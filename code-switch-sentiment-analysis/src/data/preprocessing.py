"""Preprocessing pipeline for code-mixed sentiment analysis - FIXED VERSION."""
import json
import pickle
import os
from pathlib import Path
from typing import List, Dict
import re
from tqdm import tqdm
import pandas as pd
import numpy as np


class CodeMixedPreprocessor:
    """Preprocessing for code-mixed sentiment data."""
    
    def __init__(self):
        self.label_map = {
            'positive': 0,
            'negative': 1,
            'neutral': 2
        }
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize code-mixed text while preserving sentiment cues.
        
        Args:
            text: Raw text string
            
        Returns:
            Normalized text
        """
        # Preserve original case for sentiment words (emojis, exclamation marks, etc.)
        # Only lowercase for better tokenization, but keep some sentiment indicators
        
        # Replace URLs
        text = re.sub(r'http\S+|www.\S+', '<url>', text)
        
        # Replace usernames/mentions (but keep structure)
        text = re.sub(r'@\w+', '<user>', text)
        
        # Keep hashtags but normalize (hashtags can carry sentiment)
        # Don't remove # completely as it might be part of sentiment
        text = re.sub(r'#(\w+)', r'#\1', text)
        
        # Normalize repeated characters (e.g., "sooo" -> "sooo" but limit to 3)
        # This preserves emphasis which can indicate sentiment
        text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
        
        # Lowercase for consistency (but after preserving structure)
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def calculate_cs_index(self, lang_tags: List[str]) -> float:
        """
        Calculate code-switching index.
        
        CS-Index = (# of switch points) / (# of tokens - 1)
        
        Args:
            lang_tags: List of language tags for each token
            
        Returns:
            CS-Index score between 0 and 1
        """
        if len(lang_tags) <= 1:
            return 0.0
        
        # Count language switches
        switches = 0
        for i in range(len(lang_tags) - 1):
            curr_tag = lang_tags[i]
            next_tag = lang_tags[i + 1]
            
            # Only count switches between actual languages
            if (curr_tag not in ['other', 'ambiguous', 'O', 'mixed'] and
                next_tag not in ['other', 'ambiguous', 'O', 'mixed'] and
                curr_tag != next_tag):
                switches += 1
        
        # Normalize by number of possible switch points
        cs_index = switches / (len(lang_tags) - 1)
        
        return cs_index
    
    def process_example(self, example: Dict) -> Dict:
        """
        Process a single example.
        
        Args:
            example: Dictionary with 'text', 'label', 'words', 'lang_tags'
            
        Returns:
            Processed example dictionary
        """
        # Normalize text
        text = self.normalize_text(example['text'])
        
        # Map label to integer with robust handling
        label_value = example.get('label', 'neutral')
        
        # Handle different label formats
        if isinstance(label_value, (int, float)):
            # Handle numeric labels (0=positive, 1=negative, 2=neutral)
            label_map_numeric = {0: 'positive', 1: 'negative', 2: 'neutral'}
            label_name = label_map_numeric.get(int(label_value), 'neutral')
        else:
            # Handle string labels
            label_name = str(label_value).lower().strip()
            
            # More robust label matching
            if label_name not in self.label_map:
                # Handle various label variations
                if any(x in label_name for x in ['pos', 'good', 'happy', 'love', 'like', 'great', 'excellent']):
                    label_name = 'positive'
                elif any(x in label_name for x in ['neg', 'bad', 'hate', 'sad', 'angry', 'terrible', 'awful']):
                    label_name = 'negative'
                elif any(x in label_name for x in ['neu', 'none', 'mixed', 'other']):
                    label_name = 'neutral'
                else:
                    # Default to neutral if unclear
                    print(f"⚠️  Warning: Unknown label '{label_value}', defaulting to 'neutral'")
                    label_name = 'neutral'
        
        label_id = self.label_map[label_name]
        
        # Calculate CS-Index
        lang_tags = example.get('lang_tags', [])
        cs_index = self.calculate_cs_index(lang_tags)
        
        # Get words
        words = example.get('words', text.split())
        
        return {
            'text': text,
            'label': label_id,
            'label_name': label_name,
            'cs_index': cs_index,
            'lang_tags': lang_tags,
            'words': words
        }
    
    def process_file(self, input_file: str, output_file: str) -> pd.DataFrame:
        """
        Process entire file.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output pickle file
            
        Returns:
            Processed DataFrame
        """
        print(f"\nProcessing {input_file}...")
        
        # Load data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"  Loaded {len(data)} examples")
        
        # Process examples
        processed = []
        for example in tqdm(data, desc="  Processing", leave=False):
            try:
                processed.append(self.process_example(example))
            except Exception as e:
                print(f"  ⚠️  Error processing example: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(processed)
        
        # Save to pickle
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_pickle(output_file)
        
        print(f"  ✓ Saved {len(df)} examples to {output_file}")
        
        return df
    
    def compute_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute dataset statistics.
        
        Args:
            df: DataFrame with processed data
            
        Returns:
            Dictionary of statistics (JSON-serializable)
        """
        # Convert all numpy/pandas types to Python native types
        stats = {
            'total_examples': int(len(df)),
            'label_distribution': {
                str(k): int(v) for k, v in df['label_name'].value_counts().to_dict().items()
            },
            'avg_text_length': float(df['text'].str.len().mean()),
            'avg_num_tokens': float(df['text'].str.split().str.len().mean()),
            'avg_cs_index': float(df['cs_index'].mean()),
            'cs_index_std': float(df['cs_index'].std()),
            'cs_index_min': float(df['cs_index'].min()),
            'cs_index_max': float(df['cs_index'].max()),
            'cs_density_bins': {
                'low (0-0.3)': int((df['cs_index'] < 0.3).sum()),
                'medium (0.3-0.6)': int(((df['cs_index'] >= 0.3) & (df['cs_index'] < 0.6)).sum()),
                'high (0.6+)': int((df['cs_index'] >= 0.6).sum())
            }
        }
        return stats


def process_language_pair(language: str):
    """
    Process all splits for a language pair.
    
    Args:
        language: Language code (e.g., 'es-en', 'hi-en')
    """
    print(f"\n{'=' * 60}")
    print(f"Processing {language.upper()} dataset")
    print('=' * 60)
    
    preprocessor = CodeMixedPreprocessor()
    
    # Process each split
    splits = ['train', 'validation', 'test']
    statistics = {}
    
    for split in splits:
        input_file = f"data/raw/{language}/{split}.json"
        
        # Check if file exists
        if not os.path.exists(input_file):
            print(f"\n⚠️  {split}.json not found, skipping")
            continue
        
        output_file = f"data/processed/{language}/{split}.pkl"
        
        # Process
        df = preprocessor.process_file(input_file, output_file)
        
        # Compute statistics
        stats = preprocessor.compute_statistics(df)
        statistics[split] = stats
        
        # Print statistics
        print(f"\n  {split.upper()} Statistics:")
        print(f"    Total examples: {stats['total_examples']}")
        print(f"    Label distribution:")
        for label, count in stats['label_distribution'].items():
            print(f"      {label}: {count} ({count/stats['total_examples']*100:.1f}%)")
        print(f"    Avg text length: {stats['avg_text_length']:.1f} chars")
        print(f"    Avg tokens: {stats['avg_num_tokens']:.1f}")
        print(f"    CS-Index: {stats['avg_cs_index']:.3f} ± {stats['cs_index_std']:.3f}")
        print(f"    CS-Index range: [{stats['cs_index_min']:.3f}, {stats['cs_index_max']:.3f}]")
        print(f"    CS density bins:")
        for bin_name, count in stats['cs_density_bins'].items():
            print(f"      {bin_name}: {count} ({count/stats['total_examples']*100:.1f}%)")
    
    # Save statistics (now JSON-serializable)
    if statistics:
        stats_file = f"data/processed/{language}/statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2)
        print(f"\n  ✓ Saved statistics to {stats_file}")


def main():
    """Main preprocessing pipeline."""
    print("CODE-MIXED SENTIMENT ANALYSIS - PREPROCESSING")
    print("=" * 60)
    
    # Check which languages have raw data
    languages = []
    
    if os.path.exists("data/raw/es-en") and any(
        os.path.exists(f"data/raw/es-en/{split}.json") 
        for split in ['train', 'validation', 'test']
    ):
        languages.append("es-en")
    
    if os.path.exists("data/raw/hi-en") and any(
        os.path.exists(f"data/raw/hi-en/{split}.json")
        for split in ['train', 'validation', 'test']
    ):
        languages.append("hi-en")
    
    if not languages:
        print("\n✗ No raw data found!")
        print("\nPlease run data conversion scripts first:")
        print("  1. python data/scripts/convert_spanish_lince.py")
        print("  2. python data/scripts/convert_hindi_semeval.py")
        return
    
    print(f"\nFound languages: {', '.join(languages)}")
    
    # Create output directories
    for language in languages:
        os.makedirs(f"data/processed/{language}", exist_ok=True)
    
    # Process each language
    for language in languages:
        try:
            process_language_pair(language)
        except Exception as e:
            print(f"\n✗ Error processing {language}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")



if __name__ == "__main__":
    main()