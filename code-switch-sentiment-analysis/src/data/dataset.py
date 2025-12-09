"""PyTorch Dataset classes for code-mixed sentiment analysis."""
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from typing import Dict, Optional


class CodeMixedDataset(Dataset):
    """PyTorch Dataset for code-mixed sentiment analysis."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str,
        max_length: int = 128,
        include_lang_tags: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to processed pickle file (.pkl)
            tokenizer_name: Hugging Face tokenizer name (e.g., 'xlm-roberta-base')
            max_length: Maximum sequence length for tokenization
            include_lang_tags: Whether to include language tags in output
        """
        # Load preprocessed data
        self.df = pd.read_pickle(data_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.include_lang_tags = include_lang_tags
        
        print(f"Loaded {len(self.df)} examples from {data_path}")
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary containing:
                - input_ids: Token IDs (shape: [max_length])
                - attention_mask: Attention mask (shape: [max_length])
                - labels: Sentiment label (0=positive, 1=negative, 2=neutral)
                - cs_index (optional): Code-switching index
        """
        row = self.df.iloc[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            row['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare output dictionary
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(row['label'], dtype=torch.long)
        }
        
        # Optionally include additional features
        if self.include_lang_tags:
            item['cs_index'] = torch.tensor(row['cs_index'], dtype=torch.float)
        
        return item
    
    def get_label_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of labels in the dataset.
        
        Returns:
            Dictionary mapping label names to counts
        """
        return self.df['label_name'].value_counts().to_dict()
    
    def get_examples_by_cs_density(
        self, 
        min_cs: float, 
        max_cs: float
    ) -> pd.DataFrame:
        """
        Get examples within a specific CS density range.
        
        Args:
            min_cs: Minimum CS-Index (inclusive)
            max_cs: Maximum CS-Index (exclusive)
            
        Returns:
            Filtered DataFrame
        """
        return self.df[
            (self.df['cs_index'] >= min_cs) & (self.df['cs_index'] < max_cs)
        ]
    
    def get_cs_statistics(self) -> Dict[str, float]:
        """
        Get code-switching statistics for the dataset.
        
        Returns:
            Dictionary with CS statistics
        """
        return {
            'mean_cs_index': self.df['cs_index'].mean(),
            'std_cs_index': self.df['cs_index'].std(),
            'min_cs_index': self.df['cs_index'].min(),
            'max_cs_index': self.df['cs_index'].max(),
            'median_cs_index': self.df['cs_index'].median()
        }


def load_datasets(
    language: str,
    tokenizer_name: str,
    max_length: int = 128,
    include_lang_tags: bool = False
) -> Dict[str, CodeMixedDataset]:
    """
    Load train, validation, and test datasets for a language pair.
    
    Args:
        language: Language pair code (e.g., 'es-en', 'hi-en')
        tokenizer_name: Hugging Face tokenizer name
        max_length: Maximum sequence length
        include_lang_tags: Whether to include language tags
    
    Returns:
        Dictionary with 'train', 'validation', 'test' datasets
    """
    datasets = {}
    
    for split in ['train', 'validation', 'test']:
        data_path = f"data/processed/{language}/{split}.pkl"
        
        try:
            datasets[split] = CodeMixedDataset(
                data_path=data_path,
                tokenizer_name=tokenizer_name,
                max_length=max_length,
                include_lang_tags=include_lang_tags
            )
        except FileNotFoundError:
            print(f"⚠️  Warning: {data_path} not found, skipping {split}")
    
    return datasets


def test_dataset_loading():
    """Test function to verify dataset loading works correctly."""
    print("=" * 60)
    print("Testing CodeMixedDataset")
    print("=" * 60)
    
    # Test Spanish-English
    print("\n1. Testing Spanish-English dataset...")
    try:
        datasets_es = load_datasets('es-en', 'bert-base-uncased')
        
        if datasets_es:
            print(f"\n✅ Spanish-English loaded successfully!")
            print(f"   Train size: {len(datasets_es['train'])}")
            print(f"   Validation size: {len(datasets_es['validation'])}")
            print(f"   Test size: {len(datasets_es['test'])}")
            
            # Get a sample
            sample = datasets_es['train'][0]
            print(f"\n   Sample item keys: {list(sample.keys())}")
            print(f"   Input IDs shape: {sample['input_ids'].shape}")
            print(f"   Attention mask shape: {sample['attention_mask'].shape}")
            print(f"   Label: {sample['labels']} ({sample['labels'].item()})")
            
            # Label distribution
            label_dist = datasets_es['train'].get_label_distribution()
            print(f"\n   Label distribution:")
            for label, count in label_dist.items():
                print(f"      {label}: {count}")
            
            # CS statistics
            cs_stats = datasets_es['train'].get_cs_statistics()
            print(f"\n   CS Statistics:")
            print(f"      Mean: {cs_stats['mean_cs_index']:.3f}")
            print(f"      Std: {cs_stats['std_cs_index']:.3f}")
            print(f"      Range: [{cs_stats['min_cs_index']:.3f}, {cs_stats['max_cs_index']:.3f}]")
        
    except Exception as e:
        print(f"✗ Error loading Spanish-English: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Hindi-English
    print("\n2. Testing Hindi-English dataset...")
    try:
        datasets_hi = load_datasets('hi-en', 'bert-base-uncased')
        
        if datasets_hi:
            print(f"\n✅ Hindi-English loaded successfully!")
            print(f"   Train size: {len(datasets_hi['train'])}")
            print(f"   Validation size: {len(datasets_hi['validation'])}")
            print(f"   Test size: {len(datasets_hi['test'])}")
            
            # Get a sample
            sample = datasets_hi['train'][0]
            print(f"\n   Sample item keys: {list(sample.keys())}")
            print(f"   Input IDs shape: {sample['input_ids'].shape}")
            print(f"   Label: {sample['labels']}")
            
            # Label distribution
            label_dist = datasets_hi['train'].get_label_distribution()
            print(f"\n   Label distribution:")
            for label, count in label_dist.items():
                print(f"      {label}: {count}")
    
    except Exception as e:
        print(f"✗ Error loading Hindi-English: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("DATASET TESTING COMPLETE!")



if __name__ == "__main__":
    test_dataset_loading()
