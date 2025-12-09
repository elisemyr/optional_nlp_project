"""Convert LinCE Spanish-English CSV files to JSON format - FIXED VERSION."""
import pandas as pd
import json
import os

def convert_csv_to_json(csv_file, output_file):
    """Convert LinCE CSV to JSON with proper label handling."""
    print(f"Converting {os.path.basename(csv_file)}...")
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    # Check what the 'sa' column contains
    print(f"  Unique 'sa' values: {df['sa'].unique()}")
    
    data = []
    for idx, row in df.iterrows():
        # Get words (list as string)
        words_str = str(row['words'])
        try:
            words = eval(words_str) if words_str.startswith('[') else words_str.split()
        except:
            words = words_str.split()
        
        # Get text
        text = ' '.join(words) if words else ''
        
        # Get language tags
        lid_str = str(row['lid'])
        try:
            lang_tags = eval(lid_str) if lid_str.startswith('[') else ['other'] * len(words)
        except:
            lang_tags = ['other'] * len(words)
        
        # Get sentiment label - THIS IS THE FIX!
        label_value = row['sa']
        
        # LinCE uses these mappings:
        # 0 = positive
        # 1 = negative  
        # 2 = neutral
        label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
        
        # Handle different data types
        if pd.isna(label_value):
            label = 'neutral'
        elif isinstance(label_value, str):
            label = label_value.lower()
        else:
            label = label_map.get(int(label_value), 'neutral')
        
        data.append({
            'text': text,
            'label': label,
            'words': words,
            'lang_tags': lang_tags
        })
    
    # Save JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Stats
    label_counts = {}
    for d in data:
        label_counts[d['label']] = label_counts.get(d['label'], 0) + 1
    
    print(f"  ✓ Saved {len(data)} examples")
    print(f"  Label distribution:")
    for label, count in label_counts.items():
        print(f"    {label}: {count} ({count/len(data)*100:.1f}%)")

def main():
    print("=" * 60)
    print("Converting Spanish-English LinCE CSV to JSON - FIXED")
    print("=" * 60)
    
    for split in ['train', 'validation', 'test']:
        csv_file = f"data/raw/es-en/sa_spaeng_{split}.csv"
        json_file = f"data/raw/es-en/{split}.json"
        
        if os.path.exists(csv_file):
            convert_csv_to_json(csv_file, json_file)
        else:
            print(f"\n  ⚠️  {csv_file} not found")
    

if __name__ == "__main__":
    main()