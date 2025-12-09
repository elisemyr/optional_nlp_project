
"""Convert Kaggle LinCE CSV files to JSON format."""
import pandas as pd
import json
import os
import glob

def convert_csv_to_json(csv_file, output_file):
    """Convert CSV to our JSON format."""
    print(f"Converting {os.path.basename(csv_file)}...")
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    print(f"  Columns found: {list(df.columns)}")
    
    data = []
    for _, row in df.iterrows():
        # Try different possible column names
        text = row.get('text', row.get('tweet', row.get('sentence', '')))
        
        # Get words (might be space-separated or in a list column)
        if 'words' in row:
            words = eval(row['words']) if isinstance(row['words'], str) else row['words']
        else:
            words = text.split()
        
        # Get label
        label_col = None
        for col in ['sa', 'sentiment', 'label', 'Sentiment']:
            if col in df.columns:
                label_col = col
                break
        
        if label_col:
            label_val = row[label_col]
            # Map numeric labels
            if isinstance(label_val, (int, float)):
                label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
                label = label_map.get(int(label_val), 'neutral')
            else:
                label = str(label_val).lower()
        else:
            label = 'neutral'
        
        # Get language tags
        if 'lid' in row:
            lang_tags = eval(row['lid']) if isinstance(row['lid'], str) else row['lid']
        elif 'lang_tags' in row:
            lang_tags = eval(row['lang_tags']) if isinstance(row['lang_tags'], str) else row['lang_tags']
        else:
            # Generate simple tags if not available
            lang_tags = ['mixed'] * len(words)
        
        data.append({
            'text': text,
            'label': label,
            'words': words,
            'lang_tags': lang_tags
        })
    
    # Save JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Saved {len(data)} examples to {output_file}")
    
    # Show stats
    pos = sum(1 for d in data if d['label'] == 'positive')
    neg = sum(1 for d in data if d['label'] == 'negative')
    neu = sum(1 for d in data if d['label'] == 'neutral')
    print(f"  Label distribution: Pos={pos}, Neg={neg}, Neu={neu}")
    
    return data

def main():
    print("=" * 60)
    print("Converting Kaggle LinCE CSV files to JSON")
    print("=" * 60)
    
    os.makedirs("data/raw/es-en", exist_ok=True)
    
    # Find CSV files
    print("\nLooking for CSV files in data/raw/es-en/...")
    csv_files = glob.glob("data/raw/es-en/*.csv")
    
    if not csv_files:
        print("\n❌ No CSV files found!")
        print("\nPlease:")
        print("1. Download from: https://www.kaggle.com/datasets/thedevastator/unlock-universal-language-with-the-lince-dataset")
        print("2. Extract the ZIP")
        print("3. Copy Spanish-English sentiment CSV files to: data/raw/es-en/")
        print("   Look for files like: sa_spaeng_train.csv, sa_spaeng_dev.csv, sa_spaeng_test.csv")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    
    # Map CSV files to splits
    file_mapping = {}
    for csv_file in csv_files:
        basename = os.path.basename(csv_file).lower()
        if 'train' in basename:
            file_mapping['train'] = csv_file
        elif 'dev' in basename or 'val' in basename:
            file_mapping['validation'] = csv_file
        elif 'test' in basename:
            file_mapping['test'] = csv_file
    
    if not file_mapping:
        print("\n⚠️  Could not automatically detect train/dev/test files.")
        print("Please rename your CSV files to include 'train', 'dev', or 'test' in the name.")
        
        # Try to process any CSV file
        print("\nProcessing all CSV files as 'train'...")
        for csv_file in csv_files:
            convert_csv_to_json(csv_file, "data/raw/es-en/train.json")
        return
    
    print(f"\nMapped files:")
    for split, file in file_mapping.items():
        print(f"  {split}: {os.path.basename(file)}")
    
    # Convert each file
    print("\nConverting files...")
    for split, csv_file in file_mapping.items():
        output_file = f"data/raw/es-en/{split}.json"
        convert_csv_to_json(csv_file, output_file)
    
    print("\n" + "=" * 60)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 60)
    print("\nNext step: python src/data/preprocessing.py")

if __name__ == "__main__":
    main()
