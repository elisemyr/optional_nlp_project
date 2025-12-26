"""Convert LinCE Spanish-English - FIX: Test file has no labels, use validation as test."""
import pandas as pd
import json
import os
import ast
from sklearn.model_selection import train_test_split

def parse_list_string(s):
    """Parse string representation of list."""
    try:
        if isinstance(s, list):
            return s
        s = str(s)
        if s.startswith('['):
            # Remove brackets and quotes, then split
            s = s.strip('[]')
            # Handle numpy array string format with quotes
            s = s.replace("'", "").replace('"', '')
            # Split by whitespace
            items = s.split()
            return items
        return s.split()
    except Exception as e:
        print(f"Warning: Error parsing list string: {e}")
        return str(s).split()

def convert_data(df, split_name):
    """Convert DataFrame to JSON format."""
    print(f"\nProcessing {split_name}...")
    print(f"  Rows: {len(df)}")
    
    # Check labels
    if 'sa' in df.columns:
        labels_present = df['sa'].notna().sum()
        print(f"  Labels present: {labels_present}/{len(df)}")
        
        if labels_present == 0:
            print(f" NO LABELS FOUND in {split_name}!")
            return None
    
    data = []
    label_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for idx, row in df.iterrows():
        # Skip if no label
        if pd.isna(row['sa']):
            continue
            
        # Get words
        words = parse_list_string(row['words'])
        text = ' '.join(words) if words else ''
        
        # Get language tags
        lang_tags = parse_list_string(row['lid'])
        
        # Get label (already a string in this dataset)
        label = str(row['sa']).lower()
        if label not in ['positive', 'negative', 'neutral']:
            label = 'neutral'
        
        label_counts[label] += 1
        
        data.append({
            'text': text,
            'label': label,
            'words': words,
            'lang_tags': lang_tags
        })
    
    if data:
        print(f"  ✅ Converted {len(data)} examples")
        print(f"  Labels: pos={label_counts['positive']}, neg={label_counts['negative']}, neu={label_counts['neutral']}")
    
    return data

def main():
    print("\n" + "="*70)
    print("SPANISH-ENGLISH CONVERSION - FIXED FOR MISSING TEST LABELS")
    print("="*70)
    
    # Load all files
    print("\nLoading files...")
    train_df = pd.read_csv('data/raw/es-en/sa_spaeng_train.csv')
    val_df = pd.read_csv('data/raw/es-en/sa_spaeng_validation.csv')
    test_df = pd.read_csv('data/raw/es-en/sa_spaeng_test.csv')
    
    print(f"  Train: {len(train_df)} rows")
    print(f"  Validation: {len(val_df)} rows")
    print(f"  Test: {len(test_df)} rows, labels: {test_df['sa'].notna().sum()}")
    
    # Since test has no labels, use this strategy:
    # - Split train into train (80%) + validation (20%)
    # - Use original validation as test
    
    print("\n" + "="*70)
    print("STRATEGY: Test file has no labels!")
    print("  → Split train into train (80%) + validation (20%)")
    print("  → Use original validation as test")
    print("="*70)
    
    # Split train
    train_new, val_new = train_test_split(
        train_df, 
        test_size=0.2, 
        random_state=42,
        stratify=train_df['sa']  # Maintain label balance
    )
    
    print(f"\nNew splits:")
    print(f"  Train: {len(train_new)} examples")
    print(f"  Validation: {len(val_new)} examples")
    print(f"  Test: {len(val_df)} examples (original validation)")
    
    # Convert each split
    train_data = convert_data(train_new, 'train')
    val_data = convert_data(val_new, 'validation')
    test_data = convert_data(val_df, 'test')
    
    # Save
    os.makedirs('data/raw/es-en', exist_ok=True)
    
    if train_data:
        with open('data/raw/es-en/train.json', 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Saved train.json")
    
    if val_data:
        with open('data/raw/es-en/validation.json', 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved validation.json")
    
    if test_data:
        with open('data/raw/es-en/test.json', 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved test.json")
    
    # Final summary
    print("\n" + "="*70)
    print("CONVERSION COMPLETE!")
    print("="*70)
    
    for name, data in [('Train', train_data), ('Validation', val_data), ('Test', test_data)]:
        if data:
            labels = [d['label'] for d in data]
            print(f"\n{name}: {len(data)} examples")
            print(f"  Positive: {labels.count('positive')} ({labels.count('positive')/len(labels)*100:.1f}%)")
            print(f"  Negative: {labels.count('negative')} ({labels.count('negative')/len(labels)*100:.1f}%)")
            print(f"  Neutral: {labels.count('neutral')} ({labels.count('neutral')/len(labels)*100:.1f}%)")
    

if __name__ == "__main__":
    main()