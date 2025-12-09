
"""Convert SemEval-2020 Hindi-English CoNLL files to JSON."""
import json
import os
import csv

def parse_conll_file(file_path, labels_dict=None):
    """Parse CoNLL format file."""
    tweets = []
    current_tweet = {
        'words': [],
        'lang_tags': [],
        'meta_id': None,
        'label': None
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:  # Empty line = end of tweet
                if current_tweet['words']:
                    text = ' '.join(current_tweet['words'])
                    
                    if labels_dict and current_tweet['meta_id']:
                        label = labels_dict.get(current_tweet['meta_id'], 'neutral')
                    else:
                        label = current_tweet['label'] or 'neutral'
                    
                    tweets.append({
                        'text': text,
                        'label': label.lower(),
                        'words': current_tweet['words'],
                        'lang_tags': current_tweet['lang_tags']
                    })
                    
                    current_tweet = {
                        'words': [],
                        'lang_tags': [],
                        'meta_id': None,
                        'label': None
                    }
                continue
            
            parts = line.split('\t')
            
            if parts[0] == 'meta':
                current_tweet['meta_id'] = parts[1]
                if len(parts) > 2:
                    current_tweet['label'] = parts[2]
            else:
                word = parts[0]
                lang_tag = parts[1] if len(parts) > 1 else 'O'
                
                # Map language tags
                if lang_tag == 'Eng':
                    lang_tag = 'lang1'
                elif lang_tag == 'Hin':
                    lang_tag = 'lang2'
                else:
                    lang_tag = 'other'
                
                current_tweet['words'].append(word)
                current_tweet['lang_tags'].append(lang_tag)
    
    # Last tweet
    if current_tweet['words']:
        text = ' '.join(current_tweet['words'])
        if labels_dict and current_tweet['meta_id']:
            label = labels_dict.get(current_tweet['meta_id'], 'neutral')
        else:
            label = current_tweet['label'] or 'neutral'
        
        tweets.append({
            'text': text,
            'label': label.lower(),
            'words': current_tweet['words'],
            'lang_tags': current_tweet['lang_tags']
        })
    
    return tweets

def load_test_labels(labels_file):
    """Load test labels from CSV."""
    labels = {}
    with open(labels_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['Uid']] = row['Sentiment']
    return labels

def main():
    print("=" * 60)
    print("Converting Hindi-English SemEval to JSON")
    print("=" * 60)
    
    # Dev file (use as validation)
    print("\n1. Processing Dev file...")
    dev_data = parse_conll_file('data/raw/hi-en/Dev_3k_Split_Conll.txt')
    print(f"   ✓ Loaded {len(dev_data)} examples")
    
    # Test labels
    print("\n2. Loading test labels...")
    test_labels = load_test_labels('data/raw/hi-en/Test_Labels_Hinglish.txt')
    print(f"   ✓ Loaded {len(test_labels)} labels")
    
    # Test file
    print("\n3. Processing Test file...")
    test_data = parse_conll_file(
        'data/raw/hi-en/Hindi_Test_Unlabeled_CONLL_Updated.txt',
        labels_dict=test_labels
    )
    print(f"   ✓ Loaded {len(test_data)} examples")
    
    # Split test into train (80%) + test (20%)
    split_point = int(len(test_data) * 0.8)
    train_data = test_data[:split_point]
    test_data_final = test_data[split_point:]
    
    # Save
    datasets = {
        'train': train_data,
        'validation': dev_data,
        'test': test_data_final
    }
    
    for split_name, split_data in datasets.items():
        output_file = f"data/raw/hi-en/{split_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        
        pos = sum(1 for d in split_data if d['label'] == 'positive')
        neg = sum(1 for d in split_data if d['label'] == 'negative')
        neu = sum(1 for d in split_data if d['label'] == 'neutral')
        
        print(f"\n✓ Saved {split_name}.json: {len(split_data)} examples")
        print(f"  Pos: {pos}, Neg: {neg}, Neu: {neu}")
    
    print("\n✅ Hindi-English conversion complete!")

if __name__ == "__main__":
    main()
