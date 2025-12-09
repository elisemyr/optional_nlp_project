
"""Download datasets - Updated for new Hugging Face format."""
from datasets import load_dataset
import json
import os

def download_spanish_english_lince():
    """Download Spanish-English from LinCE (Hugging Face)."""
    print("=" * 60)
    print("Downloading Spanish-English from LinCE (Hugging Face)...")
    print("=" * 60)
    
    try:
        # Try new method without dataset script
        print("\nAttempt 1: Loading with trust_remote_code...")
        dataset = load_dataset("lince", "sa_spaeng", trust_remote_code=True)
    except Exception as e1:
        print(f"Attempt 1 failed: {e1}")
        try:
            # Try alternative repository
            print("\nAttempt 2: Loading from lince-benchmark...")
            dataset = load_dataset("lince-benchmark/lince", "sa_spaeng", trust_remote_code=True)
        except Exception as e2:
            print(f"Attempt 2 failed: {e2}")
            print("\n" + "=" * 60)
            print("❌ Cannot download from Hugging Face")
            print("=" * 60)
            print("\n📥 MANUAL DOWNLOAD REQUIRED:")
            print("\n1. Visit: https://ritual.uh.edu/lince/datasets")
            print("2. Register and download Spanish-English SA dataset")
            print("3. Download files:")
            print("   - sa_spaeng_train.conll")
            print("   - sa_spaeng_dev.conll")
            print("   - sa_spaeng_test.conll")
            print("\n4. Place them in: data/raw/es-en/")
            print("5. Then run: python data/scripts/convert_lince_spanish.py")
            
            os.makedirs("data/raw/es-en", exist_ok=True)
            return False
    
    os.makedirs("data/raw/es-en", exist_ok=True)
    
    for split in ['train', 'validation', 'test']:
        data = []
        split_data = dataset[split]
        
        print(f"\nProcessing {split} split ({len(split_data)} examples)...")
        
        for example in split_data:
            words = example['words']
            text = ' '.join(words)
            
            # Get sentiment label
            if 'sa' in example and example['sa'] is not None:
                label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
                label = label_map.get(example['sa'], 'neutral')
            else:
                label = 'neutral'
            
            lang_tags = example.get('lid', [])
            
            data.append({
                'text': text,
                'label': label,
                'words': words,
                'lang_tags': lang_tags
            })
        
        output_file = f"data/raw/es-en/{split}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved {len(data)} examples to {output_file}")
    
    print("\n✅ Spanish-English dataset downloaded successfully!")
    return True

if __name__ == "__main__":
    print("CODE-MIXED SENTIMENT ANALYSIS - DATASET DOWNLOADER\n")
    
    success = download_spanish_english_lince()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ DOWNLOAD COMPLETE!")
        print("=" * 60)
        print("\nNext step: python src/data/preprocessing.py")
    else:
        print("\nPlease download manually and use converter scripts.")
