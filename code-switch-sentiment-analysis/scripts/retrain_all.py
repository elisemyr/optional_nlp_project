#!/usr/bin/env python3
"""Retrain all models with improved class balancing to fix positive bias issue."""
import subprocess
import sys
import os
from pathlib import Path

# Change to script directory
script_dir = Path(__file__).parent
project_root = script_dir.parent
os.chdir(project_root)

models_config = [
    {
        'model': 'xlm-roberta-base',
        'language': 'es-en',
        'output': 'models/xlmr-es-en',
        'name': 'XLM-RoBERTa (Spanish-English)'
    },
    {
        'model': 'bert-base-multilingual-cased',
        'language': 'es-en',
        'output': 'models/mbert-es-en',
        'name': 'mBERT (Spanish-English)'
    },
    {
        'model': 'xlm-roberta-base',
        'language': 'hi-en',
        'output': 'models/xlmr-hi-en',
        'name': 'XLM-RoBERTa (Hindi-English)'
    },
    {
        'model': 'bert-base-multilingual-cased',
        'language': 'hi-en',
        'output': 'models/mbert-hi-en',
        'name': 'mBERT (Hindi-English)'
    }
]

def train_model(config):
    """Train a single model."""
    print("\n" + "=" * 70)
    print(f"Training: {config['name']}")
    print("=" * 70)
    
    cmd = [
        sys.executable,
        'src/training/train.py',
        '--model', config['model'],
        '--language', config['language'],
        '--output', config['output'],
        '--epochs', '5',
        '--batch_size', '8',
        '--learning_rate', '2e-5'
    ]
    
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode != 0:
        print(f"\n❌ Error training {config['name']}")
        return False
    
    print(f"\n✅ Successfully trained {config['name']}")
    return True

def main():
    print("=" * 70)
    print("RETRAINING ALL MODELS (FIXING POSITIVE BIAS)")
    print("=" * 70)
    print("\nThis script will retrain all models with improved class weights")
    print("to address the issue where models predict 'positive' for everything.")
    print("\nImprovements:")
    print("  - Balanced class weights (neutral +20%, negative +15%, positive unchanged)")
    print("  - Moderate penalties for positive/neutral confusion")
    print("  - Focal loss gamma (2.5) for hard examples")
    print("  - Penalties for both positive AND neutral prediction bias")
    print("\n" + "=" * 70)
    
    success_count = 0
    for i, config in enumerate(models_config, 1):
        print(f"\n[{i}/{len(models_config)}] {config['name']}")
        
        if train_model(config):
            success_count += 1
        else:
            print(f"\n⚠️  Failed to train {config['name']}, continuing with next model...")
    
    print("\n" + "=" * 70)
    print(f"RETRAINING COMPLETE: {success_count}/{len(models_config)} models trained successfully")
    print("=" * 70)
    
    if success_count == len(models_config):
        print("\n✅ All models retrained successfully!")
        print("\nYou can now test the models using the Streamlit app:")
        print("  streamlit run app.py")
        return 0
    else:
        print(f"\n⚠️  {len(models_config) - success_count} model(s) failed to train")
        return 1

if __name__ == '__main__':
    sys.exit(main())

