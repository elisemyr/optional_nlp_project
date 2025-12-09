"""Main training script for code-mixed sentiment classification."""
import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.dataset import load_datasets


def compute_metrics(pred):
    """
    Compute evaluation metrics.
    
    Args:
        pred: Predictions from the model
        
    Returns:
        Dictionary with F1 and accuracy scores
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Compute metrics
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    
    return {
        'f1': f1_macro,
        'f1_weighted': f1_weighted,
        'accuracy': acc
    }


def train_model(
    model_name: str,
    language: str,
    output_dir: str,
    epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    seed: int = 42,
    max_length: int = 128,
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 100
):
    """
    Train sentiment classification model.
    
    Args:
        model_name: Hugging Face model name (e.g., 'xlm-roberta-base')
        language: Language pair code (e.g., 'es-en', 'hi-en')
        output_dir: Directory to save the trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay for regularization
        seed: Random seed
        max_length: Maximum sequence length
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        logging_steps: Log metrics every N steps
    """
    
    print("\n" + "=" * 70)
    print(f"Training {model_name} on {language.upper()}")
    print("=" * 70)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = load_datasets(language, model_name, max_length=max_length)
    
    if not datasets or 'train' not in datasets:
        print(" Error: Could not load training data!")
        return None
    
    print(f"\n✓ Train: {len(datasets['train'])} examples")
    print(f"✓ Validation: {len(datasets['validation'])} examples")
    print(f"✓ Test: {len(datasets['test'])} examples")
    
    # Show label distribution
    label_dist = datasets['train'].get_label_distribution()
    print(f"\n Label distribution:")
    for label, count in label_dist.items():
        print(f"   {label}: {count} ({count/len(datasets['train'])*100:.1f}%)")
    
    # Load model
    print(f"\n🤖 Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,  # positive, negative, neutral
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_dir=f'{output_dir}/logs',
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",  # Changed from evaluation_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    train_result = trainer.train()
    
    # Save final model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate on validation set
    print("\n" + "=" * 70)
    print("Evaluating on validation set...")
    print("=" * 70)
    
    val_results = trainer.evaluate(datasets['validation'])
    print(f"\n✓ Validation F1 (macro): {val_results['eval_f1']:.4f}")
    print(f"✓ Validation Accuracy: {val_results['eval_accuracy']:.4f}")
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Final evaluation on test set...")
    print("=" * 70)
    
    test_results = trainer.evaluate(datasets['test'])
    print(f"\n✓ Test F1 (macro): {test_results['eval_f1']:.4f}")
    print(f"✓ Test Accuracy: {test_results['eval_accuracy']:.4f}")
    
    # Get detailed classification report
    print("\nDetailed Classification Report:")
    predictions = trainer.predict(datasets['test'])
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    
    label_names = ['Positive', 'Negative', 'Neutral']
    report = classification_report(labels, preds, target_names=label_names, digits=4)
    print(report)
    
    # Save test metrics
    test_metrics = {
        'test_f1_macro': float(test_results['eval_f1']),
        'test_accuracy': float(test_results['eval_accuracy']),
        'test_f1_weighted': float(test_results['eval_f1_weighted']),
        'classification_report': report
    }
    
    metrics_file = f"{output_dir}/test_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n✓ Saved test metrics to {metrics_file}")
    
    # Save predictions for later analysis
    predictions_df = {
        'predictions': preds.tolist(),
        'labels': labels.tolist()
    }
    predictions_file = f"{output_dir}/predictions.json"
    with open(predictions_file, 'w') as f:
        json.dump(predictions_df, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n Model saved to: {output_dir}")
    print(f" Test F1: {test_results['eval_f1']:.4f}")
    print(f" Test Accuracy: {test_results['eval_accuracy']:.4f}")
    
    return test_results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Train sentiment classification model for code-mixed data'
    )
    
    # Required arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (e.g., xlm-roberta-base, bert-base-multilingual-cased, bert-base-uncased)'
    )
    parser.add_argument(
        '--language',
        type=str,
        required=True,
        choices=['es-en', 'hi-en'],
        help='Language pair (es-en for Spanish-English, hi-en for Hindi-English)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for saving the model'
    )
    
    # Optional arguments
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs (default: 4)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate (default: 2e-5)')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps (default: 500)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (default: 0.01)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length (default: 128)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Language: {args.language}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Seed: {args.seed}")
    
    # Train model
    try:
        results = train_model(
            model_name=args.model,
            language=args.language,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            seed=args.seed,
            max_length=args.max_length
        )
        
        if results:
            print("\nTraining successful!")
            return 0
        else:
            print("\nTraining failed!")
            return 1
            
    except Exception as e:
        print(f"\n Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())