"""Main training script for code-mixed sentiment classification."""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
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
    
    # Per-class F1 scores
    f1_per_class = f1_score(labels, preds, average=None)
    
    return {
        'f1': f1_macro,
        'f1_weighted': f1_weighted,
        'accuracy': acc,
        'f1_positive': float(f1_per_class[0]),
        'f1_negative': float(f1_per_class[1]),
        'f1_neutral': float(f1_per_class[2])
    }


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples.
    
    Focal loss focuses learning on hard examples by down-weighting easy examples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedTrainer(Trainer):
    """Custom Trainer with weighted loss for class imbalance and positive/neutral differentiation."""
    
    def __init__(self, class_weights=None, use_focal_loss=False, focal_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights_raw = class_weights  # Store raw weights
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        
        if class_weights is not None:
            # Store as CPU tensor, will move to device when needed
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            
            # For focal loss, use class weights as alpha
            if use_focal_loss:
                self.focal_loss = FocalLoss(alpha=self.class_weights, gamma=focal_gamma)
            else:
                self.focal_loss = None
        else:
            if use_focal_loss:
                self.focal_loss = FocalLoss(alpha=None, gamma=focal_gamma)
            else:
                self.focal_loss = None
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute weighted or focal loss with additional penalty for positive/neutral confusion.
        
        Args:
            model: The model to train
            inputs: Input dictionary
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (for compatibility with newer transformers)
            
        Returns:
            Loss value (and optionally outputs)
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Get device from logits to ensure weights are on the same device
        device = logits.device
        
        # Reshape for loss computation
        logits_flat = logits.view(-1, self.model.config.num_labels)
        labels_flat = labels.view(-1)
        
        # Base loss: Use focal loss if specified
        if self.use_focal_loss and self.focal_loss is not None:
            # Move focal loss weights to device if needed
            if self.focal_loss.alpha is not None and self.focal_loss.alpha.device != device:
                self.focal_loss.alpha = self.focal_loss.alpha.to(device)
            base_loss = self.focal_loss(logits_flat, labels_flat)
        # Use weighted cross-entropy if class weights are provided
        elif self.class_weights is not None:
            # Move class weights to the same device as logits
            class_weights_device = self.class_weights.to(device)
            loss_fct = nn.CrossEntropyLoss(weight=class_weights_device)
            base_loss = loss_fct(logits_flat, labels_flat)
        else:
            loss_fct = nn.CrossEntropyLoss()
            base_loss = loss_fct(logits_flat, labels_flat)
        
        # Additional penalty for positive/neutral confusion
        # This helps the model better differentiate between these two classes
        # positive=0, neutral=2
        positive_idx = 0
        neutral_idx = 2
        
        # Get probabilities
        probs = torch.softmax(logits_flat, dim=-1)
        
        # Find examples where:
        # 1. True label is neutral but model predicts positive
        # 2. True label is positive but model predicts neutral
        neutral_mask = (labels_flat == neutral_idx)
        positive_mask = (labels_flat == positive_idx)
        
        # Balanced penalties for confusion between classes
        neutral_as_positive = neutral_mask * probs[:, positive_idx]
        neutral_penalty = neutral_as_positive.mean() * 0.3  # Moderate penalty (30%)
        
        positive_as_neutral = positive_mask * probs[:, neutral_idx]
        positive_penalty = positive_as_neutral.mean() * 0.2  # Moderate penalty (20%)
        
        # Penalty for positive bias (if > 50% predictions are positive)
        pred_positive_ratio = probs[:, positive_idx].mean()
        positive_bias_penalty = torch.clamp((pred_positive_ratio - 0.5) * 0.15, min=0.0)
        
        # Penalty for neutral bias (if > 40% predictions are neutral) - NEW
        pred_neutral_ratio = probs[:, neutral_idx].mean()
        neutral_bias_penalty = torch.clamp((pred_neutral_ratio - 0.4) * 0.15, min=0.0)
        
        # Combine losses
        total_loss = base_loss + neutral_penalty + positive_penalty + positive_bias_penalty + neutral_bias_penalty
        
        return (total_loss, outputs) if return_outputs else total_loss


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
    print("\n📊 Loading datasets...")
    datasets = load_datasets(language, model_name, max_length=max_length)
    
    if not datasets or 'train' not in datasets:
        print("❌ Error: Could not load training data!")
        return None
    
    print(f"\n✓ Train: {len(datasets['train'])} examples")
    print(f"✓ Validation: {len(datasets['validation'])} examples")
    print(f"✓ Test: {len(datasets['test'])} examples")
    
    # Show label distribution
    label_dist = datasets['train'].get_label_distribution()
    print(f"\n📋 Label distribution:")
    for label, count in label_dist.items():
        print(f"   {label}: {count} ({count/len(datasets['train'])*100:.1f}%)")
    
    # Compute class weights to handle imbalance
    print(f"\n⚖️  Computing class weights for balanced training...")
    train_df = datasets['train'].df
    labels = train_df['label'].values
    
    # Compute class weights (inverse frequency)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = dict(zip(np.unique(labels), class_weights))
    print(f"   Class weights: {class_weight_dict}")
    
    # Boost weights for neutral and negative classes to improve differentiation
    # This helps the model better distinguish positive from neutral
    # Format: [positive=0, negative=1, neutral=2]
    class_weights_boosted = class_weights.copy()
    neutral_idx = 2
    negative_idx = 1
    positive_idx = 0
    
    # Moderate boost for neutral (20%) to help differentiate from positive, but not too much
    class_weights_boosted[neutral_idx] = class_weights[neutral_idx] * 1.2
    # Moderate boost for negative (15%) to help with class balance
    class_weights_boosted[negative_idx] = class_weights[negative_idx] * 1.15
    # Keep positive weight as is (don't reduce it)
    # class_weights_boosted[positive_idx] = class_weights[positive_idx] * 1.0
    
    print(f"   Boosted weights (neutral +20%, negative +15%, positive unchanged): {dict(zip(np.unique(labels), class_weights_boosted))}")
    
    # Normalize weights to sum to num_classes for stability
    class_weights_normalized = class_weights_boosted / class_weights_boosted.sum() * len(class_weights_boosted)
    print(f"   Normalized weights: {dict(zip(np.unique(labels), class_weights_normalized))}")
    
    # Load model with explicit label mappings
    print(f"\n🤖 Loading model: {model_name}")
    
    # Define label mappings (consistent with preprocessing: positive=0, negative=1, neutral=2)
    id2label = {0: "positive", 1: "negative", 2: "neutral"}
    label2id = {"positive": 0, "negative": 1, "neutral": 2}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,  # positive, negative, neutral
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    
    print(f"   ✓ Model configured with labels: {id2label}")
    
    # Check for available device (CUDA, MPS, or CPU)
    # Note: MPS may have compatibility issues, so we default to CPU for stability
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS can have issues with some operations, so use CPU for now
        # Uncomment the line below to try MPS (may cause errors)
        # device = "mps"
        device = "cpu"
        print("   ⚠️  MPS detected but using CPU for compatibility")
    else:
        device = "cpu"
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
        save_steps=save_steps * 10,  # Save less frequently to avoid error
        eval_steps=eval_steps * 2,  # Evaluate less frequently
        eval_strategy="epoch",  # Evaluate only at end of each epoch (simpler)
        save_strategy="epoch",  # Save only at end of each epoch
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,  # Keep only 1 checkpoint to save space
        seed=seed,
        fp16=False,  # Disable mixed precision on CPU
        report_to="none",
        save_safetensors=True,  # Use safer saving format
    )
    
    # Create Trainer with weighted loss and positive/neutral differentiation
    # Enable focal loss for better handling of hard examples (like positive/neutral confusion)
    use_focal = True  # Enable focal loss for better positive/neutral differentiation
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        compute_metrics=compute_metrics,
        class_weights=class_weights_normalized,
        use_focal_loss=use_focal,
        focal_gamma=2.5,  # Moderate gamma for focus on hard examples without over-penalizing
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    if use_focal:
        print(f"   ✓ Using Focal Loss (gamma={2.5}) with boosted class weights")
        print(f"   ✓ Additional penalty for positive/neutral confusion")
    else:
        print(f"   ✓ Using Weighted Cross-Entropy Loss for class balance")
    
    # Train
    print("\n" + "=" * 70)
    print("🚀 Starting training...")
    print("=" * 70)
    
    train_result = trainer.train()
    
    # Save final model
    print(f"\n💾 Saving model to {output_dir}")
    trainer.save_model(output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate on validation set
    print("\n" + "=" * 70)
    print("📊 Evaluating on validation set...")
    print("=" * 70)
    
    val_results = trainer.evaluate(datasets['validation'])
    print(f"\n✓ Validation F1 (macro): {val_results['eval_f1']:.4f}")
    print(f"✓ Validation Accuracy: {val_results['eval_accuracy']:.4f}")
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("📊 Final evaluation on test set...")
    print("=" * 70)
    
    test_results = trainer.evaluate(datasets['test'])
    print(f"\n✓ Test F1 (macro): {test_results['eval_f1']:.4f}")
    print(f"✓ Test Accuracy: {test_results['eval_accuracy']:.4f}")
    
    # Get detailed classification report
    print("\n📋 Detailed Classification Report:")
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
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Model saved to: {output_dir}")
    print(f"📊 Test F1: {test_results['eval_f1']:.4f}")
    print(f"📊 Test Accuracy: {test_results['eval_accuracy']:.4f}")
    
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
            print("\n Training successful!")
            return 0
        else:
            print("\n Training failed!")
            return 1
            
    except Exception as e:
        print(f"\n Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())