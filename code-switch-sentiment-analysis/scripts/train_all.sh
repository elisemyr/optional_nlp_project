#!/bin/bash
# Train 4 models optimized for CPU

echo "=========================================="
echo "TRAINING 4 MODELS (CPU-OPTIMIZED)"
echo "=========================================="

# Spanish-English
echo "1/4: XLM-RoBERTa (Spanish-English)"
python src/training/train.py \
    --model xlm-roberta-base \
    --language es-en \
    --output models/xlmr-es-en \
    --epochs 3 \
    --batch_size 4

echo ""
echo "2/4: mBERT (Spanish-English)"
python src/training/train.py \
    --model bert-base-multilingual-cased \
    --language es-en \
    --output models/mbert-es-en \
    --epochs 3 \
    --batch_size 4

# Hindi-English
echo ""
echo "3/4: XLM-RoBERTa (Hindi-English)"
python src/training/train.py \
    --model xlm-roberta-base \
    --language hi-en \
    --output models/xlmr-hi-en \
    --epochs 3 \
    --batch_size 4

echo ""
echo "4/4: mBERT (Hindi-English)"
python src/training/train.py \
    --model bert-base-multilingual-cased \
    --language hi-en \
    --output models/mbert-hi-en \
    --epochs 3 \
    --batch_size 4

echo ""
echo "=========================================="
echo "✅ ALL 4 MODELS TRAINED!"
echo "=========================================="