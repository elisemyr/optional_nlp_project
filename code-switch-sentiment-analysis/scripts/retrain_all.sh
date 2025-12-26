#!/bin/bash
# Retrain all models with improved class balancing to fix positive bias issue

echo "=========================================="
echo "RETRAINING ALL MODELS (FIXING POSITIVE BIAS)"
echo "=========================================="
echo ""
echo "This script will retrain all models with improved class weights"
echo "to address the issue where models predict 'positive' for everything."
echo ""

# Check if we're in the right directory
if [ ! -f "src/training/train.py" ]; then
    echo "❌ Error: Please run this script from the code-switch-sentiment-analysis directory"
    exit 1
fi

# Spanish-English models
echo "=========================================="
echo "1/4: XLM-RoBERTa (Spanish-English)"
echo "=========================================="
python src/training/train.py \
    --model xlm-roberta-base \
    --language es-en \
    --output models/xlmr-es-en \
    --epochs 5 \
    --batch_size 8 \
    --learning_rate 2e-5

if [ $? -ne 0 ]; then
    echo "❌ Error training XLM-RoBERTa (Spanish-English)"
    exit 1
fi

echo ""
echo "=========================================="
echo "2/4: mBERT (Spanish-English)"
echo "=========================================="
python src/training/train.py \
    --model bert-base-multilingual-cased \
    --language es-en \
    --output models/mbert-es-en \
    --epochs 5 \
    --batch_size 8 \
    --learning_rate 2e-5

if [ $? -ne 0 ]; then
    echo "❌ Error training mBERT (Spanish-English)"
    exit 1
fi

# Hindi-English models
echo ""
echo "=========================================="
echo "3/4: XLM-RoBERTa (Hindi-English)"
echo "=========================================="
python src/training/train.py \
    --model xlm-roberta-base \
    --language hi-en \
    --output models/xlmr-hi-en \
    --epochs 5 \
    --batch_size 8 \
    --learning_rate 2e-5

if [ $? -ne 0 ]; then
    echo "❌ Error training XLM-RoBERTa (Hindi-English)"
    exit 1
fi

echo ""
echo "=========================================="
echo "4/4: mBERT (Hindi-English)"
echo "=========================================="
python src/training/train.py \
    --model bert-base-multilingual-cased \
    --language hi-en \
    --output models/mbert-hi-en \
    --epochs 5 \
    --batch_size 8 \
    --learning_rate 2e-5

if [ $? -ne 0 ]; then
    echo "❌ Error training mBERT (Hindi-English)"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ ALL MODELS RETRAINED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "The models have been retrained with:"
echo "  - Balanced class weights (neutral +20%, negative +15%, positive unchanged)"
echo "  - Moderate penalties for positive/neutral confusion"
echo "  - Focal loss gamma (2.5) for hard examples"
echo "  - Penalties for both positive AND neutral prediction bias"
echo ""
echo "You can now test the models using the Streamlit app:"
echo "  streamlit run app.py"
echo ""

