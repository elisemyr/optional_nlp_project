#!/bin/bash
# Master evaluation script - run after all training is complete

echo "=========================================="
echo "EVALUATING ALL MODELS"
echo "=========================================="

# Check if models exist
if [ ! -d "models/xlmr-es-en" ]; then
    echo " Models not found! Train models first."
    exit 1
fi

echo ""
echo " Step 1: Evaluating individual models..."
echo "=========================================="

# Spanish-English models
echo ""
echo "1/4: XLM-RoBERTa (Spanish-English)"
python src/evaluation/evaluate.py \
    --model models/xlmr-es-en \
    --data data/processed/es-en/test.pkl \
    --output results/xlmr-es-en

echo ""
echo "2/4: mBERT (Spanish-English)"
python src/evaluation/evaluate.py \
    --model models/mbert-es-en \
    --data data/processed/es-en/test.pkl \
    --output results/mbert-es-en

# Hindi-English models
echo ""
echo "3/4: XLM-RoBERTa (Hindi-English)"
python src/evaluation/evaluate.py \
    --model models/xlmr-hi-en \
    --data data/processed/hi-en/test.pkl \
    --output results/xlmr-hi-en

echo ""
echo "4/4: mBERT (Hindi-English)"
python src/evaluation/evaluate.py \
    --model models/mbert-hi-en \
    --data data/processed/hi-en/test.pkl \
    --output results/mbert-hi-en

# CS Density Analysis
echo ""
echo "=========================================="
echo " Step 2: CS Density Analysis..."
echo "=========================================="

echo ""
echo "Spanish-English:"
python src/analysis/cs_density_analysis.py \
    --model models/xlmr-es-en \
    --data data/processed/es-en/test.pkl \
    --output results/cs_density_es-en

echo ""
echo "Hindi-English:"
python src/analysis/cs_density_analysis.py \
    --model models/xlmr-hi-en \
    --data data/processed/hi-en/test.pkl \
    --output results/cs_density_hi-en

# Error Analysis
echo ""
echo "=========================================="
echo "Step 3: Error Analysis..."
echo "=========================================="

echo ""
echo "Spanish-English:"
python src/evaluation/error_analysis.py \
    --model models/xlmr-es-en \
    --data data/processed/es-en/test.pkl \
    --output results/error_analysis_es-en \
    --num_errors 30

echo ""
echo "Hindi-English:"
python src/evaluation/error_analysis.py \
    --model models/xlmr-hi-en \
    --data data/processed/hi-en/test.pkl \
    --output results/error_analysis_hi-en \
    --num_errors 30

# Model Comparison
echo ""
echo "=========================================="
echo "Step 4: Comparing all models..."
echo "=========================================="

python src/evaluation/compare_models.py

# Summary
echo ""
echo "=========================================="
echo "✅ ALL EVALUATION COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  Individual evaluations: results/xlmr-es-en/, results/mbert-es-en/, etc."
echo "  CS density analysis: results/cs_density_es-en/, results/cs_density_hi-en/"
echo "   Error analysis: results/error_analysis_es-en/, results/error_analysis_hi-en/"
echo "   Model comparison: results/comparison/"
echo ""
