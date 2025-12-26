# Multi-Lingual Sentiment Analysis with Code-Switching

---
## Installation

### Prerequisites

* **Python 3.8+** (Python 3.10 recommended)
* **CUDA 11.8+** (for GPU acceleration, optional but highly recommended)
* **8GB RAM minimum** (16GB recommended for training)
* **10GB disk space** for datasets and models
* **Git** for cloning the repository

### Setup

**Step 1: Clone the repository**

```bash
git clone https://github.com/elisemyr/optional_nlp_project.git
cd optional_nlp_project
```

**Step 2: Create a virtual environment (recommended)**

```bash
# Using venv on MAC OS X
python -m venv venv
source venv/bin/activate  

```

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Download spaCy language models**

```bash
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
```

**Step 5: Download datasets**

```bash
# Download both language pairs using our script
python src/data/download_datasets.py
```

**Step 6: Preprocess data**

```bash
python src/preprocessing.py --input data/raw --output data/processed
```

---

## Dataset

### Overview

This project uses two code-mixed sentiment analysis datasets covering Spanish-English and Hindi-English language pairs. Both datasets contain social media text (primarily tweets) with token-level language identification tags and sentence-level sentiment labels.

### Spanish-English: LinCE Benchmark (Sentiment Analysis Track)

**Source**: [LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation](https://ritual.uh.edu/lince/)  
**Paper**: [Aguilar et al., 2020](https://arxiv.org/abs/2005.04322)

| Split | Size | Positive | Negative | Neutral |
|-------|------|----------|----------|---------|
| Train | 15,023 | 6,009 (40%) | 4,507 (30%) | 4,507 (30%) |
| Dev   | 1,883 | 753 (40%) | 565 (30%) | 565 (30%) |
| Test  | 1,883 | 753 (40%) | 565 (30%) | 565 (30%) |
| **Total** | **18,789** | **7,515** | **5,637** | **5,637** |


### Hindi-English: SemEval-2020 Task 9 (SentiMix)

**Source**: [SemEval-2020 Task 9: SentiMix](https://competitions.codalab.org/competitions/20654)  
**Paper**: [Patwa et al., 2020](https://arxiv.org/abs/2008.04277)

| Split | Size | Positive | Negative | Neutral |
|-------|------|----------|----------|---------|
| Train | 12,000 | 5,040 (42%) | 3,360 (28%) | 3,600 (30%) |
| Dev   | 1,500 | 630 (42%) | 420 (28%) | 450 (30%) |
| Test  | 1,500 | 630 (42%) | 420 (28%) | 450 (30%) |
| **Total** | **15,000** | **6,300** | **4,200** | **4,500** |


### Preprocessing Steps

Our preprocessing pipeline handles the unique challenges of code-mixed text:

1. **Text Normalization**:
   * Lowercase conversion
   * URL replacement with `<URL>` token
   * Username replacement with `<USER>` token
   * Emoji standardization

2. **Script Handling**:
   * Devanagari detection for Hindi-English
   * Script separation and labeling
   * Transliteration detection

3. **Token-Level Language Identification**:
   * Leverage pre-existing language tags from datasets
   * Language tags: `lang1` (primary language), `lang2` (secondary language), `other`, `mixed`, `ambiguous`
   * Used for code-switching density calculation

4. **Code-Switching Metrics**:
   * **CS-Index calculation**: `CS-Index = (# of switch points) / (# of tokens - 1)`
   * Example: `['en', 'en', 'es', 'es', 'en']` → 2 switches / 4 positions = 0.5
   * Categorization: Low (0-0.3), Medium (0.3-0.6), High (0.6+)

5. **Tokenization**:
   * Subword tokenization using model-specific tokenizers
   * Max length: 128 tokens (covers 95%+ of tweets)
   * Padding and truncation applied

### Download Instructions

 Use our download script
```bash
python src/data/download_datasets.py --languages es-en hi-en
```

---

## Model Architecture

### Overview

We implement and compare six models across two language pairs (12 total trained models) to systematically evaluate monolingual versus multilingual approaches for code-switched sentiment analysis.

### Model Descriptions

#### 1. BERT-base-uncased (Monolingual Baseline)

**Architecture**: 12-layer transformer encoder  
**Parameters**: 110M  
**Vocabulary**: 30K tokens (English-only)  
**Pre-training**: English Wikipedia + BookCorpus

**Purpose**: Establish monolingual baseline to demonstrate the necessity of multilingual models for code-switched text. Expected to perform poorly on non-English tokens.

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3,  # Positive, Negative, Neutral
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)
```

#### 2. mBERT (Multilingual BERT)

**Architecture**: 12-layer transformer encoder  
**Parameters**: 179M  
**Vocabulary**: 110K tokens (104 languages)  
**Pre-training**: Wikipedia in 104 languages

**Purpose**: First-generation multilingual model. Tests whether shared multilingual representations can handle code-switching without explicit code-switching training data.

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=3,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)
```

#### 3. XLM-RoBERTa-base (Best Multilingual Model)

**Architecture**: 12-layer transformer encoder  
**Parameters**: 279M  
**Vocabulary**: 250K tokens (100 languages)  
**Pre-training**: 2.5TB CommonCrawl in 100 languages

**Purpose**: State-of-the-art multilingual model with larger vocabulary and cleaner pre-training data. Expected to achieve the best performance on code-switched text.

```python
from transformers import XLMRobertaForSequenceClassification

model = XLMRobertaForSequenceClassification.from_pretrained(
    'xlm-roberta-base',
    num_labels=3,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)
```

### Hyperparameter Tuning

We performed systematic hyperparameter optimization focusing on learning rate and batch size:

**Final Configuration** (applied to all models for fair comparison):

```yaml
# Optimizer
optimizer: AdamW
learning_rate: 2e-5
weight_decay: 0.01
adam_epsilon: 1e-8
adam_beta1: 0.9
adam_beta2: 0.999

# Learning Rate Schedule
warmup_steps: 500
lr_scheduler_type: linear

# Training
num_epochs: 4
batch_size: 16
gradient_accumulation_steps: 2  # Effective batch size: 32
max_grad_norm: 1.0

# Regularization
hidden_dropout_prob: 0.1
attention_probs_dropout_prob: 0.1

# Data
max_sequence_length: 128
padding: max_length
truncation: true

# Training Strategy
evaluation_strategy: epoch
save_strategy: epoch
load_best_model_at_end: true
metric_for_best_model: f1_macro
greater_is_better: true

# Reproducibility
seed: 42
```

**Hyperparameter Search Results**:

| Learning Rate | Batch Size | Spanish-English F1 | Hindi-English F1 |
|---------------|------------|-------------------|------------------|
| 1e-5 | 16 | 0.716 | 0.693 |
| **2e-5** | **16** | **0.742** | **0.721** |
| 3e-5 | 16 | 0.728 | 0.709 |
| 2e-5 | 8 | 0.734 | 0.715 |
| 2e-5 | 32 | 0.738 | 0.718 |

Selected configuration: **lr=2e-5, batch_size=16** (best balance of performance and training stability)

---

## Training

### Quick Start

**Train all models for both language pairs**:

```bash
# Spanish-English models
python src/train.py --model bert-base-uncased --language es-en --config configs/bert_es-en.yaml
python src/train.py --model bert-base-multilingual-cased --language es-en --config configs/mbert_es-en.yaml
python src/train.py --model xlm-roberta-base --language es-en --config configs/xlmr_es-en.yaml

# Hindi-English models
python src/train.py --model bert-base-uncased --language hi-en --config configs/bert_hi-en.yaml
python src/train.py --model bert-base-multilingual-cased --language hi-en --config configs/mbert_hi-en.yaml
python src/train.py --model xlm-roberta-base --language hi-en --config configs/xlmr_hi-en.yaml
```

### Training Commands

**Train individual model**:

```bash
python src/train.py \
    --model xlm-roberta-base \
    --language es-en \
    --data_dir data/processed/es-en \
    --output_dir models/xlmr-es-en \
    --epochs 4 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --seed 42
```

**Train with custom config**:

```bash
python src/train.py --config configs/xlmr_es-en.yaml
```

**Train all models (automated)**:

```bash
# Trains all 6 models sequentially
bash scripts/train_all.sh
```

---

## Evaluation

### Evaluation Metrics

We use the following metrics to comprehensively assess model performance:

**Primary Metric**:
* **F1-Score (Macro)**: Harmonic mean of precision and recall, averaged across all classes without weighting by class frequency. This metric treats all sentiment classes equally, making it robust to class imbalance.

**Secondary Metrics**:
* **Accuracy**: Overall percentage of correct predictions
* **Per-Class F1**: F1-score for each sentiment class (Positive, Negative, Neutral)
* **Precision & Recall**: For each class
* **Confusion Matrix**: Visualization of prediction patterns

**Why These Metrics?**

1. **F1-Macro over Accuracy**: Code-mixed datasets have slight class imbalance. F1-macro ensures we evaluate performance across all sentiment classes fairly, not just overall correctness.

2. **Per-Class Analysis**: Reveals which sentiments are hardest to detect. Typically, Negative sentiment is most challenging in code-switched text due to nuanced expressions across languages.

3. **Confusion Matrix**: Helps identify systematic errors (e.g., Neutral often confused with Positive in both languages).

### Running Evaluation

**Evaluate trained model**:

```bash
python src/evaluate.py \
    --model_path models/xlmr-es-en \
    --data_path data/processed/es-en/test \
    --output_dir results/xlmr-es-en
```

**Evaluate all models**:

```bash
bash scripts/evaluate_all.sh
```

**Compare models**:

```bash
python src/compare_models.py \
    --models models/bert-es-en models/mbert-es-en models/xlmr-es-en \
    --data data/processed/es-en/test \
    --output results/comparison_es-en.csv
```

### Results Analysis

#### Overall Performance Comparison

| Model | Spanish-English F1 | Spanish-English Acc | Hindi-English F1 | Hindi-English Acc | Average F1 |
|-------|-------------------|---------------------|------------------|-------------------|------------|
| BERT (English-only) | 0.623 ± 0.012 | 0.647 ± 0.009 | 0.578 ± 0.015 | 0.603 ± 0.011 | 0.601 |
| mBERT | 0.712 ± 0.008 | 0.731 ± 0.007 | 0.684 ± 0.010 | 0.702 ± 0.008 | 0.698 |
| **XLM-RoBERTa** | **0.742 ± 0.006** | **0.761 ± 0.005** | **0.721 ± 0.008** | **0.738 ± 0.007** | **0.732** |

**Key Findings**:

1. **Multilingual models significantly outperform monolingual BERT** (+11.4% F1 for mBERT, +13.1% F1 for XLM-R on average)
2. **XLM-R achieves best performance** across both language pairs (+3.0% over mBERT on Spanish-English, +3.7% on Hindi-English)
3. **Spanish-English is slightly easier** than Hindi-English (+2.1% F1 with XLM-R), likely due to script mixing in Hindi-English
4. **Low variance** across runs indicates stable training


#### Code-Switching Density Impact

We analyzed model performance across different code-switching densities:

**Spanish-English (XLM-R)**:

| CS Density | Examples | F1 Score | Accuracy |
|------------|----------|----------|----------|
| Low (0-0.3) | 687 | 0.782 | 0.801 |
| Medium (0.3-0.6) | 932 | 0.741 | 0.761 |
| High (0.6+) | 264 | 0.684 | 0.703 |

**Hindi-English (XLM-R)**:

| CS Density | Examples | F1 Score | Accuracy |
|------------|----------|----------|----------|
| Low (0-0.3) | 534 | 0.763 | 0.778 |
| Medium (0.3-0.6) | 721 | 0.715 | 0.732 |
| High (0.6+) | 245 | 0.642 | 0.661 |

**Key Finding**: Performance degrades as code-switching density increases. High CS density examples show ~10% lower F1 than low CS examples across both languages.


### Evaluation Output Files

Each evaluation run generates:

```
results/xlmr-es-en/
├── metrics.json                # All metrics in JSON format
├── predictions.csv             # Predictions with confidence scores
├── confusion_matrix.png        # Confusion matrix visualization
├── per_class_metrics.csv       # Detailed per-class results
├── errors_by_cs_density.csv    # Performance by CS density bins
└── error_examples.txt          # Sample misclassifications
```

---

## Demo

### Running the Interactive Demo

**Start the Streamlit application**:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`
---

## Docker

### Build Docker Image

```bash
docker build -t sentiment-codemix:latest .
```

### Run Docker Container

**Run Streamlit demo**:

```bash
docker run -p 8501:8501 sentiment-codemix:latest
```

Open browser to `http://localhost:8501`

**Run with GPU support**:

```bash
docker run --gpus all -p 8501:8501 sentiment-codemix:latest
```

**Run API server**:

```bash
docker run -p 8000:8000 sentiment-codemix:latest python src/api/server.py
```

**Run with custom command**:

```bash
docker run -it sentiment-codemix:latest bash
# Inside container:
python src/evaluate.py --model_path /app/models/xlmr-es-en
```

### Docker Compose

For multi-service deployment:

```bash
# Start all services (demo + API)
docker-compose up

# Run in background
docker-compose up -d

# Stop services
docker-compose down
```

Services available:
* Demo: `http://localhost:8501`
* API: `http://localhost:8000`
* API docs: `http://localhost:8000/docs`

### Dockerfile Contents

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy models
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download es_core_news_sm

# Expose ports
EXPOSE 8501 8000

# Default command
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

---

## Project Structure

```
code-switch-sentiment-analysis/
├── data/
│   ├── raw/                           # Downloaded datasets
│   │   ├── es-en/                    # Spanish-English raw data
│   │   │   ├── train.json
│   │   │   ├── dev.json
│   │   │   └── test.json
│   │   └── hi-en/                    # Hindi-English raw data
│   │       ├── train.json
│   │       ├── dev.json
│   │       └── test.json
│   ├── processed/                     # Preprocessed data
│   │   ├── es-en/
│   │   │   ├── train.pkl
│   │   │   ├── dev.pkl
│   │   │   └── test.pkl
│   │   └── hi-en/
│   │       ├── train.pkl
│   │       ├── dev.pkl
│   │       └── test.pkl
│   └── scripts/
│       ├── download_datasets.py       # Dataset download script
│       └── compute_statistics.py      # Dataset statistics
│
├── models/                            # Trained models
│   ├── bert-es-en/                   # BERT Spanish-English
│   ├── bert-hi-en/                   # BERT Hindi-English
│   ├── mbert-es-en/                  # mBERT Spanish-English
│   ├── mbert-hi-en/                  # mBERT Hindi-English
│   ├── xlmr-es-en/                   # XLM-R Spanish-English
│   └── xlmr-hi-en/                   # XLM-R Hindi-English
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                # PyTorch Dataset classes
│   │   ├── preprocessing.py          # Preprocessing pipeline
│   │   └── language_identifier.py    # Token-level language ID
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sentiment_classifier.py   # Model wrapper classes
│   │   └── utils.py                  # Model utilities
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                  # Main training script
│   │   └── trainer.py                # Custom Trainer class
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py               # Evaluation script
│   │   ├── metrics.py                # Metric computation
│   │   ├── error_analysis.py         # Error analysis script
│   │   └── compare_models.py         # Model comparison
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── cs_density_analysis.py    # CS density impact
│   │   └── visualizations.py         # Plotting functions
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py                 # FastAPI server
│   │   └── inference.py              # Inference utilities
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py                # Logging configuration
│       └── config.py                 # Configuration management
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA for both languages
│   ├── 02_preprocessing_analysis.ipynb
│   ├── 03_model_experiments.ipynb    # Training experiments
│   ├── 04_error_analysis.ipynb       # Detailed error analysis
│   └── 05_cs_density_study.ipynb     # CS density analysis
│
├── configs/
│   ├── bert_es-en.yaml               # BERT Spanish-English config
│   ├── bert_hi-en.yaml               # BERT Hindi-English config
│   ├── mbert_es-en.yaml              # mBERT Spanish-English config
│   ├── mbert_hi-en.yaml              # mBERT Hindi-English config
│   ├── xlmr_es-en.yaml               # XLM-R Spanish-English config
│   └── xlmr_hi-en.yaml               # XLM-R Hindi-English config
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py         # Test preprocessing functions
│   ├── test_models.py                # Test model forward passes
│   ├── test_metrics.py               # Test evaluation metrics
│   └── test_api.py                   # Test API endpoints
│
├── results/                           # Evaluation results
│   ├── comparison_tables/
│   ├── confusion_matrices/
│   ├── error_analysis/
│   └── visualizations/
│
├── logs/                              # Training logs
│   └── training_YYYYMMDD_HHMMSS.log
│
├── scripts/
│   ├── train_all.sh                  # Train all models
│   ├── evaluate_all.sh               # Evaluate all models
│   └── run_full_pipeline.sh          # Complete pipeline
│
├── app.py                             # Streamlit demo application
├── Dockerfile                         # Docker configuration
├── docker-compose.yml                 # Docker Compose config
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
├── .gitignore
├── README.md                          # This file
└── report.pdf                         # Technical report (7-8 pages)
```

---

## Future Work

* Mandarin-English (different script mixing challenge)
* Arabic-French (different typological features)
* Fine-grained error analysis by linguistic phenomena (negation, sarcasm, idioms)
* Cross-lingual transfer experiments (train on Spanish-English, test on Italian-English)
* Social media platform comparison (Twitter vs Facebook vs WhatsApp)
* Emotion detection (joy, anger, sadness, etc.)
* Hate speech detection in code-mixed text
* Named entity recognition across language boundaries

---

## References


1. Aguilar, G., Kar, S., & Solorio, T. (2020). **LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation**. *Proceedings of LREC 2020*. [https://arxiv.org/abs/2005.04322](https://arxiv.org/abs/2005.04322)

2. Patwa, P., Aguilar, G., Kar, S., Pandey, S., PYKL, S., Gambäck, B., Chakraborty, T., Solorio, T., & Das, A. (2020). **SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets**. *Proceedings of SemEval 2020*. [https://arxiv.org/abs/2008.04277](https://arxiv.org/abs/2008.04277)

3. Patra, B. G., Das, D., & Das, A. (2018). **Sentiment Analysis of Code-Mixed Indian Languages: An Overview of SAIL_Code-Mixed Shared Task @ICON-2017**. *Proceedings of ICON 2017*. [https://arxiv.org/abs/1803.06745](https://arxiv.org/abs/1803.06745)


4. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. *Proceedings of NAACL 2019*. [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

5. Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). **Unsupervised Cross-lingual Representation Learning at Scale**. *Proceedings of ACL 2020*. [https://arxiv.org/abs/1911.02116](https://arxiv.org/abs/1911.02116)

6. Pires, T., Schlinger, E., & Garrette, D. (2019). **How Multilingual is Multilingual BERT?**. *Proceedings of ACL 2019*. [https://arxiv.org/abs/1906.01502](https://arxiv.org/abs/1906.01502)


7. Khanuja, S., Dandapat, S., Srinivasan, A., Sitaram, S., & Choudhury, M. (2020). **GLUECoS: An Evaluation Benchmark for Code-Switched NLP**. *Proceedings of ACL 2020*. [https://arxiv.org/abs/2004.12376](https://arxiv.org/abs/2004.12376)

8. Winata, G. I., Madotto, A., Wu, C. S., & Fung, P. (2019). **Code-Switched Language Models Using Neural Based Synthetic Data from Parallel Sentences**. *Proceedings of CoNLL 2019*. [https://arxiv.org/abs/1909.08582](https://arxiv.org/abs/1909.08582)

9. Lee, N., & Wang, Z. (2015). **Emotion Detection in Code-Switching Texts via Bilingual and Sentimental Information**. *Proceedings of ACL 2015*. 

10. Pratapa, A., Bhat, G., Choudhury, M., Sitaram, S., Dandapat, S., & Bali, K. (2018). **Language Modeling for Code-Mixing: The Role of Linguistic Theory based Synthetic Data**. *Proceedings of ACL 2018*. [https://aclanthology.org/P18-1143/](https://aclanthology.org/P18-1143/)

11. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention is All You Need**. *Advances in NeurIPS 2017*. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

12. Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Scao, T. L., Gugger, S., Drame, M., Lhoest, Q., & Rush, A. M. (2020). **Transformers: State-of-the-Art Natural Language Processing**. *Proceedings of EMNLP 2020*. [https://arxiv.org/abs/1910.03771](https://arxiv.org/abs/1910.03771)

---


## Contact

**Author**: Elise Deyris
**Email**: [elisecal.deyris@student-cs.fr]  
**Course**: NLP, LLM, TextSemantic Course
**Instructor**: Benjamin Dallard (benjamin.dallard@centralesupelec.fr)

