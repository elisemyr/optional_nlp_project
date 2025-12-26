"""Streamlit Code-Mixed Sentiment Analysis"""
import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os
import numpy as np

st.set_page_config(page_title="Code-Mixed Sentiment", layout="wide")

# Sidebar
st.sidebar.header("Settings")
language = st.sidebar.selectbox("Language:", ["Spanish-English", "Hindi-English"])
model_choice = st.sidebar.selectbox("Model:", ["XLM-RoBERTa", "mBERT"])

language_map = {"Spanish-English": "es-en", "Hindi-English": "hi-en"}
lang_code = language_map[language]
model_code = "xlmr" if "RoBERTa" in model_choice else "mbert"
model_path = f"models/{model_code}-{lang_code}"

# Check if model exists
model_exists = os.path.exists(model_path) and os.path.exists(f"{model_path}/pytorch_model.bin" if os.path.exists(f"{model_path}/pytorch_model.bin") else f"{model_path}/model.safetensors")

# Calculate model size
def get_model_size(path):
    """Get model size in MB."""
    try:
        total = 0
        from pathlib import Path
        for f in Path(path).rglob('*'):
            if f.is_file():
                total += f.stat().st_size
        return total / (1024 * 1024)
    except:
        return 0

model_size_mb = get_model_size(model_path) if model_exists else 0

if model_exists:
    st.sidebar.success(f" Model found: {model_code}-{lang_code}")
    st.sidebar.metric("Model Size", f"{model_size_mb:.0f} MB")
    
    # Check if model was trained with improved script (has id2label config)
    try:
        import json
        config_path = f"{model_path}/config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'id2label' in config and config['id2label']:
                    st.sidebar.success(" Model has label mappings")
                else:
                    st.sidebar.warning("⚠️ Model may need retraining with class weights")
    except:
        pass
else:
    st.sidebar.error(f"❌ Model not found")
    st.sidebar.info("Will use base model from HuggingFace")

# Title
st.title("🌐 Code-Mixed Sentiment Analysis")
st.markdown(f"**{language}** • **{model_choice}**")

# Show info about model improvements if needed
if model_exists:
    try:
        import json
        config_path = f"{model_path}/config.json"
        needs_retrain = False
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'id2label' not in config or not config.get('id2label'):
                    needs_retrain = True
        
        if needs_retrain:
            st.info("""
            ⚠️ **Model Quality Notice**: This model may benefit from retraining with class weight balancing. 
            If you see neutral examples being misclassified as positive, consider retraining with the improved training script.
            See `RETRAIN_MODELS.md` for instructions.
            """)
    except:
        pass

# Examples
if language == "Spanish-English":
    examples = {
        "Positive": "Me encanta this movie! Best film ever",
        "Negative": "Esta comida is terrible, no me gusta",
        "Neutral": "The weather está okay today"
    }
else:  # Hindi-English
    examples = {
        "Positive": "Yeh movie bahut amazing hai!",
        "Negative": "Bahut kharab service yaar",
        "Neutral": "Movie was okay okay"
    }

# Input tabs
tab1, tab2 = st.tabs(["Text", "File"])

with tab1:
    selected = st.selectbox("Example:", [""] + list(examples.keys()))
    text = st.text_area("Enter text:", value=examples[selected] if selected else "", height=100)
    analyze_btn = st.button("🔍 Analyze", type="primary")

with tab2:
    uploaded = st.file_uploader("Upload .txt", type=['txt'])

# CACHED MODEL LOADING: Model loads once and stays in memory
@st.cache_resource(show_spinner="Loading model...")
def load_model_and_tokenizer(_model_path, _model_code):
    """
    Load model and tokenizer with caching.
    Model will only reload if model_path or model_code changes.
    
    Args:
        _model_path: Path to the model (underscore prefix for cache hashing)
        _model_code: Model code identifier (underscore prefix for cache hashing)
    
    Returns:
        Tuple of (model, tokenizer, source)
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    torch.set_num_threads(1)
    
    base_models = {
        "xlmr": "xlm-roberta-base",
        "mbert": "bert-base-multilingual-cased"
    }
    base_model = base_models[_model_code]
    
    try:
        # Try local fine-tuned model first
        model = AutoModelForSequenceClassification.from_pretrained(
            _model_path,
            num_labels=3,
            low_cpu_mem_usage=True,
            torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(_model_path, use_fast=True)
        source = "Fine-tuned"
    except Exception as e:
        # Fallback to base model from HuggingFace
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=3,
            low_cpu_mem_usage=True,
            torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        source = "Base model (HuggingFace)"
    
    model.eval()
    return model, tokenizer, source

# Initialize model loading indicator
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'current_model_key' not in st.session_state:
    st.session_state.current_model_key = None

# Create a unique key for the current model selection
current_model_key = f"{model_code}-{lang_code}"

# Check if model selection changed
if st.session_state.current_model_key != current_model_key:
    st.session_state.model_loaded = False
    st.session_state.current_model_key = current_model_key

# Load model (cached, so only loads once per model selection)
# The @st.cache_resource decorator handles the caching automatically
# It will show a spinner on first load, then use cached model on subsequent calls
model, tokenizer, source = load_model_and_tokenizer(model_path, model_code)

# Show model status in sidebar
if not st.session_state.model_loaded:
    st.sidebar.success(f"Model loaded: {source}")
    st.sidebar.caption("Model is now cached in memory")
    st.session_state.model_loaded = True
else:
    st.sidebar.success(f"Model cached: {source}")
    st.sidebar.caption("Fast inference")

def analyze_text(text, model, tokenizer):
    """Analyze single text with attention.
    
    Args:
        text: Input text to analyze
        model: Trained model
        tokenizer: Tokenizer
    """
    import torch
    
    if not text.strip():
        return None
    
    start = time.time()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, return_attention_mask=True)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = torch.argmax(probs).item()
        
        # Get attention weights (average across all heads and layers)
        attentions = outputs.attentions
        # Average attention across layers and heads
        avg_attention = torch.mean(torch.stack([torch.mean(att, dim=1) for att in attentions]), dim=0)
        # Get attention for [CLS] token to all other tokens
        attention_weights = avg_attention[0, 0, :].cpu().numpy()
    
    # Use model's label mapping if available, but map generic labels to sentiment labels
    # Training uses: 0=positive, 1=negative, 2=neutral
    if hasattr(model.config, 'id2label') and model.config.id2label:
        # Check if labels are generic (LABEL_0, LABEL_1, etc.) or actual sentiment labels
        sample_label = list(model.config.id2label.values())[0] if model.config.id2label else None
        if sample_label and ('LABEL_' in str(sample_label).upper() or 'Label_' in str(sample_label)):
            # Generic labels - map to sentiment labels based on training convention
            labels = {0: "Positive", 1: "Negative", 2: "Neutral"}
        else:
            # Try to use actual labels from config, but normalize them
            labels = {}
            for k, v in model.config.id2label.items():
                label_str = str(v).lower()
                if 'pos' in label_str or label_str == 'positive':
                    labels[int(k)] = "Positive"
                elif 'neg' in label_str or label_str == 'negative':
                    labels[int(k)] = "Negative"
                elif 'neu' in label_str or label_str == 'neutral':
                    labels[int(k)] = "Neutral"
                else:
                    # Fallback: use capitalized version
                    labels[int(k)] = str(v).capitalize()
            # Ensure we have all three labels
            if len(labels) < 3:
                labels = {0: "Positive", 1: "Negative", 2: "Neutral"}
    else:
        # Default mapping (consistent with training: 0=positive, 1=negative, 2=neutral)
        labels = {0: "Positive", 1: "Negative", 2: "Neutral"}
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Ensure prediction index is valid and labels dictionary is complete
    default_labels = {0: "Positive", 1: "Negative", 2: "Neutral"}
    if pred not in labels or len(labels) < 3:
        # Fallback: use default mapping if prediction index is out of range
        labels = default_labels
        pred = min(max(pred, 0), 2)  # Clamp to valid range [0, 2]
    
    # Ensure all three labels exist (fill missing ones with defaults)
    for i in range(3):
        if i not in labels:
            labels[i] = default_labels[i]
    
    # Calculate confidence metrics
    max_prob = float(probs[pred])
    probs_dict = {labels[i]: float(probs[i]) for i in range(3)}
    
    # Check if prediction is uncertain (low confidence or close probabilities)
    sorted_probs = sorted(probs_dict.values(), reverse=True)
    is_uncertain = max_prob < 0.6 or (sorted_probs[0] - sorted_probs[1] < 0.15)
    
    return {
        'sentiment': labels[pred],
        'confidence': max_prob,
        'probs': probs_dict,
        'time_ms': (time.time() - start) * 1000,
        'tokens': tokens,
        'attention': attention_weights,
        'is_uncertain': is_uncertain,
        'logits': logits[0].cpu().numpy().tolist()  # Include raw logits for debugging
    }

# Single text analysis
if analyze_btn and text:
    with st.spinner("Analyzing..."):
        try:
            # Model is already loaded and cached, no need to reload!
            result = analyze_text(text, model, tokenizer)
            
            if result:
                st.markdown("---")
                st.subheader("Results")
                
                emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
                
                # Show warning if prediction is uncertain
                if result.get('is_uncertain', False):
                    st.warning("**Low confidence prediction** - The model is uncertain. Check probabilities below.")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Sentiment", f"{result['sentiment']} {emoji[result['sentiment']]}")
                col2.metric("Confidence", f"{result['confidence']:.1%}")
                col3.metric("Time", f"{result['time_ms']:.0f} ms")
                
                # Show all probabilities in an expander for transparency
                with st.expander("View All Probabilities", expanded=result.get('is_uncertain', False)):
                    prob_col1, prob_col2, prob_col3 = st.columns(3)
                    for i, (label, prob) in enumerate(result['probs'].items()):
                        col = [prob_col1, prob_col2, prob_col3][i]
                        with col:
                            st.metric(
                                label,
                                f"{prob:.1%}",
                                delta=f"{prob - (1/3):.1%}" if prob > (1/3) else None,
                                delta_color="normal"
                            )
                
                
                # Bar chart
                fig = px.bar(
                    x=list(result['probs'].keys()),
                    y=list(result['probs'].values()),
                    labels={'x': 'Sentiment', 'y': 'Probability'},
                    title="Confidence Distribution"
                )
                fig.update_traces(text=[f"{v:.1%}" for v in result['probs'].values()], textposition='outside')
                fig.update_layout(showlegend=False, yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
                
                # Attention visualization
                st.subheader("Attention Weights & Token Highlighting")
                
                # Create highlighted text
                tokens = result['tokens']
                attention = result['attention']
                
                # Clean tokens (remove special tokens and word pieces)
                clean_tokens = []
                clean_attention = []
                for i, (tok, att) in enumerate(zip(tokens, attention)):
                    if tok not in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]']:
                        clean_tok = tok.replace('▁', '').replace('##', '')
                        if clean_tok:
                            clean_tokens.append(clean_tok)
                            clean_attention.append(att)
                
                # Normalize attention for visualization
                if len(clean_attention) > 0:
                    min_att = min(clean_attention)
                    max_att = max(clean_attention)
                    range_att = max_att - min_att if max_att > min_att else 1
                    
                    # Create HTML with colored tokens
                    html = "<div style='line-height: 2.5; font-size: 16px;'>"
                    for tok, att in zip(clean_tokens, clean_attention):
                        # Normalize attention to 0-1
                        norm_att = (att - min_att) / range_att if range_att > 0 else 0.5
                        opacity = 0.2 + norm_att * 0.8
                        
                        # Color based on sentiment
                        if result['sentiment'] == "Positive":
                            color = f"rgba(46, 204, 113, {opacity})"  # Green
                        elif result['sentiment'] == "Negative":
                            color = f"rgba(231, 76, 60, {opacity})"   # Red
                        else:
                            color = f"rgba(52, 152, 219, {opacity})"  # Blue
                        
                        html += f"<span style='background-color: {color}; padding: 4px 6px; margin: 2px; border-radius: 4px; display: inline-block;'>{tok}</span> "
                    html += "</div>"
                    
                    st.markdown("**Important words highlighted (darker = more important):**")
                    st.markdown(html, unsafe_allow_html=True)
                    st.caption(f"📊 Total tokens: {len(clean_tokens)} | Attention range: [{min_att:.3f}, {max_att:.3f}]")
                
                # Performance metrics
                st.subheader("⚡ Performance Metrics")
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                perf_col1.metric("Inference Time", f"{result['time_ms']:.1f} ms")
                perf_col2.metric("Model Size", f"{model_size_mb:.0f} MB" if model_size_mb > 0 else "N/A")
                perf_col3.metric("Tokens Processed", len(clean_tokens))
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# File analysis
if uploaded:
    try:
        lines = [l.strip() for l in uploaded.read().decode('utf-8').split('\n') if l.strip()]
        st.info(f"Loaded {len(lines)} lines")
        
        if st.button("Analyze File"):
            with st.spinner("Processing file..."):
                # Model is already loaded and cached, no need to reload!
                results = []
                progress = st.progress(0)
                
                for i, line in enumerate(lines):
                    r = analyze_text(line, model, tokenizer)
                    if r:
                        results.append({**r, 'text': line})
                    progress.progress((i+1)/len(lines))
                
                if results:
                    st.success(f"✅ Analyzed {len(results)} texts")
                    
                    # Stats
                    sentiments = [r['sentiment'] for r in results]
                    counts = pd.Series(sentiments).value_counts()
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Avg Confidence", f"{sum(r['confidence'] for r in results)/len(results):.1%}")
                    col2.metric("Most Common", counts.index[0])
                    
                    # Pie chart
                    fig = px.pie(values=counts.values, names=counts.index, title="Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Table
                    df = pd.DataFrame([{
                        'Text': r['text'][:50] + '...',
                        'Sentiment': r['sentiment'],
                        'Confidence': f"{r['confidence']:.1%}"
                    } for r in results])
                    st.dataframe(df, use_container_width=True)
                    
                    # Download
                    st.download_button(" Download", df.to_csv(index=False), f"results.csv")
    except Exception as e:
        st.error(f"File error: {e}")

st.markdown("---")
