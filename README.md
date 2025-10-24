# Financial Sentiment Analysis: Fine-Tuning FinBERT

A comprehensive project demonstrating the fine-tuning of FinBERT for financial sentiment analysis, achieving state-of-the-art performance on financial text classification.

## Project Overview

This project implements a complete pipeline for fine-tuning a pre-trained language model (FinBERT) for financial sentiment analysis. The model classifies financial text into three categories: positive, negative, and neutral sentiment.

### Key Features

- **Domain-Specific Model**: Uses FinBERT, a BERT model pre-trained on financial documents
- **Advanced Data Processing**: Implements quality-weighted training with multi-source data
- **Comprehensive Evaluation**: Includes baseline comparisons and hyperparameter optimization
- **Production-Ready**: Includes inference pipeline and model deployment artifacts

## Results Summary

- **Final Model Performance**: 97.9% accuracy, 0.968 F1-Macro score (HPO-tuned FinBERT)
- **Baseline Comparison**: Zero-shot FinBERT achieved 98.2% accuracy, 0.972 F1-Macro
- **Key Finding**: Fine-tuning on noisy data (S3_max) vs. clean evaluation shows domain-specific pre-training value
- **Model Size**: Optimized for production deployment

## Video Walkthrough

📹 **[Watch the Complete Project Walkthrough](http://bit.ly/4oAmiXV)**

This comprehensive video demonstrates the entire fine-tuning pipeline, including:
- Dataset preparation and quality-weighted training
- Model selection and baseline comparisons
- Hyperparameter optimization results
- Live demonstration of model inference
- Error analysis and performance insights

## Quick Start

### Prerequisites

```bash
pip install transformers datasets accelerate evaluate scikit-learn matplotlib seaborn torch
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the fine-tuned model
model_path = "path/to/finbert-sentiment-best"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Example prediction
text = "Operating profit rose to EUR 13.1 mn and guidance was upgraded."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
```

## Project Structure

```
├── src/
│   ├── fine_tuning.py          # Main implementation notebook
│   └── fine_tuning.ipynb        # Jupyter notebook version
├── data/
│   ├── raw/                     # Raw data files
│   ├── scenarios/              # Processed data splits
│   └── results/                # Experimental results
├── models/
│   └── finbert-sentiment-best/ # Final production model
└── README.md
```

## Methodology

### 1. Dataset Preparation
- **Multi-source Data**: Combines data from different agreement levels (AllAgree, 75Agree, 66Agree, 50Agree)
- **Quality Prioritization**: Implements agreement-based data prioritization
- **Gold Test Set**: Creates unbiased test set from highest-quality data (S1_clean)
- **Training Data**: Uses S3_max scenario (all data sources) for comprehensive training
- **Data Splitting**: Prevents data leakage with careful train/validation/test splits

### 2. Model Selection
- **Primary Model**: ProsusAI/finbert (domain-specific pre-training)
- **Baseline Model**: DistilBERT (generic model for comparison)
- **Architecture**: BERT-based sequence classification
- **Training Strategy**: Weighted training on noisy S3_max data, evaluation on clean gold test

### 3. Training Strategy
- **Weighted Training**: Custom WeightedTrainer for quality-based sample weighting
- **Hyperparameter Optimization**: Systematic search across learning rates, epochs, and warmup ratios
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Evaluation Metrics**: F1-Macro for imbalanced dataset handling

### 4. Advanced Techniques
- **Sample Weighting**: Prioritizes high-agreement samples during training
- **Label Smoothing**: Reduces overconfidence on noisy labels
- **Learning Rate Scheduling**: Linear warmup and decay
- **Gradient Clipping**: Prevents exploding gradients

## Key Results

### Performance Comparison

| Model | Test Accuracy | F1-Macro | Notes |
|-------|---------------|----------|-------|
| DistilBERT (Zero-shot) | 78.1% | 0.078 | Generic model baseline |
| DistilBERT (Fine-tuned) | 96.2% | 0.941 | Trained on S3_max noisy data |
| FinBERT (Zero-shot) | 98.2% | 0.972 | Domain-specific pre-training advantage |
| FinBERT (HPO-tuned) | 97.9% | 0.968 | Fine-tuned on S3_max, evaluated on clean test |

### Error Analysis
- **Primary Error Patterns**: Directionality context and financial jargon
- **Improvement Areas**: Inverted context understanding, domain-specific terminology
- **Future Enhancements**: Data augmentation for edge cases

## Technical Implementation

### Data Processing Pipeline
1. **Raw Data Parsing**: Handles multiple label formats and data sources
2. **Quality Filtering**: Removes duplicates with quality prioritization
3. **Strategic Splitting**: Creates gold test set and prevents leakage
4. **Tokenization**: Converts text to model-compatible format

### Training Configuration
- **Learning Rate**: 3e-5 (optimized through hyperparameter search)
- **Batch Size**: 16 (training), 32 (evaluation)
- **Epochs**: 3 (with early stopping)
- **Optimizer**: AdamW with linear warmup
- **Regularization**: Weight decay (0.01), label smoothing (0.05)

### Evaluation Framework
- **Metrics**: Accuracy, F1-Macro, Precision, Recall
- **Cross-Validation**: Stratified splits with consistent random seeds
- **Baseline Comparison**: Zero-shot and generic model baselines
- **Statistical Significance**: Comprehensive error analysis

## Usage Examples

### Single Text Prediction
```python
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
    return predicted_class.item(), probabilities[0].tolist()
```

### Batch Processing
```python
def predict_batch(texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        results.extend(predictions.tolist())
    return results
```

## Model Deployment

### Production Requirements
- **Memory**: ~500MB for model weights
- **Dependencies**: PyTorch, Transformers, NumPy
- **Hardware**: CPU or GPU inference supported
- **Latency**: ~50ms per prediction on CPU

### API Integration
```python
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
    
    return jsonify({
        'prediction': predicted_class.item(),
        'confidence': probabilities[0].max().item(),
        'probabilities': probabilities[0].tolist()
    })
```

## Reproducibility

### Environment Setup
```bash
# Create virtual environment
python -m venv finbert_env
source finbert_env/bin/activate  # On Windows: finbert_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set random seeds for reproducibility
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### Data Requirements
- Raw financial text data with sentiment labels
- Minimum 1000 samples per class for reliable training
- High-quality annotations (preferably >75% agreement)

## Future Improvements

### Short-term Enhancements
1. **Data Augmentation**: Synthetic data generation for edge cases
2. **Ensemble Methods**: Combining multiple model predictions
3. **Active Learning**: Iterative improvement with human feedback

### Long-term Research
1. **Multi-task Learning**: Joint sentiment and entity recognition
2. **Cross-domain Adaptation**: Transfer to other financial domains
3. **Real-time Learning**: Online adaptation to new data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

