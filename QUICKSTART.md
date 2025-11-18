# Quick Start Guide

Get started with the Stroke Outcome Pipeline in 10 minutes!

## Prerequisites

- Python 3.10 or higher
- 16GB RAM (recommended: 64GB)
- Basic knowledge of Python and command line

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/stroke-outcome-pipeline.git
cd stroke-outcome-pipeline
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Or install as package (editable mode)
pip install -e .
```

## Quick Test (5 minutes)

### Generate Synthetic Data

```bash
# Generate 50 synthetic patients for testing
python scripts/generate_synthetic_data.py
```

This creates:
- `data/synthetic_stroke_data_notes.json` - Clinical notes
- `data/synthetic_stroke_data_structured.json` - Ground truth
- `data/synthetic_stroke_data_outcomes.csv` - 3-month outcomes

### Test Data Extraction

```python
from src.data_preprocessing import ClinicalTextPreprocessor
from src.extraction_pipeline import ClinicalDataExtractor
import json

# Load synthetic note
with open('data/test_set_notes.json', 'r') as f:
    notes = json.load(f)

# Preprocess
preprocessor = ClinicalTextPreprocessor(
    schema_path='config/extraction_schema.json'
)
processed = preprocessor.preprocess(notes[0])

print(f"Token count: {processed['token_count']}")
print(f"Character count: {processed['char_count']}")
```

### Test Validation

```python
from src.validation.rule_based import RuleBasedValidator

# Sample extracted data
extracted = {
    'age': 68,
    'sex': 'male',
    'hypertension': 'yes',
    'initial_nihss': 12,
    'aspect_score': 8,
    'mri_finding': 'acute_infarction',
    'iv_tpa': 'yes'
}

# Validate
validator = RuleBasedValidator(
    schema_path='config/extraction_schema.json'
)
validated, errors = validator.validate(extracted)

print(f"Validation errors: {len(errors)}")
print(f"Validated data: {validated}")
```

## Full Pipeline (Without Fine-tuning)

You can test the pipeline structure without a fine-tuned model:

```python
import json
from src.data_preprocessing import ClinicalTextPreprocessor
from src.validation.rule_based import RuleBasedValidator
from src.validation.rag_verification import RAGVerifier
from src.validation.cosine_similarity import CosineSimilarityFlagger
from src.prediction.outcome_predictor import StrokeOutcomePredictor

# 1. Load data
with open('data/test_set_notes.json', 'r') as f:
    notes = json.load(f)

with open('data/test_set_structured.json', 'r') as f:
    ground_truth = json.load(f)

# 2. Preprocess
preprocessor = ClinicalTextPreprocessor('config/extraction_schema.json')
processed_notes = [preprocessor.preprocess(note) for note in notes]

# 3. For testing, use ground truth as "extracted" data
extracted_data = ground_truth

# 4. Validate
rule_validator = RuleBasedValidator('config/extraction_schema.json')
validated_data = []

for data in extracted_data:
    validated, errors = rule_validator.validate(data)
    validated_data.append(validated)
    print(f"Errors: {len(errors)}")

# 5. RAG verification (on first note as example)
rag_config = {
    'embedding_model': 'intfloat/multilingual-e5-large',
    'embedding_dim': 1024,
    'top_k': 3,
    'similarity_threshold': 0.7
}
rag_verifier = RAGVerifier(rag_config)
verified, score = rag_verifier.verify(validated_data[0], notes[0])
print(f"Grounding score: {score:.3f}")

# 6. Prepare for prediction
predictor = StrokeOutcomePredictor('config/model_config.yaml')

import pandas as pd
outcomes_df = pd.read_csv('data/test_set_outcomes.csv')

X, y = predictor.prepare_features(validated_data, outcomes_df['poor_outcome'].tolist())
print(f"Feature matrix: {X.shape}")
print(f"Features: {X.columns.tolist()}")
```

## Training Your Own Model

### Fine-tune Extraction Model (Requires GPU)

```python
from src.model_finetuning import StrokeLLMFineTuner, create_synthetic_training_data

# Initialize
finetuner = StrokeLLMFineTuner(config_path='config/model_config.yaml')

# Load base model (requires ~16GB RAM)
finetuner.load_base_model()

# Configure LoRA
finetuner.configure_lora()

# Prepare training data (450 samples as per paper)
training_records = create_synthetic_training_data(n_samples=450)
train_dataset = finetuner.prepare_training_data(training_records)

# Train (takes ~6 hours on M2 Max)
finetuner.train(
    train_dataset, 
    output_dir='./stroke_lora_adapters'
)
```

### Train Prediction Models

```python
from src.prediction.outcome_predictor import main_prediction_pipeline

# Train all three models (LR, CatBoost, TabPFN)
models, results = main_prediction_pipeline(
    extracted_data_path='data/synthetic_stroke_data_structured.json',
    outcomes_path='data/synthetic_stroke_data_outcomes.csv',
    output_dir='./models'
)

print(results)
```

## Expected Results

### With Synthetic Data

- Extraction: Will show pipeline structure
- Validation: Should catch format errors
- Prediction: AUROC ~0.7-0.8 (synthetic data is simplified)

### With Real Data (Your Institution)

- Extraction: 97.0% accuracy (after all validation stages)
- RAG Grounding: 93.2% accuracy
- TabPFN AUROC: 0.816 (95% CI: 0.784-0.847)

## Next Steps

1. **Read the full README.md** for detailed information
2. **Explore notebooks/** for interactive tutorials
3. **Adapt to your data** format and variables
4. **Fine-tune with your annotated records**
5. **Deploy in your institution**

## Troubleshooting

### Out of Memory

- Use smaller batch size in config
- Enable CPU offloading
- Use quantization (already enabled)

### Model Download Issues

```bash
# Pre-download models
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')"
```

### Import Errors

```bash
# Reinstall with all dependencies
pip install -r requirements.txt --force-reinstall
```

## Getting Help

- 📖 Check [README.md](README.md) for full documentation
- 🐛 Open an issue on GitHub
- 📧 Email: aromchoi@yuhs.ac

## Success! 🎉

You're now ready to use the Stroke Outcome Pipeline. Start with synthetic data, then adapt to your institutional data when ready.
