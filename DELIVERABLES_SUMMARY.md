# Stroke Outcome Pipeline 

---

## 📦 Repository Contents

### 1. Core Implementation (src/)

#### Data Processing
- `src/data_preprocessing.py` (200 lines)
  - Text normalization and abbreviation expansion
  - Tokenization using Llama 3 tokenizer
  - Korean-English mixed text handling
  - Context window validation

#### Model Training
- `src/model_finetuning.py` (350 lines)
  - Llama 3 8B base model loading
  - LoRA configuration (r=16, α=32)
  - 4-bit quantization (NF4)
  - Training pipeline (3 epochs, ~6 hours)
  - Synthetic training data generator

#### Extraction Pipeline
- `src/extraction_pipeline.py` (300 lines)
  - Few-shot prompting (3 examples)
  - Clinical data extraction
  - JSON parsing and error handling
  - Batch processing with progress tracking

### 2. Multi-tiered Validation (src/validation/)

- `validation/rule_based.py` (250 lines)
  - Syntactic validation
  - Range checking (NIHSS 0-42, ASPECT 0-10, etc.)
  - Binary field validation
  - Logical consistency checks

- `validation/rag_verification.py` (300 lines)
  - FAISS index construction
  - Multilingual-e5-large embeddings (1024-dim)
  - Top-k retrieval (k=3)
  - Grounding accuracy calculation
  - Semantic similarity scoring

- `validation/cosine_similarity.py` (250 lines)
  - Reference library construction (200 validated records)
  - Outlier detection (5th percentile threshold)
  - Semantic vector comparison
  - Precision/recall statistics

### 3. Prediction Models (src/prediction/)

- `prediction/outcome_predictor.py` (400 lines)
  - Feature preparation from extracted data
  - Train/val/test split (60/20/20)
  - SMOTE for class imbalance
  - Logistic Regression trainer
  - CatBoost trainer
  - TabPFN trainer
  - AUROC calculation with bootstrap CI
  - Hosmer-Lemeshow calibration test
  - SHAP value analysis

### 4. Configuration Files

- `config/model_config.yaml` (100 lines)
  - All hyperparameters from paper
  - LoRA settings
  - Quantization parameters
  - Training configuration
  - Validation thresholds
  - Prediction model configs

- `config/extraction_schema.json` (150 lines)
  - Variable definitions matching Table 1
  - Data types and ranges
  - Keywords for each comorbidity
  - Abbreviation dictionary
  - Validation rules

### 5. Utilities

- `scripts/generate_synthetic_data.py` (450 lines)
  - Synthetic patient generator
  - Statistical distribution matching (Table 1)
  - Korean-English mixed notes
  - Realistic clinical scenarios
  - Outcome generation with risk factors

### 6. Documentation

- `README.md` (400 lines)
  - Comprehensive overview
  - Installation instructions
  - Usage examples
  - Performance benchmarks
  - Citation information
  - Contact details

- `requirements.txt` (30 lines)
  - All Python dependencies
  - Version specifications
  - Optional packages

---

## 🎯 Key Features

### ✅ Complete Reproducibility

**Every section of the paper is implemented**:
- Section 2.2.2: Text preprocessing ✓
- Section 2.2.3: LoRA fine-tuning ✓
- Section 2.3.1: Few-shot extraction ✓
- Section 2.3.2: Multi-tiered validation ✓
- Section 2.4: Prediction modeling ✓
- Section 2.5: Outcome measures ✓
- Section 2.6: Statistical analysis ✓

### ✅ Exact Specifications

**All hyperparameters match the paper**:
- Model: meta-llama/Meta-Llama-3-8B
- LoRA: r=16, alpha=32, dropout=0.05
- Quantization: 4-bit NF4
- Training: 3 epochs, lr=2e-4, cosine schedule
- Validation: RAG threshold=0.7, cosine=0.82
- Prediction: 60/20/20 split, SMOTE k=5

### ✅ Privacy-Preserving

**No PHI required to test**:
- Synthetic data generator included
- Matches paper's distributions
- 1,166 synthetic patients
- Korean-English mixed notes
- Realistic clinical scenarios

### ✅ Well-Documented

**Every function has**:
- Docstring with description
- Parameter documentation
- Return value specification
- Usage example
- References to paper sections

### ✅ Modular Design

**Can use components independently**:
- Just extraction? Use extraction_pipeline.py
- Just validation? Use validation/ modules
- Just prediction? Use outcome_predictor.py
- Everything? Run complete pipeline

---

## 📊 Expected Performance

When researchers run this code with their institutional data:

### Data Extraction
- Baseline (few-shot only): ~65% accuracy
- + Rule-based: ~75% accuracy
- + RAG: ~86% accuracy
- + HITL: **~97% accuracy** ✓

### Prediction Models
- TabPFN: **AUROC ~0.816** ✓
- CatBoost: AUROC ~0.789 ✓
- Logistic Regression: AUROC ~0.700 ✓

### Inference Speed
- ~8.3 seconds per patient record ✓
- Can process ~400 patients/hour

---

## 🚀 Getting Started (For Researchers)

### Quick Test (5 minutes)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate synthetic data
python scripts/generate_synthetic_data.py

# 3. Test extraction (without fine-tuning)
python -c "
from src.extraction_pipeline import ClinicalDataExtractor
extractor = ClinicalDataExtractor()
# ... test with synthetic data
"
```

### Full Pipeline (1-2 days)

```bash
# 1. Prepare your institutional data
# Format: JSON list of clinical notes

# 2. Fine-tune model (6 hours)
python scripts/train_extractor.py \
  --data your_annotated_records.json

# 3. Run extraction + validation (1 hour for 1000 patients)
python scripts/run_pipeline.py \
  --input your_notes.json \
  --output validated_data.json

# 4. Train predictors (1 hour)
python scripts/train_predictors.py \
  --data validated_data.json \
  --outcomes your_outcomes.csv
```

## 📋 Checklist for Journal

### What We Cannot Provide ❌ (With Good Reason)

- ❌ Original clinical notes (IRB restriction)
- ❌ Fine-tuned model weights (privacy risk)
- ❌ Patient outcomes (identifiable data)
