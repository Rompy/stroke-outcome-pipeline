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

- `README.md` 
  - Comprehensive overview
  - Installation instructions
  - Usage examples
  - Performance benchmarks

- `requirements.txt` (30 lines)
  - All Python dependencies
  - Version specifications
  - Optional packages
