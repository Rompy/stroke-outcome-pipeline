# End-to-End Pipeline for Stroke Outcome Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of the methods described in our paper:

**"End-to-End Pipeline Integrating Local Small-Scale Language Models and Machine Learning for Data Extraction and Stroke Outcome Prediction in Emergency Department"**

Published in *Computational and Structural Biotechnology Journal* (2025)

## 🎯 Overview

We present a **privacy-preserving, locally-deployable pipeline** that:

1. **Extracts structured data** from unstructured clinical notes using a fine-tuned Llama 3 8B model
2. **Validates extractions** through multi-tiered verification (rule-based, RAG, cosine similarity, human-in-the-loop)
3. **Predicts stroke outcomes** (mRS 3-6 at 3 months) using machine learning

### Key Results

- **Data Extraction**: 97.0% accuracy (95% CI: 95.7-98.3%) after validation
- **Outcome Prediction**: AUROC 0.816 (95% CI: 0.784-0.847) using TabPFN
- **Privacy**: Fully local deployment on consumer hardware (Apple M2 Max, 64GB RAM)
- **Inference Time**: 8.3 seconds per patient record

## 🏗️ Architecture

```
Clinical Notes → Preprocessing → sLLM Extraction → Multi-tiered Validation → Prediction Model → Risk Stratification
                                   (Llama 3 8B)    (Rule/RAG/Cosine/HITL)    (TabPFN/CatBoost/LR)
```

### Multi-tiered Validation Framework

1. **Rule-Based Verification**: Syntactic and range checks
2. **RAG Verification**: FAISS-indexed semantic grounding (91.5% grounding accuracy)
3. **Cosine Similarity Flagging**: Outlier detection (flags 8% of records)
4. **Human-in-the-Loop Review**: Final quality assurance (κ = 0.89 inter-rater agreement)

## 📋 Requirements

### Hardware
- **Minimum**: 16GB RAM, 4-core CPU
- **Recommended**: 64GB RAM, 12-core CPU/38-core GPU (Apple M2 Max or equivalent)
- **Optional**: NVIDIA GPU with 16GB+ VRAM for faster training

### Software
- Python 3.10+
- PyTorch 2.0+
- See `requirements.txt` for complete dependencies

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/stroke-outcome-pipeline.git
cd stroke-outcome-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.extraction_pipeline import ClinicalDataExtractor
from src.validation.rule_based import RuleBasedValidator
from src.validation.rag_verification import RAGVerifier
from src.prediction.outcome_predictor import StrokeOutcomePredictor

# 1. Extract structured data
extractor = ClinicalDataExtractor(
    adapter_path="path/to/fine_tuned_adapters"  # Optional
)

clinical_note = """
Patient: 65-year-old male with sudden onset left-sided weakness.
PMH: Hypertension, diabetes mellitus, atrial fibrillation.
NIHSS: 12. Brain MRI: acute right MCA infarction.
Treatment: IV t-PA administered.
"""

extracted_data = extractor.extract(clinical_note)

# 2. Validate extraction
validator = RuleBasedValidator()
validated_data, errors = validator.validate(extracted_data)

rag_verifier = RAGVerifier(config)
verified_data, grounding_score = rag_verifier.verify(validated_data, clinical_note)

# 3. Predict outcome
predictor = StrokeOutcomePredictor()
X, _ = predictor.prepare_features([verified_data], [])
risk_score = predictor.models['TabPFN'].predict_proba(X)[0, 1]

print(f"Predicted risk of poor outcome (mRS 3-6): {risk_score:.2%}")
```

### Interactive Notebooks

See `notebooks/` directory for step-by-step tutorials:

- `01_data_preparation.ipynb`: Data preprocessing and tokenization
- `02_model_finetuning.ipynb`: Fine-tuning Llama 3 with LoRA
- `03_extraction_demo.ipynb`: End-to-end extraction pipeline
- `04_prediction_modeling.ipynb`: Training prediction models

## 📊 Reproducing Paper Results

### Dataset

Due to institutional privacy policies (IRB approval 4-2025-0125), we cannot share the original clinical data. However, we provide:

- **Synthetic data generator** for testing the pipeline
- **Complete methodology** to replicate with your institutional data
- **Exact hyperparameters** used in the study

### Training the Extraction Model

```bash
python scripts/train_extractor.py \
    --training_data data/annotated_records.json \
    --output_dir models/stroke_extractor \
    --epochs 3 \
    --batch_size 4
```

**Expected training time**: ~6 hours on Apple M2 Max, ~3 hours on NVIDIA A100

### Running Full Validation Pipeline

```bash
python scripts/run_validation.py \
    --input_data data/clinical_notes.json \
    --output_dir results/validated_extractions \
    --enable_rag \
    --enable_cosine_flagging \
    --hitl_review_rate 0.1
```

### Training Prediction Models

```bash
python scripts/train_predictors.py \
    --extracted_data results/validated_extractions.json \
    --outcomes data/outcomes.csv \
    --output_dir models/predictors
```

**Expected performance**:
- TabPFN: AUROC 0.816 (95% CI: 0.784-0.847)
- CatBoost: AUROC 0.789 (95% CI: 0.756-0.822)
- Logistic Regression: AUROC 0.700 (95% CI: 0.665-0.735)

## 🔬 Method Details

### Fine-tuning Configuration

Based on Section 2.2.3 of the paper:

```yaml
lora:
  r: 16                    # Low-rank dimension
  lora_alpha: 32
  target_modules: [q_proj, v_proj, k_proj, o_proj]
  lora_dropout: 0.05

quantization:
  load_in_4bit: true       # Reduces memory from 16GB to 4GB
  bnb_4bit_quant_type: nf4

training:
  num_epochs: 3
  learning_rate: 2.0e-4
  optimizer: paged_adamw_8bit
  lr_scheduler_type: cosine
```

### Validation Thresholds

```yaml
validation:
  rag:
    top_k: 3
    similarity_threshold: 0.7    # Flag if below
  
  cosine_similarity:
    threshold_percentile: 5      # Flag bottom 5%
    similarity_threshold: 0.82
  
  hitl:
    flagged_sample_rate: 1.0     # Review all flagged
    random_sample_rate: 0.1      # Review 10% of non-flagged
```

## 📈 Performance Metrics

### Data Extraction (Table 2)

| Stage | Accuracy | F1-Score | Grounding Accuracy |
|-------|----------|----------|-------------------|
| Baseline (Few-shot) | 64.9% | 0.555 | - |
| + Rule-based | 74.8% | 0.701 | - |
| + RAG | 86.0% | 0.813 | 91.5% |
| + HITL (Final) | **97.0%** | **0.920** | **93.2%** |

### Outcome Prediction (Table 3)

| Model | Accuracy | AUROC | AUPRC | Calibration |
|-------|----------|-------|-------|-------------|
| Logistic Regression | 0.732 | 0.700 | 0.315 | ✓ (p>0.05) |
| CatBoost | 0.791 | 0.789 | 0.315 | ✓ (p>0.05) |
| **TabPFN** | **0.817** | **0.816** | **0.316** | ✓ (p>0.05) |

## 🔒 Privacy & Security

### Privacy-by-Design Features

- ✅ **Local deployment**: No cloud transmission of PHI
- ✅ **Consumer hardware**: Runs on departmental workstations
- ✅ **HIPAA/GDPR compliant**: No third-party data sharing
- ✅ **Quantization**: 4-bit model reduces memory footprint
- ✅ **Institutional sovereignty**: Full control over data and models

### Limitations

- **Cannot share**: Original clinical data, fine-tuned model weights
- **Can share**: Model architecture, training code, synthetic data
- **Generalizability**: Trained on Korean-language notes from single institution

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## ⚠️ Disclaimer

This software is intended for research purposes only and has not been approved for clinical use. Healthcare professionals should not rely on this tool for making clinical decisions without proper validation and regulatory approval.

---

**Keywords**: Electronic Health Records, Small-Scale Language Model, Stroke Outcome Prediction, Privacy-Preserving AI, LoRA, TabPFN, FAISS, RAG
