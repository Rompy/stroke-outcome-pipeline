"""
Stroke Outcome Prediction Pipeline
===================================

A privacy-preserving, end-to-end pipeline for extracting structured data
from clinical notes and predicting stroke outcomes.

Modules:
    data_preprocessing: Text preprocessing and tokenization
    model_finetuning: LoRA fine-tuning of Llama 3 8B
    extraction_pipeline: Clinical data extraction
    validation: Multi-tiered validation framework
    prediction: Outcome prediction models

Example:
    >>> from src.extraction_pipeline import ClinicalDataExtractor
    >>> extractor = ClinicalDataExtractor()
    >>> extractor.load_model()
    >>> extracted = extractor.extract(clinical_note)
"""

__version__ = "1.0.0"
__author__ = "Junsu Kim, Ji Hoon Kim, Arom Choi"
__email__ = "aromchoi@yuhs.ac"

from . import data_preprocessing
from . import model_finetuning
from . import extraction_pipeline

__all__ = [
    'data_preprocessing',
    'model_finetuning',
    'extraction_pipeline',
    'validation',
    'prediction'
]
