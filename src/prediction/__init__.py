"""
Stroke Outcome Prediction Models
=================================

Predicts poor neurological outcome (mRS 3-6) at 3 months using:
- Logistic Regression (interpretable baseline)
- CatBoost (gradient boosting)
- TabPFN (transformer-based, best performance)

Expected performance:
- TabPFN: AUROC 0.816 (95% CI: 0.784-0.847)
- CatBoost: AUROC 0.789 (95% CI: 0.756-0.822)
- Logistic Regression: AUROC 0.700 (95% CI: 0.665-0.735)

Example:
    >>> from prediction.outcome_predictor import StrokeOutcomePredictor
    >>> 
    >>> predictor = StrokeOutcomePredictor()
    >>> X, y = predictor.prepare_features(extracted_data, outcomes)
    >>> 
    >>> splits = predictor.split_data(X, y)
    >>> models = predictor.train_all_models(splits)
    >>> 
    >>> metrics = predictor.evaluate_model(models['TabPFN'], X_test, y_test)
"""

from .outcome_predictor import StrokeOutcomePredictor

__all__ = ['StrokeOutcomePredictor']
