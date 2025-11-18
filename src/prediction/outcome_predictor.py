"""
Stroke Outcome Prediction Module
Based on Section 2.4: Stroke Outcome Prediction Modeling

Implements TabPFN, CatBoost, and Logistic Regression for predicting
poor neurological outcomes (mRS 3-6) at 3 months.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from imblearn.over_sampling import SMOTE
import joblib


class StrokeOutcomePredictor:
    """
    Predict stroke outcomes using automatically extracted clinical data.
    
    Paper specifications (Section 2.4):
    - Target: Poor outcome (mRS 3-6) at 3 months
    - Models: Logistic Regression, CatBoost, TabPFN
    - Train/Val/Test split: 60%/20%/20%
    - SMOTE for class imbalance
    - Features: age, sex, NIHSS, comorbidities, imaging, interventions
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Args:
            config_path: Path to model configuration
        """
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['prediction']
        
        self.models = {}
        self.feature_names = []
        self.performance_metrics = {}
    
    def prepare_features(
        self,
        extracted_data_list: List[Dict[str, Any]],
        outcomes: List[int]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix from extracted data.
        
        Paper features (Section 2.4):
        - Age, sex
        - Initial NIHSS score
        - Hypertension, diabetes mellitus, dyslipidemia
        - Atrial fibrillation
        - Previous cerebrovascular accident
        - Presence of acute infarction on MRI
        - ASPECT score
        - Administration of IV t-PA
        - Intra-arterial intervention
        
        Args:
            extracted_data_list: List of extraction dictionaries
            outcomes: List of outcomes (0 = good mRS 0-2, 1 = poor mRS 3-6)
            
        Returns:
            Tuple of (feature_dataframe, outcome_series)
        """
        features_list = []
        
        for record in extracted_data_list:
            # Demographic features
            age = record.get('age', np.nan)
            sex_male = 1 if record.get('sex', '').lower() == 'male' else 0
            
            # Clinical scores
            initial_nihss = record.get('initial_nihss', np.nan)
            aspect_score = record.get('aspect_score', np.nan)
            
            # Comorbidities (binary)
            hypertension = 1 if record.get('hypertension', '').lower() == 'yes' else 0
            diabetes = 1 if record.get('diabetes_mellitus', '').lower() == 'yes' else 0
            dyslipidemia = 1 if record.get('dyslipidemia', '').lower() == 'yes' else 0
            af = 1 if record.get('atrial_fibrillation', '').lower() == 'yes' else 0
            prior_stroke = 1 if record.get('prior_stroke', '').lower() == 'yes' else 0
            
            # Imaging
            mri_infarction = 1 if record.get('mri_finding', '').lower() == 'acute_infarction' else 0
            
            # Interventions
            iv_tpa = 1 if record.get('iv_tpa', '').lower() == 'yes' else 0
            ia_intervention = 1 if record.get('ia_intervention', '').lower() == 'yes' else 0
            
            feature_dict = {
                'age': age,
                'sex_male': sex_male,
                'initial_nihss': initial_nihss,
                'aspect_score': aspect_score,
                'hypertension': hypertension,
                'diabetes_mellitus': diabetes,
                'dyslipidemia': dyslipidemia,
                'atrial_fibrillation': af,
                'prior_stroke': prior_stroke,
                'mri_infarction': mri_infarction,
                'iv_tpa': iv_tpa,
                'ia_intervention': ia_intervention
            }
            
            features_list.append(feature_dict)
        
        # Create DataFrame
        X = pd.DataFrame(features_list)
        y = pd.Series(outcomes, name='poor_outcome')
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        return X, y
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Split data into train/validation/test sets.
        
        Paper: "60% training, 20% validation, 20% test with stratified split"
        
        Args:
            X: Feature matrix
            y: Outcome vector
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        random_state = self.config['random_state']
        
        # First split: 60% train, 40% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=0.4,
            random_state=random_state,
            stratify=y
        )
        
        # Second split: 20% val, 20% test (50/50 of temp)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"Data split:")
        print(f"  Train: {len(X_train)} samples ({100*len(X_train)/len(X):.1f}%)")
        print(f"  Val:   {len(X_val)} samples ({100*len(X_val)/len(X):.1f}%)")
        print(f"  Test:  {len(X_test)} samples ({100*len(X_test)/len(X):.1f}%)")
        print(f"\nClass distribution:")
        print(f"  Train: {y_train.mean():.3f} poor outcomes")
        print(f"  Val:   {y_val.mean():.3f} poor outcomes")
        print(f"  Test:  {y_test.mean():.3f} poor outcomes")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def apply_smote(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to handle class imbalance.
        
        Paper (Section 2.4):
        "Given the class imbalance, the Synthetic Minority Over-sampling 
        Technique (SMOTE) was applied to the training set."
        
        Args:
            X_train: Training features
            y_train: Training outcomes
            
        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        smote_config = self.config['smote']
        
        smote = SMOTE(
            sampling_strategy=smote_config['sampling_strategy'],
            k_neighbors=smote_config['k_neighbors'],
            random_state=self.config['random_state']
        )
        
        print(f"\nApplying SMOTE...")
        print(f"  Before: {len(X_train)} samples, {y_train.sum()} positive")
        
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"  After:  {len(X_resampled)} samples, {y_resampled.sum()} positive")
        
        return X_resampled, y_resampled
    
    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> LogisticRegression:
        """
        Train logistic regression model.
        
        Paper results (Table 3):
        AUROC: 0.700 (95% CI: 0.665–0.735)
        """
        lr_config = self.config['models']['logistic_regression']
        
        model = LogisticRegression(
            penalty=lr_config['penalty'],
            C=lr_config['C'],
            max_iter=lr_config['max_iter'],
            random_state=self.config['random_state']
        )
        
        print("\nTraining Logistic Regression...")
        model.fit(X_train, y_train)
        
        return model
    
    def train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ):
        """
        Train CatBoost model.
        
        Paper results (Table 3):
        AUROC: 0.789 (95% CI: 0.756–0.822)
        """
        from catboost import CatBoostClassifier
        
        cb_config = self.config['models']['catboost']
        
        model = CatBoostClassifier(
            iterations=cb_config['iterations'],
            learning_rate=cb_config['learning_rate'],
            depth=cb_config['depth'],
            l2_leaf_reg=cb_config['l2_leaf_reg'],
            random_seed=cb_config['random_seed'],
            verbose=False
        )
        
        print("\nTraining CatBoost...")
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False
        )
        
        return model
    
    def train_tabpfn(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ):
        """
        Train TabPFN model.
        
        Paper results (Table 3):
        AUROC: 0.816 (95% CI: 0.784–0.847) - BEST PERFORMANCE
        
        TabPFN is a transformer-based model for tabular data that
        requires no hyperparameter tuning.
        """
        from tabpfn import TabPFNClassifier
        
        pfn_config = self.config['models']['tabpfn']
        
        model = TabPFNClassifier(
            N_ensemble_configurations=pfn_config['N_ensemble_configurations'],
            device='cpu'  # Use 'cuda' if available
        )
        
        print("\nTraining TabPFN...")
        
        # TabPFN has limitations on dataset size
        # If dataset is too large, use subset or implement batching
        if len(X_train) > 1000:
            print("  Note: Using subset for TabPFN (max 1000 samples)")
            indices = np.random.choice(len(X_train), 1000, replace=False)
            X_train_subset = X_train.iloc[indices]
            y_train_subset = y_train.iloc[indices]
        else:
            X_train_subset = X_train
            y_train_subset = y_train
        
        model.fit(X_train_subset.values, y_train_subset.values)
        
        return model
    
    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test set.
        
        Metrics from Table 3:
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        - AUROC
        - AUPRC
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test outcomes
            model_name: Name of the model
            
        Returns:
            Dictionary of performance metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'auroc': roc_auc_score(y_test, y_pred_proba),
            'auprc': average_precision_score(y_test, y_pred_proba)
        }
        
        # Calculate 95% CI for AUROC using bootstrap
        auroc_ci = self._bootstrap_ci(y_test, y_pred_proba)
        metrics['auroc_ci_lower'] = auroc_ci[0]
        metrics['auroc_ci_upper'] = auroc_ci[1]
        
        return metrics
    
    def _bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for AUROC.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        np.random.seed(self.config['random_state'])
        
        aucs = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            try:
                auc = roc_auc_score(y_true[indices], y_pred_proba[indices])
                aucs.append(auc)
            except:
                pass
        
        alpha = 1 - confidence
        lower = np.percentile(aucs, 100 * alpha / 2)
        upper = np.percentile(aucs, 100 * (1 - alpha / 2))
        
        return lower, upper
    
    def calibration_test(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform Hosmer-Lemeshow goodness-of-fit test.
        
        Paper (Section 2.6):
        "Hosmer–Lemeshow test was conducted to evaluate the goodness-of-fit 
        of the models by statistically comparing observed versus expected 
        probabilities."
        
        All models passed: p > 0.05
        
        Args:
            y_true: True outcomes
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with calibration statistics
        """
        from scipy import stats
        
        # Group into deciles
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred_proba, bins[:-1])
        
        observed = []
        expected = []
        counts = []
        
        for i in range(1, n_bins + 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                obs = y_true[mask].sum()
                exp = y_pred_proba[mask].sum()
                observed.append(obs)
                expected.append(exp)
                counts.append(mask.sum())
        
        # Hosmer-Lemeshow chi-square statistic
        observed = np.array(observed)
        expected = np.array(expected)
        
        chi_square = np.sum((observed - expected)**2 / (expected + 1e-10))
        df = len(observed) - 2
        p_value = 1 - stats.chi2.cdf(chi_square, df)
        
        return {
            'chi_square': chi_square,
            'degrees_of_freedom': df,
            'p_value': p_value,
            'well_calibrated': p_value > 0.05
        }


def main_prediction_pipeline(
    extracted_data_path: str,
    outcomes_path: str,
    output_dir: str = "./models"
):
    """
    Complete prediction modeling pipeline.
    
    Args:
        extracted_data_path: Path to extracted clinical data (JSON)
        outcomes_path: Path to outcome labels (CSV)
        output_dir: Directory to save trained models
    """
    import json
    import os
    
    # Load data
    with open(extracted_data_path, 'r') as f:
        extracted_data = json.load(f)
    
    outcomes_df = pd.read_csv(outcomes_path)
    
    # Initialize predictor
    predictor = StrokeOutcomePredictor()
    
    # Prepare features
    X, y = predictor.prepare_features(extracted_data, outcomes_df['poor_outcome'].tolist())
    
    # Split data
    splits = predictor.split_data(X, y)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    # Apply SMOTE
    X_train_resampled, y_train_resampled = predictor.apply_smote(X_train, y_train)
    
    # Train models
    models = {}
    
    # Logistic Regression
    models['Logistic Regression'] = predictor.train_logistic_regression(
        X_train_resampled, y_train_resampled
    )
    
    # CatBoost
    models['CatBoost'] = predictor.train_catboost(
        X_train_resampled, y_train_resampled, X_val, y_val
    )
    
    # TabPFN
    models['TabPFN'] = predictor.train_tabpfn(
        X_train_resampled, y_train_resampled
    )
    
    # Evaluate all models
    print("\n" + "="*50)
    print("TEST SET PERFORMANCE")
    print("="*50)
    
    results = []
    for model_name, model in models.items():
        metrics = predictor.evaluate_model(model, X_test, y_test, model_name)
        results.append(metrics)
        
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1-Score:  {metrics['f1_score']:.3f}")
        print(f"  AUROC:     {metrics['auroc']:.3f} "
              f"({metrics['auroc_ci_lower']:.3f}-{metrics['auroc_ci_upper']:.3f})")
        print(f"  AUPRC:     {metrics['auprc']:.3f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f"{output_dir}/prediction_results.csv", index=False)
    
    # Save best model (TabPFN)
    joblib.dump(models['TabPFN'], f"{output_dir}/tabpfn_model.pkl")
    
    print(f"\nModels and results saved to {output_dir}/")
    
    return models, results_df


if __name__ == "__main__":
    print("Stroke Outcome Prediction Module")
    print("="*50)
    print("\nThis module implements the prediction models from Section 2.4")
    print("Expected AUROC: 0.816 (TabPFN), 0.789 (CatBoost), 0.700 (Logistic Regression)")
