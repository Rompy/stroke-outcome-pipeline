"""
Rule-Based Verification Module
Based on Section 2.3.2: Multi-tiered Validation Framework - Layer 1

Implements syntactic and range validation using Python rules.
"""

import json
from typing import Dict, List, Tuple, Any
import re


class RuleBasedValidator:
    """
    First validation layer: rapid syntactic and range error detection.
    
    Paper description (Section 2.3.2):
    "A set of Python scripts implemented rules such as verifying that the 
    NIHSS score was an integer between 0 and 42, that binary variables 
    contained only 'yes'/'no'/'unknown' values, and that extracted dates 
    fell within plausible ranges."
    """
    
    def __init__(self, schema_path: str = "config/extraction_schema.json"):
        """
        Args:
            schema_path: Path to extraction schema defining valid ranges
        """
        with open(schema_path, 'r') as f:
            schema_data = json.load(f)
        
        self.schema = schema_data['extraction_schema']
        self.validation_rules = self._build_validation_rules()
    
    def _build_validation_rules(self) -> Dict[str, callable]:
        """Build validation functions for each field type."""
        
        def validate_integer_range(value, min_val, max_val, field_name):
            """Validate integer within range."""
            if value is None or value == 'unknown':
                return True, None
            
            try:
                int_val = int(value)
                if min_val <= int_val <= max_val:
                    return True, None
                else:
                    return False, f"{field_name} out of range: {int_val} (expected {min_val}-{max_val})"
            except (ValueError, TypeError):
                return False, f"{field_name} must be integer, got: {type(value)}"
        
        def validate_binary(value, field_name):
            """Validate binary yes/no/unknown."""
            valid_values = ['yes', 'no', 'unknown']
            if value is None:
                value = 'unknown'
            
            if isinstance(value, str) and value.lower() in valid_values:
                return True, None
            else:
                return False, f"{field_name} must be 'yes'/'no'/'unknown', got: {value}"
        
        def validate_categorical(value, valid_values, field_name):
            """Validate categorical variable."""
            if value is None or value == 'unknown':
                return True, None
            
            if isinstance(value, str) and value.lower() in [v.lower() for v in valid_values]:
                return True, None
            else:
                return False, f"{field_name} must be one of {valid_values}, got: {value}"
        
        return {
            'integer_range': validate_integer_range,
            'binary': validate_binary,
            'categorical': validate_categorical
        }
    
    def validate(self, extracted_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        """
        Validate extracted data against rules.
        
        Args:
            extracted_data: Raw extraction from LLM
            
        Returns:
            Tuple of (corrected_data, list_of_errors)
        """
        corrected_data = extracted_data.copy()
        errors = []
        
        # Check for parsing errors
        if extracted_data.get('_parsing_error'):
            errors.append({
                'field': '_global',
                'error': 'JSON parsing failed',
                'severity': 'critical'
            })
            return corrected_data, errors
        
        # Validate demographics
        if 'age' in extracted_data:
            valid, error = self.validation_rules['integer_range'](
                extracted_data['age'], 0, 120, 'age'
            )
            if not valid:
                errors.append({'field': 'age', 'error': error, 'severity': 'high'})
                corrected_data['age'] = None
        
        if 'sex' in extracted_data:
            valid, error = self.validation_rules['categorical'](
                extracted_data['sex'], ['male', 'female'], 'sex'
            )
            if not valid:
                errors.append({'field': 'sex', 'error': error, 'severity': 'high'})
                corrected_data['sex'] = 'unknown'
        
        # Validate comorbidities (all binary)
        binary_fields = [
            'hypertension', 'diabetes_mellitus', 'dyslipidemia',
            'atrial_fibrillation', 'prior_stroke', 'cardiovascular_disease',
            'malignancy', 'esrd', 'iv_tpa', 'ia_intervention'
        ]
        
        for field in binary_fields:
            if field in extracted_data:
                valid, error = self.validation_rules['binary'](
                    extracted_data[field], field
                )
                if not valid:
                    errors.append({'field': field, 'error': error, 'severity': 'medium'})
                    corrected_data[field] = 'unknown'
        
        # Validate NIHSS (critical field)
        if 'initial_nihss' in extracted_data:
            valid, error = self.validation_rules['integer_range'](
                extracted_data['initial_nihss'], 0, 42, 'initial_nihss'
            )
            if not valid:
                errors.append({'field': 'initial_nihss', 'error': error, 'severity': 'critical'})
                corrected_data['initial_nihss'] = None
        
        # Validate ASPECT score
        if 'aspect_score' in extracted_data:
            valid, error = self.validation_rules['integer_range'](
                extracted_data['aspect_score'], 0, 10, 'aspect_score'
            )
            if not valid:
                errors.append({'field': 'aspect_score', 'error': error, 'severity': 'high'})
                corrected_data['aspect_score'] = None
        
        # Validate MRI finding
        if 'mri_finding' in extracted_data:
            valid, error = self.validation_rules['categorical'](
                extracted_data['mri_finding'],
                ['acute_infarction', 'no_lesion', 'other'],
                'mri_finding'
            )
            if not valid:
                errors.append({'field': 'mri_finding', 'error': error, 'severity': 'high'})
                corrected_data['mri_finding'] = 'unknown'
        
        # Logical consistency checks
        # Example: If IV t-PA given, NIHSS should typically be > 0
        if corrected_data.get('iv_tpa') == 'yes':
            nihss = corrected_data.get('initial_nihss')
            if nihss is not None and nihss == 0:
                errors.append({
                    'field': 'logical_consistency',
                    'error': 'IV t-PA administered but NIHSS=0 (unusual)',
                    'severity': 'low'
                })
        
        # Format standardization
        # Ensure lowercase for categorical values
        for key, value in corrected_data.items():
            if isinstance(value, str):
                corrected_data[key] = value.lower().strip()
        
        return corrected_data, errors
    
    def get_validation_statistics(self, all_errors: List[List[Dict[str, str]]]) -> Dict[str, Any]:
        """
        Calculate validation statistics across multiple records.
        
        Args:
            all_errors: List of error lists from multiple validations
            
        Returns:
            Summary statistics
        """
        total_records = len(all_errors)
        total_errors = sum(len(errors) for errors in all_errors)
        
        # Count by severity
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        field_error_counts = {}
        
        for errors in all_errors:
            for error in errors:
                severity = error.get('severity', 'unknown')
                if severity in severity_counts:
                    severity_counts[severity] += 1
                
                field = error.get('field', 'unknown')
                field_error_counts[field] = field_error_counts.get(field, 0) + 1
        
        return {
            'total_records': total_records,
            'total_errors': total_errors,
            'avg_errors_per_record': total_errors / total_records if total_records > 0 else 0,
            'severity_distribution': severity_counts,
            'most_common_error_fields': sorted(
                field_error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


def test_rule_validator():
    """Test the rule-based validator with sample data."""
    
    validator = RuleBasedValidator(schema_path="../config/extraction_schema.json")
    
    # Test case 1: Valid data
    valid_data = {
        'age': 65,
        'sex': 'male',
        'hypertension': 'yes',
        'diabetes_mellitus': 'no',
        'initial_nihss': 12,
        'aspect_score': 8,
        'mri_finding': 'acute_infarction',
        'iv_tpa': 'yes'
    }
    
    corrected, errors = validator.validate(valid_data)
    print("Test 1 - Valid data:")
    print(f"  Errors found: {len(errors)}")
    print()
    
    # Test case 2: Invalid ranges
    invalid_data = {
        'age': 150,  # Out of range
        'sex': 'unknown_sex',  # Invalid category
        'initial_nihss': 50,  # Out of range (max 42)
        'hypertension': 'maybe',  # Invalid binary value
        'aspect_score': -1  # Out of range
    }
    
    corrected, errors = validator.validate(invalid_data)
    print("Test 2 - Invalid data:")
    print(f"  Errors found: {len(errors)}")
    for error in errors:
        print(f"  - {error['field']}: {error['error']} (severity: {error['severity']})")
    print()
    
    # Test case 3: Logical inconsistency
    inconsistent_data = {
        'age': 65,
        'sex': 'male',
        'initial_nihss': 0,
        'iv_tpa': 'yes'  # Unusual to give t-PA for NIHSS=0
    }
    
    corrected, errors = validator.validate(inconsistent_data)
    print("Test 3 - Logical inconsistency:")
    print(f"  Errors found: {len(errors)}")
    for error in errors:
        print(f"  - {error['field']}: {error['error']} (severity: {error['severity']})")


if __name__ == "__main__":
    test_rule_validator()
