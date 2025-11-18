"""
Cosine Similarity Flagging Module
Based on Section 2.3.2: Multi-tiered Validation Framework - Layer 3

Flags outlier records for human review based on semantic similarity.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json


class CosineSimilarityFlagger:
    """
    Third validation layer: semantic outlier detection.
    
    Paper description (Section 2.3.2):
    "The semantic vector of the full extracted JSON object was compared 
    against a library of 200 historical, validated records from a prior 
    quality improvement. Cosine similarity scores were calculated, and 
    records falling below the 5th percentile (similarity < 0.82) were 
    automatically flagged as potential outliers."
    
    "Approximately 8% of records were flagged at this stage, of which 
    62% were confirmed to contain errors upon human review."
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Cosine similarity configuration
        """
        self.threshold_percentile = config['threshold_percentile']
        self.similarity_threshold = config['similarity_threshold']
        
        # Load embedding model (same as RAG for consistency)
        self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
        
        # Historical validated records (would be loaded from file in production)
        self.reference_library = []
        self.reference_embeddings = None
    
    def load_reference_library(self, reference_records: List[Dict[str, Any]]) -> None:
        """
        Load library of validated historical records.
        
        Paper: "a library of 200 historical, validated records from a 
        prior quality improvement"
        
        Args:
            reference_records: List of validated extraction dictionaries
        """
        self.reference_library = reference_records
        
        # Convert records to text representation for embedding
        reference_texts = [self._record_to_text(record) for record in reference_records]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(reference_records)} reference records...")
        self.reference_embeddings = self.embedding_model.encode(
            reference_texts,
            convert_to_tensor=False,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        # Calculate empirical threshold at specified percentile
        # In practice, would use validation set to determine this
        print(f"Reference library loaded with {len(reference_records)} records")
    
    def _record_to_text(self, record: Dict[str, Any]) -> str:
        """
        Convert extracted record to text representation for embedding.
        
        Exclude metadata fields that start with '_' or end with '_score'.
        
        Args:
            record: Extracted data dictionary
            
        Returns:
            Text representation
        """
        # Filter out metadata fields
        filtered_record = {
            k: v for k, v in record.items()
            if not k.startswith('_') and not k.endswith('_score') 
            and not k.endswith('_context')
        }
        
        # Create structured text representation
        text_parts = []
        for key, value in sorted(filtered_record.items()):
            if value is not None and value != 'unknown':
                text_parts.append(f"{key}: {value}")
        
        return ", ".join(text_parts)
    
    def check(self, extracted_data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if record should be flagged based on similarity to reference library.
        
        Args:
            extracted_data: Extracted and validated data
            
        Returns:
            Tuple of (should_flag, max_similarity_score)
        """
        if self.reference_embeddings is None or len(self.reference_embeddings) == 0:
            # No reference library loaded - cannot flag
            return False, 1.0
        
        # Convert record to text and embed
        record_text = self._record_to_text(extracted_data)
        record_embedding = self.embedding_model.encode(
            [record_text],
            convert_to_tensor=False,
            normalize_embeddings=True
        )
        
        # Calculate cosine similarities with all reference records
        similarities = cosine_similarity(
            record_embedding,
            self.reference_embeddings
        )[0]
        
        # Get maximum similarity (closest match)
        max_similarity = float(np.max(similarities))
        
        # Flag if below threshold
        should_flag = max_similarity < self.similarity_threshold
        
        return should_flag, max_similarity
    
    def batch_check(
        self,
        extracted_data_list: List[Dict[str, Any]]
    ) -> Tuple[List[bool], List[float]]:
        """
        Check multiple records for flagging.
        
        Args:
            extracted_data_list: List of extracted data dictionaries
            
        Returns:
            Tuple of (flag_list, similarity_scores)
        """
        flags = []
        scores = []
        
        for record in extracted_data_list:
            should_flag, score = self.check(record)
            flags.append(should_flag)
            scores.append(score)
        
        return flags, scores
    
    def calculate_statistics(
        self,
        flags: List[bool],
        confirmed_errors: List[bool]
    ) -> Dict[str, Any]:
        """
        Calculate flagging statistics.
        
        Paper reports:
        "Approximately 8% of records were flagged at this stage, of which 
        62% were confirmed to contain errors upon human review."
        
        Args:
            flags: List of boolean flags
            confirmed_errors: List of boolean indicating actual errors
            
        Returns:
            Statistics dictionary
        """
        flags_array = np.array(flags)
        errors_array = np.array(confirmed_errors)
        
        total_records = len(flags)
        flagged_count = np.sum(flags_array)
        flagging_rate = flagged_count / total_records if total_records > 0 else 0.0
        
        # Of flagged records, how many had actual errors?
        if flagged_count > 0:
            true_positives = np.sum(flags_array & errors_array)
            precision = true_positives / flagged_count
        else:
            precision = 0.0
        
        # Of all errors, how many were flagged?
        error_count = np.sum(errors_array)
        if error_count > 0:
            recall = np.sum(flags_array & errors_array) / error_count
        else:
            recall = 0.0
        
        return {
            'total_records': total_records,
            'flagged_count': int(flagged_count),
            'flagging_rate': flagging_rate,
            'precision': precision,  # % of flagged that had errors
            'recall': recall,  # % of errors that were flagged
            'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        }


def create_synthetic_reference_library(n_records: int = 200) -> List[Dict[str, Any]]:
    """
    Create synthetic reference library for demonstration.
    
    In production, this would be loaded from validated historical records.
    
    Args:
        n_records: Number of reference records (paper uses 200)
        
    Returns:
        List of validated extraction dictionaries
    """
    import random
    
    reference_library = []
    
    for _ in range(n_records):
        record = {
            'age': random.randint(50, 85),
            'sex': random.choice(['male', 'female']),
            'hypertension': random.choice(['yes', 'no']),
            'diabetes_mellitus': random.choice(['yes', 'no', 'no', 'no']),  # Lower prevalence
            'atrial_fibrillation': random.choice(['yes', 'no', 'no']),
            'dyslipidemia': random.choice(['yes', 'no']),
            'prior_stroke': random.choice(['yes', 'no', 'no']),
            'initial_nihss': random.randint(0, 25),
            'aspect_score': random.randint(5, 10),
            'mri_finding': random.choice(['acute_infarction', 'no_lesion']),
            'iv_tpa': random.choice(['yes', 'no', 'no', 'no']),  # Lower rate
            'ia_intervention': random.choice(['yes', 'no', 'no', 'no', 'no'])  # Even lower rate
        }
        reference_library.append(record)
    
    return reference_library


def test_cosine_flagger():
    """Test the cosine similarity flagger."""
    
    config = {
        'threshold_percentile': 5,
        'similarity_threshold': 0.82
    }
    
    flagger = CosineSimilarityFlagger(config)
    
    # Load synthetic reference library
    print("Creating synthetic reference library...")
    reference_library = create_synthetic_reference_library(n_records=200)
    flagger.load_reference_library(reference_library)
    
    # Test case 1: Normal record (should not flag)
    normal_record = {
        'age': 68,
        'sex': 'male',
        'hypertension': 'yes',
        'diabetes_mellitus': 'yes',
        'atrial_fibrillation': 'no',
        'initial_nihss': 12,
        'aspect_score': 8,
        'mri_finding': 'acute_infarction',
        'iv_tpa': 'yes',
        'ia_intervention': 'no'
    }
    
    should_flag, similarity = flagger.check(normal_record)
    print(f"\nTest 1 - Normal record:")
    print(f"  Flagged: {should_flag}")
    print(f"  Max similarity: {similarity:.3f}")
    
    # Test case 2: Outlier record (should flag)
    outlier_record = {
        'age': 25,  # Unusually young for stroke
        'sex': 'male',
        'hypertension': 'no',
        'diabetes_mellitus': 'no',
        'atrial_fibrillation': 'no',
        'initial_nihss': 35,  # Very severe
        'aspect_score': 2,  # Very low
        'mri_finding': 'other',  # Unusual
        'iv_tpa': 'no',
        'ia_intervention': 'yes'
    }
    
    should_flag, similarity = flagger.check(outlier_record)
    print(f"\nTest 2 - Outlier record:")
    print(f"  Flagged: {should_flag}")
    print(f"  Max similarity: {similarity:.3f}")
    
    # Test case 3: Batch processing
    test_records = [normal_record] * 90 + [outlier_record] * 10  # 10% outliers
    flags, scores = flagger.batch_check(test_records)
    
    print(f"\nTest 3 - Batch processing:")
    print(f"  Total records: {len(test_records)}")
    print(f"  Flagged: {sum(flags)} ({100*sum(flags)/len(flags):.1f}%)")
    print(f"  Average similarity: {np.mean(scores):.3f}")


if __name__ == "__main__":
    test_cosine_flagger()
