"""
RAG (Retrieval-Augmented Generation) Verification Module
Based on Section 2.3.2: Multi-tiered Validation Framework - Layer 2

Implements semantic verification using FAISS vector database.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Any
import torch
from sentence_transformers import SentenceTransformer
import faiss


class RAGVerifier:
    """
    Second validation layer: semantic verification using RAG.
    
    Paper description (Section 2.3.2):
    "An internal knowledge base was constructed by generating 1024-dimensional 
    embeddings of clinical notes and sentences using the multilingual-e5-large 
    model and indexing them in a FAISS database configured with IndexFlatIP 
    for exact inner-product similarity search."
    
    "For each extracted entity, a query embedding was used to retrieve the 
    top three semantically similar text segments (k=3) from the original note."
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: RAG configuration from model_config.yaml
        """
        self.config = config
        self.embedding_model_name = config['embedding_model']
        self.embedding_dim = config['embedding_dim']
        self.top_k = config['top_k']
        self.similarity_threshold = config['similarity_threshold']
        
        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # FAISS index (will be built per note)
        self.faiss_index = None
        self.sentence_database = []
    
    def build_knowledge_base(self, clinical_note: str) -> None:
        """
        Build FAISS index for a clinical note.
        
        Paper: "containing 1,166 document-level and approximately 
        45,000 sentence-level vectors"
        
        For per-note verification, we index all sentences in the note.
        
        Args:
            clinical_note: The original clinical note text
        """
        # Split note into sentences
        import re
        sentences = re.split(r'[.!?]\s+', clinical_note)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) == 0:
            # Fallback: use the entire note
            sentences = [clinical_note]
        
        self.sentence_database = sentences
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            sentences,
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=True  # For cosine similarity via inner product
        )
        
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index (IndexFlatIP for inner product)
        # Paper: "configured with IndexFlatIP for exact inner-product similarity search"
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(embeddings)
    
    def retrieve_supporting_context(self, query: str) -> List[Tuple[str, float]]:
        """
        Retrieve top-k similar sentences from knowledge base.
        
        Args:
            query: Query text (e.g., "hypertension: yes")
            
        Returns:
            List of (sentence, similarity_score) tuples
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_tensor=False,
            normalize_embeddings=True
        )
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        k = min(self.top_k, self.faiss_index.ntotal)
        similarities, indices = self.faiss_index.search(query_embedding, k)
        
        # Retrieve sentences
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            if idx < len(self.sentence_database):
                results.append((self.sentence_database[idx], float(sim)))
        
        return results
    
    def verify_field(
        self, 
        field_name: str, 
        extracted_value: Any,
        original_note: str
    ) -> Tuple[Any, float, List[str]]:
        """
        Verify a single extracted field using RAG.
        
        Paper: "For each extracted entity, a query embedding was used to 
        retrieve the top three semantically similar text segments (k=3) 
        from the original note, which were concatenated and provided to 
        the sLLM in a follow-up prompt to verify or correct the extracted value"
        
        Args:
            field_name: Name of the field (e.g., 'hypertension')
            extracted_value: Value extracted by LLM
            original_note: Original clinical note
            
        Returns:
            Tuple of (verified_value, confidence_score, supporting_sentences)
        """
        # Build knowledge base if not already done
        if self.faiss_index is None:
            self.build_knowledge_base(original_note)
        
        # Construct query
        query = f"{field_name}: {extracted_value}"
        
        # Retrieve supporting context
        supporting_context = self.retrieve_supporting_context(query)
        
        if not supporting_context:
            return extracted_value, 0.0, []
        
        # Check if any retrieved segment exceeds similarity threshold
        max_similarity = max(score for _, score in supporting_context)
        
        if max_similarity < self.similarity_threshold:
            # Paper: "If no retrieved segment exceeded a cosine similarity 
            # threshold of 0.7, the case was automatically flagged for 
            # human review"
            return extracted_value, max_similarity, [sent for sent, _ in supporting_context]
        
        # Extract supporting sentences
        supporting_sentences = [sent for sent, _ in supporting_context]
        
        # In full implementation, would use LLM to verify/correct
        # For now, return original value with confidence
        return extracted_value, max_similarity, supporting_sentences
    
    def verify(
        self,
        extracted_data: Dict[str, Any],
        original_note: str
    ) -> Tuple[Dict[str, Any], float]:
        """
        Verify all fields in extracted data.
        
        Args:
            extracted_data: Data after rule-based validation
            original_note: Original clinical note
            
        Returns:
            Tuple of (verified_data, average_grounding_score)
        """
        # Build knowledge base for this note
        self.build_knowledge_base(original_note)
        
        verified_data = extracted_data.copy()
        field_scores = []
        
        # Verify each field
        critical_fields = [
            'age', 'sex', 'hypertension', 'diabetes_mellitus',
            'atrial_fibrillation', 'initial_nihss', 'mri_finding',
            'iv_tpa', 'ia_intervention'
        ]
        
        for field in critical_fields:
            if field in extracted_data and extracted_data[field] not in [None, 'unknown']:
                verified_value, score, supporting = self.verify_field(
                    field,
                    extracted_data[field],
                    original_note
                )
                
                verified_data[field] = verified_value
                verified_data[f'{field}_grounding_score'] = score
                verified_data[f'{field}_supporting_context'] = supporting
                
                field_scores.append(score)
        
        # Calculate average grounding accuracy
        # Paper: "Grounding accuracy... was 91.5% (95% CI, 89.9–93.1%)"
        avg_grounding_score = np.mean(field_scores) if field_scores else 0.0
        
        return verified_data, avg_grounding_score
    
    def batch_verify(
        self,
        extracted_data_list: List[Dict[str, Any]],
        original_notes: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Verify multiple records.
        
        Args:
            extracted_data_list: List of extracted data dictionaries
            original_notes: List of original clinical notes
            
        Returns:
            Tuple of (verified_data_list, grounding_scores)
        """
        verified_list = []
        grounding_scores = []
        
        from tqdm import tqdm
        
        for extracted, note in tqdm(zip(extracted_data_list, original_notes), 
                                     total=len(extracted_data_list),
                                     desc="RAG Verification"):
            verified, score = self.verify(extracted, note)
            verified_list.append(verified)
            grounding_scores.append(score)
        
        return verified_list, grounding_scores


def calculate_grounding_accuracy(
    grounding_scores: List[float],
    threshold: float = 0.7
) -> Dict[str, float]:
    """
    Calculate grounding accuracy as reported in paper.
    
    Paper reports (Table 2):
    - RAG stage: 91.5% (95% CI, 89.9–93.1%)
    - HITL stage: 93.2% (95% CI, 91.7–94.7%)
    
    Args:
        grounding_scores: List of per-field similarity scores
        threshold: Minimum score to consider "grounded" (default: 0.7)
        
    Returns:
        Dictionary with accuracy statistics
    """
    scores_array = np.array(grounding_scores)
    
    grounded_count = np.sum(scores_array >= threshold)
    total_count = len(scores_array)
    
    accuracy = grounded_count / total_count if total_count > 0 else 0.0
    
    # Calculate 95% CI using Wilson score interval
    from scipy import stats
    if total_count > 0:
        ci_low, ci_high = stats.binom.interval(
            0.95, 
            total_count, 
            accuracy
        )
        ci_low = ci_low / total_count
        ci_high = ci_high / total_count
    else:
        ci_low = ci_high = 0.0
    
    return {
        'grounding_accuracy': accuracy,
        'ci_lower': ci_low,
        'ci_upper': ci_high,
        'mean_score': np.mean(scores_array) if len(scores_array) > 0 else 0.0,
        'median_score': np.median(scores_array) if len(scores_array) > 0 else 0.0,
        'std_score': np.std(scores_array) if len(scores_array) > 0 else 0.0
    }


def test_rag_verifier():
    """Test the RAG verifier with sample data."""
    
    config = {
        'embedding_model': 'intfloat/multilingual-e5-large',
        'embedding_dim': 1024,
        'top_k': 3,
        'similarity_threshold': 0.7
    }
    
    verifier = RAGVerifier(config)
    
    # Sample clinical note
    clinical_note = """
    Patient is a 68-year-old male with past medical history of hypertension 
    and diabetes mellitus. He presents with sudden onset right-sided weakness 
    that started 2 hours ago. On examination, blood pressure is 165/95 mmHg.
    Neurological examination reveals right facial droop and left hemiparesis.
    NIHSS score is calculated as 12 points. Brain MRI demonstrates acute 
    infarction in the right middle cerebral artery territory. IV tissue 
    plasminogen activator was administered within the therapeutic window.
    """
    
    # Sample extracted data
    extracted_data = {
        'age': 68,
        'sex': 'male',
        'hypertension': 'yes',
        'diabetes_mellitus': 'yes',
        'initial_nihss': 12,
        'mri_finding': 'acute_infarction',
        'iv_tpa': 'yes'
    }
    
    # Verify
    print("Testing RAG Verifier...")
    print("-" * 50)
    
    verified_data, grounding_score = verifier.verify(extracted_data, clinical_note)
    
    print(f"\nAverage grounding score: {grounding_score:.3f}")
    print("\nVerified fields:")
    for key, value in verified_data.items():
        if '_grounding_score' in key:
            field = key.replace('_grounding_score', '')
            print(f"  {field}: {verified_data[field]} (score: {value:.3f})")


if __name__ == "__main__":
    test_rag_verifier()
