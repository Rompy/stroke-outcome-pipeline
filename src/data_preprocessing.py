"""
Data Preprocessing Module
Based on Section 2.2.2: Text Pre-processing and Tokenization
"""

import re
import json
from typing import Dict, List, Tuple
from transformers import AutoTokenizer


class ClinicalTextPreprocessor:
    """
    Preprocesses Korean clinical notes for LLM extraction.
    Handles abbreviation expansion and text normalization.
    """
    
    def __init__(self, schema_path: str, tokenizer_name: str = "meta-llama/Meta-Llama-3-8B"):
        """
        Args:
            schema_path: Path to extraction_schema.json
            tokenizer_name: HuggingFace model name for tokenizer
        """
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        self.abbreviations = schema['abbreviation_expansion']
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand medical abbreviations based on institutional dictionary.
        
        Example:
            "Pt has HTN, DM" -> "Patient has hypertension, diabetes mellitus"
        """
        expanded_text = text
        
        # Sort by length (descending) to handle multi-word abbreviations first
        sorted_abbrevs = sorted(self.abbreviations.items(), 
                               key=lambda x: len(x[0]), 
                               reverse=True)
        
        for abbrev, full_form in sorted_abbrevs:
            # Word boundary matching to avoid partial replacements
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            expanded_text = re.sub(pattern, full_form, expanded_text, flags=re.IGNORECASE)
        
        return expanded_text
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize clinical text while preserving medical terminology.
        
        Steps:
        1. Remove excessive whitespace
        2. Standardize line breaks
        3. Preserve Korean-English mixed content
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def preprocess(self, clinical_note: str) -> Dict[str, any]:
        """
        Complete preprocessing pipeline.
        
        Args:
            clinical_note: Raw clinical note text
            
        Returns:
            Dictionary containing:
                - original_text: Raw input
                - normalized_text: After normalization
                - expanded_text: After abbreviation expansion
                - token_count: Number of tokens
                - char_count: Number of characters
                - token_char_ratio: Token/character ratio
        """
        # Step 1: Normalize
        normalized = self.normalize_text(clinical_note)
        
        # Step 2: Expand abbreviations
        expanded = self.expand_abbreviations(normalized)
        
        # Step 3: Tokenize and count
        tokens = self.tokenizer.encode(expanded, add_special_tokens=True)
        
        return {
            'original_text': clinical_note,
            'normalized_text': normalized,
            'expanded_text': expanded,
            'token_count': len(tokens),
            'char_count': len(expanded),
            'token_char_ratio': len(tokens) / len(expanded) if len(expanded) > 0 else 0,
            'tokens': tokens
        }
    
    def check_context_window(self, text: str, max_length: int = 4096) -> Tuple[bool, int]:
        """
        Check if text fits within model's context window.
        
        Args:
            text: Preprocessed text
            max_length: Maximum sequence length (default: 4096 for Llama 3)
            
        Returns:
            Tuple of (fits_in_window, token_count)
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        token_count = len(tokens)
        
        return token_count <= max_length, token_count
    
    def batch_preprocess(self, clinical_notes: List[str]) -> List[Dict[str, any]]:
        """
        Preprocess multiple clinical notes.
        
        Args:
            clinical_notes: List of raw clinical notes
            
        Returns:
            List of preprocessed dictionaries
        """
        return [self.preprocess(note) for note in clinical_notes]


def calculate_statistics(preprocessed_notes: List[Dict[str, any]]) -> Dict[str, any]:
    """
    Calculate preprocessing statistics matching paper's Section 2.2.2.
    
    Paper reports:
    - Average: 1,247 characters (SD: 342)
    - Average: 1,421 tokens (SD: 389)
    - Token-to-character ratio: 1.14
    - Maximum utilization: 58.5% of 4096 tokens
    """
    char_counts = [note['char_count'] for note in preprocessed_notes]
    token_counts = [note['token_count'] for note in preprocessed_notes]
    ratios = [note['token_char_ratio'] for note in preprocessed_notes]
    
    import numpy as np
    
    stats = {
        'total_notes': len(preprocessed_notes),
        'characters': {
            'mean': np.mean(char_counts),
            'std': np.std(char_counts),
            'min': np.min(char_counts),
            'max': np.max(char_counts)
        },
        'tokens': {
            'mean': np.mean(token_counts),
            'std': np.std(token_counts),
            'min': np.min(token_counts),
            'max': np.max(token_counts)
        },
        'token_char_ratio': {
            'mean': np.mean(ratios),
            'std': np.std(ratios)
        },
        'max_context_utilization': np.max(token_counts) / 4096 * 100
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    preprocessor = ClinicalTextPreprocessor(
        schema_path="../config/extraction_schema.json"
    )
    
    sample_note = """
    Pt is 65 yo male with HTN, DM, AF
    Presented with sudden onset Lt weakness
    NIHSS 12 on arrival
    Brain MRI: acute infarction in Rt MCA territory
    t-PA administered within 3 hours
    """
    
    result = preprocessor.preprocess(sample_note)
    print(f"Original length: {result['char_count']} characters")
    print(f"Token count: {result['token_count']} tokens")
    print(f"Token/char ratio: {result['token_char_ratio']:.2f}")
    print(f"\nExpanded text:\n{result['expanded_text']}")
