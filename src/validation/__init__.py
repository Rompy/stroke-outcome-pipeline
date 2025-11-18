"""
Multi-tiered Validation Framework
==================================

Implements defense-in-depth validation:
1. Rule-based verification (syntactic/range checks)
2. RAG verification (FAISS semantic grounding)
3. Cosine similarity flagging (outlier detection)
4. Human-in-the-loop review

Modules:
    rule_based: Rule-based validator
    rag_verification: RAG-based semantic verification
    cosine_similarity: Cosine similarity flagger
    human_review: HITL review interface

Example:
    >>> from validation.rule_based import RuleBasedValidator
    >>> from validation.rag_verification import RAGVerifier
    >>> 
    >>> validator = RuleBasedValidator()
    >>> rag = RAGVerifier(config)
    >>> 
    >>> validated, errors = validator.validate(extracted_data)
    >>> verified, score = rag.verify(validated, original_note)
"""

from .rule_based import RuleBasedValidator
from .rag_verification import RAGVerifier
from .cosine_similarity import CosineSimilarityFlagger

__all__ = [
    'RuleBasedValidator',
    'RAGVerifier',
    'CosineSimilarityFlagger'
]
