"""
Clinical Data Extraction Pipeline
Based on Section 2.3: Information Extraction and Validation Pipeline
Implements few-shot prompting with multi-tiered validation
"""

import torch
import json
import yaml
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer
from peft import PeftModel, AutoModelForCausalLM
import re


class ClinicalDataExtractor:
    """
    Main extraction pipeline using fine-tuned sLLM with few-shot prompting.
    
    Paper specifications (Section 2.3.1):
    - 3-shot prompting
    - Temperature: 0.1
    - Max tokens: 512
    - Top-p: 0.95
    - Deterministic sampling (do_sample=False)
    """
    
    def __init__(
        self,
        base_model_name: str = "meta-llama/Meta-Llama-3-8B",
        adapter_path: Optional[str] = None,
        config_path: str = "config/model_config.yaml",
        schema_path: str = "config/extraction_schema.json"
    ):
        """
        Args:
            base_model_name: Base Llama model identifier
            adapter_path: Path to fine-tuned LoRA adapters
            config_path: Path to model configuration
            schema_path: Path to extraction schema
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)['extraction_schema']
        
        self.tokenizer = None
        self.model = None
        self.adapter_path = adapter_path
        self.base_model_name = base_model_name
        
        # Few-shot examples (Section 2.3.1)
        self.few_shot_examples = self._load_few_shot_examples()
        
    def _load_few_shot_examples(self) -> List[Dict[str, str]]:
        """
        Load 3-shot examples for prompting.
        
        In production, these would be carefully curated examples
        from the training set covering diverse documentation styles.
        """
        return [
            {
                'input': """Patient: 68 year old male
PMH: Hypertension, diabetes mellitus
Presentation: Sudden onset right-sided weakness
NIHSS: 8
MRI: Acute left MCA infarction
Treatment: IV t-PA administered""",
                'output': json.dumps({
                    'age': 68,
                    'sex': 'male',
                    'hypertension': 'yes',
                    'diabetes_mellitus': 'yes',
                    'atrial_fibrillation': 'no',
                    'prior_stroke': 'no',
                    'initial_nihss': 8,
                    'mri_finding': 'acute_infarction',
                    'iv_tpa': 'yes',
                    'ia_intervention': 'no'
                }, indent=2)
            },
            {
                'input': """72세 여성 환자
과거력: 고혈압, 심방세동
증상: 갑작스런 왼쪽 마비, 언어장애
NIHSS: 15
뇌 MRI: 우측 중대뇌동맥 영역 급성 경색
치료: t-PA 투여 후 동맥내 혈전제거술 시행""",
                'output': json.dumps({
                    'age': 72,
                    'sex': 'female',
                    'hypertension': 'yes',
                    'atrial_fibrillation': 'yes',
                    'initial_nihss': 15,
                    'mri_finding': 'acute_infarction',
                    'iv_tpa': 'yes',
                    'ia_intervention': 'yes'
                }, indent=2)
            },
            {
                'input': """55 yo M with no significant PMH
Chief complaint: Transient numbness in left arm, resolved
Neuro exam: Normal, NIHSS 0
Brain MRI: No acute lesion
Diagnosis: TIA
Plan: Start antiplatelet, outpatient follow-up""",
                'output': json.dumps({
                    'age': 55,
                    'sex': 'male',
                    'hypertension': 'no',
                    'initial_nihss': 0,
                    'mri_finding': 'no_lesion',
                    'iv_tpa': 'no',
                    'ia_intervention': 'no'
                }, indent=2)
            }
        ]
    
    def load_model(self) -> None:
        """Load base model and LoRA adapters if available."""
        print("Loading tokenizer and model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model with quantization
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load LoRA adapters if provided
        if self.adapter_path:
            print(f"Loading LoRA adapters from {self.adapter_path}...")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path,
                torch_dtype=torch.float16
            )
            print("Fine-tuned adapters loaded")
        else:
            print("Using base model without fine-tuning")
    
    def construct_prompt(self, clinical_note: str) -> str:
        """
        Construct few-shot prompt as described in Section 2.3.1.
        
        Args:
            clinical_note: Preprocessed clinical note
            
        Returns:
            Complete prompt with 3 examples and instruction
        """
        prompt = """You are a meticulous medical data abstractor. Your task is to extract structured clinical information from emergency department notes and output ONLY valid JSON.

CRITICAL RULES:
1. Output MUST be valid JSON only - no explanations, no markdown, no additional text
2. Use exact field names from the schema
3. For binary fields: use "yes", "no", or "unknown" only
4. For missing information: use "unknown" for categorical, null for numeric
5. Extract NIHSS as integer 0-42
6. Be precise with numbers - no approximations

"""
        
        # Add few-shot examples
        for i, example in enumerate(self.few_shot_examples, 1):
            prompt += f"\nExample {i}:\n"
            prompt += f"Clinical Note:\n{example['input']}\n\n"
            prompt += f"Extracted JSON:\n{example['output']}\n"
            prompt += "-" * 50 + "\n"
        
        # Add the actual clinical note to extract
        prompt += f"\nNow extract from this clinical note:\n"
        prompt += f"Clinical Note:\n{clinical_note}\n\n"
        prompt += "Extracted JSON:\n"
        
        return prompt
    
    def extract(self, clinical_note: str) -> Dict[str, any]:
        """
        Extract structured data from clinical note using the fine-tuned model.
        
        Args:
            clinical_note: Preprocessed clinical note text
            
        Returns:
            Dictionary of extracted clinical variables
        """
        if self.model is None:
            self.load_model()
        
        # Construct prompt
        prompt = self.construct_prompt(clinical_note)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        # Generate (Section 2.3.1 parameters)
        generation_config = self.config['prompting']
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=generation_config['max_new_tokens'],
                temperature=generation_config['temperature'],
                top_p=generation_config['top_p'],
                do_sample=generation_config['do_sample'],
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse JSON from output
        extracted_data = self._parse_json_output(generated_text)
        
        return extracted_data
    
    def _parse_json_output(self, generated_text: str) -> Dict[str, any]:
        """
        Parse JSON from model output, handling markdown code blocks.
        
        Args:
            generated_text: Raw model output
            
        Returns:
            Parsed JSON dictionary
        """
        # Remove markdown code blocks if present
        text = generated_text.strip()
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        text = text.strip()
        
        try:
            # Parse JSON
            data = json.loads(text)
            return data
        except json.JSONDecodeError as e:
            # Fallback: try to find JSON object in text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass
            
            # If parsing fails, return error indicator
            print(f"JSON parsing error: {e}")
            print(f"Generated text: {text[:500]}...")
            return {'_parsing_error': True, '_raw_output': text}
    
    def batch_extract(
        self, 
        clinical_notes: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, any]]:
        """
        Extract from multiple clinical notes.
        
        Args:
            clinical_notes: List of preprocessed clinical notes
            show_progress: Whether to show progress bar
            
        Returns:
            List of extracted data dictionaries
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(clinical_notes) if show_progress else clinical_notes
        
        for note in iterator:
            extracted = self.extract(note)
            results.append(extracted)
        
        return results


class ValidationPipeline:
    """
    Multi-tiered validation framework from Section 2.3.2.
    
    Validation layers:
    1. Rule-based verification
    2. RAG verification
    3. Cosine similarity flagging
    4. Human-in-the-loop review
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['validation']
        
        # Import validation modules
        from validation.rule_based import RuleBasedValidator
        from validation.rag_verification import RAGVerifier
        from validation.cosine_similarity import CosineSimilarityFlagger
        
        self.rule_validator = RuleBasedValidator()
        self.rag_verifier = RAGVerifier(self.config['rag'])
        self.cosine_flagger = CosineSimilarityFlagger(self.config['cosine_similarity'])
        
        self.validation_results = {
            'baseline': [],
            'rule_based': [],
            'rag_verified': [],
            'flagged_for_hitl': [],
            'final': []
        }
    
    def validate(
        self,
        extracted_data: Dict[str, any],
        original_note: str,
        note_id: str
    ) -> Tuple[Dict[str, any], Dict[str, any]]:
        """
        Run complete validation pipeline.
        
        Args:
            extracted_data: Raw extraction from LLM
            original_note: Original clinical note text
            note_id: Unique identifier for this note
            
        Returns:
            Tuple of (validated_data, validation_metadata)
        """
        # Stage 1: Baseline (no validation)
        self.validation_results['baseline'].append({
            'note_id': note_id,
            'data': extracted_data.copy()
        })
        
        # Stage 2: Rule-based verification
        rule_validated, rule_errors = self.rule_validator.validate(extracted_data)
        self.validation_results['rule_based'].append({
            'note_id': note_id,
            'data': rule_validated,
            'errors': rule_errors
        })
        
        # Stage 3: RAG verification
        rag_validated, grounding_score = self.rag_verifier.verify(
            rule_validated,
            original_note
        )
        self.validation_results['rag_verified'].append({
            'note_id': note_id,
            'data': rag_validated,
            'grounding_score': grounding_score
        })
        
        # Stage 4: Cosine similarity flagging
        should_flag, similarity_score = self.cosine_flagger.check(rag_validated)
        
        if should_flag or grounding_score < self.config['rag']['similarity_threshold']:
            self.validation_results['flagged_for_hitl'].append({
                'note_id': note_id,
                'data': rag_validated,
                'reason': 'low_similarity' if should_flag else 'low_grounding',
                'similarity_score': similarity_score,
                'grounding_score': grounding_score
            })
        
        # Validation metadata
        metadata = {
            'rule_errors': len(rule_errors),
            'grounding_score': grounding_score,
            'similarity_score': similarity_score,
            'flagged_for_hitl': should_flag or grounding_score < self.config['rag']['similarity_threshold'],
            'validation_stages_passed': 3 if not should_flag else 2
        }
        
        return rag_validated, metadata


if __name__ == "__main__":
    # Example usage
    print("Clinical Data Extraction Pipeline")
    print("=" * 50)
    
    # Initialize extractor
    extractor = ClinicalDataExtractor(
        base_model_name="meta-llama/Meta-Llama-3-8B",
        adapter_path=None,  # Set to path of fine-tuned adapters
        config_path="../config/model_config.yaml",
        schema_path="../config/extraction_schema.json"
    )
    
    # Sample clinical note
    sample_note = """
    Patient: 65-year-old male
    Chief complaint: Sudden onset left-sided weakness
    Past medical history: Hypertension, diabetes mellitus, atrial fibrillation
    Medications: Metformin, lisinopril, warfarin
    
    Physical examination:
    - Blood pressure: 165/95 mmHg
    - Heart rate: 88 bpm, irregular
    - Neurological exam: Right facial droop, left hemiparesis
    - NIHSS score: 12
    
    Imaging:
    - Brain MRI: Acute infarction in right MCA territory
    - ASPECT score: 8
    
    Treatment:
    - IV t-PA administered at 2 hours 45 minutes from symptom onset
    - Patient transferred to neuro ICU
    """
    
    # Extract data
    print("\nExtracting structured data...")
    # extracted_data = extractor.extract(sample_note)
    # print(json.dumps(extracted_data, indent=2))
    
    print("\nNote: Actual extraction requires loaded model (~6GB memory)")
