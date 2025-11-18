"""
Model Fine-tuning Module
Based on Section 2.2.3: Parameter-Efficient Fine-Tuning (PEFT)
Implements LoRA with 4-bit quantization
"""

import torch
import yaml
from typing import Dict, List, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import json


class StrokeLLMFineTuner:
    """
    Fine-tune Llama 3 8B for clinical data extraction using LoRA.
    
    Paper specifications (Section 2.2.3):
    - Base model: Llama 3 8B
    - LoRA: r=16, alpha=32
    - 4-bit quantization (NF4)
    - Training: 450 annotated records, 3 epochs
    - Optimizer: AdamW with cosine scheduler
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Args:
            config_path: Path to model configuration YAML
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def load_base_model(self) -> None:
        """
        Load base Llama 3 8B model with 4-bit quantization.
        
        Quantization config from paper:
        - 4-bit NF4 quantization
        - Double quantization enabled
        - Compute dtype: float16
        """
        model_config = self.config['model']
        quant_config = self.config['quantization']
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config['load_in_4bit'],
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant']
        )
        
        # Load model (approximately 16GB -> 4GB after quantization)
        print("Loading base model with 4-bit quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['name'],
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        print(f"Model loaded. Memory footprint: ~4GB")
        
    def configure_lora(self) -> None:
        """
        Configure LoRA adapters as specified in paper.
        
        LoRA configuration (Section 2.2.3):
        - Rank r = 16
        - Alpha = 32
        - Target modules: attention layers (q_proj, v_proj, k_proj, o_proj)
        - Dropout = 0.05
        """
        lora_config_dict = self.config['lora']
        
        lora_config = LoraConfig(
            r=lora_config_dict['r'],
            lora_alpha=lora_config_dict['lora_alpha'],
            target_modules=lora_config_dict['target_modules'],
            lora_dropout=lora_config_dict['lora_dropout'],
            bias=lora_config_dict['bias'],
            task_type=TaskType.CAUSAL_LM
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Add LoRA adapters
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        
        print(f"\nLoRA Configuration:")
        print(f"  Rank (r): {lora_config_dict['r']}")
        print(f"  Alpha: {lora_config_dict['lora_alpha']}")
        print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"  Total parameters: {total_params:,}")
        
    def prepare_training_data(
        self, 
        annotated_records: List[Dict[str, str]]
    ) -> Dataset:
        """
        Prepare training dataset from annotated clinical records.
        
        Args:
            annotated_records: List of dicts with 'clinical_note' and 'ground_truth' keys
            
        Returns:
            HuggingFace Dataset ready for training
            
        Expected format:
        [
            {
                'clinical_note': "Patient is 65 yo male with HTN...",
                'ground_truth': '{"age": 65, "sex": "male", "hypertension": "yes", ...}'
            },
            ...
        ]
        """
        
        def format_instruction(record: Dict[str, str]) -> str:
            """Format as instruction-following prompt."""
            prompt = f"""You are a meticulous medical data abstractor. Extract structured information from the following clinical note.

Clinical Note:
{record['clinical_note']}

Extract the following information in JSON format:
{record['ground_truth']}"""
            return prompt
        
        # Format all records
        formatted_data = []
        for record in annotated_records:
            text = format_instruction(record)
            formatted_data.append({'text': text})
        
        # Create HuggingFace dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding="max_length",
                truncation=True,
                max_length=self.config['training']['max_seq_length']
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def train(
        self, 
        train_dataset: Dataset,
        output_dir: str = "./lora_adapters",
        val_dataset: Optional[Dataset] = None
    ) -> None:
        """
        Fine-tune model using LoRA.
        
        Training configuration from paper (Section 2.2.3):
        - 3 epochs
        - AdamW optimizer (8-bit)
        - Cosine learning rate scheduler
        - Learning rate: 2e-4
        - Max gradient norm: 0.3
        - Training time: ~6 hours on M2 Max
        
        Args:
            train_dataset: Tokenized training dataset
            output_dir: Directory to save LoRA adapters
            val_dataset: Optional validation dataset
        """
        training_config = self.config['training']
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config['num_epochs'],
            per_device_train_batch_size=training_config['batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            warmup_steps=training_config['warmup_steps'],
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            max_grad_norm=training_config['max_grad_norm'],
            optim=training_config['optimizer'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            fp16=True,
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=training_config['save_steps'] if val_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if val_dataset else False,
            report_to="none"  # Disable wandb/tensorboard
        )
        
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        print("\nStarting fine-tuning...")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Expected time: ~6 hours (M2 Max) or ~3 hours (A100)")
        
        trainer.train()
        
        # Save LoRA adapters only (not full model)
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"\nLoRA adapters saved to {output_dir}")
        print("Adapter size: ~50MB (vs. 16GB full model)")
    
    def load_trained_adapters(self, adapter_path: str) -> None:
        """
        Load previously trained LoRA adapters.
        
        Args:
            adapter_path: Path to saved adapters
        """
        from peft import PeftModel
        
        if self.model is None:
            self.load_base_model()
        
        print(f"Loading LoRA adapters from {adapter_path}...")
        self.peft_model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            torch_dtype=torch.float16
        )
        
        print("Adapters loaded successfully")


def create_synthetic_training_data(n_samples: int = 450) -> List[Dict[str, str]]:
    """
    Create synthetic training data for demonstration purposes.
    
    In production, this would be replaced with the 450 manually annotated
    clinical records mentioned in Section 2.2.3.
    
    Args:
        n_samples: Number of samples (paper uses 450)
        
    Returns:
        List of training examples
    """
    from faker import Faker
    import random
    
    fake = Faker()
    training_data = []
    
    for _ in range(n_samples):
        age = random.randint(45, 85)
        sex = random.choice(['male', 'female'])
        nihss = random.randint(0, 25)
        
        # Generate synthetic clinical note
        note = f"""Patient is {age} year old {sex} presenting with sudden onset neurological symptoms.
Past medical history: {'hypertension, ' if random.random() > 0.5 else ''}{'diabetes mellitus, ' if random.random() > 0.7 else ''}{'atrial fibrillation' if random.random() > 0.8 else ''}.
Initial NIHSS score: {nihss}.
Brain MRI shows {'acute infarction' if random.random() > 0.4 else 'no acute lesion'}.
{'Received IV t-PA within therapeutic window.' if random.random() > 0.9 else ''}
"""
        
        # Ground truth JSON
        ground_truth = json.dumps({
            'age': age,
            'sex': sex,
            'initial_nihss': nihss,
            'hypertension': 'yes' if 'hypertension' in note else 'no',
            'diabetes_mellitus': 'yes' if 'diabetes' in note else 'no',
            'atrial_fibrillation': 'yes' if 'atrial fibrillation' in note else 'no',
            'mri_finding': 'acute_infarction' if 'acute infarction' in note else 'no_lesion',
            'iv_tpa': 'yes' if 't-PA' in note else 'no'
        }, indent=2)
        
        training_data.append({
            'clinical_note': note,
            'ground_truth': ground_truth
        })
    
    return training_data


if __name__ == "__main__":
    # Example usage
    print("Stroke LLM Fine-tuner Example")
    print("=" * 50)
    
    # Initialize fine-tuner
    finetuner = StrokeLLMFineTuner(config_path="../config/model_config.yaml")
    
    # Load base model with 4-bit quantization
    finetuner.load_base_model()
    
    # Configure LoRA
    finetuner.configure_lora()
    
    # Prepare training data (450 samples as per paper)
    print("\nPreparing training data...")
    training_records = create_synthetic_training_data(n_samples=450)
    train_dataset = finetuner.prepare_training_data(training_records)
    
    # Train (this would take ~6 hours on M2 Max)
    # finetuner.train(train_dataset, output_dir="./stroke_lora_adapters")
    
    print("\nSetup complete. Ready for training.")
    print("Note: Actual training takes ~6 hours on Apple M2 Max")
