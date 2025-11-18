"""
Synthetic Clinical Data Generator
Creates realistic synthetic clinical notes and outcomes for testing the pipeline.

Note: This generates FAKE data for demonstration only.
Real data cannot be shared due to privacy regulations (IRB 4-2025-0125).
"""

import json
import random
import numpy as np
from faker import Faker
from typing import List, Dict, Any, Tuple
import pandas as pd


class SyntheticStrokeDataGenerator:
    """
    Generate synthetic stroke patient data matching the study population.
    
    Paper statistics (Table 1):
    - Mean age: 65.68 ± 15.90
    - Male: 56.2%
    - Hypertension: 57.4%
    - Diabetes: 24.4%
    - Atrial fibrillation: 14.9%
    - Median NIHSS: 3 (IQR 1-7)
    - MRI infarction: 59.4%
    - IV t-PA: 9%
    - IA intervention: 7.5%
    - Poor outcome (mRS 3-6): 28.4%
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        self.fake = Faker()
        Faker.seed(seed)
    
    def generate_demographics(self) -> Dict[str, Any]:
        """Generate patient demographics matching paper statistics."""
        # Age: normal distribution, mean=65.68, sd=15.90
        age = int(np.random.normal(65.68, 15.90))
        age = np.clip(age, 18, 100)
        
        # Sex: 56.2% male
        sex = 'male' if random.random() < 0.562 else 'female'
        
        return {'age': age, 'sex': sex}
    
    def generate_comorbidities(self) -> Dict[str, str]:
        """Generate comorbidities with prevalence matching paper."""
        return {
            'hypertension': 'yes' if random.random() < 0.574 else 'no',
            'diabetes_mellitus': 'yes' if random.random() < 0.244 else 'no',
            'cardiovascular_disease': 'yes' if random.random() < 0.200 else 'no',
            'atrial_fibrillation': 'yes' if random.random() < 0.149 else 'no',
            'dyslipidemia': 'yes' if random.random() < 0.197 else 'no',
            'prior_stroke': 'yes' if random.random() < 0.217 else 'no',
            'malignancy': 'yes' if random.random() < 0.115 else 'no',
            'esrd': 'yes' if random.random() < 0.024 else 'no'
        }
    
    def generate_clinical_scores(self, age: int, comorbidities: Dict) -> Dict[str, int]:
        """Generate clinical scores with realistic correlations."""
        # NIHSS: median 3, IQR 1-7
        # Higher NIHSS associated with worse outcomes
        nihss_values = [0] * 254 + list(range(1, 5)) * 118 + list(range(5, 16)) * 77 + list(range(16, 21)) * 17 + list(range(21, 43)) * 7
        nihss = random.choice(nihss_values)
        
        # ASPECT: higher is better, 9 (IQR 8-10)
        aspect = random.choices([7, 8, 9, 10], weights=[10, 30, 40, 20])[0]
        
        # Lower ASPECT if higher NIHSS
        if nihss > 15:
            aspect = random.choices([4, 5, 6, 7, 8], weights=[10, 20, 30, 25, 15])[0]
        
        return {
            'initial_nihss': nihss,
            'aspect_score': aspect
        }
    
    def generate_imaging(self, nihss: int) -> str:
        """Generate MRI findings correlated with NIHSS."""
        # 59.4% have acute infarction
        # Higher probability if higher NIHSS
        if nihss == 0:
            prob_infarct = 0.2
        elif nihss <= 4:
            prob_infarct = 0.5
        elif nihss <= 15:
            prob_infarct = 0.7
        else:
            prob_infarct = 0.9
        
        if random.random() < prob_infarct:
            return 'acute_infarction'
        else:
            return 'no_lesion'
    
    def generate_interventions(self, nihss: int, mri_finding: str, age: int) -> Dict[str, str]:
        """Generate interventions based on clinical indicators."""
        # IV t-PA: 9% overall, higher if NIHSS 4-25 and acute infarction
        if mri_finding == 'acute_infarction' and 4 <= nihss <= 25 and age < 80:
            prob_tpa = 0.25
        else:
            prob_tpa = 0.02
        
        iv_tpa = 'yes' if random.random() < prob_tpa else 'no'
        
        # IA intervention: 7.5% overall, if large vessel occlusion
        if iv_tpa == 'yes' and nihss >= 6:
            prob_ia = 0.40
        else:
            prob_ia = 0.02
        
        ia_intervention = 'yes' if random.random() < prob_ia else 'no'
        
        return {
            'iv_tpa': iv_tpa,
            'ia_intervention': ia_intervention
        }
    
    def generate_outcome(
        self,
        age: int,
        nihss: int,
        aspect: int,
        comorbidities: Dict,
        mri_finding: str,
        interventions: Dict
    ) -> int:
        """
        Generate 3-month mRS outcome with realistic probabilities.
        
        Poor outcome (mRS 3-6): 28.4% in study population
        
        Risk factors for poor outcome:
        - Higher NIHSS
        - Older age
        - Lower ASPECT
        - Atrial fibrillation
        - No reperfusion therapy
        """
        # Base probability
        prob_poor = 0.284
        
        # NIHSS effect (strongest predictor)
        if nihss == 0:
            prob_poor = 0.05
        elif nihss <= 4:
            prob_poor = 0.10
        elif nihss <= 10:
            prob_poor = 0.30
        elif nihss <= 20:
            prob_poor = 0.60
        else:
            prob_poor = 0.85
        
        # Age effect
        if age > 75:
            prob_poor *= 1.3
        
        # ASPECT effect
        if aspect <= 5:
            prob_poor *= 1.4
        
        # Comorbidity effects
        if comorbidities['atrial_fibrillation'] == 'yes':
            prob_poor *= 1.2
        if comorbidities['prior_stroke'] == 'yes':
            prob_poor *= 1.3
        
        # Treatment effects (protective)
        if interventions['iv_tpa'] == 'yes':
            prob_poor *= 0.7
        if interventions['ia_intervention'] == 'yes':
            prob_poor *= 0.6
        
        # MRI finding
        if mri_finding != 'acute_infarction':
            prob_poor *= 0.3
        
        # Cap probability
        prob_poor = np.clip(prob_poor, 0.0, 0.95)
        
        # Generate outcome (0 = good mRS 0-2, 1 = poor mRS 3-6)
        return 1 if random.random() < prob_poor else 0
    
    def generate_clinical_note(self, patient_data: Dict[str, Any]) -> str:
        """
        Generate realistic clinical note in mixed Korean-English style.
        
        Paper: Korean clinical notes with English medical abbreviations
        Average length: 1,247 characters (SD: 342)
        """
        age = patient_data['age']
        sex = patient_data['sex']
        nihss = patient_data['initial_nihss']
        
        # Template selection based on NIHSS severity
        if nihss == 0:
            symptoms = random.choice([
                "transient numbness in left arm, fully resolved",
                "brief speech difficulty, now normal",
                "mild dizziness, resolved spontaneously"
            ])
        elif nihss <= 4:
            symptoms = random.choice([
                "mild left facial droop and hand weakness",
                "dysarthria and mild right arm weakness",
                "left facial numbness and word-finding difficulty"
            ])
        elif nihss <= 15:
            symptoms = random.choice([
                "sudden onset left hemiparesis and dysarthria",
                "right-sided weakness with aphasia",
                "left hemiplegia and facial droop"
            ])
        else:
            symptoms = random.choice([
                "severe right hemiplegia, global aphasia, decreased consciousness",
                "left hemiplegia, neglect, gaze deviation",
                "quadriplegia, dysarthria, dysphagia"
            ])
        
        # Comorbidities
        pmh_items = []
        if patient_data['hypertension'] == 'yes':
            pmh_items.append('HTN')
        if patient_data['diabetes_mellitus'] == 'yes':
            pmh_items.append('DM')
        if patient_data['atrial_fibrillation'] == 'yes':
            pmh_items.append('AF')
        if patient_data['dyslipidemia'] == 'yes':
            pmh_items.append('dyslipidemia')
        
        pmh = ', '.join(pmh_items) if pmh_items else 'none'
        
        # MRI finding
        if patient_data['mri_finding'] == 'acute_infarction':
            territories = ['MCA', 'ACA', 'PCA', 'basilar', 'PICA']
            sides = ['right', 'left']
            mri_desc = f"acute infarction in {random.choice(sides)} {random.choice(territories)} territory"
        else:
            mri_desc = "no acute lesion"
        
        # Treatment
        treatment_lines = []
        if patient_data['iv_tpa'] == 'yes':
            time = random.randint(90, 270)
            treatment_lines.append(f"IV t-PA administered at {time} minutes from symptom onset")
        if patient_data['ia_intervention'] == 'yes':
            treatment_lines.append("Mechanical thrombectomy performed")
        
        treatment = '\n'.join(treatment_lines) if treatment_lines else "Conservative management"
        
        # Construct note
        note = f"""
Patient: {age}-year-old {sex}
Chief complaint: {symptoms}
Onset time: {random.randint(1, 12)} hours ago

Past medical history: {pmh}
Medications: {random.choice(['Aspirin', 'Warfarin', 'Atenolol', 'Metformin', 'None'])}

Vital signs:
- BP: {random.randint(130, 180)}/{random.randint(75, 100)} mmHg
- HR: {random.randint(60, 110)} bpm
- RR: {random.randint(16, 22)} /min
- Body temperature: {random.uniform(36.3, 37.5):.1f} °C

Neurological examination:
- Consciousness: {random.choice(['alert', 'drowsy', 'stuporous'])}
- Motor: {random.choice(['intact', 'left hemiparesis', 'right hemiparesis', 'quadriparesis'])}
- Sensation: {random.choice(['intact', 'decreased on left', 'decreased on right'])}
- Language: {random.choice(['fluent', 'dysarthric', 'aphasic'])}

NIHSS score: {nihss}
ASPECT score: {patient_data['aspect_score']}

Imaging:
- Brain CT: no hemorrhage
- Brain MRI: {mri_desc}

Treatment:
{treatment}

Plan: {random.choice(['Admit to stroke unit', 'Neurology ICU', 'Discharge with outpatient follow-up'])}
""".strip()
        
        # Add some variation in length (target: ~1247 chars)
        if len(note) < 800:
            note += f"\n\nAdditional notes: {self.fake.sentence(nb_words=20)}"
        
        return note
    
    def generate_patient(self) -> Tuple[Dict[str, Any], str, int]:
        """
        Generate one complete synthetic patient record.
        
        Returns:
            Tuple of (structured_data, clinical_note, outcome)
        """
        # Generate components
        demographics = self.generate_demographics()
        comorbidities = self.generate_comorbidities()
        scores = self.generate_clinical_scores(demographics['age'], comorbidities)
        mri_finding = self.generate_imaging(scores['initial_nihss'])
        interventions = self.generate_interventions(
            scores['initial_nihss'],
            mri_finding,
            demographics['age']
        )
        
        # Combine structured data
        structured_data = {
            **demographics,
            **comorbidities,
            **scores,
            'mri_finding': mri_finding,
            **interventions
        }
        
        # Generate outcome
        outcome = self.generate_outcome(
            demographics['age'],
            scores['initial_nihss'],
            scores['aspect_score'],
            comorbidities,
            mri_finding,
            interventions
        )
        
        # Generate clinical note
        clinical_note = self.generate_clinical_note(structured_data)
        
        return structured_data, clinical_note, outcome
    
    def generate_dataset(
        self,
        n_patients: int = 1166,
        output_prefix: str = "synthetic_stroke_data"
    ) -> None:
        """
        Generate complete dataset matching paper statistics.
        
        Paper: 1,166 patients total, 767 with outcome data
        
        Args:
            n_patients: Number of patients to generate
            output_prefix: Prefix for output files
        """
        print(f"Generating {n_patients} synthetic patient records...")
        
        structured_data_list = []
        clinical_notes = []
        outcomes = []
        
        for i in range(n_patients):
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{n_patients} patients")
            
            structured_data, note, outcome = self.generate_patient()
            
            structured_data_list.append(structured_data)
            clinical_notes.append(note)
            outcomes.append(outcome)
        
        # Save structured data (ground truth for validation)
        with open(f'{output_prefix}_structured.json', 'w') as f:
            json.dump(structured_data_list, f, indent=2)
        
        # Save clinical notes
        with open(f'{output_prefix}_notes.json', 'w') as f:
            json.dump(clinical_notes, f, indent=2)
        
        # Save outcomes
        outcomes_df = pd.DataFrame({
            'patient_id': range(n_patients),
            'poor_outcome': outcomes
        })
        outcomes_df.to_csv(f'{output_prefix}_outcomes.csv', index=False)
        
        # Print statistics
        print("\n" + "="*50)
        print("GENERATED DATASET STATISTICS")
        print("="*50)
        
        df = pd.DataFrame(structured_data_list)
        
        print(f"\nDemographics:")
        print(f"  Age: {df['age'].mean():.2f} ± {df['age'].std():.2f}")
        print(f"  Male: {(df['sex'] == 'male').mean()*100:.1f}%")
        
        print(f"\nComorbidities:")
        print(f"  Hypertension: {(df['hypertension'] == 'yes').mean()*100:.1f}%")
        print(f"  Diabetes: {(df['diabetes_mellitus'] == 'yes').mean()*100:.1f}%")
        print(f"  Atrial fibrillation: {(df['atrial_fibrillation'] == 'yes').mean()*100:.1f}%")
        
        print(f"\nClinical scores:")
        print(f"  NIHSS median: {df['initial_nihss'].median():.0f} (IQR: {df['initial_nihss'].quantile(0.25):.0f}-{df['initial_nihss'].quantile(0.75):.0f})")
        print(f"  ASPECT median: {df['aspect_score'].median():.0f}")
        
        print(f"\nImaging:")
        print(f"  MRI infarction: {(df['mri_finding'] == 'acute_infarction').mean()*100:.1f}%")
        
        print(f"\nInterventions:")
        print(f"  IV t-PA: {(df['iv_tpa'] == 'yes').mean()*100:.1f}%")
        print(f"  IA intervention: {(df['ia_intervention'] == 'yes').mean()*100:.1f}%")
        
        print(f"\nOutcomes:")
        print(f"  Poor outcome (mRS 3-6): {outcomes_df['poor_outcome'].mean()*100:.1f}%")
        
        print(f"\nClinical notes:")
        note_lengths = [len(note) for note in clinical_notes]
        print(f"  Average length: {np.mean(note_lengths):.0f} ± {np.std(note_lengths):.0f} characters")
        
        print(f"\nFiles saved:")
        print(f"  - {output_prefix}_structured.json (ground truth)")
        print(f"  - {output_prefix}_notes.json (raw clinical notes)")
        print(f"  - {output_prefix}_outcomes.csv (3-month outcomes)")


if __name__ == "__main__":
    # Generate dataset matching paper statistics
    generator = SyntheticStrokeDataGenerator(seed=42)
    
    # Full dataset
    generator.generate_dataset(
        n_patients=1166,
        output_prefix="data/synthetic_stroke_data"
    )
    
    # Also generate small test set
    generator.generate_dataset(
        n_patients=50,
        output_prefix="data/test_set"
    )
    
    print("\n✓ Synthetic data generation complete!")
    print("\nNote: This is FAKE data for demonstration only.")
    print("Real patient data cannot be shared due to IRB restrictions.")
