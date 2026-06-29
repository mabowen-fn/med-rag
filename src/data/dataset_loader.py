"""Dataset loader for Medical RAG System"""
import json
from pathlib import Path
from typing import Optional
from datasets import load_dataset
from loguru import logger
from config import config


class DatasetLoader:
    """Load and preprocess medical datasets (Huatuo-26M, CMeQA)"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or config.data.cache_dir
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def load_huatuo_26m(self, subset: str = "default", split: str = "train", max_samples: Optional[int] = None):
        """Load Huatuo-26M dataset (composed of multiple sub-datasets)"""
        logger.info(f"Loading Huatuo-26M dataset")
        
        # Huatuo-26M is split into multiple datasets on HuggingFace
        dataset_names = [
            "FreedomIntelligence/huatuo_knowledge_graph_qa",
            "FreedomIntelligence/huatuo_encyclopedia_qa",
        ]
        
        all_data = []
        
        for dataset_name in dataset_names:
            try:
                logger.info(f"Loading {dataset_name}...")
                dataset = load_dataset(
                    dataset_name,
                    cache_dir=self.cache_dir,
                )
                
                # Get the data (handle different splits)
                if isinstance(dataset, dict):
                    # Dataset has splits, use the first available
                    for split_name in dataset.keys():
                        data = dataset[split_name]
                        break
                else:
                    data = dataset
                
                # Convert to list of dicts with standard format
                for item in data:
                    # Huatuo datasets use 'questions' and 'answers' fields
                    if 'questions' in item and 'answers' in item:
                        all_data.append({
                            "instruction": item['questions'],
                            "output": item['answers']
                        })
                
                logger.info(f"Loaded {len(data)} samples from {dataset_name}")
                
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
                continue
        
        if not all_data:
            logger.warning("Failed to load any Huatuo-26M data")
            logger.info("Falling back to synthetic medical data")
            return self._load_synthetic_medical_data(max_samples)
        
        # Limit samples if requested
        if max_samples and max_samples < len(all_data):
            all_data = all_data[:max_samples]
        
        logger.info(f"Total Huatuo-26M samples loaded: {len(all_data)}")
        return all_data
    
    def load_cmeqa(self, split: str = "test"):
        """Load CMeQA dataset for multi-turn evaluation"""
        logger.info(f"Loading CMeQA dataset (split={split})")
        try:
            dataset = load_dataset(
                "CMeQA",
                split=split,
                cache_dir=self.cache_dir,
            )
            logger.info(f"Loaded {len(dataset)} samples from CMeQA")
            return dataset
        except Exception as e:
            logger.warning(f"Failed to load CMeQA from hub, trying local: {e}")
            return self._load_local_cmeqa(split)
    
    def _load_synthetic_medical_data(self, max_samples: Optional[int] = None):
        """Generate synthetic medical QA data for demonstration"""
        logger.info("Generating synthetic medical QA data")
        
        synthetic_data = [
            {
                "instruction": "What are the common symptoms of diabetes?",
                "output": "Common symptoms of diabetes include frequent urination, increased thirst, unexplained weight loss, fatigue, blurred vision, slow-healing wounds, and frequent infections. Type 1 diabetes symptoms often appear suddenly, while Type 2 diabetes develops gradually."
            },
            {
                "instruction": "How is hypertension treated?",
                "output": "Hypertension is treated through lifestyle modifications and medications. Lifestyle changes include reducing salt intake, regular exercise, maintaining healthy weight, limiting alcohol, and stress management. Medications include ACE inhibitors, ARBs, calcium channel blockers, diuretics, and beta-blockers. Treatment depends on severity and patient factors."
            },
            {
                "instruction": "What causes myocardial infarction?",
                "output": "Myocardial infarction (heart attack) is primarily caused by coronary artery disease, where plaque buildup narrows arteries. When plaque ruptures, a blood clot forms and blocks blood flow to the heart muscle. Risk factors include high blood pressure, high cholesterol, smoking, diabetes, obesity, sedentary lifestyle, and family history."
            },
            {
                "instruction": "What are the treatment options for asthma?",
                "output": "Asthma treatment includes quick-relief medications (short-acting beta agonists like albuterol) for acute symptoms and long-term control medications (inhaled corticosteroids, long-acting beta agonists, leukotriene modifiers). Avoiding triggers, using inhalers correctly, and following an asthma action plan are essential for management."
            },
            {
                "instruction": "How is pneumonia diagnosed?",
                "output": "Pneumonia is diagnosed through physical examination (listening for abnormal lung sounds), chest X-ray to confirm infection location, blood tests to identify the causative organism, sputum culture, pulse oximetry to measure oxygen levels, and sometimes CT scan or bronchoscopy for severe cases."
            },
            {
                "instruction": "What are the symptoms of stroke?",
                "output": "Stroke symptoms appear suddenly and include: facial drooping, arm weakness, speech difficulty (slurred or unable to speak), confusion, severe headache with no known cause, vision problems, dizziness, loss of balance or coordination, and difficulty walking. Remember FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency."
            },
            {
                "instruction": "How is type 2 diabetes managed?",
                "output": "Type 2 diabetes management includes: blood sugar monitoring, healthy diet with controlled carbohydrate intake, regular physical activity (150 minutes per week), oral medications (metformin, sulfonylureas, DPP-4 inhibitors), insulin therapy if needed, weight management, and regular check-ups for complications like eye, kidney, and nerve damage."
            },
            {
                "instruction": "What are the risk factors for osteoporosis?",
                "output": "Osteoporosis risk factors include: age (especially over 50), female gender, menopause, family history, low body weight, calcium and vitamin D deficiency, sedentary lifestyle, smoking, excessive alcohol consumption, certain medications (corticosteroids), and medical conditions like rheumatoid arthritis, thyroid disorders, and celiac disease."
            },
            {
                "instruction": "How is depression treated?",
                "output": "Depression treatment includes psychotherapy (cognitive behavioral therapy, interpersonal therapy), medications (SSRIs, SNRIs, tricyclic antidepressants), lifestyle changes (regular exercise, healthy sleep, social support), and in severe cases, brain stimulation therapies (ECT, TMS). Treatment often combines multiple approaches for best results."
            },
            {
                "instruction": "What are the complications of uncontrolled diabetes?",
                "output": "Uncontrolled diabetes can lead to: cardiovascular disease (heart attack, stroke), neuropathy (nerve damage), nephropathy (kidney damage), retinopathy (eye damage leading to blindness), foot problems (infections, ulcers, amputations), skin conditions, hearing impairment, and increased risk of Alzheimer's disease."
            },
        ]
        
        if max_samples and max_samples < len(synthetic_data):
            synthetic_data = synthetic_data[:max_samples]
        
        logger.info(f"Generated {len(synthetic_data)} synthetic medical QA samples")
        return synthetic_data
    
    def _load_local_cmeqa(self, split: str):
        """Fallback: load local CMeQA if available"""
        local_path = Path(self.cache_dir) / f"cmeqa_{split}.json"
        if local_path.exists():
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} samples from local CMeQA")
            return data
        raise FileNotFoundError(f"Local CMeQA not found at {local_path}")
    
    def save_processed(self, data, filename: str):
        """Save processed data to disk"""
        output_path = Path(config.data.processed_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved processed data to {output_path}")
    
    def load_processed(self, filename: str):
        """Load processed data from disk"""
        input_path = Path(config.data.processed_dir) / filename
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data not found at {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded processed data from {input_path}")
        return data

