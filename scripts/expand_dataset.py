"""
Script to expand medical QA dataset with MedQA and PubMedQA
Downloads from HuggingFace and merges with existing dataset
"""
import json
import random
from pathlib import Path
from datasets import load_dataset
from loguru import logger


def load_medqa_dataset(split="train", max_samples=200):
    """Load MedMCQA dataset (Medical MCQ questions)"""
    logger.info("Loading MedMCQA dataset...")
    try:
        dataset = load_dataset("openlifescienceai/medmcqa", split=split)
        logger.info(f"Loaded {len(dataset)} MedMCQA samples")
        
        qa_pairs = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            
            # Extract question and answer
            question = item.get("question", "")
            options = [item.get("opa", ""), item.get("opb", ""), item.get("opc", ""), item.get("opd", "")]
            answer_idx = item.get("cop", 0) - 1  # cop is 1-indexed
            
            if question and answer_idx >= 0 and answer_idx < len(options):
                answer = options[answer_idx]
                qa_pairs.append({
                    "id": f"medmcqa_{i:04d}",
                    "category": "general_medicine",
                    "question": question,
                    "answer": answer,
                    "difficulty": "medium",
                    "keywords": [],
                    "source": "MedMCQA"
                })
        
        logger.info(f"Processed {len(qa_pairs)} MedMCQA QA pairs")
        return qa_pairs
    except Exception as e:
        logger.error(f"Failed to load MedMCQA: {e}")
        return []


def load_pubmedqa_dataset(split="train", max_samples=200):
    """Load PubMedQA dataset (research paper Q&A)"""
    logger.info("Loading PubMedQA dataset...")
    try:
        dataset = load_dataset("qiaojing/pubmed_qa", split=split)
        logger.info(f"Loaded {len(dataset)} PubMedQA samples")
        
        qa_pairs = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            
            # Extract question and answer
            question = item.get("question", "")
            answer = item.get("long_answer", "")
            
            if question and answer:
                qa_pairs.append({
                    "id": f"pubmedqa_{i:04d}",
                    "category": "medical_research",
                    "question": question,
                    "answer": answer,
                    "difficulty": "hard",
                    "keywords": [],
                    "source": "PubMedQA"
                })
        
        logger.info(f"Processed {len(qa_pairs)} PubMedQA QA pairs")
        return qa_pairs
    except Exception as e:
        logger.error(f"Failed to load PubMedQA: {e}")
        return []


def expand_dataset():
    """Main function to expand the dataset"""
    logger.info("Starting dataset expansion...")
    
    # Load existing dataset
    existing_path = Path("data/medical_qa_dataset.json")
    if existing_path.exists():
        with open(existing_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        logger.info(f"Loaded {len(existing_data['qa_pairs'])} existing QA pairs")
    else:
        existing_data = {"dataset_info": {}, "qa_pairs": []}
        logger.warning("No existing dataset found, starting fresh")
    
    # Load new datasets
    medqa_pairs = load_medqa_dataset(max_samples=200)
    pubmedqa_pairs = load_pubmedqa_dataset(max_samples=200)
    
    # Merge all QA pairs
    all_pairs = existing_data["qa_pairs"] + medqa_pairs + pubmedqa_pairs
    
    # Update dataset info
    existing_data["dataset_info"] = {
        "name": "Expanded Medical QA Dataset",
        "version": "2.0",
        "description": "Comprehensive medical question-answer pairs including MedQA and PubMedQA",
        "total_pairs": len(all_pairs),
        "categories": list(set(qa["category"] for qa in all_pairs)),
        "sources": ["Original", "MedQA", "PubMedQA"]
    }
    existing_data["qa_pairs"] = all_pairs
    
    # Save expanded dataset
    output_path = Path("data/medical_qa_dataset_expanded.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Expanded dataset saved to {output_path}")
    logger.info(f"Total QA pairs: {len(all_pairs)}")
    logger.info(f"  - Original: {len(existing_data['qa_pairs']) - len(medqa_pairs) - len(pubmedqa_pairs)}")
    logger.info(f"  - MedQA: {len(medqa_pairs)}")
    logger.info(f"  - PubMedQA: {len(pubmedqa_pairs)}")
    
    return existing_data


if __name__ == "__main__":
    expand_dataset()
