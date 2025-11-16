#!/usr/bin/env python3
"""
Script to add OpenAI text-embedding-3-large comparison to existing results.
Run this after setting OPENAI_API_KEY environment variable.

Usage:
export OPENAI_API_KEY='your-api-key-here'
python add_openai_comparison.py
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from openai import OpenAI
from tqdm import tqdm

def cosine_similarity_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two matrices."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a_norm * b_norm, axis=1)

class OpenAIEmbedder:
    def __init__(self, model_name: str = "text-embedding-3-large"):
        self.client = OpenAI()
        self.model_name = model_name

    def encode(self, texts, batch_size: int = 32):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="OpenAI batches"):
            batch = texts[i: i + batch_size]
            try:
                resp = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                )
                batch_emb = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
                all_embeddings.extend(batch_emb)
            except Exception as e:
                print(f"OpenAI API Error: {e}")
                raise
        return np.vstack(all_embeddings)

def load_test_data():
    """Load a small subset for OpenAI testing (to minimize costs)."""
    from datasets import load_dataset
    
    # Load just STS-B test set (smaller sample)
    ds = load_dataset("sentence-transformers/stsb")
    df = ds['test'].to_pandas()
    
    # Take first 100 pairs to minimize API costs
    df = df.head(100)
    
    return {
        'text1': df['sentence1'].tolist(),
        'text2': df['sentence2'].tolist(),
        'gold_score': df['score'].tolist()
    }

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return

    print("🔑 OpenAI API key found!")
    print("⚠️  This will make API calls to OpenAI (costs money)")
    print("📊 Testing with 100 text pairs from STS-B to minimize costs")
    
    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Load test data
    print("\n📥 Loading test data...")
    data = load_test_data()
    
    # Initialize OpenAI embedder
    print("🤖 Initializing OpenAI embedder...")
    openai_embedder = OpenAIEmbedder()
    
    # Compute OpenAI embeddings
    print("🔄 Computing OpenAI embeddings...")
    emb1 = openai_embedder.encode(data['text1'])
    emb2 = openai_embedder.encode(data['text2'])
    
    # Compute similarities
    openai_sims = cosine_similarity_batch(emb1, emb2)
    
    # Compute correlation with gold standard
    gold = np.array(data['gold_score'])
    openai_corr, _ = spearmanr(gold, openai_sims)
    
    print(f"\n📈 OpenAI Results (100 STS-B pairs):")
    print(f"   Spearman correlation vs gold: {openai_corr:.4f}")
    
    # Load existing model results for comparison
    try:
        existing_results = pd.read_csv("spearman_vs_gold_summary.csv")
        stsb_results = existing_results[existing_results['dataset'] == 'STS-B'].copy()
        
        print(f"\n🔄 Comparison with existing models (STS-B dataset):")
        for _, row in stsb_results.iterrows():
            print(f"   {row['model']:12s}: {row['spearman_vs_gold']:.4f}")
        print(f"   {'openai':12s}: {openai_corr:.4f}")
        
        # Find best proxy for OpenAI
        print(f"\n🎯 Best proxy for OpenAI text-embedding-3-large:")
        best_proxy = stsb_results.loc[stsb_results['spearman_vs_gold'].idxmax()]
        print(f"   Model: {best_proxy['model']}")
        print(f"   Correlation: {best_proxy['spearman_vs_gold']:.4f}")
        
    except FileNotFoundError:
        print("⚠️  No existing results found. Run the main comparison first.")
    
    # Save OpenAI results
    openai_results = pd.DataFrame({
        'dataset': ['STS-B'],
        'type': ['sentence'],
        'model': ['openai'],
        'spearman_vs_gold': [openai_corr]
    })
    
    openai_results.to_csv("openai_comparison_results.csv", index=False)
    print(f"\n💾 OpenAI results saved to openai_comparison_results.csv")
    
    print(f"\n✅ OpenAI comparison complete!")
    print(f"💰 Estimated cost: ~$0.01-0.02 for 200 embeddings")

if __name__ == "__main__":
    main()
