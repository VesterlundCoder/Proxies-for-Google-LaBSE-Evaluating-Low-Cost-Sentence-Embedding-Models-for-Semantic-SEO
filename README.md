# Proxies-for-Google-LaBSE-Evaluating-Low-Cost-Sentence-Embedding-Models-for-Semantic-SEO
Comparative analysis of sentence embedding models as low-cost proxies for Google LaBSE in Semantic SEO, AI Search, content relevance, link relevance and topical authority scoring. The coding stack for the analysis is stored here.
# High-Fidelity Proxies for Google LaBSE

This repository contains the code, data summaries, and figures used in the paper:

> **High-Fidelity Proxies for Google LaBSE: Evaluating Low-Cost Sentence Embedding Models for Semantic SEO and Content Relevance**  
> David Vesterlund, IncRev SEO Research Community

## Overview

The goal of this project is to identify public sentence embedding models that act as high-fidelity, low-cost proxies for **Google LaBSE** on English text similarity tasks. The analysis is designed for **Semantic SEO**, **AI Search**, **content relevance**, **link relevance**, and **topical authority** use cases.

We compare:
- `sentence-transformers/all-mpnet-base-v2`
- `sentence-transformers/all-MiniLM-L6-v2`
- `thenlper/gte-base`
- (reference model) `sentence-transformers/LaBSE`

on:
- **STS-B** (sentence similarity)
- **SimLex-999** (word similarity)
- **SemEval STS 2012–2016** (legacy STS datasets, used in the full analysis)

Similarity is measured with **cosine similarity**, and model quality is evaluated via **Spearman correlation** with LaBSE and with human gold labels.

## Files

- `analyze_labse_correlations.py` – loads CSV results, computes LaBSE correlations, and prints/exports summary tables.
- `RESEARCH_FINDINGS.md` – high-level summary of the experiments.
- `LABSE_PROXY_ANALYSIS.md` – detailed notes and interpretation.
- `spearman_vs_gold_summary.csv` – Spearman vs. gold per dataset/model.
- `labse_proxy_rankings.csv` – global LaBSE proxy rankings.
- `global_model_correlations.csv` – inter-model correlation matrix.
- `relative_performance_vs_labse.csv` – relative performance vs. LaBSE.
- `figures/` – all plots used in the report (performance, correlations, modern-only analysis, etc.).

## Quick start

```bash
pip install -r requirements.txt

python analyze_labse_correlations.py
```

This will:
- read the CSV files in the repo,
- compute summary statistics,
- and write updated ranking tables for LaBSE proxy selection.

## Use cases

These results can be used to:
- choose a **low-cost proxy for LaBSE** in large-scale Semantic SEO pipelines,
- score **query–content** and **link relevance** in tools like QueryMatch,
- monitor **topicality** and **topical authority** across large content corpora.

## Citation

If you use this work in research or production, please cite:

> David Vesterlund. *High-Fidelity Proxies for Google LaBSE: Evaluating Low-Cost Sentence Embedding Models for Semantic SEO and Content Relevance* (2025).

## License

Code in this repository is released under the MIT License (see `LICENSE`).
