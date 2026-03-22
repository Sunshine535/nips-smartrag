# SmartRAG: Graph-Enhanced Retrieval with Adaptive Policy for Knowledge-Intensive QA

## One-Sentence Summary

We propose SmartRAG, a two-level improvement to RAG systems that combines graph-supervised contrastive learning for higher-quality retrieval with a GRPO-trained cost-aware policy that adaptively decides when, what, and how much to retrieve.

## Problem

Current RAG systems suffer from two independent but compounding limitations:

1. **Retrieval quality**: Dense retrievers (DPR, BGE) treat queries and documents as independent embedding vectors, ignoring linguistic relationships (synonyms, polysemy, hypernymy) that could disambiguate queries.
2. **Retrieval policy**: Fixed retrieval strategies (always retrieve top-k) waste compute on easy questions and under-retrieve for complex ones, with no principled way to balance accuracy and retrieval cost.

## Approach

### Component 1: Graph-Contrastive Retriever

We build a synonym/polysemy graph from Wikipedia using:
- spaCy NER for entity extraction
- WordNet for synonym/hypernym edges
- Embedding cosine similarity for soft synonym detection
- Polysemy detection via multiple WordNet senses

A multi-head Graph Attention Network (GAT) propagates information over this graph, producing graph-aware embeddings. These are fused with standard BGE embeddings via a learned gating mechanism. Training uses:
- InfoNCE contrastive loss with BM25 hard negatives
- Graph Laplacian regularization for embedding smoothness

### Component 2: Cost-Aware Retrieval Policy

We model RAG as an MDP with 7 actions:
- NO_RETRIEVE (cost 0), RETRIEVE_1/3/5/10 (cost 0.1-1.0), REWRITE (cost 0.2), MULTI_HOP (cost 1.5)

The policy is trained in two stages:
1. **Oracle warm-start**: For each question, try all actions and record the best accuracy-cost tradeoff. Train an MLP oracle on (query_embedding → best_action).
2. **GRPO fine-tuning**: Use Qwen3.5-4B with LoRA, 8 generations per prompt, reward = accuracy - λ·cost. Cost annealing: λ increases linearly from 0 to final value.

### Integration

The graph-contrastive retriever serves as the backend for the policy's retrieve actions. When the policy decides to retrieve k documents, it uses the graph-enhanced retriever instead of vanilla BGE, creating a multiplicative improvement.

## Experiments

| Phase | Details |
|-------|---------|
| Graph construction | 50K Wikipedia articles, ~500K nodes, WordNet + embedding edges |
| Retriever training | BGE-large backbone, 10 epochs, InfoNCE + Laplacian loss |
| Policy training | Qwen3.5-4B + LoRA, 10 epochs GRPO, 7 Pareto λ values |
| Combined eval | 6 configurations × 5 datasets |
| Ablations | Graph edges, action space, cost schedule, GNN depth, backbone |

## Benchmarks

- **Natural Questions** (factoid QA)
- **TriviaQA** (trivia, open-domain)
- **HotpotQA** (multi-hop reasoning)
- **AmbigQA** (ambiguous queries, tests polysemy handling)
- **FEVER** (fact verification)

Metrics: Exact Match, F1, Recall@k, MRR, avg retrieval cost, latency.

## Expected Contributions

1. Graph-contrastive retriever that handles synonyms and polysemy
2. First RL-based cost-aware retrieval policy with principled action space
3. Demonstration that retriever quality and retrieval policy improvements are complementary
4. Pareto frontier analysis of accuracy-cost tradeoffs across datasets
5. Comprehensive ablations validating each component

## NeurIPS Justification

Addresses two fundamental RAG limitations with principled solutions (graph learning + RL policy). The two-level framework is novel and the evaluation is comprehensive across 5 diverse benchmarks.
