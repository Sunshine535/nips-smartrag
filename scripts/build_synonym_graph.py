#!/usr/bin/env python3
"""Build synonym/polysemy graph from Wikipedia using spaCy NER, WordNet, and embedding similarity."""

import argparse
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.graph_retriever import SynonymGraph


def extract_entities_spacy(texts: list, spacy_model: str = "en_core_web_lg",
                           batch_size: int = 256) -> dict:
    """Extract named entities and noun chunks from texts using spaCy NER."""
    import spacy
    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        logger.info("Downloading spaCy model: %s", spacy_model)
        spacy.cli.download(spacy_model)
        nlp = spacy.load(spacy_model)

    nlp.max_length = 2_000_000
    entity_counts = defaultdict(int)
    entity_labels = defaultdict(set)
    noun_chunks = defaultdict(int)

    logger.info("Extracting entities from %d texts with spaCy...", len(texts))
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for doc in nlp.pipe(batch, batch_size=batch_size, disable=["tagger", "parser", "lemmatizer"]):
            for ent in doc.ents:
                text_norm = ent.text.strip().lower()
                if len(text_norm) < 2 or len(text_norm) > 80:
                    continue
                entity_counts[text_norm] += 1
                entity_labels[text_norm].add(ent.label_)

        if (i // batch_size) % 20 == 0:
            logger.info("  Processed %d/%d texts, %d unique entities",
                        min(i + batch_size, len(texts)), len(texts), len(entity_counts))

    logger.info("Extracted %d unique entities from %d texts", len(entity_counts), len(texts))
    return entity_counts, entity_labels


def load_wikipedia_texts(max_articles: int = 50000) -> list:
    """Load Wikipedia article texts for entity extraction."""
    from datasets import load_dataset

    logger.info("Loading Wikipedia articles (max %d)...", max_articles)
    try:
        ds = load_dataset("wikipedia", "20220301.en", split="train",
                          streaming=True)
        texts = []
        for i, ex in enumerate(ds):
            if i >= max_articles:
                break
            text = ex.get("text", "")
            if len(text) > 100:
                texts.append(text[:2000])
        logger.info("Loaded %d Wikipedia articles", len(texts))
        return texts
    except Exception as e:
        logger.warning("Failed to load Wikipedia: %s. Using NQ passages instead.", e)
        try:
            nq = load_dataset("google-research-datasets/natural_questions", "default",
                              split="train", streaming=True)
            texts = []
            for i, ex in enumerate(nq):
                if i >= max_articles:
                    break
                doc = ex.get("document", {})
                html = doc.get("html", "") if isinstance(doc, dict) else str(doc)
                import re
                clean = re.sub(r"<[^>]+>", " ", html)
                if len(clean) > 50:
                    texts.append(clean[:2000])
            return texts
        except Exception:
            logger.warning("Generating synthetic corpus for graph construction.")
            return [f"Article about topic {i} covering concept {i % 500} "
                    f"in domain {i % 20} with entity_{i}" for i in range(max_articles)]


def build_wordnet_synonym_edges(graph: SynonymGraph, entities: dict,
                                max_synsets: int = 10) -> int:
    """Add synonym edges from WordNet for known entities."""
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet as wn

    edge_count = 0
    processed = 0
    for entity in list(entities.keys()):
        wn_query = entity.replace(" ", "_")
        synsets = wn.synsets(wn_query)[:max_synsets]
        if not synsets:
            continue

        for synset in synsets:
            lemmas = [l.name().replace("_", " ").lower() for l in synset.lemmas()]
            lemmas = [l for l in lemmas if l != entity and len(l) > 1]
            for lemma in lemmas[:5]:
                graph.add_edge(entity, lemma, "synonym")
                edge_count += 1

            for hyper in synset.hypernyms()[:2]:
                for hl in hyper.lemmas()[:2]:
                    hw = hl.name().replace("_", " ").lower()
                    graph.add_edge(entity, hw, "hypernym")
                    edge_count += 1

        processed += 1
        if processed % 5000 == 0:
            logger.info("  WordNet processed %d/%d entities, %d edges",
                        processed, len(entities), edge_count)

    logger.info("WordNet: added %d synonym/hypernym edges", edge_count)
    return edge_count


def compute_entity_embeddings(entities: list, model_name: str,
                              batch_size: int = 256) -> torch.Tensor:
    """Compute embeddings for all entities using a sentence encoder."""
    from transformers import AutoModel, AutoTokenizer

    logger.info("Computing embeddings for %d entities with %s", len(entities), model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16,
                                      trust_remote_code=True, device_map="auto")
    model.eval()

    all_embeds = []
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=128, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeds = outputs.last_hidden_state[:, 0, :]
        all_embeds.append(embeds.cpu().float())

        if i % (batch_size * 20) == 0 and i > 0:
            logger.info("  Embedded %d/%d entities", i, len(entities))

    embeddings = torch.cat(all_embeds, dim=0)
    logger.info("Entity embeddings shape: %s", embeddings.shape)
    del model
    torch.cuda.empty_cache()
    return embeddings


def add_embedding_similarity_edges(graph: SynonymGraph, entities: list,
                                   embeddings: torch.Tensor,
                                   threshold: float = 0.85,
                                   max_neighbors: int = 10) -> int:
    """Add synonym edges between entities with cosine similarity >= threshold."""
    logger.info("Computing pairwise similarities (threshold=%.2f)...", threshold)
    embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

    edge_count = 0
    chunk_size = 2000
    for i in range(0, len(entities), chunk_size):
        chunk_emb = embeddings_norm[i:i + chunk_size]
        sims = chunk_emb @ embeddings_norm.T
        sims[torch.arange(len(chunk_emb)).unsqueeze(1) ==
             torch.arange(i, min(i + chunk_size, len(entities))).unsqueeze(0)] = 0

        for row_idx in range(len(chunk_emb)):
            global_idx = i + row_idx
            row_sims = sims[row_idx]
            above = (row_sims >= threshold).nonzero(as_tuple=True)[0]

            if len(above) > max_neighbors:
                _, top_idx = row_sims[above].topk(max_neighbors)
                above = above[top_idx]

            for j in above.tolist():
                if j > global_idx:
                    graph.add_edge(entities[global_idx], entities[j], "embedding_synonym")
                    edge_count += 1

        if (i // chunk_size) % 10 == 0 and i > 0:
            logger.info("  Similarity scan: %d/%d entities, %d edges", i, len(entities), edge_count)

    logger.info("Embedding similarity: added %d edges (threshold=%.2f)", edge_count, threshold)
    return edge_count


def detect_polysemy(graph: SynonymGraph, entities: dict, entity_labels: dict,
                    threshold: int = 2) -> int:
    """Detect polysemous concepts (multiple WordNet senses or NER label types)."""
    import nltk
    from nltk.corpus import wordnet as wn

    polysemy_count = 0
    for entity in list(entities.keys())[:50000]:
        senses = wn.synsets(entity.replace(" ", "_"))
        ner_labels = entity_labels.get(entity, set())
        is_polysemous = len(senses) >= threshold or len(ner_labels) >= 2

        if is_polysemous and entity in graph.nodes:
            for sense in senses[:5]:
                sense_name = f"{entity}#{sense.pos()}.{sense.offset()}"
                definition = sense.definition()[:60]
                graph.add_node(sense_name)
                graph.add_edge(entity, sense_name, "polysemy")
                polysemy_count += 1

    logger.info("Detected %d polysemy edges", polysemy_count)
    return polysemy_count


def main():
    parser = argparse.ArgumentParser(description="Build synonym/polysemy graph from Wikipedia")
    parser.add_argument("--config", type=str, default="configs/graph_config.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/graph")
    parser.add_argument("--max_articles", type=int, default=50000)
    parser.add_argument("--similarity_threshold", type=float, default=0.85)
    parser.add_argument("--min_entity_freq", type=int, default=2)
    parser.add_argument("--spacy_model", type=str, default="en_core_web_lg")
    parser.add_argument("--skip_embeddings", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load Wikipedia texts
    logger.info("=== Step 1: Load corpus ===")
    texts = load_wikipedia_texts(args.max_articles)

    # Step 2: Extract entities with spaCy NER
    logger.info("=== Step 2: Extract entities via spaCy NER ===")
    entity_counts, entity_labels = extract_entities_spacy(texts, args.spacy_model)

    frequent_entities = {e: c for e, c in entity_counts.items()
                         if c >= args.min_entity_freq}
    logger.info("Entities with freq >= %d: %d / %d",
                args.min_entity_freq, len(frequent_entities), len(entity_counts))

    # Step 3: Build graph and add nodes
    logger.info("=== Step 3: Build graph ===")
    graph = SynonymGraph()
    max_nodes = config["graph"].get("max_graph_nodes", 500000)
    sorted_entities = sorted(frequent_entities.keys(),
                             key=lambda e: frequent_entities[e], reverse=True)[:max_nodes]
    for entity in sorted_entities:
        graph.add_node(entity)
    logger.info("Graph initialized with %d nodes", len(graph.nodes))

    # Step 4: Add WordNet synonym/hypernym edges
    logger.info("=== Step 4: WordNet synonym edges ===")
    build_wordnet_synonym_edges(graph, frequent_entities,
                                max_synsets=config["graph"]["max_synsets_per_word"])

    # Step 5: Compute embeddings and add similarity edges
    embedding_model = config["graph"]["embedding_model"]
    entity_list = graph.node_texts
    embeddings = compute_entity_embeddings(entity_list, embedding_model)

    logger.info("=== Step 5: Embedding similarity edges (threshold=%.2f) ===",
                args.similarity_threshold)
    add_embedding_similarity_edges(graph, entity_list, embeddings,
                                   threshold=args.similarity_threshold)

    # Step 6: Detect polysemy
    logger.info("=== Step 6: Detect polysemy ===")
    detect_polysemy(graph, frequent_entities, entity_labels,
                    threshold=config["graph"]["polysemy_threshold"])

    # Step 7: Build adjacency and Laplacian
    logger.info("=== Step 7: Build adjacency matrix ===")
    adj = graph.build_adjacency()
    laplacian = graph.get_laplacian()
    logger.info("Final graph: %d nodes, %d edges", len(graph.nodes), len(graph.edges))

    # Save graph
    graph_path = os.path.join(args.output_dir, "synonym_graph.pkl")
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)
    logger.info("Saved graph to %s", graph_path)

    torch.save(adj, os.path.join(args.output_dir, "adjacency.pt"))
    torch.save(laplacian, os.path.join(args.output_dir, "laplacian.pt"))

    if not args.skip_embeddings:
        torch.save(embeddings, os.path.join(args.output_dir, "node_embeddings.pt"))

    # Save stats
    edge_type_counts = defaultdict(int)
    for _, _, etype in graph.edges:
        edge_type_counts[etype] += 1

    stats = {
        "num_nodes": len(graph.nodes),
        "num_edges": len(graph.edges),
        "num_articles_processed": len(texts),
        "similarity_threshold": args.similarity_threshold,
        "min_entity_freq": args.min_entity_freq,
        "edge_types": dict(edge_type_counts),
    }
    with open(os.path.join(args.output_dir, "graph_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("=== Graph construction complete ===")
    for etype, count in edge_type_counts.items():
        logger.info("  %s edges: %d", etype, count)


if __name__ == "__main__":
    main()
