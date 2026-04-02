import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset, concatenate_datasets

import config


def load_sts_sentences() -> list[str]:
    """STSBenchmarkのtrain/dev/testから全文を取得し、重複を除去する。"""
    ds = load_dataset(config.DATASET_NAME)
    all_sentences = set()
    for split in ds:
        for row in ds[split]:
            all_sentences.add(row["sentence1"])
            all_sentences.add(row["sentence2"])
    return list(all_sentences)


def sample_anchors_and_queries(
    num_anchors: int = config.NUM_ANCHORS,
    num_queries: int = config.NUM_QUERIES,
    seed: int = config.RANDOM_SEED,
) -> tuple[list[str], list[str]]:
    """アンカーとクエリを重複なしでサンプリングする。"""
    sentences = load_sts_sentences()
    print(f"STSBenchmarkから {len(sentences)} 件のユニーク文を取得")

    total_needed = num_anchors + num_queries
    if len(sentences) < total_needed:
        raise ValueError(
            f"文が足りません: {len(sentences)} < {total_needed} (anchors={num_anchors} + queries={num_queries})"
        )

    rng = random.Random(seed)
    sampled = rng.sample(sentences, total_needed)
    anchors = sampled[:num_anchors]
    queries = sampled[num_anchors:]

    return anchors, queries


def save_data(anchors: list[str], queries: list[str], data_dir: Path = config.DATA_DIR):
    """アンカーとクエリをJSONで保存する。"""
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "anchors.json", "w", encoding="utf-8") as f:
        json.dump(anchors, f, ensure_ascii=False, indent=2)
    with open(data_dir / "queries.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    print(f"保存完了: anchors={len(anchors)}件, queries={len(queries)}件 → {data_dir}")
