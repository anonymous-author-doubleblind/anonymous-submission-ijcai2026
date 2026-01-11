#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta-learning episode splitter (K-shot, per-user) with dataset-aware paths.

Reads:
  data/<dataset>/train_valid.csv
  data/<dataset>/test.csv

Writes to:
  data/<dataset>/meta/
    - meta_train_support.csv
    - meta_train_query.csv
    - meta_test_support.csv
    - meta_test_query.csv
    - meta_config.json   (compact: unique users per split + skipped count)

Columns expected:
  train_valid.csv: user_id, question_id, correct, fold
  test.csv       : user_id, question_id, correct, fold
  (fold will be preserved as-is; no filtering is applied here.)

K-shot policy (train only):
  - For each user in train_valid:
      * Randomly shuffle that user's rows with a stable per-user RNG.
      * Take exactly K_support rows to support, exactly K_query rows to query.
      * If not enough rows (len < K_support + K_query), skip this user.
  - Test split is not re-split: full train_valid -> test-support; full test -> test-query.

Usage examples:
  python meta_datasplit.py --dataset frcsub --k_support 12 --k_query 4
  python meta_datasplit.py --dataset assist2009 --k_support 20 --k_query 8
"""

import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2] / "data"


def _stable_seed_for_user(user_id, base_seed: int) -> int:
    """Derive a stable per-user seed from (base_seed, user_id)."""
    h = hashlib.md5(f"{base_seed}:{user_id}".encode("utf-8")).hexdigest()
    # use 32-bit chunk to seed numpy RandomState deterministically
    return int(h[:8], 16)


def _load_csv(path: str, required_cols: Tuple[str, ...]) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    miss = [c for c in required_cols if c not in df.columns]
    if miss:
        raise ValueError(f"{path} missing columns: {miss}; has: {list(df.columns)}")
    return df


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["user_id", "question_id", "correct"]
    if "fold" in df.columns:
        cols.append("fold")
    return df.loc[:, cols].copy()


def split_train_kshot(train_valid: pd.DataFrame,
                      k_support: int,
                      k_query: int,
                      seed: int,
                      flex_query: bool = True,
                      min_query: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Return (train_support, train_query, skipped_users).

    strict模式（flex_query=False）:
        仅当 n_items >= k_support + k_query 才保留该学生；否则舍弃。
    灵活模式（flex_query=True，默认）:
        - 至少要保证 query 留 min_query 条；
        - 若 n_items >= k_support + k_query: 正常切分 (k_support, k_query)
        - 若 n_items <  k_support + k_query 且 n_items >= k_support + min_query:
              support = k_support, query = n_items - k_support
        - 若 n_items <  k_support + min_query 且 n_items >= min_query + 1:
              support = n_items - min_query, query = min_query
        - 若 n_items <  min_query + 1: 舍弃该学生
    """
    sup_rows, qry_rows = [], []
    skipped = 0

    cols = ["user_id", "question_id", "correct"] + (["fold"] if "fold" in train_valid.columns else [])

    for uid, g in train_valid.groupby("user_id", sort=False):
        rng = np.random.RandomState(_stable_seed_for_user(uid, seed))
        idx = g.index.to_numpy()
        idx = idx[rng.permutation(len(idx))]
        n = len(idx)

        if not flex_query:
            # 原来的严格策略：不够就跳过
            if n < (k_support + k_query):
                skipped += 1
                continue
            sup_cnt, qry_cnt = k_support, k_query
        else:
            # 灵活策略：尽量保留该学生，保证 query 至少 min_query 条
            if n < (min_query + 1):
                skipped += 1
                continue
            if n >= (k_support + k_query):
                sup_cnt, qry_cnt = k_support, k_query
            elif n >= (k_support + min_query):
                sup_cnt = k_support
                qry_cnt = n - sup_cnt
            else:
                # n < k_support + min_query 且 n >= min_query + 1
                qry_cnt = min_query
                sup_cnt = n - qry_cnt

        sup_idx = idx[:sup_cnt]
        qry_idx = idx[sup_cnt:sup_cnt + qry_cnt]
        sup_rows.append(train_valid.loc[sup_idx, cols])
        qry_rows.append(train_valid.loc[qry_idx, cols])

    if sup_rows:
        train_support = pd.concat(sup_rows, ignore_index=True)
        train_query   = pd.concat(qry_rows, ignore_index=True)
    else:
        train_support = pd.DataFrame(columns=cols)
        train_query   = pd.DataFrame(columns=cols)

    return train_support, train_query, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset name, e.g., frcsub or assist2009")
    ap.add_argument("--k_support", type=int, default=65, help="K-shot support per user for training split")
    ap.add_argument("--k_query", type=int, default=37, help="K-shot query  per user for training split")
    ap.add_argument("--seed", type=int, default=42, help="Base random seed (for per-user stable RNG)")
    ap.add_argument("--flex_query", action="store_true", default=True,
                help="启用灵活模式：不足 K_support+K_query 时，尽量保留该学生，query 至少保留 min_query 条（默认开启，可用 --no_flex_query 关闭）。")
    ap.add_argument("--min_query", type=int, default=1,
                    help="灵活模式下，query 至少保留的条数（默认 1）")

    args = ap.parse_args()

    ds_dir = BASE_DIR / args.dataset
    in_train = ds_dir / "train_valid.csv"
    in_test = ds_dir / "test.csv"
    out_dir = ds_dir / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    train_valid = _load_csv(in_train, ("user_id", "question_id", "correct"))
    test = _load_csv(in_test, ("user_id", "question_id", "correct"))

    train_valid = _select_columns(train_valid)
    test = _select_columns(test)

    # TRAIN: K-shot split
    # 原：train_sup, train_qry = split_train_kshot(train_valid, args.k_support, args.k_query, args.seed)
    train_sup, train_qry, skipped = split_train_kshot(
        train_valid,
        k_support=args.k_support,
        k_query=args.k_query,
        seed=args.seed,
        flex_query=args.flex_query,
        min_query=args.min_query
    )


    # TEST: full support/query (no re-split)
    test_sup = train_valid.copy()
    test_qry = test.copy()

    # Write CSVs
    paths = {
        "meta_train_support": out_dir / "meta_train_support.csv",
        "meta_train_query":   out_dir / "meta_train_query.csv",
        "meta_test_support":  out_dir / "meta_test_support.csv",
        "meta_test_query":    out_dir / "meta_test_query.csv",
    }
    train_sup.to_csv(paths["meta_train_support"], index=False)
    train_qry.to_csv(paths["meta_train_query"], index=False)
    test_sup.to_csv(paths["meta_test_support"], index=False)
    test_qry.to_csv(paths["meta_test_query"], index=False)

    total_users = int(train_valid["user_id"].nunique())
    kept_users  = int(train_sup["user_id"].nunique())
    assert kept_users + int(skipped) == total_users  # 简单一致性校验


    # Compact meta_config.json (only user counts + skipped)
    cfg = {
        "dataset": args.dataset,
        "k_shot": {"support": args.k_support, "query": args.k_query},
        "seed": args.seed,
        "users": {
            "total_users": total_users,
            "kept_users": kept_users,
            "skipped_users": int(skipped),  # << 新增：被舍弃
            "train_support": int(train_sup["user_id"].nunique()),
            "train_query":   int(train_qry["user_id"].nunique()),
            "test_support":  int(test_sup["user_id"].nunique()),
            "test_query":    int(test_qry["user_id"].nunique()),
            "train_skipped_users": int(skipped),  # << 新增：被舍弃学生数
        }
    }

    with open(out_dir / "meta_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # Console summary
    print(f"[OK] Wrote:")
    for k, p in paths.items():
        print(f"  - {k}: {p}")
    print(f"[OK] meta_config.json saved with user counts.")

if __name__ == "__main__":
    main()
