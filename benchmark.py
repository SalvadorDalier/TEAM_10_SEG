"""
Module benchmark.py
Đo lường và so sánh hiệu năng của 3 phương pháp chỉ mục:
- Recall@k
- QPS (Queries Per Second)
- Memory (Bộ nhớ RAM chiếm dụng - MB)
Đồng thời vẽ biểu đồ trực quan hóa kết quả.
"""

import os
import time
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from vector_db import VectorDB, METHOD_INVERTED, METHOD_HNSW, METHOD_LEARNED


def get_ground_truth(doc_vectors: np.ndarray, query_vectors: np.ndarray, k: int = 3) -> List[List[int]]:
    """Tính toán Ground Truth bằng phương pháp Brute-force Cosine Similarity chính xác 100%."""
    ground_truth: List[List[int]] = []
    for q in query_vectors:
        sims = np.dot(doc_vectors, q)
        top_k_indices = np.argsort(-sims)[:k].tolist()
        ground_truth.append(top_k_indices)
    return ground_truth


def benchmark_all(
    dbs: Dict[str, VectorDB],
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int = 3,
    n_repeat: int = 1000,
) -> pd.DataFrame:
    """Đo lường Recall@k, QPS, và Memory của từng phương pháp.

    Args:
        dbs: Dictionary chứa các instance VectorDB đã build.
        doc_vectors: Ma trận vector tài liệu (10, 384).
        query_vectors: Ma trận vector câu truy vấn (10, 384).
        k: Số lượng tài liệu top-k cần lấy.
        n_repeat: Số lần lặp lại tập truy vấn để tính QPS ổn định.

    Returns:
        pd.DataFrame: Bảng số liệu tổng hợp benchmark.
    """
    print(f"[INFO] Bắt đầu tính Ground Truth (Brute-force) với k={k}...")
    ground_truth = get_ground_truth(doc_vectors, query_vectors, k=k)

    records = []
    num_queries = len(query_vectors)

    for name, db in dbs.items():
        print(f"[INFO] Đang đánh giá phương pháp: {name.upper()}...")

        # 1. Tính Recall@k
        recalls = []
        for i, q in enumerate(query_vectors):
            pred = db.search(q, k=k)
            gt_set = set(ground_truth[i])
            pred_set = set(pred)
            intersection = len(gt_set.intersection(pred_set))
            recalls.append(intersection / float(k))
        avg_recall = float(np.mean(recalls))

        # 2. Tính QPS (Lặp n_repeat lần)
        start_time = time.perf_counter()
        for _ in range(n_repeat):
            for q in query_vectors:
                _ = db.search(q, k=k)
        elapsed_time = time.perf_counter() - start_time
        total_queries = num_queries * n_repeat
        qps = total_queries / elapsed_time if elapsed_time > 0 else 0.0

        # 3. Đo bộ nhớ RAM hiện tại của tiến trình (RSS in MB)
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024.0 * 1024.0)

        records.append({
            "Method": name,
            f"Recall@{k}": round(avg_recall, 4),
            "QPS": round(qps, 2),
            "Memory (MB)": round(mem_mb, 2),
        })

    df = pd.DataFrame(records)
    return df


def plot_results(df: pd.DataFrame, save_dir: str) -> None:
    """Vẽ 3 biểu đồ cột so sánh Recall@3, QPS, Memory và lưu vào thư mục save_dir."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        methods = df["Method"].tolist()
        recall_col = [c for c in df.columns if c.startswith("Recall")][0]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = ["#4A90E2", "#50E3C2", "#F5A623"]

        # 1. Biểu đồ Recall
        axes[0].bar(methods, df[recall_col], color=colors)
        axes[0].set_title(f"So sánh {recall_col}")
        axes[0].set_ylim(0, 1.1)
        for i, v in enumerate(df[recall_col]):
            axes[0].text(i, v + 0.02, str(v), ha="center", fontweight="bold")

        # 2. Biểu đồ QPS
        axes[1].bar(methods, df["QPS"], color=colors)
        axes[1].set_title("So sánh QPS (Queries Per Second)")
        for i, v in enumerate(df["QPS"]):
            axes[1].text(i, v + (max(df["QPS"]) * 0.01), f"{v:.1f}", ha="center", fontweight="bold")

        # 3. Biểu đồ Memory
        axes[2].bar(methods, df["Memory (MB)"], color=colors)
        axes[2].set_title("So sánh Bộ nhớ RAM (MB)")
        for i, v in enumerate(df["Memory (MB)"]):
            axes[2].text(i, v + 0.5, f"{v:.1f}", ha="center", fontweight="bold")

        plt.tight_layout()
        plot_file_path = os.path.join(save_dir, "benchmark_comparison.png")
        plt.savefig(plot_file_path, dpi=300)
        plt.close()
        print(f"[DONE] Đã lưu biểu đồ kết quả tại: {plot_file_path}")
    except Exception as e:
        print(f"[ERROR] Lỗi khi vẽ biểu đồ benchmark: {e}")
