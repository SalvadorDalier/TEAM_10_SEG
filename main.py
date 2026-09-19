"""
Module main.py
Pipeline điều phối chính của dự án:
1. Đặt seed để tái lập kết quả
2. Tạo vector nhúng cho 10 documents & 10 queries
3. Xây dựng 3 chỉ mục: Inverted Index, HNSW, Learned Index
4. Chạy Benchmark so sánh Recall@3, QPS, Memory
5. Lưu kết quả ra file CSV và vẽ đồ thị
"""

import os
import sys
import numpy as np
import torch

# Đảm bảo in tiếng Việt trên Windows console không bị lỗi cp1252 UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from data import DOCS, QUERIES
from embedding import Embedder
from learned_index import LearnedIndex, LearnedIndexTrainer
from vector_db import VectorDB, METHOD_INVERTED, METHOD_HNSW, METHOD_LEARNED
from benchmark import benchmark_all, plot_results

# Đường dẫn lưu trữ kết quả
RESULTS_DIR: str = "results"
PLOTS_DIR: str = os.path.join(RESULTS_DIR, "plots")
METRICS_CSV_PATH: str = os.path.join(RESULTS_DIR, "metrics.csv")


def set_seed(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên cho Numpy và PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    print("=" * 70)
    print("  KHỞI ĐỘNG VECTOR DATABASE PROOF-OF-CONCEPT (10 DOCUMENTS)")
    print("=" * 70)

    # 1. Cố định Seed
    set_seed(42)

    # 2. Khởi tạo Embedder và mã hóa dữ liệu
    embedder = Embedder()
    print(f"\n[INFO] Đang tạo vector nhúng cho {len(DOCS)} tài liệu...")
    doc_vectors = embedder.encode(DOCS)
    print(f"[DONE] Kích thước ma trận doc_vectors: {doc_vectors.shape}")

    print(f"\n[INFO] Đang tạo vector nhúng cho {len(QUERIES)} câu truy vấn...")
    query_vectors = embedder.encode(QUERIES)
    print(f"[DONE] Kích thước ma trận query_vectors: {query_vectors.shape}")

    # 3. Xây dựng VectorDB: Inverted Index
    print("\n" + "-" * 50)
    print("1. KHỞI TẠO VÀ XÂY DỰNG INVERTED INDEX")
    print("-" * 50)
    db_inverted = VectorDB(method=METHOD_INVERTED)
    db_inverted.build(doc_vectors)

    # 4. Xây dựng VectorDB: HNSW
    print("\n" + "-" * 50)
    print("2. KHỞI TẠO VÀ XÂY DỰNG HNSW INDEX")
    print("-" * 50)
    db_hnsw = VectorDB(method=METHOD_HNSW)
    db_hnsw.build(doc_vectors)

    # 5. Huấn luyện & Xây dựng VectorDB: Learned Index (MLP + HNSW)
    print("\n" + "-" * 50)
    print("3. HUẤN LUYỆN VÀ XÂY DỰNG LEARNED INDEX")
    print("-" * 50)
    trainer = LearnedIndexTrainer(doc_vectors=doc_vectors, n_clusters=4)
    X_train, y_train = trainer.create_training_data(queries=query_vectors, doc_vectors=doc_vectors)

    learned_model = LearnedIndex()
    learned_model.train_model(X_train, y_train, epochs=100, lr=1e-3)

    db_learned = VectorDB(method=METHOD_LEARNED, learned_model=learned_model)
    db_learned.build(doc_vectors)

    # 6. Chạy Benchmark so sánh cả 3 phương pháp
    print("\n" + "=" * 70)
    print("  BẮT ĐẦU CHẠY BENCHMARK TOÀN DIỆN")
    print("=" * 70)
    dbs = {
        METHOD_INVERTED: db_inverted,
        METHOD_HNSW: db_hnsw,
        METHOD_LEARNED: db_learned,
    }

    metrics_df = benchmark_all(
        dbs=dbs,
        doc_vectors=doc_vectors,
        query_vectors=query_vectors,
        k=3,
        n_repeat=1000,
    )

    # 7. Lưu kết quả bảng số liệu ra CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    print(f"\n[DONE] Đã lưu bảng số liệu tại: {METRICS_CSV_PATH}")

    # 8. Vẽ đồ thị so sánh
    plot_results(metrics_df, PLOTS_DIR)

    # 9. In bảng tóm tắt kết quả ra màn hình console
    print("\n" + "=" * 70)
    print("  BẢNG TỔNG KẾT KẾT QUẢ BENCHMARK")
    print("=" * 70)
    print(metrics_df.to_string(index=False))
    print("=" * 70)
    print("[DONE] Toàn bộ pipeline đã hoàn thành xuất sắc!")


if __name__ == "__main__":
    main()
