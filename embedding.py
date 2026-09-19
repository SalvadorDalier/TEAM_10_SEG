"""
Module embedding.py
Chịu trách nhiệm chuyển đổi văn bản thành vector nhúng (vector embeddings)
sử dụng mô hình Sentence-Transformers chạy trên CPU.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# Các hằng số cấu hình mô hình
MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384
DEVICE_CPU: str = "cpu"


class Embedder:
    """Lớp tạo vector nhúng (Embedder) với cơ chế Singleton Cache để tránh load lại model nhiều lần."""

    _instance: Optional["Embedder"] = None
    _model: Optional[SentenceTransformer] = None

    def __new__(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = super(Embedder, cls).__new__(cls)
            try:
                print(f"[INFO] Đang tải mô hình embedding: '{MODEL_NAME}' trên {DEVICE_CPU}...")
                cls._model = SentenceTransformer(MODEL_NAME, device=DEVICE_CPU)
                print(f"[DONE] Tải mô hình '{MODEL_NAME}' thành công (Số chiều: {EMBEDDING_DIM}).")
            except Exception as e:
                print(f"[ERROR] Không thể tải mô hình embedding: {e}")
                raise e
        return cls._instance

    def encode(self, texts: List[str]) -> np.ndarray:
        """Chuyển đổi danh sách văn bản thành ma trận vector (n, 384) và chuẩn hóa L2.

        Args:
            texts: Danh sách các chuỗi văn bản cần mã hóa.

        Returns:
            np.ndarray: Ma trận numpy kích thước (n, 384) kiểu float32 đã chuẩn hóa L2.
        """
        if self._model is None:
            raise RuntimeError("Mô hình SentenceTransformer chưa được khởi tạo.")

        try:
            # normalize_embeddings=True đảm bảo độ dài vector bằng 1 (L2 normalization)
            vectors: np.ndarray = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=DEVICE_CPU,
            )
            return vectors.astype(np.float32)
        except Exception as e:
            print(f"[ERROR] Lỗi trong quá trình encode văn bản: {e}")
            raise e
