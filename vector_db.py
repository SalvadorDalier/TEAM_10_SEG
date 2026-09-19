"""
Module vector_db.py
Cung cấp giao diện đồng nhất (Unified Interface) VectorDB cho cả 3 phương pháp:
1. Inverted Index
2. HNSW
3. Learned Index (MLP + HNSW)
"""

from typing import List, Optional
import numpy as np

from inverted_index import InvertedIndex
from hnsw_index import HNSWIndex
from learned_index import LearnedIndex

METHOD_INVERTED: str = "inverted"
METHOD_HNSW: str = "hnsw"
METHOD_LEARNED: str = "learned"


class VectorDB:
    """Lớp quản lý Vector Database hỗ trợ nhiều thuật toán chỉ mục."""

    def __init__(
        self,
        method: str = METHOD_HNSW,
        learned_model: Optional[LearnedIndex] = None,
    ) -> None:
        """Khởi tạo VectorDB.

        Args:
            method: Phương pháp chỉ mục ('inverted', 'hnsw', 'learned').
            learned_model: Mô hình MLP đã huấn luyện (chỉ dùng khi method='learned').
        """
        valid_methods = {METHOD_INVERTED, METHOD_HNSW, METHOD_LEARNED}
        if method not in valid_methods:
            raise ValueError(f"Method '{method}' không hợp lệ. Chọn một trong {valid_methods}.")

        self.method: str = method
        self.learned_model: Optional[LearnedIndex] = learned_model

        # Khởi tạo các index tương ứng
        self.inverted_index: Optional[InvertedIndex] = None
        self.hnsw_index: Optional[HNSWIndex] = None

    def build(self, vectors: np.ndarray) -> None:
        """Xây dựng chỉ mục tương ứng với phương pháp đã chọn."""
        try:
            if self.method == METHOD_INVERTED:
                self.inverted_index = InvertedIndex()
                self.inverted_index.build(vectors)
            elif self.method in (METHOD_HNSW, METHOD_LEARNED):
                self.hnsw_index = HNSWIndex()
                self.hnsw_index.build(vectors)
                # Lưu ý: Khi method == 'learned', ta kết hợp LearnedIndex để phân loại cụm
                # và HNSWIndex để tra cứu vector.
        except Exception as e:
            print(f"[ERROR] Lỗi khi build index cho method '{self.method}': {e}")
            raise e

    def search(self, query: np.ndarray, k: int = 3) -> List[int]:
        """Tìm kiếm top-k tài liệu gần nhất.

        Args:
            query: Vector truy vấn (D,).
            k: Số lượng tài liệu cần lấy.

        Returns:
            List[int]: Danh sách top-k doc_id.
        """
        try:
            if self.method == METHOD_INVERTED:
                if self.inverted_index is None:
                    raise RuntimeError("Inverted Index chưa được build.")
                return self.inverted_index.search(query, k=k)

            elif self.method == METHOD_HNSW:
                if self.hnsw_index is None:
                    raise RuntimeError("HNSW Index chưa được build.")
                results = self.hnsw_index.search(query, k=k)
                return [doc_id for doc_id, _ in results]

            elif self.method == METHOD_LEARNED:
                # ----------------------------------------------------------------------
                # HẠN CHẾ & GIẢI THÍCH:
                # Thư viện hnswlib hiện tại KHÔNG hỗ trợ truyền điểm bắt đầu tùy chỉnh
                # (custom entry point). Do đó, Learned Index tại đây thực hiện:
                # 1. Dùng mô hình MLP dự đoán Cluster ID của truy vấn (đo lường độ trễ suy luận).
                # 2. Sau đó tra cứu qua HNSW để lấy top-k kết quả chính xác nhất.
                # ----------------------------------------------------------------------
                if self.learned_model is not None:
                    _ = self.learned_model.predict(query)

                if self.hnsw_index is None:
                    raise RuntimeError("HNSW Index bên dưới Learned Index chưa được build.")
                results = self.hnsw_index.search(query, k=k)
                return [doc_id for doc_id, _ in results]

            return []
        except Exception as e:
            print(f"[ERROR] Lỗi khi thực hiện tìm kiếm với method '{self.method}': {e}")
            return []
