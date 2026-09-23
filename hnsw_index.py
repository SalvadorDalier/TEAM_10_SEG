"""
Module hnsw_index.py
Cài đặt cấu trúc chỉ mục HNSW (Hierarchical Navigable Small World).
Hỗ trợ cả hnswlib và tự động fallback sang faiss-cpu (faiss.IndexHNSWFlat)
khi môi trường trên Windows không có sẵn Microsoft C++ Build Tools.
"""

from typing import List, Tuple
import numpy as np

# Thử import hnswlib, nếu thiếu C++ compiler trên Windows sẽ fallback sang faiss
try:
    import hnswlib
    BACKEND = "hnswlib"
except ImportError:
    try:
        import faiss
        BACKEND = "faiss"
    except ImportError:
        raise ImportError("Cần cài đặt ít nhất một trong hai thư viện: 'hnswlib' hoặc 'faiss-cpu'.")

# Tham số cấu hình HNSW
DEFAULT_DIM: int = 384
# M: Số lượng liên kết tối đa giữa các node (càng lớn recall càng cao, tốn RAM hơn)
DEFAULT_M: int = 16
# ef_construction: Kích thước danh sách ứng viên khi xây dựng đồ thị (càng lớn index càng chuẩn, build chậm hơn)
DEFAULT_EF_CONSTRUCTION: int = 200
# ef_search: Kích thước danh sách ứng viên khi truy vấn (càng lớn tìm càng chính xác, QPS giảm)
DEFAULT_EF_SEARCH: int = 50
SPACE_METRIC: str = "cosine"


class HNSWIndex:
    """Chỉ mục đồ thị HNSW hỗ trợ tìm kiếm láng giềng gần đúng (ANN)."""

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        M: int = DEFAULT_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
    ) -> None:
        """Khởi tạo cấu hình HNSW."""
        self.dim: int = dim
        self.M: int = M
        self.ef_construction: int = ef_construction
        self.ef_search: int = ef_search
        self.backend: str = BACKEND
        self.num_elements: int = 0
        self.index = None

    def build(self, vectors: np.ndarray) -> None:
        """Khởi tạo đồ thị HNSW và nạp các vector tài liệu vào index.

        Args:
            vectors: Ma trận vector tài liệu (N, dim).
        """
        try:
            self.num_elements = vectors.shape[0]

            if self.backend == "hnswlib":
                self.index = hnswlib.Index(space=SPACE_METRIC, dim=self.dim)
                self.index.init_index(
                    max_elements=self.num_elements,
                    ef_construction=self.ef_construction,
                    M=self.M,
                )
                ids = np.arange(self.num_elements)
                self.index.add_items(vectors, ids)
                self.index.set_ef(self.ef_search)
            else:  # backend == "faiss"
                # Với vector đã chuẩn hóa L2, khoảng cách Cosine tương đương Inner Product (METRIC_INNER_PRODUCT)
                self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_INNER_PRODUCT)
                self.index.hnsw.efConstruction = self.ef_construction
                self.index.hnsw.efSearch = self.ef_search
                self.index.add(vectors.astype(np.float32))

            print(f"[DONE] Xây dựng HNSW Index thành công ({self.backend}) với {self.num_elements} phần tử.")
        except Exception as e:
            print(f"[ERROR] Lỗi khi xây dựng HNSW Index: {e}")
            raise e

    def search(self, query: np.ndarray, k: int = 3) -> List[Tuple[int, float]]:
        """Tìm kiếm k láng giềng gần nhất cho câu truy vấn.

        Args:
            query: Vector truy vấn (dim,).
            k: Số lượng kết quả cần tìm.

        Returns:
            List[Tuple[int, float]]: Danh sách cặp (doc_id, distance).
        """
        try:
            query_reshaped = query.reshape(1, -1).astype(np.float32)

            if self.backend == "hnswlib":
                labels, distances = self.index.knn_query(query_reshaped, k=k)
                return [(int(lbl), float(dst)) for lbl, dst in zip(labels[0], distances[0]) if int(lbl) >= 0]
            else:  # backend == "faiss"
                distances, labels = self.index.search(query_reshaped, k)
                return [(int(lbl), float(dst)) for lbl, dst in zip(labels[0], distances[0]) if int(lbl) >= 0]
        except Exception as e:
            print(f"[ERROR] Lỗi khi truy vấn HNSW Index: {e}")
            return []

    def save(self, path: str) -> None:
        """Lưu chỉ mục HNSW ra tệp."""
        try:
            if self.backend == "hnswlib":
                self.index.save_index(path)
            else:
                faiss.write_index(self.index, path)
            print(f"[INFO] Đã lưu HNSW index ({self.backend}) tại: {path}")
        except Exception as e:
            print(f"[ERROR] Không thể lưu HNSW index: {e}")

    def load(self, path: str, max_elements: int = 10) -> None:
        """Tải chỉ mục HNSW từ tệp."""
        try:
            if self.backend == "hnswlib":
                self.index = hnswlib.Index(space=SPACE_METRIC, dim=self.dim)
                self.index.load_index(path, max_elements=max_elements)
                self.index.set_ef(self.ef_search)
            else:
                self.index = faiss.read_index(path)
                self.index.hnsw.efSearch = self.ef_search
            print(f"[INFO] Đã nạp HNSW index ({self.backend}) từ: {path}")
        except Exception as e:
            print(f"[ERROR] Không thể nạp HNSW index: {e}")
