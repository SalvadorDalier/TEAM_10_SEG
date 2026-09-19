"""
Module inverted_index.py
Cài đặt chỉ mục đảo (Inverted Index) thông qua lượng tử hóa (quantization) các chiều vector.
Lưu ý: Đây là phương pháp Baseline đơn giản, không tối ưu cho vector dày đặc (dense vectors),
kỳ vọng Recall sẽ thấp nhất trong 3 phương pháp.
"""

from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

# Số lượng bin mặc định để chia dải giá trị của mỗi chiều
DEFAULT_N_BINS: int = 10
# Giới hạn giá trị của vector đã chuẩn hóa L2 [-1.0, 1.0]
VAL_MIN: float = -1.0
VAL_MAX: float = 1.0


class InvertedIndex:
    """Chỉ mục đảo lượng tử hóa theo từng chiều vector."""

    def __init__(self, n_bins: int = DEFAULT_N_BINS) -> None:
        """Khởi tạo InvertedIndex.

        Args:
            n_bins: Số lượng bin chia đều dải giá trị mỗi chiều.
        """
        self.n_bins: int = n_bins
        # Cấu trúc: dict[(dim_idx, bin_id)] -> list[doc_id]
        self.index: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.num_docs: int = 0

    def _quantize_val(self, val: float) -> int:
        """Lượng tử hóa một giá trị thực về bin index từ 0 đến n_bins - 1."""
        clamped_val = max(VAL_MIN, min(VAL_MAX, val))
        norm_val = (clamped_val - VAL_MIN) / (VAL_MAX - VAL_MIN)
        bin_id = int(norm_val * self.n_bins)
        return min(bin_id, self.n_bins - 1)

    def build(self, vectors: np.ndarray) -> None:
        """Xây dựng chỉ mục đảo từ tập vector tài liệu.

        Args:
            vectors: Ma trận vector tài liệu (N, D).
        """
        try:
            self.index.clear()
            self.num_docs = vectors.shape[0]
            num_dims = vectors.shape[1]

            for doc_id in range(self.num_docs):
                for dim_idx in range(num_dims):
                    bin_id = self._quantize_val(float(vectors[doc_id, dim_idx]))
                    self.index[(dim_idx, bin_id)].append(doc_id)

            print(f"[DONE] Xây dựng Inverted Index thành công cho {self.num_docs} documents.")
        except Exception as e:
            print(f"[ERROR] Lỗi khi xây dựng Inverted Index: {e}")
            raise e

    def search(self, query: np.ndarray, k: int = 3) -> List[int]:
        """Tìm kiếm top-k tài liệu gần nhất dựa trên số lượng bin trùng khớp (Intersection count).

        Args:
            query: Vector câu truy vấn (D,).
            k: Số lượng kết quả cần lấy.

        Returns:
            List[int]: Danh sách top-k doc_id có điểm tương đồng cao nhất.
        """
        try:
            match_counts: Dict[int, int] = defaultdict(int)
            num_dims = len(query)

            for dim_idx in range(num_dims):
                bin_id = self._quantize_val(float(query[dim_idx]))
                matched_doc_ids = self.index.get((dim_idx, bin_id), [])
                for doc_id in matched_doc_ids:
                    match_counts[doc_id] += 1

            # Sắp xếp các doc_id theo số lượng bin trùng khớp giảm dần
            sorted_docs = sorted(
                range(self.num_docs),
                key=lambda doc_id: match_counts.get(doc_id, 0),
                reverse=True,
            )
            return sorted_docs[:k]
        except Exception as e:
            print(f"[ERROR] Lỗi khi tra cứu Inverted Index: {e}")
            return list(range(min(k, self.num_docs)))
