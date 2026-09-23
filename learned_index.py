"""
Module learned_index.py
Cài đặt Learned Index: Sử dụng Machine Learning / MLP để phân vùng dữ liệu thành các cụm (clusters),
hỗ trợ định hướng và tăng tốc quá trình tìm kiếm vector.
"""

from typing import Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

# Thiết lập Seed cho PyTorch để kết quả có tính tái lập (Reproducibility)
torch.manual_seed(42)

# Hằng số cấu hình mô hình
INPUT_DIM: int = 384
HIDDEN_DIM_1: int = 64
HIDDEN_DIM_2: int = 32
NUM_CLUSTERS: int = 4
DROPOUT_RATE: float = 0.1
DEFAULT_EPOCHS: int = 100
DEFAULT_LR: float = 1e-3
LOG_EVERY_EPOCHS: int = 20
AUGMENT_NOISE_STD: float = 0.02

# ==============================================================================
# [KHU VỰC IMPORT FILE MODEL TỰ TRAIN CỦA BẠN]
# Bạn có thể import class model hoặc hàm load model của bạn tại đây:
# Ví dụ:
# from your_custom_model_module import YourCustomModel
# def load_user_model(path: str) -> Any: ...
# ==============================================================================


class LearnedIndex(nn.Module):
    """Mạng nơ-ron MLP dự đoán phân vùng (Cluster ID) của vector truy vấn."""

    def __init__(self) -> None:
        super(LearnedIndex, self).__init__()
        # Kiến trúc: Linear(384, 64) -> ReLU -> Dropout(0.1) -> Linear(64, 32) -> ReLU -> Linear(32, 4)
        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_2, NUM_CLUSTERS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def train_model(
        self,
        queries: np.ndarray,
        labels: np.ndarray,
        epochs: int = DEFAULT_EPOCHS,
        lr: float = DEFAULT_LR,
    ) -> None:
        """Huấn luyện mô hình MLP bằng thuật toán Adam và hàm mất mát CrossEntropyLoss.

        Args:
            queries: Ma trận các vector truy vấn (N, 384).
            labels: Mảng nhãn cụm mục tiêu (N,).
            epochs: Số epoch huấn luyện.
            lr: Tốc độ học (learning rate).
        """
        try:
            self.train()
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            x_tensor = torch.tensor(queries, dtype=torch.float32)
            y_tensor = torch.tensor(labels, dtype=torch.long)

            print("[INFO] Bắt đầu huấn luyện mô hình Learned Index (MLP)...")
            for epoch in range(1, epochs + 1):
                optimizer.zero_grad()
                outputs = self.forward(x_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()

                if epoch % LOG_EVERY_EPOCHS == 0 or epoch == epochs:
                    print(f"[INFO] Epoch [{epoch}/{epochs}] - Loss: {loss.item():.4f}")

            print("[DONE] Huấn luyện Learned Index hoàn tất.")
        except Exception as e:
            print(f"[ERROR] Lỗi trong quá trình huấn luyện Learned Index: {e}")
            raise e

    def predict(self, query: np.ndarray) -> int:
        """Dự đoán Cluster ID cho một vector truy vấn.

        Args:
            query: Vector truy vấn (384,).

        Returns:
            int: Cluster ID được dự đoán (từ 0 đến NUM_CLUSTERS - 1).
        """
        try:
            self.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(query.reshape(1, -1), dtype=torch.float32)
                logits = self.forward(x_tensor)
                predicted_cluster = int(torch.argmax(logits, dim=1).item())
                return predicted_cluster
        except Exception as e:
            print(f"[ERROR] Lỗi khi dự đoán cụm: {e}")
            return 0


class LearnedIndexTrainer:
    """Bộ công cụ chuẩn bị dữ liệu huấn luyện và chạy K-Means trên vector tài liệu."""

    def __init__(self, doc_vectors: np.ndarray, n_clusters: int = NUM_CLUSTERS) -> None:
        """Phân cụm K-Means trên tập vector tài liệu."""
        self.n_clusters: int = n_clusters
        self.kmeans: KMeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.doc_labels: np.ndarray = self.kmeans.fit_predict(doc_vectors)
        print(f"[DONE] K-Means đã gom {len(doc_vectors)} tài liệu vào {self.n_clusters} cụm.")

    def create_training_data(
        self, queries: np.ndarray, doc_vectors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tạo tập dữ liệu (X, y): Với mỗi query, gán nhãn là cluster ID của doc gần nhất."""
        augmented_queries = self.augment_queries(queries, n_augment=4)
        labels: List[int] = []

        for q in augmented_queries:
            # Tính Cosine similarity: do vector đã chuẩn hóa L2, tích vô hướng chính là cosine sim
            similarities = np.dot(doc_vectors, q)
            nearest_doc_idx = int(np.argmax(similarities))
            cluster_id = int(self.doc_labels[nearest_doc_idx])
            labels.append(cluster_id)

        return augmented_queries, np.array(labels, dtype=np.int64)

    def augment_queries(self, queries: np.ndarray, n_augment: int = 4) -> np.ndarray:
        """Tạo thêm dữ liệu mẫu bằng cách thêm nhiễu Gaussian nhỏ để đạt ~50 training samples."""
        augmented_list = [queries]
        for _ in range(n_augment):
            noise = np.random.normal(0, AUGMENT_NOISE_STD, queries.shape)
            noisy_queries = queries + noise
            # Chuẩn hóa L2 sau khi thêm nhiễu
            norms = np.linalg.norm(noisy_queries, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            noisy_queries = noisy_queries / norms
            augmented_list.append(noisy_queries)

        all_queries = np.vstack(augmented_list)
        return all_queries.astype(np.float32)
