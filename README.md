# Vector DB 10 Docs — Proof of Concept

Dự án thử nghiệm (Proof of Concept) xây dựng Vector Database thu nhỏ trên **10 tài liệu cố định**, so sánh 3 phương pháp chỉ mục vector:
1. **Inverted Index** (Lượng tử hóa giá trị các chiều làm baseline)
2. **HNSW** (Hierarchical Navigable Small World đồ thị ANN)
3. **Learned Index** (Mạng nơ-ron MLP kết hợp HNSW)

---

## 1. Yêu cầu hệ thống & Cài đặt

Môi trường khuyến nghị: Python 3.9 - 3.11 (Chạy hoàn toàn trên CPU, không cần GPU).

Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

> **Lưu ý:** Nếu gặp lỗi khi cài đặt `hnswlib` trên Windows (do thiếu C++ Build Tools), bạn có thể cài đặt Microsoft C++ Build Tools hoặc sử dụng phiên bản wheel tương thích.

---

## 2. Cách chạy chương trình

Chạy trực tiếp pipeline đầy đủ bằng lệnh:
```bash
python main.py
```

Sau khi chạy xong, kết quả sẽ được tạo tại:
- `results/metrics.csv`: Bảng tổng hợp các chỉ số Recall@3, QPS, Memory (MB).
- `results/plots/benchmark_comparison.png`: Biểu đồ so sánh trực quan.

---

## 3. Giải thích 3 phương pháp chỉ mục

| Phương pháp | Nguyên lý hoạt động | Ưu điểm | Nhược điểm |
|---|---|---|---|
| **Inverted Index** | Chia giá trị mỗi chiều thành các bins, tạo chỉ mục đảo ánh xạ `(dim, bin) -> [doc_id]`. | Nhẹ, cấu trúc đơn giản, không cần train. | Không phù hợp với dense vector nhiều chiều, Recall rất thấp. |
| **HNSW** | Xây dựng đồ thị phân tầng nhiều lớp, duyệt tìm láng giềng gần nhất theo hàm khoảng cách Cosine. | Recall cực cao (~1.0), tốc độ tìm kiếm nhanh (QPS cao). | Tiêu tốn bộ nhớ RAM hơn để lưu đồ thị. |
| **Learned Index** | Dùng mạng MLP học phân vùng dữ liệu thành các cụm (clusters), kết hợp HNSW. | Tiềm năng tối ưu hóa vùng tìm kiếm với dữ liệu lớn. | Có thêm độ trễ suy luận của mạng nơ-ron (QPS thấp hơn HNSW thuần). |

---

## 4. Tùy biến / Import Model tự train

Trong file `learned_index.py`, đã có sẵn khu vực đánh dấu:
```python
# ==============================================================================
# [KHU VỰC IMPORT FILE MODEL TỰ TRAIN CỦA BẠN]
# ==============================================================================
```
Bạn có thể tự do import class mô hình Machine Learning hoặc nạp trọng số (.pt / .pth / .pkl) đã huấn luyện trước của bạn vào vị trí này.

---

## 5. Kết quả kỳ vọng

Do tập dữ liệu proof-of-concept gồm 10 tài liệu cố định:
- **Recall@3**: HNSW và Learned Index đạt xấp xỉ 1.0 (100%), Inverted Index đạt mức thấp (~0.2 - 0.5).
- **QPS**: HNSW thuần có QPS cao nhất, Inverted Index ở mức trung bình, Learned Index có thêm overhead suy luận MLP nên QPS sẽ thấp hơn HNSW.
- **Memory**: Tương đương giữa các phương pháp do lượng dữ liệu 10 documents rất nhỏ (~100 - 200 MB gồm cả model sentence-transformers).
