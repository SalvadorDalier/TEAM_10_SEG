"""
Module data.py
Chứa dữ liệu văn bản và các câu truy vấn mẫu (10 documents cố định).
"""

from typing import List

# Danh sách 10 văn bản cố định làm dữ liệu cho Vector Database
DOCS: List[str] = [
    "Tôi mua xe hơi màu đỏ để đi làm",
    "Chiếc ô tô sơn đỏ rất đẹp",
    "Xe đạp màu xanh phù hợp cho sinh viên",
    "Giao hàng bằng xe máy trong 30 phút",
    "Dịch vụ vận chuyển đường dài giá rẻ",
    "Xe tải chở hàng nặng 10 tấn",
    "Máy bay vận chuyển hàng quốc tế",
    "Tàu hỏa chở container từ Bắc vào Nam",
    "Xe buýt công cộng giờ cao điểm",
    "Tàu điện ngầm metro mới khai trương",
]

# Danh sách 10 câu truy vấn tương ứng để kiểm thử tìm kiếm
QUERIES: List[str] = [
    "xe hơi màu đỏ",
    "ô tô đỏ",
    "xe đạp xanh",
    "giao hàng nhanh",
    "vận chuyển giá rẻ",
    "xe tải hàng nặng",
    "máy bay quốc tế",
    "tàu hỏa container",
    "xe buýt giờ cao điểm",
    "tàu điện metro",
]
