"""
Module data.py
Chứa dữ liệu văn bản và các câu truy vấn mẫu (40 documents đa dạng ngữ nghĩa).
"""

from typing import List

# Danh sách 40 văn bản làm dữ liệu phong phú cho Vector Database
DOCS: List[str] = [
    # Nhóm 1: Giao thông đường bộ & Phương tiện cá nhân
    "Tôi mua xe hơi màu đỏ để đi làm hàng ngày",
    "Chiếc ô tô thể thao sơn màu đỏ rất đẹp mắt",
    "Xe đạp điện màu xanh lá phù hợp cho học sinh sinh viên",
    "Dịch vụ giao hàng bằng xe máy siêu tốc trong 30 phút",
    "Dịch vụ vận chuyển đồ đạc đường dài giá rẻ và an toàn",
    "Xe tải chuyên chở hàng hóa siêu trọng tải 10 tấn",
    "Xe máy tay ga tiết kiệm xăng cho người đi làm văn phòng",
    "Xe bán tải địa hình hai cầu vượt mọi cung đường khó",
    "Trạm sạc xe điện thông minh phân bố rộng khắp thành phố",
    "Dịch vụ taxi công nghệ đưa đón hành khách tận nơi giá hợp lý",

    # Nhóm 2: Hàng không, Đường sắt & Hàng hải
    "Máy bay thương mại vận chuyển hành khách và hàng hóa quốc tế",
    "Tàu hỏa hỏa xa chở container hàng hóa từ Bắc vào Nam",
    "Tàu thủy chở dầu thô và container siêu trọng tải xuyên đại dương",
    "Cảng hàng không quốc tế đón hàng triệu lượt khách mỗi năm",
    "Tàu cao tốc đường sắt hiện đại rút ngắn thời gian di chuyển",
    "Phà biển vận chuyển ô tô và hành khách qua các đảo lớn",
    "Đội bay vận tải chuyên dụng chuyển phát nhanh qua đường hàng không",
    "Ga tàu lửa trung tâm thành phố kết nối các tuyến đường sắt huyết mạch",

    # Nhóm 3: Giao thông công cộng & Đô thị
    "Xe buýt công cộng nội đô phục vụ người dân vào giờ cao điểm",
    "Tuyến tàu điện ngầm metro số 1 hiện đại mới khai trương",
    "Hệ thống cầu vượt và hầm chui giúp giảm ùn tắc giao thông đô thị",
    "Làn đường ưu tiên dành riêng cho xe buýt nhanh BRT trong thành phố",
    "Hệ thống đèn tín hiệu giao thông thông minh ứng dụng AI điều tiết",
    "Bãi đỗ xe thông minh tự động nhiều tầng tại trung tâm thương mại",

    # Nhóm 4: Kho vận, Chuỗi cung ứng & Logistics
    "Hệ thống kho bãi lưu trữ hàng hóa và logistics tự động hóa",
    "Dịch vụ giao hàng hỏa tốc trong ngày liên tỉnh chất lượng cao",
    "Ứng dụng drone máy bay không người lái giao bưu phẩm tận nhà",
    "Quản lý chuỗi cung ứng thông minh với mã vạch và cảm biến RFID",
    "Giải pháp giao hàng chặng cuối cho các sàn thương mại điện tử",
    "Kho lạnh bảo quản nông sản và thực phẩm tươi sống xuất khẩu",

    # Nhóm 5: Vector Database, AI & Tối ưu hóa Chỉ mục (Topic Đề tài)
    "Nghiên cứu của tác giả Mai Đăng Phước về tối ưu hóa Vector Database",
    "Cấu trúc Learned Index thay thế Inverted Index cho truy xuất ngữ nghĩa",
    "Thuật toán đồ thị HNSW hỗ trợ tìm kiếm láng giềng gần nhất ANN tốc độ cao",
    "Phân vùng không gian vector đa chiều bằng giải thuật KMeans clustering",
    "Mô hình nhúng ngữ nghĩa Sentence Transformers all-MiniLM trích xuất đặc trưng văn bản",
    "Đánh giá hiệu năng Vector DB dựa trên Recall@k, QPS và dung lượng bộ nhớ RAM",
    "Tác giả Mai Đăng Phước công bố giải pháp tăng tốc độ truy xuất vector quy mô lớn",
    "Hệ thống cơ sở dữ liệu vector thời gian thực phục vụ ứng dụng RAG và LLM",
    "Tối ưu hóa chỉ mục vector giảm hiện tượng lời nguyền số chiều (curse of dimensionality)",
    "Học máy phân lớp dự đoán cụm dữ liệu giúp tăng tốc truy vấn tương đồng cosine",
]

# Danh sách các câu truy vấn tương ứng để kiểm thử tìm kiếm
QUERIES: List[str] = [
    # Giao thông & Xe cộ
    "xe hơi màu đỏ",
    "ô tô đỏ đẹp",
    "xe đạp xanh sinh viên",
    "giao hàng xe máy nhanh",
    "vận chuyển đường dài giá rẻ",
    "xe tải hàng nặng",
    "xe máy tiết kiệm xăng",
    "trạm sạc xe điện",
    "xe buýt giờ cao điểm",
    "tàu điện metro mới",

    # Hàng không & Hàng hải & Logistics
    "máy bay chở hàng quốc tế",
    "tàu hỏa chở container",
    "tàu thủy chở hàng xuyên đại dương",
    "tàu cao tốc Bắc Nam",
    "kho bãi logistics tự động",
    "giao hàng hỏa tốc trong ngày",
    "drone giao hàng bưu phẩm",
    "quản lý chuỗi cung ứng",

    # Vector DB, AI & Mai Đăng Phước
    "Mai Đăng Phước",
    "tác giả Mai Đăng Phước vector database",
    "tối ưu hóa Vector Database",
    "cấu trúc Learned Index",
    "thuật toán HNSW",
    "truy xuất ngữ nghĩa tương đồng",
    "đánh giá Recall và QPS",
    "mô hình Sentence Transformers",
    "hệ thống RAG cơ sở dữ liệu vector",
]

