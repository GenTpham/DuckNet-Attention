# DuckNet-Attention

Triển khai kiến trúc DuckNet tích hợp cơ chế attention bằng PyTorch cho phân đoạn hình ảnh y tế, được thiết kế đặc biệt cho phân đoạn polyp đường tiêu hóa.

## 📋 Tổng quan

Dự án này triển khai DuckNet-Attention, một phiên bản cải tiến của kiến trúc DuckNet có tích hợp cơ chế attention để tăng cường biểu diễn đặc trưng và cải thiện hiệu suất phân đoạn. Mô hình được huấn luyện và đánh giá trên bộ dữ liệu Kvasir-SEG cho bài toán phân đoạn polyp trong hình ảnh nội soi đại tràng.

## 🏗️ Kiến trúc

### Đặc điểm của DuckNet-Attention:
- **Kiến trúc Encoder-Decoder**: Cấu trúc giống U-Net với skip connections
- **DuckBlocks**: Khối xây dựng cốt lõi sử dụng separable convolutions
- **Cơ chế Attention**: 
  - Self-attention ở lớp bottleneck
  - Attention gates ở skip connections của decoder
  - Channel và spatial attention trong DuckBlocks
- **Fusion đa tỷ lệ**: Kết nối ResPath để cải thiện gradient flow

### Các thành phần chính:
- **SeparableConv2d**: Convolution tách biệt theo chiều sâu để tối ưu hiệu suất
- **DuckBlock**: Khối residual cải tiến với attention tùy chọn
- **AttentionGate**: Cơ chế cổng cho skip connections
- **ResidualBlock**: Kết nối residual chuẩn ở bottleneck

## 📊 Hiệu suất

Mô hình đạt hiệu suất xuất sắc trên bộ dữ liệu Kvasir-SEG:

| Độ đo | DuckNet-Attention | DuckNet thường |
|-------|-------------------|----------------|
| **Dice Score** | 0.9422 ± 0.0329 | Baseline |
| **IoU Score** | 0.9289 ± 0.0580 | Baseline |
| **Độ chính xác** | 0.9864 ± 0.0147 | Baseline |
| **Precision** | 0.9825 ± 0.0075 | Baseline |
| **Recall** | 0.9444 ± 0.0577 | Baseline |

## 🗂️ Bộ dữ liệu

**Bộ dữ liệu Kvasir-SEG**: Tập hợp các hình ảnh đường tiêu hóa với mask phân đoạn polyp tương ứng, được sử dụng rộng rãi để đánh giá các mô hình phân đoạn hình ảnh y tế.

- **Nhiệm vụ**: Phân đoạn nhị phân (polyp vs nền)
- **Kích thước ảnh**: 256×256 pixels
- **Định dạng**: Ảnh RGB với mask nhị phân

## 🛠️ Cài đặt

```bash
# Clone repository
git clone https://github.com/yourusername/DuckNet-Attention.git
cd DuckNet-Attention

# Cài đặt các package cần thiết
pip install torch torchvision
pip install numpy pandas matplotlib
pip install opencv-python pillow
pip install scikit-learn
```

## 📖 Sử dụng

### Huấn luyện mô hình

Code huấn luyện chính được chứa trong Jupyter notebook `train-ducknet-att.ipynb`. Notebook bao gồm:

1. **Tiền xử lý và tăng cường dữ liệu**
2. **Triển khai kiến trúc mô hình**
3. **Vòng lặp huấn luyện với validation**
4. **Đánh giá hiệu suất và trực quan hóa**
5. **So sánh mô hình (có/không có attention)**

### Tính năng chính trong Notebook:

- **Load dữ liệu**: Class dataset tùy chỉnh cho Kvasir-SEG
- **Các biến thể mô hình**: Cả DuckNet thường và DuckNet-Attention
- **Cấu hình huấn luyện**: 
  - Optimizer: AdamW
  - Loss Function: Kết hợp Dice + BCE Loss
  - Learning Rate Scheduling
  - Early Stopping với model checkpointing

### Chạy huấn luyện:

```python
# Load notebook và chạy tất cả cells
jupyter notebook train-ducknet-att.ipynb
```

## 🧠 Chi tiết kiến trúc mô hình

### Cấu trúc DuckBlock:
```
Input → SeparableConv2d → BatchNorm → ReLU → 
        SeparableConv2d → BatchNorm → Attention (tùy chọn) → 
        Residual Connection → Output
```

### Cơ chế Attention:
- **Self-Attention**: Áp dụng ở lớp bottleneck cho ngữ cảnh toàn cục
- **Attention Gates**: Lọc các đặc trưng liên quan trong skip connections
- **Channel Attention**: Tập trung vào các kênh đặc trưng quan trọng
- **Spatial Attention**: Nhấn mạnh các vị trí không gian liên quan

## 📈 Trực quan hóa kết quả

Notebook bao gồm trực quan hóa toàn diện:
- Đường cong loss huấn luyện/validation
- Tiến trình Dice score
- Dự đoán mẫu so với ground truth
- Trực quan hóa attention maps
- Phân tích so sánh giữa các biến thể mô hình

## 🔬 Độ đo đánh giá

- **Hệ số Dice**: Đo lường độ chồng lấp giữa dự đoán và ground truth
- **IoU (Intersection over Union)**: Chỉ số Jaccard cho chất lượng phân đoạn
- **Độ chính xác Pixel**: Độ chính xác phân loại tổng thể theo pixel
- **Precision**: Tỷ lệ true positive
- **Recall**: Độ đo sensitivity

## 📝 Trích dẫn

Nếu bạn sử dụng công trình này trong nghiên cứu, vui lòng trích dẫn:

```bibtex
@article{ducknet_attention,
  title={DuckNet-Attention: Cải tiến phân đoạn hình ảnh y tế với cơ chế Attention},
  author={Tên của bạn},
  journal={Tạp chí của bạn},
  year={2024}
}
```

## 🤝 Đóng góp

Chúng tôi hoan nghênh các đóng góp! Vui lòng tạo Pull Request.

## 📄 Giấy phép

Dự án này được cấp phép theo MIT License - xem file LICENSE để biết chi tiết.

## 🙏 Lời cảm ơn

- Cảm ơn kiến trúc DuckNet gốc đã truyền cảm hứng
- Cảm ơn những người đóng góp cho bộ dữ liệu Kvasir-SEG
- Cảm ơn cộng đồng PyTorch vì tài liệu và công cụ tuyệt vời