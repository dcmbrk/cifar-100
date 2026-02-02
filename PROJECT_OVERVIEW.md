# Project Overview: Image Summarization on CIFAR-100

## 1. Giới thiệu

### 1.1 Bài toán
**Image Summarization** (Tóm tắt tập ảnh) là bài toán chọn một tập con nhỏ k ảnh đại diện từ tập dữ liệu lớn n ảnh, sao cho tập con này "bao phủ" tốt nhất toàn bộ dataset gốc.

```
Input:  V = {v₁, v₂, ..., vₙ}  (n = 50,000 ảnh CIFAR-100)
Output: S ⊆ V với |S| = k      (k = 1,000 ảnh đại diện)
```

### 1.2 Ứng dụng
- **Giảm chi phí gán nhãn**: Chỉ cần gán nhãn 2% data thay vì 100%
- **Tăng tốc training**: Train model nhanh hơn 50x với coreset
- **Tiết kiệm lưu trữ**: Lưu trữ và truyền tải ít dữ liệu hơn
- **Active Learning**: Chọn samples quan trọng nhất để học

### 1.3 Dataset
**CIFAR-100** là bộ dữ liệu benchmark phổ biến:
- 50,000 ảnh training, 10,000 ảnh test
- Kích thước: 32×32 pixels, 3 kênh RGB
- 100 fine-grained classes thuộc 20 superclasses
- Mỗi class có 500 ảnh train, 100 ảnh test

---

## 2. Phương pháp

### 2.1 Pipeline tổng quan

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CIFAR-100     │───►│   ResNet-18     │───►│   Similarity    │───►│  Greedy/Lazy    │
│    Images       │    │   Features      │    │     Matrix      │    │   Selection     │
│   (50,000)      │    │    (512-D)      │    │  (50k × 50k)    │    │    (k ảnh)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2.2 Trích xuất đặc trưng
- Sử dụng **ResNet-18 pre-trained** trên ImageNet
- Bỏ layer FC cuối → vector **512 chiều**
- Chuẩn hóa L2 để tính cosine similarity

### 2.3 Hàm mục tiêu - Facility Location

```
f(S) = Σ   max  sim(v, s)
      v∈V  s∈S
```

**Ý nghĩa**: Mỗi ảnh v trong dataset được "đại diện" bởi ảnh gần nhất s trong tập đã chọn S. Tối đa hóa f(S) = chọn S sao cho mọi ảnh đều có đại diện tốt.

**Tính chất quan trọng**:
- **Submodular**: Diminishing returns - thêm phần tử mới mang lại ít lợi ích hơn
- **Monotone**: f(A) ≤ f(B) khi A ⊆ B
- **Non-negative**: f(S) ≥ 0

### 2.4 Thuật toán Greedy

```python
def greedy_facility_location(V, k):
    S = ∅
    for i in 1 to k:
        v* = argmax_{v ∈ V\S} [f(S ∪ {v}) - f(S)]  # Marginal gain
        S = S ∪ {v*}
    return S
```

**Đảm bảo lý thuyết**: Thuật toán Greedy cho hàm submodular monotone đạt tỷ lệ xấp xỉ **(1 - 1/e) ≈ 63%** so với giải pháp tối ưu (Nemhauser et al., 1978).

### 2.5 Lazy Greedy Optimization

Do tính diminishing returns, marginal gain chỉ giảm khi S mở rộng:
- Dùng **priority queue** lưu upper bound của gain
- Chỉ tính lại gain khi cần thiết
- Giảm số evaluations từ **O(nk)** xuống **~O(n log n)**

---

## 3. Các thuật toán được cài đặt

| # | Thuật toán | Độ phức tạp | Mô tả |
|---|------------|-------------|-------|
| 1 | **Random** | O(k) | Chọn ngẫu nhiên k ảnh |
| 2 | **Stratified Random** | O(n) | Chọn đều k/C ảnh mỗi class |
| 3 | **K-Center Greedy** | O(nk) | Chọn điểm xa nhất với tập đã chọn |
| 4 | **Herding** | O(nk) | Chọn điểm gần centroid class nhất |
| 5 | **Greedy Facility Location** | O(nk) → O(n log n) | Tối ưu hàm submodular với Lazy Eval |

---

## 4. Cấu trúc project

```
cifar-100/
├── src/
│   ├── feature_extractor.py   # Trích xuất features bằng ResNet-18
│   ├── similarity.py          # Tính cosine similarity matrix
│   ├── greedy_facility_location.py  # Greedy + Lazy Greedy
│   ├── baselines.py           # Random, Stratified, K-Center, Herding
│   ├── trainer.py             # Train classifier đánh giá coreset
│   ├── evaluation.py          # Các metrics đánh giá
│   └── visualization.py       # Vẽ biểu đồ kết quả
│
├── run_experiment.py          # Script chạy thực nghiệm chính
│
├── data/                      # CIFAR-100 dataset + cache
│   ├── cifar-100-python/      # Raw dataset
│   ├── cifar100_features.npz  # Cached features (99MB)
│   └── cifar100_similarity.dat # Cached similarity matrix (9.4GB)
│
├── results/                   # Kết quả thực nghiệm
│   └── figures/               # Biểu đồ
│
├── checkpoints/               # Model checkpoints
└── venv/                      # Virtual environment
```

---

## 5. Hướng dẫn sử dụng

### 5.1 Cài đặt

```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate

# Cài đặt dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scikit-learn matplotlib tqdm
```

### 5.2 Chạy thực nghiệm

```bash
# Activate venv
source venv/bin/activate

# Chạy nhanh (chỉ selection, không train classifier)
python run_experiment.py --budget 1000

# Chạy đầy đủ với training
python run_experiment.py --budget 1000 --train --epochs 50

# Phân tích nhiều budget khác nhau
python run_experiment.py --mode budget --train
```

### 5.3 Tham số

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--budget` | 1000 | Số ảnh cần chọn (k) |
| `--mode` | full | `full`, `budget`, hoặc `select_only` |
| `--train` | False | Train classifier trên coreset |
| `--epochs` | 50 | Số epochs training |
| `--device` | auto | `cuda` hoặc `cpu` |
| `--seed` | 42 | Random seed |

---

## 6. Metrics đánh giá

| Metric | Ý nghĩa |
|--------|---------|
| **FL Score** | Facility Location score - đo mức độ coverage |
| **Class Coverage** | Tỷ lệ classes có ít nhất 1 ảnh trong S |
| **Diversity** | Trung bình khoảng cách giữa các ảnh trong S |
| **Redundancy** | Mức độ trùng lặp trong S (thấp = tốt) |
| **Downstream Accuracy** | Accuracy khi train classifier chỉ trên S |
| **Runtime** | Thời gian chạy thuật toán |

---

## 7. Kết quả kỳ vọng (k=1000, 2% data)

| Phương pháp | FL Score | Coverage | Accuracy | Time |
|-------------|----------|----------|----------|------|
| Random | ~0.72 | ~97% | ~38% | <1s |
| Stratified | ~0.74 | 100% | ~41% | <1s |
| K-Center | ~0.76 | 100% | ~44% | ~45s |
| Herding | ~0.74 | 100% | ~42% | ~12s |
| **Greedy FL** | **~0.81** | **100%** | **~46%** | ~30s |

---

## 8. Tài liệu tham khảo

1. Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). *An analysis of approximations for maximizing submodular set functions*. Mathematical Programming.

2. Wei, K., Iyer, R., & Bilmes, J. (2015). *Submodularity in data subset selection and active learning*. ICML.

3. Mirzasoleiman, B., et al. (2015). *Lazier than lazy greedy*. AAAI.

4. Sener, O., & Savarese, S. (2018). *Active learning for CNNs: A core-set approach*. ICLR.

---

## 9. Ghi chú

- **Lần chạy đầu tiên** sẽ mất ~10 phút để:
  - Download CIFAR-100 (~170MB)
  - Trích xuất features (~1 phút)
  - Tính similarity matrix (~2 phút)

- **Các lần sau** sẽ nhanh hơn nhiều (~2-3 phút) vì đã cache features và similarity matrix.

- **Yêu cầu bộ nhớ**:
  - RAM: ~4GB (dùng memmap cho similarity matrix)
  - Disk: ~10GB (similarity matrix cache)
  - GPU: Optional nhưng recommended cho feature extraction
