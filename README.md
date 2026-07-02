# AI Credit Scoring kết hợp RPA trong quy trình xét duyệt tín dụng

> Đồ án môn **Công nghệ dịch vụ tài chính** — Nhóm 3
> Hệ thống chấm điểm tín dụng dựa trên Machine Learning kết hợp tự động hóa quy trình (RPA-style) qua n8n để ghi log Google Sheets, gửi email báo cáo kết quả cho khách hàng.

---

## 🚀 Quick Start — Chạy demo trong 3 phút

Nếu bạn chỉ muốn chạy thử app ngay:

```powershell
# 1. Clone repo
git clone https://github.com/huymonsterhuman-eng/AI-credit-score-RPA-_-nh-m-3.git
cd AI-credit-score-RPA-_-nh-m-3

# 2. Tạo môi trường Python + cài thư viện
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Chạy Streamlit
streamlit run app\app.py
```

Trình duyệt tự mở `http://localhost:8501`. Nhấn nút **🟢 Good** để load case demo → **Dự đoán** → xem kết quả.

> **Ghi chú**: App chạy được ngay chỉ với Streamlit. Phần n8n (Sheets + Gmail) là **tùy chọn** — chỉ 1 người trong nhóm cần setup để làm demo.

---

## 📋 Mục lục

1. [Mục tiêu](#1-mục-tiêu)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Công nghệ sử dụng](#3-công-nghệ-sử-dụng)
4. [Cấu trúc thư mục](#4-cấu-trúc-thư-mục)
5. [Bảng phân loại FICO](#5-bảng-phân-loại-fico)
6. [Setup chi tiết](#6-setup-chi-tiết)
7. [Setup n8n workflow](#7-setup-n8n-workflow-tùy-chọn)
8. [Hành trình phát triển mô hình](#8-hành-trình-phát-triển-mô-hình)
9. [Hạn chế & Hướng phát triển](#9-hạn-chế--hướng-phát-triển)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Mục tiêu

- Train mô hình ML dự đoán **credit score** từ thông tin cá nhân/tài chính khách hàng.
- Map điểm số sang thang FICO chuẩn 300–850 với 5 nhóm rating.
- Cung cấp giao diện web (Streamlit) cho phép nhập thông tin và nhận kết quả kèm giải thích (SHAP).
- Tự động hóa hậu xử lý qua n8n: lưu Google Sheets → gửi email báo cáo chi tiết.

## 2. Kiến trúc hệ thống

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Dataset CSV    │───▶│  Colab Notebook  │───▶│    model.pkl     │
│  (96k dòng)     │    │  Train LightGBM  │    │    features.json │
└─────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                        │
                                                        ▼
┌────────────────────────────────────────────────────────────────┐
│                    Streamlit App (localhost:8501)              │
│                                                                │
│   Form nhập liệu → Predict → FICO Score + Rating + Gợi ý       │
│   ├─ Preset buttons (Poor/Standard/Good/Exceptional)           │
│   ├─ SHAP explanation                                          │
│   └─ Contribution breakdown                                    │
└──────────────────────────────┬─────────────────────────────────┘
                               │  POST webhook
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                  n8n (Docker, localhost:5678)                  │
│                                                                │
│   Webhook → Sheets Append → IF (có email?)                     │
│                              ├─ True: Build HTML → Gmail Send  │
│                              └─ False: dừng                    │
└────────────────────────────────────────────────────────────────┘
```

## 3. Công nghệ sử dụng

| Thành phần | Công cụ |
|---|---|
| Ngôn ngữ | Python 3.10 / 3.11 / 3.12 |
| Train model | Google Colab, LightGBM, XGBoost, scikit-learn |
| Giải thích | SHAP (TreeExplainer) |
| Feature engineering | Temporal features (lag, delta, rolling, streak, trend) |
| Giao diện web | Streamlit, Plotly |
| Tự động hóa | n8n (self-hosted qua Docker) |
| Lưu trữ log | Google Sheets |
| Thông báo | Gmail (OAuth 2.0 qua n8n) |
| Version control | Git + GitHub |

## 4. Cấu trúc thư mục

```
.
├── README.md                           # Tài liệu này
├── WORKLOAD.md                         # Checklist tiến độ 10 phase
├── requirements.txt                    # Python dependencies
├── .gitignore
│
├── data/
│   ├── Credit_score_classification/    # Dataset chính (Kaggle)
│   │   ├── train.csv                   # 100k dòng
│   │   └── test.csv
│   └── synthetic_personal_finance_dataset.csv  # Dataset đầu tiên (đã loại)
│
├── notebooks/                          # 4 notebooks — hành trình train model
│   ├── 01_train_model.ipynb            # Baseline (dataset synthetic — R²≈0, fail)
│   ├── 02_train_credit_classification.ipynb  # Classification, F1 = 0.682
│   ├── 03_improve_model.ipynb          # Aggregate features (F1 = 0.612, fail)
│   └── 04_temporal_features.ipynb      # ✅ Model final, F1 = 0.687
│
├── models/                             # Artifact
│   ├── model.pkl                       # LightGBM pipeline (10MB)
│   └── features.json                   # Schema input + metrics
│
├── app/                                # Streamlit app
│   ├── app.py                          # Entry point
│   ├── config.py                       # Config: paths, webhook URL
│   └── utils.py                        # Load model, predict, FICO conversion, SHAP
│
├── n8n/                                # RPA workflow
│   ├── docker-compose.yml              # n8n self-hosted
│   ├── workflow.json                   # Workflow export
│   └── README.md                       # Setup credentials + workflow
│
└── docs/                               # Tài liệu, screenshots (chưa)
    └── images/
```

## 5. Bảng phân loại FICO

| Điểm | Rating | Ý nghĩa |
|---|---|---|
| < 580 | **Poor** | Rủi ro cao, hầu hết ngân hàng từ chối |
| 580–669 | **Fair** | Dưới trung bình, có thể vay với lãi cao |
| 670–739 | **Good** | Xấp xỉ trung bình, được xem là điểm tốt |
| 740–799 | **Very Good** | Trên trung bình, đáng tin cậy |
| ≥ 800 | **Exceptional** | Xuất sắc, top 5% khách hàng |

*Nguồn: [MyFICO — What is a good credit score?](https://www.myfico.com/credit-education/credit-scores)*

## 6. Setup chi tiết

### Yêu cầu

- **Python** 3.10 / 3.11 / 3.12 (khuyên 3.12)
- **Git** để clone repo
- **Docker Desktop** (chỉ nếu setup n8n)
- Trình duyệt hiện đại (Chrome/Edge/Firefox)

### Bước 1 — Clone repo

```powershell
git clone https://github.com/huymonsterhuman-eng/AI-credit-score-RPA-_-nh-m-3.git
cd AI-credit-score-RPA-_-nh-m-3
```

### Bước 2 — Tạo virtual environment

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

Prompt sẽ có `(venv)` ở đầu.

**Nếu PowerShell chặn script** (`running scripts is disabled`):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
Rồi Activate lại.

### Bước 3 — Cài dependencies

```powershell
pip install -r requirements.txt
```

Chờ ~2-3 phút (nhiều thư viện nặng như lightgbm, shap).

### Bước 4 — Chạy Streamlit

```powershell
streamlit run app\app.py
```

Trình duyệt tự mở `http://localhost:8501`. Nếu không mở tự, mở tay URL đó.

### Bước 5 — Test demo

Trong app:
1. Bấm 1 trong 4 nút preset (🔴 Poor / 🟠 Standard / 🟢 Good / 💚 Exceptional).
2. Cuộn xuống bấm **🔍 Dự đoán điểm tín dụng**.
3. Xem kết quả: FICO Score + Rating + Gợi ý cải thiện.
4. Mở expander **🔬 Chi tiết kỹ thuật** để xem cách tính điểm + SHAP.

## 7. Setup n8n workflow (tùy chọn)

Phần này để tự động hóa lưu Sheets + gửi email. **Chỉ cần 1 người trong nhóm setup** để demo.

Xem hướng dẫn chi tiết tại [n8n/README.md](n8n/README.md).

Tóm tắt:

```powershell
# 1. Khởi động n8n
cd n8n
docker compose up -d
```

2. Mở `http://localhost:5678` → tạo account owner.
3. Setup **Google Sheets OAuth** + **Gmail OAuth** (Google Cloud Console → tạo project → enable Sheets/Gmail API → OAuth Client).
4. **Workflows** → Import from File → chọn `n8n/workflow.json`.
5. Link 2 credentials vào node `Append to Sheet` và `Send Email`.
6. Sửa Sheet ID trong node `Append to Sheet` cho phù hợp Sheet của bạn.
7. **Publish** workflow.
8. Copy Production URL từ node Webhook → dán vào `app/config.py` biến `N8N_WEBHOOK_URL`.
9. Restart Streamlit → submit form → check Sheet + email.

## 8. Hành trình phát triển mô hình

Đây là câu chuyện thực tế của nhóm — 4 notebooks phản ánh 4 giai đoạn:

| # | Notebook | Approach | F1 Macro | Kết luận |
|---|---|---|---|---|
| 1 | Baseline | Regression trên dataset synthetic | R² ≈ 0 | ❌ Dataset không có tín hiệu |
| 2 | Classification | LightGBM 3 lớp trên `Credit_score_classification` | **0.682** | ✅ Baseline solid |
| 3 | Aggregate | Aggregate per Customer_ID | 0.612 | ❌ Mất 8× sample size |
| 4 | Temporal | Row-level + lag/delta/rolling features | **0.687** | ✅ Model final |

**Bài học**: aggregate features làm giảm performance trên dataset panel-data này vì trade sample-size lấy feature richness không đáng — cải thiện nhỏ hơn mất mát.

## 9. Hạn chế & Hướng phát triển

### Hạn chế

- Dataset `Credit_score_classification` là **synthetic của Kaggle**, không phản ánh phân phối rủi ro tín dụng thực tế → mô hình mang tính minh họa quy trình.
- n8n là **workflow automation** platform, không phải RPA "chuẩn" theo nghĩa UI automation (UiPath). Trong báo cáo được sử dụng như công cụ orchestration thay thế RPA.
- Chưa có xác thực người dùng / phân quyền — chỉ phục vụ mục đích demo.
- Model output là probability trên 3 lớp (Poor/Standard/Good), FICO Score được suy ra bằng **weighted anchor** — là design choice của nhóm chứ không phải công thức chuẩn công nghiệp.

### Hướng phát triển

- Mở rộng workflow n8n: thêm PDF report attachment, notify Telegram/Discord khi score < 580, sync database Postgres backup.
- Thêm model thứ 2 cho **Loan Approval** dùng `loan_data.csv` (đã có sẵn).
- Deploy Streamlit + n8n vào cùng 1 `docker-compose` để đóng gói, giúp chấm điểm demo dễ hơn.
- A/B test các FICO anchor khác nhau để calibrate score với distribution tham chiếu.

## 10. Troubleshooting

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| `ModuleNotFoundError` khi chạy Streamlit | Chưa activate venv | `.\venv\Scripts\Activate.ps1` |
| `Can't get attribute '_RemainderColsList'` | sklearn version mismatch | `pip install scikit-learn==1.6.1` |
| Streamlit hiện nhưng warning "chưa cấu hình webhook" | Chưa setup n8n | Bỏ qua nếu chỉ test model — hoặc setup n8n |
| n8n báo "Column names were updated" | Sheet header và mapping không khớp | Refresh schema trong node Sheets |
| Google Sheets ra `#ERROR!` | Value bị dính dấu `=` ở đầu | Đổi mode Fixed/Expression cho các field |
| Email không nhận được | Chưa activate workflow / OAuth scope thiếu | Check Executions tab trong n8n |
| PowerShell không chạy được `.\venv\Scripts\Activate.ps1` | Execution policy | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` |

---

## 👥 Nhóm 3

- Github: https://github.com/huymonsterhuman-eng/AI-credit-score-RPA-_-nh-m-3

## 📖 Tham khảo

- Siddiqi, N. (2005). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley.
- MyFICO. (2024). What's in your credit score? https://www.myfico.com/credit-education/whats-in-your-credit-score
- Kaggle Dataset: [Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)
- n8n Documentation: https://docs.n8n.io/
