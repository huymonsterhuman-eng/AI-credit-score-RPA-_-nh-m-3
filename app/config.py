"""Cấu hình cho Streamlit app."""

from pathlib import Path

APP_ROOT = Path(__file__).parent
PROJECT_ROOT = APP_ROOT.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'model.pkl'
FEATURES_PATH = PROJECT_ROOT / 'models' / 'features.json'

# n8n webhook — dán URL webhook sau khi setup n8n
N8N_WEBHOOK_URL = 'http://localhost:5678/webhook/credit-scoring'  # e.g. 'https://n8n.example.com/webhook/credit-scoring'
WEBHOOK_TIMEOUT = 8  # seconds

# FICO rating bins — phải khớp với features.json
FICO_BINS = [
    (0, 580, 'Poor', '#d9534f'),
    (580, 670, 'Fair', '#e67e22'),
    (670, 740, 'Good', '#7cb342'),
    (740, 800, 'Very Good', '#43a047'),
    (800, 850, 'Exceptional', '#1b5e20'),
]

# Cảnh báo trong Streamlit
DISCLAIMER = (
    'Mô hình được huấn luyện trên dataset Credit Score Classification (Kaggle). '
    'Kết quả chỉ mang tính minh họa cho đồ án học tập, không dùng cho quyết định '
    'tín dụng thực tế.'
)

APP_TITLE = 'AI Credit Scoring'
APP_SUBTITLE = 'Chấm điểm tín dụng cá nhân dựa trên Machine Learning'
