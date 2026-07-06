"""Cấu hình cho Streamlit app."""

from pathlib import Path

APP_ROOT = Path(__file__).parent
PROJECT_ROOT = APP_ROOT.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'model.pkl'
FEATURES_PATH = PROJECT_ROOT / 'models' / 'features.json'

# n8n webhook — dán URL webhook sau khi setup n8n
N8N_WEBHOOK_URL = 'http://localhost:5678/webhook/credit-scoring'
WEBHOOK_TIMEOUT = 8  # seconds

# Cảnh báo trong Streamlit
DISCLAIMER = (
    'Mô hình được huấn luyện trên dataset Credit Score Classification (Kaggle). '
    'Kết quả chỉ mang tính minh họa cho đồ án học tập, không dùng cho quyết định '
    'tín dụng thực tế.'
)

APP_TITLE = 'AI Credit Classification'
APP_SUBTITLE = 'Phân loại mức tín dụng cá nhân dựa trên Machine Learning'

# ============ Admin Dashboard ============
ADMIN_PASSWORD = 'admin123'  # ĐỔI trước khi demo/production
GOOGLE_SHEET_ID = '1ASE9CUCseFaU2cr99PrSVEfxWbxUkNGZ1xQhox5AKLM'
SERVICE_ACCOUNT_PATH = PROJECT_ROOT / 'models' / 'service_account.json'
SHEET_NAME = 'Sheet1'  # tab name trong workbook (thường là 'Sheet1' hoặc 'Trang tính1')
