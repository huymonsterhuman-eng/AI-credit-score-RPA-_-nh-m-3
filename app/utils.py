"""Hàm dùng lại cho Streamlit app: load model, build input, predict, SHAP, admin dashboard."""

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib

from config import (ADMIN_PASSWORD, FEATURES_PATH, GOOGLE_SHEET_ID, MODEL_PATH,
                     SERVICE_ACCOUNT_PATH, SHEET_NAME)


# Màu cho 3 nhãn phân loại
CLASS_COLORS = {
    'Poor': '#d9534f',
    'Standard': '#f0ad4e',
    'Good': '#5cb85c',
}

CLASS_LABELS_VI = {
    'Poor': 'Kém',
    'Standard': 'Trung bình',
    'Good': 'Tốt',
}


@lru_cache(maxsize=1)
def load_artifacts():
    """Load model + schema. Cached để Streamlit không load lại mỗi lần rerun."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f'Không tìm thấy {MODEL_PATH}. Chạy notebook 04 và sync file từ Drive về '
            f'thư mục models/ trước.'
        )
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, encoding='utf-8') as f:
        schema = json.load(f)
    return model, schema


def build_input_row(user_inputs: dict, schema: dict) -> pd.DataFrame:
    """
    Xây DataFrame 1 dòng có đúng cột model expect.
    User chỉ nhập current-month values. Temporal features được auto-fill.
    """
    row = {}
    for col in schema['input_columns']:
        if col in user_inputs:
            row[col] = user_inputs[col]
        elif col.endswith('_lag1'):
            base = col[:-len('_lag1')]
            row[col] = user_inputs.get(base, np.nan)
        elif col.endswith('_delta1'):
            row[col] = 0.0
        elif col.endswith('_roll_mean3'):
            base = col[:-len('_roll_mean3')]
            row[col] = user_inputs.get(base, np.nan)
        elif col.endswith('_roll_std3'):
            row[col] = 0.0
        elif col == 'delay_streak_prev':
            row[col] = 0
        elif col == 'debt_trend_prev':
            row[col] = 0.0
        elif col == 'month_idx':
            row[col] = user_inputs.get('month_idx', 6)
        else:
            row[col] = user_inputs.get(col, np.nan)
    return pd.DataFrame([row])


def predict(user_inputs: dict) -> dict:
    """Chạy model và trả về dict phân loại + xác suất."""
    model, schema = load_artifacts()
    X = build_input_row(user_inputs, schema)
    proba = model.predict_proba(X)[0]  # [P_poor, P_standard, P_good]
    hard_class_idx = int(np.argmax(proba))
    hard_class = schema['target_order'][hard_class_idx]

    return {
        'proba': {
            'Poor': float(proba[0]),
            'Standard': float(proba[1]),
            'Good': float(proba[2]),
        },
        'predicted_class': hard_class,
        'predicted_class_vi': CLASS_LABELS_VI.get(hard_class, hard_class),
        'confidence': float(proba[hard_class_idx]),
        'color': CLASS_COLORS.get(hard_class, '#666'),
        'input_row': X,
    }


def suggestions_for(result: dict, user_inputs: dict) -> list[str]:
    """Sinh gợi ý cải thiện dựa trên class dự đoán và feature."""
    tips = []
    cls = result['predicted_class']

    if cls == 'Poor':
        tips.append('⚠️ Nhóm tín dụng Kém — hầu hết ngân hàng sẽ từ chối hoặc đưa lãi suất cao.')
    elif cls == 'Standard':
        tips.append('Nhóm tín dụng Trung bình — có thể được duyệt nhưng lãi suất chưa tối ưu.')
    else:
        tips.append('✅ Nhóm tín dụng Tốt — đủ điều kiện vay với lãi suất ưu đãi.')

    dti = user_inputs.get('Debt_to_Income_Annual')
    if dti is not None and dti > 0.4:
        tips.append(f'Tỷ lệ nợ trên thu nhập năm là {dti:.1%} — nên giảm dưới 40% để cải thiện.')

    delay = user_inputs.get('Delay_from_due_date')
    if delay is not None and delay > 10:
        tips.append('Trả trễ trung bình > 10 ngày — cải thiện thanh toán đúng hạn là yếu tố ảnh hưởng lớn nhất.')

    util = user_inputs.get('Credit_Utilization_Ratio')
    if util is not None and util > 60:
        tips.append(f'Tỷ lệ sử dụng tín dụng {util:.0f}% — nên giữ dưới 30% để tối ưu.')

    inquiries = user_inputs.get('Num_Credit_Inquiries')
    if inquiries is not None and inquiries > 5:
        tips.append('Số lần bị tra cứu tín dụng cao — hạn chế mở thẻ/vay mới trong 6 tháng tới.')

    if len(tips) == 1 and cls == 'Good':
        tips.append('Tiếp tục duy trì thói quen tài chính hiện tại.')
    return tips


FEATURE_LABELS = {
    'Payment_of_Min_Amount_Yes': 'Có trả tối thiểu',
    'Payment_of_Min_Amount_No': 'Không trả tối thiểu',
    'Payment_of_Min_Amount_NM': 'Không xác định',
    'Outstanding_Debt': 'Nợ tồn đọng',
    'Interest_Rate': 'Lãi suất khoản vay',
    'Credit_History_Months': 'Lịch sử tín dụng (tháng)',
    'Num_Credit_Inquiries': 'Số lần bị tra cứu',
    'Delay_from_due_date': 'Số ngày trả trễ',
    'Num_of_Delayed_Payment': 'Số lần trả trễ',
    'Spending_Level_High': 'Mức chi tiêu cao',
    'Spending_Level_Low': 'Mức chi tiêu thấp',
    'Payment_Value_Small': 'Trả từng khoản nhỏ',
    'Payment_Value_Medium': 'Trả khoản trung bình',
    'Payment_Value_Large': 'Trả khoản lớn',
    'Changed_Credit_Limit': 'Thay đổi hạn mức',
    'Num_Credit_Card': 'Số thẻ tín dụng',
    'Num_Bank_Accounts': 'Số tài khoản ngân hàng',
    'Num_of_Loan': 'Số khoản vay',
    'Debt_to_Income_Annual': 'Tỷ lệ nợ / thu nhập năm',
    'EMI_to_Salary_Ratio': 'Tỷ lệ EMI / lương',
    'Investment_Rate': 'Tỷ lệ đầu tư / lương',
    'CC_per_Bank': 'Số thẻ / số tài khoản',
    'Loan_per_CC': 'Số vay / số thẻ',
    'Age_Start_Credit': 'Tuổi bắt đầu có tín dụng',
    'Debt_per_Loan': 'Nợ trung bình mỗi khoản',
    'Balance_to_Salary': 'Số dư / lương',
    'Annual_Income': 'Thu nhập hàng năm',
    'Monthly_Inhand_Salary': 'Lương thực nhận',
    'Monthly_Balance': 'Số dư cuối tháng',
    'Total_EMI_per_month': 'Tổng EMI hàng tháng',
    'Amount_invested_monthly': 'Đầu tư hàng tháng',
    'Credit_Utilization_Ratio': 'Tỷ lệ sử dụng tín dụng',
    'Age': 'Tuổi',
    'Num_Loan_Types': 'Số loại vay',
    'Credit_Mix': 'Chất lượng tổ hợp tín dụng',
    'month_idx': 'Tháng ghi nhận',
    'delay_streak_prev': 'Chuỗi tháng trả trễ',
    'debt_trend_prev': 'Xu hướng nợ',
}


def clean_feature_name(raw: str) -> str:
    """Bỏ prefix num__/ord__/cat__ và đổi sang label tiếng Việt nếu có."""
    for prefix in ('num__', 'ord__', 'cat__'):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    for suffix, label in [('_lag1', ' (tháng trước)'),
                           ('_delta1', ' (thay đổi so tháng trước)'),
                           ('_roll_mean3', ' (TB 3 tháng)'),
                           ('_roll_std3', ' (dao động 3 tháng)')]:
        if raw.endswith(suffix):
            base = raw[:-len(suffix)]
            return FEATURE_LABELS.get(base, base.replace('_', ' ')) + label
    return FEATURE_LABELS.get(raw, raw.replace('_', ' '))


def get_shap_values(user_inputs: dict, top_n: int = 8) -> pd.DataFrame | None:
    """Tính SHAP cho input hiện tại. Trả về DataFrame top_n features theo |shap|."""
    try:
        import shap
    except ImportError:
        return None

    model, schema = load_artifacts()
    X = build_input_row(user_inputs, schema)

    pre = model.named_steps['pre']
    clf = model.named_steps['model']

    X_trans = pre.transform(X)
    feat_names = pre.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_trans)

    # Multi-class: lấy class 'Good' (index 2) — features đẩy về/kéo ra Good
    if isinstance(shap_values, list):
        sv = shap_values[2][0]
    else:
        sv = shap_values[0, :, 2] if shap_values.ndim == 3 else shap_values[0]

    df = pd.DataFrame({
        'feature_raw': feat_names,
        'feature': [clean_feature_name(f) for f in feat_names],
        'shap_value': sv,
    })
    hidden_keywords = ('month_idx', 'Tháng ghi nhận')
    df = df[~df['feature'].str.contains('|'.join(hidden_keywords), regex=True)]
    df = df[~df['feature_raw'].str.contains('month_idx')]
    df['abs_shap'] = df['shap_value'].abs()
    df = df.sort_values('abs_shap', ascending=False).head(top_n)
    return df.drop(columns=['abs_shap']).reset_index(drop=True)


# ============ Admin Dashboard helpers ============

def check_admin_password(input_password: str) -> bool:
    """So sánh với ADMIN_PASSWORD trong config."""
    return input_password == ADMIN_PASSWORD


def load_sheet_data() -> pd.DataFrame:
    """Đọc toàn bộ Google Sheet qua service account.

    Trả về DataFrame với:
      - 'Thời gian' → datetime
      - 'Xác suất Poor/Standard/Good' → float
      - Các cột khác giữ nguyên string
    """
    import gspread
    from google.oauth2.service_account import Credentials

    if not SERVICE_ACCOUNT_PATH.exists():
        raise FileNotFoundError(
            f'Chưa cấu hình service account. Đọc README, tạo credential tại '
            f'Google Cloud Console và đặt file JSON tại {SERVICE_ACCOUNT_PATH}.'
        )

    scopes = [
        'https://www.googleapis.com/auth/spreadsheets.readonly',
        'https://www.googleapis.com/auth/drive.readonly',
    ]
    creds = Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_PATH), scopes=scopes
    )
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(GOOGLE_SHEET_ID)

    # Try named tab; fallback to first tab if not found (tab name có thể khác nhau
    # trên account tiếng Anh vs tiếng Việt: 'Sheet1' vs 'Trang tính1')
    if SHEET_NAME:
        try:
            ws = sh.worksheet(SHEET_NAME)
        except gspread.WorksheetNotFound:
            ws = sh.sheet1
    else:
        ws = sh.sheet1

    records = ws.get_all_records()
    df = pd.DataFrame(records)

    if df.empty:
        return df

    # Parse datetime
    if 'Thời gian' in df.columns:
        df['Thời gian'] = pd.to_datetime(df['Thời gian'], errors='coerce')

    # Parse xác suất về float [0, 1]
    for col in ('Xác suất Poor', 'Xác suất Standard', 'Xác suất Good'):
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '.'),  # Vietnamese decimal
                errors='coerce'
            ).astype(float)  # force float64 để tránh conflict khi chia 100
            # Normalize: nếu value > 1 → chia 100 (Sheet lưu 30.24 thay vì 0.3024)
            df[col] = df[col].where(df[col] <= 1, df[col] / 100)

    return df.sort_values('Thời gian', ascending=False) if 'Thời gian' in df.columns else df


def derive_behavior(user_inputs: dict) -> tuple[str, str]:
    """Suy ra Spending_Level & Payment_Value từ số liệu tài chính của user."""
    salary = max(user_inputs.get('Monthly_Inhand_Salary', 1), 1)
    emi = user_inputs.get('Total_EMI_per_month', 0)
    invested = user_inputs.get('Amount_invested_monthly', 0)
    debt = user_inputs.get('Outstanding_Debt', 0)

    spending_ratio = (emi + debt * 0.02) / salary
    spending_level = 'High' if spending_ratio > 0.5 else 'Low'

    payment_ratio = (emi + invested) / salary
    if payment_ratio < 0.2:
        payment_value = 'Small'
    elif payment_ratio < 0.5:
        payment_value = 'Medium'
    else:
        payment_value = 'Large'

    return spending_level, payment_value
