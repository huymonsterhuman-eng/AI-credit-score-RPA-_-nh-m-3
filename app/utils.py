"""Hàm dùng lại cho Streamlit app: load model, build input, predict, convert FICO."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from config import FEATURES_PATH, FICO_BINS, MODEL_PATH


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
    User chỉ nhập current-month values. Temporal features được auto-fill:
      - _lag1 = current value (giả định tháng trước giống hệt)
      - _delta1 = 0
      - _roll_mean3 = current value
      - _roll_std3 = 0
      - delay_streak_prev = 0
      - debt_trend_prev = 0
    Nếu user_inputs đã có key ứng với temporal feature, giá trị đó được giữ.
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
    """Chạy model và trả về dict đầy đủ kết quả."""
    model, schema = load_artifacts()
    X = build_input_row(user_inputs, schema)
    proba = model.predict_proba(X)[0]  # [P_poor, P_standard, P_good]
    hard_class_idx = int(np.argmax(proba))
    hard_class = schema['target_order'][hard_class_idx]

    fico = proba_to_fico(proba)
    rating, color = fico_to_rating(fico)

    return {
        'proba': {
            'Poor': float(proba[0]),
            'Standard': float(proba[1]),
            'Good': float(proba[2]),
        },
        'predicted_class': hard_class,
        'fico_score': float(fico),
        'rating': rating,
        'rating_color': color,
        'input_row': X,
    }


FICO_ANCHORS = (400.0, 620.0, 830.0)  # Poor, Standard, Good anchors


def proba_to_fico(proba: np.ndarray) -> float:
    """proba theo thứ tự [Poor, Standard, Good] → FICO 300-850.
    Dùng weighted anchor để tránh nghịch lý argmax=Standard nhưng FICO=Poor.
    """
    return float(np.dot(proba, FICO_ANCHORS))


def fico_to_rating(score: float) -> tuple[str, str]:
    """Trả về (rating_label, hex_color)."""
    for lo, hi, label, color in FICO_BINS:
        if lo <= score < hi:
            return label, color
    return FICO_BINS[-1][2], FICO_BINS[-1][3]


def suggestions_for(result: dict, user_inputs: dict) -> list[str]:
    """Sinh gợi ý cải thiện đơn giản dựa trên score và một số feature."""
    tips = []
    score = result['fico_score']
    if score < 580:
        tips.append('⚠️ Điểm tín dụng thấp — hầu hết ngân hàng sẽ từ chối hoặc đưa lãi suất cao.')
    elif score < 670:
        tips.append('Điểm ở mức Fair — có thể được duyệt nhưng lãi suất chưa tối ưu.')

    dti = user_inputs.get('Debt_to_Income_Annual')
    if dti is not None and dti > 0.4:
        tips.append(f'Tỷ lệ nợ trên thu nhập năm là {dti:.1%} — nên giảm dưới 40% để cải thiện điểm.')

    delay = user_inputs.get('Delay_from_due_date')
    if delay is not None and delay > 10:
        tips.append('Trả trễ trung bình > 10 ngày — cải thiện thanh toán đúng hạn là yếu tố ảnh hưởng lớn nhất.')

    util = user_inputs.get('Credit_Utilization_Ratio')
    if util is not None and util > 60:
        tips.append(f'Tỷ lệ sử dụng tín dụng {util:.0f}% — nên giữ dưới 30% để tối ưu điểm.')

    inquiries = user_inputs.get('Num_Credit_Inquiries')
    if inquiries is not None and inquiries > 5:
        tips.append('Số lần bị tra cứu tín dụng cao — hạn chế mở thẻ/vay mới trong 6 tháng tới.')

    if not tips:
        tips.append('✅ Hồ sơ tài chính tốt, tiếp tục duy trì thói quen hiện tại.')
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

    # Model là Pipeline: preprocessor + classifier
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
    # Loại các feature không actionable (user không control được)
    hidden_keywords = ('month_idx', 'Tháng ghi nhận')
    df = df[~df['feature'].str.contains('|'.join(hidden_keywords), regex=True)]
    df = df[~df['feature_raw'].str.contains('month_idx')]
    df['abs_shap'] = df['shap_value'].abs()
    df = df.sort_values('abs_shap', ascending=False).head(top_n)
    return df.drop(columns=['abs_shap']).reset_index(drop=True)


def derive_behavior(user_inputs: dict) -> tuple[str, str]:
    """Suy ra Spending_Level & Payment_Value từ số liệu tài chính của user.
    Tránh dropdown chủ quan.
    """
    salary = max(user_inputs.get('Monthly_Inhand_Salary', 1), 1)
    emi = user_inputs.get('Total_EMI_per_month', 0)
    invested = user_inputs.get('Amount_invested_monthly', 0)
    debt = user_inputs.get('Outstanding_Debt', 0)

    # Spending level = tỷ lệ nghĩa vụ tài chính / lương
    spending_ratio = (emi + debt * 0.02) / salary  # 0.02: giả định 2% nợ trả mỗi tháng
    spending_level = 'High' if spending_ratio > 0.5 else 'Low'

    # Payment value = mức độ giao dịch (đầu tư + EMI) so với lương
    payment_ratio = (emi + invested) / salary
    if payment_ratio < 0.2:
        payment_value = 'Small'
    elif payment_ratio < 0.5:
        payment_value = 'Medium'
    else:
        payment_value = 'Large'

    return spending_level, payment_value
