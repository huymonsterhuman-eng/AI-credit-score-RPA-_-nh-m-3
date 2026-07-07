"""Hàm dùng lại cho Streamlit app: load model, build input, predict, SHAP, admin dashboard."""

import json
import sqlite3
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from config import (ADMIN_PASSWORD, FEATURES_PATH, GOOGLE_SHEET_ID, MODEL_PATH,
                     PROJECT_ROOT, SERVICE_ACCOUNT_PATH, SHEET_NAME)

# ============ SQLite predictions store ============
PREDICTIONS_DB = PROJECT_ROOT / 'data' / 'predictions.db'


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
    """Sinh gợi ý cải thiện dựa trên class dự đoán và feature.

    Các ngưỡng threshold dưới đây được chọn dựa trên chuẩn công nghiệp
    (FICO, CFPB, NHNN Việt Nam) — chi tiết nguồn ở từng tip.
    """
    tips = []
    cls = result['predicted_class']

    # ==== Tip 1: Overview theo class model ====
    if cls == 'Poor':
        tips.append('⚠️ Nhóm tín dụng Kém — hầu hết ngân hàng sẽ từ chối hoặc đưa lãi suất cao.')
    elif cls == 'Standard':
        tips.append('Nhóm tín dụng Trung bình — có thể được duyệt nhưng lãi suất chưa tối ưu.')
    else:
        tips.append('✅ Nhóm tín dụng Tốt — đủ điều kiện vay với lãi suất ưu đãi.')

    # ==== Tip 2: Debt-to-income ratio ====
    # Nguồn: US CFPB khuyến nghị DTI ≤ 43% cho Qualified Mortgage.
    # Fannie Mae/Freddie Mac chấp nhận tối đa 45%. Ngưỡng 40% chọn conservative
    # để cảnh báo sớm. Ref: https://www.consumerfinance.gov/ask-cfpb/what-is-a-debt-to-income-ratio-en-1791/
    dti = user_inputs.get('Debt_to_Income_Annual')
    if dti is not None and dti > 0.4:
        tips.append(f'Tỷ lệ nợ trên thu nhập năm là {dti:.1%} — nên giảm dưới 40% để cải thiện.')

    # ==== Tip 3: Delay from due date ====
    # Nguồn: Thông tư 11/2021/TT-NHNN — nợ quá hạn 10 ngày phân loại vào nhóm 2
    # (nợ cần chú ý). Đây là ngưỡng CIC bắt đầu ghi nhận rủi ro tín dụng.
    delay = user_inputs.get('Delay_from_due_date')
    if delay is not None and delay > 10:
        tips.append(f'Trả trễ trung bình {delay:.0f} ngày — cải thiện thanh toán đúng hạn là yếu tố ảnh hưởng lớn nhất.')

    # ==== Tip 4: Credit utilization ====
    # Nguồn: FICO/Experian best practice — tối ưu dưới 30%, trên 30% là high risk.
    # Ngưỡng 60% chọn để cảnh báo case rõ ràng.
    # Ref: https://www.myfico.com/credit-education/blog/credit-utilization-ratio
    util = user_inputs.get('Credit_Utilization_Ratio')
    if util is not None and util > 60:
        tips.append(f'Tỷ lệ sử dụng tín dụng {util:.0f}% — nên giữ dưới 30% để tối ưu.')

    # ==== Tip 5: Credit inquiries ====
    # Nguồn: Experian — mỗi hard inquiry giảm FICO 5-10 điểm, ảnh hưởng 12 tháng.
    # >5 inquiries trong 6 tháng được coi là pattern rủi ro cao.
    # Ref: https://www.experian.com/blogs/ask-experian/credit-education/credit-inquiries/
    inquiries = user_inputs.get('Num_Credit_Inquiries')
    if inquiries is not None and inquiries > 5:
        tips.append(f'Số lần bị tra cứu tín dụng ({inquiries}) khá cao — hạn chế mở thẻ/vay mới trong 6 tháng tới.')

    # ==== Tip 6: Số lần trả trễ ====
    # Nguồn: không có chuẩn công nghiệp cứng. Ngưỡng 10 chọn dựa trên
    # phân phối dataset + logic: >10 lần trễ = pattern hành vi, không phải sự cố đơn lẻ.
    # Design choice minh họa.
    num_delay = user_inputs.get('Num_of_Delayed_Payment')
    if num_delay is not None and num_delay > 10:
        tips.append(f'Đã trả trễ {num_delay} lần — con số cao, ngân hàng coi là khách rủi ro. Nên setup autopay để không trễ tiếp.')

    # ==== Tip 7: Credit mix ====
    # Nguồn: FICO — Credit Mix chiếm 10% điểm FICO. Giá trị "Bad" là label từ dataset,
    # phản ánh cơ cấu tín dụng mất cân bằng (chỉ có 1 loại vay hoặc quá tập trung).
    # Ref: https://www.myfico.com/credit-education/whats-in-your-credit-score
    credit_mix = user_inputs.get('Credit_Mix')
    if credit_mix == 'Bad':
        tips.append('Cơ cấu tín dụng "Bad" — cân bằng thêm giữa thẻ tín dụng, vay tín chấp, vay có bảo đảm để cải thiện.')

    # ==== Tip 8: Credit history length ====
    # Nguồn: FICO chính thức — Length of Credit History chiếm 15% điểm FICO.
    # Hồ sơ <24 tháng được coi là "thin file", điểm thấp do thiếu data lịch sử.
    # Ngưỡng 2 năm = ranh giới thoát khỏi thin file status.
    # Ref: https://www.myfico.com/credit-education/whats-in-your-credit-score
    history = user_inputs.get('Credit_History_Months')
    if history is not None and history < 24:
        tips.append(f'Lịch sử tín dụng chỉ {history} tháng — quá ngắn. Duy trì tài khoản/thẻ hiện có càng lâu càng tốt.')

    # ==== Tip 9: Payment of minimum amount ====
    # Nguồn: CFPB + Investopedia — trả tối thiểu là dấu hiệu tài chính căng thẳng,
    # tổng lãi trả gấp 3-5 lần so với trả full balance. Đây là boolean behavior,
    # không có threshold số.
    # Ref: https://www.investopedia.com/terms/m/minimum-monthly-payment.asp
    pay_min = user_inputs.get('Payment_of_Min_Amount')
    if pay_min == 'Yes':
        tips.append('Bạn chỉ đang trả khoản tối thiểu — dấu hiệu tài chính căng thẳng. Cố gắng trả nhiều hơn mức minimum ít nhất 20%.')

    # ==== Tip 10: EMI vs salary ====
    # Nguồn: NHTM Việt Nam (Vietcombank, VPBank, Techcombank) thường yêu cầu
    # EMI/thu nhập <40-50% để duyệt vay. >50% = không đủ dòng tiền dự phòng
    # cho tình huống khẩn cấp.
    emi_ratio = user_inputs.get('EMI_to_Salary_Ratio')
    if emi_ratio is not None and emi_ratio > 0.5:
        tips.append(f'EMI hàng tháng chiếm {emi_ratio:.0%} lương — quá cao. Nên duy trì dưới 40% để có dòng tiền dự phòng.')

    # ==== Fallback: nếu Good mà không có tip nào cụ thể ====
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


# ============ SQLite predictions store ============

def init_predictions_db() -> None:
    """Tạo file DB + schema nếu chưa có. Idempotent — an toàn gọi mỗi lần app start."""
    PREDICTIONS_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(PREDICTIONS_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                customer_name TEXT,
                timestamp TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                p_poor REAL NOT NULL,
                p_standard REAL NOT NULL,
                p_good REAL NOT NULL,
                inputs_json TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_email ON predictions(email)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)")


def save_prediction(email: str, customer_name: str, timestamp: str,
                     inputs: dict, result: dict) -> int:
    """INSERT 1 row vào predictions DB. Trả về id vừa insert.

    - email: normalized lowercase strip (rỗng nếu user không nhập)
    - timestamp: ISO string
    - inputs: dict full user_inputs (sẽ JSON serialize)
    - result: dict từ predict() — có 'predicted_class' và 'proba'
    """
    init_predictions_db()  # ensure schema exists

    email_norm = (email or '').strip().lower()

    # Convert numpy types trong inputs → native để JSON serialize được
    def _clean(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, np.ndarray): return v.tolist()
        return v
    inputs_clean = {k: _clean(v) for k, v in inputs.items()}

    with sqlite3.connect(PREDICTIONS_DB) as conn:
        cursor = conn.execute(
            """INSERT INTO predictions
                (email, customer_name, timestamp, predicted_class,
                 p_poor, p_standard, p_good, inputs_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                email_norm,
                customer_name or '',
                timestamp,
                result['predicted_class'],
                float(result['proba']['Poor']),
                float(result['proba']['Standard']),
                float(result['proba']['Good']),
                json.dumps(inputs_clean, ensure_ascii=False),
            )
        )
        return cursor.lastrowid


def get_previous_prediction(email: str, exclude_id: int | None = None) -> dict | None:
    """Tìm prediction gần nhất của email (loại trừ exclude_id nếu có).

    Trả về dict:
      { 'id', 'timestamp', 'predicted_class', 'proba': {...}, 'inputs': {...} }
    Hoặc None nếu email rỗng / không tìm thấy record trước đó.
    """
    email_norm = (email or '').strip().lower()
    if not email_norm:
        return None
    if not PREDICTIONS_DB.exists():
        return None

    with sqlite3.connect(PREDICTIONS_DB) as conn:
        conn.row_factory = sqlite3.Row
        query = """
            SELECT id, timestamp, predicted_class, p_poor, p_standard, p_good,
                   inputs_json, customer_name
            FROM predictions
            WHERE email = ? AND (? IS NULL OR id != ?)
            ORDER BY timestamp DESC
            LIMIT 1
        """
        row = conn.execute(query, (email_norm, exclude_id, exclude_id)).fetchone()

    if row is None:
        return None
    return {
        'id': row['id'],
        'timestamp': row['timestamp'],
        'customer_name': row['customer_name'],
        'predicted_class': row['predicted_class'],
        'proba': {
            'Poor': row['p_poor'],
            'Standard': row['p_standard'],
            'Good': row['p_good'],
        },
        'inputs': json.loads(row['inputs_json']),
    }


# Feature nào là "actionable" (user có thể control) và hướng "good" là gì.
# Direction: 'down' = giảm là tốt, 'up' = tăng là tốt
ACTIONABLE_FEATURES = {
    # Field: (label_vi, direction, unit_format)
    'Annual_Income':              ('Thu nhập hàng năm',       'up',   '${:,.0f}'),
    'Monthly_Inhand_Salary':      ('Lương thực nhận',         'up',   '${:,.0f}'),
    'Monthly_Balance':            ('Số dư cuối tháng',        'up',   '${:,.0f}'),
    'Total_EMI_per_month':        ('EMI hàng tháng',          'down', '${:,.0f}'),
    'Amount_invested_monthly':    ('Đầu tư hàng tháng',       'up',   '${:,.0f}'),
    'Num_Bank_Accounts':          ('Số tài khoản NH',         'up',   '{:.0f}'),
    'Num_Credit_Card':            ('Số thẻ tín dụng',         'down', '{:.0f}'),
    'Num_of_Loan':                ('Số khoản vay',            'down', '{:.0f}'),
    'Interest_Rate':              ('Lãi suất TB',             'down', '{:.1f}%'),
    'Outstanding_Debt':           ('Nợ tồn đọng',             'down', '${:,.0f}'),
    'Credit_Utilization_Ratio':   ('Tỷ lệ dùng tín dụng',     'down', '{:.1f}%'),
    'Delay_from_due_date':        ('Số ngày trả trễ',         'down', '{:.0f} ngày'),
    'Num_of_Delayed_Payment':     ('Số lần trả trễ',          'down', '{:.0f}'),
    'Num_Credit_Inquiries':       ('Số lần tra cứu',          'down', '{:.0f}'),
    'Changed_Credit_Limit':       ('Thay đổi hạn mức',        'down', '{:.1f}'),
    # Categorical (special handling)
    'Payment_of_Min_Amount':      ('Trả khoản tối thiểu',     'cat',  None),
    'Credit_Mix':                 ('Chất lượng cơ cấu tín dụng', 'cat', None),
}

# Ordering cho categorical features (index cao hơn = tốt hơn)
CATEGORICAL_ORDER = {
    'Payment_of_Min_Amount': {'Yes': 0, 'NM': 1, 'No': 2},
    'Credit_Mix': {'Bad': 0, 'Standard': 1, 'Good': 2},
}


def diff_inputs(prev_inputs: dict, curr_inputs: dict, top_n: int = 8) -> list[dict]:
    """So sánh 2 dict inputs, trả về top_n feature có thay đổi lớn nhất.

    Mỗi item là dict:
      { 'feature', 'label', 'prev', 'curr', 'delta', 'delta_pct',
        'impact': 'good'|'bad'|'neutral', 'format' }
    """
    diffs = []
    for field, (label, direction, fmt) in ACTIONABLE_FEATURES.items():
        prev_v = prev_inputs.get(field)
        curr_v = curr_inputs.get(field)
        if prev_v is None or curr_v is None:
            continue

        if direction == 'cat':
            # Categorical: dùng ordering để tính "delta rank"
            order = CATEGORICAL_ORDER.get(field, {})
            prev_rank = order.get(str(prev_v), 1)
            curr_rank = order.get(str(curr_v), 1)
            if prev_rank == curr_rank:
                continue  # không thay đổi
            impact = 'good' if curr_rank > prev_rank else 'bad'
            diffs.append({
                'feature': field, 'label': label,
                'prev': str(prev_v), 'curr': str(curr_v),
                'delta': curr_rank - prev_rank,
                'delta_pct': None,
                'impact': impact, 'is_categorical': True,
                'format': fmt,
            })
        else:
            # Numeric
            try:
                prev_num = float(prev_v)
                curr_num = float(curr_v)
            except (TypeError, ValueError):
                continue
            delta = curr_num - prev_num
            if abs(delta) < 1e-6:
                continue  # không thay đổi
            # Impact: giảm là tốt (down) → delta<0 là good
            if direction == 'down':
                impact = 'good' if delta < 0 else 'bad'
            else:  # up
                impact = 'good' if delta > 0 else 'bad'
            delta_pct = (delta / prev_num * 100) if prev_num != 0 else None
            diffs.append({
                'feature': field, 'label': label,
                'prev': prev_num, 'curr': curr_num,
                'delta': delta, 'delta_pct': delta_pct,
                'impact': impact, 'is_categorical': False,
                'format': fmt,
            })

    # Sort theo magnitude — categorical được ưu tiên (rank change nhỏ nhưng ý nghĩa)
    # Chuẩn hóa magnitude: numeric dùng abs(delta_pct) nếu có, categorical dùng
    # abs(delta) * 50 để boost lên top
    def _mag(d):
        if d['is_categorical']:
            return abs(d['delta']) * 50
        if d['delta_pct'] is not None:
            return abs(d['delta_pct'])
        return abs(d['delta'])

    diffs.sort(key=_mag, reverse=True)
    return diffs[:top_n]


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
