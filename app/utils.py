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


def suggestions_for(result: dict, user_inputs: dict,
                     previous: dict | None = None) -> list[str]:
    """Sinh gợi ý cải thiện theo ngữ cảnh: class hiện tại + lịch sử lần trước.

    5 tình huống:
      - Poor lần đầu     : giải thích + danh sách ưu tiên khẩn
      - Poor không đổi   : thẳng thắn, nhắc lại gợi ý chưa làm
      - Poor đang cải thiện (vẫn Poor nhưng P(Good) tăng): động viên + tiếp tục
      - Standard         : lộ trình cụ thể lên Good
      - Good             : khuyến nghị sản phẩm + duy trì
    """
    tips = []
    cls = result['predicted_class']

    # ── Xác định ngữ cảnh lịch sử ──────────────────────────────────────────
    prev_cls = previous.get('predicted_class') if previous else None
    prev_p_good = previous['proba']['Good'] if previous else None
    curr_p_good = result['proba']['Good']
    improving = (prev_p_good is not None) and (curr_p_good - prev_p_good > 0.05)
    stagnant = (prev_cls == 'Poor') and (cls == 'Poor') and not improving
    first_time = previous is None

    # ── Overview theo ngữ cảnh ─────────────────────────────────────────────
    if cls == 'Poor' and first_time:
        tips.append(
            '**Nhóm tín dụng Kém (Poor)** — lần đầu tra cứu. '
            'Hầu hết ngân hàng sẽ từ chối hoặc áp lãi suất >15%/năm. '
            'Tin tốt: với 6–12 tháng kỷ luật tài chính, bạn hoàn toàn có thể lên Standard.'
        )
    elif cls == 'Poor' and stagnant:
        tips.append(
            '**Vẫn ở nhóm Kém (Poor)** — chưa có cải thiện so với lần trước. '
            'Các gợi ý bên dưới là những việc cần làm ngay, không nên trì hoãn thêm.'
        )
    elif cls == 'Poor' and improving:
        tips.append(
            f'**Vẫn Poor nhưng đang tiến bộ** — xác suất Good tăng '
            f'{(curr_p_good - prev_p_good)*100:.1f}% so với lần trước. '
            'Đi đúng hướng rồi, tiếp tục duy trì!'
        )
    elif cls == 'Standard' and prev_cls == 'Poor':
        tips.append(
            '**Đã lên Standard từ Poor** — cải thiện rõ rệt! '
            'Bạn đã có thể được duyệt vay cơ bản. '
            'Bây giờ tập trung lên Good để hưởng lãi suất tốt hơn 2–3%/năm.'
        )
    elif cls == 'Standard':
        tips.append(
            '**Nhóm Trung bình (Standard)** — có thể được duyệt vay nhưng lãi suất chưa tối ưu. '
            'Khoảng cách lên Good không xa — thường chỉ cần cải thiện 2–3 yếu tố.'
        )
    elif cls == 'Good' and prev_cls in ('Poor', 'Standard'):
        tips.append(
            f'**Đã đạt nhóm Tốt (Good)** từ {prev_cls}! '
            'Bạn đủ điều kiện vay với lãi suất ưu đãi. Duy trì thói quen hiện tại.'
        )
    else:
        tips.append(
            '**Nhóm Tốt (Good)** — đủ điều kiện vay với lãi suất ưu đãi. '
            'Tiếp tục duy trì là cách tốt nhất.'
        )

    # ── Lấy các feature cần dùng ───────────────────────────────────────────
    dti          = user_inputs.get('Debt_to_Income_Annual', 0) or 0
    delay        = user_inputs.get('Delay_from_due_date', 0) or 0
    num_delay    = user_inputs.get('Num_of_Delayed_Payment', 0) or 0
    util         = user_inputs.get('Credit_Utilization_Ratio', 0) or 0
    inquiries    = user_inputs.get('Num_Credit_Inquiries', 0) or 0
    history      = user_inputs.get('Credit_History_Months', 0) or 0
    pay_min      = user_inputs.get('Payment_of_Min_Amount', '')
    emi_ratio    = user_inputs.get('EMI_to_Salary_Ratio', 0) or 0
    credit_mix   = user_inputs.get('Credit_Mix', '')
    outstanding  = user_inputs.get('Outstanding_Debt', 0) or 0
    balance      = user_inputs.get('Monthly_Balance', 0) or 0
    invested     = user_inputs.get('Amount_invested_monthly', 0) or 0
    num_loans    = user_inputs.get('Num_of_Loan', 0) or 0
    payday       = user_inputs.get('Has_Payday_Loan', 0)

    # ── GỢI Ý THEO TỪNG TÌNH HUỐNG ────────────────────────────────────────

    if cls == 'Poor':
        # ── Phần 1: Vấn đề KHẨN CẤP (ảnh hưởng điểm nhiều nhất) ──────────
        tips.append('─── ƯU TIÊN XỬ LÝ NGAY ───')

        if delay > 30:
            tips.append(
                f'Trả trễ {delay:.0f} ngày (rất nghiêm trọng) — '
                'Đây là yếu tố phá điểm số 1. Bắt đầu từ ngay hôm nay: '
                'setup nhắc nhở tự động hoặc GIRO tự động trả trước ngày đến hạn 2–3 ngày. '
                'Chỉ cần 3 tháng trả đúng hạn liên tiếp, điểm sẽ bắt đầu cải thiện.'
            )
        elif delay > 10:
            tips.append(
                f'Trả trễ TB {delay:.0f} ngày — vượt ngưỡng CIC ghi nhận rủi ro (TT 11/2021/TT-NHNN). '
                'Hành động: bật nhắc lịch thanh toán trên điện thoại, trả trước ít nhất 5 ngày.'
            )

        if num_delay > 20:
            tips.append(
                f'{num_delay} lần trả trễ — hồ sơ đã bị đánh dấu "khách rủi ro cao" ở CIC. '
                'Không có cách nào xóa lịch sử này nhanh chóng — chỉ có thể xây dựng lại bằng cách '
                'trả đúng hạn liên tục 12+ tháng tới. Bắt đầu càng sớm càng tốt.'
            )
        elif num_delay > 10:
            tips.append(
                f'{num_delay} lần trả trễ — cần dừng ngay. '
                'Setup autopay cho tất cả khoản vay/thẻ hiện có.'
            )

        if pay_min == 'Yes':
            tips.append(
                'Đang chỉ trả khoản tối thiểu — dấu hiệu dòng tiền căng thẳng. '
                'Tiền lãi tích lũy gấp 3–5× nếu chỉ trả minimum. '
                'Mục tiêu: tăng dần lên trả ít nhất 30% số dư mỗi tháng.'
            )

        if payday:
            tips.append(
                'Đang có vay ngắn hạn (Payday Loan) — lãi suất loại này thường 200–400%/năm. '
                'Ưu tiên tất toán khoản này trước tất cả các khoản khác.'
            )

        if dti > 0.6:
            tips.append(
                f'Tỷ lệ nợ/thu nhập {dti:.0%} — cực kỳ nguy hiểm. '
                f'Nợ hiện tại {outstanding:,.0f} USD chiếm quá lớn so với thu nhập. '
                'Cần lập kế hoạch trả nợ ngay: ưu tiên khoản lãi suất cao nhất trước (phương pháp Avalanche).'
            )
        elif dti > 0.4:
            tips.append(
                f'Tỷ lệ nợ/thu nhập {dti:.0%} — vượt ngưỡng an toàn 40%. '
                'Tránh vay thêm bất kỳ khoản nào cho đến khi giảm được nợ hiện tại.'
            )

        # ── Phần 2: Cải thiện TRUNG HẠN (3–6 tháng) ──────────────────────
        tips.append('─── CẢI THIỆN TRONG 3–6 THÁNG ───')

        if util > 80:
            tips.append(
                f'Tỷ lệ sử dụng tín dụng {util:.0f}% — quá ngưỡng nguy hiểm. '
                'Mục tiêu giảm xuống dưới 60% trước, sau đó dưới 30%. '
                'Cách: trả bớt dư nợ thẻ, hoặc xin tăng hạn mức (nếu ngân hàng đồng ý).'
            )
        elif util > 60:
            tips.append(
                f'Tỷ lệ sử dụng tín dụng {util:.0f}%. '
                'Mục tiêu: giảm xuống dưới 30% — đây là ngưỡng FICO coi là "safe zone".'
            )

        if emi_ratio > 0.5:
            tips.append(
                f'EMI chiếm {emi_ratio:.0%} lương — không còn dư địa tài chính. '
                'Xem xét đàm phán gia hạn kỳ hạn vay để giảm số tiền trả hàng tháng, '
                'hoặc tìm nguồn thu nhập phụ.'
            )

        if inquiries > 10:
            tips.append(
                f'{inquiries} lần bị tra cứu tín dụng — quá nhiều, báo hiệu đang tìm vay khắp nơi. '
                'Dừng hoàn toàn việc nộp đơn vay mới trong 12 tháng tới.'
            )
        elif inquiries > 5:
            tips.append(
                f'{inquiries} lần tra cứu — hạn chế mở thêm thẻ/vay mới trong 6 tháng tới.'
            )

        if credit_mix == 'Bad':
            tips.append(
                'Cơ cấu tín dụng "Bad" — đang quá tập trung vào 1 loại (thường là thẻ tín dụng). '
                'Sau khi ổn định thanh toán, cân nhắc thêm 1 khoản vay tín chấp nhỏ để đa dạng hóa.'
            )

        # ── Phần 3: Nếu là lần 2+ mà vẫn Poor ───────────────────────────
        if stagnant:
            tips.append('─── NHÌN LẠI SO VỚI LẦN TRƯỚC ───')
            tips.append(
                'Điểm chưa cải thiện kể từ lần tra cứu trước. '
                'Hãy trung thực: trong các gợi ý lần trước, bạn đã thực hiện được gợi ý nào? '
                'Tập trung vào đúng 1 việc dễ nhất trước — thường là "trả đúng hạn tháng này".'
            )

        # ── Phần 4: Lộ trình lên Standard ────────────────────────────────
        tips.append('─── LỘ TRÌNH LÊN STANDARD (6–12 THÁNG) ───')
        tips.append(
            '1. Tháng 1–3: Trả đúng hạn 100%, dừng mọi khoản vay mới. '
            '2. Tháng 3–6: Giảm credit utilization xuống dưới 60%. '
            '3. Tháng 6–12: Duy trì streak thanh toán đúng hạn, giảm dần số lần trễ trong hồ sơ CIC.'
        )

    elif cls == 'Standard':
        # ── Standard: tập trung vào lộ trình lên Good ─────────────────────
        tips.append('─── PHÂN TÍCH ĐỂ LÊN GOOD ───')

        blockers = []  # các yếu tố đang cản trở lên Good

        if delay > 10:
            blockers.append(f'trả trễ {delay:.0f} ngày')
            tips.append(
                f'Trả trễ TB {delay:.0f} ngày — đây là rào cản chính. '
                'Giảm xuống dưới 5 ngày trong 3 tháng tới sẽ tạo ra sự khác biệt rõ rệt.'
            )

        if util > 30:
            blockers.append(f'utilization {util:.0f}%')
            tips.append(
                f'Tỷ lệ sử dụng tín dụng {util:.0f}% — mục tiêu là dưới 30%. '
                f'Cần giảm khoảng {max(0, util - 30):.0f}% nữa. '
                'Cách nhanh nhất: trả bớt dư nợ thẻ tín dụng trước ngày sao kê.'
            )

        if dti > 0.35:
            blockers.append(f'DTI {dti:.0%}')
            tips.append(
                f'Tỷ lệ nợ/thu nhập {dti:.0%} — nên giảm xuống dưới 35% để vào vùng "Good". '
                f'Ưu tiên trả bớt {outstanding:,.0f} USD nợ tồn đọng.'
            )

        if pay_min == 'Yes':
            blockers.append('chỉ trả minimum')
            tips.append(
                'Đang trả khoản tối thiểu — tăng lên trả ít nhất 50% số dư mỗi tháng '
                'sẽ vừa giảm lãi, vừa cải thiện Payment Value trong mắt ngân hàng.'
            )

        if num_delay > 5:
            tips.append(
                f'{num_delay} lần trả trễ trong hồ sơ — mỗi tháng trả đúng hạn sẽ '
                '"pha loãng" các lần trễ cũ. Cần khoảng 6 tháng liên tiếp đúng hạn.'
            )

        if history < 36:
            tips.append(
                f'Lịch sử tín dụng {history} tháng — còn ngắn. '
                'Đừng đóng bất kỳ thẻ/tài khoản nào, dù ít dùng. '
                'Tuổi lịch sử tín dụng tăng thụ động theo thời gian.'
            )

        if inquiries > 3:
            tips.append(
                f'{inquiries} lần tra cứu — hạn chế nộp đơn vay mới 6 tháng tới. '
                'Nhiều inquiry làm ngân hàng nghĩ bạn đang "desperate" cần tiền.'
            )

        if invested < 100:
            tips.append(
                f'Đầu tư hàng tháng chỉ {invested:,.0f} USD — tăng lên ít nhất 10% lương. '
                'Không chỉ cải thiện hồ sơ tài chính mà còn tạo đệm dự phòng.'
            )

        # Tóm tắt rào cản
        if blockers:
            tips.append('─── TÓM TẮT: CẦN XỬ LÝ ĐỂ LÊN GOOD ───')
            tips.append(
                f'Rào cản chính hiện tại: **{" | ".join(blockers)}**. '
                'Giải quyết được 2/3 trong số này trong 3 tháng tới có thể đủ để lên Good.'
            )
        else:
            tips.append(
                'Không có rào cản rõ ràng — bạn đang ở gần ranh giới Good. '
                'Duy trì ổn định 2–3 tháng nữa, điểm có thể tự đẩy lên.'
            )

    else:  # Good
        tips.append('─── DUY TRÌ & TỐI ƯU HÓA ───')

        if util < 10:
            tips.append(
                f'Utilization {util:.0f}% — rất tốt. '
                'Lưu ý: nếu xuống 0% (không dùng thẻ) thì điểm có thể giảm nhẹ vì không có activity. '
                'Duy trì 5–15% là lý tưởng.'
            )
        else:
            tips.append(
                f'Tiếp tục giữ utilization dưới 30% (hiện {util:.0f}%) — đang tốt.'
            )

        if balance > 0:
            tips.append(
                f'Số dư cuối tháng {balance:,.0f} USD — có đệm tài chính tốt. '
                'Cân nhắc chuyển phần dư sang tài khoản tiết kiệm có lãi suất cao hơn.'
            )

        if invested < 200:
            tips.append(
                'Tăng đầu tư hàng tháng lên ít nhất 15–20% lương. '
                'Ở mức Good, bạn đủ điều kiện mở thêm tài khoản đầu tư (chứng khoán, quỹ mở).'
            )

        tips.append('─── SẢN PHẨM TÍN DỤNG PHÙ HỢP VỚI NHÓM GOOD ───')
        tips.append(
            '- Thẻ tín dụng hoàn tiền (cashback) hoặc tích điểm — lãi suất ưu đãi 10–12%/năm.\n'
            '- Vay mua nhà (Mortgage) — ngân hàng sẵn sàng duyệt với lãi suất tốt.\n'
            '- Vay tín chấp tiêu dùng — lãi suất thường 10–13%/năm thay vì 15–18% như Standard.'
        )

        if history < 60:
            tips.append(
                f'Lịch sử tín dụng {history} tháng — đang xây dựng tốt. '
                'Giữ nguyên các tài khoản cũ nhất, đừng đóng dù không dùng.'
            )

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
    'Annual_Income':              ('Thu nhập hàng năm',       'up',   '{:,.0f} USD'),
    'Monthly_Inhand_Salary':      ('Lương thực nhận',         'up',   '{:,.0f} USD'),
    'Monthly_Balance':            ('Số dư cuối tháng',        'up',   '{:,.0f} USD'),
    'Total_EMI_per_month':        ('EMI hàng tháng',          'down', '{:,.0f} USD'),
    'Amount_invested_monthly':    ('Đầu tư hàng tháng',       'up',   '{:,.0f} USD'),
    'Num_Bank_Accounts':          ('Số tài khoản NH',         'up',   '{:.0f}'),
    'Num_Credit_Card':            ('Số thẻ tín dụng',         'down', '{:.0f}'),
    'Num_of_Loan':                ('Số khoản vay',            'down', '{:.0f}'),
    'Interest_Rate':              ('Lãi suất TB',             'down', '{:.1f}%'),
    'Outstanding_Debt':           ('Nợ tồn đọng',             'down', '{:,.0f} USD'),
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

