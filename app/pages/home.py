"""Streamlit app — AI Credit Classification."""

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from config import (APP_SUBTITLE, APP_TITLE, DISCLAIMER,
                     N8N_WEBHOOK_URL, WEBHOOK_TIMEOUT)
from utils import (CLASS_COLORS, CLASS_LABELS_VI, derive_behavior,
                    diff_inputs, get_previous_prediction,
                    load_artifacts, predict, save_prediction, suggestions_for)

st.set_page_config(page_title='CreditAI — Đánh giá tín dụng', page_icon=None,
                   layout='wide', initial_sidebar_state='collapsed')

# ── Auth guard ───────────────────────────────────────────────────────────────
if st.session_state.get('role') not in ('customer', 'admin'):
    st.switch_page('login.py')

# ── Ẩn sidebar ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"]        { display: none !important; }
  [data-testid="collapsedControl"] { display: none !important; }
  #MainMenu, footer                { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── CSS toàn trang ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Font & background */
  html, body, [class*="css"] { font-family: 'Segoe UI', Arial, sans-serif; }
  .main { background: #f0f4f8; }

  /* Ẩn header mặc định Streamlit */
  header[data-testid="stHeader"] { background: transparent; }
  #MainMenu, footer { visibility: hidden; }

  /* Navbar top */
  .navbar {
    background: linear-gradient(135deg, #1a237e 0%, #1565c0 100%);
    padding: 16px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-radius: 0 0 16px 16px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(21,101,192,0.3);
  }
  .navbar-brand { color: white; font-size: 22px; font-weight: 700; letter-spacing: 1px; }
  .navbar-sub { color: rgba(255,255,255,0.75); font-size: 13px; margin-top: 2px; }

  /* Hero banner */
  .hero {
    background: linear-gradient(135deg, #1565c0 0%, #0d47a1 50%, #1a237e 100%);
    border-radius: 16px;
    padding: 40px;
    color: white;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(21,101,192,0.25);
  }
  .hero h1 { font-size: 32px; font-weight: 800; margin: 0 0 10px; }
  .hero p { font-size: 16px; opacity: 0.88; margin: 0; }

  /* Card wrapper */
  .card {
    background: white;
    border-radius: 14px;
    padding: 28px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    margin-bottom: 20px;
  }
  .card-title {
    font-size: 16px; font-weight: 700; color: #1565c0;
    border-left: 4px solid #1565c0;
    padding-left: 12px; margin-bottom: 12px;
    text-transform: uppercase; letter-spacing: 0.5px;
  }

  /* Preset buttons */
  div[data-testid="column"] button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
  }

  /* Result card */
  .result-badge {
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    color: white;
    box-shadow: 0 6px 24px rgba(0,0,0,0.15);
  }
  .result-badge .label { font-size: 13px; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; }
  .result-badge .value { font-size: 44px; font-weight: 800; margin: 8px 0 4px; }
  .result-badge .sub   { font-size: 15px; opacity: 0.85; }
  .result-badge .conf  { font-size: 13px; margin-top: 10px; opacity: 0.8; }

  /* Tip sections */
  .tip-section {
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    line-height: 1.7;
    font-size: 14px;
  }
  .tip-urgent   { background: #fff0f0; border-left: 4px solid #e53935; color: #333; }
  .tip-midterm  { background: #fff8e1; border-left: 4px solid #f9a825; color: #333; }
  .tip-longterm { background: #e3f2fd; border-left: 4px solid #1565c0; color: #333; }
  .tip-strength { background: #e8f5e9; border-left: 4px solid #2e7d32; color: #333; }
  .tip-header   { font-weight: 700; font-size: 13px; text-transform: uppercase;
                  letter-spacing: 0.5px; margin-bottom: 6px; }

  /* Divider */
  .section-divider {
    border: none; border-top: 2px solid #e8edf5; margin: 24px 0;
  }

  /* Disclaimer */
  .disclaimer {
    background: #e8edf5; border-radius: 10px; padding: 14px 18px;
    font-size: 12px; color: #546e7a; margin-top: 20px;
  }

  /* Required field asterisk */
  .req::after {
    content: " *";
    color: #e53935;
    font-weight: 700;
  }

  div[data-testid="stForm"] button[type="submit"] {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 14px !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(21,101,192,0.3) !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Helper: label có dấu * đỏ cho field bắt buộc ──────────────────────────
def req(label: str) -> str:
    """Trả về label kèm dấu * đỏ — dùng trong st.markdown trước widget."""
    return f'{label} <span style="color:#e53935;font-weight:700;">*</span>'

def req_label(col, label: str) -> None:
    """Hiển thị label có * đỏ phía trên widget trong column."""
    col.markdown(f'<p style="font-size:14px;margin-bottom:4px;color:#31333f;">{label} <span style="color:#e53935;font-weight:700;font-size:15px;">*</span></p>',
                 unsafe_allow_html=True)

# ── Load model (validate ngay, dừng nếu lỗi) ───────────────────────────────
try:
    load_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ── Navbar ──────────────────────────────────────────────────────────────────
_username = st.session_state.get('username', 'Khách hàng')
nav_col1, nav_col2 = st.columns([8, 1])
with nav_col1:
    st.markdown("""
    <div class="navbar">
      <div>
        <div class="navbar-brand">CreditAI</div>
        <div class="navbar-sub">Hệ thống đánh giá tín dụng thông minh</div>
      </div>
      <div style="color:rgba(255,255,255,0.7); font-size:13px;">
        Powered by Machine Learning
      </div>
    </div>
    """, unsafe_allow_html=True)
with nav_col2:
    st.markdown('<div style="margin-top:12px;">', unsafe_allow_html=True)
    if st.button('Đăng xuất', use_container_width=True, key='logout_home'):
        st.session_state.role = None
        st.session_state.username = None
        st.switch_page('login.py')
    st.markdown('</div>', unsafe_allow_html=True)
# ── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Đánh giá mức tín dụng cá nhân</h1>
  <p>Nhập thông tin tài chính — nhận kết quả phân loại tín dụng và gợi ý cải thiện ngay lập tức</p>
</div>
""", unsafe_allow_html=True)

# ── Presets & Session State ─────────────────────────────────────────────────
DEFAULTS = {
    'Age': 30, 'Occupation_vi': 'Kỹ sư', 'Credit_History_Months': 60,
    'Annual_Income': 50000.0, 'Monthly_Inhand_Salary': 3500.0, 'Monthly_Balance': 400.0,
    'Total_EMI_per_month': 150.0, 'Amount_invested_monthly': 100.0,
    'Payment_of_Min_Amount_vi': 'Có',
    'Num_Bank_Accounts': 3, 'Num_Credit_Card': 4, 'Num_of_Loan': 2,
    'Interest_Rate': 12, 'Outstanding_Debt': 1200.0, 'Credit_Utilization_Ratio': 30.0,
    'Delay_from_due_date': 5, 'Num_of_Delayed_Payment': 3, 'Num_Credit_Inquiries': 2,
    'Credit_Mix': 'Standard', 'Changed_Credit_Limit': 5.0,
    'Has_Credit_Builder_Loan': False, 'Has_Home_Equity_Loan': False,
    'Has_Mortgage_Loan': False, 'Has_Student_Loan': False,
    'Has_Payday_Loan': False,
    'customer_name': '', 'customer_email': '',
}

PRESETS = {
    'Poor': {
        'Age': 60, 'Occupation_vi': 'Thợ máy', 'Credit_History_Months': 3,
        'Annual_Income': 9000.0, 'Monthly_Inhand_Salary': 700.0, 'Monthly_Balance': 5.0,
        'Total_EMI_per_month': 800.0, 'Amount_invested_monthly': 0.0,
        'Payment_of_Min_Amount_vi': 'Có',
        'Num_Bank_Accounts': 0, 'Num_Credit_Card': 10, 'Num_of_Loan': 10,
        'Interest_Rate': 40, 'Outstanding_Debt': 50000.0, 'Credit_Utilization_Ratio': 100.0,
        'Delay_from_due_date': 60, 'Num_of_Delayed_Payment': 50, 'Num_Credit_Inquiries': 30,
        'Credit_Mix': 'Bad', 'Changed_Credit_Limit': 30.0,
        'Has_Credit_Builder_Loan': False, 'Has_Home_Equity_Loan': False,
        'Has_Mortgage_Loan': False, 'Has_Student_Loan': False, 'Has_Payday_Loan': True,
    },
    'Standard': DEFAULTS,
    'Good': {
        'Age': 35, 'Occupation_vi': 'Kế toán', 'Credit_History_Months': 60,
        'Annual_Income': 55000.0, 'Monthly_Inhand_Salary': 4500.0, 'Monthly_Balance': 700.0,
        'Total_EMI_per_month': 300.0, 'Amount_invested_monthly': 250.0,
        'Payment_of_Min_Amount_vi': 'Không',
        'Num_Bank_Accounts': 2, 'Num_Credit_Card': 4, 'Num_of_Loan': 2,
        'Interest_Rate': 11, 'Outstanding_Debt': 1500.0, 'Credit_Utilization_Ratio': 28.0,
        'Delay_from_due_date': 4, 'Num_of_Delayed_Payment': 1, 'Num_Credit_Inquiries': 2,
        'Credit_Mix': 'Good', 'Changed_Credit_Limit': 6.0,
        'Has_Credit_Builder_Loan': False, 'Has_Home_Equity_Loan': True,
        'Has_Mortgage_Loan': True, 'Has_Student_Loan': False, 'Has_Payday_Loan': False,
    },
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v

def reset_form():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

# ── Preset buttons ───────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">Tải nhanh hồ sơ mẫu</div>', unsafe_allow_html=True)
pc1, pc2, pc3, pc4 = st.columns(4)
pc1.button('Hồ sơ Kém (Poor)',     on_click=apply_preset, args=('Poor',),     use_container_width=True)
pc2.button('Hồ sơ Trung bình',     on_click=apply_preset, args=('Standard',), use_container_width=True)
pc3.button('Hồ sơ Tốt (Good)',     on_click=apply_preset, args=('Good',),     use_container_width=True)
pc4.button('Đặt lại mặc định',     on_click=reset_form,                       use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Form ─────────────────────────────────────────────────────────────────────
with st.form('credit_form'):

    # Section 1
    st.markdown('<div class="card"><div class="card-title">Thông tin cá nhân</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    Age = c1.number_input('Tuổi', min_value=18, max_value=100, key='Age')
    _occ_map = {
        'Kế toán': 'Accountant', 'Kiến trúc sư': 'Architect',
        'Lập trình viên': 'Developer', 'Bác sĩ': 'Doctor',
        'Kỹ sư': 'Engineer', 'Doanh nhân': 'Entrepreneur',
        'Nhà báo': 'Journalist', 'Luật sư': 'Lawyer',
        'Quản lý': 'Manager', 'Thợ máy': 'Mechanic',
        'Truyền thông': 'Media_Manager', 'Nhạc sĩ': 'Musician',
        'Nhà khoa học': 'Scientist', 'Giáo viên': 'Teacher', 'Nhà văn': 'Writer',
        'Khác': 'Engineer',
    }
    _occ_vi = c2.selectbox('Nghề nghiệp', list(_occ_map.keys()), key='Occupation_vi')
    Occupation = _occ_map[_occ_vi]
    Credit_History_Months = c3.number_input('Lịch sử tín dụng (tháng)', min_value=0, max_value=500, key='Credit_History_Months')
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 2
    st.markdown('<div class="card"><div class="card-title">Thu nhập và Chi tiêu</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    Annual_Income          = c1.number_input('Thu nhập hàng năm (USD) *',       min_value=0.0, step=1000.0, key='Annual_Income')
    Monthly_Inhand_Salary  = c2.number_input('Lương thực nhận / tháng (USD) *', min_value=0.0, step=100.0,  key='Monthly_Inhand_Salary')
    Monthly_Balance        = c3.number_input('Số dư cuối tháng (USD)',          min_value=0.0, step=50.0,   key='Monthly_Balance')
    c1, c2, c3 = st.columns(3)
    Total_EMI_per_month        = c1.number_input('Trả nợ EMI / tháng (USD) *',      min_value=0.0, step=10.0,  key='Total_EMI_per_month')
    Amount_invested_monthly    = c2.number_input('Đầu tư / tháng (USD)',          min_value=0.0, step=10.0,  key='Amount_invested_monthly')
    _pmt_map = {'Có': 'Yes', 'Không': 'No', 'Không xác định': 'NM'}
    _pmt_vi  = c3.selectbox('Trả khoản tối thiểu hàng tháng?', list(_pmt_map.keys()), key='Payment_of_Min_Amount_vi')
    Payment_of_Min_Amount = _pmt_map[_pmt_vi]
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 3
    st.markdown('<div class="card"><div class="card-title">Tình trạng tín dụng</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    Num_Bank_Accounts  = c1.number_input('Số tài khoản ngân hàng',      min_value=0, max_value=20, key='Num_Bank_Accounts')
    Num_Credit_Card    = c2.number_input('Số thẻ tín dụng',              min_value=0, max_value=20, key='Num_Credit_Card')
    Num_of_Loan        = c3.number_input('Số khoản vay hiện tại',        min_value=0, max_value=20, key='Num_of_Loan')
    c1, c2, c3 = st.columns(3)
    Interest_Rate      = c1.number_input('Lãi suất trung bình (%) *',      min_value=0, max_value=40, key='Interest_Rate')
    Outstanding_Debt   = c2.number_input('Nợ tồn đọng (USD) *',            min_value=0.0, step=100.0, key='Outstanding_Debt')
    Credit_Utilization_Ratio = c3.slider('Tỷ lệ sử dụng tín dụng (%) *',  0.0, 100.0, key='Credit_Utilization_Ratio')
    c1, c2, c3 = st.columns(3)
    Delay_from_due_date    = c1.number_input('Số ngày trả trễ (TB)',         min_value=0, max_value=60, key='Delay_from_due_date')
    Num_of_Delayed_Payment = c2.number_input('Số lần trả trễ',               min_value=0, max_value=50, key='Num_of_Delayed_Payment')
    Num_Credit_Inquiries   = c3.number_input('Số lần bị tra cứu tín dụng',   min_value=0, max_value=30, key='Num_Credit_Inquiries')
    c1, c2 = st.columns(2)
    Credit_Mix           = c1.selectbox('Chất lượng tổ hợp tín dụng', ['Bad', 'Standard', 'Good'], key='Credit_Mix')
    Changed_Credit_Limit = c2.number_input('Thay đổi hạn mức tín dụng', step=0.5, key='Changed_Credit_Limit')
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 4
    st.markdown('<div class="card"><div class="card-title">Loại khoản vay đang có</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    Has_Credit_Builder_Loan = int(c1.checkbox('Vay xây dựng tín dụng',      key='Has_Credit_Builder_Loan'))
    Has_Home_Equity_Loan    = int(c2.checkbox('Vay thế chấp nhà',            key='Has_Home_Equity_Loan'))
    Has_Mortgage_Loan       = int(c3.checkbox('Vay mua nhà',                 key='Has_Mortgage_Loan'))
    Has_Student_Loan        = int(c4.checkbox('Vay sinh viên',               key='Has_Student_Loan'))
    Has_Payday_Loan         = int(c5.checkbox('Vay ngắn hạn (Payday)',       key='Has_Payday_Loan'))
    Num_Loan_Types = sum([Has_Credit_Builder_Loan, Has_Home_Equity_Loan,
                          Has_Mortgage_Loan, Has_Student_Loan, Has_Payday_Loan])
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 5 — Thông tin liên hệ
    st.markdown('<div class="card"><div class="card-title">Thông tin liên hệ (tùy chọn — để nhận kết quả qua email)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    customer_name  = c1.text_input('Họ tên',  key='customer_name')
    customer_email = c2.text_input('Email',   key='customer_email')
    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button('Phân tích tín dụng ngay', use_container_width=True)


# ── Xử lý kết quả ───────────────────────────────────────────────────────────
if submitted:
    _annual = Annual_Income if Annual_Income > 0 else 1
    _salary = Monthly_Inhand_Salary if Monthly_Inhand_Salary > 0 else 1
    _cc     = Num_Credit_Card    if Num_Credit_Card    > 0 else 1
    _bank   = Num_Bank_Accounts  if Num_Bank_Accounts  > 0 else 1
    _loan   = Num_of_Loan        if Num_of_Loan        > 0 else 1

    Spending_Level, Payment_Value = derive_behavior({
        'Monthly_Inhand_Salary':    Monthly_Inhand_Salary,
        'Total_EMI_per_month':      Total_EMI_per_month,
        'Amount_invested_monthly':  Amount_invested_monthly,
        'Outstanding_Debt':         Outstanding_Debt,
    })

    user_inputs = {
        'Age': Age, 'Occupation': Occupation,
        'Annual_Income': Annual_Income, 'Monthly_Inhand_Salary': Monthly_Inhand_Salary,
        'Monthly_Balance': Monthly_Balance, 'Total_EMI_per_month': Total_EMI_per_month,
        'Amount_invested_monthly': Amount_invested_monthly,
        'Payment_of_Min_Amount': Payment_of_Min_Amount,
        'Num_Bank_Accounts': Num_Bank_Accounts, 'Num_Credit_Card': Num_Credit_Card,
        'Num_of_Loan': Num_of_Loan, 'Interest_Rate': Interest_Rate,
        'Outstanding_Debt': Outstanding_Debt,
        'Credit_Utilization_Ratio': Credit_Utilization_Ratio,
        'Delay_from_due_date': Delay_from_due_date,
        'Num_of_Delayed_Payment': Num_of_Delayed_Payment,
        'Num_Credit_Inquiries': Num_Credit_Inquiries,
        'Credit_Mix': Credit_Mix, 'Changed_Credit_Limit': Changed_Credit_Limit,
        'Credit_History_Months': Credit_History_Months,
        'Spending_Level': Spending_Level, 'Payment_Value': Payment_Value,
        'Num_Loan_Types': Num_Loan_Types,
        'Has_Credit_Builder_Loan': Has_Credit_Builder_Loan,
        'Has_Home_Equity_Loan':    Has_Home_Equity_Loan,
        'Has_Mortgage_Loan':       Has_Mortgage_Loan,
        'Has_Student_Loan':        Has_Student_Loan,
        'Has_Payday_Loan':         Has_Payday_Loan,
        'Debt_to_Income_Annual':   Outstanding_Debt / _annual,
        'EMI_to_Salary_Ratio':     Total_EMI_per_month / _salary,
        'Investment_Rate':         Amount_invested_monthly / _salary,
        'CC_per_Bank':             Num_Credit_Card / _bank,
        'Loan_per_CC':             Num_of_Loan / _cc,
        'Age_Start_Credit':        Age - Credit_History_Months / 12,
        'Debt_per_Loan':           Outstanding_Debt / _loan,
        'Balance_to_Salary':       Monthly_Balance / _salary,
    }

    with st.spinner('Đang phân tích...'):
        result = predict(user_inputs)

    prediction_timestamp = datetime.now().isoformat()
    previous = None
    try:
        previous = get_previous_prediction(customer_email)
        save_prediction(email=customer_email, customer_name=customer_name,
                        timestamp=prediction_timestamp, inputs=user_inputs, result=result)
    except Exception:
        pass

    # ── Kết quả chính ────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col_badge, col_chart = st.columns([1, 2])
    with col_badge:
        bg = result['color']
        cls_vi = result['predicted_class_vi']
        cls_en = result['predicted_class']
        conf   = result['confidence'] * 100
        st.markdown(f"""
        <div class="result-badge" style="background: linear-gradient(135deg, {bg}, {bg}cc);">
          <div class="label">Kết quả phân loại</div>
          <div class="value">{cls_vi}</div>
          <div class="sub">({cls_en})</div>
          <div class="conf">Độ tin cậy: {conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        proba_df = pd.DataFrame({
            'Nhóm':    list(result['proba'].keys()),
            'Xac suat': list(result['proba'].values()),
        })
        fig = go.Figure(go.Bar(
            x=proba_df['Nhóm'], y=proba_df['Xac suat'],
            marker_color=[CLASS_COLORS[c] for c in proba_df['Nhóm']],
            text=[f'{p:.1%}' for p in proba_df['Xac suat']],
            textposition='outside',
            marker_line_width=0,
        ))
        fig.update_layout(
            title=dict(text='Xác suất theo từng nhóm tín dụng', font=dict(size=15, color='#1565c0')),
            yaxis=dict(range=[0, 1.1], tickformat='.0%', gridcolor='#f0f0f0'),
            xaxis=dict(tickfont=dict(size=13, color='#333')),
            plot_bgcolor='white', paper_bgcolor='white',
            height=300, margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Gợi ý cải thiện ──────────────────────────────────────────────────────
    tips = suggestions_for(result, user_inputs, previous=previous)

    # Lọc bỏ các dòng header phân section (dòng chứa ───)
    clean_tips = [t for t in tips if '───' not in t]

    # Render tất cả trong 1 box duy nhất
    tips_html = ''.join(
        f'<p style="margin:8px 0; line-height:1.7; color:#333;">{t.replace("**", "<b>", 1).replace("**", "</b>", 1)}</p>'
        for t in clean_tips
    )
    st.markdown(
        f'<div class="card">'
        f'<div class="card-title">Gợi ý cải thiện</div>'
        f'{tips_html}'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── So sánh lần trước ────────────────────────────────────────────────────
    if previous:
        from datetime import datetime as _dt
        try:
            prev_dt  = _dt.fromisoformat(previous['timestamp'])
            days_ago = (_dt.now() - prev_dt).days
            date_str = prev_dt.strftime('%d/%m/%Y %H:%M')
        except Exception:
            days_ago = None
            date_str = str(previous['timestamp'])

        days_label = f'({days_ago} ngày trước)' if days_ago is not None else ''

        st.markdown(f"""
        <div class="card">
          <div class="card-title">So sánh với lần tra cứu trước</div>
          <p style="color:#546e7a;font-size:13px;margin-bottom:16px;">
            Lần tra cứu ngày <b>{date_str}</b> {days_label}
          </p>
        </div>
        """, unsafe_allow_html=True)

        d_poor = result['proba']['Poor']     - previous['proba']['Poor']
        d_std  = result['proba']['Standard'] - previous['proba']['Standard']
        d_good = result['proba']['Good']     - previous['proba']['Good']

        prev_cls = previous['predicted_class']
        curr_cls = result['predicted_class']
        class_ranks = {'Poor': 0, 'Standard': 1, 'Good': 2}
        cls_delta = class_ranks.get(curr_cls, 1) - class_ranks.get(prev_cls, 1)

        cc1, cc2, cc3, cc4 = st.columns(4)
        indicator = 'Cải thiện' if cls_delta > 0 else ('Xấu đi' if cls_delta < 0 else 'Giữ nguyên')
        cc1.metric('Nhóm phân loại', f'{prev_cls} → {curr_cls}', delta=indicator,
                   delta_color='normal' if cls_delta >= 0 else 'inverse')
        cc2.metric('P(Poor)',    f'{result["proba"]["Poor"]*100:.1f}%',
                   delta=f'{d_poor*100:+.1f}%', delta_color='inverse')
        cc3.metric('P(Standard)', f'{result["proba"]["Standard"]*100:.1f}%',
                   delta=f'{d_std*100:+.1f}%',  delta_color='off')
        cc4.metric('P(Good)',   f'{result["proba"]["Good"]*100:.1f}%',
                   delta=f'{d_good*100:+.1f}%', delta_color='normal')

        if cls_delta > 0:
            st.success('Chúc mừng! Nhóm tín dụng của bạn đã được cải thiện. Tiếp tục duy trì!')
        elif cls_delta < 0:
            st.warning('Nhóm tín dụng đã hạ. Xem gợi ý ở trên để quay lại quỹ đạo.')
        elif d_good > 0.05:
            st.info('Chưa đổi nhóm nhưng xác suất Good đang tăng — bạn đang đi đúng hướng.')

        with st.expander('Xem chi tiết thay đổi từng yếu tố', expanded=False):
            diffs = diff_inputs(previous['inputs'], user_inputs, top_n=8)
            if not diffs:
                st.info('Không có yếu tố nào thay đổi đáng kể giữa 2 lần.')
            else:
                chart_df = pd.DataFrame([{
                    'Feature': d['label'],
                    'Delta':   d['delta_pct'] if d.get('delta_pct') is not None
                               else (d['delta'] * 50 if d['is_categorical'] else d['delta']),
                    'Impact':  d['impact'],
                    'Prev':    d['prev'], 'Curr': d['curr'],
                    'IsCat':   d['is_categorical'], 'Format': d.get('format'),
                } for d in diffs]).sort_values('Delta')

                fig_diff = go.Figure(go.Bar(
                    y=chart_df['Feature'], x=chart_df['Delta'], orientation='h',
                    marker_color=['#2e7d32' if i == 'good' else '#c62828' for i in chart_df['Impact']],
                    text=[(f"{r['Prev']} → {r['Curr']}" if r['IsCat']
                           else (r['Format'] or '{:.1f}').format(r['Prev']) + ' → ' +
                                (r['Format'] or '{:.1f}').format(r['Curr']))
                          for _, r in chart_df.iterrows()],
                    textposition='outside',
                ))
                fig_diff.update_layout(
                    height=max(280, 40 + 40 * len(chart_df)),
                    margin=dict(l=20, r=120, t=10, b=10),
                    xaxis_title='Mức độ thay đổi (%)',
                    plot_bgcolor='white', paper_bgcolor='white',
                    showlegend=False,
                )
                fig_diff.add_vline(x=0, line_width=1, line_color='#ccc')
                st.plotly_chart(fig_diff, use_container_width=True)

    # ── Gửi webhook n8n ──────────────────────────────────────────────────────
    if N8N_WEBHOOK_URL:
        import numpy as np
        def _to_native(v):
            if isinstance(v, np.integer): return int(v)
            if isinstance(v, np.floating): return float(v)
            if isinstance(v, np.ndarray): return v.tolist()
            return v

        payload = {
            'timestamp':         datetime.now().isoformat(),
            'customer_name':     customer_name or 'Ẩn danh',
            'customer_email':    customer_email or '',
            'predicted_class':   result['predicted_class'],
            'predicted_class_vi': result['predicted_class_vi'],
            'proba':             {k: round(float(v), 4) for k, v in result['proba'].items()},
            'inputs':            {k: _to_native(v) for k, v in user_inputs.items()},
            'previous':          ({
                'timestamp':       previous['timestamp'],
                'predicted_class': previous['predicted_class'],
                'proba':           {k: round(float(v), 4) for k, v in previous['proba'].items()},
            } if previous else None),
        }
        try:
            r = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=WEBHOOK_TIMEOUT)
            if r.ok:
                st.toast('Kết quả đã được lưu và gửi email thành công.')
            else:
                st.warning(f'Webhook trả về status {r.status_code}. Kết quả chưa được lưu.')
        except requests.RequestException as e:
            st.warning(f'Không kết nối được n8n: {e}')

    # ── Footer disclaimer ────────────────────────────────────────────────────
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)
