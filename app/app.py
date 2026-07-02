"""Streamlit app cho AI Credit Scoring."""

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from config import (APP_SUBTITLE, APP_TITLE, DISCLAIMER, FICO_BINS,
                     N8N_WEBHOOK_URL, WEBHOOK_TIMEOUT)
from utils import (FICO_ANCHORS, derive_behavior, get_shap_values,
                    load_artifacts, predict, suggestions_for)


st.set_page_config(page_title=APP_TITLE, page_icon='💳', layout='wide')


# ============ Sidebar ============
with st.sidebar:
    st.title('💳 ' + APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.markdown('---')

    try:
        _, schema = load_artifacts()
        metrics = schema.get('metrics', {})
        st.subheader('Thông số mô hình')
        c1, c2 = st.columns(2)
        c1.metric('F1 macro', f"{metrics.get('f1_macro', 0):.3f}")
        c2.metric('AUC (ovr)', f"{metrics.get('auc_ovr', 0):.3f}")
        st.metric('Accuracy', f"{metrics.get('accuracy', 0):.3f}")
        st.caption(f"Model: **{schema.get('best_model', 'N/A')}**")
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    st.markdown('---')
    st.info(DISCLAIMER)


# ============ Header ============
st.title('Credit Scoring — Chấm điểm tín dụng cá nhân')
st.write(
    'Nhập thông tin bên dưới để nhận điểm tín dụng dự đoán, phân loại theo thang FICO '
    'và gợi ý cải thiện.'
)

# ============ Presets & Session State ============
DEFAULTS = {
    'Age': 30, 'Occupation_vi': 'Kỹ sư', 'Credit_History_Months': 60,
    'Annual_Income': 50000.0, 'Monthly_Inhand_Salary': 3500.0, 'Monthly_Balance': 400.0,
    'Total_EMI_per_month': 150.0, 'Amount_invested_monthly': 100.0,
    'Payment_of_Min_Amount_vi': 'Có',
    'Num_Bank_Accounts': 3, 'Num_Credit_Card': 4, 'Num_of_Loan': 2,
    'Interest_Rate': 12, 'Outstanding_Debt': 1200.0, 'Credit_Utilization_Ratio': 30.0,
    'Delay_from_due_date': 5, 'Num_of_Delayed_Payment': 3, 'Num_Credit_Inquiries': 2,
    'Credit_Mix': 'Standard', 'Changed_Credit_Limit': 5.0,
    'Has_Credit_Builder_Loan': False, 'Has_Personal_Loan': False,
    'Has_Debt_Consolidation_Loan': False, 'Has_Student_Loan': False,
    'Has_Payday_Loan': False,
    'customer_name': '', 'customer_email': '',
}

PRESETS = {
    'Poor': {
        'Age': 22, 'Occupation_vi': 'Thợ máy', 'Credit_History_Months': 6,
        'Annual_Income': 14000.0, 'Monthly_Inhand_Salary': 1100.0, 'Monthly_Balance': 30.0,
        'Total_EMI_per_month': 650.0, 'Amount_invested_monthly': 0.0,
        'Payment_of_Min_Amount_vi': 'Có',
        'Num_Bank_Accounts': 1, 'Num_Credit_Card': 8, 'Num_of_Loan': 8,
        'Interest_Rate': 32, 'Outstanding_Debt': 12000.0, 'Credit_Utilization_Ratio': 96.0,
        'Delay_from_due_date': 45, 'Num_of_Delayed_Payment': 28, 'Num_Credit_Inquiries': 15,
        'Credit_Mix': 'Bad', 'Changed_Credit_Limit': 22.0,
        'Has_Credit_Builder_Loan': False, 'Has_Personal_Loan': True,
        'Has_Debt_Consolidation_Loan': True, 'Has_Student_Loan': False,
        'Has_Payday_Loan': True,
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
        'Has_Credit_Builder_Loan': False, 'Has_Personal_Loan': True,
        'Has_Debt_Consolidation_Loan': False, 'Has_Student_Loan': False,
        'Has_Payday_Loan': False,
    },
    'Exceptional': {
        'Age': 48, 'Occupation_vi': 'Bác sĩ', 'Credit_History_Months': 240,
        'Annual_Income': 150000.0, 'Monthly_Inhand_Salary': 11500.0, 'Monthly_Balance': 3500.0,
        'Total_EMI_per_month': 50.0, 'Amount_invested_monthly': 2500.0,
        'Payment_of_Min_Amount_vi': 'Không',
        'Num_Bank_Accounts': 4, 'Num_Credit_Card': 2, 'Num_of_Loan': 1,
        'Interest_Rate': 5, 'Outstanding_Debt': 100.0, 'Credit_Utilization_Ratio': 5.0,
        'Delay_from_due_date': 0, 'Num_of_Delayed_Payment': 0, 'Num_Credit_Inquiries': 0,
        'Credit_Mix': 'Good', 'Changed_Credit_Limit': 2.0,
        'Has_Credit_Builder_Loan': False, 'Has_Personal_Loan': True,
        'Has_Debt_Consolidation_Loan': False, 'Has_Student_Loan': False,
        'Has_Payday_Loan': False,
    },
}

# Init session state với default
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


def reset_form():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


st.markdown('##### ⚡ Load nhanh case demo')
pc1, pc2, pc3, pc4, pc5 = st.columns(5)
pc1.button('🔴 Poor', on_click=apply_preset, args=('Poor',),
            use_container_width=True, help='Load case điểm tín dụng thấp')
pc2.button('🟠 Standard', on_click=apply_preset, args=('Standard',),
            use_container_width=True, help='Load case điểm tín dụng trung bình')
pc3.button('🟢 Good', on_click=apply_preset, args=('Good',),
            use_container_width=True, help='Load case điểm tín dụng tốt')
pc4.button('💚 Exceptional', on_click=apply_preset, args=('Exceptional',),
            use_container_width=True, help='Load case điểm tín dụng xuất sắc')
pc5.button('🔄 Reset', on_click=reset_form,
            use_container_width=True, help='Trở về giá trị mặc định')
st.markdown('---')

# ============ Form ============
with st.form('credit_form'):
    st.subheader('1. Thông tin cá nhân')
    c1, c2, c3 = st.columns(3)
    Age = c1.number_input('Tuổi', min_value=18, max_value=100, key='Age')
    _occ_map = {
        'Kế toán': 'Accountant', 'Kiến trúc sư': 'Architect',
        'Lập trình viên': 'Developer', 'Bác sĩ': 'Doctor',
        'Kỹ sư': 'Engineer', 'Doanh nhân': 'Entrepreneur',
        'Nhà báo': 'Journalist', 'Luật sư': 'Lawyer',
        'Quản lý': 'Manager', 'Thợ máy': 'Mechanic',
        'Truyền thông': 'Media_Manager', 'Nhạc sĩ': 'Musician',
        'Nhà khoa học': 'Scientist', 'Giáo viên': 'Teacher',
        'Nhà văn': 'Writer',
    }
    _occ_vi = c2.selectbox('Nghề nghiệp', list(_occ_map.keys()), key='Occupation_vi')
    Occupation = _occ_map[_occ_vi]
    Credit_History_Months = c3.number_input('Lịch sử tín dụng (tháng)',
                                             min_value=0, max_value=500,
                                             key='Credit_History_Months')

    st.subheader('2. Thu nhập & Chi tiêu')
    c1, c2, c3 = st.columns(3)
    Annual_Income = c1.number_input('Thu nhập hàng năm (USD)',
                                     min_value=0.0, step=1000.0, key='Annual_Income')
    Monthly_Inhand_Salary = c2.number_input('Lương thực nhận / tháng (USD)',
                                              min_value=0.0, step=100.0,
                                              key='Monthly_Inhand_Salary')
    Monthly_Balance = c3.number_input('Số dư cuối tháng (USD)',
                                       min_value=0.0, step=50.0, key='Monthly_Balance')

    c1, c2, c3 = st.columns(3)
    Total_EMI_per_month = c1.number_input('Trả nợ EMI / tháng (USD)',
                                            min_value=0.0, step=10.0,
                                            key='Total_EMI_per_month')
    Amount_invested_monthly = c2.number_input('Đầu tư / tháng (USD)',
                                                min_value=0.0, step=10.0,
                                                key='Amount_invested_monthly')
    _pmt_map = {'Có': 'Yes', 'Không': 'No', 'Không xác định': 'NM'}
    _pmt_vi = c3.selectbox('Trả khoản tối thiểu hàng tháng?',
                            list(_pmt_map.keys()), key='Payment_of_Min_Amount_vi')
    Payment_of_Min_Amount = _pmt_map[_pmt_vi]

    st.subheader('3. Tình trạng tín dụng')
    c1, c2, c3 = st.columns(3)
    Num_Bank_Accounts = c1.number_input('Số tài khoản ngân hàng',
                                          min_value=0, max_value=20,
                                          key='Num_Bank_Accounts')
    Num_Credit_Card = c2.number_input('Số thẻ tín dụng',
                                        min_value=0, max_value=20, key='Num_Credit_Card')
    Num_of_Loan = c3.number_input('Số khoản vay hiện tại',
                                    min_value=0, max_value=20, key='Num_of_Loan')

    c1, c2, c3 = st.columns(3)
    Interest_Rate = c1.number_input('Lãi suất trung bình (%)',
                                     min_value=0, max_value=40, key='Interest_Rate')
    Outstanding_Debt = c2.number_input('Nợ tồn đọng (USD)',
                                        min_value=0.0, step=100.0, key='Outstanding_Debt')
    Credit_Utilization_Ratio = c3.slider('Tỷ lệ sử dụng tín dụng (%)',
                                           0.0, 100.0, key='Credit_Utilization_Ratio')

    c1, c2, c3 = st.columns(3)
    Delay_from_due_date = c1.number_input('Số ngày trả trễ (TB)',
                                            min_value=0, max_value=60,
                                            key='Delay_from_due_date')
    Num_of_Delayed_Payment = c2.number_input('Số lần trả trễ',
                                                min_value=0, max_value=50,
                                                key='Num_of_Delayed_Payment')
    Num_Credit_Inquiries = c3.number_input('Số lần bị tra cứu tín dụng',
                                             min_value=0, max_value=30,
                                             key='Num_Credit_Inquiries')

    c1, c2 = st.columns(2)
    Credit_Mix = c1.selectbox('Chất lượng tổ hợp tín dụng',
                                ['Bad', 'Standard', 'Good'], key='Credit_Mix')
    Changed_Credit_Limit = c2.number_input('Thay đổi hạn mức tín dụng',
                                             step=0.5, key='Changed_Credit_Limit')

    st.subheader('4. Loại khoản vay đang có')
    c1, c2, c3, c4, c5 = st.columns(5)
    Has_Credit_Builder_Loan = int(c1.checkbox('Vay xây dựng tín dụng',
                                               key='Has_Credit_Builder_Loan'))
    Has_Personal_Loan = int(c2.checkbox('Vay cá nhân', key='Has_Personal_Loan'))
    Has_Debt_Consolidation_Loan = int(c3.checkbox('Vay hợp nhất nợ',
                                                   key='Has_Debt_Consolidation_Loan'))
    Has_Student_Loan = int(c4.checkbox('Vay sinh viên', key='Has_Student_Loan'))
    Has_Payday_Loan = int(c5.checkbox('Vay ngắn hạn', key='Has_Payday_Loan'))
    Num_Loan_Types = sum([Has_Credit_Builder_Loan, Has_Personal_Loan,
                           Has_Debt_Consolidation_Loan, Has_Student_Loan,
                           Has_Payday_Loan])

    with st.expander('📝 Thông tin liên hệ (tùy chọn — để gửi email kết quả)'):
        c1, c2 = st.columns(2)
        customer_name = c1.text_input('Họ tên', key='customer_name')
        customer_email = c2.text_input('Email', key='customer_email')

    submitted = st.form_submit_button('🔍 Dự đoán điểm tín dụng', use_container_width=True)


# ============ Kết quả ============
if submitted:
    # Tính các ratio features
    _annual = Annual_Income if Annual_Income > 0 else 1
    _salary = Monthly_Inhand_Salary if Monthly_Inhand_Salary > 0 else 1
    _cc = Num_Credit_Card if Num_Credit_Card > 0 else 1
    _bank = Num_Bank_Accounts if Num_Bank_Accounts > 0 else 1
    _loan = Num_of_Loan if Num_of_Loan > 0 else 1

    # Suy ra hành vi từ số liệu tài chính (tránh dropdown chủ quan)
    _base_inputs = {
        'Monthly_Inhand_Salary': Monthly_Inhand_Salary,
        'Total_EMI_per_month': Total_EMI_per_month,
        'Amount_invested_monthly': Amount_invested_monthly,
        'Outstanding_Debt': Outstanding_Debt,
    }
    Spending_Level, Payment_Value = derive_behavior(_base_inputs)

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
        'Has_Personal_Loan': Has_Personal_Loan,
        'Has_Debt_Consolidation_Loan': Has_Debt_Consolidation_Loan,
        'Has_Student_Loan': Has_Student_Loan,
        'Has_Payday_Loan': Has_Payday_Loan,
        # Ratios
        'Debt_to_Income_Annual': Outstanding_Debt / _annual,
        'EMI_to_Salary_Ratio': Total_EMI_per_month / _salary,
        'Investment_Rate': Amount_invested_monthly / _salary,
        'CC_per_Bank': Num_Credit_Card / _bank,
        'Loan_per_CC': Num_of_Loan / _cc,
        'Age_Start_Credit': Age - Credit_History_Months / 12,
        'Debt_per_Loan': Outstanding_Debt / _loan,
        'Balance_to_Salary': Monthly_Balance / _salary,
    }

    with st.spinner('Đang dự đoán...'):
        result = predict(user_inputs)

    st.success('Dự đoán hoàn tất.')
    st.markdown('---')

    # ---- Kết quả chính (tối giản cho end-user) ----
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.metric('FICO Score', f"{result['fico_score']:.0f}",
                    help='Điểm tín dụng theo thang FICO chuẩn 300-850, dùng cho quyết định tín dụng.')
    with c2:
        st.markdown(
            f"<div style='padding:20px;border-radius:8px;background:{result['rating_color']};"
            f"color:white;text-align:center;font-size:22px;font-weight:bold'>"
            f"{result['rating']}</div>",
            unsafe_allow_html=True
        )
    with c3:
        def hex_to_rgba(hex_color, alpha=0.25):
            h = hex_color.lstrip('#')
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f'rgba({r},{g},{b},{alpha})'

        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=result['fico_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [300, 850]},
                'bar': {'color': result['rating_color']},
                'steps': [
                    {'range': [lo, hi], 'color': hex_to_rgba(color)}
                    for lo, hi, _, color in FICO_BINS
                ],
                'threshold': {'line': {'color': 'black', 'width': 3},
                              'value': result['fico_score']}
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # ---- Gợi ý (main view) ----
    st.subheader('💡 Gợi ý cải thiện')
    for tip in suggestions_for(result, user_inputs):
        st.write('•', tip)

    st.markdown('---')

    # ---- Expander: chi tiết kỹ thuật (progressive disclosure) ----
    with st.expander('🔬 Chi tiết kỹ thuật — Điểm này được tính thế nào?',
                       expanded=False):

        # === 1. Contribution chart ===
        st.markdown('#### 📊 Cách tính điểm FICO')
        st.caption(
            'Điểm FICO là **tổng đóng góp** từ 3 nhóm phân loại của mô hình. '
            'Mỗi nhóm có điểm neo (anchor) riêng, được nhân với xác suất tương ứng.'
        )
        anchor_poor, anchor_std, anchor_good = FICO_ANCHORS
        p_poor = result['proba']['Poor']
        p_std = result['proba']['Standard']
        p_good = result['proba']['Good']
        contrib_poor = p_poor * anchor_poor
        contrib_std = p_std * anchor_std
        contrib_good = p_good * anchor_good
        total_score = contrib_poor + contrib_std + contrib_good

        fig_contrib = go.Figure()
        fig_contrib.add_trace(go.Bar(
            y=['FICO Score'], x=[contrib_poor], name='Nhóm Poor',
            orientation='h', marker_color='#d9534f',
            text=f'{contrib_poor:.0f} điểm<br>({p_poor*100:.1f}% × {anchor_poor:.0f})',
            textposition='inside', hovertemplate='%{text}<extra></extra>'
        ))
        fig_contrib.add_trace(go.Bar(
            y=['FICO Score'], x=[contrib_std], name='Nhóm Standard',
            orientation='h', marker_color='#6c8ebf',
            text=f'{contrib_std:.0f} điểm<br>({p_std*100:.1f}% × {anchor_std:.0f})',
            textposition='inside', hovertemplate='%{text}<extra></extra>'
        ))
        fig_contrib.add_trace(go.Bar(
            y=['FICO Score'], x=[contrib_good], name='Nhóm Good',
            orientation='h', marker_color='#5cb85c',
            text=f'{contrib_good:.0f} điểm<br>({p_good*100:.1f}% × {anchor_good:.0f})',
            textposition='inside', hovertemplate='%{text}<extra></extra>'
        ))
        fig_contrib.update_layout(
            barmode='stack', height=180,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(range=[0, 850], title='Điểm đóng góp'),
            legend=dict(orientation='h', y=-0.3),
            showlegend=True,
        )
        st.plotly_chart(fig_contrib, use_container_width=True)
        st.info(
            f'**Công thức:** FICO = {contrib_poor:.0f} + {contrib_std:.0f} + '
            f'{contrib_good:.0f} = **{total_score:.0f}**'
        )

        # === 2. Model confidence (probability breakdown) ===
        st.markdown('#### 🎯 Mức độ tin cậy của mô hình')
        st.caption(
            'Model dự đoán khách hàng có bao nhiêu **khả năng** thuộc từng nhóm. '
            'Đây KHÔNG phải kết quả cuối — điểm FICO đã tổng hợp từ 3 số này.'
        )
        proba_df = pd.DataFrame({
            'Nhóm': list(result['proba'].keys()),
            'Xác suất': list(result['proba'].values())
        })
        fig2 = go.Figure(go.Bar(
            x=proba_df['Nhóm'], y=proba_df['Xác suất'],
            marker_color=['#d9534f', '#6c8ebf', '#5cb85c'],
            text=[f'{p:.1%}' for p in proba_df['Xác suất']], textposition='outside'
        ))
        fig2.update_layout(yaxis=dict(range=[0, 1], tickformat='.0%'),
                            height=260, margin=dict(l=20, r=20, t=20, b=20),
                            showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        # === 3. SHAP ===
        st.markdown('#### 🔍 Yếu tố nào ảnh hưởng đến điểm của bạn?')
        with st.spinner('Đang phân tích...'):
            shap_df = get_shap_values(user_inputs, top_n=10)
        if shap_df is not None:
            shap_df = shap_df.sort_values('shap_value', ascending=True)
            colors = ['#5cb85c' if v > 0 else '#d9534f' for v in shap_df['shap_value']]
            fig_shap = go.Figure(go.Bar(
                y=shap_df['feature'],
                x=shap_df['shap_value'],
                orientation='h',
                marker_color=colors,
                text=[f"{'+' if v > 0 else ''}{v:.2f}" for v in shap_df['shap_value']],
                textposition='outside',
            ))
            fig_shap.update_layout(
                height=380,
                margin=dict(l=20, r=40, t=20, b=20),
                xaxis_title='Mức độ ảnh hưởng',
                yaxis_title='',
                showlegend=False,
            )
            fig_shap.add_vline(x=0, line_width=1, line_color='gray')
            st.plotly_chart(fig_shap, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('**🟢 Yếu tố giúp tăng điểm:**')
                pos = shap_df[shap_df['shap_value'] > 0].sort_values('shap_value', ascending=False).head(5)
                if len(pos):
                    for _, r in pos.iterrows():
                        st.write(f"• {r['feature']}")
                else:
                    st.caption('Không có yếu tố tích cực nổi bật.')
            with c2:
                st.markdown('**🔴 Yếu tố kéo điểm xuống:**')
                neg = shap_df[shap_df['shap_value'] < 0].sort_values('shap_value').head(5)
                if len(neg):
                    for _, r in neg.iterrows():
                        st.write(f"• {r['feature']}")
                else:
                    st.caption('Không có yếu tố tiêu cực nổi bật.')
        else:
            st.info('Cài `shap` để xem giải thích: `pip install shap`')

    # ---- Gửi n8n ----
    if N8N_WEBHOOK_URL:
        import numpy as np

        def to_native(v):
            if isinstance(v, (np.integer,)): return int(v)
            if isinstance(v, (np.floating,)): return float(v)
            if isinstance(v, np.ndarray): return v.tolist()
            return v

        payload = {
            'timestamp': datetime.now().isoformat(),
            'customer_name': customer_name or 'Ẩn danh',
            'customer_email': customer_email or '',
            'fico_score': round(float(result['fico_score']), 1),
            'rating': result['rating'],
            'predicted_class': result['predicted_class'],
            'proba': {k: round(float(v), 4) for k, v in result['proba'].items()},
            'inputs': {k: to_native(v) for k, v in user_inputs.items()},
        }
        try:
            r = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=WEBHOOK_TIMEOUT)
            if r.ok:
                st.toast('✅ Đã lưu kết quả và gửi email.', icon='📧')
            else:
                st.warning(f'Webhook trả về status {r.status_code}. Kết quả chưa được lưu.')
        except requests.RequestException as e:
            st.warning(f'Không kết nối được n8n: {e}. Kết quả không được lưu.')
    else:
        st.info('💡 Chưa cấu hình `N8N_WEBHOOK_URL` — kết quả chỉ hiển thị, chưa lưu tự động.')
