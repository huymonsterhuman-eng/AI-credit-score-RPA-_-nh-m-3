"""Streamlit app cho AI Credit Classification."""

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from config import (APP_SUBTITLE, APP_TITLE, DISCLAIMER,
                     N8N_WEBHOOK_URL, WEBHOOK_TIMEOUT)
from utils import (CLASS_COLORS, CLASS_LABELS_VI, derive_behavior,
                    diff_inputs, get_previous_prediction, get_shap_values,
                    load_artifacts, predict, save_prediction, suggestions_for)


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
st.title('Credit Classification — Phân loại mức tín dụng cá nhân')
st.write(
    'Nhập thông tin bên dưới để nhận kết quả phân loại tín dụng (Poor / Standard / Good) '
    'kèm gợi ý cải thiện.'
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
        'Has_Mortgage_Loan': False, 'Has_Student_Loan': False,
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
        'Has_Credit_Builder_Loan': False, 'Has_Home_Equity_Loan': True,
        'Has_Mortgage_Loan': True, 'Has_Student_Loan': False,
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
pc1, pc2, pc3, pc4 = st.columns(4)
pc1.button('🔴 Kém (Poor)', on_click=apply_preset, args=('Poor',),
            use_container_width=True, help='Load case tín dụng kém')
pc2.button('🟠 Trung bình (Standard)', on_click=apply_preset, args=('Standard',),
            use_container_width=True, help='Load case tín dụng trung bình')
pc3.button('🟢 Tốt (Good)', on_click=apply_preset, args=('Good',),
            use_container_width=True, help='Load case tín dụng tốt')
pc4.button('🔄 Reset', on_click=reset_form,
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
    Has_Home_Equity_Loan = int(c2.checkbox('Vay thế chấp nhà (Home Equity)',
                                            key='Has_Home_Equity_Loan'))
    Has_Mortgage_Loan = int(c3.checkbox('Vay mua nhà (Mortgage)',
                                         key='Has_Mortgage_Loan'))
    Has_Student_Loan = int(c4.checkbox('Vay sinh viên', key='Has_Student_Loan'))
    Has_Payday_Loan = int(c5.checkbox('Vay ngắn hạn (Payday)', key='Has_Payday_Loan'))
    Num_Loan_Types = sum([Has_Credit_Builder_Loan, Has_Home_Equity_Loan,
                           Has_Mortgage_Loan, Has_Student_Loan,
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
        'Has_Home_Equity_Loan': Has_Home_Equity_Loan,
        'Has_Mortgage_Loan': Has_Mortgage_Loan,
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

    # Lưu snapshot vào SQLite để tracking progress lần sau
    prediction_timestamp = datetime.now().isoformat()
    saved_row_id = None
    previous = None
    try:
        # Lấy lần trước TRƯỚC khi save (để không lẫn lộn với chính lần này)
        previous = get_previous_prediction(customer_email)
        saved_row_id = save_prediction(
            email=customer_email,
            customer_name=customer_name,
            timestamp=prediction_timestamp,
            inputs=user_inputs,
            result=result,
        )
    except Exception as e:
        # Không block flow chính nếu DB lỗi
        st.caption(f'⚠️ Không lưu được vào lịch sử local: {e}')

    st.success('Dự đoán hoàn tất.')
    st.markdown('---')

    # ---- Kết quả chính ----
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            f"<div style='padding:32px 20px; border-radius:12px; "
            f"background:{result['color']}; color:white; text-align:center;'>"
            f"<div style='font-size:14px; opacity:0.9;'>Nhóm phân loại</div>"
            f"<div style='font-size:42px; font-weight:bold; margin-top:8px;'>"
            f"{result['predicted_class_vi']}</div>"
            f"<div style='font-size:16px; opacity:0.95; margin-top:4px;'>"
            f"({result['predicted_class']})</div>"
            f"<div style='font-size:13px; margin-top:12px; opacity:0.9;'>"
            f"Độ tin cậy: {result['confidence']*100:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with c2:
        st.subheader('Xác suất theo từng nhóm')
        proba_df = pd.DataFrame({
            'Nhóm': list(result['proba'].keys()),
            'Xác suất': list(result['proba'].values())
        })
        fig2 = go.Figure(go.Bar(
            x=proba_df['Nhóm'], y=proba_df['Xác suất'],
            marker_color=[CLASS_COLORS[c] for c in proba_df['Nhóm']],
            text=[f'{p:.1%}' for p in proba_df['Xác suất']], textposition='outside'
        ))
        fig2.update_layout(yaxis=dict(range=[0, 1], tickformat='.0%'),
                            height=280, margin=dict(l=20, r=20, t=20, b=20),
                            showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ---- Gợi ý (main view) ----
    st.subheader('💡 Gợi ý cải thiện')
    for tip in suggestions_for(result, user_inputs, previous=previous):
        st.write(tip)

    # ---- So sánh với lần tra cứu trước (nếu có) ----
    if previous:
        from datetime import datetime as _dt
        try:
            prev_dt = _dt.fromisoformat(previous['timestamp'])
            days_ago = (_dt.now() - prev_dt).days
            date_str = prev_dt.strftime('%d/%m/%Y %H:%M')
        except (ValueError, TypeError):
            days_ago = None
            date_str = str(previous['timestamp'])

        st.markdown('---')
        st.subheader('📊 So sánh với lần tra cứu trước')

        days_label = f'({days_ago} ngày trước)' if days_ago is not None else ''
        st.caption(f'Bạn đã tra cứu ngày **{date_str}** {days_label}.')

        # Delta các probability
        d_poor = result['proba']['Poor'] - previous['proba']['Poor']
        d_std = result['proba']['Standard'] - previous['proba']['Standard']
        d_good = result['proba']['Good'] - previous['proba']['Good']

        cc1, cc2, cc3, cc4 = st.columns(4)

        # Class change
        prev_cls = previous['predicted_class']
        curr_cls = result['predicted_class']
        class_ranks = {'Poor': 0, 'Standard': 1, 'Good': 2}
        cls_delta = class_ranks.get(curr_cls, 1) - class_ranks.get(prev_cls, 1)
        if cls_delta > 0:
            cls_indicator = '🎉 Cải thiện'
        elif cls_delta < 0:
            cls_indicator = '⚠️ Xấu đi'
        else:
            cls_indicator = '→ Giữ nguyên'

        cc1.metric(
            'Nhóm phân loại',
            f'{prev_cls} → {curr_cls}',
            delta=cls_indicator,
            delta_color='normal' if cls_delta >= 0 else 'inverse',
        )
        # P(Poor): giảm là tốt → dùng delta_color inverse
        cc2.metric(
            'P(Poor)',
            f'{result["proba"]["Poor"]*100:.1f}%',
            delta=f'{d_poor*100:+.1f}%',
            delta_color='inverse',
        )
        cc3.metric(
            'P(Standard)',
            f'{result["proba"]["Standard"]*100:.1f}%',
            delta=f'{d_std*100:+.1f}%',
            delta_color='off',
        )
        # P(Good): tăng là tốt → delta_color normal
        cc4.metric(
            'P(Good)',
            f'{result["proba"]["Good"]*100:.1f}%',
            delta=f'{d_good*100:+.1f}%',
            delta_color='normal',
        )

        if cls_delta > 0:
            st.success('🎉 Chúc mừng bạn đã cải thiện nhóm tín dụng! Tiếp tục duy trì thói quen tốt.')
        elif cls_delta < 0:
            st.warning('⚠️ Nhóm tín dụng của bạn đã hạ. Xem gợi ý cải thiện ở trên để quay lại quỹ đạo.')
        elif d_good > 0.05:
            st.info('📈 Chưa đổi nhóm nhưng xác suất Good đang tăng — bạn đi đúng hướng, cố gắng thêm.')

        # ---- Expander: factor-level diff ----
        with st.expander('🔍 Xem yếu tố nào thay đổi giữa 2 lần', expanded=False):
            diffs = diff_inputs(previous['inputs'], user_inputs, top_n=8)
            if not diffs:
                st.info('Không có yếu tố actionable nào thay đổi đáng kể giữa 2 lần tra cứu.')
            else:
                st.caption(
                    'Bar xanh = thay đổi có lợi (tăng khả năng vào Good). '
                    'Bar đỏ = thay đổi bất lợi. Sort theo mức độ thay đổi.'
                )
                # Build DataFrame for chart
                chart_df = pd.DataFrame([
                    {
                        'Feature': d['label'],
                        'Delta': d['delta_pct'] if d.get('delta_pct') is not None else (d['delta'] * 50 if d['is_categorical'] else d['delta']),
                        'Impact': d['impact'],
                        'Prev': d['prev'],
                        'Curr': d['curr'],
                        'IsCat': d['is_categorical'],
                        'Format': d.get('format'),
                    }
                    for d in diffs
                ])
                # Sort ascending cho horizontal bar (nhỏ nhất trên)
                chart_df = chart_df.sort_values('Delta')

                fig_diff = go.Figure(go.Bar(
                    y=chart_df['Feature'],
                    x=chart_df['Delta'],
                    orientation='h',
                    marker_color=['#5cb85c' if i == 'good' else '#d9534f' for i in chart_df['Impact']],
                    text=[
                        (f"{r['Prev']} → {r['Curr']}" if r['IsCat']
                         else (r['Format'] or '{:.1f}').format(r['Prev']) + ' → ' +
                              (r['Format'] or '{:.1f}').format(r['Curr']))
                        for _, r in chart_df.iterrows()
                    ],
                    textposition='outside',
                ))
                fig_diff.update_layout(
                    height=max(280, 40 + 40 * len(chart_df)),
                    margin=dict(l=20, r=100, t=20, b=20),
                    xaxis_title='Mức độ thay đổi (%)',
                    yaxis_title='',
                    showlegend=False,
                )
                fig_diff.add_vline(x=0, line_width=1, line_color='gray')
                st.plotly_chart(fig_diff, use_container_width=True)

                # Text summary — top 3 tích cực + top 3 tiêu cực
                pos = [d for d in diffs if d['impact'] == 'good'][:3]
                neg = [d for d in diffs if d['impact'] == 'bad'][:3]
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown('**🟢 Cải thiện nổi bật:**')
                    if pos:
                        for d in pos:
                            if d['is_categorical']:
                                st.write(f"• {d['label']}: {d['prev']} → **{d['curr']}**")
                            else:
                                fmt = d.get('format') or '{:.1f}'
                                st.write(f"• {d['label']}: {fmt.format(d['prev'])} → **{fmt.format(d['curr'])}**")
                    else:
                        st.caption('Không có cải thiện nổi bật.')
                with sc2:
                    st.markdown('**🔴 Xấu đi:**')
                    if neg:
                        for d in neg:
                            if d['is_categorical']:
                                st.write(f"• {d['label']}: {d['prev']} → **{d['curr']}**")
                            else:
                                fmt = d.get('format') or '{:.1f}'
                                st.write(f"• {d['label']}: {fmt.format(d['prev'])} → **{fmt.format(d['curr'])}**")
                    else:
                        st.caption('Không có yếu tố xấu đi.')

    st.markdown('---')

    # ---- Expander: chi tiết kỹ thuật ----
    with st.expander('🔬 Chi tiết kỹ thuật — Yếu tố ảnh hưởng đến kết quả',
                       expanded=False):
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
            'predicted_class': result['predicted_class'],
            'predicted_class_vi': result['predicted_class_vi'],
            'proba': {k: round(float(v), 4) for k, v in result['proba'].items()},
            'inputs': {k: to_native(v) for k, v in user_inputs.items()},
            'previous': ({
                'timestamp': previous['timestamp'],
                'predicted_class': previous['predicted_class'],
                'proba': {k: round(float(v), 4) for k, v in previous['proba'].items()},
            } if previous else None),
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
