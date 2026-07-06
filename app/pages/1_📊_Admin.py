"""Admin Dashboard — báo cáo tổng hợp cho nhân viên."""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Cho phép import từ app/ (parent của pages/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ADMIN_PASSWORD, APP_TITLE
from utils import CLASS_COLORS, check_admin_password, load_sheet_data


st.set_page_config(page_title=f'{APP_TITLE} — Admin', page_icon='📊', layout='wide')

# ============ Session state ============
if 'admin_logged_in' not in st.session_state:
    st.session_state.admin_logged_in = False


# ============ Login form ============
def show_login():
    st.title('🔐 Admin Dashboard — Đăng nhập')
    st.caption('Trang này dành cho nhân viên quản lý hệ thống.')

    with st.form('login_form'):
        password = st.text_input('Mật khẩu', type='password')
        submitted = st.form_submit_button('Đăng nhập', use_container_width=True)
        if submitted:
            if check_admin_password(password):
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error('❌ Mật khẩu không đúng.')


def logout():
    st.session_state.admin_logged_in = False
    st.rerun()


# ============ Dashboard ============
@st.cache_data(ttl=30, show_spinner='Đang tải dữ liệu từ Google Sheets...')
def get_data():
    """Wrapper cache 30 giây cho load_sheet_data."""
    return load_sheet_data()


def show_dashboard():
    # Header
    c1, c2, c3 = st.columns([6, 1, 1])
    with c1:
        st.title('📊 Admin Dashboard')
        st.caption('Báo cáo tổng hợp các lượt tra cứu tín dụng')
    with c2:
        if st.button('🔄 Refresh', use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with c3:
        if st.button('🚪 Logout', use_container_width=True):
            logout()

    # Load data
    try:
        df = get_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            'Hướng dẫn setup service account:\n\n'
            '1. Vào https://console.cloud.google.com → chọn project\n'
            '2. APIs & Services → Credentials → Create Service Account\n'
            '3. Tạo JSON key, download về, đặt tại `models/service_account.json`\n'
            '4. Share Google Sheet với email service account (quyền Viewer)'
        )
        return
    except Exception as e:
        st.error(f'Lỗi khi đọc Google Sheet: {e}')
        return

    if df.empty:
        st.warning('Chưa có dữ liệu trong Google Sheet. Submit vài form ở trang chính trước.')
        return

    # ============ Filter bar ============
    st.markdown('---')
    f1, f2 = st.columns([1, 1])
    with f1:
        min_date = df['Thời gian'].min().date() if 'Thời gian' in df.columns else date.today() - timedelta(days=30)
        max_date = df['Thời gian'].max().date() if 'Thời gian' in df.columns else date.today()
        default_start = max(min_date, date.today() - timedelta(days=30))
        date_range = st.date_input(
            'Khoảng thời gian',
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    with f2:
        st.metric('Tổng lượt trong Sheet', len(df))

    # Apply date filter
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        mask = (df['Thời gian'].dt.date >= start) & (df['Thời gian'].dt.date <= end)
        dff = df[mask].copy()
    else:
        dff = df.copy()

    if dff.empty:
        st.warning('Không có dữ liệu trong khoảng thời gian đã chọn.')
        return

    # ============ KPI Cards ============
    st.markdown('### 📈 Chỉ số tổng quan')
    total = len(dff)
    class_col = 'Nhóm dự đoán' if 'Nhóm dự đoán' in dff.columns else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Tổng lượt', total)

    if class_col:
        counts = dff[class_col].value_counts()
        n_poor = int(counts.get('Poor', 0))
        n_std = int(counts.get('Standard', 0))
        n_good = int(counts.get('Good', 0))
        k2.metric('Poor', f'{n_poor} ({n_poor/total*100:.1f}%)',
                    delta=None, delta_color='inverse')
        k3.metric('Standard', f'{n_std} ({n_std/total*100:.1f}%)')
        k4.metric('Good', f'{n_good} ({n_good/total*100:.1f}%)')

    # ============ Chart 1: Trend theo ngày ============
    st.markdown('### 📅 Lượt tra cứu theo ngày')
    if 'Thời gian' in dff.columns and class_col:
        daily = (dff.groupby([dff['Thời gian'].dt.date, class_col])
                    .size().reset_index(name='count'))
        daily.columns = ['Ngày', 'Nhóm', 'Số lượt']
        fig1 = px.line(daily, x='Ngày', y='Số lượt', color='Nhóm',
                        color_discrete_map=CLASS_COLORS, markers=True)
        fig1.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info('Không có cột "Thời gian" hoặc "Nhóm dự đoán" — không vẽ được chart.')

    # ============ Chart 2: Phân bố class ============
    if class_col:
        st.markdown('### 🎯 Phân bố nhóm phân loại')
        counts_df = dff[class_col].value_counts().reset_index()
        counts_df.columns = ['Nhóm', 'Số lượt']
        fig2 = go.Figure(go.Bar(
            x=counts_df['Nhóm'], y=counts_df['Số lượt'],
            marker_color=[CLASS_COLORS.get(c, '#888') for c in counts_df['Nhóm']],
            text=counts_df['Số lượt'], textposition='outside',
        ))
        fig2.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20),
                            showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ============ Section: Case Poor cần review ============
    st.markdown('### ⚠️ Case Poor cần review')
    if class_col and 'Xác suất Poor' in dff.columns:
        confidence_threshold = st.slider(
            'Ngưỡng xác suất Poor tối thiểu (%)',
            0, 100, 60, step=5,
            help='Chỉ hiển thị case có xác suất Poor ≥ ngưỡng này'
        ) / 100

        poor_cases = dff[
            (dff[class_col] == 'Poor') &
            (dff['Xác suất Poor'] >= confidence_threshold)
        ].copy()

        if not poor_cases.empty:
            display_cols = [c for c in
                            ['Thời gian', 'Họ tên', 'Email', 'Xác suất Poor', 'Ghi chú']
                            if c in poor_cases.columns]
            st.dataframe(
                poor_cases[display_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Xác suất Poor': st.column_config.ProgressColumn(
                        'Xác suất Poor', min_value=0, max_value=1,
                        format='%.1f%%',
                    ),
                    'Thời gian': st.column_config.DatetimeColumn(
                        'Thời gian', format='DD/MM/YYYY HH:mm'
                    ),
                }
            )
            st.caption(f'Tìm thấy **{len(poor_cases)} case** cần review (ngưỡng {confidence_threshold*100:.0f}%).')
        else:
            st.success('✅ Không có case Poor nào vượt ngưỡng — hệ thống trong tình trạng bình thường.')

    # ============ Section: Search theo email ============
    st.markdown('### 🔍 Tra cứu lịch sử khách hàng')
    if 'Email' in dff.columns:
        search_email = st.text_input('Nhập email khách hàng',
                                       placeholder='vd: khach@gmail.com')
        if search_email:
            history = dff[dff['Email'].str.contains(search_email, case=False, na=False)]
            if not history.empty:
                st.dataframe(history.reset_index(drop=True),
                            use_container_width=True, hide_index=True)
                st.caption(f'Tìm thấy **{len(history)} lượt** cho email khớp `{search_email}`.')

                # Nếu có nhiều hơn 1 lượt → vẽ trend cho customer
                if len(history) > 1 and 'Xác suất Good' in history.columns:
                    st.markdown('#### Xu hướng cải thiện của khách')
                    trend_df = history.sort_values('Thời gian').copy()
                    fig_trend = px.line(
                        trend_df, x='Thời gian',
                        y=['Xác suất Poor', 'Xác suất Standard', 'Xác suất Good'],
                        color_discrete_map={
                            'Xác suất Poor': CLASS_COLORS['Poor'],
                            'Xác suất Standard': CLASS_COLORS['Standard'],
                            'Xác suất Good': CLASS_COLORS['Good'],
                        }, markers=True,
                    )
                    fig_trend.update_layout(height=300,
                                             yaxis=dict(range=[0, 1], tickformat='.0%'),
                                             margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info(f'Không tìm thấy lượt tra cứu nào với email `{search_email}`.')

    # ============ Export CSV ============
    st.markdown('### 💾 Xuất dữ liệu')
    csv = dff.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label=f'📥 Tải CSV ({len(dff)} dòng)',
        data=csv,
        file_name=f'credit_scoring_log_{date.today()}.csv',
        mime='text/csv',
    )


# ============ Main ============
if st.session_state.admin_logged_in:
    show_dashboard()
else:
    show_login()
