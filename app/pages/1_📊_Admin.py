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
        st.error(f'Lỗi khi đọc Google Sheet: **{type(e).__name__}** — {e}')
        st.info(
            'Các nguyên nhân phổ biến:\n'
            '- Chưa share Sheet với email service account (email có dạng `xxx@yyy.iam.gserviceaccount.com`)\n'
            '- Tab name không đúng — thử đổi `SHEET_NAME` trong config.py (hiện: `Sheet1`)\n'
            '- Sheet ID trong config.py không khớp với Sheet thật'
        )
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

    # ============ KPI Cards — Hàng 1: Phân loại ============
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

    # ============ KPI Cards — Hàng 2: Engagement ============
    m1, m2, m3, m4 = st.columns(4)

    # Trung bình lượt/ngày
    if 'Thời gian' in dff.columns:
        n_days = max((dff['Thời gian'].dt.date.max() - dff['Thời gian'].dt.date.min()).days + 1, 1)
        avg_per_day = total / n_days
        m1.metric('TB lượt/ngày', f'{avg_per_day:.1f}',
                    help=f'{total} lượt / {n_days} ngày')

    # Số lượt có email vs không
    if 'Email' in dff.columns:
        has_email = dff['Email'].astype(str).str.strip().str.len() > 0
        has_email &= dff['Email'].astype(str).str.contains('@', na=False)
        n_with = int(has_email.sum())
        n_without = total - n_with
        m2.metric('Có email', f'{n_with} ({n_with/total*100:.0f}%)')
        m3.metric('Không có email', f'{n_without} ({n_without/total*100:.0f}%)')

        # Repeat rate — khách xuất hiện ≥ 2 lần
        emails_valid = dff[has_email]['Email'].astype(str).str.strip().str.lower()
        if len(emails_valid) > 0:
            email_counts = emails_valid.value_counts()
            repeat_customers = int((email_counts >= 2).sum())
            unique_customers = int(email_counts.nunique())
            repeat_rate = repeat_customers / unique_customers * 100 if unique_customers else 0
            m4.metric(
                'Tỷ lệ khách quay lại',
                f'{repeat_rate:.1f}%',
                help=f'{repeat_customers} khách quay lại / {unique_customers} khách có email',
            )
        else:
            m4.metric('Tỷ lệ khách quay lại', '—')

    # ============ Chart: Phân bố số lần tra cứu / khách ============
    if 'Email' in dff.columns:
        has_email = dff['Email'].astype(str).str.strip().str.len() > 0
        has_email &= dff['Email'].astype(str).str.contains('@', na=False)
        emails_valid = dff[has_email]['Email'].astype(str).str.strip().str.lower()

        if len(emails_valid) > 0:
            st.markdown('### 🔁 Phân bố số lần tra cứu của mỗi khách')
            email_counts = emails_valid.value_counts()

            # Bin số lần
            def bin_label(n):
                if n == 1: return '1 lần'
                if n == 2: return '2 lần'
                if n == 3: return '3 lần'
                if n <= 5: return '4-5 lần'
                return '6+ lần'

            binned = email_counts.apply(bin_label).value_counts()
            order = ['1 lần', '2 lần', '3 lần', '4-5 lần', '6+ lần']
            binned = binned.reindex([o for o in order if o in binned.index])

            fig_hist = go.Figure(go.Bar(
                x=binned.index, y=binned.values,
                marker_color='#667eea',
                text=binned.values, textposition='outside',
            ))
            fig_hist.update_layout(
                height=280, margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title='Số lần tra cứu', yaxis_title='Số khách hàng',
                showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption(
                f'Tổng: **{len(email_counts)} khách hàng có email** với '
                f'trung bình **{email_counts.mean():.1f} lượt/khách**.'
            )

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

    # ============ Section: Case cần review (sort by P(Poor)) ============
    st.markdown('### ⚠️ Case cần review — sắp xếp theo rủi ro cao nhất')
    if 'Xác suất Poor' in dff.columns:
        # Tổng số case
        total_review = len(dff)
        n_poor_class = int((dff[class_col] == 'Poor').sum()) if class_col else 0
        st.caption(
            f'Tổng: **{total_review} lượt** tra cứu · '
            f'Trong đó **{n_poor_class} case** được model phân loại là Poor.'
        )

        # Filter mode
        mode = st.radio(
            'Bộ lọc',
            ['Tất cả case (sort theo P(Poor) giảm dần)',
             'Chỉ case phân loại Poor',
             'Case có P(Poor) ≥ ngưỡng'],
            horizontal=True,
        )

        review_df = dff.copy()
        if mode == 'Chỉ case phân loại Poor' and class_col:
            review_df = review_df[review_df[class_col] == 'Poor']
        elif mode.startswith('Case có P(Poor)'):
            threshold = st.slider(
                'Ngưỡng P(Poor) tối thiểu (%)',
                0, 100, 30, step=5,
            ) / 100
            review_df = review_df[review_df['Xác suất Poor'] >= threshold]

        review_df = review_df.sort_values('Xác suất Poor', ascending=False)

        if not review_df.empty:
            display_cols = [c for c in
                            ['Thời gian', 'Họ tên', 'Email', class_col,
                             'Xác suất Poor', 'Ghi chú']
                            if c and c in review_df.columns]

            col_config = {}
            if 'Xác suất Poor' in display_cols:
                col_config['Xác suất Poor'] = st.column_config.ProgressColumn(
                    'Xác suất Poor', min_value=0, max_value=1, format='%.1f%%',
                )
            if 'Thời gian' in display_cols:
                col_config['Thời gian'] = st.column_config.DatetimeColumn(
                    'Thời gian', format='DD/MM/YYYY HH:mm',
                )

            st.dataframe(
                review_df[display_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                column_config=col_config,
                height=min(400, 50 + 35 * min(len(review_df), 10)),
            )
            st.caption(f'Hiển thị **{len(review_df)} case**.')
        else:
            st.info('Không có case nào khớp bộ lọc.')
    else:
        st.info('Sheet chưa có cột "Xác suất Poor" — không hiển thị section này được.')

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
