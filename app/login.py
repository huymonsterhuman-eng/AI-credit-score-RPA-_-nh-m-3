"""Trang đăng nhập — phân quyền theo username/password."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from config import USERS

st.set_page_config(
    page_title='CreditAI — Đăng nhập',
    page_icon=None,
    layout='centered',
    initial_sidebar_state='collapsed',
)

# ── Ẩn hoàn toàn sidebar và menu ────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"]          { display: none !important; }
  [data-testid="collapsedControl"]   { display: none !important; }
  #MainMenu, footer, header          { visibility: hidden; }
  html, body, [class*="css"]         { font-family: 'Segoe UI', Arial, sans-serif; }
  .main                              { background: #f0f4f8; }

  .login-card {
    max-width: 420px;
    margin: 80px auto 0;
    background: white;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(21,101,192,0.18);
    padding: 36px 32px;
  }
  .login-title {
    font-size: 28px; font-weight: 800; color: #1a237e;
    text-align: center; margin-bottom: 4px;
  }
  .login-sub {
    font-size: 14px; color: #78909c;
    text-align: center; margin-bottom: 28px;
  }

  div[data-testid="stForm"] button[type="submit"] {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(21,101,192,0.3) !important;
    margin-top: 8px !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Redirect nếu đã đăng nhập ───────────────────────────────────────────────
if st.session_state.get('role') == 'customer':
    st.switch_page('pages/home.py')
elif st.session_state.get('role') == 'admin':
    st.switch_page('pages/admin.py')


with st.form('login_form'):
    username = st.text_input('Tên đăng nhập', placeholder='Nhập tên đăng nhập')
    password = st.text_input('Mật khẩu', type='password', placeholder='Nhập mật khẩu')
    submitted = st.form_submit_button('Đăng nhập', use_container_width=True)

    if submitted:
        username = username.strip().lower()
        user = USERS.get(username)
        if user and password == user[0]:
            st.session_state.role     = user[1]
            st.session_state.username = username
            if user[1] == 'admin':
                st.switch_page('pages/admin.py')
            else:
                st.switch_page('pages/home.py')
        else:
            st.error('Tên đăng nhập hoặc mật khẩu không đúng.')
