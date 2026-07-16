# n8n Setup — Credit Scoring Automation

Hướng dẫn dựng n8n self-hosted qua Docker để tự động hóa lưu Sheets + gửi email + đính kèm PDF report.

## Yêu cầu

- **Docker Desktop for Windows** — cài từ https://docs.docker.com/desktop/setup/install/windows-install/
- Tài khoản Google (cho Google Sheets + Gmail)

## Bước 1 — Khởi động n8n

Từ folder này (`n8n/`), chạy:

```powershell
docker compose up -d
```

Chờ ~30 giây cho n8n download image lần đầu (~500 MB). Kiểm tra:

```powershell
docker ps
```

Thấy container `credit-scoring-n8n` với status `Up` là ok.

Mở trình duyệt: **http://localhost:5678**

Lần đầu, n8n yêu cầu tạo tài khoản owner (email + password bất kỳ, dùng local). Chỉ dùng cho local development.

## Bước 2 — Cài community node cho HTML → PDF

Vào **Settings** (icon bánh răng ↖ hoặc menu) → **Community Nodes** → **Install a community node**.

Nhập:
```
n8n-nodes-puppeteer
```

Bấm **Install** → chờ 1–2 phút. Đây là node giúp render HTML thành PDF.

> **Nếu node không install được**: dùng fallback — workflow sẽ gửi email HTML không kèm PDF. Vẫn chấp nhận được cho đồ án.

## Bước 3 — Setup Google Sheets credential

1. Vào **Credentials** → **New** → tìm **Google Sheets OAuth2 API**.
2. n8n hướng dẫn tạo Google Cloud project:
   - Đi tới https://console.cloud.google.com/apis/credentials
   - Create project → Enable **Google Sheets API** + **Gmail API**
   - Create OAuth 2.0 Client ID
   - Application type: Web application
   - Authorized redirect URI: `http://localhost:5678/rest/oauth2-credential/callback`
   - Copy Client ID + Secret vào n8n → **Sign in with Google** → cho phép access
3. Tạo Google Sheet mới, đặt tên **"Credit Scoring Log"** với các cột:
   ```
   timestamp | customer_name | customer_email | fico_score | rating |
   predicted_class | P_poor | P_standard | P_good | notes
   ```
4. Copy Sheet ID từ URL (đoạn giữa `/d/` và `/edit`) — sẽ dùng lúc build workflow.

## Bước 4 — Setup Gmail credential

1. **Credentials** → **New** → **Gmail OAuth2 API**.
2. Dùng cùng OAuth Client vừa tạo (thêm scope `gmail.send` nếu thiếu).
3. Redirect URI: `http://localhost:5678/rest/oauth2-credential/callback` (giống Sheets).
4. **Sign in with Google** → cho phép gửi email.

## Bước 5 — Import workflow

Sẽ có sau — file `workflow.json` trong folder này. Import bằng:
**Workflows** → **⋯** menu → **Import from File** → chọn `workflow.json`.

Sau khi import:
- Chọn credential đúng cho các node Google Sheets và Gmail.
- Sửa Sheet ID và Sheet Name trong node "Append to Sheet".
- Tạo thêm tab **Weekly_Report** trong cùng Google Sheet với các cột:
  ```
  Week | Total | Poor | Standard | Good | With_Email | Return_Rate | Avg_Per_Day
  ```
- **Activate** workflow (switch ↖ trên góc phải).
- Copy Webhook URL (trong node Webhook) và dán vào `app/config.py` → biến `N8N_WEBHOOK_URL`.
- Webhook báo cáo tuần (thủ công): `http://localhost:5678/webhook/weekly-report` → `N8N_WEEKLY_WEBHOOK_URL`.

## Bước 6 — Test

Trong Streamlit, submit 1 form. Kiểm tra:
- Google Sheet có dòng mới ✓
- Nhận được email kèm PDF ✓ (hoặc chỉ HTML nếu không cài puppeteer)

## Dừng / khởi động lại n8n

```powershell
# Dừng
docker compose stop

# Khởi động lại
docker compose start

# Dừng hẳn và xóa container (data trong volume vẫn giữ)
docker compose down

# Xem log realtime
docker compose logs -f
```

## Trouble shooting

- **Cổng 5678 đã dùng** → sửa `ports: - "5678:5678"` thành `- "5679:5678"` trong `docker-compose.yml`, mở `http://localhost:5679`.
- **Không kết nối Google được** → chắc chắn Client ID redirect URI khớp `http://localhost:5678/rest/oauth2-credential/callback` (không dấu `/` cuối).
- **Community node install fail** → thử `docker compose down && docker compose up -d` rồi cài lại. Nếu vẫn fail, bỏ qua — workflow có fallback không dùng PDF.
