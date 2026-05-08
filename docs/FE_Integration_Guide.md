# Report Tích Hợp Frontend (FE) - ChatbotCSKHv2

Tài liệu này cung cấp các thông số kỹ thuật và API chi tiết để đội ngũ Frontend tích hợp hệ thống **Chatbot Chăm Sóc Khách Hàng (ChatbotCSKHv2)** lên môi trường Production.

---

## 1. Thông Tin Chung
Hệ thống Chatbot Backend hiện tại đang chạy hoàn toàn bằng API (không có Web UI đính kèm). Frontend (Portal UI) sẽ gọi trực tiếp hoặc thông qua API Gateway đến các endpoint này.

- **Base URL (Gateway)**: `https://llmerp.hextech.vn/primer-diamond` *(Vui lòng confirm lại Route từ Gateway proxy)*
- **Protocol**: HTTP/HTTPS
- **Content-Type mặc định**: `application/json`

---

## 2. Cơ Chế Xác Thực & Phân Quyền (Headers)

Khi gọi API, Frontend cần truyền đầy đủ các Headers sau đây (nếu gọi qua Gateway, Gateway sẽ tự động chặn/pass các header này).

| Header Key | Ý Nghĩa | Bắt buộc | Ví dụ |
|---|---|---|---|
| `Authorization` | JWT Token của hệ thống đối tác (Gateway sẽ tự động giải mã và cấp quyền SSO). | Có | `Bearer eyJhbGci...` |
| `X-Session-ID` | Định danh phiên chat. Rất quan trọng để Bot ghi nhớ ngữ cảnh! | Có (Trừ lần đầu) | `sess_xyz123` |

> **💡 CƠ CHẾ CẤP QUYỀN TỰ ĐỘNG (SSO AUTO-PROVISIONING):**
> API Gateway được cấu hình để tự động nhận dạng JWT của đối tác (BYOT - Bring Your Own Token). Frontend KHÔNG cần truyền các header như `X-User-ID`, `X-Role-Level` vì Gateway sẽ tự động trích xuất `email` từ Payload của Token. 
> - Nếu email chứa cụm từ `admin` (VD: `sysadmin@ctg.com`), user tự động nhận quyền Quản trị (Level D).
> - Ngược lại, user nhận quyền Nhân viên/Khách (Level E).

> **⚠️ LƯU Ý CHO FE (SESSION ID):**
> Lần đầu người dùng nhắn tin, FE không cần truyền `X-Session-ID`. Backend sẽ trả về một `session_id` mới trong Response. Các câu hỏi tiếp theo của cùng cuộc trò chuyện, FE **BẮT BUỘC** phải đính kèm `session_id` này vào Header `X-Session-ID` hoặc trong Body để bot nhớ được lịch sử chat.

---

## 3. Danh Sách API Endpoints

### 3.1. API Hỏi Đáp Chatbot - Dạng Stream (`POST /.../chat/stream`)

Đây là endpoint cốt lõi dùng để giao tiếp với AI Agent. Agent tự động nhận diện ý định và phản hồi theo **thời gian thực (Server-Sent Events - SSE)**.

**Endpoint (qua Gateway)**: `POST /api/{tenant_id}/chat/stream` (Ví dụ: `/api/primer-diamond/chat/stream`)

**Ví dụ JS (Dùng Fetch API & SSE):**
```javascript
const res = await fetch("https://llmerp.hextech.vn/api/primer-diamond/chat/stream", {
    method: "POST",
    headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer <YOUR_JWT_TOKEN>"
    },
    body: JSON.stringify({
        "message": "Căn hộ 2PN tại dự án Elysian giá bao nhiêu?",
        "session_id": "sess_abc123" // Truyền nếu đang tiếp tục chat
    })
});

const reader = res.body.getReader();
const decoder = new TextDecoder("utf-8");

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value, { stream: true });
    // Parse chuỗi chunk (format: data: { JSON } \n\n)
    // - data.text: Chứa text sinh ra từng chữ
    // - data.suggested_questions: Gợi ý trả về ở chunk cuối
}
```

**Cấu trúc dữ liệu Stream trả về (SSE Format)**:
Mỗi cục dữ liệu trả về sẽ bắt đầu bằng `data: ` và kết thúc bằng `\n\n`. Khi kết thúc stream, server trả về `data: [DONE]`.

```json
data: {"text": "Hiện tại "}
data: {"text": "căn hộ 2PN "}
...
data: {"text": "", "suggested_questions": ["Chính sách thanh toán?", "Có vay ngân hàng không?"], "sources": [{"doc_name": "CSBH.pdf"}]}
data: [DONE]
```

**Chi tiết thêm cho FE**:
- **`suggested_questions`**: Danh sách 3 câu hỏi gợi ý tiếp theo để hiển thị thành các bong bóng (chips) cho người dùng bấm chọn nhanh.
- **`answer`**: Nội dung câu trả lời, trả về định dạng Markdown. FE cần sử dụng thư viện render Markdown (như `react-markdown`).
- **`sources`**: Nguồn tài liệu Bot đã tham khảo. FE có thể tạo UI "Trích xuất từ tài liệu..." dưới mỗi câu trả lời.

---

### 3.2. API Hỏi Đáp Chatbot - Dạng Đồng Bộ (`POST /.../chat`)

Nếu hệ thống Frontend/Mobile của bạn không hỗ trợ SSE (Server-Sent Events) hoặc không muốn dùng Stream, bạn có thể dùng API Đồng Bộ. Agent sẽ xử lý xong toàn bộ câu trả lời rồi mới trả về một JSON cục duy nhất. **Lưu ý: API này sẽ phải đợi khá lâu (5-15s) trước khi nhận được phản hồi.**

**Endpoint (qua Gateway)**: `POST /{tenant_id}/chat` (Ví dụ: `/api/primer-diamond/chat`)

**Ví dụ JS (Fetch API):**
```javascript
const res = await fetch("https://llmerp.hextech.vn/api/primer-diamond/chat", {
    method: "POST",
    headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer <YOUR_JWT_TOKEN>"
    },
    body: JSON.stringify({
        "message": "Căn hộ 2PN tại dự án Elysian giá bao nhiêu?",
        "session_id": "sess_abc123"
    })
});

const data = await res.json();
// data.answer -> Nội dung trả lời (Markdown)
// data.suggested_questions -> Mảng câu hỏi gợi ý
// data.sources -> Mảng tài liệu nguồn
// data.session_id -> Lưu lại để dùng cho lượt chat sau
```

---

### 3.3. API Lấy Danh Sách Dự Án (`GET /api/v1/projects`)

Dùng để hiển thị danh sách các dự án hiện có trên Dropdown của giao diện Chat.

**Endpoint**: `GET /api/v1/projects`

**Body Response (JSON)**:
```json
{
  "projects": [
    "Elysian",
    "Vinhomes_GrandPark",
    "Metro_Star"
  ]
}
```

---

### 3.4. API Kiểm Tra Trạng Thái Hệ Thống (`GET /health`)

Dùng để Frontend/Load Balancer kiểm tra tình trạng kết nối.

**Endpoint**: `GET /health`

**Body Response (JSON)**:
```json
{
  "status": "healthy",
  "dependencies": {
    "redis": "ok",
    "vector_db": "ok"
  }
}
```

---

### 3.5. Các Mã Lỗi (HTTP Status Codes)

Trong quá trình gọi API, Frontend cần xử lý các mã trạng thái HTTP sau:

| HTTP Code | Ý Nghĩa | Cách Xử Lý (Dành cho FE) |
|---|---|---|
| **`200 OK`** | Thành công. | Đọc kết quả trong `answer` và cập nhật UI. |
| **`400 Bad Request`** | Dữ liệu gửi lên không hợp lệ (sai định dạng JSON, thiếu trường bắt buộc). | Kiểm tra lại dữ liệu trước khi gửi (VD: `message` bị rỗng). |
| **`401 Unauthorized`**| Sai hoặc thiếu JWT Token ở Header `Authorization`, Token hết hạn. | Chuyển hướng người dùng về trang Đăng nhập. |
| **`403 Forbidden`** | User không có quyền truy cập chức năng này hoặc sai Tenant. | Hiển thị thông báo "Bạn không có quyền truy cập". |
| **`404 Not Found`** | Gọi sai URL API Endpoint. | Kiểm tra lại Base URL hoặc Path API. |
| **`500 Internal Error`**| Lỗi hệ thống Backend (AI bị timeout, đứt kết nối Database). | Hiển thị giao diện báo lỗi (VD: *"Hệ thống đang bảo trì, vui lòng thử lại sau"*). |

---

## 4. Các Chú Ý Quan Trọng Cho Đội Phát Triển

1. **Lịch sử hội thoại**: 
   - Lịch sử sẽ tự động được lưu trữ tại PostgreSQL trên Backend.
   - FE không cần gửi kèm toàn bộ mảng lịch sử chat. Chỉ cần gửi đúng chuỗi text `message` hiện tại và Header/Body `session_id`. Backend sẽ tự động móc nối toàn bộ lịch sử.

2. **Dữ liệu phân quyền (Role Level)**:
   - Các Account có `role_level` thấp (như mức 1) sẽ tự động bị giấu đi một số thông tin nội bộ. Gateway sẽ inject header `X-Role-Level` này dựa vào Token. Cấp bậc: 1=E, 2=C, 3=M, 4=D.

3. **Timeouts**:
   - Agent xử lý có thể mất từ 2-10 giây tùy độ khó câu hỏi vì phải quét qua cơ sở dữ liệu Vector và dùng LLM tổng hợp thông tin. Frontend nên set Timeout ở mức tối thiểu 30s và hiển thị các Loading Skeleton sinh động để giữ chân người dùng.

4. **Xử lý lỗi (Error Handling)**:
   - Khi API trả về HTTP Code `500` hoặc trong cấu trúc JSON có `fallback: true`, FE nên render UI báo lỗi thân thiện (VD: *"Hệ thống đang quá tải, vui lòng để lại số điện thoại để chuyên viên tư vấn gọi lại"*).
