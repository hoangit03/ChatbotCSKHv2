# Report Tích Hợp Frontend (FE) - ChatbotCSKHv2

Tài liệu này cung cấp các thông số kỹ thuật và API chi tiết để đội ngũ Frontend tích hợp hệ thống **Chatbot Chăm Sóc Khách Hàng (ChatbotCSKHv2)** lên môi trường Production.

---

## 1. Thông Tin Chung
Hệ thống Chatbot Backend hiện tại đang chạy hoàn toàn bằng API (không có Web UI đính kèm). Frontend (Portal UI) sẽ gọi trực tiếp hoặc thông qua API Gateway đến các endpoint này.

- **Base URL (Gateway)**: `https://llmerp.hextech.vn/prim-diamond` *(Vui lòng confirm lại Route từ Gateway proxy)*
- **Protocol**: HTTP/HTTPS
- **Content-Type mặc định**: `application/json`

---

## 2. Cơ Chế Xác Thực & Phân Quyền (Headers)

Khi gọi API, Frontend cần truyền đầy đủ các Headers sau đây (nếu gọi qua Gateway, Gateway sẽ tự động chặn/pass các header này).

| Header Key | Ý Nghĩa | Bắt buộc | Ví dụ |
|---|---|---|---|
| `Authorization` | Token xác thực của người dùng đăng nhập (Gateway sẽ xử lý). | Có | `Bearer eyJhbGci...` |
| `X-User-ID` | ID của người dùng đang đăng nhập | Không (nhưng khuyến nghị) | `user_123` |
| `X-Tenant-ID` | ID của đối tác/công ty (để cách ly dữ liệu) | Không | `prim-diamond` |
| `X-Role-Level` | Cấp bậc phân quyền (để lọc tài liệu hiển thị). 1-E, 2-C, 3-M, 4-D. | Có (Mặc định `1`) | `2` |
| `X-Session-ID` | Định danh phiên chat. Rất quan trọng để Bot ghi nhớ ngữ cảnh! | CÓ (Khi chat liên tiếp) | `sess_xyz123` |

> **⚠️ LƯU Ý CHO FE (SESSION ID):**
> Lần đầu người dùng nhắn tin, FE không cần truyền `X-Session-ID`. Backend sẽ trả về một `session_id` mới trong Response. Các câu hỏi tiếp theo của cùng cuộc trò chuyện, FE **BẮT BUỘC** phải đính kèm `session_id` này vào Header `X-Session-ID` hoặc trong Body để bot nhớ được lịch sử chat.

---

## 3. Danh Sách API Endpoints

### 3.1. API Hỏi Đáp Chatbot (`POST /api/v1/chat`)

Đây là endpoint cốt lõi dùng để giao tiếp với AI Agent. Agent tự động nhận diện ý định khách hàng (Hỏi dự án, Xin giá, Đặt cọc) và điều phối công việc.

**Endpoint**: `POST /api/v1/chat`

**Ví dụ cURL gọi API:**
```bash
curl -X POST "https://llmerp.hextech.vn/prim-diamond/api/v1/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_JWT_TOKEN>" \
  -H "X-Role-Level: 1" \
  -d '{
    "message": "Căn hộ 2PN tại dự án Elysian giá bao nhiêu?",
    "session_id": "sess_abc123"
  }'
```

**Body Request (JSON)**:
```json
{
  "message": "Căn hộ 2PN tại dự án Elysian giá bao nhiêu?",
  "session_id": "sess_abc123", // (Optional) Truyền nếu đang tiếp tục chat
  "project_name": "Elysian", // (Optional) Gợi ý cho AI biết đang hỏi dự án nào
  "customer_name": "Nguyen Van A", // (Optional) Truyền nếu user đang hỏi đặt chỗ
  "customer_phone": "0901234567"   // (Optional) Truyền nếu user đang hỏi đặt chỗ
}
```

**Body Response (JSON)**:
```json
{
  "session_id": "sess_abc123",
  "answer": "Hiện tại căn hộ 2PN tại Elysian có giá tham khảo từ...",
  "intent": "sales_inquiry", // Intent AI nhận diện: customer_support, sales_inquiry, v.v.
  "sources": [
    {
      "document_code": "CSBH_Elysian_01",
      "document_name": "Chính sách bán hàng",
      "doc_group": "chinh_sach",
      "excerpt": "Căn 2PN giá từ...",
      "page": 1
    }
  ],
  "tool_calls": [
    {
      "tool_name": "inventory_lookup",
      "input_summary": "Tìm căn 2PN",
      "output_summary": "Còn 5 căn",
      "duration_ms": 150,
      "success": true
    }
  ],
  "fallback": false, // True nếu Bot bị lỗi/không thể trả lời
  "fallback_reason": "",
  "was_injected": false,
  "project_name": "Elysian", // Dự án AI thực tế đã nhận diện được
  "response_time_ms": 2340,
  "suggested_questions": [
    "Cho tôi xem thiết kế căn 2PN",
    "Chính sách thanh toán thế nào?",
    "Có hỗ trợ vay ngân hàng không?"
  ],
  "sales_data": {} // Dữ liệu thô (nếu FE cần render UI dạng thẻ/danh sách)
}
```

**Chi tiết thêm cho FE**:
- **`suggested_questions`**: Danh sách 3 câu hỏi gợi ý tiếp theo để hiển thị thành các bong bóng (chips) cho người dùng bấm chọn nhanh.
- **`answer`**: Nội dung câu trả lời, trả về định dạng Markdown. FE cần sử dụng thư viện render Markdown (như `react-markdown`).
- **`sources`**: Nguồn tài liệu Bot đã tham khảo. FE có thể tạo UI "Trích xuất từ tài liệu..." dưới mỗi câu trả lời.

---

### 3.2. API Lấy Danh Sách Dự Án (`GET /api/v1/projects`)

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

### 3.3. API Kiểm Tra Trạng Thái Hệ Thống (`GET /health`)

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

### 3.4. Các Mã Lỗi (HTTP Status Codes)

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
