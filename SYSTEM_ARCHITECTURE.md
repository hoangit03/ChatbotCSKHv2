# Kiến trúc hệ thống Chatbot CSKH & Bán hàng (LangGraph)

Tài liệu này mô tả chi tiết kiến trúc của Agentic Chatbot sử dụng LangGraph, giúp kiểm soát luồng hội thoại và trạng thái khách hàng một cách chặt chẽ.

## 1. Đồ thị luồng xử lý (Agent Graph)

Dưới đây là sơ đồ Mermaid mô tả các Node và Edge trong hệ thống:

```mermaid
graph TD
    %% Nodes
    START((Bắt đầu))
    Classify[<b>1. Intent Classifier</b><br/>Phân loại ý định khách hàng]
    Guard[<b>2. Project Guard</b><br/>Kiểm soát ngữ cảnh dự án]
    Support[<b>3a. Support Node</b><br/>Tra cứu RAG / Kiến thức]
    Sales[<b>3b. Sales Node</b><br/>Xử lý Bán hàng / API Tồn kho]
    Synth[<b>4. Synthesizer</b><br/>Tổng hợp câu trả lời]
    END((Kết thúc))

    %% Edges
    START --> Classify
    Classify --> Guard

    %% Conditional Edges sau Guard
    Guard -- "Yêu cầu chọn dự án" --> END
    Guard -- "Hỏi thông tin chung/pháp lý" --> Support
    Guard -- "Hỏi giá/tồn kho/đặt chỗ" --> Sales
    Guard -- "Chào hỏi/Tán gẫu" --> Synth

    %% Luồng hội tụ
    Support --> Synth
    Sales --> Synth
    Synth --> END

    %% Styling
    style START fill:#f9f,stroke:#333,stroke-width:2px
    style END fill:#f9f,stroke:#333,stroke-width:2px
    style Guard fill:#fff4dd,stroke:#d4a017,stroke-width:2px
    style Classify fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style Support fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    style Sales fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    style Synth fill:#fff3e0,stroke:#e65100,stroke-width:2px
```

## 2. Chi tiết các Node xử lý

### 1. Intent Classifier
Sử dụng LLM để phân loại yêu cầu của người dùng vào các nhóm chính:
- `customer_support`: Câu hỏi về thông tin, pháp lý, vị trí.
- `sales_inquiry`: Câu hỏi về giá, tình trạng căn hộ.
- `booking_intent`: Muốn giữ chỗ hoặc đặt lịch.
- `chitchat`: Chào hỏi thông thường.

### 2. Project Guard (Chốt chặn quan trọng)
Kiểm tra xem câu hỏi có yêu cầu ngữ cảnh dự án cụ thể hay không. Nếu khách hàng chưa chọn dự án, node này sẽ trả về yêu cầu chọn dự án và kết thúc luồng ngay lập tức để tránh trả lời sai.

### 3a. Support Node (RAG Pipeline)
Kết nối với **Qdrant (Vector Database)** để trích xuất các đoạn văn bản liên quan đến dự án đã chọn. Sử dụng kỹ thuật RAG để trả lời chính xác dựa trên tài liệu pháp lý và tài liệu dự án.

### 3b. Sales Node (Sales Intelligence)
Tương tác với **Sales API Adapter** để lấy dữ liệu thời gian thực:
- Tra cứu bảng hàng, giá bán.
- Tính toán mức độ khan hiếm (`ScarcityLevel`).
- Áp dụng kịch bản bán hàng dựa trên giai đoạn tâm lý (`CustomerStage`).

### 4. Synthesizer
Nhận dữ liệu từ các node chuyên môn và biên soạn lại thành một câu trả lời hoàn chỉnh. Đảm bảo:
- Văn phong chuyên nghiệp, thân thiện.
- Đầy đủ nguồn trích dẫn (`SourceRef`).
- Đề xuất bước tiếp theo (Call to Action).

## 3. Quản lý trạng thái (Agent State)

Hệ thống sử dụng một State chung truyền qua các node, bao gồm:
- `messages`: Lịch sử hội thoại.
- `customer_stage`: Giai đoạn của khách hàng (Awareness, Consideration, Decision).
- `usps_used`: Danh sách các điểm bán hàng độc đáo đã được giới thiệu.
- `scarcity_level`: Mức độ khan hiếm căn hộ để tạo FOMO phù hợp.
- `project_name`: Tên dự án đang được thảo luận.

---
*Tài liệu này được tạo tự động bởi AI Coding Assistant.*
