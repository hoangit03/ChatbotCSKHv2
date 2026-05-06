import streamlit as st
import httpx
import json
import time
import uuid
from datetime import datetime

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CTlotus AI | Chuyên viên tư vấn BĐS",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

def apply_custom_style():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;500;600;800&display=swap');
        
        :root {
            --primary: #4F46E5;
            --primary-light: #818CF8;
            --secondary: #ec4899;
            --bg-dark: #0f172a;
            --surface: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
        }

        /* Base Typography */
        .stApp {
            font-family: 'Inter', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 800 !important;
        }
        
        /* Gradient Titles */
        .gradient-text {
            background: linear-gradient(135deg, var(--primary-light), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            font-family: 'Outfit', sans-serif;
        }
        
        .sub-text {
            color: var(--text-muted);
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: var(--surface);
            border-right: 1px solid var(--border);
        }
        
        /* Chat bubble glassmorphism */
        .stChatMessage {
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 16px;
            border: 1px solid var(--border);
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(10px);
            transition: transform 0.2s ease, border-color 0.2s ease;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        
        .stChatMessage:hover {
            border-color: var(--primary-light);
            transform: translateY(-2px);
        }
        
        .stChatMessage[data-testid="stChatMessageUser"] {
            background: linear-gradient(145deg, rgba(79, 70, 229, 0.15), rgba(79, 70, 229, 0.05));
            border: 1px solid rgba(79, 70, 229, 0.3);
        }

        /* Source card styling */
        .source-card {
            background: var(--surface);
            border-radius: 12px;
            padding: 16px;
            border-left: 4px solid var(--primary);
            margin: 12px 0;
            font-size: 0.95rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            transition: all 0.2s ease;
            color: var(--text-main);
        }
        
        .source-card:hover {
            transform: translateX(4px);
            border-left-color: var(--secondary);
            background: #253347;
        }

        .source-title {
            color: var(--primary-light);
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 1.05rem;
        }

        /* Beautiful Buttons */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border: none;
            color: white;
            transition: all 0.3s ease;
            padding: 0.5rem 1rem;
        }
        
        .stButton>button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(236, 72, 153, 0.3);
            color: white;
        }
        
        /* Expander UI */
        .streamlit-expanderHeader {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            color: var(--primary-light);
            border-radius: 8px;
        }
        
        /* Tabs UI */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            font-weight: 600;
            padding: 10px 20px;
        }
        
        /* Spinner */
        .stSpinner > div > div {
            border-color: var(--primary) transparent var(--secondary) transparent !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# API CLIENT
# ─────────────────────────────────────────────────────────────────

class ChatbotAPI:
    def __init__(self, base_url="http://localhost:8000", api_key="chatbot-ctlotus"):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.Client(timeout=60.0)

    def check_health(self):
        try:
            resp = self.client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except:
            return False

    def send_message(self, message, session_id=None, project_name=None):
        payload = {
            "message": message,
            "session_id": session_id,
            "project_name": project_name
        }
        headers = {"X-API-Key": self.api_key}
        resp = self.client.post(f"{self.base_url}/api/v1/chat", json=payload, headers=headers)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Lỗi kết nối API ({resp.status_code}): {resp.text}")
            return None

    def get_projects(self):
        try:
            headers = {"X-API-Key": self.api_key}
            resp = self.client.get(f"{self.base_url}/api/v1/projects", headers=headers)
            if resp.status_code == 200:
                return resp.json().get("projects", [])
            return []
        except:
            return []

    def upload_document(self, file, project_name, doc_group="Tài liệu dự án"):
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {
            "project_name": project_name, 
            "doc_group": doc_group,
            "version": "1.0",
            "effective_date": datetime.utcnow().isoformat()
        }
        headers = {"X-API-Key": self.api_key}
        resp = self.client.post(f"{self.base_url}/api/v1/documents/upload", files=files, data=data, headers=headers)
        if resp.status_code == 201 or resp.status_code == 200:
            return resp.json()
        st.error(f"Lỗi Upload ({resp.status_code}): {resp.text}")
        return None

# ─────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "api_url" not in st.session_state:
        st.session_state.api_url = "http://localhost:8000"
    if "api_key" not in st.session_state:
        st.session_state.api_key = "chatbot-ctlotus"
    if "last_bot_time" not in st.session_state:
        st.session_state.last_bot_time = None     # Timestamp sau bot reply cuối
    if "follow_up_sent" not in st.session_state:
        st.session_state.follow_up_sent = False   # Chỉ gửi follow-up 1 lần/turn


def _send_message_and_append(api_client: ChatbotAPI, text: str):
    """
    Gửi tin nhắn lên API, cập nhật session state và rerun.
    Hàm tái sử dụng cho cả chat input và suggested question click.
    """
    st.session_state.messages.append({"role": "user", "content": text})
    response = api_client.send_message(
        message=text,
        session_id=st.session_state.session_id,
        project_name=st.session_state.get("project_name"),
    )
    if response:
        detected_project = response.get("project_name")
        if detected_project and detected_project != st.session_state.get("project_name"):
            st.session_state.project_name = detected_project
        st.session_state.messages.append({
            "role":               "assistant",
            "content":            response["answer"],
            "sources":            response.get("sources", []),
            "tool_calls":         response.get("tool_calls", []),
            "suggested_questions": response.get("suggested_questions", []),
        })
        # Reset follow-up timer
        st.session_state.last_bot_time = time.time()
        st.session_state.follow_up_sent = False
    st.rerun()


def _render_suggested_questions(
    api_client: ChatbotAPI,
    questions: list[str],
    key_prefix: str = "sq",
):
    """
    Hiển thị 3 câu hỏi gợi ý dưới dạng pill buttons.
    Khi click → tự động gửi cho agent.
    """
    if not questions:
        return
    st.markdown(
        '<div style="margin-top: 10px; margin-bottom: 4px; font-size: 0.82rem; color: #94a3b8;">'
        '✨ Gợi ý câu hỏi:'
        '</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(questions))
    for i, q in enumerate(questions[:3]):
        if cols[i].button(
            q,
            key=f"{key_prefix}_q{i}",
            use_container_width=True,
        ):
            _send_message_and_append(api_client, q)

def render_sidebar(api_client: ChatbotAPI):
    with st.sidebar:
        # Header Sidebar
        st.markdown("<h2 style='text-align: center; margin-bottom: 0;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; margin-bottom: 20px;'>Hệ thống quản trị</p>", unsafe_allow_html=True)
        
        # System Health
        is_alive = api_client.check_health()
        status_color = "#22c55e" if is_alive else "#ef4444"
        status_text = "Hoạt động tốt" if is_alive else "Mất kết nối"
        
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.5); padding: 15px; border-radius: 10px; border: 1px solid #334155; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span style="font-weight: 600; color: #f8fafc;">Trạng thái Server</span>
                <span style="display: flex; align-items: center; gap: 6px; color: {status_color}; font-size: 0.9rem;">
                    <div style="width: 8px; height: 8px; border-radius: 50%; background-color: {status_color}; box-shadow: 0 0 8px {status_color};"></div>
                    {status_text}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Project Selection
        st.markdown("### 📌 Bối cảnh dự án")
        raw_projects = api_client.get_projects()
        projects = ["Tất cả dự án"] + raw_projects if raw_projects else ["Tất cả dự án"]
            
        current_selection = st.session_state.get("project_selection", "Tất cả dự án")
        try:
            default_idx = projects.index(current_selection)
        except ValueError:
            default_idx = 0
            
        st.session_state.project_selection = st.selectbox(
            "Chọn dự án để Agent tập trung hỗ trợ:",
            projects,
            index=default_idx,
            label_visibility="collapsed"
        )
        
        # Map selection to project_name for API
        if st.session_state.project_selection == "Tất cả dự án":
            st.session_state.project_name = None
        else:
            st.session_state.project_name = st.session_state.project_selection
        
        st.markdown("<hr style='border-color: #334155; margin: 25px 0;'>", unsafe_allow_html=True)
        
        # Settings Expander
        with st.expander("🛠️ Cấu hình hệ thống"):
            st.session_state.api_url = st.text_input("Backend URL", value=st.session_state.api_url)
            st.session_state.api_key = st.text_input("API Key", value=st.session_state.api_key, type="password")
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Xóa lịch sử trò chuyện"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.toast("Đã làm mới phiên trò chuyện!", icon="✨")
            st.rerun()

def render_chat_interface(api_client: ChatbotAPI):
    # Header
    st.markdown('<div class="gradient-text">CTlotus AI Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Chuyên viên tư vấn & chăm sóc khách hàng bất động sản 24/7</div>', unsafe_allow_html=True)

    # Khung chứa Chat
    chat_container = st.container()

    with chat_container:
        # Initial greeting if empty
        if not st.session_state.messages:
            st.info("👋 Chào bạn! Tôi là chuyên viên tư vấn AI của CTlotus. Bạn cần tìm hiểu thông tin, pháp lý, bảng giá hay đặt chỗ cho dự án nào?")
            # Gợi ý câu hỏi mặc định khi chưa có hội thoại
            _render_suggested_questions(
                api_client,
                [
                    "🏢 Hiện tại có những dự án nào đang mở bán?",
                    "💰 Mức giá tối thiểu của các dự án là bao nhiêu?",
                    "📍 Chính sách thanh toán và vay vốn như thế nào?",
                ],
                key_prefix="welcome",
            )

        # Display chat history
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Hiển thị Nguồn (Sources)
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Xem nguồn tham khảo (RAG/QA)"):
                        for src in message["sources"]:
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-title">📄 {src['document_name']}</div>
                                <div style="color: #cbd5e1; line-height: 1.5;">{src['excerpt']}</div>
                            </div>
                            """, unsafe_allow_html=True)

                # Hiển thị Agent Tool Traces
                if "tool_calls" in message and message["tool_calls"]:
                    with st.expander("⚙️ Xem quá trình Agent suy luận"):
                        for tool in message["tool_calls"]:
                            status_icon = "✅" if tool.get("success") else "❌"
                            st.markdown(f"**{status_icon} {tool['tool_name']}** `{tool['duration_ms']}ms`")
                            st.markdown(f"""
                            <div style="background: rgba(15, 23, 42, 0.6); padding: 10px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 10px;">
                                <div style="font-family: monospace; font-size: 0.85rem; color: #94a3b8;">
                                    <b>Input:</b> {tool['input_summary']}<br/>
                                    <b>Output:</b> {tool['output_summary']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # [NEW] Hiển thị suggested_questions ở cuối tin nhắn bot cuối cùng
                is_last_bot = (
                    message["role"] == "assistant"
                    and idx == len(st.session_state.messages) - 1
                )
                if is_last_bot and message.get("suggested_questions"):
                    _render_suggested_questions(
                        api_client,
                        message["suggested_questions"],
                        key_prefix=f"suggest_{idx}",
                    )

    # Chat Input Box
    if prompt := st.chat_input("Nhập câu hỏi của bạn (VD: Giá căn hộ 2PN dự án Metro Star bao nhiêu?)..."):
        # Reset follow-up state khi user chủ động gỏ
        st.session_state.follow_up_sent = False
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent đang phân tích và tìm kiếm dữ liệu..."):
                response = api_client.send_message(
                    message=prompt,
                    session_id=st.session_state.session_id,
                    project_name=st.session_state.get("project_name"),
                )

                if response:
                    answer            = response["answer"]
                    sources           = response.get("sources", [])
                    tool_calls        = response.get("tool_calls", [])
                    suggested         = response.get("suggested_questions", [])
                    detected_project  = response.get("project_name")
                    sales_data        = response.get("sales_data", {})

                    if detected_project and detected_project != st.session_state.get("project_name"):
                        st.session_state.project_name = detected_project
                        st.toast(f"Hệ thống đã tự động chuyển bối cảnh sang dự án: **{detected_project}**", icon="🎯")

                    st.markdown(answer)

                    # Dự án list buttons
                    project_list = sales_data.get("project_list", [])
                    if project_list:
                        st.markdown("---")
                        st.markdown("### 🎯 Dự án đề xuất")
                        cols = st.columns(min(len(project_list), 4))
                        for pidx, p_data in enumerate(project_list):
                            if isinstance(p_data, dict):
                                prj_name = p_data.get("name", "Unknown")
                                prj_loc = p_data.get("province") or p_data.get("address") or ""
                                btn_text = f"Chọn: {prj_name}"
                                if prj_loc:
                                    btn_text += f" ({prj_loc})"
                            else:
                                prj_name = str(p_data)
                                btn_text = f"Chọn: {prj_name}"

                            if cols[pidx % 4].button(btn_text, key=f"btn_prj_{pidx}"):
                                _send_message_and_append(
                                    api_client,
                                    f"Tôi muốn tìm hiểu về dự án {prj_name}",
                                )

                    # Lưu message vào session (kèm suggested_questions)
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.messages.append({
                        "role":               "assistant",
                        "content":            answer,
                        "sources":            sources,
                        "tool_calls":         tool_calls,
                        "suggested_questions": suggested,
                    })

                    # Cập nhật timer cho auto follow-up
                    st.session_state.last_bot_time = time.time()
                    st.session_state.follow_up_sent = False

                    # Hiển thị sources / tool traces cho response hiện tại
                    if sources:
                        with st.expander("📚 Xem nguồn tham khảo (RAG/QA)"):
                            for src in sources:
                                st.markdown(f"""
                                <div class="source-card">
                                    <div class="source-title">📄 {src['document_name']}</div>
                                    <div style="color: #cbd5e1; line-height: 1.5;">{src['excerpt']}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    if tool_calls:
                        with st.expander("⚙️ Xem quá trình Agent suy luận"):
                            for tool in tool_calls:
                                status_ic = "✅" if tool.get("success") else "❌"
                                st.markdown(f"**{status_ic} {tool['tool_name']}** `{tool['duration_ms']}ms`")
                                st.markdown(f"""
                                <div style="background: rgba(15, 23, 42, 0.6); padding: 10px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 10px;">
                                    <div style="font-family: monospace; font-size: 0.85rem; color: #94a3b8;">
                                        <b>Input:</b> {tool['input_summary']}<br/>
                                        <b>Output:</b> {tool['output_summary']}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.rerun()
                else:
                    st.error("Không thể kết nối đến AI Server. Vui lòng thử lại sau.")

    # [NEW] Auto follow-up sau 15s nếu khách không phản hồi
    FOLLOW_UP_DELAY = 15  # giây
    FOLLOW_UP_MSG = (
        "👋 Bạn có cần hỗ trợ thêm không? "
        "Tôi rất vui được giải đáp thắc mắc về giá, pháp lý hoặc cập nhật tình hình dự án!"
    )
    last_time = st.session_state.get("last_bot_time")
    follow_sent = st.session_state.get("follow_up_sent", False)
    if (
        last_time is not None
        and not follow_sent
        and st.session_state.messages  # có hội thoại
        and st.session_state.messages[-1]["role"] == "assistant"  # bot vừa reply
        and (time.time() - last_time) >= FOLLOW_UP_DELAY
    ):
        st.session_state.follow_up_sent = True
        st.session_state.messages.append({
            "role":               "assistant",
            "content":            FOLLOW_UP_MSG,
            "sources":            [],
            "tool_calls":         [],
            "suggested_questions": [
                "💰 Giá dự án cụ thể bao nhiêu?",
                "📊 Tình trạng tồn kho hiện tại thế nào?",
                "📅 Tôi muốn đặt lịch xem nhà mẫu.",
            ],
        })
        st.rerun()

def render_upload_page(api_client: ChatbotAPI):
    st.markdown('<div class="gradient-text">Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Huấn luyện AI bằng cách nạp thêm tài liệu, chính sách, Q&A</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("upload_form", clear_on_submit=True):
            st.markdown("### 📤 Tải lên tài liệu mới")
            uploaded_files = st.file_uploader("Kéo thả file (PDF, DOCX, XLSX)", accept_multiple_files=True)
            
            project = st.text_input("Tên dự án (Ví dụ: Vinhomes_GrandPark)", value=st.session_state.get("project_name", ""))
            group = st.selectbox("Phân loại tài liệu", ["Chính sách bán hàng", "Pháp lý", "Tiến độ", "Kỹ thuật", "Brochure", "Bộ Q&A"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("🚀 Huấn luyện AI ngay")
            
            if submit:
                if not uploaded_files:
                    st.warning("Vui lòng chọn ít nhất 1 file!")
                elif not project.strip():
                    st.warning("Vui lòng nhập tên dự án!")
                else:
                    for file in uploaded_files:
                        with st.status(f"Đang phân tích và nhúng vector cho {file.name}...") as status:
                            res = api_client.upload_document(file, project, group)
                            if res:
                                status.update(label=f"✅ Nạp thành công: {file.name}!", state="complete")
                                st.balloons()
                            else:
                                status.update(label=f"❌ Xử lý thất bại: {file.name}", state="error")

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    apply_custom_style()
    init_session_state()
    
    api_client = ChatbotAPI(
        base_url=st.session_state.api_url,
        api_key=st.session_state.api_key
    )
    
    render_sidebar(api_client)
    
    # Elegant Navigation
    tab1, tab2 = st.tabs(["💬 Tương tác AI", "📂 Quản trị Dữ liệu"])
    
    with tab1:
        render_chat_interface(api_client)
    
    with tab2:
        render_upload_page(api_client)

if __name__ == "__main__":
    main()
