// Global State
let currentUser = null;
let currentTenant = null;
let currentSessionId = null;

// DOM Elements
const loginScreen = document.getElementById('login-screen');
const appScreen = document.getElementById('app-screen');
const pageContainer = document.getElementById('page-container');
const tenantSelect = document.getElementById('tenant-select');

// Generate UUID
function uuidv4() {
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
        (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
    );
}

// Initialization
async function init() {
    const token = api.getToken();
    if (token) {
        try {
            currentUser = await api.getMe();
            showApp();
        } catch (e) {
            api.removeToken();
            showLogin();
        }
    } else {
        showLogin();
    }
    
    setupEventListeners();
}

function showLogin() {
    loginScreen.classList.remove('hidden');
    appScreen.classList.add('hidden');
}

function showApp() {
    loginScreen.classList.add('hidden');
    appScreen.classList.remove('hidden');
    
    // Fill user info
    document.getElementById('user-fullname').textContent = currentUser.full_name;
    document.getElementById('user-email').textContent = currentUser.email;
    
    // Fill tenants
    tenantSelect.innerHTML = '';
    const roles = currentUser.roles || {};
    const tenants = Object.keys(roles);
    
    if (tenants.length === 0) {
        pageContainer.innerHTML = '<div class="page-header"><h2>Không có quyền truy cập Domain nào</h2></div>';
        return;
    }
    
    tenants.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t;
        opt.textContent = t.toUpperCase();
        tenantSelect.appendChild(opt);
    });
    
    // Check if user is global admin or has 'D' role to show Admin tab
    let isAdmin = currentUser.email === 'admin@ctgroup.vn';
    Object.values(roles).forEach(r => { if (r.role === 'D') isAdmin = true; });
    
    // Check URL path for specific tenant demo routing
    const pathParts = window.location.pathname.split('/').filter(p => p);
    const urlTenantId = pathParts.length > 0 ? pathParts[0] : null;
    
    if (urlTenantId && tenants.includes(urlTenantId)) {
        // Lock the UI to this specific tenant for demo purposes
        changeTenant(urlTenantId);
        document.querySelector('.domain-selector').style.display = 'none';
        
        // Hide Admin and ETL tabs regardless of role to prevent confusion in demo mode
        document.getElementById('nav-admin').style.display = 'none';
        document.getElementById('nav-etl').style.display = 'none';
    } else {
        // Normal mode
        if (isAdmin) document.getElementById('nav-admin').style.display = 'flex';
        // Set initial tenant
        changeTenant(tenants[0]);
    }
    
    // Default page
    navigate('chat');
}

function changeTenant(tenant) {
    currentTenant = tenant;
    tenantSelect.value = tenant;
    const roleInfo = currentUser.roles[tenant];
    document.getElementById('role-info').textContent = `Role: ${roleInfo.role} (Level ${roleInfo.level})`;
    
    // ETL Page visibility logic: Only Level D (level 4) can see the ETL tab for this tenant
    // Check URL to not override demo mode hidden tabs
    const pathParts = window.location.pathname.split('/').filter(p => p);
    const urlTenantId = pathParts.length > 0 ? pathParts[0] : null;
    
    if (!urlTenantId || !currentUser.roles[urlTenantId]) {
        if (roleInfo.level >= 4 || currentUser.email === 'admin@ctgroup.vn') {
            document.getElementById('nav-etl').style.display = 'flex';
        } else {
            document.getElementById('nav-etl').style.display = 'none';
            // If they are on ETL page, kick them to Chat
            if (document.querySelector('.nav-item.active').dataset.page === 'etl') {
                navigate('chat');
            }
        }
    }
    
    currentSessionId = uuidv4();
    if (document.querySelector('.nav-item.active').dataset.page === 'chat') {
        navigate('chat'); // Refresh chat page
    }
}

function navigate(page) {
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    document.querySelector(`.nav-item[data-page="${page}"]`).classList.add('active');
    
    const tpl = document.getElementById(`tpl-${page}`).content.cloneNode(true);
    pageContainer.innerHTML = '';
    pageContainer.appendChild(tpl);
    
    // Initialize page specific scripts
    if (page === 'chat') initChat();
    if (page === 'etl') initETL();
    if (page === 'admin') initAdmin();
}

function setupEventListeners() {
    // Login
    document.getElementById('login-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = document.getElementById('login-btn');
        const loader = btn.querySelector('.loader');
        const text = btn.querySelector('.btn-text');
        const error = document.getElementById('login-error');
        
        loader.classList.remove('hidden');
        text.classList.add('hidden');
        error.classList.add('hidden');
        
        try {
            await api.login(
                document.getElementById('email').value,
                document.getElementById('password').value
            );
            currentUser = await api.getMe();
            showApp();
        } catch (err) {
            error.classList.remove('hidden');
        } finally {
            loader.classList.add('hidden');
            text.classList.remove('hidden');
        }
    });

    // Logout
    document.getElementById('logout-btn').addEventListener('click', () => {
        api.removeToken();
        showLogin();
    });

    // Sidebar Nav
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            navigate(e.currentTarget.dataset.page);
        });
    });

    // Tenant change
    tenantSelect.addEventListener('change', (e) => {
        changeTenant(e.target.value);
    });
}

// Chat Page Logic
function initChat() {
    document.querySelector('.chat-page .tenant-name').textContent = currentTenant.toUpperCase();
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const messagesList = document.getElementById('chat-messages');
    // (Project selection is no longer used for chat as the LLM auto-detects it)


    // Handle Enter to submit, Shift+Enter for new line
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text) return;
        
        chatInput.value = '';
        appendMessage('user', text);
        
        // Assistant placeholder
        const msgEl = appendMessage('assistant', '<span class="loader"></span>');
        const contentEl = msgEl.querySelector('.msg-content');
        
        try {
            // Setup fetch for Server Sent Events / Streaming
            const roleSelect = document.getElementById('chat-role-select');
            const isSale = roleSelect && roleSelect.value === 'sale';
            const endpoint = isSale ? 'sale/chat/stream' : 'chat/stream';
            
            // Map payload field
            const payloadBody = { 
                message: text, 
                session_id: currentSessionId
            };

            const res = await fetch(`${GATEWAY_URL}/${currentTenant}/${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...api.getHeaders()
                },
                body: JSON.stringify(payloadBody)
            });

            if (!res.ok) throw new Error('API Error');

            const reader = res.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let fullText = "";
            contentEl.innerHTML = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');
                
                for (let line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6).trim();
                        if (dataStr === '[DONE]') break;
                        try {
                            const data = JSON.parse(dataStr);
                            if (data.session_id && !currentSessionId) {
                                currentSessionId = data.session_id;
                                localStorage.setItem('chat_session_id', currentSessionId);
                                await loadSessions();
                            }
                            if (data.text) {
                                fullText += data.text;
                                contentEl.innerHTML = marked.parse(fullText);
                                messagesList.scrollTop = messagesList.scrollHeight;
                            }
                            if (data.suggested_questions && data.suggested_questions.length > 0) {
                                let sugHtml = '<div class="suggested-questions" style="margin-top: 15px; display: flex; gap: 8px; flex-wrap: wrap;">';
                                data.suggested_questions.forEach(q => {
                                    sugHtml += `<button class="suggestion-btn" onclick="document.getElementById('chat-input').value='${q.replace(/'/g, "\\'")}'; document.getElementById('chat-form').dispatchEvent(new Event('submit'))">${q}</button>`;
                                });
                                sugHtml += '</div>';
                                contentEl.innerHTML += sugHtml;
                            }
                            if (data.sources && data.sources.length > 0) {
                                const sourcesHtml = `<div style="margin-top:10px; font-size:12px; color:var(--text-secondary)"><i>Nguồn: ${data.sources.map(s => s.doc || 'Tài liệu').join(', ')}</i></div>`;
                                contentEl.innerHTML += sourcesHtml;
                            }
                        } catch(e) {}
                    }
                }
            }
        } catch (err) {
            contentEl.innerHTML = `<span style="color: var(--danger)">Lỗi kết nối.</span>`;
        }
    });
}

function appendMessage(role, htmlContent) {
    const messagesList = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = `message ${role}`;
    
    const icon = role === 'user' ? 'fa-user' : 'fa-robot';
    
    div.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid ${icon}"></i></div>
        <div class="msg-content">${htmlContent}</div>
    `;
    messagesList.appendChild(div);
    messagesList.scrollTop = messagesList.scrollHeight;
    return div;
}

// ETL Page Logic
function initETL() {
    document.querySelector('.etl-page .tenant-name').textContent = currentTenant.toUpperCase();
    const form = document.getElementById('etl-form');
    const status = document.getElementById('upload-status');
    const btn = document.getElementById('upload-btn');
    const etlProjectInput = document.getElementById('etl-project-input');
    
    // Load projects
    if (etlProjectInput) {
        api.getProjects(currentTenant).then(data => {
            const projects = data.projects || [];
            etlProjectInput.innerHTML = '<option value="" disabled selected>-- Chọn dự án --</option>';
            projects.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p;
                opt.textContent = p;
                etlProjectInput.appendChild(opt);
            });
        }).catch(err => {
            console.error("Error loading projects for ETL:", err);
            etlProjectInput.innerHTML = '<option value="">-- Lỗi tải danh sách --</option>';
        });
    }
    
    // The UI only shows this page if level >= 4. Let's make sure the options are correct.
    const select = document.getElementById('doc-security-level');
    const level = currentUser.roles[currentTenant]?.level || 1;
    
    // Disable options above user's level
    Array.from(select.options).forEach(opt => {
        if (parseInt(opt.value) > level) {
            opt.disabled = true;
        }
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const files = document.getElementById('file-upload').files;
        if (!files.length) return;
        
        btn.disabled = true;
        status.style.color = 'white';
        status.innerHTML = `<span class="loader" style="display:inline-block; vertical-align:middle; width:14px; height:14px; margin-right:8px;"></span> Đang tải lên ${files.length} file...`;
        
        let successCount = 0;
        const minRole = select.value;
        const projectInput = document.getElementById('etl-project-input');
        const projectName = projectInput ? projectInput.value.trim() : "";
        if (!projectName) {
            alert("Vui lòng chọn dự án!");
            btn.disabled = false;
            return;
        }
        
        for (let i = 0; i < files.length; i++) {
            try {
                await api.uploadETL(currentTenant, files[i], minRole, false, projectName);
                successCount++;
            } catch (err) {
                if (err.message === "CONFIRM_OVERWRITE") {
                    if (confirm(`File "${files[i].name}" đã tồn tại nhưng có nội dung mới.\nBạn có chắc chắn muốn GHI ĐÈ và XÓA dữ liệu cũ trong cơ sở dữ liệu không?`)) {
                        try {
                            await api.uploadETL(currentTenant, files[i], minRole, true, projectName);
                            successCount++;
                        } catch (err2) {
                            console.error(err2);
                            alert(`Lỗi khi ghi đè ${files[i].name}: ${err2.message}`);
                        }
                    }
                } else {
                    console.error(err);
                    alert(`Lỗi upload ${files[i].name}: ${err.message}`);
                }
            }
        }
        
        status.style.color = successCount === files.length ? 'var(--success)' : 'var(--danger)';
        status.innerHTML = `Hoàn tất: ${successCount}/${files.length} file thành công.`;
        btn.disabled = false;
        form.reset();
    });
}

// Admin Page Logic
function initAdmin() {
    const tabBtns = document.querySelectorAll('.admin-tabs .tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.admin-tabs .tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            e.target.classList.add('active');
            document.getElementById(`tab-${e.target.dataset.tab}`).classList.add('active');
        });
    });

    loadAdminData();
}

async function loadAdminData() {
    try {
        const users = await api.getUsers();
        const uBody = document.getElementById('users-tbody');
        uBody.innerHTML = users.map(u => {
            const roles = u.roles.map(r => `<span class="role-badge">${r.tenant_id}(${r.role})</span>`).join(' ');
            return `
                <tr>
                    <td>${u.email}</td>
                    <td>${u.full_name}</td>
                    <td>${roles}</td>
                    <td>${u.is_active ? '✅' : '❌'}</td>
                </tr>
            `;
        }).join('');
    } catch(e) {}

    try {
        const sessions = await api.getSessions(currentTenant);
        const sBody = document.getElementById('sessions-tbody');
        if (sessions.length === 0) {
            sBody.innerHTML = '<tr><td colspan="4" style="text-align:center">Chưa có phiên chat nào</td></tr>';
        } else {
            sBody.innerHTML = sessions.map(s => `
                <tr class="clickable-row" data-id="${s.id}" style="cursor: pointer;" title="Bấm để xem lịch sử">
                    <td>${s.id.slice(0,8)}...</td>
                    <td>${s.user_email}</td>
                    <td>${s.message_count}</td>
                    <td>${new Date(s.created_at).toLocaleString('vi-VN')}</td>
                </tr>
            `).join('');
            
            // Thêm sự kiện click
            document.querySelectorAll('.clickable-row').forEach(row => {
                row.addEventListener('click', () => openSessionModal(row.dataset.id));
            });
        }
    } catch(e) {}
}

async function openSessionModal(sessionId) {
    const modal = document.getElementById('session-modal');
    const msgList = document.getElementById('session-messages-list');
    document.getElementById('session-modal-title').innerText = 'Lịch sử Chat: ' + sessionId.slice(0,8) + '...';
    
    msgList.innerHTML = '<div style="text-align:center; padding:20px;">Đang tải...</div>';
    modal.classList.remove('hidden');
    
    document.getElementById('close-session-modal').onclick = () => {
        modal.classList.add('hidden');
    };
    
    try {
        const msgs = await api.getSessionMessages(currentTenant, sessionId);
        if(msgs.length === 0) {
            msgList.innerHTML = '<div style="text-align:center; padding:20px;">Không có tin nhắn nào</div>';
            return;
        }
        
        msgList.innerHTML = '';
        msgs.forEach(m => {
            const el = document.createElement('div');
            el.className = `message ${m.role === 'user' ? 'user-message' : 'ai-message'}`;
            el.innerHTML = `
                <div class="msg-avatar"><i class="fa-solid ${m.role === 'user' ? 'fa-user' : 'fa-robot'}"></i></div>
                <div class="msg-content">
                    <div style="font-size:12px; opacity:0.7; margin-bottom:5px;">
                        ${new Date(m.created_at).toLocaleString('vi-VN')}
                    </div>
                    ${m.role === 'user' ? m.content : marked.parse(m.content)}
                </div>
            `;
            msgList.appendChild(el);
        });
        // Cuộn xuống cuối
        msgList.scrollTop = msgList.scrollHeight;
    } catch(e) {
        msgList.innerHTML = '<div style="color:red; text-align:center; padding:20px;">Lỗi: ' + e.message + '</div>';
    }
}

// Start App
init();
