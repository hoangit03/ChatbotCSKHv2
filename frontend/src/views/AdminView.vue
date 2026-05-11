<template>
  <div class="admin-page page-content active">
    <div class="page-header">
      <h2>QUẢN TRỊ HỆ THỐNG</h2>
      <p class="text-sm text-gray">Quản lý User & Giám sát Phiên Chat</p>
    </div>

    <div class="admin-tabs mt-20">
      <button :class="['tab-btn', currentTab === 'users' ? 'active' : '']" @click="currentTab = 'users'">
        <i class="fa-solid fa-users"></i> Người Dùng
      </button>
      <button :class="['tab-btn', currentTab === 'sessions' ? 'active' : '']" @click="currentTab = 'sessions'">
        <i class="fa-solid fa-comments"></i> Lịch Sử Chat
      </button>
    </div>

    <!-- USERS TAB -->
    <div v-show="currentTab === 'users'" class="tab-content active mt-20 fade-in">
      <div class="flex justify-between items-center mb-20">
        <h3 class="gradient-text">Quản Lý Người Dùng</h3>
        <button class="btn-glow-small" @click="openUserModal()">
          <i class="fa-solid fa-plus"></i> Thêm User Mới
        </button>
      </div>
      
      <div class="table-responsive glass-panel glass-morphism">
        <table class="data-table">
          <thead>
            <tr>
              <th>Email</th>
              <th>Họ tên</th>
              <th>Phân quyền (Domain)</th>
              <th>Trạng thái</th>
              <th>Hành động</th>
            </tr>
          </thead>
          <tbody>
            <tr v-if="loadingUsers">
              <td colspan="5" class="text-center"><span class="loader"></span></td>
            </tr>
            <tr v-else-if="users.length === 0">
              <td colspan="5" class="text-center text-gray">Chưa có người dùng nào.</td>
            </tr>
            <tr v-else v-for="u in users" :key="u.id">
              <td class="font-bold">{{ u.email }}</td>
              <td>{{ u.full_name }}</td>
              <td>
                <div class="role-badges-container">
                  <span v-for="r in u.roles" :key="r.tenant_id" class="role-badge" :class="'role-' + r.role.toLowerCase()">
                    {{ r.tenant_id.toUpperCase() }} ({{ r.role }})
                  </span>
                </div>
              </td>
              <td>
                <span class="status-badge" :class="u.is_active ? 'status-active' : 'status-inactive'">
                  <i :class="u.is_active ? 'fa-solid fa-check-circle' : 'fa-solid fa-ban'"></i> 
                  {{ u.is_active ? 'Hoạt động' : 'Đã khóa' }}
                </span>
              </td>
              <td>
                <div class="action-buttons">
                  <button class="btn-icon btn-edit" @click="openUserModal(u)" title="Sửa">
                    <i class="fa-solid fa-pen"></i>
                  </button>
                  <button class="btn-icon btn-delete" @click="deleteUser(u.id)" title="Xóa">
                    <i class="fa-solid fa-trash"></i>
                  </button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- SESSIONS TAB -->
    <div v-show="currentTab === 'sessions'" class="tab-content active mt-20 fade-in">
      <div class="flex justify-between items-center mb-20">
        <h3 class="gradient-text">Lịch Sử Phiên Chat <span class="text-sm text-gray">(Domain: {{ props.tenant.toUpperCase() }})</span></h3>
      </div>
      
      <div class="table-responsive glass-panel glass-morphism">
        <table class="data-table">
          <thead>
            <tr>
              <th>Mã Phiên</th>
              <th>Email User</th>
              <th>Số Tin Nhắn</th>
              <th>Thời gian tạo</th>
            </tr>
          </thead>
          <tbody>
            <tr v-if="loadingSessions">
              <td colspan="4" class="text-center"><span class="loader"></span></td>
            </tr>
            <tr v-else-if="sessions.length === 0">
              <td colspan="4" class="text-center text-gray">Chưa có phiên chat nào.</td>
            </tr>
            <tr v-else v-for="s in sessions" :key="s.id" class="clickable-row" @click="openSessionModal(s.id)" title="Bấm để xem chi tiết">
              <td class="font-mono text-primary"><i class="fa-solid fa-hashtag"></i> {{ s.id.substring(0,8) }}</td>
              <td class="font-bold">{{ s.user_email }}</td>
              <td><span class="count-badge">{{ s.message_count }}</span></td>
              <td class="text-gray">{{ new Date(s.created_at).toLocaleString('vi-VN') }}</td>
            </tr>
          </tbody>
        </table>
        
        <!-- Pagination controls -->
        <div class="pagination-container" v-if="totalPages > 1">
          <button class="btn-page" :disabled="currentPage === 1" @click="changePage(currentPage - 1)">
            <i class="fa-solid fa-chevron-left"></i>
          </button>
          <span class="page-info">Trang {{ currentPage }} / {{ totalPages }}</span>
          <button class="btn-page" :disabled="currentPage === totalPages" @click="changePage(currentPage + 1)">
            <i class="fa-solid fa-chevron-right"></i>
          </button>
        </div>
      </div>
    </div>

    <!-- USER MODAL -->
    <div v-if="showUserModal" class="modal-overlay custom-scroll">
      <div class="modal-content glass-panel glass-morphism premium-modal" style="width: 550px;">
        <div class="modal-header">
          <h3 class="gradient-text"><i class="fa-solid fa-user-shield"></i> {{ editingUser ? 'Sửa Người Dùng' : 'Thêm Người Dùng' }}</h3>
          <button class="close-btn" @click="showUserModal = false"><i class="fa-solid fa-xmark"></i></button>
        </div>
        <div class="modal-body mt-20">
          <form @submit.prevent="saveUser">
            <div class="form-group mb-15">
              <label><i class="fa-solid fa-envelope"></i> Email <span v-if="editingUser" class="text-xs text-gray">(Không thể sửa)</span></label>
              <input type="email" v-model="userForm.email" class="glass-input custom-input" required :disabled="editingUser" placeholder="admin@domain.com" />
            </div>
            <div class="form-group mb-15">
              <label><i class="fa-solid fa-id-card"></i> Họ tên</label>
              <input type="text" v-model="userForm.full_name" class="glass-input custom-input" required placeholder="Nhập họ tên" />
            </div>
            <div class="form-group mb-15">
              <label><i class="fa-solid fa-lock"></i> Mật khẩu <span v-if="editingUser" class="text-xs text-gray">(Để trống nếu không đổi)</span></label>
              <input type="password" v-model="userForm.password" class="glass-input custom-input" :required="!editingUser" placeholder="Nhập mật khẩu" />
            </div>
            <div class="form-group mb-15" v-if="editingUser">
              <label><i class="fa-solid fa-toggle-on"></i> Trạng thái hoạt động</label>
              <div class="status-toggle mt-10">
                <label class="radio-label">
                  <input type="radio" v-model="userForm.is_active" :value="true" />
                  <span class="custom-radio active-radio">Hoạt động</span>
                </label>
                <label class="radio-label">
                  <input type="radio" v-model="userForm.is_active" :value="false" />
                  <span class="custom-radio inactive-radio">Khóa</span>
                </label>
              </div>
            </div>
            
            <div class="form-group mb-15">
              <label><i class="fa-solid fa-key"></i> Phân quyền Domain (Role)</label>
              <div class="domain-roles-list custom-scroll glass-panel">
                <div v-if="availableDomains.length === 0" class="text-center text-gray p-10">Đang tải danh sách domain...</div>
                <div v-for="d in availableDomains" :key="d.id" class="domain-role-item">
                  <span class="domain-name"><i class="fa-solid fa-globe"></i> {{ d.name.toUpperCase() }}</span>
                  <div class="custom-select-wrapper">
                    <select v-model="userForm.roles[d.id]" class="glass-input custom-select">
                      <option value="">-- Cấm --</option>
                      <option value="E">User</option>
                      <option value="M">Manager</option>
                      <option value="D">Admin</option>
                    </select>
                    <i class="fa-solid fa-chevron-down select-icon"></i>
                  </div>
                </div>
              </div>
            </div>

            <div class="modal-actions mt-30">
              <button type="button" class="btn-glow-small btn-cancel" @click="showUserModal = false">Hủy</button>
              <button type="submit" class="btn-glow-small btn-save"><i class="fa-solid fa-check"></i> Lưu</button>
            </div>
          </form>
        </div>
      </div>
    </div>

    <!-- SESSION DETAILS MODAL -->
    <div v-if="showSessionModal" class="modal-overlay">
      <div class="modal-content glass-panel glass-morphism chat-history-modal">
        <div class="modal-header">
          <div class="header-info">
            <h3 class="gradient-text"><i class="fa-solid fa-message"></i> Phiên Chat</h3>
            <span class="session-id-badge">ID: {{ selectedSessionId?.substring(0,8) }}</span>
          </div>
          <button class="close-btn" @click="showSessionModal = false"><i class="fa-solid fa-xmark"></i></button>
        </div>
        
        <div class="modal-body chat-messages-container custom-scroll" ref="sessionMsgList">
          <div v-if="loadingSessionDetails" class="loading-state">
            <span class="loader"></span>
            <p>Đang tải tin nhắn...</p>
          </div>
          <div v-else-if="sessionMessages.length === 0" class="empty-state">
            <i class="fa-solid fa-inbox"></i>
            <p>Không có tin nhắn nào.</p>
          </div>
          <div v-else class="messages-list">
            <!-- Newest message at the top due to backend ordering and CSS flex-direction -->
            <div v-for="(m, i) in sessionMessages" :key="i" class="message-wrapper" :class="m.role === 'user' ? 'user-wrapper' : 'ai-wrapper'">
              <div class="msg-avatar">
                <i :class="['fa-solid', m.role === 'user' ? 'fa-user' : 'fa-robot']"></i>
              </div>
              <div class="msg-bubble glass-panel">
                <div class="msg-meta">{{ new Date(m.created_at).toLocaleString('vi-VN') }}</div>
                <div class="msg-content markdown-body" v-html="m.role === 'user' ? m.content : renderMarkdown(m.content)"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';
import { api } from '../api';
import { marked } from 'marked';

const props = defineProps({
  tenant: { type: String, required: true },
  role: { type: String, required: true }
});

const currentTab = ref('users');

// User Management State
const users = ref([]);
const loadingUsers = ref(false);
const showUserModal = ref(false);
const editingUser = ref(null);
const availableDomains = ref([]);

const userForm = ref({
  email: '',
  full_name: '',
  password: '',
  is_active: true,
  roles: {}
});

// Sessions State
const sessions = ref([]);
const loadingSessions = ref(false);
const showSessionModal = ref(false);
const selectedSessionId = ref(null);
const sessionMessages = ref([]);
const loadingSessionDetails = ref(false);
const currentPage = ref(1);
const totalPages = ref(1);
const limit = 20;

const renderMarkdown = (text) => marked.parse(text || '');

const loadTenants = async () => {
  try {
    const tenants = await api.getTenants();
    availableDomains.value = tenants;
  } catch(e) {
    console.error("Lỗi tải danh sách domains:", e);
    // Fallback if API fails
    availableDomains.value = [
      {id: 'qtqd', name: 'qtqd'}, 
      {id: 'primer-diamond', name: 'primer-diamond'}
    ];
  }
};

const loadUsers = async () => {
  loadingUsers.value = true;
  try {
    users.value = await api.getUsers();
  } catch(e) {
    console.error(e);
  } finally {
    loadingUsers.value = false;
  }
};

const loadSessions = async (page = 1) => {
  loadingSessions.value = true;
  try {
    const response = await api.getSessions(props.tenant, page, limit);
    sessions.value = response.data || [];
    currentPage.value = response.page || 1;
    totalPages.value = Math.ceil((response.total || 0) / limit);
  } catch(e) {
    console.error(e);
    sessions.value = [];
  } finally {
    loadingSessions.value = false;
  }
};

const changePage = (newPage) => {
  if (newPage >= 1 && newPage <= totalPages.value) {
    loadSessions(newPage);
  }
};

watch(currentTab, (newVal) => {
  if (newVal === 'users') loadUsers();
  else if (newVal === 'sessions') {
    currentPage.value = 1;
    loadSessions();
  }
});

watch(() => props.tenant, () => {
  if (currentTab.value === 'sessions') {
    currentPage.value = 1;
    loadSessions();
  }
});

onMounted(async () => {
  await loadTenants();
  loadUsers();
});

// User Actions
const openUserModal = (user = null) => {
  editingUser.value = user;
  
  // Initialize roles object for all available domains
  const initialRoles = {};
  availableDomains.value.forEach(d => {
    initialRoles[d.id] = '';
  });

  if (user) {
    userForm.value = {
      email: user.email,
      full_name: user.full_name,
      password: '',
      is_active: user.is_active,
      roles: initialRoles
    };
    user.roles.forEach(r => {
      userForm.value.roles[r.tenant_id] = r.role;
    });
  } else {
    userForm.value = {
      email: '',
      full_name: '',
      password: '',
      is_active: true,
      roles: initialRoles
    };
  }
  showUserModal.value = true;
};

const saveUser = async () => {
  try {
    const payloadRoles = {};
    for (const [d, r] of Object.entries(userForm.value.roles)) {
      if (r) payloadRoles[d] = r;
    }
    
    if (editingUser.value) {
      const payload = {
        full_name: userForm.value.full_name,
        is_active: userForm.value.is_active,
        roles: payloadRoles
      };
      if (userForm.value.password) payload.password = userForm.value.password;
      await api.updateUser(editingUser.value.id, payload);
    } else {
      const payload = {
        email: userForm.value.email,
        full_name: userForm.value.full_name,
        password: userForm.value.password,
        roles: payloadRoles
      };
      await api.createUser(payload);
    }
    showUserModal.value = false;
    loadUsers();
  } catch (err) {
    alert("Lỗi lưu người dùng: " + (err.response?.data?.detail || err.message));
  }
};

const deleteUser = async (id) => {
  if (confirm("Bạn có chắc muốn xóa người dùng này?")) {
    try {
      await api.deleteUser(id);
      loadUsers();
    } catch(err) {
      alert("Lỗi xóa người dùng");
    }
  }
};

// Session Actions
const openSessionModal = async (id) => {
  selectedSessionId.value = id;
  showSessionModal.value = true;
  loadingSessionDetails.value = true;
  sessionMessages.value = [];
  try {
    const msgs = await api.getSessionMessages(props.tenant, id);
    sessionMessages.value = msgs;
  } catch (err) {
    console.error(err);
  } finally {
    loadingSessionDetails.value = false;
  }
};
</script>

<style scoped>
/* Tabs */
.admin-tabs {
  display: flex;
  gap: 15px;
  border-bottom: 2px solid rgba(255,255,255,0.05);
  padding-bottom: 15px;
}

.tab-btn {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255,255,255,0.1);
  color: #8892b0;
  padding: 10px 20px;
  border-radius: 12px;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 8px;
}

.tab-btn:hover {
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
}

.tab-btn.active {
  background: var(--primary-color);
  border-color: var(--primary-color);
  color: #000;
  box-shadow: 0 0 15px rgba(0, 243, 255, 0.3);
}

/* Typography */
.gradient-text {
  background: linear-gradient(135deg, #00f3ff, #0066ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0;
}

/* Tables */
.glass-morphism {
  background: rgba(15, 20, 35, 0.6);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 16px;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
  padding: 20px;
}

.data-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
}

.data-table th {
  padding: 15px;
  text-align: left;
  color: #8892b0;
  font-weight: 600;
  text-transform: uppercase;
  font-size: 0.85rem;
  letter-spacing: 1px;
  border-bottom: 1px solid rgba(255,255,255,0.1);
}

.data-table td {
  padding: 15px;
  text-align: left;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  vertical-align: middle;
}

.clickable-row {
  cursor: pointer;
  transition: background 0.2s;
}

.clickable-row:hover {
  background: rgba(0, 243, 255, 0.05);
}

/* Badges */
.role-badges-container {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}

.role-badge {
  font-size: 0.75rem;
  padding: 4px 8px;
  border-radius: 6px;
  font-weight: 600;
  background: rgba(255,255,255,0.1);
}
.role-e { background: rgba(0, 243, 255, 0.1); color: #00f3ff; border: 1px solid rgba(0, 243, 255, 0.2); }
.role-m { background: rgba(138, 43, 226, 0.1); color: #b975ff; border: 1px solid rgba(138, 43, 226, 0.2); }
.role-d { background: rgba(255, 77, 79, 0.1); color: #ff4d4f; border: 1px solid rgba(255, 77, 79, 0.2); }

.status-badge {
  font-size: 0.8rem;
  padding: 5px 10px;
  border-radius: 20px;
  font-weight: 600;
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.status-active { background: rgba(0, 255, 136, 0.1); color: #00ff88; border: 1px solid rgba(0, 255, 136, 0.2); }
.status-inactive { background: rgba(255, 77, 79, 0.1); color: #ff4d4f; border: 1px solid rgba(255, 77, 79, 0.2); }

.count-badge {
  background: rgba(255,255,255,0.1);
  padding: 4px 12px;
  border-radius: 12px;
  font-weight: bold;
}

/* Action Buttons */
.action-buttons {
  display: flex;
  gap: 8px;
}

.btn-icon {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.1);
  background: rgba(255,255,255,0.05);
  color: #fff;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}
.btn-edit:hover { background: rgba(0, 243, 255, 0.2); border-color: #00f3ff; color: #00f3ff; }
.btn-delete:hover { background: rgba(255, 77, 79, 0.2); border-color: #ff4d4f; color: #ff4d4f; }

.btn-glow-small {
  padding: 10px 20px;
  font-size: 0.9rem;
  font-weight: 600;
  border-radius: 10px;
  background: linear-gradient(135deg, var(--primary-color), #00a8ff);
  border: none;
  color: #000;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 8px;
}
.btn-glow-small:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(0, 243, 255, 0.4);
}

.btn-cancel {
  background: transparent;
  border: 1px solid rgba(255,255,255,0.2);
  color: #fff;
}
.btn-cancel:hover {
  background: rgba(255,255,255,0.1);
  box-shadow: none;
}
.btn-save {
  background: linear-gradient(135deg, #00ff88, #00cc6a);
}
.btn-save:hover {
  box-shadow: 0 5px 15px rgba(0, 255, 136, 0.4);
}

/* Pagination */
.pagination-container {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 15px;
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid rgba(255,255,255,0.05);
}

.btn-page {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  color: #fff;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}
.btn-page:not(:disabled):hover {
  background: var(--primary-color);
  color: #000;
  border-color: var(--primary-color);
}
.btn-page:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.page-info {
  font-size: 0.9rem;
  color: #8892b0;
  font-weight: 600;
}

/* Modals */
.premium-modal {
  border: 1px solid rgba(0, 243, 255, 0.2);
  box-shadow: 0 20px 50px rgba(0,0,0,0.5), inset 0 0 0 1px rgba(255,255,255,0.05);
}

.modal-header {
  border-bottom: 1px solid rgba(255,255,255,0.1);
  padding-bottom: 15px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.close-btn {
  background: rgba(255,255,255,0.05);
  border: none;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  color: #8892b0;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}
.close-btn:hover {
  background: rgba(255, 77, 79, 0.2);
  color: #ff4d4f;
}

/* Forms */
.custom-input {
  width: 100%;
  height: 44px;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  color: #fff;
  padding: 0 15px;
  margin-top: 8px;
  transition: all 0.3s;
}
.custom-input:focus {
  border-color: var(--primary-color);
  background: rgba(0, 0, 0, 0.5);
  box-shadow: 0 0 0 2px rgba(0, 243, 255, 0.1);
}
.custom-input:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.form-group label {
  font-size: 0.9rem;
  color: #e2e8f0;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* Radio Toggle */
.status-toggle {
  display: flex;
  gap: 15px;
}

.radio-label {
  cursor: pointer;
}
.radio-label input {
  display: none;
}
.custom-radio {
  display: inline-block;
  padding: 8px 20px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.1);
  background: rgba(255,255,255,0.05);
  transition: all 0.2s;
  font-size: 0.9rem;
}
.radio-label input:checked + .active-radio {
  background: rgba(0, 255, 136, 0.2);
  border-color: #00ff88;
  color: #00ff88;
}
.radio-label input:checked + .inactive-radio {
  background: rgba(255, 77, 79, 0.2);
  border-color: #ff4d4f;
  color: #ff4d4f;
}

/* Scrollable Domains List */
.domain-roles-list {
  max-height: 200px;
  overflow-y: auto;
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.05);
  background: rgba(0,0,0,0.2);
  margin-top: 8px;
}

.domain-role-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 15px;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}
.domain-role-item:last-child {
  border-bottom: none;
}

.domain-name {
  font-weight: 600;
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  gap: 8px;
}

.custom-select-wrapper {
  position: relative;
  width: 140px;
}
.custom-select {
  width: 100%;
  margin: 0;
  height: 36px;
  appearance: none;
  padding-right: 30px;
}
.select-icon {
  position: absolute;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  pointer-events: none;
  color: #8892b0;
  font-size: 0.8rem;
}

/* Chat History Modal Specifics */
.chat-history-modal {
  width: 700px;
  height: 85vh;
  display: flex;
  flex-direction: column;
  padding: 0;
  overflow: hidden;
}

.chat-history-modal .modal-header {
  padding: 20px;
  background: rgba(0,0,0,0.2);
}

.header-info {
  display: flex;
  align-items: center;
  gap: 15px;
}

.session-id-badge {
  background: rgba(255,255,255,0.1);
  padding: 4px 10px;
  border-radius: 6px;
  font-family: monospace;
  font-size: 0.8rem;
}

.chat-messages-container {
  flex-grow: 1;
  padding: 20px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.messages-list {
  display: flex;
  flex-direction: column-reverse; /* Reverses the flow to put newest at the top */
  gap: 20px;
}

.message-wrapper {
  display: flex;
  gap: 15px;
  width: 100%;
}

.user-wrapper {
  flex-direction: row-reverse;
}

.msg-avatar {
  width: 40px;
  height: 40px;
  border-radius: 12px;
  background: rgba(255,255,255,0.1);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  font-size: 1.2rem;
}
.user-wrapper .msg-avatar {
  background: rgba(0, 243, 255, 0.2);
  color: #00f3ff;
}
.ai-wrapper .msg-avatar {
  background: rgba(138, 43, 226, 0.2);
  color: #b975ff;
}

.msg-bubble {
  max-width: 80%;
  padding: 15px;
  border-radius: 16px;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.05);
}
.user-wrapper .msg-bubble {
  background: rgba(0, 243, 255, 0.05);
  border: 1px solid rgba(0, 243, 255, 0.1);
  border-top-right-radius: 4px;
}
.ai-wrapper .msg-bubble {
  background: rgba(138, 43, 226, 0.05);
  border: 1px solid rgba(138, 43, 226, 0.1);
  border-top-left-radius: 4px;
}

.msg-meta {
  font-size: 0.75rem;
  color: #8892b0;
  margin-bottom: 8px;
}

.empty-state, .loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #8892b0;
  gap: 15px;
}
.empty-state i {
  font-size: 3rem;
  opacity: 0.5;
}

/* Custom Scrollbar */
.custom-scroll::-webkit-scrollbar {
  width: 6px;
}
.custom-scroll::-webkit-scrollbar-track {
  background: rgba(0,0,0,0.1);
  border-radius: 3px;
}
.custom-scroll::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,0.2);
  border-radius: 3px;
}
.custom-scroll::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 243, 255, 0.5);
}

/* Utilities */
.flex { display: flex; }
.justify-between { justify-content: space-between; }
.items-center { align-items: center; }
.text-center { text-align: center; }
.text-gray { color: #8892b0; }
.text-sm { font-size: 0.9rem; }
.text-xs { font-size: 0.8rem; }
.font-bold { font-weight: bold; }
.font-mono { font-family: monospace; }
.text-primary { color: var(--primary-color); }
.mt-10 { margin-top: 10px; }
.mt-20 { margin-top: 20px; }
.mt-30 { margin-top: 30px; }
.mb-15 { margin-bottom: 15px; }
.mb-20 { margin-bottom: 20px; }
.p-10 { padding: 10px; }
.fade-in { animation: fadeIn 0.3s ease; }

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
