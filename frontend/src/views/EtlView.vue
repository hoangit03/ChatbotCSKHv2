<template>
  <div class="etl-page page-content active">
    <div class="page-header">
      <h2>NẠP DỮ LIỆU - {{ props.tenant.toUpperCase() }}</h2>
      <p class="text-sm text-gray">Quản lý cơ sở tri thức AI</p>
    </div>

    <div class="etl-grid mt-20">
      <!-- Nạp dữ liệu -->
      <div class="etl-upload-section glass-panel glass-morphism">
        <div class="section-header">
          <h3 class="gradient-text"><i class="fa-solid fa-cloud-arrow-up"></i> Upload Tài Liệu</h3>
          <p class="text-xs text-gray mt-5">Hỗ trợ PDF, Word, Excel, Markdown</p>
        </div>
        
        <form @submit.prevent="handleUpload" id="etl-form" class="mt-20">
          <div v-if="props.tenant === 'primer-diamond'" class="form-group mb-20">
            <label><i class="fa-solid fa-building"></i> Chọn dự án:</label>
            <div class="custom-select-wrapper mt-5">
              <select v-model="selectedProject" class="glass-input custom-input" required>
                <option value="" disabled>-- Chọn dự án --</option>
                <option v-for="p in projects" :key="p" :value="p">{{ p }}</option>
              </select>
              <i class="fa-solid fa-chevron-down select-icon"></i>
            </div>
          </div>

          <div class="form-group mb-20">
            <label><i class="fa-solid fa-shield-halved"></i> Mức độ bảo mật (Role Level):</label>
            <div class="custom-select-wrapper mt-5">
              <select v-model="minRoleLevel" class="glass-input custom-input">
                <option value="1">Level 1 (Tất cả nhân viên)</option>
                <option value="2">Level 2 (Quản lý cấp trung)</option>
                <option value="3">Level 3 (Quản lý cấp cao)</option>
                <option value="4" :disabled="userLevel < 4">Level 4 (Giám đốc)</option>
              </select>
              <i class="fa-solid fa-chevron-down select-icon"></i>
            </div>
          </div>
          
          <div class="form-group mb-30">
            <label><i class="fa-solid fa-file-pdf"></i> Tệp đính kèm:</label>
            <div class="file-upload-wrapper mt-5">
              <input type="file" ref="fileInput" class="file-input-hidden" multiple required accept=".pdf,.doc,.docx,.xls,.xlsx,.md" @change="handleFileChange" />
              <div class="file-upload-box" @click="$refs.fileInput.click()">
                <i class="fa-solid fa-cloud-arrow-up upload-icon"></i>
                <p v-if="selectedFileNames.length === 0">Kéo thả file hoặc <span>nhấn để chọn</span></p>
                <p v-else class="text-primary">{{ selectedFileNames.length }} file đã được chọn</p>
              </div>
            </div>
          </div>
          
          <button type="submit" class="btn-glow w-full" :disabled="loading">
            <span v-if="loading" class="loader-small mr-10"></span>
            <i v-else class="fa-solid fa-upload mr-10"></i>
            {{ loading ? 'Đang nạp dữ liệu...' : 'Bắt đầu nạp dữ liệu' }}
          </button>
          
          <div v-if="statusMessage" class="status-box mt-20" :class="statusClass">
            <p v-html="statusMessage"></p>
          </div>
        </form>
      </div>

      <!-- Danh sách file đã nạp -->
      <div class="etl-files-section glass-panel glass-morphism">
        <div class="section-header flex justify-between items-center">
          <h3 class="gradient-text"><i class="fa-solid fa-database"></i> Dữ Liệu Đã Nạp</h3>
          <button class="btn-icon btn-refresh" @click="fetchFilesData" title="Làm mới">
            <i class="fa-solid fa-rotate-right" :class="{'spinning': fetchingFiles}"></i>
          </button>
        </div>
        
        <div class="file-tree-container mt-20">
          <div v-if="fetchingFiles" class="loading-state">
            <span class="loader"></span>
            <p class="mt-10">Đang tải cấu trúc dữ liệu...</p>
          </div>
          <div v-else-if="!filesData || Object.keys(filesData).length === 0" class="empty-state">
            <i class="fa-solid fa-folder-open"></i>
            <p class="mt-10">Hệ thống chưa có dữ liệu nào.</p>
          </div>
          <div v-else class="file-tree custom-scroll">
            <div v-for="(projectsMap, t_id) in filesData" :key="t_id" class="tenant-node mb-15">
              <div class="tree-header tenant-header">
                <i class="fa-solid fa-server"></i> Domain: {{ t_id.toUpperCase() }}
              </div>
              <div class="tenant-children ml-15 mt-10">
                <div v-for="(fileList, pName) in projectsMap" :key="pName" class="project-node mb-10">
                  <div class="tree-header project-header">
                    <i class="fa-solid fa-folder-open text-primary"></i> 
                    {{ pName === '_no_project' ? 'Dữ liệu chung' : 'Dự án: ' + pName }}
                    <span class="file-count badge">{{ fileList.length }}</span>
                  </div>
                  <ul class="file-list mt-5">
                    <li v-for="f in fileList" :key="f" class="file-item">
                      <i class="fa-solid fa-file-lines file-icon"></i> <span class="file-name">{{ f }}</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue';
import { api } from '../api';

const props = defineProps({
  tenant: { type: String, required: true },
  role: { type: String, required: true }
});

const projects = ref([]);
const selectedProject = ref('');
const minRoleLevel = ref('1');
const fileInput = ref(null);
const selectedFileNames = ref([]);
const loading = ref(false);
const statusMessage = ref('');
const statusClass = ref('');

const filesData = ref({});
const fetchingFiles = ref(false);

const userLevel = computed(() => {
  if (props.role === 'D') return 4;
  if (props.role === 'M') return 3;
  if (props.role === 'C') return 2;
  return 1;
});

const loadProjects = async () => {
  if (props.tenant === 'primer-diamond') {
    try {
      const data = await api.getProjects();
      projects.value = data.projects || [];
    } catch (err) {
      console.error(err);
    }
  }
};

const fetchFilesData = async () => {
  fetchingFiles.value = true;
  try {
    // Only fetch for current tenant
    // Wait, the backend returns all tenants, but we should just show the ones returned by the proxy
    const data = await api.getETLFiles(props.tenant);
    filesData.value = data;
  } catch (err) {
    console.error("Error fetching files", err);
  } finally {
    fetchingFiles.value = false;
  }
};

watch(() => props.tenant, () => {
  selectedProject.value = '';
  statusMessage.value = '';
  loadProjects();
  fetchFilesData();
});

onMounted(() => {
  loadProjects();
  fetchFilesData();
});

const handleFileChange = () => {
  if (fileInput.value && fileInput.value.files) {
    selectedFileNames.value = Array.from(fileInput.value.files).map(f => f.name);
  } else {
    selectedFileNames.value = [];
  }
};

const handleUpload = async () => {
  const files = fileInput.value.files;
  if (!files.length) return;
  
  if (props.tenant === 'primer-diamond' && !selectedProject.value) {
    alert("Vui lòng chọn dự án!");
    return;
  }

  loading.value = true;
  statusClass.value = '';
  statusMessage.value = `<span class="loader" style="display:inline-block; vertical-align:middle; width:14px; height:14px; margin-right:8px;"></span> Đang tải lên ${files.length} file...`;
  
  let successCount = 0;

  for (let i = 0; i < files.length; i++) {
    try {
      await api.uploadETL(props.tenant, files[i], minRoleLevel.value, false, selectedProject.value);
      successCount++;
    } catch (err) {
      if (err.message === "CONFIRM_OVERWRITE") {
        if (confirm(`File "${files[i].name}" đã tồn tại.\nBạn có chắc chắn muốn GHI ĐÈ và XÓA dữ liệu cũ trong cơ sở dữ liệu không?`)) {
          try {
            await api.uploadETL(props.tenant, files[i], minRoleLevel.value, true, selectedProject.value);
            successCount++;
          } catch (err2) {
            alert(`Lỗi khi ghi đè ${files[i].name}: ${err2.message}`);
          }
        }
      } else {
        alert(`Lỗi upload ${files[i].name}: ${err.message}`);
      }
    }
  }
  
  loading.value = false;
  statusClass.value = successCount === files.length ? 'status-success' : 'status-error';
  statusMessage.value = `<i class="fa-solid ${successCount === files.length ? 'fa-circle-check' : 'fa-triangle-exclamation'}"></i> Hoàn tất: ${successCount}/${files.length} file thành công.`;
  fileInput.value.value = '';
  selectedFileNames.value = [];
  
  fetchFilesData(); // Refresh list
};
</script>

<style scoped>
/* Grid Layout */
.etl-grid {
  display: grid;
  grid-template-columns: 1fr 1.2fr;
  gap: 25px;
  align-items: start;
}

@media (max-width: 1024px) {
  .etl-grid {
    grid-template-columns: 1fr;
  }
}

.glass-morphism {
  background: rgba(15, 20, 35, 0.6);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 16px;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
  padding: 25px;
}

.gradient-text {
  background: linear-gradient(135deg, #00f3ff, #0066ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0;
}

.section-header {
  border-bottom: 1px solid rgba(255,255,255,0.05);
  padding-bottom: 15px;
}

/* Form Styles */
.form-group label {
  font-size: 0.95rem;
  color: #e2e8f0;
  font-weight: 600;
}

.custom-input {
  width: 100%;
  height: 48px;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  color: #fff;
  padding: 0 15px;
  transition: all 0.3s;
}

.custom-input:focus {
  border-color: var(--primary-color);
  background: rgba(0, 0, 0, 0.5);
  box-shadow: 0 0 0 2px rgba(0, 243, 255, 0.1);
}

.custom-select-wrapper {
  position: relative;
  width: 100%;
}
.custom-select-wrapper select {
  appearance: none;
  padding-right: 40px;
}
.select-icon {
  position: absolute;
  right: 15px;
  top: 50%;
  transform: translateY(-50%);
  pointer-events: none;
  color: #8892b0;
}

/* File Upload Area */
.file-input-hidden {
  display: none;
}

.file-upload-box {
  border: 2px dashed rgba(255, 255, 255, 0.2);
  border-radius: 16px;
  padding: 30px 20px;
  text-align: center;
  background: rgba(0, 0, 0, 0.2);
  cursor: pointer;
  transition: all 0.3s;
}
.file-upload-box:hover {
  border-color: var(--primary-color);
  background: rgba(0, 243, 255, 0.05);
}

.upload-icon {
  font-size: 2.5rem;
  color: #8892b0;
  margin-bottom: 10px;
  transition: all 0.3s;
}
.file-upload-box:hover .upload-icon {
  color: var(--primary-color);
}

.file-upload-box p {
  color: #8892b0;
  margin: 0;
  font-size: 0.95rem;
}
.file-upload-box span {
  color: var(--primary-color);
  font-weight: 600;
}

/* Buttons */
.btn-glow {
  height: 48px;
  font-size: 1rem;
  font-weight: 600;
  letter-spacing: 1px;
  border-radius: 12px;
  background: linear-gradient(135deg, var(--primary-color), #00a8ff);
  border: none;
  color: #000;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}
.btn-glow:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(0, 243, 255, 0.4);
}
.btn-glow:disabled {
  opacity: 0.7;
  cursor: not-allowed;
  transform: none;
}

.btn-icon {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.1);
  background: rgba(255,255,255,0.05);
  color: #8892b0;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}
.btn-icon:hover {
  background: rgba(0, 243, 255, 0.1);
  color: var(--primary-color);
  border-color: var(--primary-color);
}
.spinning {
  animation: spin 1s linear infinite;
}
@keyframes spin {
  100% { transform: rotate(360deg); }
}

/* Status Box */
.status-box {
  padding: 12px 15px;
  border-radius: 10px;
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
}
.status-success {
  background: rgba(0, 255, 136, 0.1);
  border: 1px solid rgba(0, 255, 136, 0.2);
  color: #00ff88;
}
.status-error {
  background: rgba(255, 77, 79, 0.1);
  border: 1px solid rgba(255, 77, 79, 0.2);
  color: #ff4d4f;
}

/* File Tree */
.file-tree-container {
  max-height: 500px;
}

.empty-state, .loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 200px;
  color: #8892b0;
}
.empty-state i {
  font-size: 3rem;
  opacity: 0.3;
}

.file-tree {
  max-height: 420px;
  overflow-y: auto;
  padding-right: 5px;
}

.tree-header {
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 10px;
}

.tenant-header {
  font-size: 1.05rem;
  color: #fff;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  padding-bottom: 8px;
}

.project-header {
  font-size: 0.95rem;
  color: #e2e8f0;
}

.file-count.badge {
  background: rgba(255,255,255,0.1);
  color: #8892b0;
  font-size: 0.7rem;
  padding: 2px 8px;
  border-radius: 10px;
  margin-left: auto;
}

.file-list {
  list-style: none;
  padding: 0;
  margin: 0;
  border-left: 1px solid rgba(255,255,255,0.1);
  padding-left: 15px;
  margin-left: 6px;
}

.file-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 0;
  color: #8892b0;
  font-size: 0.9rem;
  position: relative;
  transition: color 0.2s;
}
.file-item:hover {
  color: #fff;
}
.file-item::before {
  content: '';
  position: absolute;
  left: -15px;
  top: 15px;
  width: 10px;
  height: 1px;
  background: rgba(255,255,255,0.1);
}

.file-icon {
  color: #586069;
}

.loader-small {
  display: inline-block;
  width: 18px;
  height: 18px;
  border: 2px solid rgba(0,0,0,0.3);
  border-radius: 50%;
  border-top-color: #000;
  animation: spin 1s ease-in-out infinite;
}

/* Custom Scrollbar */
.custom-scroll::-webkit-scrollbar { width: 6px; }
.custom-scroll::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); border-radius: 3px; }
.custom-scroll::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 3px; }
.custom-scroll::-webkit-scrollbar-thumb:hover { background: rgba(0, 243, 255, 0.5); }

/* Utilities */
.flex { display: flex; }
.justify-between { justify-content: space-between; }
.items-center { align-items: center; }
.text-center { text-align: center; }
.text-gray { color: #8892b0; }
.text-sm { font-size: 0.9rem; }
.text-xs { font-size: 0.8rem; }
.text-primary { color: var(--primary-color); }
.w-full { width: 100%; }
.mt-5 { margin-top: 5px; }
.mt-10 { margin-top: 10px; }
.mt-20 { margin-top: 20px; }
.mt-30 { margin-top: 30px; }
.mb-10 { margin-bottom: 10px; }
.mb-15 { margin-bottom: 15px; }
.mb-20 { margin-bottom: 20px; }
.mb-30 { margin-bottom: 30px; }
.mr-10 { margin-right: 10px; }
.ml-15 { margin-left: 15px; }
</style>
