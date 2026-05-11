import axios from 'axios';

// Function to create a client with dynamic tenant baseURL
const getClient = (tenantId) => {
  const tenant = (tenantId || 'primer-diamond').toLowerCase().replace(/_/g, '-');
  return axios.create({
    baseURL: `/api/${tenant}`,
    headers: {
      'X-API-Key': 'ak_guest_3rd_party_ctlotus_998877'
    }
  });
};

export const api = {
  // Projects
  getProjects: async (tenantId) => {
    const { data } = await getClient(tenantId).get('/projects');
    return data;
  },

  // ETL
  uploadETL: async (tenantId, file, minRoleLevel, forceOverwrite = false, projectName = "") => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('project_name', projectName || tenantId);
    formData.append('doc_group', "Tài liệu dự án");
    formData.append('version', "1.0");
    formData.append('min_role_level', minRoleLevel);
    formData.append('force_overwrite', forceOverwrite);
    
    // Gọi thẳng vào Nginx proxy /shared_etl/ (chạy port 8010 trên server)
    try {
      const { data } = await axios.post('/shared_etl/etl/extract', formData, {
        headers: { 
          'Content-Type': 'multipart/form-data',
          'X-Tenant-Id': tenantId,
          'X-API-Key': 'ak_guest_3rd_party_ctlotus_998877'
        }
      });
      return data;
    } catch (error) {
      const errBody = error.response?.data;
      if (errBody?.detail?.code === "CONFIRM_OVERWRITE") {
        throw new Error("CONFIRM_OVERWRITE");
      }
      throw new Error(errBody?.detail?.message || errBody?.detail || 'Lỗi upload dữ liệu');
    }
  },

  getETLFiles: async (tenantId) => {
    const { data } = await axios.get('/shared_etl/etl/files', {
      headers: {
        'X-API-Key': 'ak_guest_3rd_party_ctlotus_998877'
      }
    });
    return data;
  },

  // Chat/Sessions 
  // (Stubbed since ChatbotCSKHv2 doesn't have an admin session history endpoint out of the box in this UI)
  getSessions: async (tenantId, page = 1, limit = 20) => {
    return { sessions: [], total: 0 };
  },
  getSessionMessages: async (tenantId, sessionId) => {
    return { messages: [] };
  },

  // Send message
  sendMessage: async (tenantId, sessionId, message) => {
    const payload = {
      message: message,
      session_id: sessionId,
      project_name: tenantId
    };
    const { data } = await getClient(tenantId).post('/chat', payload);
    return data;
  }
};

