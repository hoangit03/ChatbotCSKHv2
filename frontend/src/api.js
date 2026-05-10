import axios from 'axios';

// Gateway to local Nginx which will proxy /api to the backend
const GATEWAY_URL = '/api/v1';

const apiClient = axios.create({
  baseURL: GATEWAY_URL,
});

export const api = {
  // Projects
  getProjects: async () => {
    const { data } = await apiClient.get('/projects');
    return data;
  },

  // ETL
  uploadETL: async (tenantId, file, minRoleLevel, forceOverwrite = false, projectName = "") => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('project_name', projectName || tenantId);
    formData.append('doc_group', "Tài liệu dự án");
    formData.append('version', "1.0");
    
    const { data } = await apiClient.post('/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return data;
  },

  getETLFiles: async (tenantId) => {
    // Dummy implementation if there is no get API
    return { files: [] };
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
    const { data } = await apiClient.post('/chat', payload);
    return data;
  }
};

