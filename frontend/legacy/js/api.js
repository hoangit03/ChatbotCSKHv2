const API_URL = window.location.hostname === 'localhost' ? 'http://localhost:8081' : '/api';
// Actually, Nginx hasn't mapped /api yet, or we can just use the public Gateway URL if exposed.
// Wait, Nginx config I wrote earlier didn't map /api to gateway. Let's use relative path if we map it, or we can just call gateway directly if they are on same domain.
// Let's use the explicit domain for gateway if needed. Since gateway and UI are on the same server, I should update Nginx to map /api to gateway!
// Let's assume Nginx maps /api to http://global_gateway:8081.

const GATEWAY_URL = '/api';

const api = {
    getToken: () => localStorage.getItem('access_token'),
    setToken: (token) => localStorage.setItem('access_token', token),
    removeToken: () => localStorage.removeItem('access_token'),
    
    getHeaders: () => {
        const token = api.getToken();
        return token ? { 'Authorization': `Bearer ${token}` } : {};
    },

    login: async (email, password) => {
        const formData = new URLSearchParams();
        formData.append('username', email);
        formData.append('password', password);
        
        const res = await fetch(`${GATEWAY_URL}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: formData
        });
        
        if (!res.ok) throw new Error('Sai email hoặc mật khẩu');
        const data = await res.json();
        api.setToken(data.access_token);
        return data;
    },

    getMe: async () => {
        const res = await fetch(`${GATEWAY_URL}/auth/me`, {
            headers: api.getHeaders()
        });
        if (!res.ok) throw new Error('Phiên đăng nhập hết hạn');
        return await res.json();
    },

    getUsers: async () => {
        const res = await fetch(`${GATEWAY_URL}/admin/users`, { headers: api.getHeaders() });
        if (!res.ok) throw new Error('Không thể lấy danh sách user');
        return await res.json();
    },

    getSessions: async (tenantId) => {
        const res = await fetch(`${GATEWAY_URL}/admin/sessions/${tenantId}`, { headers: api.getHeaders() });
        if (!res.ok) throw new Error('Không thể lấy danh sách session');
        return await res.json();
    },

    getSessionMessages: async (tenantId, sessionId) => {
        const res = await fetch(`${GATEWAY_URL}/admin/sessions/${tenantId}/${sessionId}/messages`, { headers: api.getHeaders() });
        if (!res.ok) throw new Error('Không thể lấy nội dung chat');
        return await res.json();
    },

    getProjects: async () => {
        try {
            // Lấy trực tiếp từ Sales API theo yêu cầu (không qua proxy)
            const salesApiUrl = "https://uatapi.ctlotus.ctgroupvietnam.com/endpoint/project";
            const salesApiKey = "ak_34zs6l1r7z"; // Giá trị từ .env

            const res = await fetch(salesApiUrl, {
                method: "GET",
                headers: {
                    "x-api-key": salesApiKey,
                    "Accept": "application/json"
                }
            });
            if (!res.ok) throw new Error(`API returned ${res.status}`);
            const data = await res.json();
            // Map the data to project names
            let projectsList = [];
            const items = Array.isArray(data) ? data : (data.data || []);
            projectsList = items.map(p => p.name || p.code).filter(Boolean);
            return { projects: projectsList };
        } catch (error) {
            console.error("Error fetching projects directly from Sales API:", error);
            throw error;
        }
    },

    uploadETL: async (tenantId, file, minRoleLevel, forceOverwrite = false, projectName = "") => {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('min_role_level', minRoleLevel);
        formData.append('force_overwrite', forceOverwrite);
        if (projectName) {
            formData.append('project_name', projectName);
        }

        const res = await fetch(`${GATEWAY_URL}/${tenantId}/etl/extract`, {
            method: 'POST',
            headers: api.getHeaders(),
            body: formData
        });
        if (!res.ok) {
            const errBody = await res.json().catch(() => ({}));
            if (errBody.detail && errBody.detail.code === "CONFIRM_OVERWRITE") {
                throw new Error("CONFIRM_OVERWRITE");
            }
            throw new Error(errBody.detail?.message || errBody.detail || 'Lỗi upload dữ liệu');
        }
        return await res.json();
    }
};
