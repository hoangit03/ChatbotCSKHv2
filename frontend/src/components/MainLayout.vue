<template>
  <div class="app-container">
    <Sidebar :tenant="currentTenant" :role="currentRole" @tenant-changed="handleTenantChange" @logout="handleLogout" />
    <main class="main-content glass-panel">
      <router-view :tenant="currentTenant" :role="currentRole" />
    </main>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import Sidebar from './Sidebar.vue';
import { api } from '../api';

const route = useRoute();
const router = useRouter();

const user = ref(null);

const fetchUser = async () => {
  // Không gọi api.getMe() nữa vì đã bỏ SSO
  // Role sẽ được lấy từ URL query
};

onMounted(() => {
  fetchUser();
});

const currentTenant = computed(() => route.params.tenantId || 'qtqd');
const currentRole = computed(() => {
  const urlParams = new URLSearchParams(window.location.search);
  const roleParam = route.query.role || urlParams.get('role');
  return roleParam || 'E'; // Default 'E'
});

const handleTenantChange = (newTenant) => {
  // If we change tenant, stay on the same route if possible, or redirect to chat
  const currentPath = route.path;
  const pathParts = currentPath.split('/');
  pathParts[1] = newTenant; // Replace tenant in path
  router.push(pathParts.join('/'));
};

const handleLogout = () => {
  // Không có chức năng logout khi nhúng
};
</script>

<style scoped>
.app-container {
  display: flex;
  height: 100vh;
  width: 100vw;
  background-color: var(--bg-color, #1a1a1a); /* Adjust based on style.css */
}

.main-content {
  flex: 1;
  margin: 20px;
  border-radius: 16px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
</style>
