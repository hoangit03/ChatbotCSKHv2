<template>
  <aside class="sidebar glass-panel">
    <div class="logo">
      <h2 style="color: var(--primary-color)">AI ORCHESTRATION</h2>
    </div>

    <div class="domain-selector mb-20">
      <label class="text-sm text-gray">CHỌN DOMAIN</label>
      <select :value="tenant" @change="$emit('tenant-changed', $event.target.value)" class="glass-input mt-10">
        <option value="qtqd">QUY TRÌNH QUY ĐỊNH</option>
        <option value="primer-diamond">CSKH PRIMER-DIAMOND</option>
      </select>
    </div>

    <nav class="nav-menu">
      <router-link :to="`/${tenant}/chat`" class="nav-item glass-button" active-class="active">
        <i>💬</i> Chat
      </router-link>

      <router-link v-if="role === 'M' || role === 'D'" :to="`/${tenant}/etl`" class="nav-item glass-button" active-class="active">
        <i>📂</i> Nạp Dữ Liệu
      </router-link>

      <router-link v-if="role === 'D'" :to="`/${tenant}/admin`" class="nav-item glass-button" active-class="active">
        <i>⚙️</i> Quản Trị Hệ Thống
      </router-link>
    </nav>

    <div class="user-profile mt-auto">
      <div class="user-info">
        <div class="avatar">U</div>
        <div class="details">
          <span class="role-badge">{{ roleName }}</span>
        </div>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  tenant: {
    type: String,
    required: true
  },
  role: {
    type: String,
    required: true
  }
});

defineEmits(['tenant-changed', 'logout']);

const roleName = computed(() => {
  if (props.role === 'D') return 'Admin';
  if (props.role === 'M') return 'Manager';
  return 'User';
});
</script>

<style scoped>
.sidebar {
  width: 280px;
  height: calc(100vh - 40px);
  margin: 20px;
  display: flex;
  flex-direction: column;
  padding: 20px;
  border-radius: 16px;
}

.logo h2 {
  font-size: 1.2rem;
  margin-bottom: 30px;
  text-align: center;
  font-weight: 700;
  letter-spacing: 1px;
}

.nav-menu {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  text-decoration: none;
  color: var(--text-color);
  border-radius: 8px;
  transition: all 0.3s ease;
  font-weight: 500;
}

.nav-item.active {
  background: var(--primary-color);
  color: #fff;
}

.mt-auto {
  margin-top: auto;
}

.user-profile {
  padding-top: 20px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.user-info {
  display: flex;
  align-items: center;
  gap: 10px;
}

.avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: var(--primary-color);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
}

.role-badge {
  font-size: 0.8rem;
  background: rgba(0, 243, 255, 0.2);
  color: var(--primary-color);
  padding: 2px 8px;
  border-radius: 12px;
}
</style>
