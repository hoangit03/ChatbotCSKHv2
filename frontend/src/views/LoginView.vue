<template>
  <div class="login-container">
    <div class="background-effects">
      <div class="glow-orb orb-1"></div>
      <div class="glow-orb orb-2"></div>
    </div>
    <div class="login-box glass-panel glass-morphism">
      <div class="login-header">
        <div class="logo-icon"><i class="fa-solid fa-cube"></i></div>
        <h2>AI ORCHESTRATION</h2>
        <p class="subtitle">Đăng nhập để tiếp tục</p>
      </div>
      <form @submit.prevent="handleLogin" class="login-form">
        <div class="input-group">
          <i class="fa-solid fa-envelope input-icon"></i>
          <input type="email" v-model="email" placeholder="Email đăng nhập" class="glass-input custom-input" required />
        </div>
        <div class="input-group">
          <i class="fa-solid fa-lock input-icon"></i>
          <input type="password" v-model="password" placeholder="Mật khẩu" class="glass-input custom-input" required />
        </div>
        <button type="submit" class="glass-button w-full btn-glow" :disabled="loading">
          <span v-if="loading" class="loader-small"></span>
          <span v-else>Đăng nhập</span>
        </button>
        <p v-if="error" class="error-msg"><i class="fa-solid fa-circle-exclamation"></i> {{ error }}</p>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { api } from '../api';

const router = useRouter();
const email = ref('');
const password = ref('');
const error = ref('');
const loading = ref(false);

const handleLogin = async () => {
  error.value = '';
  loading.value = true;
  try {
    await api.login(email.value, password.value);
    router.push('/');
  } catch (err) {
    error.value = 'Sai email hoặc mật khẩu';
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  width: 100vw;
  background-color: #0b0f19;
  position: relative;
  overflow: hidden;
}

.background-effects {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 0;
}

.glow-orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(80px);
  opacity: 0.5;
  animation: float 10s infinite ease-in-out alternate;
}

.orb-1 {
  width: 300px;
  height: 300px;
  background: var(--primary-color);
  top: 20%;
  left: 30%;
}

.orb-2 {
  width: 400px;
  height: 400px;
  background: #8a2be2;
  bottom: 10%;
  right: 20%;
  animation-delay: -5s;
}

@keyframes float {
  0% { transform: translate(0, 0) scale(1); }
  100% { transform: translate(30px, -50px) scale(1.1); }
}

.login-box {
  width: 420px;
  padding: 40px;
  border-radius: 20px;
  display: flex;
  flex-direction: column;
  gap: 30px;
  z-index: 1;
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
  background: rgba(20, 25, 40, 0.6);
  backdrop-filter: blur(15px);
  -webkit-backdrop-filter: blur(15px);
}

.login-header {
  text-align: center;
}

.logo-icon {
  font-size: 2.5rem;
  color: var(--primary-color);
  margin-bottom: 15px;
  filter: drop-shadow(0 0 10px var(--primary-color));
}

.login-header h2 {
  color: #ffffff;
  margin: 0;
  font-size: 1.5rem;
  letter-spacing: 2px;
  font-weight: 700;
}

.subtitle {
  color: #8892b0;
  font-size: 0.9rem;
  margin-top: 5px;
}

.login-form {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.input-group {
  position: relative;
  display: flex;
  align-items: center;
}

.input-icon {
  position: absolute;
  left: 15px;
  color: #8892b0;
  font-size: 1rem;
}

.custom-input {
  width: 100%;
  padding-left: 45px;
  height: 48px;
  background: rgba(0, 0, 0, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 12px;
  color: #fff;
  transition: all 0.3s ease;
}

.custom-input:focus {
  border-color: var(--primary-color);
  box-shadow: 0 0 0 2px rgba(0, 243, 255, 0.1);
  background: rgba(0, 0, 0, 0.4);
}

.btn-glow {
  height: 48px;
  font-size: 1rem;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  border-radius: 12px;
  background: linear-gradient(135deg, var(--primary-color), #00a8ff);
  border: none;
  color: #000;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
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

.loader-small {
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 3px solid rgba(0,0,0,0.3);
  border-radius: 50%;
  border-top-color: #000;
  animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.error-msg {
  color: #ff4d4f;
  text-align: center;
  margin-top: 5px;
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  background: rgba(255, 77, 79, 0.1);
  padding: 10px;
  border-radius: 8px;
}
</style>
