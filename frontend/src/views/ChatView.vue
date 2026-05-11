<template>
  <div class="chat-page page-content active">
    <div class="page-header glass-panel glass-morphism chat-header">
      <div class="header-title">
        <h2><i class="fa-solid fa-robot text-primary"></i> {{ props.tenant.toUpperCase() }}</h2>
        <p class="text-sm text-gray">Trợ lý ảo thông minh</p>
      </div>
      
      <!-- Role selector specifically for primer-diamond -->
      <div v-if="props.tenant === 'primer-diamond'" class="ml-auto role-selector">
        <div class="custom-select-wrapper">
          <select v-model="chatRole" class="glass-input custom-select">
            <option value="user">Tư vấn khách hàng</option>
            <option value="sale">Hỗ trợ Sale</option>
          </select>
          <i class="fa-solid fa-chevron-down select-icon"></i>
        </div>
      </div>
    </div>

    <div class="chat-container">
      <div class="chat-messages custom-scroll" id="chat-messages" ref="messagesList">
        <div class="message-wrapper ai-wrapper">
          <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
          <div class="msg-bubble glass-panel">Xin chào! Tôi có thể giúp gì cho bạn hôm nay?</div>
        </div>
        
        <div v-for="(msg, index) in messages" :key="index" class="message-wrapper" :class="msg.role === 'user' ? 'user-wrapper' : 'ai-wrapper'">
          <div class="msg-avatar">
            <i :class="['fa-solid', msg.role === 'user' ? 'fa-user' : 'fa-robot']"></i>
          </div>
          <div class="msg-bubble glass-panel markdown-body" v-html="renderMarkdown(msg.content)"></div>
        </div>
        
        <div v-if="loading" class="message-wrapper ai-wrapper">
          <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
          <div class="msg-bubble glass-panel"><span class="loader-dots"></span></div>
        </div>
      </div>

      <div class="chat-input-area glass-morphism">
        <form @submit.prevent="sendMessage" class="chat-form">
          <textarea 
            v-model="inputText" 
            placeholder="Nhập câu hỏi của bạn (Enter để gửi, Shift+Enter để xuống dòng)..." 
            class="glass-input custom-textarea custom-scroll" 
            rows="1"
            @keydown="handleKeydown"
            @input="adjustTextareaHeight"
            ref="chatInputRef"
            :disabled="loading"
          ></textarea>
          <button type="submit" class="btn-send btn-glow-small" :disabled="!inputText.trim() || loading">
            <i class="fa-solid fa-paper-plane"></i>
          </button>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue';
import { api } from '../api';
import { marked } from 'marked';

const props = defineProps({
  tenant: { type: String, required: true },
  role: { type: String, required: true }
});

const chatRole = ref('user');
const inputText = ref('');
const messages = ref([]);
const loading = ref(false);
const sessionId = ref(null);
const messagesList = ref(null);

watch(() => props.tenant, () => {
  // Reset chat when tenant changes
  messages.value = [];
  sessionId.value = null;
  chatRole.value = 'user';
});

const scrollToBottom = async () => {
  await nextTick();
  if (messagesList.value) {
    messagesList.value.scrollTop = messagesList.value.scrollHeight;
  }
};

const renderMarkdown = (text) => {
  return marked.parse(text || '');
};

const chatInputRef = ref(null);

const adjustTextareaHeight = () => {
  if (chatInputRef.value) {
    chatInputRef.value.style.height = 'auto';
    chatInputRef.value.style.height = (chatInputRef.value.scrollHeight) + 'px';
  }
};

const handleKeydown = (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
};

const appendMessage = (role, content) => {
  messages.value.push({ role, content });
  scrollToBottom();
};

const sendMessage = async () => {
  const text = inputText.value.trim();
  if (!text || loading.value) return;

  inputText.value = '';
  appendMessage('user', text);
  
  loading.value = true;
  
  // Create a placeholder for assistant's streaming response
  const assistantMsgIndex = messages.value.push({ role: 'assistant', content: '' }) - 1;
  let fullText = '';
  
  try {
    const isSale = chatRole.value === 'sale';
    const endpoint = isSale ? 'sale/chat/stream' : 'chat/stream';
    const payloadBody = {
      message: text,
      session_id: sessionId.value || "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c => (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16))
    };

    const token = api.getToken() || '';
    const res = await fetch(`/api/v1/${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(payloadBody)
    });

    if (!res.ok) throw new Error('API Error');

    loading.value = false; // Hide global loader
    
    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");

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
            if (data.session_id && !sessionId.value) {
              sessionId.value = data.session_id;
            }
            if (data.text) {
              fullText += data.text;
              messages.value[assistantMsgIndex].content = fullText;
              scrollToBottom();
            }
            if (data.suggested_questions && data.suggested_questions.length > 0) {
              let sugHtml = '<div class="suggested-questions" style="margin-top: 15px; display: flex; gap: 8px; flex-wrap: wrap;">';
              data.suggested_questions.forEach(q => {
                sugHtml += `<button class="suggestion-btn" onclick="window.dispatchEvent(new CustomEvent('suggest', {detail: '${q.replace(/'/g, "\\'")}'}))">${q}</button>`;
              });
              sugHtml += '</div>';
              messages.value[assistantMsgIndex].content += sugHtml;
            }
            if (data.sources && data.sources.length > 0) {
              const sourcesHtml = `<div style="margin-top:10px; font-size:12px; color:var(--text-secondary)"><i>Nguồn: ${data.sources.map(s => s.doc || 'Tài liệu').join(', ')}</i></div>`;
              messages.value[assistantMsgIndex].content += sourcesHtml;
            }
          } catch(e) {}
        }
      }
    }
  } catch (err) {
    loading.value = false;
    messages.value[assistantMsgIndex].content = `<span style="color: var(--danger)">Lỗi kết nối.</span>`;
  }
};

// Global event listener for suggested questions
if (typeof window !== 'undefined') {
  window.addEventListener('suggest', (e) => {
    inputText.value = e.detail;
    sendMessage();
  });
}
</script>

<style scoped>
.chat-page {
  display: flex;
  flex-direction: column;
  height: 100vh;
  padding: 0;
  overflow: hidden;
}

.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 30px;
  border-radius: 0;
  border-left: none;
  border-right: none;
  border-top: none;
  z-index: 10;
  flex-shrink: 0;
}

.header-title h2 {
  margin: 0;
  font-size: 1.4rem;
  letter-spacing: 1px;
}

.chat-container {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  position: relative;
  overflow: hidden;
  max-width: 1000px;
  margin: 0 auto;
  width: 100%;
}

.chat-messages {
  flex-grow: 1;
  overflow-y: auto;
  padding: 30px 20px;
  display: flex;
  flex-direction: column;
  gap: 25px;
}

.message-wrapper {
  display: flex;
  gap: 15px;
  width: 100%;
  animation: slideUp 0.3s ease;
}

@keyframes slideUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

.user-wrapper {
  flex-direction: row-reverse;
}

.msg-avatar {
  width: 45px;
  height: 45px;
  border-radius: 12px;
  background: rgba(255,255,255,0.1);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  font-size: 1.3rem;
  box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.user-wrapper .msg-avatar {
  background: rgba(0, 243, 255, 0.15);
  color: #00f3ff;
  border: 1px solid rgba(0, 243, 255, 0.3);
}
.ai-wrapper .msg-avatar {
  background: rgba(138, 43, 226, 0.15);
  color: #b975ff;
  border: 1px solid rgba(138, 43, 226, 0.3);
}

.msg-bubble {
  max-width: 80%;
  padding: 18px 24px;
  border-radius: 20px;
  background: rgba(15, 20, 35, 0.6);
  backdrop-filter: blur(10px);
  font-size: 1.05rem;
  line-height: 1.6;
}
.user-wrapper .msg-bubble {
  background: rgba(0, 243, 255, 0.08);
  border: 1px solid rgba(0, 243, 255, 0.2);
  border-top-right-radius: 4px;
}
.ai-wrapper .msg-bubble {
  background: rgba(138, 43, 226, 0.05);
  border: 1px solid rgba(138, 43, 226, 0.2);
  border-top-left-radius: 4px;
}

/* Chat Input Area */
.chat-input-area {
  flex-shrink: 0;
  padding: 20px;
  background: rgba(10, 15, 25, 0.8);
  border-top: 1px solid rgba(255,255,255,0.05);
  backdrop-filter: blur(20px);
}

.chat-form {
  position: relative;
  display: flex;
  align-items: flex-end;
  gap: 15px;
  max-width: 900px;
  margin: 0 auto;
}

.custom-textarea {
  width: 100%;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  color: #fff;
  padding: 15px 60px 15px 20px;
  font-size: 1rem;
  transition: all 0.3s;
  resize: none;
  min-height: 54px;
  max-height: 200px;
}
.custom-textarea:focus {
  border-color: var(--primary-color);
  box-shadow: 0 0 0 2px rgba(0, 243, 255, 0.1);
  outline: none;
}

.btn-send {
  position: absolute;
  right: 10px;
  bottom: 8px;
  width: 40px;
  height: 40px;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
}

/* Custom Select for Role */
.custom-select-wrapper {
  position: relative;
  width: 180px;
}
.custom-select {
  width: 100%;
  margin: 0;
  height: 40px;
  appearance: none;
  padding-right: 30px;
  border-radius: 10px;
  background: rgba(0,0,0,0.3);
  border: 1px solid rgba(255,255,255,0.1);
}
.custom-select:focus {
  border-color: var(--primary-color);
}
.select-icon {
  position: absolute;
  right: 15px;
  top: 50%;
  transform: translateY(-50%);
  pointer-events: none;
  color: #8892b0;
  font-size: 0.8rem;
}

/* Loading Dots */
.loader-dots {
  display: inline-block;
  position: relative;
  width: 60px;
  height: 20px;
}
.loader-dots::after {
  content: '...';
  font-size: 1.5rem;
  font-weight: bold;
  letter-spacing: 2px;
  animation: typing 1.5s infinite;
}
@keyframes typing {
  0% { content: '.'; }
  33% { content: '..'; }
  66% { content: '...'; }
}

/* Custom Scrollbar */
.custom-scroll::-webkit-scrollbar {
  width: 6px;
}
.custom-scroll::-webkit-scrollbar-track {
  background: rgba(0,0,0,0.1);
}
.custom-scroll::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,0.2);
  border-radius: 3px;
}
.custom-scroll::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 243, 255, 0.5);
}

.glass-morphism {
  background: rgba(15, 20, 35, 0.6);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.05);
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

.btn-glow-small {
  background: linear-gradient(135deg, var(--primary-color), #00a8ff);
  border: none;
  color: #000;
  cursor: pointer;
  transition: all 0.3s ease;
}
.btn-glow-small:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(0, 243, 255, 0.4);
}
.btn-glow-small:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  background: rgba(255,255,255,0.1);
  color: #8892b0;
}
.text-primary { color: var(--primary-color); }
</style>
