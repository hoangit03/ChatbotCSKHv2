import { createRouter, createWebHistory } from 'vue-router';
import { api } from '../api';

const routes = [
  {
    path: '/',
    redirect: '/qtqd/chat'
  },
  {
    path: '/:tenantId',
    component: () => import('../components/MainLayout.vue'),
    meta: { requiresAuth: true },
    children: [
      {
        path: '',
        redirect: to => {
          return `/${to.params.tenantId}/chat`;
        }
      },
      {
        path: 'chat',
        name: 'Chat',
        component: () => import('../views/ChatView.vue')
      },
      {
        path: 'etl',
        name: 'ETL',
        component: () => import('../views/EtlView.vue')
      },
      {
        path: 'admin',
        name: 'Admin',
        component: () => import('../views/AdminView.vue')
      }
    ]
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

router.beforeEach((to, from, next) => {
  // Bỏ logic SSO, cho phép vào thẳng
  next();
});

export default router;
