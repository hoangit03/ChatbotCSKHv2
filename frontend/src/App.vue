<template>
  <router-view />
</template>

<style>
/* Global styles can go here, but we will rely mostly on src/assets/style.css */
@import './assets/style.css';

body {
  margin: 0;
  padding: 0;
  font-family: 'Inter', sans-serif;
  background-color: var(--bg-color, #1a1a1a);
  color: var(--text-color, #ffffff);
}
</style>
