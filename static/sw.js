// Minimal service worker - just enough for "Add to Home Screen" / install
// prompts on Android/Chrome, which require an active service worker with a
// fetch handler. This deliberately does NOT cache API responses (session
// state, history, audio clips) - only the static app shell - so you always
// see live data, never stale cached session results.

const CACHE_NAME = 'focusbuddy-shell-v1';
const SHELL_FILES = ['/static/style.css', '/static/app.js', '/static/icons/icon-192.png'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_FILES))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  // Never intercept API calls or dynamic pages - only the static shell files.
  if (url.pathname.startsWith('/api/') || event.request.method !== 'GET') return;
  if (!SHELL_FILES.includes(url.pathname)) return;

  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
