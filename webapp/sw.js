// Offline cache + cross-origin isolation (COOP/COEP) so onnxruntime can use multiple threads
// on static hosts (GitHub Pages, Netlify, …) that don't let you set response headers.
const SHELL_CACHE = "bangla-asr-shell-v1";
const MODEL_CACHE = "bangla-asr-model-v1";      // big files; kept across app updates
const SHELL = ["./", "./index.html", "./style.css", "./app.js", "./worker.js", "./asr-core.js", "./mic-worklet.js",
  "./manifest.webmanifest", "./icon.svg", "./icon-192.png", "./icon-512.png",
  "./vendor/ort.wasm.min.mjs", "./vendor/ort-wasm-simd-threaded.mjs", "./vendor/ort-wasm-simd-threaded.wasm"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL_CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k !== SHELL_CACHE && k !== MODEL_CACHE) await caches.delete(k);
    await self.clients.claim();
  })());
});

function isolate(resp) {
  if (!resp || resp.status === 0 || resp.type === "opaque") return resp;
  const h = new Headers(resp.headers);
  h.set("Cross-Origin-Embedder-Policy", "require-corp");
  h.set("Cross-Origin-Opener-Policy", "same-origin");
  h.set("Cross-Origin-Resource-Policy", "same-origin");
  return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers: h });
}

self.addEventListener("fetch", (e) => {
  const req = e.request;
  const url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== self.location.origin) return;
  e.respondWith((async () => {
    const isModel = url.pathname.includes("/model/");
    const cacheName = isModel ? MODEL_CACHE : SHELL_CACHE;
    const key = req.mode === "navigate" ? new URL("./index.html", self.location.href).href : req.url;
    const cached = await caches.match(key, { ignoreSearch: true });
    if (cached) return isolate(cached);
    try {
      const resp = await fetch(req);
      if (resp.ok) { const copy = resp.clone(); e.waitUntil(caches.open(cacheName).then(c => c.put(key, copy))); }
      return isolate(resp);
    } catch (err) {
      return new Response("Offline and not cached yet", { status: 503 });
    }
  })());
});
