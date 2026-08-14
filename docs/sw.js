/* 오프라인에서도 마지막 결과를 볼 수 있게 하는 최소 서비스워커.
   - 앱 셸: 캐시 우선
   - 데이터: 네트워크 우선, 실패 시 캐시된 마지막 결과 */
const CACHE = "screener-v4";
const SHELL = ["./", "./index.html", "./manifest.json", "./icon-192.png", "./icon-512.png"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET") return;
  const isData = new URL(req.url).pathname.includes("/data/");

  if (isData) {
    e.respondWith(
      fetch(req)
        .then(res => {
          const copy = res.clone();
          caches.open(CACHE).then(c => c.put(req, copy));
          return res;
        })
        .catch(() => caches.match(req, { ignoreSearch: true }))
    );
  } else {
    e.respondWith(caches.match(req).then(hit => hit || fetch(req)));
  }
});
