const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

const ui = {
  loading: $("loading"), loadText: $("loadText"), loadBar: $("loadBar"), loadDetail: $("loadDetail"),
  micBtn: $("micBtn"), micLabel: $("micLabel"), micStatus: $("micStatus"), level: $("level"),
  fileInput: $("fileInput"), fileStatus: $("fileStatus"),
  silence: $("silence"), silenceOut: $("silenceOut"), ns: $("ns"), engineInfo: $("engineInfo"),
  results: $("results"), empty: $("empty"), offlineBadge: $("offlineBadge"),
};

let worker, items = store.get("bn-asr-items", []);
let mic = null; // { ctx, stream, node }

// ---------- settings ----------
ui.silence.value = store.get("bn-asr-silence", 600);
ui.ns.checked = store.get("bn-asr-ns", false);
const showSilence = () => (ui.silenceOut.textContent = `${ui.silence.value} ms`);
showSilence();
ui.silence.oninput = () => { showSilence(); store.set("bn-asr-silence", +ui.silence.value); worker?.postMessage({ type: "config", silenceMs: +ui.silence.value }); };
ui.ns.onchange = () => store.set("bn-asr-ns", ui.ns.checked);

// ---------- transcript list ----------
function render() {
  ui.results.replaceChildren(...items.map((it) => {
    const li = document.createElement("li");
    const t = document.createElement("div"); t.className = "text"; t.lang = "bn"; t.textContent = it.text;
    const m = document.createElement("div"); m.className = "meta";
    m.textContent = `${it.label} · ${it.dur.toFixed(1)} s speech · ${(it.ms / 1000).toFixed(2)} s to transcribe`;
    const c = document.createElement("button"); c.className = "ghost copy"; c.textContent = "Copy";
    c.onclick = () => copy(it.text, c);
    li.append(t, c, m);
    return li;
  }));
  ui.empty.hidden = items.length > 0;
}
async function copy(text, btn) {
  try { await navigator.clipboard.writeText(text); const o = btn.textContent; btn.textContent = "Copied"; setTimeout(() => (btn.textContent = o), 1200); } catch {}
}
$("copyAll").onclick = (e) => copy(items.map((i) => i.text).join("\n"), e.currentTarget);
$("clear").onclick = () => { items = []; store.set("bn-asr-items", items); render(); };
$("download").onclick = () => {
  const blob = new Blob([items.map((i) => i.text).join("\n") + "\n"], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = `transcript-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.txt`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};
render();

// ---------- service worker: offline cache + cross-origin isolation for threads ----------
async function setupServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  const sw = navigator.serviceWorker;
  const hadActive = !!(await sw.getRegistration())?.active;
  await sw.register("sw.js");
  if (!sw.controller && !hadActive) {
    ui.loadText.textContent = "Setting up offline mode…";
    // first visit: the worker pre-caches the app shell (~15 MB) then takes control; don't wait forever
    await Promise.race([
      new Promise((r) => sw.addEventListener("controllerchange", r, { once: true })),
      new Promise((r) => setTimeout(r, 60000)),
    ]);
  }
  // COOP/COEP headers only apply after a reload under the service worker
  let reloaded = false; try { reloaded = sessionStorage.getItem("bn-asr-coi") === "1"; } catch {}
  if (sw.controller && !self.crossOriginIsolated && !reloaded) {
    try { sessionStorage.setItem("bn-asr-coi", "1"); } catch {}
    location.reload();
    await new Promise(() => {});
  }
  try { await navigator.storage?.persist?.(); } catch {}
}

async function checkOffline() {
  if (!("caches" in self)) return;
  try {
    const man = await (await fetch("model/manifest.json")).json();
    const files = ["model/manifest.json", "model/mel_fb.bin", "model/window.bin", "model/tokens.txt", "model/silero_vad.onnx", ...man.parts.map((p) => "model/" + p.name)];
    // the service worker caches in the background; give it a moment for the last chunk
    for (let attempt = 0; attempt < 10; attempt++) {
      const hits = await Promise.all(files.map((f) => caches.match(new URL(f, location.href).href)));
      if (hits.every(Boolean)) { ui.offlineBadge.hidden = false; return; }
      await new Promise((r) => setTimeout(r, 1000));
    }
  } catch {}
}

// ---------- engine ----------
function startWorker() {
  const isolated = self.crossOriginIsolated;
  const threads = isolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;
  worker = new Worker("worker.js", { type: "module" });
  worker.onerror = (e) => fail(e.message || "Worker failed to start");
  worker.onmessage = ({ data: m }) => {
    switch (m.type) {
      case "progress": {
        const pct = (m.loaded / m.total) * 100;
        ui.loadText.textContent = "Loading the speech model (first time only: ~165 MB)…";
        ui.loadBar.style.width = `${pct.toFixed(1)}%`;
        ui.loadDetail.textContent = `${(m.loaded / 1e6).toFixed(0)} / ${(m.total / 1e6).toFixed(0)} MB`;
        break;
      }
      case "stage": ui.loadText.textContent = m.stage; ui.loadBar.style.width = "100%"; ui.loadDetail.textContent = ""; break;
      case "ready":
        ui.loading.hidden = true; ui.micBtn.disabled = false; ui.fileInput.disabled = false;
        ui.engineInfo.textContent = `Engine: onnxruntime-web (WebAssembly, ${m.threads} thread${m.threads > 1 ? "s" : ""}${isolated ? "" : ", single-thread mode"}) · ready in ${(m.ms / 1000).toFixed(1)} s`;
        document.body.dataset.ready = "1";
        checkOffline();
        break;
      case "vad":
        ui.micBtn.classList.toggle("speaking", m.speaking);
        if (mic) ui.micStatus.textContent = m.speaking ? "Hearing you…" : "Listening — speak, then pause.";
        break;
      case "result":
        items.unshift({ text: m.text, dur: m.dur, ms: m.ms, label: m.source === "file" ? m.name : new Date().toLocaleTimeString() });
        store.set("bn-asr-items", items.slice(0, 500));
        render();
        break;
      case "fileProgress": ui.fileStatus.textContent = `${m.name}: ${Math.round((m.done / m.total) * 100)}%`; break;
      case "fileDone": ui.fileStatus.textContent = `${m.name}: done`; ui.fileInput.disabled = false; document.body.dataset.fileDone = m.name; break;
      case "error": fail(m.message); break;
    }
  };
  worker.postMessage({ type: "init", threads, silenceMs: +ui.silence.value });
}

function fail(msg) {
  ui.loading.hidden = false;
  ui.loadText.textContent = "Something went wrong: " + msg;
  ui.loadDetail.textContent = "Try reloading the page.";
  console.error(msg);
}

// ---------- microphone ----------
ui.micBtn.onclick = async () => (mic ? stopMic() : startMic());

async function startMic() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: ui.ns.checked, autoGainControl: true },
    });
    const ctx = new AudioContext({ sampleRate: 16000 });
    await ctx.audioWorklet.addModule("mic-worklet.js");
    const src = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, "mic-capture");
    node.port.onmessage = ({ data }) => {
      ui.level.style.width = `${Math.min(100, data.rms * 400)}%`;
      worker.postMessage({ type: "audio", samples: data.samples }, [data.samples.buffer]);
    };
    src.connect(node);
    worker.postMessage({ type: "micStart" });
    mic = { ctx, stream, node };
    ui.micBtn.classList.add("on"); ui.micBtn.setAttribute("aria-pressed", "true");
    ui.micLabel.textContent = "Stop"; ui.micStatus.textContent = "Listening — speak, then pause.";
  } catch (e) {
    ui.micStatus.textContent = e.name === "NotAllowedError" ? "Microphone permission was denied." : `Microphone error: ${e.message}`;
  }
}

function stopMic() {
  mic.node.port.onmessage = null;
  mic.stream.getTracks().forEach((t) => t.stop());
  mic.ctx.close();
  mic = null;
  worker.postMessage({ type: "micStop" });
  ui.level.style.width = "0";
  ui.micBtn.classList.remove("on", "speaking"); ui.micBtn.setAttribute("aria-pressed", "false");
  ui.micLabel.textContent = "Start listening"; ui.micStatus.textContent = "Speak, then pause — text appears after each pause.";
}

// ---------- audio files ----------
ui.fileInput.onchange = async () => {
  const f = ui.fileInput.files[0]; if (!f) return;
  ui.fileInput.disabled = true; ui.fileStatus.textContent = `${f.name}: decoding…`;
  try {
    const buf = await f.arrayBuffer();
    const probe = new OfflineAudioContext(1, 1, 16000);
    const audio = await probe.decodeAudioData(buf);            // resampled to 16 kHz
    const n = audio.length, mono = new Float32Array(n);
    for (let c = 0; c < audio.numberOfChannels; c++) { const d = audio.getChannelData(c); for (let i = 0; i < n; i++) mono[i] += d[i] / audio.numberOfChannels; }
    worker.postMessage({ type: "file", samples: mono, name: f.name }, [mono.buffer]);
  } catch (e) {
    ui.fileStatus.textContent = `${f.name}: could not decode this file`; ui.fileInput.disabled = false;
  }
  ui.fileInput.value = "";
};

(async () => {
  try { await setupServiceWorker(); } catch (e) { console.warn("service worker unavailable", e); }
  startWorker();
})();
