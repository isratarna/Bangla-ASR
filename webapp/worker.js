// Runs onnxruntime-web, the VAD and the ASR model off the main thread.
import * as ort from "./vendor/ort.wasm.min.mjs";
import { Recognizer, Vad, SR } from "./asr-core.js";

let rec = null, micVad = null, opts = {};
let queue = Promise.resolve();          // process messages strictly in order
let segId = 0;

self.onmessage = (e) => { queue = queue.then(() => handle(e.data)).catch(err => post({ type: "error", message: String(err?.message || err) })); };
const post = (m, t) => self.postMessage(m, t || []);

async function fetchBytes(url, onBytes) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const reader = r.body.getReader(); const chunks = []; let n = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); n += value.length; onBytes?.(value.length); }
  const out = new Uint8Array(n); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

async function handle(msg) {
  switch (msg.type) {
    case "init": {
      opts = msg;
      ort.env.wasm.wasmPaths = new URL("./vendor/", self.location.href).href;
      ort.env.wasm.numThreads = msg.threads;
      ort.env.wasm.simd = true;
      const t0 = performance.now();
      const man = await (await fetch("./model/manifest.json")).json();
      let loaded = 0; const total = man.total;
      const tick = (n) => { loaded += n; post({ type: "progress", loaded, total }); };
      const [melFb, win, tokens, vadBytes, ...parts] = await Promise.all([
        fetchBytes("./model/mel_fb.bin", tick), fetchBytes("./model/window.bin", tick),
        fetchBytes("./model/tokens.txt", tick), fetchBytes("./model/silero_vad.onnx", tick),
        ...man.parts.map(p => fetchBytes("./model/" + p.name, tick)),
      ]);
      const model = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
      { let o = 0; for (const p of parts) { model.set(p, o); o += p.length; } }
      post({ type: "stage", stage: "Starting model…" });
      rec = await Recognizer.create(ort, {
        modelBytes: model, tokensText: new TextDecoder().decode(tokens),
        melFb: new Float32Array(melFb.buffer), window: new Float32Array(win.buffer),
      });
      micVad = await Vad.create(ort, vadBytes, { minSilenceMs: msg.silenceMs, maxSpeechSec: 20 });
      self.vadBytes = vadBytes;
      // warm-up so the first sentence isn't slow
      await rec.transcribe(new Float32Array(SR));
      post({ type: "ready", ms: performance.now() - t0, threads: msg.threads });
      break;
    }
    case "config":
      if (micVad) micVad.minSilenceSamples = Math.round(msg.silenceMs * SR / 1000);
      break;
    case "micStart": micVad.reset(); break;
    case "audio": {
      const segs = await micVad.accept(msg.samples);
      post({ type: "vad", speaking: micVad.inSpeech, prob: micVad.lastProb });
      for (const s of segs) await emit(s, "mic");
      break;
    }
    case "micStop": {
      const s = micVad.flush();
      if (s) await emit(s, "mic");
      post({ type: "vad", speaking: false, prob: 0 });
      break;
    }
    case "file": {
      const vad = await Vad.create(ort, self.vadBytes, { minSilenceMs: 500, maxSpeechSec: 20 });
      const a = msg.samples, step = SR; let pos = 0, offset = 0;
      for (let i = 0; i < a.length; i += step) {
        const segs = await vad.accept(a.subarray(i, i + step));
        pos = i + step;
        for (const s of segs) await emit(s, "file", msg.name);
        post({ type: "fileProgress", done: Math.min(pos, a.length), total: a.length, name: msg.name });
      }
      const tail = vad.flush(); if (tail) await emit(tail, "file", msg.name);
      post({ type: "fileDone", name: msg.name });
      break;
    }
  }
}

async function emit(samples, source, name) {
  const t0 = performance.now();
  const { text } = await rec.transcribe(samples);
  if (!text) return;
  post({ type: "result", id: ++segId, text, source, name, dur: samples.length / SR, ms: performance.now() - t0 });
}
