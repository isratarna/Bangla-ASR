// Core speech recognition engine: NeMo-exact log-mel features, Silero VAD (v4), CTC greedy decoding.
// Pure ES module with no DOM use, so it runs in a Web Worker and in Node (for tests).

export const SR = 16000;
const N_FFT = 512, HOP = 160, WIN = 400, N_MELS = 80, N_BINS = N_FFT / 2 + 1;
const PREEMPH = 0.97, LOG_GUARD = 2 ** -24, STD_EPS = 1e-5;

// ---------- FFT (radix-2, in place) ----------
const fftCos = new Float64Array(N_FFT / 2), fftSin = new Float64Array(N_FFT / 2), fftRev = new Uint16Array(N_FFT);
for (let i = 0; i < N_FFT / 2; i++) { fftCos[i] = Math.cos(-2 * Math.PI * i / N_FFT); fftSin[i] = Math.sin(-2 * Math.PI * i / N_FFT); }
for (let i = 0, bits = Math.log2(N_FFT); i < N_FFT; i++) {
  let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); fftRev[i] = r;
}
function fft(re, im) {
  for (let i = 0; i < N_FFT; i++) { const j = fftRev[i]; if (j > i) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let size = 2; size <= N_FFT; size <<= 1) {
    const half = size >> 1, step = N_FFT / size;
    for (let start = 0; start < N_FFT; start += size) {
      for (let k = 0; k < half; k++) {
        const c = fftCos[k * step], s = fftSin[k * step], a = start + k, b = a + half;
        const tr = re[b] * c - im[b] * s, ti = re[b] * s + im[b] * c;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
  }
}

// ---------- NeMo AudioToMelSpectrogramPreprocessor (eval mode) ----------
// melFb: Float32Array(80*257) row-major, window: Float32Array(400) — both exported from the .nemo featurizer.
export function nemoFeatures(audio, melFb, window) {
  const L = audio.length;
  const validFrames = Math.floor(L / HOP);          // NeMo seq_len
  const nFrames = validFrames + 1;                  // torch.stft(center=True) frame count
  // pre-emphasis
  const x = new Float32Array(L);
  if (L) x[0] = audio[0];
  for (let i = 1; i < L; i++) x[i] = audio[i] - PREEMPH * audio[i - 1];
  // window zero-padded to n_fft, centred (torch.stft behaviour)
  const winPad = new Float64Array(N_FFT), off = (N_FFT - WIN) >> 1;
  for (let i = 0; i < WIN; i++) winPad[off + i] = window[i];

  const feats = new Float32Array(N_MELS * nFrames);   // layout [mel][frame]
  const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT), pow = new Float64Array(N_BINS);
  for (let t = 0; t < validFrames; t++) {
    const start = t * HOP - N_FFT / 2;               // centre padding with zeros
    for (let n = 0; n < N_FFT; n++) {
      const idx = start + n;
      re[n] = (idx >= 0 && idx < L) ? x[idx] * winPad[n] : 0; im[n] = 0;
    }
    fft(re, im);
    for (let k = 0; k < N_BINS; k++) pow[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < N_MELS; m++) {
      let s = 0; const row = m * N_BINS;
      for (let k = 0; k < N_BINS; k++) s += melFb[row + k] * pow[k];
      feats[m * nFrames + t] = Math.log(s + LOG_GUARD);
    }
  }
  // per-feature normalisation over valid frames (unbiased std), padded frame stays 0
  for (let m = 0; m < N_MELS; m++) {
    const row = m * nFrames;
    let mean = 0; for (let t = 0; t < validFrames; t++) mean += feats[row + t];
    mean /= validFrames;
    let v = 0; for (let t = 0; t < validFrames; t++) { const d = feats[row + t] - mean; v += d * d; }
    let std = validFrames > 1 ? Math.sqrt(v / (validFrames - 1)) : 0;
    std += STD_EPS;
    for (let t = 0; t < validFrames; t++) feats[row + t] = (feats[row + t] - mean) / std;
    feats[row + validFrames] = 0;
  }
  return { feats, nFrames, validFrames };
}

// ---------- tokens / CTC ----------
export function parseTokens(text) {
  const vocab = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const sp = line.lastIndexOf(" ");
    vocab[parseInt(line.slice(sp + 1), 10)] = line.slice(0, sp);
  }
  return vocab;
}

export function ctcGreedy(logprobs, T, V, vocab) {
  const blank = V - 1; const ids = []; let prev = -1;
  for (let t = 0; t < T; t++) {
    let best = 0, bv = -Infinity; const o = t * V;
    for (let v = 0; v < V; v++) if (logprobs[o + v] > bv) { bv = logprobs[o + v]; best = v; }
    if (best !== prev && best !== blank) ids.push(best);
    prev = best;
  }
  return { ids, text: ids.map(i => vocab[i]).join("").replace(/▁/g, " ").replace(/\s+/g, " ").trim() };
}

// ---------- ASR model ----------
export class Recognizer {
  static async create(ort, { modelBytes, tokensText, melFb, window, sessionOptions = {} }) {
    const r = new Recognizer();
    r.ort = ort; r.vocab = parseTokens(tokensText); r.melFb = melFb; r.window = window;
    r.session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all", ...sessionOptions });
    return r;
  }
  async transcribe(audio) {
    if (audio.length < HOP * 2) return { text: "", ids: [] };
    const { feats, nFrames, validFrames } = nemoFeatures(audio, this.melFb, this.window);
    const out = await this.session.run({
      audio_signal: new this.ort.Tensor("float32", feats, [1, N_MELS, nFrames]),
      length: new this.ort.Tensor("int64", BigInt64Array.from([BigInt(validFrames)]), [1]),
    });
    const lp = out[this.session.outputNames[0]];
    const [, T, V] = lp.dims;
    const res = ctcGreedy(lp.data, T, V, this.vocab);
    for (const k in out) out[k].dispose?.();
    return res;
  }
}

// ---------- Silero VAD v4 (inputs x[1,512], h, c) with VADIterator-style segmentation ----------
export class Vad {
  static async create(ort, modelBytes, opts = {}) {
    const v = new Vad();
    v.ort = ort;
    v.session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
    v.threshold = opts.threshold ?? 0.5;
    v.minSilenceSamples = Math.round((opts.minSilenceMs ?? 600) * SR / 1000);
    v.minSpeechSamples = Math.round((opts.minSpeechMs ?? 250) * SR / 1000);
    v.padSamples = Math.round((opts.speechPadMs ?? 200) * SR / 1000);
    v.maxSpeechSamples = Math.round((opts.maxSpeechSec ?? 20) * SR);
    v.reset();
    return v;
  }
  reset() {
    this.h = new Float32Array(128); this.c = new Float32Array(128);
    this.pending = new Float32Array(0);
    this.history = [];        // recent windows kept for pre-roll padding
    this.speech = null;       // array of windows while in speech
    this.silence = 0; this.lastProb = 0;
  }
  async prob(win) {
    const T = this.ort.Tensor;
    const out = await this.session.run({ x: new T("float32", win, [1, 512]), h: new T("float32", this.h, [2, 1, 64]), c: new T("float32", this.c, [2, 1, 64]) });
    this.h = out.new_h.data.slice(); this.c = out.new_c.data.slice();
    return out.prob.data[0];
  }
  // Feed audio; returns array of finished speech segments (Float32Array).
  async accept(samples) {
    const segs = [];
    const buf = new Float32Array(this.pending.length + samples.length);
    buf.set(this.pending); buf.set(samples, this.pending.length);
    let i = 0;
    for (; i + 512 <= buf.length; i += 512) {
      const win = buf.slice(i, i + 512);
      const p = await this.prob(win); this.lastProb = p;
      if (!this.speech) {
        this.history.push(win);
        const keep = Math.ceil(this.padSamples / 512) + 1;
        if (this.history.length > keep) this.history.shift();
        if (p >= this.threshold) { this.speech = this.history.slice(); this.history = []; this.silence = 0; }
      } else {
        this.speech.push(win);
        if (p < this.threshold - 0.15) this.silence += 512; else this.silence = 0;
        const len = this.speech.length * 512;
        if (this.silence >= this.minSilenceSamples || len >= this.maxSpeechSamples) {
          // trim trailing silence but keep padding
          const trim = Math.max(0, Math.floor((this.silence - this.padSamples) / 512));
          const wins = this.speech.slice(0, this.speech.length - trim);
          if (wins.length * 512 - this.silence + trim * 512 >= this.minSpeechSamples) segs.push(concat(wins));
          this.speech = null; this.silence = 0;
        }
      }
    }
    this.pending = buf.slice(i);
    return segs;
  }
  // Flush any speech in progress (e.g. when recording stops or a file ends).
  flush() {
    const s = this.speech && this.speech.length * 512 >= this.minSpeechSamples ? concat(this.speech) : null;
    this.speech = null; this.silence = 0; this.pending = new Float32Array(0);
    return s;
  }
  get inSpeech() { return !!this.speech; }
}

function concat(arrs) {
  let n = 0; for (const a of arrs) n += a.length;
  const out = new Float32Array(n); let o = 0;
  for (const a of arrs) { out.set(a, o); o += a.length; }
  return out;
}
