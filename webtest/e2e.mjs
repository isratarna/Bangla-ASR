// End-to-end test of the web app in real (headless) Chrome.
//   cd webtest && node e2e.mjs
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import puppeteer from "puppeteer-core";
import { serve } from "./serve.mjs";

const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const ROOT = path.resolve("..");
const URL_ = "http://localhost:8080/webapp/";
const ref = JSON.parse(fs.readFileSync("ref.json", "utf8"));
const profile = fs.mkdtempSync(path.join(os.tmpdir(), "bnasr-"));
let failures = 0;
const check = (ok, msg) => { console.log(`${ok ? "PASS" : "FAIL"}  ${msg}`); if (!ok) failures++; };

let server = await serve(8080);
const browser = await puppeteer.launch({
  executablePath: CHROME, headless: true, userDataDir: profile,
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
    `--use-file-for-fake-audio-capture=${path.join(ROOT, "webtest", "mic_input.wav")}`, "--autoplay-policy=no-user-gesture-required"],
});
let page = await browser.newPage();
page.on("console", (m) => { if (m.type() === "error" || m.type() === "warn") console.log("  [page]", m.type(), m.text()); });
page.on("pageerror", (e) => console.log("  [pageerror]", e.message));

// ---- 1. first load (downloads + caches everything) ----
let t0 = Date.now();
await page.goto(URL_);
await page.waitForSelector("body[data-ready='1']", { timeout: 300000 });
const info = await page.evaluate(() => ({ coi: self.crossOriginIsolated, engine: document.getElementById("engineInfo").textContent }));
console.log(`first load ready in ${((Date.now() - t0) / 1000).toFixed(1)}s — ${info.engine}`);
check(info.coi, "page is cross-origin isolated (multi-threaded wasm) via service worker");

// ---- 2. feature parity inside Chrome ----
const parity = await page.evaluate(async (names) => {
  const core = await import("/webapp/asr-core.js");
  const f32 = async (u) => new Float32Array(await (await fetch(u)).arrayBuffer());
  const fb = await f32("/webapp/model/mel_fb.bin"), win = await f32("/webapp/model/window.bin");
  const out = {};
  for (const n of names) {
    const buf = await (await fetch(`/test_wavs/${n}.wav`)).arrayBuffer();
    const dv = new DataView(buf); let o = 12, audio;
    while (o < buf.byteLength) { const id = String.fromCharCode(...new Uint8Array(buf, o, 4)), sz = dv.getUint32(o + 4, true);
      if (id === "data") { audio = new Float32Array(sz / 2); for (let i = 0; i < audio.length; i++) audio[i] = dv.getInt16(o + 8 + 2 * i, true) / 32768; break; } o += 8 + sz; }
    const { feats } = core.nemoFeatures(audio, fb, win);
    const refF = await f32(`/webtest/${n}.feat.bin`);
    let m = 0; for (let i = 0; i < refF.length; i++) m = Math.max(m, Math.abs(refF[i] - feats[i]));
    out[n] = m;
  }
  return out;
}, Object.keys(ref));
for (const [n, d] of Object.entries(parity)) check(d < 1e-3, `${n}: features match NeMo (max diff ${d.toExponential(2)})`);

// ---- 3. file transcription ----
async function transcribeFile(file) {
  const before = await page.evaluate(() => document.querySelectorAll("#results li").length);
  const input = await page.$("#fileInput");
  await page.waitForFunction(() => !document.getElementById("fileInput").disabled);
  await input.uploadFile(file);
  await page.waitForFunction((f) => document.body.dataset.fileDone === f, { timeout: 120000 }, path.basename(file));
  await page.evaluate(() => delete document.body.dataset.fileDone);
  return page.evaluate((b) => [...document.querySelectorAll("#results li .text")].slice(0, document.querySelectorAll("#results li").length - b).map((e) => e.textContent).reverse(), before);
}
for (const [n, r] of Object.entries(ref)) {
  const texts = await transcribeFile(path.join(ROOT, "test_wavs", `${n}.wav`));
  check(texts.join(" ") === r.text, `${n} file → "${texts.join(" ")}"  (NeMo: "${r.text}")`);
}
const meta = await page.$$eval("#results li .meta", (els) => els.slice(0, 3).map((e) => e.textContent));
console.log("  timings:", meta.join(" | "));

// ---- 4. microphone path (Chrome fake mic plays mic_input.wav: t1, t2, t3 with pauses, looping) ----
await page.click("#clear");
await page.click("#micBtn");
await page.waitForFunction(() => document.querySelectorAll("#results li").length >= 3, { timeout: 60000 }).catch(() => {});
await page.click("#micBtn");
const micTexts = await page.$$eval("#results li .text", (els) => els.map((e) => e.textContent).reverse());
console.log("  mic results:", micTexts);
const want = Object.values(ref).map((r) => r.text);
check(micTexts.length >= 3 && want.every((w) => micTexts.some((t) => t === w)), "mic: all three sentences recognised exactly");

// ---- 5. offline: stop the server, reload, transcribe again ----
server.closeAllConnections(); await new Promise((r) => server.close(r)); server = null;
await page.evaluate(() => new Promise((r) => setTimeout(r, 1500)));
const badge = await page.evaluate(() => !document.getElementById("offlineBadge").hidden);
check(badge, "'Works offline' badge shown (all model files cached)");
// (puppeteer's page.reload() uses a CDP path that skips the service worker; use a real reload / fresh open)
for (const how of ["reload (F5)", "fresh open in new tab"]) {
  t0 = Date.now();
  if (how.startsWith("reload")) {
    await Promise.all([page.waitForNavigation(), page.evaluate(() => location.reload())]);
  } else {
    await page.close(); page = await browser.newPage(); await page.goto(URL_);
  }
  await page.waitForSelector("body[data-ready='1']", { timeout: 120000 });
  const off = await transcribeFile(path.join(ROOT, "test_wavs", "t2.wav"));
  check(off.join(" ") === ref.t2.text, `offline ${how}: ready in ${((Date.now() - t0) / 1000).toFixed(1)}s, transcription → "${off.join(" ")}"`);
}

await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2 });
await page.screenshot({ path: "screenshot-phone.png", fullPage: true });
const scrollW = await page.evaluate(() => document.documentElement.scrollWidth);
check(scrollW <= 390, `no horizontal scroll at phone width (scrollWidth ${scrollW})`);

await browser.close();
fs.rmSync(profile, { recursive: true, force: true });
console.log(failures ? `\n${failures} FAILED` : "\nALL PASSED");
process.exit(failures ? 1 : 0);
