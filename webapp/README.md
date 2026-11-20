# Bangla ASR: offline web app

Bangla speech-to-text, including regional dialects (`kazalbrur/bangla-asr-conformer-120m-dialects`), running entirely
in the browser with onnxruntime-web (WebAssembly). There is no backend: audio never leaves the device.

## Run it
The folder must be opened from a web address once. Chrome won't run WebAssembly, ES modules or service
workers from a double-clicked `file://` page. After the first visit it works with no internet at all.

- **This laptop:** run `node webtest/serve.mjs` from `E:\Projects\asr` and open http://localhost:8080/webapp/.
  `python -m http.server 8080` inside `webapp/` also works.
- **Phone or any device:** upload the `webapp/` folder to a static host (HTTPS is needed for the microphone):
  - **GitHub Pages:** push the folder to a repo, then go to Settings → Pages → deploy from branch.
  - **Netlify Drop:** drag the folder onto https://app.netlify.com/drop.
  - **Cloudflare Pages:** use "Direct upload". Every file is under its 25 MB limit; the model is split into 19 MB parts.

  Open the URL once on the device and wait for the "Works offline" badge. Then choose Chrome menu → **Install app**
  (or **Add to Home screen** on Android).

## How it works
| File | Role |
|---|---|
| `asr-core.js` | NeMo-exact log-mel features (filterbank exported from the `.nemo`), Silero VAD, CTC greedy decoding |
| `worker.js` | Loads the model and runs VAD + ASR off the main thread |
| `sw.js` | Caches everything for offline use and adds COOP/COEP headers so WebAssembly can use 4 threads |
| `model/` | int8 ONNX model in 8 parts, tokens, mel filterbank, window, Silero VAD |
| `vendor/` | onnxruntime-web 1.31.1 (WebAssembly build) |

## Updating
After changing any app file, bump `SHELL_CACHE` in `sw.js` (for example `-v2`); otherwise the cached copy keeps being served.
If you replace the model, also bump `MODEL_CACHE`.

## Tests
`cd webtest && node e2e.mjs` runs the app in headless Chrome. It checks that features match NeMo, that file and
microphone transcripts match NeMo exactly, and that the app works offline with the server stopped.
