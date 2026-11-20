# Bangla ASR — on-device speech recognition

Runs the [`kazalbrur/bangla-asr-conformer-120m-dialects`](https://huggingface.co/kazalbrur/bangla-asr-conformer-120m-dialects)
NeMo Conformer-CTC model (121M params, Bangla incl. regional dialects) fully offline, with no server.

| Folder / file | What it is |
|---|---|
| [`webapp/`](webapp/) | Offline web app (PWA) for Chrome — model runs in the browser via onnxruntime-web. See [webapp/README.md](webapp/README.md) |
| `export_onnx.py` | Converts the `.nemo` model to ONNX (fp32 + int8) for sherpa-onnx / the web app |
| `test_sherpa.py`, `compare.py` | Check the ONNX model against NeMo |
| `build_apk.sh`, `patch_app.py` | Build an Android APK (sherpa-onnx VAD + ASR demo pointed at this model) |
| `webtest/` | Headless-Chrome end-to-end test of the web app (`cd webtest && npm i && node e2e.mjs`) |
| `test_wavs/` | Short Bangla test clips |

## Quick start (web app)
```
node webtest/serve.mjs
```
Open http://localhost:8080/webapp/ in Chrome, wait for **Works offline**, then it keeps working without the server.
For phones, host the `webapp/` folder on any HTTPS static host (e.g. GitHub Pages) and open it once.

## Regenerating the large files
`sherpa-model/` and the APK are not in the repo (GitHub's 100 MB limit):
```
python export_onnx.py path/to/bangla-stt-conformer-120m-dialects.nemo   # -> sherpa-model/
bash build_apk.sh                                                       # -> BanglaASR.apk (run in WSL/Linux)
```

Model licence: Apache-2.0 (kazalbrur). onnxruntime-web: MIT. Silero VAD: MIT.
