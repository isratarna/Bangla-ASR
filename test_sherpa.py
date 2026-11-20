"""
Check the exported ONNX model on the laptop BEFORE moving to the phone.
Compare its output with what NeMo's model.transcribe() gives for the same file.

    pip install sherpa-onnx soundfile
    python test_sherpa.py audio_16k_mono.wav [sherpa-model/model.int8.onnx]
"""
import sys
import time

import sherpa_onnx
import soundfile as sf

wav = sys.argv[1]
onnx_model = sys.argv[2] if len(sys.argv) > 2 else "sherpa-model/model.int8.onnx"

rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
    model=onnx_model,
    tokens="sherpa-model/tokens.txt",
    num_threads=4,
    decoding_method="greedy_search",
)

samples, sr = sf.read(wav, dtype="float32", always_2d=True)
samples = samples[:, 0]  # mono

t = time.time()
s = rec.create_stream()
s.accept_waveform(sr, samples)
rec.decode_stream(s)
dt = time.time() - t

print(s.result.text)
print(f"audio {len(samples)/sr:.2f}s, decode {dt:.2f}s, RTF {dt/(len(samples)/sr):.3f}")
