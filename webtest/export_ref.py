"""Export NeMo featurizer constants for the web app + reference outputs for parity tests."""
import sys, json, numpy as np, soundfile as sf, torch, glob, os
import nemo.collections.asr as nemo_asr
m = nemo_asr.models.ASRModel.restore_from(sys.argv[1], map_location="cpu").eval()
fz = m.preprocessor.featurizer
fb = fz.fb[0].numpy().astype("<f4")          # (80, 257)
win = fz.window.numpy().astype("<f4")         # (400,)
print("fb", fb.shape, "win", win.shape, "n_fft", fz.n_fft, "hop", fz.hop_length, "preemph", fz.preemph,
      "guard", fz.log_zero_guard_value, "exact_pad", fz.exact_pad, "pad_to", fz.pad_to, "mag", fz.mag_power)
fb.tofile("webapp/model/mel_fb.bin"); win.tofile("webapp/model/window.bin")
ref = {}
for w in sorted(glob.glob("test_wavs/*.wav")):
    a, sr = sf.read(w, dtype="float32")
    with torch.inference_mode():
        feat, fl = m.preprocessor(input_signal=torch.tensor(a)[None], length=torch.tensor([len(a)]))
        lp = m.forward(processed_signal=feat, processed_signal_length=fl)[0][0].numpy()
        txt = m.transcribe([w], verbose=False)[0]
    name = os.path.basename(w)[:-4]
    feat[0].numpy().astype("<f4").tofile(f"webtest/{name}.feat.bin")
    ref[name] = {"feat_shape": list(feat[0].shape), "len": int(fl[0]), "text": getattr(txt, "text", txt),
                 "argmax": lp.argmax(-1).tolist()}
json.dump(ref, open("webtest/ref.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: (v["feat_shape"], v["len"], v["text"]) for k, v in ref.items()}, ensure_ascii=False))
