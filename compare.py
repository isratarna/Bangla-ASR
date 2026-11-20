"""Compare NeMo vs sherpa-onnx (fp32 / int8) transcripts + speed on test_wavs/*.wav"""
import glob, sys, time, os
import soundfile as sf, torch, sherpa_onnx
import nemo.collections.asr as nemo_asr
torch.set_num_threads(4)
wavs = sorted(glob.glob("test_wavs/*.wav"))
nemo_path = sys.argv[1]
m = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location="cpu").eval()
def onnx_rec(p):
    return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(model=p, tokens="sherpa-model/tokens.txt", num_threads=4)
recs = {"fp32": onnx_rec("sherpa-model/model.onnx"), "int8": onnx_rec("sherpa-model/model.int8.onnx")}
for w in wavs:
    a, sr = sf.read(w, dtype="float32")
    dur = len(a)/sr
    with torch.inference_mode():
        t=time.time(); out = m.transcribe([w], verbose=False)[0]; tn=time.time()-t
    print(f"\n== {os.path.basename(w)} ({dur:.1f}s)")
    print(f"  nemo  ({tn:.2f}s): {getattr(out,'text',out)}")
    for k, r in recs.items():
        s = r.create_stream(); t=time.time(); s.accept_waveform(sr, a); r.decode_stream(s)
        print(f"  {k:5} ({time.time()-t:.2f}s): {s.result.text}")
