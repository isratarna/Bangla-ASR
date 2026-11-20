"""
Export kazalbrur/bangla-asr-conformer-120m-dialects (.nemo) to ONNX for sherpa-onnx.

Usage (on the laptop, in the same env where NeMo already works):
    pip install onnx onnxruntime
    python export_onnx.py path/to/bangla-stt-conformer-120m-dialects.nemo

Output folder ./sherpa-model/:
    model.onnx        fp32  (~480 MB)
    model.int8.onnx   int8  (~125 MB)  <- use this on the phone
    tokens.txt
"""
import os
import sys

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
import nemo.collections.asr as nemo_asr

NEMO_PATH = sys.argv[1] if len(sys.argv) > 1 else "bangla-stt-conformer-120m-dialects.nemo"
OUT = "sherpa-model"
os.makedirs(OUT, exist_ok=True)

model = nemo_asr.models.ASRModel.restore_from(NEMO_PATH, map_location="cpu")
model.eval()

# 1) tokens.txt  (CTC blank is the last id in NeMo)
vocab = model.decoder.vocabulary
with open(f"{OUT}/tokens.txt", "w", encoding="utf-8") as f:
    for i, tok in enumerate(vocab):
        f.write(f"{tok} {i}\n")
    f.write(f"<blk> {len(vocab)}\n")

# 2) ONNX export (encoder + CTC head; feature extraction is done by sherpa-onnx)
fp32_path = f"{OUT}/model.onnx"
model.export(fp32_path)

# 3) metadata sherpa-onnx needs to pick the right feature normalisation / decoding
normalize = model.cfg.preprocessor.get("normalize", "per_feature")
if normalize not in ("per_feature", "all_features"):
    normalize = ""
meta = {
    "vocab_size": model.decoder.num_classes_with_blank,
    "normalize_type": normalize,
    "subsampling_factor": 4,
    "model_type": "EncDecCTCModelBPE",
    "version": "1",
    "model_author": "NeMo",
    "comment": "kazalbrur/bangla-asr-conformer-120m-dialects",
}
m = onnx.load(fp32_path)
for k, v in meta.items():
    p = m.metadata_props.add()
    p.key, p.value = k, str(v)
onnx.save(m, fp32_path)
print("metadata:", meta)

# 4) int8 quantisation for the phone
quantize_dynamic(
    model_input=fp32_path,
    model_output=f"{OUT}/model.int8.onnx",
    weight_type=QuantType.QUInt8,
)
print(f"Done -> {OUT}/model.int8.onnx, {OUT}/tokens.txt")
