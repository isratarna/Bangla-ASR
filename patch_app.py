"""Point the sherpa-onnx VAD+ASR Android demo at the Bangla Conformer model."""
import pathlib, re, sys
app = pathlib.Path(sys.argv[1])
ma = app / "app/src/main/java/com/k2fsa/sherpa/onnx/MainActivity.kt"
s = ma.read_text()
if "bangla-conformer" not in s:
    s = s.replace("import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig\n",
        "import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig\n"
        "import com.k2fsa.sherpa.onnx.OfflineModelConfig\n"
        "import com.k2fsa.sherpa.onnx.OfflineNemoEncDecCtcModelConfig\n")
    s = s.replace("modelConfig = getOfflineModelConfig(type = asrModelType)!!,",
        'modelConfig = OfflineModelConfig(\n'
        '                nemo = OfflineNemoEncDecCtcModelConfig(model = "bangla-conformer/model.int8.onnx"),\n'
        '                tokens = "bangla-conformer/tokens.txt",\n'
        '                numThreads = 4,\n'
        '            ),')
    s = s.replace("val config = getVadModelConfig(type)\n",
        "val base = getVadModelConfig(type)!!\n"
        "        // end a sentence after a 0.5 s pause (laptop script used 0.7 s)\n"
        "        val config = base.copy(sileroVadModelConfig = base.sileroVadModelConfig.copy(minSilenceDuration = 0.5F))\n")
    assert s.count("bangla-conformer") == 2 and "minSilenceDuration = 0.5F" in s, "patch failed"
    ma.write_text(s)
g = app / "app/build.gradle"
gs = g.read_text()
gs = gs.replace('applicationId "com.k2fsa.sherpa.onnx"', 'applicationId "com.k2fsa.sherpa.onnx.bangla"')
gs = re.sub(r"(android \{\n)(?!    aaptOptions)", r'\1    aaptOptions { noCompress "onnx" }\n', gs)
g.write_text(gs)
st = app / "app/src/main/res/values/strings.xml"
st.write_text(re.sub(r'<string name="app_name">.*?</string>', '<string name="app_name">Bangla ASR</string>', st.read_text()))
print("patched", ma)
