#!/bin/bash
# Builds the Bangla ASR Android APK inside WSL. Re-runnable.
set -euo pipefail
R=~/android; cd $R
APP=$R/sherpa-onnx/android/SherpaOnnxVadAsr
V=2.0.2

# 1) JDK 17 (no sudo)
if [ ! -d jdk17 ]; then
  echo ">> JDK"; curl -sSL -o jdk.tgz "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse"
  mkdir jdk17 && tar xzf jdk.tgz -C jdk17 --strip-components=1 && rm jdk.tgz
fi
export JAVA_HOME=$R/jdk17 PATH=$R/jdk17/bin:$PATH

# 2) Android SDK
export ANDROID_HOME=$R/sdk ANDROID_SDK_ROOT=$R/sdk
if [ ! -d sdk/cmdline-tools/latest ]; then
  echo ">> Android cmdline-tools"; curl -sSL -o clt.zip https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
  mkdir -p sdk/cmdline-tools && ~/asr311/bin/python -c "import zipfile;zipfile.ZipFile('clt.zip').extractall('sdk/cmdline-tools')"
  mv sdk/cmdline-tools/cmdline-tools sdk/cmdline-tools/latest && chmod +x sdk/cmdline-tools/latest/bin/* && rm clt.zip
fi
SDKM=sdk/cmdline-tools/latest/bin/sdkmanager
yes | $SDKM --licenses >/dev/null 2>&1 || true
$SDKM "platforms;android-33" "build-tools;30.0.3" "platform-tools" >/dev/null

# 3) prebuilt sherpa-onnx native libs
if [ ! -f $APP/app/src/main/jniLibs/arm64-v8a/libsherpa-onnx-jni.so ]; then
  echo ">> native libs"; [ -f libs.tar.bz2 ] || curl -sSL -o libs.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/v$V/sherpa-onnx-v$V-android.tar.bz2
  mkdir -p libs && ~/asr311/bin/python -c "import tarfile;tarfile.open('libs.tar.bz2').extractall('libs', filter='data')" && rm libs.tar.bz2
  cp -v $(find libs -path '*arm64-v8a*' -name '*.so') $APP/app/src/main/jniLibs/arm64-v8a/
fi

# 4) assets: VAD + Bangla model
A=$APP/app/src/main/assets; mkdir -p $A/bangla-conformer
[ -f $A/silero_vad.onnx ] || curl -sSL -o $A/silero_vad.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
cp /mnt/e/Projects/asr/sherpa-model/model.int8.onnx /mnt/e/Projects/asr/sherpa-model/tokens.txt $A/bangla-conformer/

# 5) patch app (idempotent)
~/asr311/bin/python /mnt/e/Projects/asr/patch_app.py $APP

# 6) build
cd $APP && chmod +x gradlew && ./gradlew --no-daemon -q assembleDebug
OUT=$APP/app/build/outputs/apk/debug/app-debug.apk
cp $OUT /mnt/e/Projects/asr/BanglaASR.apk && ls -la /mnt/e/Projects/asr/BanglaASR.apk
