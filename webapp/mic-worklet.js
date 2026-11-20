// Collects mic audio (already resampled to the AudioContext's 16 kHz) into 100 ms blocks.
class MicCapture extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(1600); this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      for (let i = 0; i < ch.length; i++) {
        this.buf[this.n++] = ch[i];
        if (this.n === this.buf.length) {
          let s = 0; for (let j = 0; j < this.n; j++) s += this.buf[j] * this.buf[j];
          this.port.postMessage({ samples: this.buf, rms: Math.sqrt(s / this.n) }, [this.buf.buffer]);
          this.buf = new Float32Array(1600); this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("mic-capture", MicCapture);
