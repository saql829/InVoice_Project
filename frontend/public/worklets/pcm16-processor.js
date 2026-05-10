// public/worklets/pcm16-processor.js
// AudioWorklet: 48k (device) -> 16k mono, 20ms frames (320 samples), PCM16LE packets.

class Pcm16Downsampler extends AudioWorkletProcessor {
    constructor() {
      super();
      this.targetRate = 16000;
      this.ratio = sampleRate / this.targetRate; // sampleRate = AudioContext rate (usually 48001)
      this.inbuf = new Float32Array(0); // accumulate input @ sampleRate
      this.rpos = 0; // read pointer (float) over inbuf
      this.outbuf = new Float32Array(0); // accumulated @16k (float)
      this.FRAME_SAMPLES = 320; // 20ms at 16k
    }
  
    appendFloat(a, b) {
      // concat 2 Float32Array
      const out = new Float32Array(a.length + b.length);
      out.set(a, 0);
      out.set(b, a.length);
      return out;
    }
  
    process(inputs) {
      const input = inputs[0] && inputs[0][0] ? inputs[0][0] : null;
      if (!input) return true;
  
      // 1) Accumulate input (mono)
      this.inbuf = this.appendFloat(this.inbuf, input);
  
      // 2) Produce as many 16k samples as possible (linear interpolation)
      const produced = [];
      const len = this.inbuf.length;
      while (true) {
        const i0 = Math.floor(this.rpos);
        const i1 = i0 + 1;
        if (i1 >= len) break;
        const frac = this.rpos - i0;
        const s0 = this.inbuf[i0];
        const s1 = this.inbuf[i1];
        const s = s0 + (s1 - s0) * frac;
        produced.push(s);
        this.rpos += this.ratio;
      }
  
      if (produced.length) {
        const chunk = new Float32Array(produced.length);
        for (let i = 0; i < produced.length; i++) {
          chunk[i] = produced[i];
        }
        this.outbuf = this.appendFloat(this.outbuf, chunk);
      }
  
      // 3) Drop consumed input
      const drop = Math.floor(this.rpos);
      if (drop > 0) {
        this.inbuf = this.inbuf.subarray(drop);
        this.rpos -= drop;
      }
  
      // 4) Flush 20ms frames as PCM16LE
      while (this.outbuf.length >= this.FRAME_SAMPLES) {
        const frame = this.outbuf.subarray(0, this.FRAME_SAMPLES);
        this.outbuf = this.outbuf.subarray(this.FRAME_SAMPLES);
  
        const pcm = new Int16Array(this.FRAME_SAMPLES);
        for (let i = 0; i < this.FRAME_SAMPLES; i++) {
          // clamp & convert
          let v = Math.max(-1, Math.min(1, frame[i]));
          pcm[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
        }
        // send raw 20ms (320 * 2 = 640 bytes) packet to main thread
        this.port.postMessage({ type: 'packet', buffer: pcm.buffer }, [pcm.buffer]);
      }
  
      return true;
    }
  }
  
  registerProcessor('pcm16-writer', Pcm16Downsampler);
  