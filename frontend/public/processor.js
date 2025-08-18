class PcmCapture extends AudioWorkletProcessor {
    process(inputs) {
      const input = inputs[0];
      if (input && input[0]) {
        // Float32 mono block @ context sample rate (usually 48k)
        this.port.postMessage(input[0]);
      }
      return true;
    }
  }
  registerProcessor("pcm-capture", PcmCapture);
  