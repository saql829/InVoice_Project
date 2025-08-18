import { useEffect, useRef, useState } from "react";
import { downsampleTo16k, float32ToInt16 } from "../utils/audio";

export function useRecorder(options = {}) {
  const { onPcmChunk } = options || {};   // safe destructure

  const [recording, setRecording] = useState(false);
  const [error, setError] = useState(null);
  const audioCtxRef = useRef(null);
  const workletRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    return () => stop();
  }, []);

  async function start() {
    if (recording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 48000,
      });
      audioCtxRef.current = audioCtx;

      await audioCtx.audioWorklet.addModule("/processor.js");
      const source = audioCtx.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(audioCtx, "pcm-capture");
      workletRef.current = node;

      node.port.onmessage = (e) => {
        const float32_48k = e.data;
        const energy = Math.sqrt(
          float32_48k.reduce((acc, val) => acc + val * val, 0) /
            float32_48k.length
        );

        // VAD: sirf jab volume kaafi high ho
        if (energy > 0.01) {
          const float32_16k = downsampleTo16k(float32_48k, 48000);
          const int16 = float32ToInt16(float32_16k);

          if (typeof onPcmChunk === "function") {
            onPcmChunk(int16.buffer);
          }
        }
      };

      source.connect(node);
      setRecording(true);
    } catch (err) {
      setError(String(err));
      stop();
    }
  }

  function stop() {
    setRecording(false);
    try {
      workletRef.current?.disconnect();
    } catch {}
    try {
      audioCtxRef.current?.close();
    } catch {}
    try {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    } catch {}
    workletRef.current = null;
    audioCtxRef.current = null;
    streamRef.current = null;
  }

  return { start, stop, recording, error };
}
