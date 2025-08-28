// frontend/src/hooks/useRecorder.js
import { useEffect, useRef, useState } from "react";
import { downsampleTo16k, float32ToInt16 } from "../utils/audio";

export function useRecorder(options = {}) {
  const { onPcmChunk, vadThreshold = 0.01 } = options; // energy threshold configurable

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
        if (!float32_48k || !float32_48k.length) return;

        // energy calc
        const energy = Math.sqrt(
          float32_48k.reduce((acc, val) => acc + val * val, 0) /
            float32_48k.length
        );

        // VAD
        if (energy > vadThreshold) {
          // 🔹 Normalize before downsampling
          const normalized = normalizeFloat32(float32_48k);

          const float32_16k = downsampleTo16k(normalized, 48000);
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

  // 🔹 Normalize input audio to avoid mishearing ("Pakistan" → "aksan")
  function normalizeFloat32(float32) {
    let max = 0;
    for (let i = 0; i < float32.length; i++) {
      max = Math.max(max, Math.abs(float32[i]));
    }
    if (max < 1e-5) return float32;
    const factor = 1 / max;
    return float32.map(v => v * factor);
  }

  return { start, stop, recording, error };
}
