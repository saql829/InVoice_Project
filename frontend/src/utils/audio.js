// frontend/src/utils/audio.js

export function downsampleTo16k(float32, inputRate) {
  if (inputRate === 16000) return float32;
  const ratio = inputRate / 16000;
  const newLen = Math.floor(float32.length / ratio);
  const out = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) out[i] = float32[Math.floor(i * ratio)];
  return out;
}

export function float32ToInt16(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    let s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return out;
}
