// frontend/src/services/voiceSocket.js
// Handles JSON messages and binary PCM16 streaming for TTS + STT.

export function createVoiceSocket({ url = "ws://localhost:8000/ws/voice", onEvent }) {
  let ws = null;
  let audioPlayer = createAudioPlayer();
  const notify = (evt) => { try { onEvent?.(evt); } catch (e) { console.warn(e); } };

  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      notify({ type: "socket_open" });
      try { ws.send(JSON.stringify({ type: "start", sample_rate: 16000 })); } catch {}
    };

    ws.onmessage = (e) => {
      try {
        if (e.data instanceof ArrayBuffer) {
          try { audioPlayer.playPcm16Chunk(e.data); } catch (err) { console.warn("audio play error:", err); }
          return;
        }

        let data;
        try { data = JSON.parse(e.data); } catch (err) { notify({ type: "raw", text: String(e.data) }); return; }

        if (data.type === "tts_chunk" && data.b64) {
          const bin = atob(data.b64);
          const arr = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
          audioPlayer.playPcm16Chunk(arr.buffer);
          notify(data);
          return;
        }

        if (data.type === "tts_start") {
          const sr = data.sample_rate || 22050;
          audioPlayer.setSampleRate(sr);
          notify({ type: "tts_start", sample_rate: sr, format: data.format });
          return;
        }
        if (data.type === "tts_done") {
          notify({ type: "tts_done" });
          return;
        }

        notify(data);

      } catch (err) {
        console.warn("onmessage handling error:", err);
      }
    };

    ws.onerror = (err) => { notify({ type: "error", error: String(err) }); };
    ws.onclose = () => { notify({ type: "socket_close" }); try { audioPlayer.close(); } catch {}; ws = null; };
  }

  function sendStart(sampleRate = 16000) { if (!ws || ws.readyState !== WebSocket.OPEN) return; try { ws.send(JSON.stringify({ type: "start", sample_rate: sampleRate })); } catch {} }
  function sendDone() { if (!ws || ws.readyState !== WebSocket.OPEN) return; try { ws.send("done"); } catch {} }
  function sendAudio(buffer) { if (!ws || ws.readyState !== WebSocket.OPEN) return; try { ws.send(buffer); } catch {} }
  function stop() { try { ws?.close(); } catch {}; ws = null; try { audioPlayer.close(); } catch {} }

  connect();

  return {
    sendStart,
    sendDone,
    sendAudio,
    stop,
    playTTS: (arrayBuffer) => audioPlayer.playPcm16Chunk(arrayBuffer),
    get ready() { return ws?.readyState === WebSocket.OPEN; },
  };
}

// ------------------- audio player -------------------
function createAudioPlayer() {
  let audioCtx = null;
  let sampleRate = 22050;
  let playTime = 0;

  function ensureCtx() {
    if (!audioCtx) {
      try { audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate }); } catch { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); }
      document.addEventListener("click", resumeIfNeeded, { once: true });
      document.addEventListener("keydown", resumeIfNeeded, { once: true });
    }
  }

  function resumeIfNeeded() { if (audioCtx && audioCtx.state === "suspended") { audioCtx.resume().catch(()=>{}); } }
  function setSampleRate(sr) { if (!sr) return; if (Number(sr) !== sampleRate) { if (audioCtx) { try { audioCtx.close(); } catch {} audioCtx = null; } sampleRate = Number(sr); playTime = 0; } ensureCtx(); }
  function close() { try { audioCtx?.close(); } catch {} audioCtx = null; playTime = 0; }

  function pcm16ToFloat32(ab) {
    const view = new DataView(ab);
    const l = view.byteLength / 2;
    const out = new Float32Array(l);
    for (let i = 0; i < l; i++) { out[i] = view.getInt16(i*2, true) / 32768.0; }
    return out;
  }

  function playPcm16Chunk(arrayBuffer) {
    ensureCtx();
    if (!audioCtx) return;
    const float32 = pcm16ToFloat32(arrayBuffer);
    const buffer = audioCtx.createBuffer(1, float32.length, audioCtx.sampleRate);
    buffer.copyToChannel(float32, 0, 0);
    const src = audioCtx.createBufferSource();
    src.buffer = buffer;
    src.connect(audioCtx.destination);
    const now = audioCtx.currentTime;
    const startAt = Math.max(now + 0.05, playTime || now + 0.05);
    try { src.start(startAt); } catch { try { src.start(); } catch {} }
    playTime = startAt + buffer.duration;
  }

  return { playPcm16Chunk, setSampleRate, close };
}
