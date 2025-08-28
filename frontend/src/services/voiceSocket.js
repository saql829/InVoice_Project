export function createVoiceSocket({ url = "ws://localhost:8000/ws/voice", onEvent, personality = "default", voice = "alloy" }) {
  let ws = null;
  let audioPlayer = createAudioPlayer();
  let sessionId = localStorage.getItem("session_id") || null;

  const notify = (evt) => { try { onEvent?.(evt); } catch (e) { console.warn(e); } };

  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      notify({ type: "socket_open" });

      ws.send(JSON.stringify({
        type: "start",
        sample_rate: 16000,
        session: sessionId || undefined,
        personality,
        voice,
      }));
    };

    ws.onmessage = async (e) => {
      try {
        if (e.data instanceof ArrayBuffer) {
          audioPlayer.playPcm16Chunk(e.data);
          notify({ type: "tts_chunk_bin" });
          return;
        }

        let data;
        try { data = JSON.parse(e.data); } catch {
          notify({ type: "raw", text: String(e.data) });
          return;
        }

        if (data.type === "ack" && data.session) {
          sessionId = data.session;
          localStorage.setItem("session_id", sessionId);
          console.log(" Saved session:", sessionId);
        }

        switch (data.type) {
          case "tts_chunk":
            if (data.b64) {
              const bin = atob(data.b64);
              const arr = new Uint8Array(bin.length);
              for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
              audioPlayer.playPcm16Chunk(arr.buffer);
            }
            break;

          case "tts_start":
            audioPlayer.setSampleRate(data.sample_rate || 22050);
            break;

          case "tts_done":
            audioPlayer.flush();
            break;
        }

        notify(data);
      } catch (err) {
        console.warn("onmessage handling error:", err);
      }
    };

    ws.onerror = (err) => notify({ type: "error", error: String(err) });
    ws.onclose = () => {
      notify({ type: "socket_close" });
      try { audioPlayer.close(); } catch {}
      ws = null;
    };
  }

  // sendStart now accepts dynamic personality + voice
  function sendStart(sampleRate = 16000, personalityOverride = personality, voiceOverride = voice) {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: "start",
        sample_rate: sampleRate,
        session: sessionId || undefined,
        personality: personalityOverride,
        voice: voiceOverride,
      }));
    }
  }

  function sendDone() {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop", session: sessionId }));
    }
  }

  function sendAudio(buffer) {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(buffer);
    }
  }

  function stop() {
    try { ws?.close(); } catch {};
    ws = null;
    try { audioPlayer.close(); } catch {}
  }

  connect();

  return {
    sendStart,
    sendDone,
    sendAudio,
    stop,
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
      audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
      document.addEventListener("click", resumeIfNeeded, { once: true });
      document.addEventListener("keydown", resumeIfNeeded, { once: true });
    }
  }

  function resumeIfNeeded() {
    if (audioCtx?.state === "suspended") {
      audioCtx.resume().catch(()=>{});
    }
  }

  function setSampleRate(sr) { 
    if (!sr) return; 
    if (Number(sr) !== sampleRate) { 
      if (audioCtx) { try { audioCtx.close(); } catch {} audioCtx = null; } 
      sampleRate = Number(sr); 
      playTime = 0; 
    } 
    ensureCtx(); 
  }

  function close() { 
    try { audioCtx?.close(); } catch {} 
    audioCtx = null; 
    playTime = 0; 
  }

  function pcm16ToFloat32(ab) {
    const view = new DataView(ab);
    const l = view.byteLength / 2;
    const out = new Float32Array(l);
    for (let i = 0; i < l; i++) out[i] = view.getInt16(i*2, true) / 32768.0;
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
    src.start(startAt);
    playTime = startAt + buffer.duration;
  }

  function flush() {
    if (audioCtx && audioCtx.state === "suspended") {
      audioCtx.resume().catch(()=>{});
    }
    playTime = 0;
  }

  return { playPcm16Chunk, setSampleRate, close, flush };
}
