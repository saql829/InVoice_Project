import { useEffect, useRef, useState, useCallback } from "react";
import MicButton from "./components/MicButton";
import TranscriptBox from "./components/TranscriptBox";
import StreamBox from "./components/StreamBox";
import HistoryPanel from "./components/HistoryPanel";   // ✅ history panel
import VoiceBubble from "./components/VoiceBubble";     // ✅ bubble
import { useRecorder } from "./hooks/useRecorder";
import { createVoiceSocket } from "./services/voiceSocket";
import "./index.css";

const FALLBACK = {
  default: "friendly",
  personas: [
    { key: "friendly", label: "Friendly", tts_preset: "friendly_voice" },
    { key: "neutral", label: "Neutral", tts_preset: "neutral_voice" },
    { key: "teacher", label: "Teacher", tts_preset: "teacher_voice" },
  ],
  tts_presets: ["friendly_voice", "neutral_voice", "teacher_voice"],
};

export default function App() {
  const [status, setStatus] = useState("disconnected");
  const [partial, setPartial] = useState("");
  const [assistantStream, setAssistantStream] = useState("");
  const [messages, setMessages] = useState([]);
  const [socketReady, setSocketReady] = useState(false);

  const [options, setOptions] = useState({ personas: [], tts_presets: [], default: "friendly" });
  const [persona, setPersona] = useState("");
  const [ttsPreset, setTtsPreset] = useState("");

  const [history, setHistory] = useState([]);   // ✅ history state

  const socketRef = useRef(null);
  const assistantRef = useRef("");
  const ttsStartedRef = useRef(false);

  // ---------- history fetch ----------
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch("http://localhost:8001/api/history?limit=20");
        if (!r.ok) throw new Error("bad status");
        const data = await r.json();
        setHistory(data.history || []);
      } catch (e) {
        console.error("history fetch failed", e);
      }
    })();
  }, []);

  const speakFallback = (text) => {
    if (!text) return;
    try {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = "en-US";
      window.speechSynthesis.speak(u);
    } catch {}
  };

  const handleEvent = useCallback((msg) => {
    switch (msg.type) {
      case "socket_open":
        setStatus("ready");
        setSocketReady(true);
        break;
      case "socket_close":
        setStatus("disconnected");
        setSocketReady(false);
        break;
      case "ack":
      case "ready":
        setStatus("ready");
        break;
      case "started":
        ttsStartedRef.current = false;
        setStatus("streaming");
        setAssistantStream("");
        assistantRef.current = "";
        break;
      case "tts_start":
        ttsStartedRef.current = true;
        break;
      case "config-ack":
        setStatus(`Config → persona: ${msg.persona}, voice: ${msg.tts_preset}`);
        break;
      case "stt_partial":
        setPartial(msg.text || "");
        break;
      case "transcript":
        if (msg.final) {
          setMessages((m) => [...m, { role: "user", text: msg.text || "" }]);
          setPartial("");
        }
        break;
      case "delta":
        setAssistantStream((s) => {
          const next = s + (msg.text || "");
          assistantRef.current = next;
          return next;
        });
        break;
      case "assistant_final":
      case "done": {
        const finalText = msg.text || assistantRef.current || "";
        if (finalText) {
          setMessages((m) => [...m, { role: "assistant", text: finalText }]);
          if (!ttsStartedRef.current) speakFallback(finalText);
        }
        setAssistantStream("");
        assistantRef.current = "";
        setStatus("ready");
        break;
      }
      case "error":
        setStatus(`error: ${msg.error || msg.message || ""}`);
        break;
    }
  }, []);

  // ---------- open WS ----------
  useEffect(() => {
    if (socketRef.current) return;
    const url = import.meta.env.VITE_WS_URL || "ws://localhost:8001/api/ws/voice";
    socketRef.current = createVoiceSocket({ url, onEvent: handleEvent });
    return () => {
      socketRef.current?.stop();
      socketRef.current = null;
    };
  }, [handleEvent]);

  // ---------- load personas ----------
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch("http://localhost:8001/api/personas");
        if (!r.ok) throw new Error("bad status");
        const data = await r.json();
        const good = Array.isArray(data?.personas) && data.personas.length > 0;
        const out = good ? data : FALLBACK;
        setOptions(out);
        const def = out.default || "friendly";
        const defTts = (out.personas.find((p) => p.key === def)?.tts_preset) || out.tts_presets[0];
        setPersona(def);
        setTtsPreset(defTts);
        const trySend = () => socketRef.current?.sendConfig?.({ persona: def, ttsPreset: defTts });
        if (socketRef.current?.ready) trySend(); else setTimeout(trySend, 400);
      } catch {
        setOptions(FALLBACK);
        setPersona(FALLBACK.default);
        setTtsPreset("friendly_voice");
        const trySend = () => socketRef.current?.sendConfig?.({ persona: FALLBACK.default, ttsPreset: "friendly_voice" });
        if (socketRef.current?.ready) trySend(); else setTimeout(trySend, 400);
      }
    })();
  }, []);

  // ---------- send config when persona/preset changes ----------
  useEffect(() => {
    if (!persona || !socketRef.current?.ready) return;
    socketRef.current.sendConfig?.({ persona, ttsPreset });
  }, [persona, ttsPreset]);

  const { start, stop, recording, error } = useRecorder({
    onPcmChunk: (buf) => socketRef.current?.sendAudio(buf),
  });

  const toggleMic = async () => {
    if (!socketReady) return;
    if (recording) {
      stop();
      socketRef.current?.sendDone();
    } else {
      socketRef.current?.sendStart(16000);
      await start();
    }
  };

  return (
    <div className="page">
      <div className="container">
        <header className="header">
          <h1>Voice Mode (STT → LLM)</h1>
          <span className="status">Status: {status}</span>
        </header>

        {/* Persona + TTS preset controls */}
        <div className="card" style={{ display: "flex", gap: 12, alignItems: "end", marginBottom: 12 }}>
          <label>
            <div>Persona</div>
            <select value={persona} disabled={!socketReady} onChange={(e) => setPersona(e.target.value)}>
              {options.personas.map((p) => (
                <option key={p.key} value={p.key}>{p.label}</option>
              ))}
            </select>
          </label>

          <label>
            <div>TTS preset</div>
            <select value={ttsPreset} disabled={!socketReady} onChange={(e) => setTtsPreset(e.target.value)}>
              {options.tts_presets.map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
          </label>
        </div>

        <div className="row">
          <MicButton recording={recording} onClick={toggleMic} />
          {/* ✅ Bubble mic indicator */}
          <VoiceBubble active={recording} speaking={status === "streaming"} />
          {!socketReady && <span className="text-warn">Connecting to backend...</span>}
          {error && <span className="text-error">Mic error: {error}</span>}
        </div>

        <StreamBox text={assistantStream} />
        <TranscriptBox partial={partial} messages={messages} />

        {/* ✅ history panel at bottom */}
        <HistoryPanel history={history} />
      </div>
    </div>
  );
}
