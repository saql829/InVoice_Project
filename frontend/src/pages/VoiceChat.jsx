import { useEffect, useRef, useState } from "react";
import { createVoiceSocket } from "../services/voiceSocket";
import { useRecorder } from "../hooks/useRecorder";

export default function VoiceChat() {
  const socketRef = useRef(null);
  const [partial, setPartial] = useState("");
  const [finals, setFinals] = useState([]);
  const [assistant, setAssistant] = useState([]);
  const [detectedLang, setDetectedLang] = useState(null);

  // 🔹 Personality & Voice state
  const [personality, setPersonality] = useState("friendly");
  const [voice, setVoice] = useState("alloy");

  useEffect(() => {
    socketRef.current = createVoiceSocket({
      url: "ws://localhost:8000/ws/voice",
      personality,
      voice,
      onEvent: (evt) => {
        console.log("Socket event:", evt);

        if (evt.type === "stt_partial") setPartial(evt.text);
        if (evt.type === "stt_final") {
          setFinals((f) => [...f, evt.text]);
          setPartial("");
        }
        if (evt.type === "delta") {
          setAssistant((a) => [...a.slice(0, -1), (a[a.length - 1] || "") + evt.text]);
        }
        if (evt.type === "assistant_final") {
          setAssistant((a) => [...a, evt.text]);
          setDetectedLang(evt.language || null);
        }
      },
    });

    return () => {
      socketRef.current?.stop();
    };
  }, [personality, voice]);

  const { start, stop, recording } = useRecorder({
    onPcmChunk: (chunk) => {
      if (socketRef.current?.ready) {
        socketRef.current.sendAudio(chunk);
      }
    }
  });

  function handleStart() {
    if (socketRef.current?.ready) {
      socketRef.current.sendStart(16000);
      start();
    }
  }

  function handleStop() {
    stop();
    socketRef.current?.sendDone();
  }

  return (
    <div style={{ padding: 20, fontFamily: "sans-serif" }}>
      <h2>🎙 Voice Chat</h2>

      {/* 🔹 Personality Selector */}
      <label>
        Personality:
        <select value={personality} onChange={(e) => setPersonality(e.target.value)}>
          <option value="friendly">Friendly</option>
          <option value="professional">Professional</option>
          <option value="funny">Funny</option>
          <option value="serious">Serious</option>
        </select>
      </label>

      {/* 🔹 Voice Selector */}
      <label style={{ marginLeft: 10 }}>
        Voice:
        <select value={voice} onChange={(e) => setVoice(e.target.value)}>
          <option value="alloy">Alloy</option>
          <option value="verse">Verse</option>
          <option value="echo">Echo</option>
          <option value="nova">Nova</option>
        </select>
      </label>

      <div style={{ marginTop: 10 }}>
        <button onClick={handleStart} disabled={recording}>🎙 Start</button>
        <button onClick={handleStop} disabled={!recording}>⏹ Stop</button>
      </div>

      <div style={{ marginTop: 20 }}>
        <h3>🗣 You:</h3>
        {finals.map((t, i) => <div key={i}>{t}</div>)}
        {partial && <div style={{ opacity: 0.6 }}>{partial}</div>}
      </div>

      <div style={{ marginTop: 20 }}>
        <h3>🤖 Assistant:</h3>
        {assistant.map((line, i) => (
          <div key={i} style={{ marginBottom: "8px", whiteSpace: "pre-wrap" }}>
            {line}
          </div>
        ))}
      </div>

      {detectedLang && (
        <div style={{ marginTop: 10, fontSize: "0.9em", opacity: 0.7 }}>
          🌐 Detected Language: <b>{detectedLang}</b>
        </div>
      )}
    </div>
  );
}