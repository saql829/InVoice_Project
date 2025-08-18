// src/App.jsx
import { useEffect, useRef, useState, useCallback } from "react";
import MicButton from "./components/MicButton";
import TranscriptBox from "./components/TranscriptBox";
import StreamBox from "./components/StreamBox";
import { useRecorder } from "./hooks/useRecorder";
import { createVoiceSocket } from "./services/voiceSocket";
import "./index.css";

export default function App() {
  const [status, setStatus] = useState("disconnected");
  const [partial, setPartial] = useState("");
  const [assistantStream, setAssistantStream] = useState("");
  const [messages, setMessages] = useState([]);
  const [socketReady, setSocketReady] = useState(false);

  const socketRef = useRef(null);
  const assistantRef = useRef("");

  // TTS helper function
  const speakText = (text) => {
    if (!text) return;
    // cancel previous speech if any
    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US"; // change language if needed
    speechSynthesis.speak(utterance);
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
        setStatus("streaming");
        setAssistantStream("");
        assistantRef.current = "";
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
      case "done":
        const finalText = msg.text || assistantRef.current || "";
        if (finalText) {
          setMessages((m) => [...m, { role: "assistant", text: finalText }]);
          speakText(finalText); // TTS here
        }
        setAssistantStream("");
        assistantRef.current = "";
        setStatus("ready");
        break;
      case "error":
        setStatus(`error: ${msg.error || msg.message || ""}`);
        break;
    }
  }, []);

  useEffect(() => {
    if (socketRef.current) return;
    const url = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/voice";
    socketRef.current = createVoiceSocket({ url, onEvent: handleEvent });
    return () => {
      socketRef.current?.stop();
      socketRef.current = null;
    };
  }, [handleEvent]);

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

        <div className="row">
          <MicButton recording={recording} onClick={toggleMic} />
          {!socketReady && <span className="text-warn">Connecting to backend...</span>}
          {error && <span className="text-error">Mic error: {error}</span>}
        </div>

        <StreamBox text={assistantStream} />
        <TranscriptBox partial={partial} messages={messages} />
      </div>
    </div>
  );
}
