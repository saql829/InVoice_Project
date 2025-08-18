import { useEffect, useRef } from "react";
import { createVoiceSocket } from "../services/voiceSocket";
import { useRecorder } from "../hooks/useRecorder";

export default function VoiceChat() {
  const socketRef = useRef(null);

  useEffect(() => {
    socketRef.current = createVoiceSocket({
      url: "ws://localhost:8000/ws/voice",
      onEvent: (evt) => {
        console.log("Socket event:", evt);
      },
    });
  }, []);

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
    <div>
      <button onClick={handleStart} disabled={recording}>🎙 Start</button>
      <button onClick={handleStop} disabled={!recording}>⏹ Stop</button>
    </div>
  );
}
