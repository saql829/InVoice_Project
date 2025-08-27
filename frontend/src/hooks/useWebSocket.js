// hooks/useWebSocket.js
import { useEffect, useRef } from 'react';

export default function useWebSocket({ onTranscript, onSpeaking, onAudio }) {
  const wsRef = useRef(null);

  useEffect(() => {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${wsProtocol}://localhost:8000/ws/audio`;  // ✅ FIXED

    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer"; // ✅ important for audio streaming
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('✅ WebSocket connected:', wsUrl);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
          case 'stt': // speech-to-text partial/final
            onTranscript?.(msg.text);
            break;
          case 'tts_chunk':   // ✅ backend sends audio pieces
            onAudio?.(msg.b64);
            onSpeaking?.(true);
            break;
          case 'tts_done':    // ✅ speech finished
            onSpeaking?.(false);
            break;
          default:
            console.warn('Unknown message type:', msg.type);
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };

    ws.onerror = (err) => {
      console.error('❌ WebSocket error:', err);
    };

    ws.onclose = () => {
      console.warn('⚠️ WebSocket closed');
    };

    return () => {
      ws.close();
    };
  }, []);

  const sendAudio = (blob) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      blob.arrayBuffer().then((buffer) => {
        wsRef.current.send(buffer);
      });
    } else {
      console.warn('WebSocket not open, cannot send audio');
    }
  };

  return { sendAudio };
}
