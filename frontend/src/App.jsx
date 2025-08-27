import React, { useState, useRef } from 'react';
import MicButton from './components/MicButton';
import Transcript from './components/Transcript';
import SpeakingIndicator from './components/SpeakingIndicator';
import AudioPlayer from './components/AudioPlayer';
import useWebSocket from './hooks/useWebSocket';
import './styles/app.css';

export default function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [audioQueue, setAudioQueue] = useState([]);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);

  //  WebSocket hooks
  const { sendAudio } = useWebSocket({
    onTranscript: (text) =>
      setTranscript((prev) =>
        prev.trim() ? `${prev} ${text}` : text
      ),
    onSpeaking: (value) => setIsSpeaking(value),
    onAudio: (base64) =>
      setAudioQueue((q) => [...q, base64]),
  });

  //  Start mic recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          sendAudio(e.data);
        }
      };

      mediaRecorder.start(250); // send every 250ms
      setIsRecording(true);
    } catch (err) {
      console.error('❌ Mic access denied:', err);
      alert('Please allow microphone access.');
    }
  };

  //  Stop mic recording
  const stopRecording = () => {
    try {
      mediaRecorderRef.current?.stop();
      streamRef.current?.getTracks().forEach((track) => track.stop());
    } catch (err) {
      console.error('❌ Error stopping recording:', err);
    }
    setIsRecording(false);
  };

  return (
    <div className="app-container">
      <h1>🎤 Voice Assistant</h1>

      <MicButton
        isRecording={isRecording}
        onStart={startRecording}
        onStop={stopRecording}
      />

      <SpeakingIndicator isSpeaking={isSpeaking} />
      <Transcript text={transcript} />

      {/*  Audio queue consume karega */}
      <AudioPlayer audioQueue={audioQueue} setAudioQueue={setAudioQueue} />
    </div>
  );
}
