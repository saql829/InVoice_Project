import React from 'react';

export default function MicButton({ isRecording, onStart, onStop }) {
  return (
    <button
      className={`mic-btn ${isRecording ? 'recording' : ''}`}
      onClick={isRecording ? onStop : onStart}
    >
      {isRecording ? '⏹ Stop' : '🎙 Start'}
    </button>
  );
}
