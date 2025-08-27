import React from 'react';

export default function SpeakingIndicator({ isSpeaking }) {
  return (
    <div className={`speaking-indicator ${isSpeaking ? 'active' : ''}`}>
      {isSpeaking ? '🔴 Speaking...' : '⚪ Silent'}
    </div>
  );
}
