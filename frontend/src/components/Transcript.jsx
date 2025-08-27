import React from 'react';

export default function Transcript({ text }) {
  return (
    <div className="transcript">
      <h2>Live Transcript</h2>
      <p>{text || 'Say something...'}</p>
    </div>
  );
}
