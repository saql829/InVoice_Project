import React from "react";
//import "./VoiceBubble.css";

export default function VoiceBubble({ active, speaking }) {
  return (
    <div className={`voice-bubble ${active ? "active" : ""} ${speaking ? "speaking" : ""}`}>
      <div className="inner" />
    </div>
  );
}
