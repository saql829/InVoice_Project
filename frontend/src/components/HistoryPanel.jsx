import React from "react";

export default function HistoryPanel({ history }) {
  if (!history || history.length === 0) return <div>No history yet</div>;

  return (
    <div className="history-panel">
      <h3>Conversation History</h3>
      <ul>
        {history.map((msg, idx) => (
          <li key={idx}>
            <strong>{msg.role}:</strong> {msg.text}
          </li>
        ))}
      </ul>
    </div>
  );
}
