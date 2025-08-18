export default function TranscriptBox({ partial, messages }) {
    return (
      <div className="stack">
        {partial && (
          <div className="card card-warn">
            <strong>User (live):</strong> {partial}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`card ${m.role === "user" ? "card-user" : "card-assistant"}`}>
            <strong>{m.role === "user" ? "You" : "Assistant"}:</strong> {m.text}
          </div>
        ))}
      </div>
    );
  }
  