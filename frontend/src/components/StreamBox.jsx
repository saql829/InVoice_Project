export default function StreamBox({ text }) {
    if (!text) return null;
    return (
      <div className="card card-info">
        <strong>Assistant (streaming):</strong> {text}
      </div>
    );
  }
  