export default function MicButton({ recording, onClick }) {
    return (
      <button onClick={onClick} className={`btn ${recording ? "btn-stop" : "btn-start"}`}>
        {recording ? "Stop Mic" : "Start Mic"}
      </button>
    );
  }
  