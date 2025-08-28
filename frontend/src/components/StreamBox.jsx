export default function StreamBox({ text, voice }) {
  if (!text) return null;

  // Extra cleanup (safety net agar backend se kuch slip ho jaye)
  const fixedText = text
    .replace(/([.,!?])([A-Za-z])/g, "$1 $2")
    .replace(/([a-z])([A-Z])/g, "$1 $2");

  return (
    <div className="card card-info">
      <strong>Assistant ({voice || "default"}):</strong> {fixedText}
    </div>
  );
}
