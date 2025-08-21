import { useEffect, useState } from "react";
import { fetchPersonas } from "../services/voiceSocket";

export default function VoiceControls({ onConfig, disabled }) {
  const [opts, setOpts] = useState({ personas: [], tts_presets: [], default: "neutral" });
  const [persona, setPersona] = useState("");
  const [ttsPreset, setTtsPreset] = useState("");
  const [rate, setRate] = useState(1.0);
  const [pitch, setPitch] = useState(0);

  useEffect(() => {
    fetchPersonas()
      .then((data) => {
        setOpts(data);
        const def = data.default || "neutral";
        setPersona(def);
        const defTts = (data.personas.find(p => p.key === def)?.tts_preset) || data.tts_presets?.[0] || "";
        setTtsPreset(defTts);
        onConfig?.({ persona: def, ttsPreset: defTts, rate, pitch });
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (!persona) return;
    onConfig?.({ persona, ttsPreset, rate, pitch });
  }, [persona, ttsPreset, rate, pitch]);

  return (
    <div className="card" style={{ display: "flex", gap: 12, alignItems: "end", flexWrap: "wrap", marginBottom: 12 }}>
      <label>
        <div>Persona</div>
        <select value={persona} disabled={disabled} onChange={e => setPersona(e.target.value)}>
          {opts.personas.map(p => <option key={p.key} value={p.key}>{p.label}</option>)}
        </select>
      </label>

      <label>
        <div>TTS preset</div>
        <select value={ttsPreset} disabled={disabled} onChange={e => setTtsPreset(e.target.value)}>
          {opts.tts_presets.map(k => <option key={k} value={k}>{k}</option>)}
        </select>
      </label>

      <label style={{ minWidth: 180 }}>
        <div>Rate: {rate.toFixed(2)}</div>
        <input type="range" min="0.85" max="1.2" step="0.01" value={rate} disabled={disabled} onChange={e => setRate(parseFloat(e.target.value))}/>
      </label>

      <label style={{ minWidth: 180 }}>
        <div>Pitch: {pitch}</div>
        <input type="range" min="-3" max="3" step="1" value={pitch} disabled={disabled} onChange={e => setPitch(parseInt(e.target.value))}/>
      </label>
    </div>
  );
}
