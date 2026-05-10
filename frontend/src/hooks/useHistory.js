import { useEffect, useState } from "react";

export function useHistory(limit = 20, sessionId = null) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchHistory() {
      try {
        setLoading(true);
        let url = `http://localhost:8001/api/history?limit=${limit}`;
        if (sessionId) {
          url += `&session_id=${sessionId}`;
        }

        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // ✅ Correct: only take data.history
        setHistory(data.history || []);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchHistory();
  }, [limit, sessionId]);

  return { history, loading, error };
}
