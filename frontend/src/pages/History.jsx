import { useEffect, useState } from "react";
import { diagnosisApi } from "../api/client";

export default function History() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    diagnosisApi
      .history()
      .then(({ data }) => {
        if (!cancelled) setRecords(data);
      })
      .catch(() => {
        if (!cancelled) setError("Couldn't load your prediction history.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="page page-history">
      <h1>Prediction history</h1>
      <p className="page-subtitle">Every image you've analyzed, most recent first.</p>

      {loading && <p>Loading…</p>}
      {error && <div className="alert alert-error">{error}</div>}

      {!loading && !error && records.length === 0 && (
        <p className="empty-state">No predictions yet. Upload an image to get started.</p>
      )}

      <div className="history-grid">
        {records.map((r) => (
          <div key={r.id} className={`history-card history-${r.predicted_class === "DR" ? "positive" : "negative"}`}>
            <img src={r.image} alt={`Scan predicted ${r.predicted_class}`} className="history-thumb" />
            <div className="history-body">
              <div className="history-top-row">
                <span className="history-label">{r.predicted_class}</span>
                <span className="history-confidence">{(r.confidence * 100).toFixed(1)}%</span>
              </div>
              <p className="history-date">{new Date(r.created_at).toLocaleString()}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
