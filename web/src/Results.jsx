// web/src/Results.jsx
import React from "react";

export default function Results({ result }) {
  if (!result) return (
    <div className="card"><p>No prediction yet.</p></div>
  );

  return (
    <div className="card">
      <h3>Prediction</h3>
      <p>Performance score: {result.performance_score.toFixed(1)}</p>
      <p>Injury risk: {result.injury_risk ? "High" : "Low"}</p>

      {result.top_features?.length > 0 && (
        <>
          <h4>Top features</h4>
          <ul>
            {result.top_features.map((f, i) => {
              const [k, v] = Object.entries(f)[0];
              return <li key={i}>{k}: {v.toFixed(3)}</li>;
            })}
          </ul>
        </>
      )}

      {result.hotspots?.length > 0 && (
        <>
          <h4>Risk hotspots</h4>
          <ul>
            {result.hotspots.map((h, i) => (
              <li key={i}>
                {h.region}: {Math.round(h.score * 100)}%
              </li>
            ))}
          </ul>
        </>
      )}

      {result.preview && (
        <>
          <h4>Overlay preview</h4>
          <img
            alt="preview"
            style={{ width: "100%", maxWidth: 520, border: "1px solid #ddd", borderRadius: 8, marginTop: 8 }}
            src={`data:image/jpeg;base64,${result.preview}`}
          />
        </>
      )}
    </div>
  );
}
