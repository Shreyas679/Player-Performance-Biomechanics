import React, { useEffect, useRef, useState } from "react";

export default function Live() {
  const [status, setStatus] = useState("disconnected");
  const [pred, setPred] = useState(null);
  const wsRef = useRef(null);

  const connect = () => {
    if (wsRef.current) return; // already connected
    const ws = new WebSocket("ws://localhost:8000/ws/realtime");
    ws.onopen = () => setStatus("connected");
    ws.onclose = () => { setStatus("disconnected"); wsRef.current = null; };
    ws.onmessage = (e) => {
      try { setPred(JSON.parse(e.data)); } catch {}
    };
    wsRef.current = ws;
  };

  const disconnect = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setPred(null);
    setStatus("disconnected");
  };

  useEffect(() => {
    // optional: auto-connect on mount
    connect();
    return () => disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const badge = (s) => {
    const pct = Math.round((s || 0) * 100);
    const color = pct > 66 ? "#e74c3c" : pct > 33 ? "#f39c12" : "#2ecc71";
    return <span style={{
      background: color, color: "white", padding: "2px 8px",
      borderRadius: 12, marginLeft: 8, fontSize: 12
    }}>{pct}%</span>;
  };

  return (
    <div className="card">
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <h3 style={{ margin: 0 }}>Live Predictions ({status})</h3>
        {status === "connected" ? (
          <button onClick={disconnect}>Stop</button>
        ) : (
          <button onClick={connect}>Start</button>
        )}
      </div>

      {typeof pred?.miss_ratio === "number" && (
        <p>Pose success: {Math.max(0, 100 - Math.round(pred.miss_ratio * 100))}%</p>
      )}
      {typeof pred?.performance_score === "number" && (
        <p>Performance score: {pred.performance_score.toFixed(1)}</p>
      )}
      {typeof pred?.injury_risk === "number" && (
        <p>Injury risk: {pred.injury_risk ? "High" : "Low"}</p>
      )}
      {pred?.hotspots?.length > 0 && (
        <>
          <h4>Risk hotspots</h4>
          <ul>
            {pred.hotspots.map((h, i) => (
              <li key={i}>
                {h.region}{badge(h.score)}
              </li>
            ))}
          </ul>
        </>
      )}
      {pred?.frame && (
        <img
          alt="overlay"
          style={{ width: "100%", maxWidth: 520, border: "1px solid #ddd", borderRadius: 8, marginTop: 8 }}
          src={`data:image/jpeg;base64,${pred.frame}`}
        />
      )}
      {!pred && status === "connected" && <p>Waiting for stream...</p>}
      {status !== "connected" && <p>Live capture is stopped.</p>}
    </div>
  );
}
