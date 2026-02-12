import React, { useState } from "react";
import { uploadKeypoints } from "./api";

export default function Upload({ onResult }) {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    if (!file) return;
    setBusy(true);
    try {
      const data = await uploadKeypoints(file);
      onResult(data);
    } catch (e) {
      alert(e?.response?.data?.detail || "Upload failed");
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="card">
      <h3>Upload Cricket Pose JSON or Video</h3>
      <input
        type="file"
        accept=".json,video/mp4,video/quicktime,video/x-matroska,video/x-msvideo"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button disabled={busy} onClick={submit}>{busy ? "Predicting..." : "Predict"}</button>
    </div>
  );
}
