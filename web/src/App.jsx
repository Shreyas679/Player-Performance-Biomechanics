import React, { useState } from "react";
import Upload from "./Upload";
import Results from "./Results";
import Live from "./Live";
import "./styles.css";

export default function App() {
  const [result, setResult] = useState(null);
  return (
    <div className="container">
      <h2>Player Performance Biomechanics Predictor</h2>
      <div className="card">
        <p>Upload a keypoints JSON or a cricket video to get a prediction.</p>
      </div>
      <Upload onResult={setResult} />
      <Results result={result} />
      <Live />
    </div>
  );
}
