import { useState, useRef } from "react";
import { diagnosisApi } from "../api/client";

export default function Upload() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const selectFile = (selected) => {
    if (!selected) return;
    if (!selected.type.startsWith("image/")) {
      setError("Please select an image file.");
      return;
    }
    setError("");
    setResult(null);
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
  };

  const handleFileChange = (e) => selectFile(e.target.files?.[0]);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    selectFile(e.dataTransfer.files?.[0]);
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("image", file);
      const { data } = await diagnosisApi.predict(formData);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.error || "Prediction failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="page page-upload">
      <h1>Upload a retinal image</h1>
      <p className="page-subtitle">
        Select or drag in a fundus photo, then click Predict to run the model.
      </p>

      <div
        className={`dropzone ${dragActive ? "dropzone-active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        {previewUrl ? (
          <img src={previewUrl} alt="Selected retinal scan preview" className="preview-image" />
        ) : (
          <div className="dropzone-placeholder">
            <div className="dropzone-icon">🖼️</div>
            <p>Click to browse or drag an image here</p>
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          hidden
        />
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      <div className="upload-actions">
        <button
          className="btn btn-primary btn-lg"
          onClick={handlePredict}
          disabled={!file || loading}
        >
          {loading ? "Analyzing…" : "Predict"}
        </button>
        {file && (
          <button className="btn btn-secondary" onClick={reset} disabled={loading}>
            Clear
          </button>
        )}
      </div>

      {result && (
        <div className={`result-card result-${result.predicted_class === "DR" ? "positive" : "negative"}`}>
          <h2>
            Prediction: <span>{result.predicted_class}</span>
          </h2>
          <p className="result-confidence-label">
            Confidence: {(result.confidence * 100).toFixed(1)}%
          </p>

          <div className="prob-bars">
            <div className="prob-row">
              <span className="prob-label">DR</span>
              <div className="prob-track">
                <div
                  className="prob-fill prob-fill-dr"
                  style={{ width: `${result.dr_probability * 100}%` }}
                />
              </div>
              <span className="prob-value">{(result.dr_probability * 100).toFixed(1)}%</span>
            </div>
            <div className="prob-row">
              <span className="prob-label">No DR</span>
              <div className="prob-track">
                <div
                  className="prob-fill prob-fill-nodr"
                  style={{ width: `${result.no_dr_probability * 100}%` }}
                />
              </div>
              <span className="prob-value">{(result.no_dr_probability * 100).toFixed(1)}%</span>
            </div>
          </div>

          <p className="result-note">
            This is an automated screening result, not a clinical diagnosis. Please consult a
            doctor for confirmation.
          </p>
        </div>
      )}
    </div>
  );
}
