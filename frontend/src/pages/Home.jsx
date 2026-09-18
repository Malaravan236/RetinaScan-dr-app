import { Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function Home() {
  const { isAuthenticated, username } = useAuth();

  return (
    <div className="page page-home">
      <section className="hero">
        <h1>Diabetic Retinopathy Screening</h1>
        <p className="hero-subtitle">
          Upload a retinal fundus image and get an instant, AI-assisted screening result
          powered by a MobileNetV2 model trained with Cuckoo &amp; Pelican search-optimized
          hyperparameters.
        </p>

        {isAuthenticated ? (
          <div className="hero-actions">
            <p className="hero-welcome">Welcome back, {username} 👋</p>
            <Link to="/upload" className="btn btn-primary btn-lg">
              Upload an image
            </Link>
            <Link to="/history" className="btn btn-secondary btn-lg">
              View history
            </Link>
          </div>
        ) : (
          <div className="hero-actions">
            <Link to="/signup" className="btn btn-primary btn-lg">
              Get started
            </Link>
            <Link to="/login" className="btn btn-secondary btn-lg">
              Log in
            </Link>
          </div>
        )}
      </section>

      <section className="feature-grid">
        <div className="feature-card">
          <div className="feature-icon">📤</div>
          <h3>Upload</h3>
          <p>Upload a retinal fundus photo directly from your device.</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">🧠</div>
          <h3>Predict</h3>
          <p>Our trained MobileNetV2 model classifies the image in seconds.</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">📊</div>
          <h3>Confidence</h3>
          <p>See the model's confidence for DR and No DR classes.</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">🕘</div>
          <h3>History</h3>
          <p>Every prediction is saved so you can track results over time.</p>
        </div>
      </section>

      <p className="disclaimer">
        ⚠️ This tool is for educational/screening-assistance purposes only and is not a
        substitute for a professional medical diagnosis. Always consult an ophthalmologist.
      </p>
    </div>
  );
}
