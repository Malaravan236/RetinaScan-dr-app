# RetinaScan — Diabetic Retinopathy Detection

A full-stack app for screening retinal fundus images for diabetic retinopathy (DR):

- **`frontend/`** — React (Vite) SPA: login/signup, home, image upload, prediction result
  with confidence bars, prediction history, responsive UI.
- **`backend/`** — Django + Django REST Framework API: JWT auth, image upload handling,
  prediction endpoint, SQLite database, prediction history, CORS.
- **`backend/ml/`** — the AI layer: model loading + preprocessing (`inference.py`), the
  trained MobileNetV2 model (`models/my_model.keras`), and your original training
  scripts for the Cuckoo Search and Pelican Optimization hyperparameter tuning
  pipelines (`training/`).

The prediction pipeline has been tested end-to-end (signup → login → upload → predict →
history) against the actual trained model.

## Project layout

```
dr-app/
├── backend/
│   ├── manage.py
│   ├── requirements.txt
│   ├── config/            # Django project settings/urls
│   ├── accounts/          # signup / login / JWT auth
│   ├── diagnosis/         # prediction model, endpoints, history
│   └── ml/
│       ├── inference.py   # model loading + preprocessing + prediction
│       ├── models/        # trained my_model.keras (MobileNetV2)
│       └── training/      # your original Cuckoo/Pelican optimization scripts
└── frontend/
    ├── src/
    │   ├── pages/          # Home, Login, Signup, Upload, History
    │   ├── components/     # Navbar, ProtectedRoute
    │   ├── context/         # AuthContext (JWT storage + refresh)
    │   └── api/             # axios client
    └── ...
```

## 1. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python manage.py migrate
python manage.py createsuperuser   # optional, for /admin/
python manage.py runserver         # http://127.0.0.1:8000
```

The model is loaded lazily on the first prediction request (from
`backend/ml/models/my_model.keras`), so `runserver` itself starts instantly.

### API endpoints

| Method | Endpoint                | Auth | Description                                  |
|--------|--------------------------|------|-----------------------------------------------|
| POST   | `/api/auth/signup/`      | No   | Create a user, returns JWT access/refresh     |
| POST   | `/api/auth/login/`       | No   | Log in, returns JWT access/refresh            |
| POST   | `/api/auth/refresh/`     | No   | Exchange a refresh token for a new access one |
| GET    | `/api/auth/me/`          | Yes  | Current user profile                          |
| POST   | `/api/predict/`          | Yes  | Upload an image (`multipart/form-data`, field `image`) → prediction |
| GET    | `/api/history/`          | Yes  | List the current user's past predictions      |

## 2. Frontend setup

```bash
cd frontend
npm install
npm run dev    # http://127.0.0.1:5173
```

`frontend/.env` sets `VITE_API_BASE_URL` (defaults to `http://127.0.0.1:8000/api`) —
change it if your backend runs elsewhere.

## Notes

- **Auth**: JWT (via `djangorestframework-simplejwt`). Tokens are stored in
  `localStorage` on the frontend and auto-refreshed on 401 responses.
- **Model**: `backend/outputs/my_model.keras` and `.../models/my_model.keras` had the
  same name in the original project — the `models/` copy turned out to be an empty
  placeholder file, so this project uses the real trained model. If you retrain, drop
  the new `.keras` file at `backend/ml/models/my_model.keras`.
- **CORS**: pre-configured for `localhost:5173`/`3000` in `backend/config/settings.py`
  (`CORS_ALLOWED_ORIGINS`) — update this list for other ports or a deployed frontend
  origin.
- **Training scripts**: `backend/ml/training/` contains your original MobileNetV2 +
  Cuckoo Search / Pelican Optimization training pipelines, kept as-is for retraining;
  they aren't used at request-time (only the exported `.keras` file is).
- This is a screening aid, not a diagnostic tool — the UI includes a disclaimer to that
  effect; keep it in any deployment.
