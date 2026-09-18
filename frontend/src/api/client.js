import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api";

const client = axios.create({ baseURL: API_BASE_URL });

// Attach the access token to every request, if we have one.
client.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// On a 401, try refreshing the access token once, then retry the request.
client.interceptors.response.use(
  (response) => response,
  async (error) => {
    const original = error.config;
    if (error.response?.status === 401 && !original._retry) {
      original._retry = true;
      const refresh = localStorage.getItem("refresh_token");
      if (refresh) {
        try {
          const { data } = await axios.post(`${API_BASE_URL}/auth/refresh/`, { refresh });
          localStorage.setItem("access_token", data.access);
          original.headers.Authorization = `Bearer ${data.access}`;
          return client(original);
        } catch (refreshError) {
          localStorage.removeItem("access_token");
          localStorage.removeItem("refresh_token");
          localStorage.removeItem("username");
          window.location.href = "/login";
        }
      }
    }
    return Promise.reject(error);
  }
);

export default client;

export const authApi = {
  signup: (payload) => client.post("/auth/signup/", payload),
  login: (payload) => client.post("/auth/login/", payload),
  me: () => client.get("/auth/me/"),
};

export const diagnosisApi = {
  predict: (formData) =>
    client.post("/predict/", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    }),
  history: () => client.get("/history/"),
};
