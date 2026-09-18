import { createContext, useContext, useState, useCallback } from "react";
import { authApi } from "../api/client";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [username, setUsername] = useState(() => localStorage.getItem("username"));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const isAuthenticated = Boolean(username && localStorage.getItem("access_token"));

  const persistSession = (data) => {
    localStorage.setItem("access_token", data.access);
    localStorage.setItem("refresh_token", data.refresh);
    localStorage.setItem("username", data.user.username);
    setUsername(data.user.username);
  };

  const login = useCallback(async (creds) => {
    setLoading(true);
    setError("");
    try {
      const { data } = await authApi.login(creds);
      persistSession(data);
      return true;
    } catch (err) {
      setError(err.response?.data?.detail || "Invalid username or password.");
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const signup = useCallback(async (payload) => {
    setLoading(true);
    setError("");
    try {
      const { data } = await authApi.signup(payload);
      persistSession(data);
      return true;
    } catch (err) {
      const data = err.response?.data;
      const message = data
        ? Object.entries(data)
            .map(([field, msgs]) => `${field}: ${Array.isArray(msgs) ? msgs.join(" ") : msgs}`)
            .join(" ")
        : "Signup failed. Please try again.";
      setError(message);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    localStorage.removeItem("username");
    setUsername(null);
  }, []);

  return (
    <AuthContext.Provider value={{ username, isAuthenticated, loading, error, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
