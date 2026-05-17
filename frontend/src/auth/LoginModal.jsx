import { GoogleLogin } from "@react-oauth/google";
import axios from "axios";
import { useAuth } from "./AuthContext";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:5001/api";

export function LoginModal() {
  const { isLoginOpen, closeLogin, setSession } = useAuth();

  if (!isLoginOpen) return null;

  const handleSuccess = async (credentialResponse) => {
    try {
      const { data } = await axios.post(`${API_BASE_URL}/auth/google`, {
        credential: credentialResponse.credential
      });
      setSession(data.token, data.user);
    } catch (err) {
      console.error("Google sign-in failed", err);
    }
  };

  return (
    <div
      onClick={closeLogin}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.55)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "#111",
          color: "#fff",
          borderRadius: 16,
          padding: "2rem",
          minWidth: 320,
          maxWidth: 400,
          boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "1.25rem"
        }}
      >
        <button
          onClick={closeLogin}
          aria-label="Close"
          style={{
            position: "absolute",
            top: 12,
            right: 16,
            background: "transparent",
            border: "none",
            color: "#aaa",
            fontSize: 22,
            cursor: "pointer"
          }}
        >
          ×
        </button>
        <h2 style={{ margin: 0, fontSize: "1.25rem" }}>Sign in to continue</h2>
        <p style={{ margin: 0, color: "#aaa", fontSize: "0.9rem", textAlign: "center" }}>
          You need a Google account to upload and process videos.
        </p>
        <GoogleLogin
          onSuccess={handleSuccess}
          onError={() => console.error("Google sign-in failed")}
          theme="filled_black"
        />
      </div>
    </div>
  );
}
