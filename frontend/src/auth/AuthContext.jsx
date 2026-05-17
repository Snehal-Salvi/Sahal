import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";

const AuthContext = createContext(null);

let currentToken = null;

export function getAuthToken() {
  return currentToken;
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setTokenState] = useState(null);
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const pendingActionRef = useRef(null);

  const setSession = useCallback((nextToken, nextUser) => {
    currentToken = nextToken;
    setTokenState(nextToken);
    setUser(nextUser);
    setIsLoginOpen(false);
    const action = pendingActionRef.current;
    pendingActionRef.current = null;
    if (typeof action === "function") {
      Promise.resolve().then(action);
    }
  }, []);

  const clearSession = useCallback(() => {
    currentToken = null;
    setTokenState(null);
    setUser(null);
  }, []);

  const signOut = clearSession;

  const requireLogin = useCallback(
    (action) => {
      if (currentToken) {
        action?.();
        return;
      }
      pendingActionRef.current = typeof action === "function" ? action : null;
      setIsLoginOpen(true);
    },
    []
  );

  const closeLogin = useCallback(() => {
    pendingActionRef.current = null;
    setIsLoginOpen(false);
  }, []);

  const value = useMemo(
    () => ({
      user,
      token,
      setSession,
      clearSession,
      signOut,
      requireLogin,
      isLoginOpen,
      closeLogin
    }),
    [user, token, setSession, clearSession, signOut, requireLogin, isLoginOpen, closeLogin]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}
