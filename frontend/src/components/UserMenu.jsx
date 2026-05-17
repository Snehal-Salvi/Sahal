import { useEffect, useRef, useState } from "react";
import { useAuth } from "../auth/AuthContext";

export default function UserMenu() {
  const { user, requireLogin, signOut } = useAuth();
  const [open, setOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    function onDocClick(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setOpen(false);
      }
    }
    if (open) document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  if (!user) {
    return (
      <button className="nav-signin" onClick={() => requireLogin()}>
        Sign in
      </button>
    );
  }

  const initial = (user.name || user.email || "?").trim().charAt(0).toUpperCase();
  const displayName = user.name || user.email;

  return (
    <div className="nav-user" ref={menuRef}>
      <button
        className="nav-user-trigger"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        {user.picture ? (
          <img
            className="nav-user-avatar"
            src={user.picture}
            alt=""
            referrerPolicy="no-referrer"
            onError={(e) => {
              e.currentTarget.style.display = "none";
              e.currentTarget.nextSibling.style.display = "flex";
            }}
          />
        ) : null}
        <div
          className="nav-user-avatar-fallback"
          style={{ display: user.picture ? "none" : "flex" }}
        >
          {initial}
        </div>
        <span className="nav-user-name">Welcome, {user.name?.split(" ")[0] || "you"}</span>
      </button>

      {open && (
        <div className="nav-user-menu" role="menu">
          <div className="nav-user-info">
            <div className="nav-user-info-name">{displayName}</div>
            {user.email && user.name ? (
              <div className="nav-user-info-email">{user.email}</div>
            ) : null}
          </div>
          <button
            className="nav-user-action"
            onClick={() => {
              setOpen(false);
              signOut();
            }}
          >
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
