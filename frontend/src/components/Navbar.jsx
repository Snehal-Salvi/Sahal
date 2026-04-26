const STEP_LABELS = ["Upload", "Detect", "Filters", "Processing", "Download"];

export default function Navbar({ step, goTo }) {
  return (
    <nav className="topnav">
      <div className="logo">
        <div className="logo-dot">S</div>
        Sahal
      </div>
      <div className="steps-nav">
        {STEP_LABELS.flatMap((label, i) => {
          const n = i + 1;
          const isDone = step > n;
          const isOn = step === n;
          const items = [];
          if (i > 0) items.push(<div key={`line-${n}`} className="step-line" />);
          items.push(
            <button
              key={n}
              className={`step${isOn ? " on" : ""}${isDone ? " done" : ""}`}
              onClick={() => { if (n <= step) goTo(n); }}
            >
              <div className="step-n">{isDone ? "✓" : n}</div>
              <span className="step-l">{label}</span>
            </button>
          );
          return items;
        })}
      </div>
      <div className="nav-av" />
    </nav>
  );
}
