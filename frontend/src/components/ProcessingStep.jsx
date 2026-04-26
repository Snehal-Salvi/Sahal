export default function ProcessingStep({ icon, iconClass, name, desc, state }) {
  return (
    <div className={`srow${state === "active" ? " act" : ""}`}>
      <div className={`sico ${iconClass}`}>{icon}</div>
      <div className="sbody">
        <div className="sn">{name}</div>
        <div className="sd">{desc}</div>
      </div>
      {state === "done"   && <div className="ss-done">✓</div>}
      {state === "active" && <div className="ss-spin" />}
      {state === "wait"   && <div className="ss-wait">Pending</div>}
    </div>
  );
}
