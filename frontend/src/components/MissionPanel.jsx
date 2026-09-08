import React from "react";

export default function MissionPanel({ mode, selected, onSelect }) {
  return (
    <aside className="hud legend mission">
      <h2 style={{ color: mode.accent }}>
        <span>{mode.icon}&nbsp;{mode.label.toUpperCase()}</span>
        <span className="cnt">{mode.sites.length}</span>
      </h2>
      <div className="mq" style={{ borderColor: mode.accent }}>“{mode.query}”</div>
      <p className="mdesc">{mode.desc}</p>
      <div className="mlist">
        {mode.sites.map((s, i) => (
          <div
            key={i}
            className={"ms" + (selected === s ? " on" : "")}
            onClick={() => onSelect(s)}
          >
            <span className="msdot" style={{ background: s.color || mode.accent }} />
            <span className="msn">{s.n}</span>
            {s.change && <span className="msc">{s.change}</span>}
          </div>
        ))}
      </div>
      <div className="hint">Hover a marker → the detected change is outlined and explained.</div>
    </aside>
  );
}
