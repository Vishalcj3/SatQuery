import React from "react";

export default function ModeSelector({ modes, activeId, onChange }) {
  return (
    <div className="hud modes">
      {modes.map((m) => (
        <button
          key={m.id}
          className={"mode" + (m.id === activeId ? " act" : "")}
          style={m.id === activeId ? { borderColor: m.accent, color: m.accent } : undefined}
          onClick={() => onChange(m.id)}
          title={m.label}
        >
          <span className="mi">{m.icon}</span>
          <span className="ml">{m.label}</span>
        </button>
      ))}
    </div>
  );
}
