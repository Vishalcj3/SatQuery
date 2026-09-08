import React from "react";

export default function Legend({ types, counts, total, hidden, onToggle }) {
  const keys = Object.keys(types).filter((k) => counts[k]);
  const visibleCount = keys.reduce((n, k) => n + (hidden.has(k) ? 0 : counts[k]), 0);

  return (
    <aside className="hud legend">
      <h2>
        DISASTER TYPES <span className="cnt">{visibleCount}/{total}</span>
      </h2>
      <div>
        {keys.map((k) => (
          <div
            key={k}
            className={"lg" + (hidden.has(k) ? " off" : "")}
            style={{ color: types[k].color }}
            onClick={() => onToggle(k)}
          >
            <span className="sw" style={{ background: types[k].color }} />
            <span className="n">{types[k].label}</span>
            <span className="c">{counts[k]}</span>
          </div>
        ))}
      </div>
      <div className="hint">
        Hover a marker for before / after imagery + brief. Click to fly in. Scroll to zoom to street level.
      </div>
    </aside>
  );
}
