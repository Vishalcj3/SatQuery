import React from "react";

const OPTS = [
  { id: "sat", label: "◐ SAT" },
  { id: "hybrid", label: "⌖ HYBRID" },
  { id: "map", label: "▦ MAP" },
  { id: "dark", label: "☾ DARK" },
];

export default function BasemapToggle({ value, onChange }) {
  return (
    <div className="hud basemap">
      {OPTS.map((o) => (
        <button key={o.id} className={value === o.id ? "act" : ""} onClick={() => onChange(o.id)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
