import React, { useEffect, useState } from "react";

function zulu() {
  const d = new Date();
  const z = (n) => String(n).padStart(2, "0");
  return `${z(d.getUTCHours())}:${z(d.getUTCMinutes())}:${z(d.getUTCSeconds())}Z`;
}

export default function Header({ mission, events, zoom, cursor }) {
  const [clock, setClock] = useState(zulu());
  useEffect(() => {
    const t = setInterval(() => setClock(zulu()), 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <header className="hud topbar">
      <div className="brand">
        <svg className="eye" viewBox="0 0 48 48" fill="none" stroke="#e8b23a" strokeWidth="2.4">
          <path d="M4 24s8-13 20-13 20 13 20 13-8 13-20 13S4 24 4 24z" />
          <circle cx="24" cy="24" r="7" fill="rgba(232,178,58,.15)" />
          <circle cx="24" cy="24" r="2.4" fill="#e8b23a" stroke="none" />
        </svg>
        <div>
          <h1>SATQUERY</h1>
          <small>INDIA&nbsp;·&nbsp;CHANGE&nbsp;INTEL</small>
        </div>
      </div>
      <div className="stats">
        <div className="stat"><b>ZULU</b> <span className="v">{clock}</span></div>
        <div className="stat"><b>MISSION</b> <span className="live">{mission}</span></div>
        <div className="stat"><b>SITES</b> <span className="v">{events}</span></div>
        <div className="stat"><b>ZOOM</b> <span className="v">{zoom}</span></div>
        <div className="stat"><b>CURSOR</b> <span className="v">{cursor}</span></div>
      </div>
    </header>
  );
}
