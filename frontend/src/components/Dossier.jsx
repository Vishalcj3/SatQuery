import React, { useEffect, useRef } from "react";
import L from "leaflet";

const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";

// non-interactive satellite mini-map. geo -> highlight only the changed area.
function miniMap(el, opts) {
  const m = L.map(el, {
    center: opts.center,
    zoom: opts.zoom || 11,
    zoomControl: false,
    attributionControl: false,
    dragging: false,
    scrollWheelZoom: false,
    doubleClickZoom: false,
    boxZoom: false,
    keyboard: false,
    touchZoom: false,
    fadeAnimation: false,
  });
  L.tileLayer(ESRI, { maxZoom: 19, className: "realistic-sat" }).addTo(m);

  if (opts.geo) {
    L.geoJSON(opts.geo, {
      style: {
        color: opts.zoneColor,
        weight: 1.8,
        opacity: 0.95,
        fillColor: opts.zoneColor,
        fillOpacity: 0.32,
        className: "affzone",
      },
    }).addTo(m);
  } else if (opts.rkm != null) {
    L.circle(opts.center, {
      radius: opts.rkm * 1000,
      color: opts.zoneColor,
      weight: 1.6,
      fillColor: opts.zoneColor,
      fillOpacity: 0.28,
      className: "affzone",
    }).addTo(m);
  }

  if (opts.bounds) m.fitBounds(opts.bounds, { padding: [10, 10], maxZoom: 15 });
  setTimeout(() => m.invalidateSize(), 60);
  return m;
}

const FOOT = {
  live: { t: "LIVE SENTINEL", c: "#41c98a" },
  loading: { t: "COMPUTING…", c: "#e8b23a" },
  modelled: { t: "MODELLED", c: "#7a8f99" },
  empty: { t: "NO CHANGE", c: "#7a8f99" },
  historic: { t: "MODELLED · PRE-SAT", c: "#7a8f99" },
};
function FootChip({ status }) {
  const s = FOOT[status] || FOOT.modelled;
  return (
    <span className="dsr-foot" style={{ color: s.c, borderColor: s.c }}>
      {status === "loading" ? "◌ " : "◉ "}
      {s.t}
    </span>
  );
}

// area of a GeoJSON Polygon/MultiPolygon in km² (equirectangular, good at these scales)
function polyAreaKm2(geo) {
  if (!geo) return null;
  let rings = [];
  if (geo.type === "Polygon") rings = [geo.coordinates[0]];
  else if (geo.type === "MultiPolygon") rings = geo.coordinates.map((p) => p[0]);
  else return null;
  if (!rings.length || !rings[0] || !rings[0].length) return null;
  const latRef = rings[0][0][1];
  const kx = 111.32 * Math.cos((latRef * Math.PI) / 180);
  const ky = 110.574;
  let total = 0;
  for (const ring of rings) {
    let a = 0;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      a += ring[j][0] * kx * (ring[i][1] * ky) - ring[i][0] * kx * (ring[j][1] * ky);
    }
    total += Math.abs(a / 2);
  }
  return total;
}

export default function Dossier({ event, mode, footprint, footStatus }) {
  const beforeRef = useRef(null);
  const afterRef = useRef(null);
  const zone = (mode && mode.accent) || "#e5484d";
  const geo = footprint || (event && event.geo) || null;

  useEffect(() => {
    if (!event || !beforeRef.current || !afterRef.current) return;
    const center = [event.lat, event.lng];
    let bounds = null;
    if (geo) {
      try {
        bounds = L.geoJSON(geo).getBounds();
      } catch (e) {
        bounds = null;
      }
    }
    const mB = miniMap(beforeRef.current, { center, zoom: event.z, bounds });
    const mA = miniMap(afterRef.current, {
      center,
      zoom: event.z,
      bounds,
      geo: geo,
      rkm: geo ? null : event.r_km,
      zoneColor: zone,
    });
    return () => {
      mB.remove();
      mA.remove();
    };
  }, [event, zone, geo]);

  if (!event) {
    return (
      <aside className="hud dossier">
        <div className="dsr-empty">
          <b>NO SITE SELECTED</b>
          Hover a marker — the detected change is outlined on the AFTER map and explained here.
        </div>
      </aside>
    );
  }

  const chip = event.typeLabel || (mode && mode.label) || "SITE";
  const chipColor = event.color || zone;
  const query = event.q || (mode && mode.query);

  return (
    <aside className="hud dossier">
      <div className="dsr-head" style={{ borderColor: chipColor }}>
        <span className="dsr-type" style={{ background: chipColor }}>
          {chip.toUpperCase()}
        </span>
        <h3>{event.n}</h3>
        <div className="dsr-meta">
          {event.date ? event.date + " · " : ""}
          {event.sev ? <span className="sev">{event.sev}</span> : null}
          {event.sev ? " · " : ""}
          {event.lat.toFixed(3)}°N {event.lng.toFixed(3)}°E
        </div>
      </div>

      {query && <div className="dsr-query">“{query}”</div>}

      <div className="dsr-imgs">
        <figure className="before">
          <div className="dsr-map" ref={beforeRef} />
          <figcaption>BEFORE</figcaption>
        </figure>
        <figure className="after">
          <span className="dsr-affected" style={{ background: zone }}>
            ◉ {event.change || "CHANGED AREA"}
          </span>
          {footStatus && <FootChip status={footStatus} />}
          <div className="dsr-map" ref={afterRef} />
          <figcaption>AFTER · CHANGE DETECTED</figcaption>
        </figure>
      </div>

      <div className="dsr-impact">
        <div className="imp-title">IMPACT ASSESSMENT{footStatus === "live" ? " · measured" : " · estimated"}</div>
        <div className="imp-grid">
          {(() => {
            const km2 = polyAreaKm2(geo);
            const val = km2 == null ? "—" : `${Math.round(km2 * 247.105).toLocaleString()} ac · ${km2.toFixed(1)} km²`;
            return (
              <div className="imp-row"><span>Area affected</span><b>{val}</b></div>
            );
          })()}
          {(event.impact || []).map(([label, v], i) => (
            <div className="imp-row" key={i}><span>{label}</span><b>{v}</b></div>
          ))}
          <div className="imp-row"><span>Whereabouts</span><b>{event.where || "—"}</b></div>
          <div className="imp-row"><span>Coordinates</span><b>{event.lat.toFixed(3)}°N {event.lng.toFixed(3)}°E</b></div>
        </div>
      </div>

      <ul className="dsr-facts">
        {(event.f || []).map((f, i) => (
          <li key={i} dangerouslySetInnerHTML={{ __html: f }} />
        ))}
      </ul>
      <div className="dsr-src">◉ Esri World Imagery · coloured outline = detected change footprint</div>
    </aside>
  );
}
