import React, { useEffect, useMemo, useRef, useState } from "react";
import Header from "./components/Header.jsx";
import ModeSelector from "./components/ModeSelector.jsx";
import MissionPanel from "./components/MissionPanel.jsx";
import Dossier from "./components/Dossier.jsx";
import BasemapToggle from "./components/BasemapToggle.jsx";
import MapView from "./components/MapView.jsx";
import QueryBox from "./components/QueryBox.jsx";
import VlmChat from "./components/VlmChat.jsx";
import { health, interpret, computeChange } from "./api.js";

export default function App() {
  const [disasterData, setDisasterData] = useState(null);
  const [missionsData, setMissionsData] = useState(null);
  const [error, setError] = useState(null);
  const [activeId, setActiveId] = useState("disaster");
  const [selected, setSelected] = useState(null);
  const [basemap, setBasemap] = useState("sat");
  const [zoom, setZoom] = useState(5);
  const [cursor, setCursor] = useState("--.--, --.--");
  const [focus, setFocus] = useState(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [live, setLive] = useState({});      // name -> geojson | null
  const [fstatus, setFstatus] = useState({}); // name -> 'loading'|'live'|'modelled'|'empty'

  useEffect(() => {
    Promise.all([
      fetch("disasters.json").then((r) => r.json()),
      fetch("missions.json").then((r) => r.json()),
    ])
      .then(([d, m]) => {
        setDisasterData(d);
        setMissionsData(m);
      })
      .catch(() => setError("Could not load data — serve the folder over http (npm run dev)."));
    health().then((h) => setBackendOnline(!!(h && h.ok && !h.offline)));
  }, []);

  const modes = useMemo(() => {
    if (!disasterData || !missionsData) return [];
    const types = disasterData.types || {};
    const disasterMode = {
      id: "disaster",
      label: "Disaster Response",
      icon: "◉",
      accent: "#e5484d",
      query: "Which villages are now under water compared to last week?",
      desc:
        "SAR flood-extent change detection sees through cloud — a flood map in minutes instead of an analyst's days. Real lives, real time-pressure.",
      sites: (disasterData.disasters || []).map((d) => ({
        ...d,
        color: (types[d.t] || {}).color,
        typeLabel: (types[d.t] || {}).label,
      })),
    };
    const missionModes = (missionsData.modes || []).map((m) => ({
      ...m,
      sites: m.sites.map((s) => ({ ...s, color: m.accent, typeLabel: m.label })),
    }));
    return [disasterMode, ...missionModes];
  }, [disasterData, missionsData]);

  const activeMode = modes.find((m) => m.id === activeId) || modes[0] || null;

  const changeMode = (id) => {
    setActiveId(id);
    setSelected(null);
  };

  const selectSite = (site) => {
    setSelected(site);
    if (site) setFocus({ lat: site.lat, lng: site.lng, z: Math.min(site.z || 12, 13), k: Date.now() });
  };

  // ---- natural-language query: backend interpreter, local fallback ----
  const localMatch = (q) => {
    const ql = q.toLowerCase();
    const tokens = ql.match(/[a-z]{3,}/g) || [];
    let best = null,
      bestMode = null,
      score = -1;
    modes.forEach((m) => {
      const modeHit = (m.label + " " + m.query).toLowerCase();
      const mBoost = tokens.some((t) => modeHit.includes(t)) ? 2 : 0;
      m.sites.forEach((s) => {
        const hay = (s.n + " " + (s.q || "") + " " + (s.change || "") + " " + m.label).toLowerCase();
        let sc = tokens.reduce((n, t) => n + (hay.includes(t) ? 1 : 0), 0) + mBoost;
        if (tokens.some((t) => s.n.toLowerCase().includes(t))) sc += 3;
        if (sc > score) {
          score = sc;
          best = s;
          bestMode = m;
        }
      });
    });
    if (!best || score <= 0) return { ok: false, message: "no matching site found" };
    return { ok: true, source: "local", mode: bestMode.id, name: best.n, lat: best.lat, lng: best.lng, _site: best };
  };

  const handleQuery = async (q) => {
    let res = null;
    if (backendOnline) {
      try {
        res = await interpret(q);
      } catch {
        res = null;
      }
    }
    if (!res || !res.ok) res = localMatch(q);
    if (res && res.ok) {
      const mode = modes.find((m) => m.id === res.mode) || activeMode;
      const site = res._site || (mode && mode.sites.find((s) => s.n === res.name)) || (mode && mode.sites[0]);
      if (mode) setActiveId(mode.id);
      if (site) selectSite(site);
    }
    return res;
  };

  // ---- live footprint fetch (debounced + cached per site) ----
  const timerRef = useRef(null);
  useEffect(() => {
    clearTimeout(timerRef.current);
    if (!selected || !backendOnline) return;
    const key = selected.n;
    if (live[key] !== undefined) return; // already resolved
    // no Sentinel coverage before ~2015 → keep modelled polygon, don't call
    if (!selected.dates) {
      setLive((m) => ({ ...m, [key]: null }));
      setFstatus((s) => ({ ...s, [key]: "historic" }));
      return;
    }
    timerRef.current = setTimeout(() => {
      setFstatus((s) => ({ ...s, [key]: "loading" }));
      computeChange(selected, activeId)
        .then((res) => {
          const g = res && res.ok && res.geojson && res.geojson.type ? res.geojson : null;
          const usable = g && !(g.type === "FeatureCollection" && (!g.features || !g.features.length));
          setLive((m) => ({ ...m, [key]: usable ? g : null }));
          setFstatus((s) => ({ ...s, [key]: usable ? "live" : res && res.empty ? "empty" : "modelled" }));
        })
        .catch(() => {
          setLive((m) => ({ ...m, [key]: null }));
          setFstatus((s) => ({ ...s, [key]: "modelled" }));
        });
    }, 700);
    return () => clearTimeout(timerRef.current);
  }, [selected, backendOnline, activeId, live]);

  if (error) {
    return (
      <aside className="hud dossier">
        <div className="dsr-empty">
          <b>DATA LOAD FAILED</b>
          {error}
        </div>
      </aside>
    );
  }

  const selKey = selected ? selected.n : null;
  const liveGeo = selKey ? live[selKey] : undefined;
  const footprint = liveGeo || (selected ? selected.geo : null);
  const footStatus = selKey ? fstatus[selKey] || (backendOnline ? "loading" : "modelled") : null;

  return (
    <>
      <MapView
        sites={activeMode ? activeMode.sites : []}
        accent={activeMode ? activeMode.accent : "#e5484d"}
        basemap={basemap}
        selected={selected}
        focus={focus}
        onSelect={setSelected}
        onZoom={setZoom}
        onCursor={setCursor}
      />
      <Header
        mission={activeMode ? activeMode.label : "—"}
        events={activeMode ? activeMode.sites.length : 0}
        zoom={zoom}
        cursor={cursor}
      />
      <QueryBox onResolve={handleQuery} backendOnline={backendOnline} />
      <ModeSelector modes={modes} activeId={activeId} onChange={changeMode} />
      {activeMode && <MissionPanel mode={activeMode} selected={selected} onSelect={selectSite} />}
      <Dossier event={selected} mode={activeMode} footprint={footprint} footStatus={footStatus} />
      <BasemapToggle value={basemap} onChange={setBasemap} />
      <VlmChat backendOnline={backendOnline} />
    </>
  );
}
