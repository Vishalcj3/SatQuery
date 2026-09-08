import React, { useEffect, useRef } from "react";
import L from "leaflet";

function makeIcon(color, sel) {
  return L.divIcon({
    className: "",
    html: `<div class="dmark${sel ? " sel" : ""}" style="--c:${color}"><i></i></div>`,
    iconSize: [16, 16],
    iconAnchor: [8, 8],
  });
}

export default function MapView({ sites, accent, basemap, selected, focus, onSelect, onZoom, onCursor }) {
  const elRef = useRef(null);
  const mapRef = useRef(null);
  const basesRef = useRef({});
  const groupRef = useRef(null);
  const markersRef = useRef([]); // { marker, site, color }
  const focusTimeRef = useRef(0); // when a site-focus fly last happened
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  // ---- init map once ----
  useEffect(() => {
    // perf options shared by every tile layer: don't fetch tiles mid-zoom
    // (kills the "reloading while zooming" lag), keep a small reuse buffer.
    // maxNativeZoom = deepest zoom the source actually has tiles for; Leaflet
    // upscales that tile for closer zooms instead of showing "no data" blanks.
    const perf = { updateWhenZooming: false, updateWhenIdle: true, keepBuffer: 2, maxZoom: 22 };

    const map = L.map(elRef.current, {
      center: [22.6, 79.5],
      zoom: 5,
      minZoom: 3,
      maxZoom: 22,
      zoomControl: true,
      zoomSnap: 0.5,
      zoomDelta: 0.5,
      wheelPxPerZoomLevel: 120,
      worldCopyJump: true,
    });
    mapRef.current = map;

    // dedicated pane for satellite imagery so the colour filter composites ONCE
    // for the whole layer instead of per-tile (a big zoom-perf win).
    map.createPane("satfx");
    map.getPane("satfx").style.filter = "saturate(1.24) contrast(1.07) brightness(1.07)";

    const esriSat = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { maxNativeZoom: 19, pane: "satfx", attribution: "Imagery © Esri, Maxar, Earthstar Geographics", ...perf }
    );
    const esriRef = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
      { maxNativeZoom: 19, opacity: 0.9, ...perf }
    );
    const sat = L.layerGroup([esriSat, esriRef]);
    // OSM stays crisp (vector-drawn) to the deepest zoom — streets, buildings, ponds.
    const osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxNativeZoom: 19, attribution: "© OpenStreetMap contributors", ...perf,
    });
    const dark = L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      maxNativeZoom: 20, subdomains: "abcd", attribution: "© OpenStreetMap · © CARTO", ...perf,
    });
    // HYBRID: satellite imagery + roads + place labels (see ground AND streets/ponds)
    const esriSat2 = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { maxNativeZoom: 19, pane: "satfx", ...perf }
    );
    const esriRoads = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}",
      { maxNativeZoom: 19, ...perf }
    );
    const esriRef2 = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
      { maxNativeZoom: 19, opacity: 0.9, ...perf }
    );
    const hybrid = L.layerGroup([esriSat2, esriRoads, esriRef2]);
    basesRef.current = { sat, map: osm, dark, hybrid, current: sat };
    sat.addTo(map);
    groupRef.current = L.layerGroup().addTo(map);
    setTimeout(() => map.invalidateSize(), 250);

    map.on("zoomend", () => onZoom && onZoom(map.getZoom().toFixed(2)));
    map.on("mousemove", (e) => onCursor && onCursor(`${e.latlng.lat.toFixed(3)}, ${e.latlng.lng.toFixed(3)}`));
    onZoom && onZoom(map.getZoom());

    return () => map.remove();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- rebuild markers when the active mode's sites change ----
  useEffect(() => {
    const map = mapRef.current;
    const group = groupRef.current;
    if (!map || !group) return;
    group.clearLayers();
    markersRef.current = [];
    const valid = sites.filter((s) => Number.isFinite(s.lat) && Number.isFinite(s.lng));
    valid.forEach((s) => {
      const color = s.color || accent;
      const m = L.marker([s.lat, s.lng], { icon: makeIcon(color, false), riseOnHover: true });
      m.on("mouseover", () => onSelectRef.current(s));
      m.on("click", () => {
        onSelectRef.current(s);
        map.flyTo([s.lat, s.lng], Math.min(s.z || 12, 13), { duration: 1.3 });
      });
      m.addTo(group);
      markersRef.current.push({ marker: m, site: s, color });
    });
    // frame the theatre for the new set (deferred + size-guarded — a 0-size map
    // makes Leaflet compute a NaN zoom and throw)
    if (valid.length) {
      const b = L.latLngBounds(valid.map((s) => [s.lat, s.lng]));
      setTimeout(() => {
        // if a query/panel just focused a specific site, don't yank the camera back out
        if (Date.now() - focusTimeRef.current < 1600) return;
        try {
          map.invalidateSize();
          if (b.isValid() && map.getSize().x > 0) map.fitBounds(b, { padding: [70, 70], maxZoom: 6 });
        } catch (e) {
          /* ignore transient sizing races */
        }
      }, 90);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sites, accent]);

  // ---- swap basemap ----
  useEffect(() => {
    const map = mapRef.current;
    const bs = basesRef.current;
    if (!map || !bs[basemap]) return;
    map.removeLayer(bs.current);
    const next = bs[basemap];
    next.addTo(map);
    if (basemap !== "sat" && next.bringToBack) next.bringToBack();
    bs.current = next;
  }, [basemap]);

  // ---- highlight selected ----
  useEffect(() => {
    markersRef.current.forEach(({ marker, site, color }) => {
      marker.setIcon(makeIcon(color, selected && site === selected));
    });
  }, [selected]);

  // ---- fly to a query/panel-driven focus target ----
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !focus || !Number.isFinite(focus.lat) || !Number.isFinite(focus.lng)) return;
    focusTimeRef.current = Date.now();
    setTimeout(() => {
      try {
        if (map.getSize().x > 0) map.flyTo([focus.lat, focus.lng], focus.z || 12, { duration: 1.4 });
      } catch (e) {
        /* ignore */
      }
    }, 120);
  }, [focus]);

  return <div id="map" ref={elRef} />;
}
