"""Load the mission/disaster catalog from the frontend data files and flatten it
into a list of sites the API can match queries against and locate on the map."""
import json
import os

HERE = os.path.dirname(__file__)
PUB = os.path.normpath(os.path.join(HERE, "..", "frontend", "public"))

# default Sentinel index + change direction per mode.
# sar_* use Sentinel-1 (cloud-penetrating); the rest use optical Sentinel-2.
MODE_INDEX = {
    "disaster":    ("sar_water", "gain"),  # S1 SAR: new water appeared (flood, through cloud)
    "agri":        ("ndvi", "loss"),       # S2: vegetation lower (unsown / fallow)
    "urban":       ("ndbi", "gain"),       # S2: built-up increased
    "forestwater": ("ndvi", "loss"),       # S2: canopy/water loss (reservoirs override to water/loss)
    "defence":     ("sar_struct", "gain"), # S1 SAR: new bright targets / structures, day-or-night
}


def _bbox_from_geo(geo, lat, lng):
    try:
        ring = geo["coordinates"][0]
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return [min(xs), min(ys), max(xs), max(ys)]
    except Exception:
        d = 0.05
        return [lng - d, lat - d, lng + d, lat + d]


def load_catalog():
    sites = []
    # disaster mode
    try:
        d = json.load(open(os.path.join(PUB, "disasters.json"), encoding="utf-8"))
        types = d.get("types", {})
        for i, s in enumerate(d.get("disasters", [])):
            sites.append(_site("disaster", "Disaster Response", i, s,
                               (types.get(s["t"], {}) or {}).get("label", s["t"])))
    except Exception as e:
        print("[catalog] disasters.json:", e)
    # mission modes
    try:
        m = json.load(open(os.path.join(PUB, "missions.json"), encoding="utf-8"))
        for mode in m.get("modes", []):
            for i, s in enumerate(mode["sites"]):
                sites.append(_site(mode["id"], mode["label"], i, s, mode["label"]))
    except Exception as e:
        print("[catalog] missions.json:", e)
    return sites


def _site(mode_id, mode_label, idx, s, type_label):
    idx_name, direction = MODE_INDEX.get(mode_id, ("ndvi", "loss"))
    kw = " ".join([
        s.get("n", ""), s.get("change", ""), s.get("q", ""), type_label, mode_label,
    ]).lower()
    return {
        "mode": mode_id,
        "mode_label": mode_label,
        "site_index": idx,
        "name": s.get("n", ""),
        "lat": s.get("lat"),
        "lng": s.get("lng"),
        "z": s.get("z", 12),
        "change": s.get("change", ""),
        "query": s.get("q", ""),
        "index": s.get("index", idx_name),
        "direction": direction,
        "bbox": _bbox_from_geo(s.get("geo"), s.get("lat"), s.get("lng")),
        "keywords": kw,
    }
