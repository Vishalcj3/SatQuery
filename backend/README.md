# SatQuery India — Geospatial Intelligence API

FastAPI backend powering two things the frontend can't do alone:

1. **`POST /api/query`** — interprets a plain-language question → the best mission + site.
2. **`POST /api/change`** — runs **live Sentinel-2 change detection** and returns the
   exact changed-area polygon (GeoJSON) — the pinpoint footprint.

The frontend degrades gracefully: if this API is offline it uses a local matcher for
the query box and the stored (modelled) polygons for footprints. When the API is up
and credentials are set, footprints switch to **LIVE SENTINEL**.

## Run (base — works immediately, no keys)

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate      # optional
pip install fastapi "uvicorn[standard]" requests pydantic
uvicorn app:app --reload --port 8000
```

- `GET  http://127.0.0.1:8000/health`
- Query interpretation works out of the box via a keyword/place matcher.
- `/api/change` returns a `no_credentials` fallback until you set the keys below.

The frontend looks for the API at `http://127.0.0.1:8000` by default. To point it
elsewhere, create `frontend/.env` with `VITE_API=https://your-host`.

## Go live — pinpoint Sentinel footprints (free)

1. Create a free **Copernicus Data Space** account: https://dataspace.copernicus.eu
2. In its **Sentinel Hub** dashboard, create an **OAuth client** (client id + secret).
3. Install the geo libraries and export the keys:

```bash
pip install numpy rasterio shapely
set SH_CLIENT_ID=your-client-id            # PowerShell: $env:SH_CLIENT_ID="..."
set SH_CLIENT_SECRET=your-client-secret
uvicorn app:app --reload --port 8000
```

Now `/api/change` fetches a "before" and "after" scene, differences + thresholds them to a
change mask, and vectorises the exact changed pixels to a GeoJSON polygon. Sensor per mode:

| Mode | Sensor | Index |
|---|---|---|
| Disaster Response | **Sentinel-1 SAR** (through cloud) | `sar_water` — new low-backscatter = flood water |
| Border Monitoring | **Sentinel-1 SAR** (day/night) | `sar_struct` — new bright targets = structures |
| Agriculture | Sentinel-2 | `ndvi` loss (unsown / fallow) |
| Urban | Sentinel-2 | `ndbi` gain (built-up) |
| Forest & Water | Sentinel-2 | `ndvi` loss / `water` loss (reservoirs) |

Each site carries an event-pinned `dates` window (before/after) so the analysis targets
the actual event, not just recent change. In the app the dossier footprint flips to
**LIVE SENTINEL**. Events before ~2015 have no Sentinel coverage and stay **MODELLED · PRE-SAT**.

## Go smarter — Claude-interpreted queries (optional)

```bash
pip install anthropic
set ANTHROPIC_API_KEY=sk-ant-...
set ANTHROPIC_MODEL=claude-3-5-sonnet-latest   # optional override
```

Free-form questions are then routed by Claude instead of the keyword matcher
(the matcher stays as the fallback).

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{ok, sites, sentinel_configured, llm_configured}` |
| POST | `/api/query` | `{"q": "..."}` | `{ok, mode, name, lat, lng, source, confidence}` |
| POST | `/api/change` | `{lat, lng, bbox?, index?, direction?, date_before?, date_after?}` | `{ok, geojson, changed_pixels, approx_area_km2}` or `{ok:false, fallback:true}` |

`catalog.py` reads `../frontend/public/disasters.json` + `missions.json`, so adding a
site to those files automatically makes it queryable and analysable here too.
