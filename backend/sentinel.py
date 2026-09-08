"""Live change detection via the Copernicus Data Space Sentinel Hub API.

Supports BOTH sensors:
  * Sentinel-2 optical indices — NDWI (water), NDVI (vegetation), NDBI (built-up)
  * Sentinel-1 SAR backscatter — sees THROUGH cloud, day or night:
        sar_water : flood mapping (new low-backscatter = new water)
        sar_struct: new bright targets (structures / cleared ground)

Set two env vars from a FREE Copernicus Data Space account:
    SH_CLIENT_ID, SH_CLIENT_SECRET

Note: Sentinel-2 covers ~2015+, Sentinel-1 ~2014+. Older events have no coverage
and return {ok:true, empty:true} — the app then keeps the modelled polygon.
"""
import datetime as dt
import os

from catalog import MODE_INDEX

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# per index: (satellite collection, evalscript, default threshold, water-is-low?)
NDWI = """//VERSION=3
function setup(){return{input:["B03","B08","dataMask"],output:{bands:1,sampleType:"FLOAT32"}}}
function evaluatePixel(s){ if(s.dataMask===0) return [-999]; return [ (s.B03-s.B08)/(s.B03+s.B08+1e-6) ]; }"""
NDVI = """//VERSION=3
function setup(){return{input:["B04","B08","dataMask"],output:{bands:1,sampleType:"FLOAT32"}}}
function evaluatePixel(s){ if(s.dataMask===0) return [-999]; return [ (s.B08-s.B04)/(s.B08+s.B04+1e-6) ]; }"""
NDBI = """//VERSION=3
function setup(){return{input:["B08","B11","dataMask"],output:{bands:1,sampleType:"FLOAT32"}}}
function evaluatePixel(s){ if(s.dataMask===0) return [-999]; return [ (s.B11-s.B08)/(s.B11+s.B08+1e-6) ]; }"""
# Sentinel-1 VV backscatter in dB
S1VV = """//VERSION=3
function setup(){return{input:["VV","dataMask"],output:{bands:1,sampleType:"FLOAT32"}}}
function evaluatePixel(s){ if(s.dataMask===0) return [-999]; var v=Math.max(s.VV,1e-6); return [ 10*Math.log(v)/Math.LN10 ]; }"""

# index -> (collection_type, evalscript, threshold, rule)
#   rule: 'high_gain'  -> value crosses UP over threshold  (before<t, after>=t)
#         'high_loss'  -> value crosses DOWN below threshold(before>=t, after<t)
INDEX = {
    "water":      ("sentinel-2-l2a", NDWI, 0.05, None),   # NDWI>0.05 = water
    "ndvi":       ("sentinel-2-l2a", NDVI, 0.25, None),
    "ndbi":       ("sentinel-2-l2a", NDBI, 0.00, None),
    "sar_water":  ("sentinel-1-grd", S1VV, -17.0, None),  # VV < -17 dB = smooth water
    "sar_struct": ("sentinel-1-grd", S1VV, -6.0, None),   # VV >= -6 dB = bright target
}
# how a "gain" of the feature maps to a value crossing, per index:
#   water/sar_water: feature (water) is LOW value  -> gain = value drops below t
#   ndvi/ndbi/sar_struct: feature is HIGH value     -> gain = value rises above t
LOW_IS_FEATURE = {"water", "sar_water"}


def available():
    return bool(os.environ.get("SH_CLIENT_ID") and os.environ.get("SH_CLIENT_SECRET"))


def _token():
    import requests
    r = requests.post(TOKEN_URL, data={
        "grant_type": "client_credentials",
        "client_id": os.environ["SH_CLIENT_ID"],
        "client_secret": os.environ["SH_CLIENT_SECRET"],
    }, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def _fetch(token, bbox, collection, evalscript, t0, t1, size=512):
    import requests
    data = {"type": collection, "dataFilter": {"timeRange": {"from": t0 + "T00:00:00Z", "to": t1 + "T23:59:59Z"}}}
    if collection == "sentinel-2-l2a":
        data["dataFilter"]["maxCloudCoverage"] = 70
        data["dataFilter"]["mosaickingOrder"] = "leastCC"
    else:  # sentinel-1-grd
        data["dataFilter"]["acquisitionMode"] = "IW"
        data["dataFilter"]["polarization"] = "DV"
        data["processing"] = {"backCoeff": "GAMMA0_TERRAIN", "orthorectify": True}
    body = {
        "input": {"bounds": {"bbox": bbox,
                             "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                  "data": [data]},
        "output": {"width": size, "height": size,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": evalscript,
    }
    r = requests.post(PROCESS_URL, json=body, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    r.raise_for_status()
    return r.content


def compute_change(p):
    """p: lat,lng,bbox?,mode?,index?,direction?,date_before_from?,date_before?,date_after_from?,date_after?"""
    if not available():
        return {"ok": False, "reason": "no_credentials", "fallback": True,
                "message": "Set SH_CLIENT_ID / SH_CLIENT_SECRET (free Copernicus Data Space) to compute live footprints."}
    try:
        import io
        import numpy as np
        import rasterio
        from rasterio.features import shapes
        from rasterio.transform import from_bounds
        from shapely.geometry import shape, mapping
        from shapely.ops import unary_union
    except Exception as e:
        return {"ok": False, "reason": "geo_libs_missing", "fallback": True,
                "message": f"Install geo libs: pip install numpy rasterio shapely ({e})"}

    lat, lng = p["lat"], p["lng"]
    bbox = p.get("bbox") or [lng - 0.06, lat - 0.05, lng + 0.06, lat + 0.05]
    mode = p.get("mode")
    index = p.get("index") or (MODE_INDEX.get(mode, ("water", "gain"))[0])
    direction = p.get("direction") or (MODE_INDEX.get(mode, ("water", "gain"))[1])
    if index not in INDEX:
        index = "water"
    collection, evalscript, thr, _ = INDEX[index]

    today = dt.date.today()
    b0 = p.get("date_before_from") or (today - dt.timedelta(days=440)).isoformat()
    b1 = p.get("date_before") or (today - dt.timedelta(days=380)).isoformat()
    a0 = p.get("date_after_from") or (today - dt.timedelta(days=90)).isoformat()
    a1 = p.get("date_after") or (today - dt.timedelta(days=20)).isoformat()

    try:
        tok = _token()
        buf_b = _fetch(tok, bbox, collection, evalscript, b0, b1)
        buf_a = _fetch(tok, bbox, collection, evalscript, a0, a1)

        def read(buf):
            with rasterio.open(io.BytesIO(buf)) as ds:
                return ds.read(1).astype("float32")
        before, after = read(buf_b), read(buf_a)
        valid = (before > -900) & (after > -900)

        # a "gain" of the feature:
        if index in LOW_IS_FEATURE:
            # feature = value below threshold; gain = became feature
            feat_before = before < thr
            feat_after = after < thr
        else:
            feat_before = before >= thr
            feat_after = after >= thr
        if direction == "gain":
            mask = valid & (~feat_before) & feat_after
        else:  # loss
            mask = valid & feat_before & (~feat_after)
        mask = mask.astype("uint8")

        if int(mask.sum()) == 0:
            return {"ok": True, "source": "sentinel", "sensor": collection, "index": index, "empty": True,
                    "geojson": {"type": "FeatureCollection", "features": []},
                    "message": "No change above threshold in this window (or no scenes in range)."}

        h, w = mask.shape
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
        polys = [shape(g) for g, v in shapes(mask, mask=mask.astype(bool), transform=transform) if v == 1]
        merged = unary_union(polys).simplify(0.0004, preserve_topology=True)
        px = float(mask.sum())
        area_km2 = round(px * abs((bbox[2] - bbox[0]) / w) * abs((bbox[3] - bbox[1]) / h) * 111 * 111 * 0.9, 2)
        return {"ok": True, "source": "sentinel", "sensor": collection, "index": index, "direction": direction,
                "changed_pixels": int(px), "approx_area_km2": area_km2,
                "window": {"before": [b0, b1], "after": [a0, a1]},
                "geojson": mapping(merged)}
    except Exception as e:
        return {"ok": False, "reason": "error", "fallback": True, "message": str(e)}
