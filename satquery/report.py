"""Downloadable report. Standalone HTML with base64-embedded images — no
Playwright / WeasyPrint install needed, opens in any browser, prints to PDF
with Ctrl+P. Robust beats fancy the night before a demo."""
import base64
import io
import time
import html as _html

from PIL import Image


def _img_tag(img: Image.Image, max_w=520) -> str:
    if img is None:
        return ""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img style="max-width:{max_w}px;border:1px solid #ccc;border-radius:6px" src="data:image/png;base64,{b64}">'


def build_report(query, result, trace, inputs: list[Image.Image], out_path="satquery_report.html") -> str:
    t = trace.to_dict()
    tools = ", ".join(x["tool"] for x in t["tools"]) or "(none)"
    steps = "\n".join(_html.escape(s) for s in t["steps"])
    inp = "".join(_img_tag(im, 300) for im in inputs)
    ev = _img_tag(result.get("evidence"))

    doc = f"""<!doctype html><html><head><meta charset=utf8>
<title>SatQuery Report</title>
<style>
body{{font:15px system-ui,Segoe UI,sans-serif;max-width:820px;margin:24px auto;padding:0 16px;color:#141a24}}
h1{{font-size:22px}} h2{{font-size:16px;border-bottom:1px solid #ddd;padding-bottom:4px;margin-top:28px}}
.kv{{background:#f4f6f9;border:1px solid #d7dee8;border-radius:8px;padding:12px 16px}}
.kv b{{display:inline-block;min-width:110px;color:#455066}}
pre{{background:#0f1620;color:#d6e2ef;padding:14px;border-radius:8px;overflow-x:auto;font-size:12.5px;white-space:pre-wrap}}
.ans{{background:#e0f2ea;border-left:4px solid #127a5b;padding:12px 16px;border-radius:6px}}
.imgrow{{display:flex;gap:12px;flex-wrap:wrap}}
small{{color:#6b7688}}
</style></head><body>
<h1>SatQuery AI — Analysis Report</h1>
<small>Generated {time.strftime('%Y-%m-%d %H:%M:%S')}</small>

<h2>Query</h2>
<div class="kv"><b>Question</b> {_html.escape(query or '(none)')}<br>
<b>Task</b> {_html.escape(t['task'])}<br>
<b>Images</b> {t['n_images']} &nbsp; modalities={_html.escape(str(t['modalities']))}<br>
<b>Tools run</b> {_html.escape(tools)}<br>
<b>Confidence</b> {t['confidence']}<br>
<b>Elapsed</b> {t['elapsed_s']}s</div>

<h2>Answer</h2>
<div class="ans">{_html.escape(result.get('answer','')).replace(chr(10),'<br>')}</div>

<h2>Input image(s)</h2><div class="imgrow">{inp}</div>

{('<h2>Visual evidence</h2>' + ev) if ev else ''}

<h2>Execution trace</h2>
<pre>{steps}</pre>

<hr><small>SatQuery MVP · pretrained RS models + small LoRA adaptation. Description-level
optical-SAR fusion and pixel-diff change detection are first-pass methods; scaling the
fine-tune and Cartosat/RISAT tuning are stated future work.</small>
</body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path
