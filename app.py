"""SatQuery AI — Gradio app. Single entry point: `python app.py`.

Upload 1 image (VQA / caption / grounding) or 2 images (change / optical-SAR
fusion), type a query, and the agent routes it to the right tool and shows the
answer + visual evidence + the execution trace."""
import os
import json
import tempfile

import gradio as gr

from satquery import handlers, io_utils, report, config


def _previews(paths, mods):
    out = []
    for p, m in zip(paths, mods):
        try:
            out.append(io_utils.load_image(p, io_utils.guess_modality(p, m)))
        except Exception as e:  # noqa
            print(f"[app] preview failed for {p}: {e}")
    return out


def analyze(file1, file2, mod1, mod2, query):
    paths = [p for p in (file1, file2) if p]
    mods = [mod1, mod2][: len(paths)]
    if not paths:
        return "Upload at least one image.", None, "", None, None

    result, trace = handlers.run(query, paths, mods)
    previews = _previews(paths, mods)

    rep_path = os.path.join(tempfile.gettempdir(), "satquery_report.html")
    try:
        report.build_report(query, result, trace, previews, out_path=rep_path)
    except Exception as e:  # noqa
        print(f"[app] report build failed: {e}")
        rep_path = None

    return (result.get("answer", ""),
            result.get("evidence"),
            trace.pretty(),
            rep_path,
            previews[0] if previews else None)


def load_presets():
    """Load presets/presets.json if present. Each entry:
       {"label","file1","file2","mod1","mod2","query"} (files relative to repo root)."""
    pj = os.path.join(os.path.dirname(__file__), "presets", "presets.json")
    if not os.path.exists(pj):
        return []
    try:
        with open(pj, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa
        print(f"[app] presets.json parse error: {e}")
        return []
    rows = []
    root = os.path.dirname(__file__)
    for d in data:
        f1 = os.path.join(root, d["file1"]) if d.get("file1") else None
        f2 = os.path.join(root, d["file2"]) if d.get("file2") else None
        if f1 and not os.path.exists(f1):
            continue  # skip presets whose files aren't there yet
        rows.append([f1, f2, d.get("mod1", "auto"), d.get("mod2", "auto"), d.get("query", "")])
    return rows


with gr.Blocks(title="SatQuery AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛰️ SatQuery AI\nAgentic remote-sensing assistant — ask in plain "
                "language, the agent picks the tool and shows its work.")
    gr.Markdown(f"`{config.summary()}`")

    with gr.Row():
        with gr.Column(scale=1):
            file1 = gr.File(label="Image 1 (required)", type="filepath",
                            file_types=["image", ".tif", ".tiff"])
            mod1 = gr.Radio(["auto", "optical", "sar"], value="auto", label="Image 1 sensor")
            file2 = gr.File(label="Image 2 (optional — for change / fusion)", type="filepath",
                            file_types=["image", ".tif", ".tiff"])
            mod2 = gr.Radio(["auto", "optical", "sar"], value="auto", label="Image 2 sensor")
            query = gr.Textbox(label="Your query",
                               placeholder="e.g. How many buildings are visible? / "
                                           "Highlight the water body / What changed between these?")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=1):
            answer = gr.Textbox(label="Answer", lines=6)
            evidence = gr.Image(label="Visual evidence (box / change map / composite)")
            preview = gr.Image(label="Input preview")
            trace_box = gr.Textbox(label="Execution trace (what the agent did)", lines=14)
            report_file = gr.File(label="Download report (HTML — Ctrl+P for PDF)")

    btn.click(analyze, [file1, file2, mod1, mod2, query],
              [answer, evidence, trace_box, report_file, preview])

    presets = load_presets()
    if presets:
        gr.Markdown("### Demo presets")
        gr.Examples(presets, inputs=[file1, file2, mod1, mod2, query],
                    label="Click a preset, then Analyze")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", inbrowser=True)
