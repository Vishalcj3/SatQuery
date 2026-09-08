import React, { useRef, useState } from "react";
import { vlmQuery } from "../api.js";

// Satellite-image VQA assistant backed by the fine-tuned Qwen-VL + RAG pipeline
// (POST /api/vlm). Drop 1 image (VQA / caption / grounding / count) or 2
// (change / optical–SAR fusion); ask in plain language.
const EXAMPLES = [
  "How many runways are visible?",
  "How many acres are under water?",
  "Highlight the water body",
  "What changed between these two images?",
];

export default function VlmChat({ backendOnline }) {
  const [open, setOpen] = useState(false);
  const [imgs, setImgs] = useState([]); // [{file, url, mod}]
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  const [showTrace, setShowTrace] = useState(false);
  const [drag, setDrag] = useState(false);
  const inputRef = useRef(null);

  const addFiles = (fileList) => {
    const next = Array.from(fileList || [])
      .filter((f) => f.type.startsWith("image/") || /\.tiff?$/i.test(f.name))
      .map((f) => ({ file: f, url: URL.createObjectURL(f), mod: "auto" }));
    if (next.length) setImgs((cur) => [...cur, ...next]);
    setErr(null);
  };
  const removeImg = (i) => setImgs((cur) => cur.filter((_, k) => k !== i));
  const setMod = (i, mod) => setImgs((cur) => cur.map((x, k) => (k === i ? { ...x, mod } : x)));

  const submit = async (e) => {
    if (e) e.preventDefault();
    if (!imgs.length) return setErr("Add at least one image.");
    if (!query.trim()) return setErr("Type a question.");
    setBusy(true);
    setErr(null);
    setRes(null);
    try {
      const out = await vlmQuery({
        image1: imgs[0].file,
        image2: imgs[1] ? imgs[1].file : undefined,
        query: query.trim(),
        mod1: imgs[0].mod,
        mod2: imgs[1] ? imgs[1].mod : "auto",
      });
      setRes(out);
    } catch (e2) {
      setErr(e2.message || "request failed");
    } finally {
      setBusy(false);
    }
  };

  if (!open) {
    return (
      <button className="vlm-fab" onClick={() => setOpen(true)} title="Ask the satellite-image VLM">
        🛰 IMAGE&nbsp;VLM
      </button>
    );
  }

  return (
    <div className="vlm-panel">
      <div className="vlm-head">
        <div>
          <div className="vlm-title">SATQUERY&nbsp;VLM</div>
          <div className="vlm-sub">Qwen-VL + RAG · image intelligence</div>
        </div>
        <button className="vlm-x" onClick={() => setOpen(false)}>×</button>
      </div>

      {!backendOnline && (
        <div className="vlm-warn">Backend offline — start it: <code>uvicorn app:app --port 8000</code></div>
      )}

      <form onSubmit={submit}>
        <div
          className={"vlm-drop" + (drag ? " over" : "")}
          onClick={() => inputRef.current && inputRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); addFiles(e.dataTransfer.files); }}
        >
          {imgs.length === 0 ? (
            <span>Drop image(s) or click — 1 for VQA/count/caption, 2 for change / optical–SAR fusion</span>
          ) : (
            <div className="vlm-thumbs">
              {imgs.map((im, i) => (
                <div className="vlm-thumb" key={i} onClick={(e) => e.stopPropagation()}>
                  <img src={im.url} alt={"img" + i} />
                  <button type="button" className="vlm-rm" onClick={() => removeImg(i)}>×</button>
                  <select value={im.mod} onChange={(e) => setMod(i, e.target.value)}>
                    <option value="auto">auto</option>
                    <option value="optical">optical</option>
                    <option value="sar">SAR</option>
                  </select>
                  {i > 1 && <span className="vlm-extra">extra</span>}
                </div>
              ))}
            </div>
          )}
          <input
            ref={inputRef} type="file" accept="image/*,.tif,.tiff" multiple hidden
            onChange={(e) => addFiles(e.target.files)}
          />
        </div>
        {imgs.length > 2 && (
          <div className="vlm-note">Pipeline uses the first 2 images (change / fusion); the rest are ignored.</div>
        )}

        <div className="vlm-examples">
          {EXAMPLES.map((x) => (
            <button type="button" key={x} onClick={() => setQuery(x)}>{x}</button>
          ))}
        </div>

        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about the image(s) — e.g. how many runways / acres affected / what changed?"
          rows={2}
        />
        <button className="vlm-ask" type="submit" disabled={busy}>
          {busy ? "ANALYSING… (first call loads the model)" : "ANALYSE"}
        </button>
      </form>

      {err && <div className="vlm-warn">{err}</div>}

      {res && (
        <div className="vlm-res">
          <div className="vlm-task">
            task: <b>{res.task || "?"}</b>{res.extra && res.extra.rag_used ? " · RAG" : ""}
          </div>
          <div className="vlm-answer">{res.answer}</div>
          {res.evidence && <img className="vlm-evidence" src={res.evidence} alt="visual evidence" />}
          <button type="button" className="vlm-tracebtn" onClick={() => setShowTrace((s) => !s)}>
            {showTrace ? "hide" : "show"} execution trace
          </button>
          {showTrace && <pre className="vlm-trace">{res.trace}</pre>}
        </div>
      )}
    </div>
  );
}
