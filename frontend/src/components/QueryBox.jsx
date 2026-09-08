import React, { useState } from "react";

// district-officer style query box. Calls the backend interpreter; if the
// backend is unreachable, falls back to a local matcher so it still works.
export default function QueryBox({ onResolve, backendOnline }) {
  const [q, setQ] = useState("");
  const [status, setStatus] = useState(null); // {text, kind}

  const submit = async (e) => {
    if (e) e.preventDefault();
    if (!q.trim()) return;
    setStatus({ text: "interpreting…", kind: "busy" });
    const res = await onResolve(q.trim());
    if (res && res.ok) {
      setStatus({
        text: `→ ${res.name}${res.source === "llm" ? "  · AI" : res.source === "local" ? "  · local" : ""}`,
        kind: "ok",
      });
    } else {
      setStatus({ text: res && res.message ? res.message : "no matching site found", kind: "err" });
    }
  };

  const examples = [
    "Which villages are under water in coastal Odisha?",
    "Which paddy fields aren't sown in the Cauvery delta?",
    "Where has built-up area grown around Bellandur lake?",
    "Has Nagarjuna Sagar reservoir dropped?",
  ];

  return (
    <form className="hud querybox" onSubmit={submit}>
      <div className="qb-row">
        <span className="qb-ic">⌕</span>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Ask in plain language — e.g. “which villages are now under water compared to last week?”"
          list="qb-examples"
          spellCheck="false"
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              submit();
            }
          }}
        />
        <datalist id="qb-examples">
          {examples.map((x) => (
            <option key={x} value={x} />
          ))}
        </datalist>
        <button type="submit">ANALYSE</button>
      </div>
      {status && <div className={"qb-status " + status.kind}>{status.text}</div>}
      {!backendOnline && (
        <div className="qb-status warn">backend offline — using local matching &amp; stored footprints</div>
      )}
    </form>
  );
}
