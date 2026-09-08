# SatQuery task types and image metadata

## Task types (BigEarthNet.txt `type` / `category`)
- **binary** — yes/no questions. Categories: `presence` (is a class present?),
  `area` (does a class cover a given fraction/area?), `count` (are there N
  instances?), `adjacency` (do two classes adjoin?).
- **mcq** — multiple-choice questions over the same categories plus `climate zone`,
  `country`, `season`.
- **captioning** — free-text description of the scene's land cover and layout.
- **bounding box** — grounding: given a `<point>(x, y)</point>` or a
  `<ref>class</ref>`, return a box `[x1 y1, x2 y2]`. Coordinates are on a
  0..1000 grid (top-left origin, x right, y down).

## Answer conventions
- Binary answers are exactly `yes` or `no`.
- Grounding boxes are `[x1 y1, x2 y2]` integers in 0..1000.
- Be concise; do not restate the question. Abstain or say "uncertain" when the
  evidence is weak rather than guessing.

## Köppen–Geiger climate zones (values seen in this dataset)
- **Cold, no dry season, warm summer** (Dfb) — humid continental; much of
  central/eastern Europe.
- **Temperate, no dry season, warm summer** (Cfb) — oceanic; western Europe.
- **Temperate, dry/hot summer** (Csa/Csb) — Mediterranean.
- Climate zone can often be inferred from vegetation cues (e.g. sclerophyllous
  scrub suggests Mediterranean).

## Acquisition metadata
- **Seasons**: Spring, Summer, Fall, Winter — affect vegetation greenness and
  snow cover, which changes class appearance.
- **Countries in the corpus**: Austria, Belgium, Finland, Ireland, Kosovo,
  Lithuania, Luxembourg, Portugal, Serbia, Switzerland.
- Imagery is Sentinel-2 optical RGB previews (bands B04/B03/B02). Do not expect
  SAR or full 12-band spectral cues in the preview.
