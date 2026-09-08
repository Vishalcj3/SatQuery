# Demo presets

Drop your demo images in this folder and name them to match `presets.json`
(or edit `presets.json` to match your filenames). Presets whose files are
missing are skipped automatically, so the app still runs with none present.

Suggested files (from LEVIR-CD / RSVQA samples you download):

- `vqa_airport.png`      — one optical scene with countable objects
- `ground_water.png`     — one optical scene containing a lake/river
- `change_before.png` + `change_after.png` — a LEVIR-CD bi-temporal pair
- `fusion_optical.png` + `fusion_sar.png`  — an optical + SAR pair of one area

Keep them small (≤1024 px). These make the live demo instant — no waiting on
uploads while judges watch.
