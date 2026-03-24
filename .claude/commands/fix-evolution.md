# Fix Evolution

Check all 6 HF Space evolution islands for issues:

```bash
for url in \
  "https://lbjlincoln-nomos-nba-quant.hf.space/api/status" \
  "https://lbjlincoln-nomos-nba-quant-2.hf.space/api/status" \
  "https://lbjlincoln26-nba-evo-3.hf.space/api/status" \
  "https://lbjlincoln26-nba-evo-4.hf.space/api/status" \
  "https://nomos42-nba-evo-5.hf.space/api/status" \
  "https://nomos42-nba-evo-6.hf.space/api/status"; do
  curl -s --max-time 15 "$url"
done
```

Diagnose: stagnation (same best_brier >10 gens), crashes (DOWN status), stuck in BUILDING FEATURES (>15 min).
Fix via S10 API: POST /api/command {"command":"diversify"} or POST /api/config with tuned params.
If a space is crashed, redeploy with: `python3 hf-space/deploy_island.py SPACE_ID ROLE TOKEN_VAR`
