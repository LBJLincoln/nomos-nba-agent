---
name: deploy-islands
description: Deploy feature engine or config updates to all 6 HF evolution islands
---

Deploy updates to one or all 6 HF evolution islands.

Arguments: $ARGUMENTS (required: "all" or space names like "S10 S12", optional: "config-only" or "engine-only")

## Steps

1. **Pre-flight checks**:
   - Verify feature engine parity: `sha256sum features/engine.py hf-space/features/engine.py`
   - If different, copy: `cp features/engine.py hf-space/features/engine.py`
   - Run syntax check: `python3 -c "from hf_space.features.engine import NBAFeatureEngine; e=NBAFeatureEngine(); print(f'{len(e.feature_names)} features OK')"`
   - Check hf-space/app.py exists and is valid

2. **Determine target spaces** from $ARGUMENTS:
   ```
   S10: Nomos42/nba-quant (HF_TOKEN_3)
   S11: Nomos42/nba-quant-2 (HF_TOKEN_3)
   S12: Nomos42/nba-evo-3 (HF_TOKEN_3)
   S13: Nomos42/nba-evo-4 (HF_TOKEN_3)
   S14: Nomos42/nba-evo-5 (HF_TOKEN_3)
   S15: Nomos42/nba-evo-6 (HF_TOKEN_3)
   ```

3. **Deploy** using git subtree push for each target:
   ```bash
   # MUST use subtree push, NEVER push full repo
   cd /home/termius/nomos-nba-agent
   git subtree push --prefix=hf-space https://$TOKEN@huggingface.co/spaces/$SPACE main
   ```
   If subtree push fails (non-fast-forward), use:
   ```bash
   git subtree split --prefix=hf-space -b hf-deploy
   git push -f https://$TOKEN@huggingface.co/spaces/$SPACE hf-deploy:main
   git branch -D hf-deploy
   ```

4. **Verify deployment** — wait 60s then check /api/status:
   ```bash
   sleep 60 && curl -s "https://$SPACE_URL/api/status" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'OK: gen={d.get(\"generation\")}')"
   ```

5. **Report**:
   ```
   ## Island Deployment Report
   | Space | Status | Generation | Engine Hash |
   ```

## Constraints
- MUST use subtree push (never push full repo to HF)
- All spaces are on Nomos42 account (HF_TOKEN_3)
- Wait at least 60s between deployments to avoid rate limiting
- Verify each space comes back online before deploying next
