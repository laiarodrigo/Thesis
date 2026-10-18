# Protected Pearmut on Railway Free with automatic wake-up

This is the production-like external annotation path:

```text
Railway HTTPS domain
  -> protective annotation-only proxy on Railway $PORT
  -> Pearmut on container loopback 127.0.0.1:8001
  -> PEARMUT_ROOT=/pearmut-volume/pearmut
  -> Railway persistent volume mounted at /pearmut-volume
```

It does not use Rodrigo's laptop or a Cloudflare Quick Tunnel. Once Railway
has deployed it, the laptop may be shut down completely. Railway Serverless
may sleep the container while nobody is annotating and wakes it on public
traffic. A separate static GitHub Pages launcher hides that cold start from
annotators, polls the nonsensitive health endpoint, and redirects only after
Pearmut is ready. Use one service and one replica. Do not enable a TCP proxy,
do not expose port 8001, and do not upload checkpoints or source corpora.

The image contains Pearmut 1.1.5, the filtering proxy, the completion-only
per-user backup exporter, and only the public campaign JSON files. Its
`.railwayignore` upload allow list and
`.dockerignore` build allow list exclude private rosters/mappings and all
generation/model data. The first runtime campaign is
only `ted2020_pilot_external_smoke_01_cesa`; its sole user is
`SMOKE-FRIEND-01`.

## Exact setup after `railway login`

Run every command in this section from the repository root:

```bash
cd ~/repos/Thesis
railway whoami
```

Create and link one empty project/service, then attach its persistent volume:

```bash
railway init --name cmu-pearmut
railway add --service pearmut-annotations
railway service link pearmut-annotations
railway volume add --mount-path /pearmut-volume
railway status
railway volume list
```

The volume list must show a volume attached to `pearmut-annotations` at exactly
`/pearmut-volume`. Do not continue with another mount path.

Create a retained local backup-signing key. This file is Git-ignored. Keep it:
losing it prevents creation of new valid personal links, while rotating the
Railway value invalidates every link already sent.

```bash
install -d -m 700 cmu_evaluator_project/runtime/secrets
umask 077
python -c 'import secrets; print(secrets.token_urlsafe(32))' > \
  cmu_evaluator_project/runtime/secrets/pearmut_backup_signing_key
PEARMUT_BACKUP_KEY_FILE="$PWD/cmu_evaluator_project/runtime/secrets/pearmut_backup_signing_key"
```

Set all application variables without triggering an incomplete deployment.
The Pearmut admin secret is generated directly into Railway and is not printed
or stored locally:

```bash
railway variable set --skip-deploys \
  PEARMUT_ROOT=/pearmut-volume/pearmut \
  PEARMUT_CAMPAIGN_IDS=ted2020_pilot_external_smoke_01_cesa

python -c 'import secrets; print(secrets.token_urlsafe(32))' | \
  railway variable set --skip-deploys --stdin PEARMUT_TOKEN_MAIN

railway variable set --skip-deploys --stdin PEARMUT_BACKUP_SIGNING_KEY \
  < "$PEARMUT_BACKUP_KEY_FILE"
```

Railway supplies `PORT`, `RAILWAY_VOLUME_MOUNT_PATH=/pearmut-volume`, and
`RAILWAY_VOLUME_NAME`. Never define or override those variables.

Open the linked dashboard:

```bash
railway open
```

Before the first deploy, apply this service checklist:

1. **Settings -> Deploy -> Healthcheck Path:** `/healthz`.
2. **Settings -> Deploy -> Serverless:** enabled. This setting applies when a
   new container is deployed; redeploy if it was enabled afterward.
3. **Settings -> Deploy -> Restart Policy:** `Always`.
4. **Settings -> Scale:** one replica in one region; Free-plan limits are fine.
5. Leave build/start commands empty: the Dockerfile supplies both.
6. Do not add a pre-deploy command; the volume exists only at runtime.
7. Do not create a TCP proxy or expose Pearmut's internal port 8001.
8. In workspace **Usage**, set an alert and a hard limit at the available free
   credit. Do not configure an external uptime monitor.

The container launcher verifies both Railway's volume variable and the Linux mount
table, probes volume writability, and fails before starting Pearmut if the
volume is absent. It also refuses any `PEARMUT_ROOT` other than the exact path
above.

## Deploy and obtain HTTPS

From the repository root, with the Railway project/service linked:

```bash
railway up cmu_evaluator_project --path-as-root \
  --service pearmut-annotations
```

Railway uses `cmu_evaluator_project/Dockerfile`; its exact container command
is:

```text
/app/scripts/start_railway_pearmut.sh
```

Check the deployment logs and health, then generate the Railway-provided HTTPS
domain:

```bash
railway logs --service pearmut-annotations --lines 100
railway domain --service pearmut-annotations
railway domain list --service pearmut-annotations
```

The dashboard equivalent is service **Settings -> Networking -> Public
Networking -> Generate Domain**. No custom domain is required. If the listed
hostname is `<HOST>.up.railway.app`, set it once in the current shell and check
the protected surface:

```bash
PEARMUT_HOST="https://<HOST>.up.railway.app"
curl --retry 20 --retry-all-errors --retry-delay 3 --fail \
  "$PEARMUT_HOST/healthz"
curl -i "$PEARMUT_HOST/dashboard"
curl -i "$PEARMUT_HOST/download-annotations"
```

The first command must return `ok`; both protected research requests must
return HTTP 403.

## Publish and use the automatic launcher

The repository is public and the generic launcher file contains no campaign,
participant, annotation, or secret data. Publish that one file once:

```bash
git add docs/cmu-evaluator-launcher/index.html
git commit --only docs/cmu-evaluator-launcher/index.html \
  -m "Add Pearmut annotation wake launcher"
git push origin main
```

Then enable GitHub Pages:

1. Push `docs/cmu-evaluator-launcher/index.html` to the repository's `main`
   branch.
2. Open GitHub repository **Settings -> Pages**.
3. Under **Build and deployment**, select **Deploy from a branch**, branch
   `main`, folder `/docs`, and click **Save**.
4. Wait until
   `https://laiarodrigo.github.io/Thesis/cmu-evaluator-launcher/` loads.

After Railway gives you `<HOST>.up.railway.app`, generate the exact friend
link locally:

```bash
python cmu_evaluator_project/scripts/make_pearmut_launcher_url.py \
  "$PEARMUT_HOST" \
  ted2020_pilot_external_smoke_01_cesa \
  SMOKE-FRIEND-01 \
  --signing-key-file "$PEARMUT_BACKUP_KEY_FILE"
```

To retain it in the current shell and print the exact URL:

```bash
SMOKE_URL="$(python cmu_evaluator_project/scripts/make_pearmut_launcher_url.py \
  "$PEARMUT_HOST" \
  ted2020_pilot_external_smoke_01_cesa \
  SMOKE-FRIEND-01 \
  --signing-key-file "$PEARMUT_BACKUP_KEY_FILE")"
printf '%s\n' "$SMOKE_URL"
```

Send the friend only the printed GitHub Pages launcher URL. Do not send the
raw form below as the invitation link; it is shown only to document the final
redirect target:

```text
https://<HOST>.up.railway.app/annotate?campaign_id=ted2020_pilot_external_smoke_01_cesa&user_id=SMOKE-FRIEND-01#backup_token=<SCOPED_SIGNATURE>
```

The scoped signature is stored after `#`, so the browser never sends it to
GitHub Pages, Railway, Pearmut, or their access logs. The annotator can open
the launcher at any time. It wakes Railway, retries for
up to two minutes, and redirects automatically when `/healthz` returns `ok`.
No researcher action or manual browser refresh is required. The health route
allows cross-origin reads because it returns only `ok` or an availability
error; annotation and research routes do not receive cross-origin access.

Do not send A01-A08 links. The start script loads a campaign only when both
its persistent campaign file and progress entry are absent. If it is already
loaded, it changes nothing. Inconsistent/orphan state causes startup to fail
rather than overwrite annotations.

## Allowed and blocked public surface

The public Railway port belongs only to the protective proxy. Raw Pearmut is
bound to `127.0.0.1:8001` and is not published. Annotators can fetch the
annotation page/static assets and call only `get-next-item`, `get-i-item`,
`log-response`, and the capability-protected `annotator-backup`. The proxy
returns 403 for everything else, including:

```text
/download-annotations  /download-campaigns  /download-progress
/dashboard             /dashboard/*         /dashboard-data
/dashboard-results     /export-results      /add-campaign
/reset-task            /purge-campaign
```

`/healthz` is a non-sensitive proxy health endpoint and reports healthy only
when loopback Pearmut also responds.

## Completion-only personal backup

When Pearmut displays **All done**, the protected proxy adds a **Download
backup** button. It is optional and does not replace researcher retrieval from
the Railway volume. The downloaded ZIP contains exactly:

```text
annotations.jsonl  only raw rows whose user_id matches this signed link
progress.json      only {campaign_id: {user_id: that user's progress}}
metadata.json      campaign_id, user_id, assignment_id, passage ID(s),
                   UTC export time, purpose, eligibility, and row count
```

It contains no other annotator, real name, roster, private label-to-system
mapping, campaign JSON, or researcher token. Smoke metadata is explicitly
`eligible_for_gold: false`. The endpoint rejects unfinished assignments and
uses an HMAC capability bound to the exact `campaign_id + user_id`; an A01
token cannot request A02. Tokens are read-only but should still be distributed
only through the intended personal launcher URL. General Pearmut download and
admin endpoints remain blocked.

Filenames are deterministic and readable, for example:

```text
A01_pilot_screening_backup.zip
A01_test_ted2020-1234_s1-s10_backup.zip
SMOKE-FRIEND-01_external_smoke_test_backup.zip
```

The persistent Railway JSONL and progress files remain authoritative. The
annotator ZIP is only a personal recovery copy.

## Mandatory smoke and restart/persistence check

Before any A01-A08 link is sent:

1. Open the printed `SMOKE_URL` in a browser. If Railway is sleeping, the
   launcher must display its starting message and redirect automatically.
2. Complete all 10 disposable smoke items. The completion page must display
   **Download backup**. Click it and confirm the browser downloads
   `SMOKE-FRIEND-01_external_smoke_test_backup.zip`. Its `metadata.json` must
   say `eligible_for_gold: false`; its JSONL contains only
   `SMOKE-FRIEND-01`. This copy is optional—the volume remains authoritative.
3. Inspect the server logs. Query strings and backup request bodies must not be
   printed:

   ```bash
   railway logs --service pearmut-annotations --lines 150
   ```

   Expected startup lines include `Protected Pearmut is ready` and either
   `Loading absent campaign ...` on the first start or `already loaded;
   preserving all runtime state` afterward.
4. Retrieve authoritative snapshot 1:

   ```bash
   CAMPAIGN_ID=ted2020_pilot_external_smoke_01_cesa
   bash cmu_evaluator_project/scripts/download_hosted_pearmut_campaign.sh \
     "$CAMPAIGN_ID"

   SNAPSHOT_ONE="$(find cmu_evaluator_project/runtime/hosted_downloads \
     -mindepth 1 -maxdepth 1 -type d -name "${CAMPAIGN_ID}_*" \
     -printf '%p\n' | sort | tail -n 1)"
   RAW_ONE="$SNAPSHOT_ONE/state/data/outputs/$CAMPAIGN_ID.jsonl"
   PROGRESS_ONE="$SNAPSHOT_ONE/state/data/progress.json"
   sha256sum "$RAW_ONE" "$PROGRESS_ONE"

   python - "$RAW_ONE" "$PROGRESS_ONE" "$CAMPAIGN_ID" <<'PY'
   import json
   import sys
   from pathlib import Path

   raw_path, progress_path, campaign_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
   rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
   progress = json.loads(progress_path.read_text(encoding="utf-8"))[campaign_id]
   print("raw rows:", len(rows))
   print("users:", sorted({row["user_id"] for row in rows}))
   print("progress:", json.dumps(progress, indent=2))
   PY
   ```

   This must print 10 raw rows, only `SMOKE-FRIEND-01`, and ten completed
   progress entries.
5. Restart only the Railway service, wait for health, and confirm the loader
   preserved state:

   ```bash
   railway service restart --service pearmut-annotations --yes
   curl --retry 20 --retry-all-errors --retry-delay 3 --fail \
     "$PEARMUT_HOST/healthz"
   railway logs --service pearmut-annotations --lines 100
   ```

6. Retrieve snapshot 2 and compare exact raw rows plus only this campaign's
   progress entry:

   ```bash
   bash cmu_evaluator_project/scripts/download_hosted_pearmut_campaign.sh \
     "$CAMPAIGN_ID"

   SNAPSHOT_TWO="$(find cmu_evaluator_project/runtime/hosted_downloads \
     -mindepth 1 -maxdepth 1 -type d -name "${CAMPAIGN_ID}_*" \
     -printf '%p\n' | sort | tail -n 1)"
   RAW_TWO="$SNAPSHOT_TWO/state/data/outputs/$CAMPAIGN_ID.jsonl"
   PROGRESS_TWO="$SNAPSHOT_TWO/state/data/progress.json"
   cmp "$RAW_ONE" "$RAW_TWO"

   python - "$PROGRESS_ONE" "$PROGRESS_TWO" "$CAMPAIGN_ID" <<'PY'
   import json
   import sys
   from pathlib import Path

   first, second, campaign_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
   a = json.loads(first.read_text(encoding="utf-8"))[campaign_id]
   b = json.loads(second.read_text(encoding="utf-8"))[campaign_id]
   if a != b:
       raise SystemExit("ERROR: campaign progress changed across restart")
   print("restart persistence: PASS")
   PY
   ```

7. Test Serverless sleep/wake. Make no requests for at least twelve minutes
   (Railway currently documents a five-to-ten-minute inactivity window).
   Then open the same `SMOKE_URL`; the launcher must wake and redirect without
   manual refresh, Pearmut must remain on the completed screen, and the backup
   button must still work. Retrieve once more and compare:

   ```bash
   bash cmu_evaluator_project/scripts/download_hosted_pearmut_campaign.sh \
     "$CAMPAIGN_ID"

   SNAPSHOT_THREE="$(find cmu_evaluator_project/runtime/hosted_downloads \
     -mindepth 1 -maxdepth 1 -type d -name "${CAMPAIGN_ID}_*" \
     -printf '%p\n' | sort | tail -n 1)"
   RAW_THREE="$SNAPSHOT_THREE/state/data/outputs/$CAMPAIGN_ID.jsonl"
   PROGRESS_THREE="$SNAPSHOT_THREE/state/data/progress.json"
   cmp "$RAW_ONE" "$RAW_THREE"

   python - "$PROGRESS_ONE" "$PROGRESS_THREE" "$CAMPAIGN_ID" <<'PY'
   import json
   import sys
   from pathlib import Path

   first, third, campaign_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
   a = json.loads(first.read_text(encoding="utf-8"))[campaign_id]
   b = json.loads(third.read_text(encoding="utf-8"))[campaign_id]
   if a != b:
       raise SystemExit("ERROR: campaign progress changed across sleep/wake")
   print("sleep/wake persistence: PASS")
   PY
   ```

8. Redeploy once and repeat the health call plus snapshot comparison. This
   proves replacement of the entire container also preserves the volume:

   ```bash
   railway up cmu_evaluator_project --path-as-root \
     --service pearmut-annotations
   curl --retry 20 --retry-all-errors --retry-delay 3 --fail \
     "$PEARMUT_HOST/healthz"
   ```

9. Normalize the downloaded raw file with the local private manifest:

   ```bash
   python cmu_evaluator_project/scripts/normalize_pearmut_annotations.py \
     --candidate-dataset cmu_evaluator_project/data/prepared/pilot_01/combined/candidates.json \
     --assignment-manifest cmu_evaluator_project/data/pearmut/external_smoke_01/private/assignment_manifest_PRIVATE.json \
     --raw-export external_smoke_01__SMOKE-FRIEND-01="$RAW_THREE" \
     --output cmu_evaluator_project/runtime/hosted_downloads/smoke_normalized.json \
     --repeat-policy preserve --overwrite
   ```

All normalized smoke rows must remain `eligible_for_gold: false`. The smoke
campaign is disposable and never shares an ID, JSONL file, progress namespace,
or private mapping with Pool 1 or DEV.

## Download, backup, and restore

The normal researcher command is:

```bash
bash cmu_evaluator_project/scripts/download_hosted_pearmut_campaign.sh \
  <CAMPAIGN_ID>
```

It uses Railway's non-interactive volume-file API to download the persistent
campaign JSON, raw JSONL, and global progress into a new timestamped,
Git-ignored directory under `cmu_evaluator_project/runtime/hosted_downloads/`.
It validates them and creates a campaign-only `.tar.gz` plus SHA-256 file. The
archive adds the matching private assignment/system mapping from the local
repository, but excludes the real-name roster. Nothing is copied back into
curated `data/annotations/` automatically.

That same command is the researcher-triggered timestamped backup. In addition,
enable a **Daily** volume backup in the service's **Backups** tab during live
collection. Railway's native backup can restore the whole volume: select the
dated backup, click **Restore**, review the staged replacement volume, then
click **Deploy**. Railway retains the previous volume unmounted, making this
the preferred disaster-recovery route.

To restore one downloaded campaign archive instead:

1. In **Settings -> Scale -> Regions**, set the service's only region to zero
   replicas and wait until it is stopped. Never restore while Pearmut can write.
2. Run:

   ```bash
   bash cmu_evaluator_project/scripts/restore_hosted_pearmut_campaign.sh \
     <PRINTED_BACKUP.tar.gz> <CAMPAIGN_ID> --confirm-service-stopped
   ```

   The script validates the archive, downloads current global progress, merges
   only the chosen campaign entry (preserving other campaigns), uploads its
   campaign and JSONL, and uploads merged progress last.
3. Set the region back to one replica, wait for `/healthz`, open the annotator
   URL, and immediately make a fresh downloaded backup.

## Adding later campaigns safely

The image already packages the public Pool 1 pilot-screening campaign, but the
initial environment variable names **only** the smoke campaign, so A01-A08 are
not loaded or linked during this test. After the mandatory restart check and
the decision to begin production, append the production campaign IDs to the
comma-separated `PEARMUT_CAMPAIGN_IDS` value and redeploy. For example, the
pilot-screening ID is `ted2020_pilot_screening_pool_01_cesa`.

The TEST campaign must first be generated and reviewed locally. A subsequent
image build will package its public JSON from `data/pearmut/pool_01/campaigns/`;
only then may its ID be appended. Never replace an existing campaign JSON on a
live volume and never run a purge/reset operation. Campaign IDs remain the
storage namespaces, so each campaign has a distinct JSONL file and progress
entry.

## Cost and capacity

Start with Railway Free. It currently provides $1 monthly usage, one replica,
0.5 GB RAM, and a 0.5 GB volume. The packaged smoke campaign is about 42 KB and
the current screening campaign is about 305 KB, so storage is ample. Enabling
Serverless avoids paying for idle RAM/CPU and is the intended zero-cost path
for intermittent annotation sessions. Builds are not billed.

Do not configure an uptime monitor or scheduled health ping: it would prevent
sleeping and consume the free allowance. Monitor the Usage page and set the
hard usage limit to the available free credit. A cold service's first direct
request may fail while it wakes; this is why annotators receive the launcher
rather than the raw Railway URL. If later traffic exceeds the free allowance
or collection requires a formal availability target with no cold starts,
upgrade to Hobby; it currently has a $5 monthly minimum including $5 usage.

Current Railway references:

- <https://docs.railway.com/volumes>
- <https://docs.railway.com/cli/deploying>
- <https://docs.railway.com/cli/volume>
- <https://docs.railway.com/networking/domains/working-with-domains>
- <https://docs.railway.com/deployments/healthchecks>
- <https://docs.railway.com/deployments/serverless>
- <https://docs.railway.com/volumes/backups>
- <https://railway.com/pricing>
