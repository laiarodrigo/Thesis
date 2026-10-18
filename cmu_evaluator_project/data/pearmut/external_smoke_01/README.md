# External Pearmut smoke test

This disposable campaign verifies the real public-browser path before any
Pool 1 link is sent. `SMOKE-FRIEND-01` is not a Pool 1 annotator and every
record from this campaign is permanently ineligible for DEV or TEST gold.

- Campaign ID: `ted2020_pilot_external_smoke_01_cesa`
- Public campaign: `campaigns/ted2020_pilot_external_smoke_01_cesa.json`
- Purpose: `external_web_smoke_test`
- Gold eligible: `false`
- Disposable: `true`

The campaign contains the established 10-item pilot and four canonical
systems, but no Rodrigo answers. Its private mapping and assignment manifest
remain in the Git-ignored local `private/` directory. The Pool 1 manifest and
allocation are not used.

## Real hosted path

The actual friend test runs continuously on Railway, behind the annotation-only
proxy and backed by a Railway persistent volume. Follow the complete setup,
URL, mandatory restart/redeploy persistence check, retrieval, normalization,
backup, and restore instructions in
[`deployment/railway/README.md`](../../../deployment/railway/README.md).
Rodrigo's laptop may be completely off after deployment.

## Local-only debugging path

The old localhost and Cloudflare Quick Tunnel helpers remain only for debugging.
They are **not** the hosted campaign and stop working when this laptop sleeps,
closes, loses connectivity, or turns off.

Start localhost Pearmut/proxy in one terminal:

```bash
bash cmu_evaluator_project/scripts/start_external_smoke_pearmut.sh
```

Optionally start a temporary Quick Tunnel in another terminal:

```bash
export SMOKE_ALLOWED_EMAIL='<friend-email>'
bash cmu_evaluator_project/scripts/start_external_smoke_tunnel.sh
```

Use those commands only to reproduce a local browser problem. Never distribute
their temporary URL as the production annotation link.
