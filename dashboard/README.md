# RiskPulse Operations

An interactive fraud-operations dashboard for RiskPulse. It gives analysts a focused view of scoring volume, alert traffic, review-queue activity, API latency, and model drift.

The current release uses an explicitly labeled demo stream so the interface can be evaluated independently of a running RiskPulse API. Analyst actions update the local review queue and transaction detail view.

[Open the live dashboard](https://riskpulse-operations.yasserj.chatgpt.site)

## Run locally

```bash
npm run install:ci
npm run dev
```

## Validate

```bash
npm run build
node --test tests/*.test.mjs
npm run lint
npm run deploy:cloudflare -- --dry-run
```

## Deploy free on Cloudflare

The dashboard builds to a Cloudflare Worker and can use the free
`*.workers.dev` address assigned to your Cloudflare account.

For a manual deployment after signing in with Wrangler:

```bash
npm run deploy
```

For Cloudflare's GitHub integration, import this repository and configure:

- Root directory: `dashboard`
- Build command: `npm run build`
- Deploy command: `npm run deploy:cloudflare`
- Worker name: `riskpulse-operations`

Cloudflare then deploys updates from the production branch automatically.

## Production integration

The next integration step is to replace the demo stream with the RiskPulse scoring, review, metrics, and drift endpoints while retaining the same dashboard states and interactions.
