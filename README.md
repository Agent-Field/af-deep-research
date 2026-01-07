<p align="center">
  <h1 align="center">Deep Research</h1>
  <p align="center">
    Ask a question. Get a research report with entities, relationships, and sources.
  </p>
</p>

<p align="center">
  <a href="https://github.com/Agent-Field/af-deep-research/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://github.com/Agent-Field/agentfield"><img src="https://img.shields.io/badge/built%20on-AgentField-8A2BE2" alt="Built on AgentField"></a>
</p>

---

```bash
curl -X POST localhost:8080/api/v1/execute/async/meta_deep_research.execute_deep_research \
  -d '{"input": {"query": "Who are Tesla competitors and what is their market position?"}}'
```

```json
{
  "entities": [
    {"name": "Tesla", "type": "Company", "summary": "Leading US EV maker, $800B market cap"},
    {"name": "BYD", "type": "Company", "summary": "Chinese EV giant, overtook Tesla in Q4 2023 sales"},
    {"name": "Rivian", "type": "Company", "summary": "US EV startup, focus on trucks and SUVs"},
    {"name": "Ford", "type": "Company", "summary": "Legacy automaker, $7B EV investment"}
  ],
  "relationships": [
    {"source": "BYD", "target": "Tesla", "type": "Overtook_In_Sales"},
    {"source": "Ford", "target": "SK_Innovation", "type": "Battery_Partnership"},
    {"source": "Rivian", "target": "Amazon", "type": "Delivery_Van_Deal"}
  ],
  "key_findings": [
    "BYD outsold Tesla globally in Q4 2023, primarily driven by China market",
    "Legacy automakers investing $50B+ collectively in EV transition",
    "Tesla's margin advantage shrinking as competition intensifies"
  ],
  "document": "# Tesla Competitive Landscape\n\n## Executive Summary..."
}
```

That's real output. Entities extracted, relationships mapped, sources cited.

---

## How it works

You send a query. The system spawns thousands of parallel reasoning agents—entity extractors, relationship mappers, evidence gatherers, gap detectors. When something's missing, more agents spin up to find it. Three iterative cycles later, you get structured data back.

![Workflow Execution](assets/workflow-execution.png)

One query. 57 entities. 351 relationships. 93 sources. Done in 16 minutes.

---

## Try it

```bash
git clone https://github.com/Agent-Field/af-deep-research.git && cd af-deep-research
cp .env.example .env  # add your API keys
docker-compose -f docker-compose.hub.yml up -d
```

Then:

```bash
curl -X POST http://localhost:8080/api/v1/execute/async/meta_deep_research.execute_deep_research \
  -H "Content-Type: application/json" \
  -d '{"input": {"query": "What companies are most exposed to rising interest rates?"}}'
```

You'll get an `execution_id`. Stream progress via SSE, grab results when it's done.

---

## What comes back

Every query returns:

- **Entities** — companies, people, products, concepts. Each with a summary.
- **Relationships** — who competes with whom, who acquired what, who partners with who.
- **Evidence** — facts and quotes traced back to source articles.
- **Document** — a cited research report you can actually use.

It's structured. Feed it into your app, your database, your dashboard. Build on it.

---

## Stream it

```bash
curl -N http://localhost:8080/api/ui/v1/workflows/{run_id}/notes/events
```

```
data: {"message": "Spawning research agents..."}
data: {"message": "Found 34 relevant articles"}
data: {"message": "Extracting entities: 12 companies, 8 executives"}
data: {"message": "Gap detected: missing Q4 earnings data. Researching..."}
data: {"message": "Building relationship graph..."}
data: {"message": "Done. 47 entities, 156 relationships."}
```

Build your own UI on top. Show users what's happening. It's just SSE.

---

## Tune it

| Parameter | What it does |
|-----------|--------------|
| `research_focus` | Depth. 1 = quick scan, 5 = deep dive |
| `research_scope` | Breadth. 1 = narrow, 5 = wide net |
| `max_research_loops` | How many iterative cycles |
| `tension_lens` | `balanced`, `bull`, or `bear` perspective |

Want bearish analysis on a stock? Set `tension_lens: "bear"`. The system will dig for risks, red flags, counter-arguments.

---

## Run it locally

Don't want API calls leaving your network? Use Ollama.

```bash
# .env
OLLAMA_BASE_URL=http://host.docker.internal:11434
DEFAULT_MODEL=ollama/llama3.2
```

No telemetry. No phone home. Your data stays yours.

<details>
<summary>Model options</summary>

| Model | Cost | Notes |
|-------|------|-------|
| `deepseek-chat-v3.1` | $0.15/0.75 per 1M | Default. Solid. |
| `claude-sonnet-4` | $3/$15 | Better synthesis |
| `ollama/llama3.2` | Free | Local |
| `ollama/qwen2.5:72b` | Free | Best local quality |

</details>

---

## The stack

Built on [AgentField](https://github.com/Agent-Field/agentfield) — open-source infrastructure for production AI agents.

- Long-running workflows (16+ minutes, no timeouts)
- SSE streaming to your frontend
- Async execution with webhooks
- Cryptographic audit trails if you need compliance

<p align="center">
  <a href="https://github.com/Agent-Field/agentfield">
    <img src="https://img.shields.io/badge/Powered%20by-AgentField-8A2BE2?style=for-the-badge" alt="Powered by AgentField">
  </a>
</p>

---

## Use cases

<details>
<summary>Due diligence</summary>

```json
{"query": "What are the risks of investing in Rivian?", "tension_lens": "bear"}
```
→ Risk factors, competitive threats, cash burn analysis, supplier dependencies
</details>

<details>
<summary>Competitive intel</summary>

```json
{"query": "How is AMD positioning against NVIDIA in the AI chip market?"}
```
→ Product comparisons, customer wins, pricing strategies, partnership map
</details>

<details>
<summary>Market research</summary>

```json
{"query": "What's driving growth in the weight loss drug market?"}
```
→ Key players (Novo Nordisk, Eli Lilly), pipeline drugs, insurance coverage trends
</details>

---

## Contribute

Using this for something interesting? Open a PR, file an issue, or come hang out:

**[Discord](https://discord.com/invite/aBHaXMkpqh)**

---

## License

Apache 2.0

---

<p align="center">
  <sub>Built by <a href="https://agentfield.ai">AgentField</a></sub>
</p>
