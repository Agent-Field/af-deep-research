<p align="center">
  <h1 align="center">Deep Research</h1>
  <p align="center">Autonomous research that keeps digging until the question is actually answered.</p>
</p>
<p align="center">
  <a href="https://github.com/Agent-Field/af-deep-research/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://github.com/Agent-Field/agentfield"><img src="https://img.shields.io/badge/built%20on-AgentField-8A2BE2" alt="Built on AgentField"></a>
  <a href="https://discord.com/invite/aBHaXMkpqh"><img src="https://img.shields.io/badge/Discord-Join-7289da" alt="Discord"></a>
</p>

---

Most research tools search once and summarize. This one searches, finds gaps, spawns more agents to fill them, and repeats until coverage is complete. You get structured data back—entities, relationships, sources—not a text blob.

```bash
curl -X POST localhost:8080/api/v1/execute/async/meta_deep_research.execute_deep_research \
  -d '{"input": {"query": "Who are Tesla main competitors and how are they positioned?"}}'
```

```json
{
  "entities": [
    {"name": "Tesla", "type": "Company", "summary": "Leading US EV maker, $800B market cap"},
    {"name": "BYD", "type": "Company", "summary": "Chinese EV giant, overtook Tesla in Q4 2023"},
    {"name": "Rivian", "type": "Company", "summary": "US EV startup, trucks and SUVs focus"}
  ],
  "relationships": [
    {"source": "BYD", "target": "Tesla", "type": "Overtook_In_Sales"},
    {"source": "Rivian", "target": "Amazon", "type": "Delivery_Van_Deal"}
  ],
  "key_findings": [
    "BYD outsold Tesla globally in Q4 2023",
    "Legacy automakers investing $50B+ in EV transition"
  ],
  "document": "# Tesla Competitive Landscape\n\n## Executive Summary..."
}
```

---

## How it differs

| | This | ChatGPT / Perplexity |
|---|---|---|
| **Process** | Iterative—finds gaps, researches more | Single pass |
| **Output** | Entities + relationships + document | Text summary |
| **Integration** | REST API, SSE streaming | Chat window |
| **Hosting** | Self-host with Docker | SaaS only |
| **Agents** | Thousands spawned dynamically | None |

---

## Try it

```bash
git clone https://github.com/Agent-Field/af-deep-research.git && cd af-deep-research
cp .env.example .env
docker-compose -f docker-compose.hub.yml up -d
```

```bash
curl -X POST http://localhost:8080/api/v1/execute/async/meta_deep_research.execute_deep_research \
  -H "Content-Type: application/json" \
  -d '{"input": {"query": "What companies are most exposed to rising interest rates?"}}'
```

Returns `execution_id`. Stream progress via SSE. Grab results when done.

---

## Stream progress

```bash
curl -N http://localhost:8080/api/ui/v1/workflows/{run_id}/notes/events
```

```
data: {"message": "Spawning research agents..."}
data: {"message": "Found 34 articles"}
data: {"message": "Extracting: 12 companies, 8 executives"}
data: {"message": "Gap: missing Q4 data. Researching..."}
data: {"message": "Building relationship graph..."}
```

Build your own frontend on top.

---

## Parameters

| Parameter | Effect |
|-----------|--------|
| `research_focus` | Depth (1-5) |
| `research_scope` | Breadth (1-5) |
| `max_research_loops` | Iteration cycles |
| `tension_lens` | `balanced` / `bull` / `bear` |

Set `tension_lens: "bear"` for risk-focused analysis.

---

## Run locally

```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434
DEFAULT_MODEL=ollama/llama3.2
```

No API calls leaving your network. No telemetry.

<details>
<summary>Models</summary>

| Model | Cost |
|-------|------|
| `deepseek-chat-v3.1` | $0.15/0.75 per 1M |
| `claude-sonnet-4` | $3/$15 |
| `ollama/llama3.2` | Free |
| `ollama/qwen2.5:72b` | Free |

</details>

---

## Stack

Built on [AgentField](https://github.com/Agent-Field/agentfield)—infrastructure for production AI agents.

| Feature | |
|---------|---|
| Long workflows | 16+ minutes, no timeouts |
| Streaming | SSE to your frontend |
| Audit trails | Cryptographic verification |
| Scaling | Horizontal across nodes |

<p align="center">
  <a href="https://github.com/Agent-Field/agentfield">
    <img src="https://img.shields.io/badge/Powered%20by-AgentField-8A2BE2?style=for-the-badge" alt="AgentField">
  </a>
</p>

---

## Examples

| Use case | Query |
|----------|-------|
| Due diligence | `"Risks of investing in Rivian?"` with `tension_lens: "bear"` |
| Competitive intel | `"How is AMD positioning against NVIDIA?"` |
| Market research | `"What's driving the weight loss drug market?"` |

---

## Contribute

PRs welcome. Join [Discord](https://discord.com/invite/aBHaXMkpqh).

---

Apache 2.0

<p align="center">
  <a href="https://github.com/Agent-Field/af-deep-research">Star this repo if it saves you research time</a><br>
  <sub>Built by <a href="https://agentfield.ai">AgentField</a></sub>
</p>
