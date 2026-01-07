<p align="center">
  <h1 align="center">AF Deep Research</h1>
  <p align="center">Autonomous research backend for AI applications.</p>
</p>
<p align="center">
  <a href="https://github.com/Agent-Field/af-deep-research/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://github.com/Agent-Field/agentfield"><img src="https://img.shields.io/badge/built%20on-AgentField-8A2BE2" alt="Built on AgentField"></a>
  <a href="https://discord.com/invite/aBHaXMkpqh"><img src="https://img.shields.io/badge/Discord-Join-7289da" alt="Discord"></a>
</p>
<p align="center"><b>Early Preview</b> — APIs may change. Feedback welcome.</p>

---

Most research tools search once and summarize. You've used them. You know the pattern: ask a question, get a paragraph, move on. But real research doesn't work that way. Real research finds something, realizes what's missing, goes back, digs deeper, connects dots across sources.

That's what this does. You send a query. The system spawns thousands of parallel agents—entity extractors, relationship mappers, evidence gatherers—all working simultaneously. When gaps are found, more agents spin up to fill them. Three iterative cycles later, you get structured data back: typed entities, mapped relationships, traced evidence, and a cited document.

It's not a chatbot. It's an [AI backend](https://www.agentfield.ai/blog/posts/ai-backend)—infrastructure that returns structured data for your applications.

## How it works

**You call the API:**

```bash
curl -X POST http://localhost:8080/api/v1/execute/async/meta_deep_research.execute_deep_research \
  -H "Content-Type: application/json" \
  -d '{"input": {"query": "What companies are investing in AI chips?"}}'
```

**You get back structured research:**

**Entities** — typed and summarized. Companies, investors, executives, technologies, market trends.

```json
{
  "entities": [
    {"name": "NVIDIA", "type": "Company", "summary": "Dominant AI chip maker, 80%+ datacenter GPU share"},
    {"name": "Jensen Huang", "type": "Founder", "summary": "CEO of NVIDIA since 1993"},
    {"name": "Sequoia Capital", "type": "Investor", "summary": "Early NVIDIA investor, major AI fund"},
    {"name": "H100", "type": "Technology", "summary": "Flagship AI training GPU, $30k+ ASP"},
    {"name": "AI Chip Market", "type": "Market_Trend", "summary": "Projected $300B by 2030"}
  ]
}
```

**Relationships** — who competes with whom, who invested in what, who acquired who. The kind of connections that take hours to map manually.

```json
{
  "relationships": [
    {"source": "AMD", "target": "NVIDIA", "type": "Competes_With", "description": "Direct competition in AI accelerators"},
    {"source": "Sequoia Capital", "target": "NVIDIA", "type": "Invests_In", "description": "Series A investor, 1993"},
    {"source": "NVIDIA", "target": "Mellanox", "type": "Acquires", "description": "$7B acquisition for networking"},
    {"source": "Microsoft", "target": "OpenAI", "type": "Partners_With", "description": "$10B strategic investment"}
  ]
}
```

**Evidence** — every claim traced to source. Facts extracted. Quotes captured. Nothing hallucinated.

```json
{
  "article_evidence": [{
    "article_id": 1,
    "facts": [
      "NVIDIA datacenter revenue reached $18.4B in Q4",
      "H100 backlog extends into late 2025"
    ],
    "quotes": ["We are seeing unprecedented demand — Jensen Huang"]
  }]
}
```

**Document** — a hierarchical research report with sections, citations, and bibliography. If you just want to read something, this is ready to go.

```json
{
  "document_title": "AI Chip Investment Landscape",
  "executive_summary": "The AI chip market is experiencing...",
  "sections": [
    {"title": "Market Dynamics", "content": "The accelerator market reached $45B [1]..."},
    {"title": "Competitive Landscape", "content": "NVIDIA maintains 80%+ share [2]..."}
  ],
  "source_notes": [
    {"citation_id": 1, "title": "AI Chip Report 2024", "domain": "reuters.com"}
  ]
}
```

**Metadata** — quality scores, timing, iteration counts. Use these for quality gates in your pipeline.

```json
{
  "metadata": {
    "iterations_completed": 3,
    "total_entities": 47,
    "total_relationships": 156,
    "total_sources": 89,
    "final_quality_score": 0.82
  }
}
```

## Why this exists

There's a gap between "ask ChatGPT" and "hire a research analyst." ChatGPT gives you a paragraph in seconds but hallucinates and doesn't cite. An analyst gives you a 20-page report with sources but takes days and costs thousands.

This sits in the middle. It runs for 15 minutes instead of 15 seconds, but it iterates, it cites, it structures. And because it's an API, you can embed it in your own applications—due diligence tools, competitive intelligence dashboards, market research pipelines.

| | AF Deep Research | ChatGPT / Perplexity |
|---|---|---|
| Process | Iterative (finds gaps, researches more) | Single pass |
| Entities | Typed (Company, Investor, Technology) | None |
| Relationships | Mapped (Competes_With, Invests_In) | None |
| Evidence | Facts + quotes with source URLs | Basic citations |
| Document | Hierarchical sections + bibliography | Flat text |
| Integration | REST API + SSE streaming | Chat window |
| Hosting | Self-host, local LLMs, air-gapped | SaaS only |

## Getting started

```bash
git clone https://github.com/Agent-Field/af-deep-research.git && cd af-deep-research
cp .env.example .env
docker-compose -f docker-compose.hub.yml up -d
```

You get back an `execution_id`. Stream progress via SSE while it runs, then fetch the results when it's done.

## Build on it

This is an API. Pull exactly the parts you need:

- `response.research_package.entities` — pipe to your graph database
- `response.research_package.relationships` — feed to Neo4j
- `response.research_package.sections` — render in your UI
- `response.research_package.source_notes` — show bibliography
- `response.metadata` — use quality scores as gates

The SSE stream lets you build real-time UIs. Show users the research happening: "Found 34 articles... Extracting entities... Gap detected, researching funding data..."

```bash
curl -N http://localhost:8080/api/ui/v1/workflows/{run_id}/notes/events
```

## Parameters

Control depth, breadth, and perspective:

| Parameter | What it does |
|-----------|--------------|
| `research_focus` | Depth (1-5). Higher goes deeper. |
| `research_scope` | Breadth (1-5). Higher casts wider net. |
| `max_research_loops` | How many iterative cycles to run. |
| `tension_lens` | `balanced`, `bull`, or `bear` perspective. |
| `source_strictness` | `strict`, `mixed`, or `permissive` source filtering. |

Set `tension_lens: "bear"` when you want the system to dig for risks and red flags. Set `source_strictness: "strict"` to filter to reputable sources only.

## Run locally

Don't want API calls leaving your network? Point at Ollama:

```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434
DEFAULT_MODEL=ollama/llama3.2
```

No telemetry. No phone home. Your queries and your data stay on your infrastructure.

<details>
<summary>Model options</summary>

| Model | Cost |
|-------|------|
| `deepseek-chat-v3.1` | $0.15/0.75 per 1M tokens |
| `claude-sonnet-4` | $3/$15 per 1M tokens |
| `ollama/llama3.2` | Free (local) |
| `ollama/qwen2.5:72b` | Free (local) |

</details>

## The stack

AF Deep Research runs on [AgentField](https://github.com/Agent-Field/agentfield), open-source infrastructure for production AI agents. That's what makes the long-running, multi-agent orchestration possible. Workflows run for 16+ minutes without timeout. Progress streams via SSE. Results persist. Audit trails are cryptographically signed if you need compliance.

<p align="center">
  <a href="https://github.com/Agent-Field/agentfield"><img src="https://img.shields.io/badge/Powered%20by-AgentField-8A2BE2?style=for-the-badge" alt="AgentField"></a>
</p>

## Examples

A few queries to try:

| Use case | Query |
|----------|-------|
| Due diligence | `"What are the risks of investing in Rivian?"` with `tension_lens: "bear"` |
| Competitive intel | `"How is AMD positioning against NVIDIA in AI chips?"` |
| Market research | `"What's driving growth in the weight loss drug market?"` |

## Links

- **Docs** — [agentfield.ai/docs](https://agentfield.ai/docs)
- **GitHub** — [Agent-Field/agentfield](https://github.com/Agent-Field/agentfield)
- **Discord** — [Join the community](https://discord.com/invite/aBHaXMkpqh)

## Contribute

This is an early preview. We're actively developing and want feedback. File issues, open PRs, or come chat in [Discord](https://discord.com/invite/aBHaXMkpqh).

<br>

<p align="center">
  <a href="https://github.com/Agent-Field/af-deep-research">Star if this saves you research time</a><br>
  <sub>Built by <a href="https://agentfield.ai">AgentField</a> · <a href="https://github.com/Agent-Field/af-deep-research/blob/main/LICENSE">Apache 2.0</a></sub>
</p>
