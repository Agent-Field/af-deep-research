# AF Deep Research

End-to-end AI-powered deep research. One query, full document output with real-time progress streaming.

![Workflow Execution](assets/workflow-execution.png)
*A single research query spawns 170+ parallel reasoning steps - entity extraction, relationship mapping, evidence synthesis, and document generation.*

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT                                         │
│                                                                             │
│   1. POST /async → get workflow_id                                          │
│   2. GET /workflows/:id/notes/events → subscribe to SSE for this workflow   │
│   3. GET /executions/:id → fetch final result                               │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONTROL PLANE (:8080)                               │
│         Workflow orchestration • SSE streaming • Agent routing              │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DEEP RESEARCH AGENT (:8001)                            │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     execute_deep_research()                           │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Phase 1: Research Collection                                   │  │  │
│  │  │  • Query classification                                         │  │  │
│  │  │  • Multi-stream parallel web search                             │  │  │
│  │  │  • Evidence extraction from articles                            │  │  │
│  │  │  • Entity & relationship graph building                         │  │  │
│  │  │  • Gap analysis → targeted follow-up (iterative)                │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Phase 2: Document Generation                                   │  │  │
│  │  │  • Source filtering & adjudication                              │  │  │
│  │  │  • Perspective lens application                                 │  │  │
│  │  │  • Formatted document with citations                            │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OpenRouter or Ollama (LLM)  •  Jina AI / Tavily / Firecrawl / Serper (Search)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Option A: Using Docker Hub (Recommended)

No need to build - just pull and run:

```bash
# 1. Clone the repo (for config files only)
git clone https://github.com/Agent-Field/af-deep-research.git
cd af-deep-research

# 2. Configure API keys
cp .env.example .env
# Edit .env with your API keys

# 3. Start with pre-built image
docker-compose -f docker-compose.hub.yml up -d
```

### Option B: Build Locally

```bash
# 1. Clone the repo
git clone https://github.com/Agent-Field/af-deep-research.git
cd af-deep-research

# 2. Configure API keys
cp .env.example .env
# Edit .env with your API keys

# 3. Build and start
docker-compose up -d --build
```

### 1. Configure API Keys

```bash
cat > .env << 'EOF'
OPENROUTER_API_KEY=sk-or-your-key-here
JINA_API_KEY=jina_your-key-here
EOF
```

### 2. Start Services

For a fresh start (rebuilds images, removes old containers):

```bash
docker-compose down && docker-compose up -d --force-recreate
```

To watch startup logs in real-time:

```bash
docker-compose up
```

> **Tip**: Use `docker-compose up` without `-d` to see logs directly. Press `Ctrl+C` to stop.

### 2.5. Verify Services Are Ready

The agent takes a moment to register with the control plane. Verify it's ready:

```bash
# Check container status
docker-compose ps

# Or visit the UI
open http://localhost:8080
```

You should see `deep-research-agent` listed in the Agent Nodes page. If not, wait a few seconds and refresh.

> **Note**: The control plane starts immediately, but the agent may take 10-15 seconds to fully register.

### 3. Execute Research (3-step async flow)

**Step 1: Start the workflow (async)**

```bash
curl -X POST http://localhost:8080/api/v1/execute/async/meta_deep_research.execute_deep_research \
  -H "Content-Type: application/json" \
  -d '{"input": {"query": "What is the current state of quantum computing startups?"}}'
```

Response (HTTP 202 Accepted):

```json
{
	"execution_id": "exec_abc123",
	"run_id": "run_xyz789",
	"status": "queued",
	"target": "meta_deep_research.execute_deep_research"
}
```

**Step 2: Subscribe to SSE events for your workflow (in a separate terminal)**

Use the `run_id` from Step 1 to subscribe to events for this workflow:

```bash
curl -N http://localhost:8080/api/ui/v1/workflows/run_xyz789/notes/events
```

You'll see real-time progress events:

```
data: {"type":"connected","workflow_id":"run_xyz789","message":"Workflow node notes stream connected"}

data: {"type":"note","workflow_id":"run_xyz789","data":{"message":"Starting research collection..."}}

data: {"type":"note","workflow_id":"run_xyz789","data":{"message":"Found 15 relevant articles"}}

data: {"type":"heartbeat","timestamp":"2025-01-15T10:00:30Z"}
```

**Step 3: Get the final result**

Use the `execution_id` from Step 1 to grab final results for this workflow (You need to wait til the workflow is complete check http://localhost:8080/ui/workflows):

```bash
curl http://localhost:8080/api/v1/executions/exec_abc123
```

Response:

```json
{
	"execution_id": "exec_abc123",
	"status": "succeeded",
	"result": {
		"mode": "general",
		"version": "1.0.0",
		"research_package": {
			"document": "# Research Report\n\n## Executive Summary\n..."
		}
	},
	"duration_ms": 145000
}
```

<details>
<summary><strong>Example: Full Research Output</strong> (click to expand)</summary>

Below is a real response from the query *"What is the current state of quantum computing startups?"* — completed in ~16 minutes with 170 reasoning steps, 57 entities, 351 relationships, and 93 sources.

```json
{
  "execution_id": "exec_20260106_172519_6hqcvbe5",
  "run_id": "run_20260106_172519_od6wrp6g",
  "status": "succeeded",
  "result": {
    "mode": "general",
    "version": "1.0.0",
    "metadata": {
      "query": "What is the current state of quantum computing startups?",
      "total_orchestration_time_seconds": 990.02,
      "research_phase_metadata": {
        "iterations_completed": 3,
        "total_entities": 57,
        "total_relationships": 351,
        "total_sources": 93,
        "final_quality_score": 0.7,
        "iteration_summaries": [
          "Iteration 1: +19 entities, +85 relationships (381.3s)",
          "Iteration 2: +9 entities, +137 relationships (224.0s)",
          "Iteration 3: +29 entities, +129 relationships (234.8s)"
        ]
      }
    },
    "research_package": {
      "document_title": "Quantum Computing Startup Landscape Blueprint",
      "executive_summary": "The executive summary of the research document reveals a symbiotic relationship between hardware and software advancements in quantum computing startups, where breakthroughs in one area significantly accelerate the other, driving market growth and providing a competitive edge. Key findings include hardware driving software innovation and software pushing hardware boundaries, with startups benefiting from this symbiosis. Notably, the rapid progress in quantum machine learning and optimization algorithms, such as VQE and QAOA, demonstrates the strides made in quantum software, which in turn requires more advanced hardware. The document also highlights the increasing competition and investment in the quantum computing landscape, with major players like IBM, Google, and Microsoft leading the way.",
      "disclaimers": [
        "Caution: This research may rely on unverified sources.",
        "Note: This study's findings may be contentious due to a high peak disagreement score."
      ],
      "sections": [
        {
          "title": "The Symbiotic Relationship between Hardware and Software",
          "content": "# The Symbiotic Relationship between Hardware and Software\n\nThe development of quantum hardware and software in startups is not two separate races, but a symbiotic dance. Advancements in one area significantly accelerate the other, creating a mutually beneficial relationship that drives market growth and provides a competitive advantage.\n\n## Key Findings\n\n1. **Hardware Drives Software Innovation**\n   - Breakthroughs in quantum hardware, such as those by Google and IBM, pave the way for advanced quantum software and algorithms.\n   - More powerful hardware enables the development of more complex and efficient quantum software and algorithms.\n\n2. **Software Pushes Hardware Boundaries**\n   - Advancements in quantum software and algorithms push the limits of current hardware.\n   - These advancements require more powerful hardware to run effectively, driving hardware innovation.\n\n## Rapid Progress in Quantum Machine Learning and Optimization\n\n| Algorithm/Tool | Developer | Key Features |\n| --- | --- | --- |\n| Variational Quantum Eigensolver (VQE) | IBM, Google, Rigetti | Solves complex optimization problems |\n| Quantum Approximate Optimization Algorithm (QAOA) | Google, IBM | Solves combinatorial optimization problems |\n| Quantum Machine Learning (QML) | IBM, Microsoft, Zapata Computing | Applies quantum principles to machine learning |"
        },
        {
          "title": "Growth and Competition in the Quantum Computing Landscape",
          "content": "# Growth and Competition in the Quantum Computing Landscape\n\n## Increasing Major Players\n\nThe quantum computing landscape has witnessed a significant influx of major players, growing from 21 companies in 2020 to 76 in 2025. This rapid expansion underscores the immense potential and attractiveness of the market.\n\n## Top 10 Leading Quantum Computing Companies in 2025\n\n| Rank | Company | Quantum Volume (Estimated) | Funding Raised (USD Billion) |\n| --- | --- | --- | --- |\n| 1 | IBM Quantum | 127 | 15 |\n| 2 | Google Quantum AI | 64 | 12 |\n| 3 | Microsoft Quantum | 48 | 10 |\n| 4 | IonQ | 32 | 8 |\n| 5 | Rigetti Computing | 24 | 7 |\n| 6 | D-Wave Systems | 16 | 6 |\n| 7 | Honeywell Quantum Solutions | 12 | 5 |\n| 8 | Quantinuum | 8 | 4 |\n| 9 | Quantum Circuits Inc. | 4 | 3 |\n| 10 | Pasqal | 2 | 2 |"
        },
        {
          "title": "Key Collaborations between Hardware and Software Startups",
          "content": "..."
        },
        {
          "title": "Investment Trends in Quantum Computing Startups",
          "content": "..."
        }
      ],
      "source_notes": [
        {
          "citation_id": 1,
          "title": "Google & IBM: New Age of Quantum Computing is About to Begin",
          "domain": "technologymagazine.com",
          "url": "https://technologymagazine.com/news/google-ibm-new-age-of-quantum-computing-is-about-to-begin"
        },
        {
          "citation_id": 2,
          "title": "Good Old IBM Is Leading the Way in the Race for Quantum...",
          "domain": "wsj.com",
          "url": "https://www.wsj.com/tech/ibm-quantum-computer-b443bf5c"
        },
        {
          "citation_id": 3,
          "title": "Quantum Computing Companies in 2025 (76 Major Players)",
          "domain": "thequantuminsider.com",
          "url": "https://thequantuminsider.com/2025/09/23/top-quantum-computing-companies/"
        }
      ]
    }
  },
  "started_at": "2026-01-06T17:25:19Z",
  "completed_at": "2026-01-06T17:41:49Z",
  "duration_ms": 990047
}
```

</details>

## API Reference

### Start Async Execution

**URL**: `POST http://localhost:8080/api/v1/execute/async/meta_deep_research.execute_deep_research`

**Request**:

```json
{
	"input": {
		"query": "Your research question",
		"research_focus": 3,
		"research_scope": 3,
		"max_research_loops": 3,
		"num_parallel_streams": 2,
		"tension_lens": "balanced",
		"source_strictness": "mixed"
	}
}
```

**Response** (HTTP 202):

```json
{
	"execution_id": "exec_abc123",
	"run_id": "run_xyz789",
	"workflow_id": "run_xyz789",
	"status": "queued",
	"target": "meta_deep_research.execute_deep_research",
	"created_at": "2025-01-15T10:00:00Z"
}
```

### Subscribe to Workflow SSE Events

**URL**: `GET http://localhost:8080/api/ui/v1/workflows/:run_id/notes/events`

Use `curl -N` to disable buffering and see events in real-time:

```bash
curl -N http://localhost:8080/api/ui/v1/workflows/run_xyz789/notes/events
```

**Event Types**:

- `connected` - Initial connection confirmation
- `heartbeat` - Keepalive (every 30s)
- `note` - Progress updates from `app.note()` calls in the agent

**Event Format**:

```
data: {"type":"note","workflow_id":"run_xyz789","data":{"message":"Processing step 3 of 5..."}}
```

> **Note**: This endpoint streams events only for the specified workflow ID.

### Get Execution Result

**URL**: `GET http://localhost:8080/api/v1/executions/:execution_id`

**Response**:

```json
{
	"execution_id": "exec_abc123",
	"run_id": "run_xyz789",
	"status": "succeeded",
	"result": {
		"mode": "general",
		"version": "1.0.0",
		"research_package": {
			"document": "# Research Report\n\n## Executive Summary\n...",
			"core_thesis": "Central finding",
			"key_discoveries": ["Discovery 1", "Discovery 2"],
			"entities": [
				{ "name": "Company X", "type": "Company", "summary": "..." }
			],
			"relationships": [
				{
					"source_entity": "A",
					"target_entity": "B",
					"relationship_type": "Funded_By"
				}
			],
			"source_articles": [
				{ "id": 1, "title": "...", "url": "...", "content": "..." }
			],
			"article_evidence": [
				{ "article_id": 1, "facts": ["..."], "quotes": ["..."] }
			]
		},
		"metadata": {
			"total_orchestration_time_seconds": 145.2
		}
	},
	"started_at": "2025-01-15T10:00:00Z",
	"completed_at": "2025-01-15T10:02:25Z",
	"duration_ms": 145000
}
```

**Status Values**: `queued`, `running`, `succeeded`, `failed`, `cancelled`, `timeout`

### Parameters

| Parameter              | Default      | Description                             |
| ---------------------- | ------------ | --------------------------------------- |
| `query`                | _required_   | Research question to investigate        |
| `research_focus`       | `3`          | Depth (1-5): 1=surface, 5=deep dive     |
| `research_scope`       | `3`          | Breadth (1-5): 1=narrow, 5=wide         |
| `max_research_loops`   | `3`          | Iterative research cycles               |
| `num_parallel_streams` | `2`          | Parallel research angles                |
| `tension_lens`         | `"balanced"` | `"balanced"` / `"bull"` / `"bear"`      |
| `source_strictness`    | `"mixed"`    | `"strict"` / `"mixed"` / `"permissive"` |

## Environment Variables

| Variable             | Required     | Description                                                   |
| -------------------- | ------------ | ------------------------------------------------------------- |
| `OPENROUTER_API_KEY` | Yes*         | LLM API key (*not required when using Ollama)                 |
| `OLLAMA_BASE_URL`    | No           | Local Ollama server URL (e.g., `http://host.docker.internal:11434`) |
| `JINA_API_KEY`       | One of these | Jina AI search                                                |
| `TAVILY_API_KEY`     | One of these | Tavily search                                                 |
| `FIRECRAWL_API_KEY`  | One of these | Firecrawl                                                     |
| `SERPER_API_KEY`     | One of these | Serper                                                        |
| `DEFAULT_MODEL`      | No           | LLM model (default: `openrouter/deepseek/deepseek-chat-v3.1`) |

## Model Configuration

The system uses [OpenRouter](https://openrouter.ai) to access LLMs. You can change the model by setting `DEFAULT_MODEL` in your `.env` file.

### Changing Models

```bash
# Edit your .env file
echo 'DEFAULT_MODEL=openrouter/google/gemini-2.5-flash' >> .env

# Restart to apply
docker-compose restart
```

> **Format**: `openrouter/<provider>/<model-name>` - The `openrouter/` prefix is required for LiteLLM routing.

### Recommended Models

All models below are verified to work with the system's complex JSON/Pydantic schema requirements.

#### Best Value (Default)

| Model         | ID                                       | Cost (Input/Output per 1M tokens) | Notes                                           |
| ------------- | ---------------------------------------- | --------------------------------- | ----------------------------------------------- |
| DeepSeek V3.1 | `openrouter/deepseek/deepseek-chat-v3.1` | $0.15 / $0.75                     | **Default** - Excellent reasoning, 128K context |
| DeepSeek V3.2 | `openrouter/deepseek/deepseek-v3.2`      | ~$0.15 / $0.75                    | Latest version                                  |

#### Premium (Highest Quality)

| Model           | ID                                     | Cost (Input/Output per 1M tokens) | Notes                        |
| --------------- | -------------------------------------- | --------------------------------- | ---------------------------- |
| Claude Sonnet 4 | `openrouter/anthropic/claude-sonnet-4` | $3 / $15                          | Best balance of quality/cost |
| Claude Opus 4   | `openrouter/anthropic/claude-opus-4`   | $15 / $75                         | Top-tier reasoning           |

#### Budget-Friendly

| Model                 | ID                                        | Cost (Input/Output per 1M tokens) | Notes                   |
| --------------------- | ----------------------------------------- | --------------------------------- | ----------------------- |
| Gemini 2.5 Flash      | `openrouter/google/gemini-2.5-flash`      | $0.30 / $2.50                     | Fast, 1M context window |
| Gemini 2.5 Flash Lite | `openrouter/google/gemini-2.5-flash-lite` | ~$0.15 / $1.25                    | Ultra-low latency       |

#### Open Source

| Model          | ID                                              | Cost               | Notes                 |
| -------------- | ----------------------------------------------- | ------------------ | --------------------- |
| Qwen 2.5 72B   | `openrouter/qwen/qwen2.5-72b-instruct`          | Varies by provider | Excellent JSON output |
| Qwen 3 235B    | `openrouter/qwen/qwen3-235b-a22b`               | Varies by provider | Latest, very capable  |
| Llama 3.3 70B  | `openrouter/meta-llama/llama-3.3-70b-instruct`  | Often free         | Good general purpose  |
| Llama 3.1 405B | `openrouter/meta-llama/llama-3.1-405b-instruct` | Varies             | Largest open model    |

> **Note**: Prices are approximate and may change. Check [OpenRouter Models](https://openrouter.ai/models) for current pricing.

### Model Requirements

The deep research system requires models that can:

- Generate valid JSON matching complex Pydantic schemas
- Handle multi-step reasoning chains
- Process long context (research synthesis)

All recommended models above meet these requirements. Smaller models (< 7B parameters) may struggle with the JSON schema complexity.

## Local Ollama Configuration

You can use a local [Ollama](https://ollama.ai) deployment instead of OpenRouter. This is useful for:

- Air-gapped environments
- Cost savings with local hardware
- Testing with self-hosted LLaMA models

### Setup

1. **Start Ollama** on your host machine with a model:

```bash
ollama run llama3.2
```

2. **Configure your `.env`**:

```bash
# Point to Ollama (use host.docker.internal for Docker to reach host)
OLLAMA_BASE_URL=http://host.docker.internal:11434
DEFAULT_MODEL=ollama/llama3.2

# OpenRouter key not needed when using Ollama
# OPENROUTER_API_KEY=...
```

3. **Restart the service**:

```bash
docker-compose down && docker-compose up -d
```

### Recommended Ollama Models

| Model | Command | Parameters | Notes |
| ----- | ------- | ---------- | ----- |
| Llama 3.2 | `ollama run llama3.2` | 3B | Fast, good for testing |
| Llama 3.1 8B | `ollama run llama3.1:8b` | 8B | Good balance |
| Llama 3.1 70B | `ollama run llama3.1:70b` | 70B | High quality (needs ~40GB VRAM) |
| Qwen 2.5 72B | `ollama run qwen2.5:72b` | 72B | Excellent structured output |
| DeepSeek R1 | `ollama run deepseek-r1:14b` | 14B | Reasoning model |

> **Note**: For Docker deployments, use `host.docker.internal` instead of `localhost` to reach Ollama running on your host machine.

## Troubleshooting

```bash
docker-compose ps              # Check status
docker-compose logs -f         # Stream all logs
docker-compose restart         # Restart services
docker-compose down -v && docker-compose up -d --build  # Full reset with rebuild
```

## License

MIT License - see [LICENSE](LICENSE) for details.
