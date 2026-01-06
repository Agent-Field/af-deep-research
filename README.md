# AF Deep Research

End-to-end AI-powered deep research. One query, full document output with real-time progress streaming.

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
│    OpenRouter (LLM)  •  Jina AI / Tavily / Firecrawl / Serper (Search)      │
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
open http://localhost:8080/agent-nodes
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

Use the `workflow_id` (same as `run_id`) from Step 1 to subscribe to events for just this workflow:

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

**URL**: `GET http://localhost:8080/api/ui/v1/workflows/:workflowId/notes/events`

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
      "entities": [{"name": "Company X", "type": "Company", "summary": "..."}],
      "relationships": [{"source_entity": "A", "target_entity": "B", "relationship_type": "Funded_By"}],
      "source_articles": [{"id": 1, "title": "...", "url": "...", "content": "..."}],
      "article_evidence": [{"article_id": 1, "facts": ["..."], "quotes": ["..."]}]
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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `query` | *required* | Research question to investigate |
| `research_focus` | `3` | Depth (1-5): 1=surface, 5=deep dive |
| `research_scope` | `3` | Breadth (1-5): 1=narrow, 5=wide |
| `max_research_loops` | `3` | Iterative research cycles |
| `num_parallel_streams` | `2` | Parallel research angles |
| `tension_lens` | `"balanced"` | `"balanced"` / `"bull"` / `"bear"` |
| `source_strictness` | `"mixed"` | `"strict"` / `"mixed"` / `"permissive"` |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | LLM API key |
| `JINA_API_KEY` | One of these | Jina AI search |
| `TAVILY_API_KEY` | One of these | Tavily search |
| `FIRECRAWL_API_KEY` | One of these | Firecrawl |
| `SERPER_API_KEY` | One of these | Serper |
| `DEFAULT_MODEL` | No | LLM model (default: `openrouter/deepseek/deepseek-chat-v3.1`) |

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
| Model | ID | Cost (Input/Output per 1M tokens) | Notes |
|-------|----|------------------------------------|-------|
| DeepSeek V3.1 | `openrouter/deepseek/deepseek-chat-v3.1` | $0.15 / $0.75 | **Default** - Excellent reasoning, 128K context |
| DeepSeek V3.2 | `openrouter/deepseek/deepseek-v3.2` | ~$0.15 / $0.75 | Latest version |

#### Premium (Highest Quality)
| Model | ID | Cost (Input/Output per 1M tokens) | Notes |
|-------|----|------------------------------------|-------|
| Claude Sonnet 4 | `openrouter/anthropic/claude-sonnet-4` | $3 / $15 | Best balance of quality/cost |
| Claude Opus 4 | `openrouter/anthropic/claude-opus-4` | $15 / $75 | Top-tier reasoning |

#### Budget-Friendly
| Model | ID | Cost (Input/Output per 1M tokens) | Notes |
|-------|----|------------------------------------|-------|
| Gemini 2.5 Flash | `openrouter/google/gemini-2.5-flash` | $0.30 / $2.50 | Fast, 1M context window |
| Gemini 2.5 Flash Lite | `openrouter/google/gemini-2.5-flash-lite` | ~$0.15 / $1.25 | Ultra-low latency |

#### Open Source
| Model | ID | Cost | Notes |
|-------|----|------------------------------------|-------|
| Qwen 2.5 72B | `openrouter/qwen/qwen2.5-72b-instruct` | Varies by provider | Excellent JSON output |
| Qwen 3 235B | `openrouter/qwen/qwen3-235b-a22b` | Varies by provider | Latest, very capable |
| Llama 3.3 70B | `openrouter/meta-llama/llama-3.3-70b-instruct` | Often free | Good general purpose |
| Llama 3.1 405B | `openrouter/meta-llama/llama-3.1-405b-instruct` | Varies | Largest open model |

> **Note**: Prices are approximate and may change. Check [OpenRouter Models](https://openrouter.ai/models) for current pricing.

### Model Requirements

The deep research system requires models that can:
- Generate valid JSON matching complex Pydantic schemas
- Handle multi-step reasoning chains
- Process long context (research synthesis)

All recommended models above meet these requirements. Smaller models (< 7B parameters) may struggle with the JSON schema complexity.

## Troubleshooting

```bash
docker-compose ps              # Check status
docker-compose logs -f         # Stream all logs
docker-compose restart         # Restart services
docker-compose down -v && docker-compose up -d --build  # Full reset with rebuild
```

## License

MIT License - see [LICENSE](LICENSE) for details.
