#!/usr/bin/env python3
"""
Deep Research Engine - Adaptive Intelligence System

A meta-level research system that performs comprehensive multi-stream intelligence gathering.
It dynamically classifies queries, executes parallel research streams, and iteratively
refines hypotheses to produce well-sourced analytical packages for any domain.
"""

import asyncio
import datetime
import hashlib
import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import aiohttp
from agentfield import Agent, AIConfig
from pydantic import BaseModel, Field
from temporal_context import get_temporal_context
from doc_generation_pipeline import (
    DocumentResponse as DocGenDocumentResponse,
    FinalDocument as DocGenFinalDocument,
    generate_document_from_package_core,
)

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
# The maximum number of concurrent AI calls to make at once.
# Adjust this based on your model provider's rate limits.


AI_CALL_CONCURRENCY_LIMIT = 20
MAX_ARTICLES_PER_TASK = 10
NUM_SEARCH_TERMS_PER_TASK = 3
# A hard safety limit on the number of task execution loops.
MAX_BLUEPRINT_EXECUTION_LOOPS = int(os.getenv("MAX_BLUEPRINT_EXECUTION_LOOPS", "3"))
ADJUDICATION_BATCH_SIZE = 50

# Batch size for entity deduplication - larger values may overwhelm smaller models
ENTITY_BATCH_SIZE = 10

# --- AI Configuration & Agent Setup ---
# Build litellm_params - supports local Ollama deployments
litellm_params = {"drop_params": True}

# If OLLAMA_BASE_URL is set, configure for local Ollama deployment
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
if ollama_base_url:
    litellm_params["api_base"] = ollama_base_url
    print(f"🦙 Using local Ollama at: {ollama_base_url}")

ai_config = AIConfig(
    model=os.getenv("DEFAULT_MODEL", "openrouter/deepseek/deepseek-chat-v3.1"),
    temperature=float(os.getenv("TEMPERATURE", "0.6")),
    max_tokens=8192,
    litellm_params=litellm_params,
)

app = Agent(
    node_id="meta_deep_research",
    agentfield_server=os.getenv('AGENTFIELD_SERVER', 'http://localhost:8080'),
    version="3.0.0",
    dev_mode=True,
    callback_url=os.getenv("AGENT_CALLBACK_URL", None),
    ai_config=ai_config,
)


# ==============================================================================
# CORE UTILITIES & HELPERS
# ==============================================================================
async def search_web_for_content(query: str) -> List[Dict]:
    """
    Performs a web search using the available search provider.

    Automatically detects and uses the first available provider from:
    Jina, Tavily, Firecrawl, or Serper (in priority order).

    Set SEARCH_PROVIDER env var to force a specific provider.
    """
    from skills.search import search, list_provider_status

    try:
        # Log provider status on first call for debugging
        status = list_provider_status()
        available = [name for name, is_available in status.items() if is_available]
        if available:
            print(f"🔍 DEBUG: Available search providers: {available}")
        else:
            print("⚠️ WARNING: No search providers available! Configure at least one API key.")
            return []

        response = await search(query)
        print(f"🔍 DEBUG: Search completed via {response.provider} - {len(response.results)} results for: {query[:50]}...")

        # Convert SearchResult objects to dicts matching expected format
        return [
            {
                "url": result.url,
                "title": result.title,
                "content": result.content,
                "description": result.description,
            }
            for result in response.results
        ]
    except RuntimeError as e:
        print(f"⚠️ WARNING: No search providers available: {e}")
        return []
    except Exception as e:
        print(f"❌ ERROR: Search failed for query '{query}': {e}")
        return []


async def ai_with_dynamic_params(
    *args, model: Optional[str] = None, api_key: Optional[str] = None, **kwargs
) -> Any:
    """A wrapper for app.ai calls to allow dynamic model and API key overrides."""
    dynamic_params = {}
    if model:
        dynamic_params["model"] = model
    if api_key:
        dynamic_params["api_key"] = api_key

    merged_kwargs = {**kwargs, **dynamic_params}
    return await app.ai(*args, **merged_kwargs)


def create_content_hash(content: str) -> str:
    """Creates a unique MD5 hash for a string of content to identify duplicates."""
    return hashlib.md5(content.encode()).hexdigest()


async def run_in_batches(tasks: List, batch_size: int):
    """Executes a list of asyncio tasks in batches to manage rate limits."""
    results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        results.extend(batch_results)
    return [r for r in results if not isinstance(r, Exception)]


def translate_control_knob_to_description(level: int) -> str:
    """Translates a 1-5 integer into a qualitative descriptor for AI prompts."""
    return {
        1: "Minimal: Stay strictly on topic. Surface level.",
        2: "Constrained: Explore only immediate, directly connected concepts.",
        3: "Balanced: Investigate the main topic and its most important related areas.",
        4: "Expansive: Actively seek out adjacent topics and secondary connections.",
        5: "Maximum: Explore all potentially relevant tangents, contexts, and implications.",
    }.get(level, "Balanced")


# ==============================================================================
# DATA SCHEMAS
# ==============================================================================


# --- Query Classification & Intelligence Schemas ---


class VCQueryClassification(BaseModel):
    """Classifies the user's query into a research category."""

    query_type: str = Field(
        description="The category of the research query (e.g., 'Market_Intelligence', 'Entity_Analysis', 'Competitive_Analysis', 'Technology_Assessment', 'Strategic_Assessment')."
    )
    core_subject: str = Field(
        description="The primary company, technology, or market being investigated."
    )
    key_question: str = Field(
        description="A refined, single-sentence question that captures the user's core intent."
    )


class MarketIntel(BaseModel):
    """Structured insights from the Market Intelligence stream."""

    market_size_summary: str = Field(
        description="TAM/SAM/SOM insights or directional market growth trends."
    )
    key_market_drivers: List[str] = Field(
        description="List of key technological, economic, or regulatory drivers."
    )
    competitive_landscape_summary: str = Field(
        description="A summary of the main competitors and market concentration."
    )
    regulatory_snapshot: str = Field(
        description="Brief on any critical regulations impacting the market."
    )


class EntityPair(BaseModel):
    """Represents a potential duplicate entity pair."""

    entity1_name: str
    entity2_name: str
    similarity_reason: str


class EntityPairList(BaseModel):
    """List of potential duplicate entity pairs."""

    pairs: List[EntityPair]


class MergedEntity(BaseModel):
    """Result of merging two entities."""

    name: str
    type: str
    summary: str


class BusinessIntel(BaseModel):
    """Structured insights from the Business Intelligence stream."""

    business_model_summary: str = Field(
        description="How the company makes money and its value proposition."
    )
    revenue_traction_indicators: List[str] = Field(
        description="Indicators of financial performance or customer adoption."
    )
    customer_segment_summary: str = Field(
        description="Description of the target customer profile and needs."
    )
    go_to_market_strategy: str = Field(
        description="How the company reaches its customers."
    )


class NetworkIntel(BaseModel):
    """Structured insights from the Network Intelligence stream."""

    key_personnel_backgrounds: List[str] = Field(
        description="Summaries of the founding team or key hires' track records."
    )
    investor_syndicate_quality: str = Field(
        description="Analysis of the existing investors and their reputation."
    )
    strategic_partnerships: List[str] = Field(
        description="Notable partnerships that provide a competitive advantage."
    )
    hiring_patterns: str = Field(
        description="Trends in recent hiring that indicate strategic direction."
    )


class TechnicalIntel(BaseModel):
    """Structured insights from the Technical Intelligence stream."""

    tech_stack_summary: str = Field(
        description="Description of the core technology and its components."
    )
    product_differentiation: List[str] = Field(
        description="Key technical or product features that create a moat."
    )
    ip_position_summary: str = Field(
        description="Summary of patent filings or proprietary technology claims."
    )
    innovation_velocity: str = Field(
        description="Indicators of the company's R&D and product development speed."
    )


class InvestmentHypothesis(BaseModel):
    """A structured investment thesis based on synthesized intelligence."""

    core_thesis: str = Field(
        description="The central argument for why this is or is not a good investment, framed as a hypothesis."
    )
    supporting_strengths: List[str] = Field(
        description="The top 3-4 factors that support the thesis."
    )
    counter_risks: List[str] = Field(
        description="The most significant risks or weaknesses that challenge the thesis."
    )
    key_unknowns: List[str] = Field(
        description="Critical unanswered questions that require deeper investigation."
    )


class TrajectoryDecision(BaseModel):
    """Determines the next step in the research process."""

    next_step: str = Field(
        description="The action to take next ('Deep_Dive', 'Multi_Perspective_Analysis', or 'Final_Synthesis')."
    )
    rationale: str = Field(
        description="A brief justification for the chosen next step."
    )
    focus_areas: List[str] = Field(
        description="Specific topics or questions to focus on in the next step."
    )


class BullBearBaseCase(BaseModel):
    """Defines the bull, bear, and base case scenarios for an investment."""

    bull_case_summary: str = Field(
        description="The optimistic scenario: what has to go right for massive success."
    )
    bear_case_summary: str = Field(
        description="The pessimistic scenario: key risks and potential failure modes."
    )
    base_case_summary: str = Field(
        description="The most probable outcome based on the current evidence."
    )


# --- Core Data Schemas ---


class Article(BaseModel):
    """Represents a single source article found during research."""

    id: int
    title: str
    url: str
    content: str
    content_hash: str


class ArticleEvidence(BaseModel):
    """Contains structured data extracted from a single source article."""

    article_id: int
    relevance_summary: str
    facts: List[str]
    quotes: List[str]


class Entity(BaseModel):
    """Represents a person, organization, concept, or other key noun."""

    name: str
    type: str = Field(
        description="The entity category (e.g., 'Organization', 'Person', 'Technology', 'Market_Trend', 'Concept', 'Metric')."
    )
    summary: str


class Relationship(BaseModel):
    """Describes a connection between two entities."""

    source_entity: str
    target_entity: str
    description: str
    relationship_type: str = Field(
        description="The connection type (e.g., 'Competes_With', 'Partners_With', 'Founded_By', 'Influences', 'Depends_On')."
    )


class InquiryProbe(BaseModel):
    """A suggested question for future investigation."""

    question: str
    rationale: str
    suggested_method: str


class UniversalResearchPackage(BaseModel):
    """The final, comprehensive output. Schema is compatible with the original."""

    query: str
    core_thesis: str
    key_discoveries: List[str]
    confidence_assessment: str
    entities: List[Entity]
    relationships: List[Relationship]
    observed_causal_chains: List[str]
    hypothesized_implications: List[str]
    next_inquiry_probes: List[InquiryProbe]
    source_articles: List[Article]
    article_evidence: List[ArticleEvidence]


class ResearchResponse(BaseModel):
    """Base response format for all research APIs."""

    mode: str
    version: str
    research_package: dict
    metadata: dict


# Type aliases for API-specific responses (same structure, different semantic meaning)
ModeAwareResearchResponse = ResearchResponse  # For prepare/continue APIs
ResearchBriefingResponse = ResearchResponse   # For briefing API
DocumentResponse = ResearchResponse           # For document generation API


class VCEntityRelationshipPackage(BaseModel):
    """A container for entities and relationships extracted from intelligence reports."""

    entities: List[Entity]
    relationships: List[Relationship]


class VCKeyDiscoveries(BaseModel):
    """A container for high-level, synthesized discoveries."""

    key_discoveries: List[str] = Field(
        description="A list of 3-5 non-obvious, critical insights synthesized from all intelligence streams."
    )


class StreamOutput(BaseModel):
    """Holds all outputs from a single intelligence stream."""

    stream_type: str
    synthesized_intel: Dict[str, Any]  # This will hold MarketIntel, BusinessIntel, etc.
    source_articles: List[Article]
    article_evidence: List[ArticleEvidence]


class AdaptiveInquiryProbes(BaseModel):
    """Container for adaptive inquiry probes."""

    probes: List[InquiryProbe]


# === NEW INTERNAL SCHEMAS (unchanged external APIs) ===


class ResearchQualityScore(BaseModel):
    """Simple quality assessment of current research state."""

    confidence_score: float = Field(description="Research confidence from 0.0 to 1.0")
    evidence_adequacy: str = Field(
        description="'sufficient', 'moderate', or 'insufficient'"
    )
    critical_gaps_present: bool = Field(
        description="True if major knowledge gaps remain"
    )


class ResearchGap(BaseModel):
    """Individual knowledge gap that needs investigation."""

    gap_description: str = Field(description="What specific knowledge is missing")
    gap_priority: str = Field(description="'high', 'medium', or 'low'")
    gap_type: str = Field(
        description="'entity', 'relationship', 'evidence', or 'context'"
    )


class GapFillingQuery(BaseModel):
    """Targeted query to fill specific research gaps."""

    search_query: str = Field(description="Specific search query to address gaps")
    expected_insights: str = Field(description="What this query should reveal")
    query_priority: str = Field(description="'high', 'medium', or 'low'")


class LoopDecision(BaseModel):
    """Decision on whether to continue research iterations."""

    should_continue: bool = Field(description="True if another research loop is needed")
    termination_reason: str = Field(description="Why stopping or continuing")
    focus_areas: str = Field(description="What to focus on in next iteration")


# ==============================================================================
# META-REASONER AGENTS
# ==============================================================================

# --- Parallelized Processing Utilities ---


@app.reasoner()
async def merge_entity_pair(
    entity1: Entity,
    entity2: Entity,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> MergedEntity:
    """Merges two specific entities into one."""
    prompt = f"""
<entity1>
Name: {entity1.name}
Type: {entity1.type}
Summary: {entity1.summary}
</entity1>

<entity2>
Name: {entity2.name}
Type: {entity2.type}
Summary: {entity2.summary}
</entity2>

<instructions>
Merge these two entities into a single, comprehensive entity.
- Choose the most complete/accurate name
- Use the most appropriate type
- Create a summary that combines the best information from both
</instructions>
"""

    return await ai_with_dynamic_params(
        system="You are an Entity Merger. Combine two entities into one comprehensive entity without losing information.",
        user=prompt,
        schema=MergedEntity,
        model=model,
        api_key=api_key,
    )


@app.reasoner()
async def detect_entity_duplicates_batch(
    entities_batch: List[Entity],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> EntityPairList:
    """Detects potential duplicate entities in a batch."""
    entities_text = "\n".join(
        [f"- {e.name} ({e.type}): {e.summary}" for e in entities_batch]
    )

    prompt = f"""
<entities>
{entities_text}
</entities>

<instructions>
Analyze the entities above and identify pairs that likely represent the same real-world entity.
Look for:
- Same names with slight variations
- Same organizations/people described differently
- Concepts that are essentially the same thing

Only include pairs where you're confident they represent the same entity.
Provide a brief reason for each potential duplicate pair.
</instructions>
"""

    return await ai_with_dynamic_params(
        system="You are an Entity Deduplication Specialist. Identify potential duplicate entities with high confidence.",
        user=prompt,
        schema=EntityPairList,
        model=model,
        api_key=api_key,
    )


async def process_entity_consolidation_parallel(
    all_entities: List[Entity],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Entity]:
    """Consolidates entities using parallel processing."""
    if len(all_entities) <= 1:
        return all_entities

    # Create batches for parallel processing
    if len(all_entities) <= 50:
        entity_batches = [all_entities]
    else:
        entity_batches = batch_list(all_entities, ENTITY_BATCH_SIZE)

    # Step 1: Detect duplicates in all batches in parallel
    # Show sample entities being processed
    sample_entities = [e.name for e in all_entities[:3]]
    entities_preview = ", ".join(sample_entities)
    if len(all_entities) > 3:
        entities_preview += "…"
    app.note(f"Verifying if {entities_preview} are the same organizations…")

    batch_tasks = [
        detect_entity_duplicates_batch(batch, model=model, api_key=api_key)
        for batch in entity_batches
    ]
    batch_results = await run_in_batches(batch_tasks, AI_CALL_CONCURRENCY_LIMIT)

    # Step 2: Collect all duplicate pairs
    all_duplicate_pairs = []
    for result in batch_results:
        if result and hasattr(result, "pairs"):
            all_duplicate_pairs.extend(result.pairs)

    # Step 3: Create entity lookup and prepare merge tasks
    entities_dict = {entity.name: entity for entity in all_entities}
    merge_tasks = []
    pairs_to_process = []

    for pair in all_duplicate_pairs:
        if pair.entity1_name in entities_dict and pair.entity2_name in entities_dict:
            entity1 = entities_dict[pair.entity1_name]
            entity2 = entities_dict[pair.entity2_name]
            merge_tasks.append(
                merge_entity_pair(entity1, entity2, model=model, api_key=api_key)
            )
            pairs_to_process.append(pair)

    # Step 4: Execute all merges in parallel
    if merge_tasks:
        # Show specific entities being merged
        merge_examples = []
        for pair in pairs_to_process[:2]:  # Show first 2 examples
            merge_examples.append(f"{pair.entity1_name} and {pair.entity2_name}")

        if len(pairs_to_process) == 1:
            app.note(
                f"Found {merge_examples[0]} refer to the same entity — consolidating information…"
            )
        elif len(pairs_to_process) == 2:
            app.note(f"Consolidating information for: {' | '.join(merge_examples)}…")
        else:
            app.note(
                f"Consolidating profiles for {merge_examples[0]} and other organizations…"
            )

        merge_results = await run_in_batches(merge_tasks, AI_CALL_CONCURRENCY_LIMIT)

        # Step 5: Apply merge results
        for i, merged in enumerate(merge_results):
            if merged and i < len(pairs_to_process):
                pair = pairs_to_process[i]
                merged_entity = Entity(
                    name=merged.name, type=merged.type, summary=merged.summary
                )

                # Update entities dict
                entities_dict[merged.name] = merged_entity
                if pair.entity1_name != merged.name:
                    entities_dict.pop(pair.entity1_name, None)
                if pair.entity2_name != merged.name:
                    entities_dict.pop(pair.entity2_name, None)

    return list(entities_dict.values())


class EvidenceDeduplication(BaseModel):
    """Result of evidence deduplication."""

    is_duplicate: bool
    reason: str


class RelationshipPair(BaseModel):
    """Represents a potential duplicate relationship pair."""

    relationship1_description: str
    relationship2_description: str
    similarity_reason: str


class RelationshipPairList(BaseModel):
    """List of potential duplicate relationship pairs."""

    pairs: List[RelationshipPair]


@app.reasoner()
async def detect_relationship_duplicates_batch(
    relationships_batch: List[Relationship],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> RelationshipPairList:
    """Detects potential duplicate relationships in a batch."""
    rels_text = "\n".join(
        [
            f"- {r.source_entity} {r.description} {r.target_entity} ({r.relationship_type})"
            for r in relationships_batch
        ]
    )

    prompt = f"""
<relationships>
{rels_text}
</relationships>

<instructions>
Identify pairs of relationships that describe essentially the same connection between entities.
Look for:
- Same relationship expressed differently
- Redundant descriptions of the same connection
- Similar relationships that should be consolidated

Only include pairs where consolidation would reduce redundancy without losing meaning.
</instructions>
"""

    return await ai_with_dynamic_params(
        system="You are a Relationship Deduplication Specialist. Identify redundant relationships.",
        user=prompt,
        schema=RelationshipPairList,
        model=model,
        api_key=api_key,
    )


class MergedRelationship(BaseModel):
    """Result of merging two relationships."""

    source_entity: str
    target_entity: str
    description: str
    relationship_type: str


@app.reasoner()
async def merge_relationship_pair(
    rel1: Relationship,
    rel2: Relationship,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> MergedRelationship:
    """Merges two specific relationships into one."""
    prompt = f"""
<relationship1>
Source: {rel1.source_entity}
Target: {rel1.target_entity}
Description: {rel1.description}
Type: {rel1.relationship_type}
</relationship1>

<relationship2>
Source: {rel2.source_entity}
Target: {rel2.target_entity}
Description: {rel2.description}
Type: {rel2.relationship_type}
</relationship2>

<instructions>
Merge these relationships into a single, comprehensive relationship.
- Use the most accurate entity names
- Create a description that captures the complete relationship
- Choose the most appropriate relationship type
</instructions>
"""

    return await ai_with_dynamic_params(
        system="You are a Relationship Merger. Combine relationships without losing information.",
        user=prompt,
        schema=MergedRelationship,
        model=model,
        api_key=api_key,
    )


def batch_list(items: List, batch_size: int) -> List[List]:
    """Split a list into batches of specified size."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


async def process_relationship_consolidation_parallel(
    all_relationships: List[Relationship],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Relationship]:
    """Consolidates relationships using parallel processing."""
    if len(all_relationships) <= 1:
        return all_relationships

    # Create batches for parallel processing
    if len(all_relationships) <= 30:
        rel_batches = [all_relationships]
    else:
        rel_batches = batch_list(all_relationships, 15)

    # Step 1: Detect duplicates in all batches in parallel
    # Get dynamic context from relationships being processed
    if all_relationships:
        sample_entities = set()
        for rel in all_relationships[:3]:
            sample_entities.add(rel.source_entity)
            sample_entities.add(rel.target_entity)
        entities_context = ", ".join(list(sample_entities)[:3])
        if len(sample_entities) > 3:
            entities_context += "…"
        app.note(f"Mapping connections between {entities_context}…")
    else:
        app.note("Mapping connections between key players…")
    batch_tasks = [
        detect_relationship_duplicates_batch(batch, model=model, api_key=api_key)
        for batch in rel_batches
    ]
    batch_results = await run_in_batches(batch_tasks, AI_CALL_CONCURRENCY_LIMIT)

    # Step 2: Collect all duplicate pairs and create lookup
    all_duplicate_pairs = []
    for result in batch_results:
        if result and hasattr(result, "pairs"):
            all_duplicate_pairs.extend(result.pairs)

    # Step 3: Create relationship lookup and prepare merge tasks
    rel_dict = {}
    for rel in all_relationships:
        key = (rel.source_entity, rel.target_entity, rel.description)
        rel_dict[key] = rel

    merge_tasks = []
    pairs_to_process = []

    for pair in all_duplicate_pairs:
        # Find relationships by description
        rel1 = next(
            (
                r
                for r in all_relationships
                if r.description == pair.relationship1_description
            ),
            None,
        )
        rel2 = next(
            (
                r
                for r in all_relationships
                if r.description == pair.relationship2_description
            ),
            None,
        )

        if rel1 and rel2:
            merge_tasks.append(
                merge_relationship_pair(rel1, rel2, model=model, api_key=api_key)
            )
            pairs_to_process.append((rel1, rel2))

    # Step 4: Execute all merges in parallel
    if merge_tasks:
        app.note("Connecting the dots between relationships…")
        merge_results = await run_in_batches(merge_tasks, AI_CALL_CONCURRENCY_LIMIT)

        # Step 5: Apply merge results
        for i, merged in enumerate(merge_results):
            if merged and i < len(pairs_to_process):
                rel1, rel2 = pairs_to_process[i]
                merged_rel = Relationship(
                    source_entity=merged.source_entity,
                    target_entity=merged.target_entity,
                    description=merged.description,
                    relationship_type=merged.relationship_type,
                )

                # Update relationships dict
                new_key = (
                    merged.source_entity,
                    merged.target_entity,
                    merged.description,
                )
                rel_dict[new_key] = merged_rel

                # Remove originals
                old_key1 = (rel1.source_entity, rel1.target_entity, rel1.description)
                old_key2 = (rel2.source_entity, rel2.target_entity, rel2.description)
                rel_dict.pop(old_key1, None)
                rel_dict.pop(old_key2, None)

    return list(rel_dict.values())


@app.reasoner()
async def check_evidence_duplication(
    evidence1: ArticleEvidence,
    evidence2: ArticleEvidence,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> EvidenceDeduplication:
    """Checks if two pieces of evidence are duplicates."""
    prompt = f"""
<evidence1>
Summary: {evidence1.relevance_summary}
Facts: {evidence1.facts[:3]}
Quotes: {evidence1.quotes[:2]}
</evidence1>

<evidence2>
Summary: {evidence2.relevance_summary}
Facts: {evidence2.facts[:3]}
Quotes: {evidence2.quotes[:2]}
</evidence2>

<instructions>
Determine if these represent duplicate/redundant evidence.
Consider them duplicates if they convey essentially the same information.
</instructions>
"""

    return await ai_with_dynamic_params(
        system="You are an Evidence Analyst. Determine if evidence pieces are duplicates.",
        user=prompt,
        schema=EvidenceDeduplication,
        model=model,
        api_key=api_key,
    )


async def process_evidence_deduplication_parallel(
    all_evidence: List[ArticleEvidence],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[ArticleEvidence]:
    """Advanced evidence deduplication using parallel AI analysis."""
    if len(all_evidence) <= 1:
        return all_evidence

    # Step 1: Basic hash-based deduplication
    unique_evidence = []
    seen_hashes = set()

    for evidence in all_evidence:
        evidence_content = f"{evidence.relevance_summary}{''.join(evidence.facts[:3])}"
        evidence_hash = create_content_hash(evidence_content)

        if evidence_hash not in seen_hashes:
            seen_hashes.add(evidence_hash)
            unique_evidence.append(evidence)

    # Step 2: AI-based similarity checking for remaining evidence
    if len(unique_evidence) <= 20:  # Only use AI for manageable sizes
        similarity_tasks = []
        pairs_to_check = []

        for i in range(len(unique_evidence)):
            for j in range(i + 1, len(unique_evidence)):
                similarity_tasks.append(
                    check_evidence_duplication(
                        unique_evidence[i],
                        unique_evidence[j],
                        model=model,
                        api_key=api_key,
                    )
                )
                pairs_to_check.append((i, j))

        if similarity_tasks:
            # Show what we're cross-referencing
            sample_sources = [
                f"source {unique_evidence[i].article_id}"
                for i in range(min(3, len(unique_evidence)))
            ]
            sources_context = ", ".join(sample_sources)
            if len(unique_evidence) > 3:
                sources_context += f" and {len(unique_evidence) - 3} others"
            app.note(f"Analyzing insights from {sources_context}…")
            similarity_results = await run_in_batches(
                similarity_tasks, AI_CALL_CONCURRENCY_LIMIT
            )

            # Remove duplicates based on AI analysis
            indices_to_remove = set()
            for k, result in enumerate(similarity_results):
                if result and result.is_duplicate and k < len(pairs_to_check):
                    i, j = pairs_to_check[k]
                    indices_to_remove.add(j)  # Remove the second one

            unique_evidence = [
                evidence
                for i, evidence in enumerate(unique_evidence)
                if i not in indices_to_remove
            ]

    return unique_evidence


@app.reasoner()
async def synthesize_key_discoveries_from_intel(
    query: str,
    market_intel: MarketIntel,
    business_intel: BusinessIntel,
    network_intel: NetworkIntel,
    technical_intel: TechnicalIntel,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> VCKeyDiscoveries:
    """
    Synthesizes disparate intelligence streams into high-level, non-obvious discoveries.
    """
    # Show what we're synthesizing
    intel_sources = []
    if market_intel.key_market_drivers:
        intel_sources.append(f"market trends like {market_intel.key_market_drivers[0]}")
    if business_intel.business_model_summary:
        intel_sources.append("business model insights")
    if network_intel.key_personnel_backgrounds:
        intel_sources.append("team analysis")

    sources_context = ", ".join(intel_sources[:2])
    if sources_context:
        app.note(f"Synthesizing insights from {sources_context}…")
    else:
        app.note("Identifying the key takeaways from our research…")

    prompt = f"""
<task>
You are a sharp, insightful VC Analyst. Your superpower is synthesis. You can look at reports from different teams (market, business, network, tech) and connect the dots to find the hidden story. Your goal is to identify the 3-5 most critical, non-obvious insights that a partner needs to know.
</task>

<original_research_query>{query}</original_research_query>

<intelligence_reports>
<market_intelligence>
- Market Size: {market_intel.market_size_summary}
- Key Drivers: {', '.join(market_intel.key_market_drivers)}
- Competition: {market_intel.competitive_landscape_summary}
</market_intelligence>

<business_intelligence>
- Business Model: {business_intel.business_model_summary}
- Traction: {', '.join(business_intel.revenue_traction_indicators)}
- GTM Strategy: {business_intel.go_to_market_strategy}
</business_intelligence>

<network_intelligence>
- Team: {', '.join(network_intel.key_personnel_backgrounds)}
- Investors: {network_intel.investor_syndicate_quality}
</network_intelligence>

<technical_intelligence>
- Technology: {technical_intel.tech_stack_summary}
- Differentiation: {', '.join(technical_intel.product_differentiation)}
</technical_intelligence>
</intelligence_reports>

<instructions>
1.  **Do NOT simply summarize each section.** Read across all the reports.
2.  **Find the Connections:** How does a market driver relate to a technical differentiator? How does the team's background explain their GTM strategy?
3.  **Identify Tensions and Synergies:** Is there a conflict between the market trend and the business model? Is there a powerful synergy between the investor network and strategic partnerships?
4.  **Distill into Discoveries:** Formulate 3-5 single-sentence `key_discoveries`. Each discovery should be a complete thought and a valuable insight on its own. Focus on the 'so what'.

**Example of a WEAK discovery (just a summary):** "The market is large and the team is experienced."
**Example of a STRONG discovery (a synthesis):** "The experienced team's deep industry connections are a key enabler for their land-and-expand GTM strategy in a market that is rapidly consolidating."
</instructions>
"""
    return await ai_with_dynamic_params(
        system="You are a VC Analyst specializing in synthesizing disparate data points into critical, non-obvious insights.",
        user=prompt,
        schema=VCKeyDiscoveries,
        model=model,
        api_key=api_key,
    )


@app.reasoner()
async def extract_vc_entities_and_relationships(
    core_subject: str,
    all_discoveries: List[str],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> VCEntityRelationshipPackage:
    """
    Extracts a tight, relevant knowledge graph from the synthesized discoveries.
    """
    # Show what we're mapping
    if all_discoveries:
        # Extract entity names mentioned in discoveries
        discovery_preview = all_discoveries[0][:60] + "..."
        app.note(
            f"Mapping the network around {core_subject} — starting with: {discovery_preview}"
        )
    else:
        app.note(f"Mapping the key players and connections around {core_subject}…")

    prompt = f"""
<task>
You are a Knowledge Graph Specialist for a VC firm. Your job is to read a list of high-level discoveries and extract the most critical entities and their relationships to build a strategic map of the investment landscape.
</task>

<core_subject_of_analysis>{core_subject}</core_subject_of_analysis>

<synthesized_discoveries>
{chr(10).join(f'- {d}' for d in all_discoveries)}
</synthesized_discoveries>

<instructions>
1.  **Identify Key Entities**: Read through the discoveries and identify the most important actors. Always include the `<core_subject_of_analysis>`. Use the following VC-specific entity types:
    - `Company`: A specific company (e.g., "{core_subject}").
    - `Investor`: A VC firm or angel investor.
    - `Founder`: A key person who started a company.
    - `Technology`: A specific technology or platform (e.g., "AI-powered logistics").
    - `Market_Trend`: A high-level market dynamic (e.g., "Market Consolidation").
    - `Metric`: A key business metric (e.g., "Customer Acquisition Cost").
2.  **Define Relationships**: Identify the connections between the entities you extracted. Use the following VC-specific relationship types:
    - `Invests_In`
    - `Competes_With`
    - `Partners_With`
    - `Founded_By`
    - `Acquires`
    - `Impacted_By` (e.g., a Company is `Impacted_By` a Market_Trend).
    - `Demonstrates` (e.g., a Company `Demonstrates` a Metric).
3.  **Be Selective**: Only extract the most central entities and relationships that define the strategic picture. Aim for quality over quantity. The graph should be tight and meaningful.
</instructions>
"""
    return await ai_with_dynamic_params(
        system="You are a VC Knowledge Graph Specialist who extracts strategically important entities and relationships.",
        user=prompt,
        schema=VCEntityRelationshipPackage,
        model=model,
        api_key=api_key,
    )


@app.reasoner()
async def generate_adaptive_hypothesis(
    query: str,
    query_type: str,
    key_discoveries: List[str],
    entities: List[Entity],
    relationships: List[Relationship],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> InvestmentHypothesis:
    """Generates adaptive hypothesis based on query type and domain."""

    # Create rich context from extracted network
    entities_xml = "\n".join(
        [
            f'<entity name="{e.name}" type="{e.type}">{e.summary}</entity>'
            for e in entities[:20]
        ]
    )

    relationships_xml = "\n".join(
        [
            f"<relationship>{r.source_entity} {r.description} {r.target_entity} (type: {r.relationship_type})</relationship>"
            for r in relationships[:15]
        ]
    )

    discoveries_xml = "\n".join(
        [f"<discovery>{discovery}</discovery>" for discovery in key_discoveries]
    )

    # Adaptive system prompts based on query type
    domain_frameworks = {
        "Market_Intelligence": {
            "persona": "You are a Market Strategy Analyst synthesizing market intelligence into strategic insights.",
            "thesis_focus": "market opportunity, competitive dynamics, and strategic positioning",
            "strengths_focus": "market drivers, competitive advantages, and growth opportunities",
            "risks_focus": "market risks, competitive threats, and adoption barriers",
        },
        "Entity_Analysis": {
            "persona": "You are a Strategic Analyst evaluating entities and their systemic importance.",
            "thesis_focus": "entity capabilities, strategic position, and systemic impact",
            "strengths_focus": "core capabilities, strategic assets, and competitive advantages",
            "risks_focus": "operational risks, strategic vulnerabilities, and external threats",
        },
        "Competitive_Analysis": {
            "persona": "You are a Competitive Intelligence Analyst mapping competitive landscapes.",
            "thesis_focus": "competitive positioning, differentiation, and market dynamics",
            "strengths_focus": "competitive advantages, market position, and strategic assets",
            "risks_focus": "competitive threats, market shifts, and strategic vulnerabilities",
        },
        "Technology_Assessment": {
            "persona": "You are a Technology Strategy Analyst evaluating technological capabilities and trends.",
            "thesis_focus": "technological capabilities, innovation potential, and adoption dynamics",
            "strengths_focus": "technical advantages, innovation capacity, and adoption drivers",
            "risks_focus": "technical limitations, obsolescence risks, and adoption barriers",
        },
    }

    framework = domain_frameworks.get(query_type, domain_frameworks["Entity_Analysis"])

    prompt = f"""
<mission>
{framework["persona"]} Your task is to synthesize comprehensive research findings into a testable strategic hypothesis.
</mission>

<research_context>
<original_query>{query}</original_query>
<query_classification>{query_type}</query_classification>
</research_context>

<comprehensive_research_findings>
<key_discoveries>
{discoveries_xml}
</key_discoveries>

<extracted_entities>
{entities_xml}
</extracted_entities>

<discovered_relationships>
{relationships_xml}
</discovered_relationships>
</comprehensive_research_findings>

<hypothesis_synthesis_framework>
**Core Thesis Development:**
Focus on {framework["thesis_focus"]}. Synthesize the research findings into a single, testable central argument that addresses the original query. The thesis should:
- Integrate insights from entities, relationships, and discoveries
- Be specific and actionable rather than generic
- Make a clear claim that can be supported or refuted by evidence

**Supporting Strengths Analysis:**
Identify factors related to {framework["strengths_focus"]} that support the core thesis:
- Draw from entity capabilities and relationship advantages
- Reference specific discoveries that strengthen the position
- Focus on sustainable competitive advantages or strategic assets

**Counter Risks Assessment:**
Identify factors related to {framework["risks_focus"]} that could challenge the thesis:
- Consider entity vulnerabilities and relationship weaknesses
- Include external threats and market risks from discoveries
- Focus on factors that could invalidate the core argument

**Knowledge Gaps Identification:**
What critical questions remain unanswered that would strengthen or weaken the thesis:
- Missing information about key entities or relationships
- Uncertain causation or correlation in discovered patterns
- External factors not yet investigated that could impact conclusions
</hypothesis_synthesis_framework>

<instructions>
Synthesize the comprehensive research findings into a structured hypothesis:
1. **Core Thesis**: A single, clear statement that synthesizes the main finding/conclusion
2. **Supporting Strengths**: 3-5 key factors that support the thesis based on the research
3. **Counter Risks**: 3-5 significant risks or weaknesses that could challenge the thesis
4. **Key Unknowns**: 3-5 critical unanswered questions that need investigation

Ensure the hypothesis directly addresses the original query and integrates insights from the entity/relationship network analysis.
</instructions>
"""

    return await ai_with_dynamic_params(
        system=framework["persona"]
        + " You synthesize comprehensive research into testable strategic hypotheses that integrate network analysis with domain expertise.",
        user=prompt,
        schema=InvestmentHypothesis,  # Reuse schema but make it domain-adaptive
        model=model,
        api_key=api_key,
    )


@app.reasoner()
async def merge_and_refine_hypothesis(
    original_hypothesis: InvestmentHypothesis,
    new_intel_summary: str,
    sub_query: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> InvestmentHypothesis:
    """Merges new intelligence from a deep dive to refine an existing hypothesis."""
    # Show what we're integrating
    intel_preview = (
        new_intel_summary[:50] + "..."
        if len(new_intel_summary) > 50
        else new_intel_summary
    )
    app.note(f"Integrating new insights: {intel_preview}")

    prompt = f"""
<task>
You are a General Partner reviewing an updated diligence report. An analyst has just completed a deep dive on a specific question. Your job is to integrate the new findings with the original investment hypothesis to create a stronger, more refined version.
</task>

<original_hypothesis>
- **Thesis**: {original_hypothesis.core_thesis}
- **Strengths**: {original_hypothesis.supporting_strengths}
- **Risks**: {original_hypothesis.counter_risks}
- **Unknowns**: {original_hypothesis.key_unknowns}
</original_hypothesis>

<deep_dive_request>{sub_query}</deep_dive_request>
<new_intelligence_summary>{new_intel_summary}</new_intelligence_summary>

<instructions>
1.  **Integrate Findings**: Read the `new_intelligence_summary`. How does it confirm, contradict, or clarify the `original_hypothesis`?
2.  **Refine Core Thesis**: Rewrite the `core_thesis` to incorporate the new information. It should be more specific and confident.
3.  **Update Strengths & Risks**: Modify the lists of strengths and risks. Add new ones, remove invalidated ones, or strengthen existing ones based on the new intel.
4.  **Resolve Unknowns**: The `key_unknowns` should be updated. Remove the questions that have now been answered by the deep dive and add any new critical questions that have emerged.
</instructions>
"""
    return await ai_with_dynamic_params(
        system="You are a VC General Partner refining an investment thesis with new diligence findings.",
        user=prompt,
        schema=InvestmentHypothesis,
        model=model,
        api_key=api_key,
    )


@app.reasoner()
async def classify_vc_query(
    query: str, model: Optional[str] = None, api_key: Optional[str] = None
) -> VCQueryClassification:
    """Phase 1: Classifies the initial query into a research workflow category."""
    app.note(
        f"Analyzing your question: '{query[:60]}{'...' if len(query) > 60 else ''}'"
    )
    prompt = f"""
<task>
You are a Principal at a top-tier Venture Capital firm. Your role is to receive a research request from a Partner and immediately classify it to assign it to the right analyst team. You must determine the primary intent behind the query.
</task>

<query_categories>
- **Market_Intelligence**: Broad questions about market size, trends, TAM/SAM/SOM, and competitive dynamics. (e.g., "What is the market size for developer tools in AI?")
- **Company_Diligence**: Specific questions about a named company's business model, traction, team, or technology. (e.g., "Analyze the business model of 'Acme Corp'.")
- **Competitor_Analysis**: Questions focused on comparing a company to its rivals. (e.g., "How does 'Acme Corp' differentiate from its main competitors?")
- **Team_Assessment**: Questions specifically about the founders, executives, or key hires of a company. (e.g., "What is the track record of the founding team of 'Acme Corp'?")
- **Thesis_Validation**: Questions designed to prove or disprove a specific investment idea. (e.g., "Is now the right time to invest in vertical SaaS for logistics?")
</query_categories>

<user_query>{query}</user_query>

<instructions>
1.  Analyze the `<user_query>`.
2.  Select the single best `query_type` from the list of categories.
3.  Identify the `core_subject` (the company, market, or technology).
4.  Rewrite the query into a `key_question` that is a clear, actionable research directive for an analyst.
</instructions>
"""
    return await ai_with_dynamic_params(
        system="You are a VC Principal classifying research requests. Your output must be precise and actionable.",
        user=prompt,
        schema=VCQueryClassification,
        model=model,
        api_key=api_key,
    )


@app.reasoner()
async def gather_intelligence_stream(
    stream_type: str,
    subject: str,
    key_question: str,
    start_article_id: int,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> StreamOutput:
    """
    Executes a full intelligence stream: search, evidence extraction, and synthesis.
    This version correctly processes and returns all source materials.
    """
    app.note(f"Researching {subject}'s {stream_type.lower()} landscape…")

    # Configuration for each stream type
    stream_configs = {
        "Market": {
            "persona": "You are a Market Research Analyst at a VC firm specializing in market dynamics, competitive landscapes, and growth forecasts.",
            "schema": MarketIntel,
            "search_queries": [
                f"{subject} market size TAM SAM growth forecast 2025",
                f"Key market trends driving {subject} sector dynamics",
                f"Top competitors market share {subject} competitive analysis",
                f"Regulatory environment {subject} compliance requirements",
            ],
        },
        "Business": {
            "persona": "You are a Business Intelligence Analyst at a VC firm specializing in business models, revenue mechanics, and customer acquisition strategies.",
            "schema": BusinessIntel,
            "search_queries": [
                f"{subject} business model revenue streams monetization",
                f"{subject} customer acquisition go-to-market strategy",
                f"{subject} funding rounds traction metrics customer growth",
                f"{subject} target customer profile market positioning",
            ],
        },
        "Network": {
            "persona": "You are a Network Intelligence Analyst at a VC firm specializing in team assessment, investor syndication, and strategic partnerships.",
            "schema": NetworkIntel,
            "search_queries": [
                f"{subject} founders executives management team background",
                f"{subject} investors board members funding syndicate",
                f"{subject} strategic partnerships integrations alliances",
                f"{subject} key hires talent acquisition recruiting 2024 2025",
            ],
        },
        "Technical": {
            "persona": "You are a Technical Diligence Analyst at a VC firm specializing in technology assessment, IP analysis, and product differentiation.",
            "schema": TechnicalIntel,
            "search_queries": [
                f"{subject} technology stack technical architecture",
                f"{subject} product features differentiation competitive advantages",
                f"{subject} patents intellectual property proprietary technology",
                f"{subject} technical team engineering capabilities R&D",
            ],
        },
    }

    config = stream_configs.get(stream_type)
    if not config:
        raise ValueError(f"Invalid stream type: {stream_type}")

    # --- 1. Search and Article Creation ---
    search_tasks = [search_web_for_content(query) for query in config["search_queries"]]
    search_results_lists = await asyncio.gather(*search_tasks)

    # Flatten and deduplicate articles
    unique_articles_map = {
        res["url"]: res
        for res in [item for sublist in search_results_lists for item in sublist]
        if "url" in res and res.get("content", "").strip()
    }

    source_articles: List[Article] = []
    article_id_counter = start_article_id
    for url, result in unique_articles_map.items():
        content = result.get("content", "")
        if content.strip():
            source_articles.append(
                Article(
                    id=article_id_counter,
                    title=result.get("title", "No Title"),
                    url=url,
                    content=content,
                    content_hash=create_content_hash(content),
                )
            )
            article_id_counter += 1

    # --- 2. Parallel Evidence Extraction ---
    async def extract_evidence(article: Article) -> Optional[ArticleEvidence]:
        prompt = f"""
<task>
You are a VC Intelligence Analyst specializing in {stream_type.lower()} intelligence. Your mission is to extract evidence that directly supports investment decision-making.
</task>

<research_context>
<original_query>{key_question}</original_query>
<subject_company>{subject}</subject_company>
<intelligence_focus>{stream_type} Analysis</intelligence_focus>
</research_context>

<article_content>
{article.content[:4000]}
</article_content>

<extraction_framework>
<relevance_assessment>
Assess WHY this article matters for VC decision-making on {subject}:
- Does it provide market sizing or competitive intelligence?
- Does it reveal business model insights or traction signals?
- Does it offer team assessment or investor validation data?
- Does it contain technical differentiation or IP insights?
</relevance_assessment>

<fact_extraction>
Extract discrete, investment-relevant facts:
- Quantitative metrics (revenue, growth rates, market size, funding amounts)
- Qualitative indicators (customer testimonials, expert opinions, competitive advantages)
- Strategic developments (partnerships, acquisitions, product launches)
- Risk factors (regulatory challenges, competitive threats, technical limitations)
</fact_extraction>

<quote_extraction>
Extract impactful quotes that:
- Provide expert validation or criticism
- Reveal strategic intent from leadership
- Demonstrate market sentiment or customer feedback
- Support or challenge investment thesis
</quote_extraction>
</extraction_framework>

<instructions>
Extract evidence through the lens of VC investment analysis. Focus on information that would influence an investment committee's decision. Ensure the relevance_summary explicitly connects the article to investment decision-making for {subject}.
</instructions>
"""

        class TempEvidence(BaseModel):
            relevance_summary: str
            facts: List[str]
            quotes: List[str]

        try:
            ai_output = await ai_with_dynamic_params(
                system=config["persona"]
                + " Your extraction must focus on investment-relevant intelligence that supports VC decision-making.",
                user=prompt,
                schema=TempEvidence,
                model=model,
                api_key=api_key,
            )
            return ArticleEvidence(article_id=article.id, **ai_output.dict())
        except Exception as e:
            return None

    extraction_tasks = [extract_evidence(article) for article in source_articles]
    article_evidence_results = await run_in_batches(
        extraction_tasks, AI_CALL_CONCURRENCY_LIMIT
    )
    article_evidence: List[ArticleEvidence] = [
        ev for ev in article_evidence_results if ev
    ]

    # --- 3. Synthesize Final Intel from Structured Evidence ---
    evidence_context = "\n".join(
        [
            f"<evidence source_id='{ev.article_id}' relevance='{ev.relevance_summary}'>\n<facts>{'</fact><fact>'.join(ev.facts)}</facts>\n<quotes>{'</quote><quote>'.join(ev.quotes)}</quotes>\n</evidence>"
            for ev in article_evidence
        ]
    )

    synthesis_prompt = f"""
<task>
You are a {stream_type} Intelligence Synthesist at a premier VC firm. Your mission is to synthesize structured evidence into actionable {stream_type.lower()} intelligence for investment decision-making.
</task>

<research_context>
<investment_target>{subject}</investment_target>
<key_research_question>{key_question}</key_research_question>
<intelligence_stream>{stream_type} Analysis</intelligence_stream>
</research_context>

<synthesis_framework>
<investment_lens>
All analysis must be framed through investment considerations:
- Market opportunity size and timing
- Business model viability and scalability
- Competitive positioning and differentiation
- Team execution capability and track record
- Technology moats and intellectual property
- Risk factors and potential failure modes
</investment_lens>

<evidence_integration>
Synthesize evidence to populate the required {stream_type} intelligence fields:
- Base ALL conclusions on the provided structured evidence
- Cross-reference facts across multiple sources for validation
- Identify patterns and themes across the evidence base
- Flag contradictions or uncertainty where evidence conflicts
- Quantify insights where possible (market sizes, growth rates, etc.)
</evidence_integration>
</synthesis_framework>

<structured_evidence>
{evidence_context}
</structured_evidence>

<instructions>
Synthesize the structured evidence above into comprehensive {stream_type.lower()} intelligence. Ensure every field in your response is grounded in the provided evidence. If evidence is insufficient for a field, explicitly state "Evidence insufficient" rather than speculating.

Focus on actionable insights that would inform an investment committee's decision on {subject}.
</instructions>
"""

    synthesized_intel = await ai_with_dynamic_params(
        system=config["persona"]
        + " You synthesize evidence into actionable VC intelligence, basing all conclusions strictly on provided evidence.",
        user=synthesis_prompt,
        schema=config["schema"],
        model=model,
        api_key=api_key,
    )

    # Show what we discovered in this stream
    if article_evidence:
        sample_insights = []
        for ev in article_evidence[:2]:
            if ev.facts:
                sample_insights.append(ev.facts[0][:50] + "...")
        insights_preview = " | ".join(sample_insights)
        app.note(f"Found key insights about {subject}: {insights_preview}")
    else:
        app.note(f"Completed {stream_type.lower()} analysis of {subject}")

    return StreamOutput(
        stream_type=stream_type,
        synthesized_intel=synthesized_intel.dict(),
        source_articles=source_articles,
        article_evidence=article_evidence,
    )


@app.reasoner()
async def synthesize_investment_hypothesis(
    query: str,
    market_intel: MarketIntel,
    business_intel: BusinessIntel,
    network_intel: NetworkIntel,
    technical_intel: TechnicalIntel,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> InvestmentHypothesis:
    """Phase 3: Synthesizes findings from all streams into a unified investment hypothesis."""
    # Show what we're synthesizing
    key_elements = []
    if market_intel.key_market_drivers:
        key_elements.append(f"market drivers like {market_intel.key_market_drivers[0]}")
    if business_intel.revenue_traction_indicators:
        key_elements.append(
            f"traction signals including {business_intel.revenue_traction_indicators[0]}"
        )

    elements_context = " and ".join(key_elements[:2])
    if elements_context:
        app.note(f"Connecting the dots between {elements_context}…")
    else:
        app.note("Synthesizing our research findings…")

    prompt = f"""
<task>
You are a General Partner at a VC firm leading an investment committee meeting. You have just heard from your four analyst teams (Market, Business, Network, Technical). Your job is to synthesize all their findings into a clear, actionable investment hypothesis.
</task>

<original_research_query>{query}</original_research_query>

<analyst_briefings>
<market_intelligence>
- Market Size: {market_intel.market_size_summary}
- Key Drivers: {', '.join(market_intel.key_market_drivers)}
- Competition: {market_intel.competitive_landscape_summary}
- Regulation: {market_intel.regulatory_snapshot}
</market_intelligence>

<business_intelligence>
- Business Model: {business_intel.business_model_summary}
- Traction: {', '.join(business_intel.revenue_traction_indicators)}
- GTM Strategy: {business_intel.go_to_market_strategy}
</business_intelligence>

<network_intelligence>
- Team: {', '.join(network_intel.key_personnel_backgrounds)}
- Investors: {network_intel.investor_syndicate_quality}
- Partnerships: {', '.join(network_intel.strategic_partnerships)}
</network_intelligence>

<technical_intelligence>
- Technology: {technical_intel.tech_stack_summary}
- Differentiation: {', '.join(technical_intel.product_differentiation)}
- IP Position: {technical_intel.ip_position_summary}
</technical_intelligence>
</analyst_briefings>

<instructions>
1.  **Formulate Core Thesis**: Write a single, concise sentence that acts as a testable investment hypothesis. It should state the primary reason to either invest or pass. (e.g., "We should invest because the company's proprietary tech creates a strong moat in a rapidly growing market.")
2.  **Identify Supporting Strengths**: List the top 3-4 most compelling points from the briefings that support your thesis.
3.  **Identify Counter Risks**: List the top 3-4 most significant weaknesses or threats that could invalidate your thesis.
4.  **Define Key Unknowns**: What are the most critical pieces of information still missing that you would need before making a final decision? These will guide the next phase of research.
</instructions>
"""
    return await ai_with_dynamic_params(
        system="You are a VC General Partner synthesizing analyst reports into a clear investment thesis.",
        user=prompt,
        schema=InvestmentHypothesis,
        model=model,
        api_key=api_key,
    )


@app.reasoner()
async def decide_research_trajectory(
    hypothesis: InvestmentHypothesis,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> TrajectoryDecision:
    """Phase 4: Acts as the dynamic router, deciding what to do next based on the hypothesis."""
    # Show what we're evaluating
    unknowns_preview = ""
    if hypothesis.key_unknowns:
        unknowns_preview = f" — need to explore {hypothesis.key_unknowns[0][:40]}..."
    app.note(f"Evaluating research progress{unknowns_preview}")

    prompt = f"""
<task>
You are a Meta-Cognitive Research Strategist. Your job is to analyze an investment hypothesis and determine the most effective next step to improve its quality and confidence.
</task>

<current_hypothesis>
- **Thesis**: {hypothesis.core_thesis}
- **Strengths**: {hypothesis.supporting_strengths}
- **Risks**: {hypothesis.counter_risks}
- **Unknowns**: {hypothesis.key_unknowns}
</current_hypothesis>

<next_step_options>
- **Deep_Dive**: Choose this if the `key_unknowns` are specific, factual questions that can be answered with more targeted research. This is for filling known gaps.
- **Multi_Perspective_Analysis**: Choose this if the hypothesis is well-formed but contentious, with strong points on both sides (strengths vs. risks). This is for exploring the upside and downside scenarios.
- **Final_Synthesis**: Choose this if the evidence is overwhelming, the risks are minimal, and the unknowns are minor. This means the research is largely complete.
</next_step_options>

<instructions>
1.  Analyze the `<current_hypothesis>`. Pay close attention to the `key_unknowns`.
2.  Choose the most logical `next_step` from the options provided.
3.  Write a brief `rationale` explaining your choice.
4.  List the most important questions from the `key_unknowns` or `risks` as `focus_areas` for the next step.
</instructions>
"""
    return await ai_with_dynamic_params(
        system="You are a research strategist determining the next logical step in an investigation.",
        user=prompt,
        schema=TrajectoryDecision,
        model=model,
        api_key=api_key,
    )


@app.reasoner()
async def execute_deep_dive(
    focus_areas: List[str],
    initial_context: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[ArticleEvidence]:
    """Phase 5a: Conducts targeted research to fill specific knowledge gaps."""
    focus_preview = focus_areas[0] if focus_areas else "key questions"
    app.note(f"Exploring {focus_preview} in more detail…")

    # 1. Generate targeted search queries
    class GeneratedQueries(BaseModel):
        queries: List[str]

    query_gen_prompt = f"Generate 2-3 precise search queries to find factual answers for these topics: {', '.join(focus_areas)}. The queries should be optimized for finding specific data, not general articles."

    generated_queries = await ai_with_dynamic_params(
        system="You are a search query generation expert for financial and technical diligence. Remember, make sure to add sources one or two of them in the search query as well. For instance, if a scientific use archive, if it is market use something, you know, you don't need to add everything just mention the site name and that should be fine, like 'carbon nano tube' ar5iv.labs.arxiv.org or something remember we are using Jina ai search so make it such that the query is searchable in public internet if you are not sure about good sources then dont add and it can come up on its own",
        user=query_gen_prompt,
        schema=GeneratedQueries,
        model=model,
        api_key=api_key,
    )

    # 2. Perform searches and gather content
    search_tasks = [search_web_for_content(q) for q in generated_queries.queries]
    search_results_lists = await asyncio.gather(*search_tasks)
    all_results = [item for sublist in search_results_lists for item in sublist]
    unique_articles_map = {
        article["url"]: article for article in all_results if "url" in article
    }

    # 3. Extract evidence from new sources
    class EvidenceList(BaseModel):
        evidence: List[str]

    evidence_extraction_tasks = []
    articles_for_evidence = list(unique_articles_map.values())[
        :5
    ]  # Limit to top 5 new articles

    for i, article in enumerate(articles_for_evidence):
        extract_prompt = f"""
<task>You are an Intelligence Analyst extracting specific facts.</task>
<topics_of_interest>{', '.join(focus_areas)}</topics_of_interest>
<article_content>{article.get('content', '')[:4000]}</article_content>
<instructions>Extract a list of direct facts or data points from the article that are highly relevant to the topics of interest. Ignore general information.</instructions>
"""
        evidence_extraction_tasks.append(
            ai_with_dynamic_params(
                system="Extract specific, relevant facts from text.",
                user=extract_prompt,
                schema=EvidenceList,
                model=model,
                api_key=api_key,
            )
        )

    extracted_facts_results = await run_in_batches(
        evidence_extraction_tasks, AI_CALL_CONCURRENCY_LIMIT
    )

    # 4. Format as ArticleEvidence
    new_evidence = []
    for i, result in enumerate(extracted_facts_results):
        if result and articles_for_evidence[i]:
            article_ref = articles_for_evidence[i]
            article_id = i + 1000  # Avoid collision with initial IDs
            new_evidence.append(
                ArticleEvidence(
                    article_id=article_id,
                    relevance_summary=f"Deep dive fact-finding for topics: {', '.join(focus_areas)}.",
                    facts=result.evidence,
                    quotes=[],
                )
            )
            # We would also create and add the Article object here in a real implementation

    # Show what we discovered in the deep dive
    if new_evidence and new_evidence[0].facts:
        sample_finding = new_evidence[0].facts[0][:60] + "..."
        app.note(f"Deep dive complete — found insights like: {sample_finding}")
    else:
        app.note(
            f"Deep dive complete — gathered {len(new_evidence)} additional data points"
        )
    return new_evidence


@app.reasoner()
async def execute_multi_perspective_analysis(
    hypothesis: InvestmentHypothesis,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BullBearBaseCase:
    """Phase 5b: Generates bull, bear, and base cases based on the existing hypothesis."""
    # Show what scenarios we're exploring
    strength_preview = (
        hypothesis.supporting_strengths[0][:50] + "..."
        if hypothesis.supporting_strengths
        else "key strengths"
    )
    risk_preview = (
        hypothesis.counter_risks[0][:50] + "..."
        if hypothesis.counter_risks
        else "potential risks"
    )
    app.note(f"Modeling scenarios from {strength_preview} to {risk_preview}…")

    prompt = f"""
<task>
You are a seasoned VC Partner playing devil's advocate in an investment committee. Your role is to articulate the bull, bear, and base case scenarios for a potential investment based on an existing hypothesis.
</task>

<current_hypothesis>
- **Thesis**: {hypothesis.core_thesis}
- **Strengths**: {hypothesis.supporting_strengths}
- **Risks**: {hypothesis.counter_risks}
</current_hypothesis>

<instructions>
Based *only* on the hypothesis provided, write a concise summary for each of the following scenarios:
1.  **bull_case_summary**: The most optimistic outcome. Extrapolate from the `Strengths`. What is the absolute best-case scenario if everything goes right?
2.  **bear_case_summary**: The most pessimistic outcome. Extrapolate from the `Risks`. What does failure look like, and what are the primary causes?
3.  **base_case_summary**: The most realistic, probabilistic outcome. Synthesize the strengths and risks into the most likely trajectory for the company.
</instructions>
"""
    return await ai_with_dynamic_params(
        system="You are a VC Partner articulating bull, bear, and base case investment scenarios.",
        user=prompt,
        schema=BullBearBaseCase,
        model=model,
        api_key=api_key,
    )


# ==============================================================================
# ORCHESTRATOR & PUBLIC API REASONERS
# ==============================================================================


@app.reasoner()
async def assess_research_completeness(
    query: str,
    entities: List[Entity],
    relationships: List[Relationship],
    key_discoveries: List[str],
    hypothesis_confidence: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ResearchQualityScore:
    """Evaluates the completeness and quality of current research state."""

    entity_count = len(entities)
    relationship_count = len(relationships)
    discovery_count = len(key_discoveries)

    prompt = f"""
<mission>
Evaluate research completeness for: "{query}"
</mission>

<current_research_state>
- Entities mapped: {entity_count}
- Relationships discovered: {relationship_count}
- Key insights: {discovery_count}
- Current hypothesis confidence: {hypothesis_confidence}
</current_research_state>

<evaluation_criteria>
**Confidence Assessment (0.0-1.0):**
- 0.8-1.0: Strong evidence, clear patterns, well-connected network
- 0.5-0.7: Moderate evidence, some gaps but workable conclusions
- 0.0-0.4: Insufficient evidence, major gaps, unclear patterns

**Evidence Adequacy:**
- 'sufficient': Multiple sources support key claims, network well-mapped
- 'moderate': Some supporting evidence, partial network coverage
- 'insufficient': Weak evidence base, sparse network connections

**Critical Gaps Assessment:**
- True: Major players missing, key relationships unknown, core questions unanswered
- False: Main elements mapped, relationships clear, minor gaps only
</evaluation_criteria>

<instructions>
Rate the current research quality based on whether we have enough insight to confidently address the original query.
</instructions>
"""

    return await ai_with_dynamic_params(
        system="You are a Research Quality Evaluator. Assess completeness objectively based on evidence strength and network coverage.",
        user=prompt,
        schema=ResearchQualityScore,
        model=model,
        api_key=api_key,
    )


@app.reasoner()
async def identify_knowledge_gaps_batch(
    query: str,
    entities: List[Entity],
    relationships: List[Relationship],
    current_evidence_summary: str,
    batch_focus: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[ResearchGap]:
    """Identifies specific knowledge gaps that need investigation."""

    key_entities = [f"{e.name} ({e.type})" for e in entities[:15]]
    entity_context = ", ".join(key_entities)

    key_relationships = [
        f"{r.source_entity} → {r.target_entity}" for r in relationships[:10]
    ]
    relationship_context = " | ".join(key_relationships)

    prompt = f"""
<mission>
Identify specific knowledge gaps for: "{query}"
Focus area: {batch_focus}
</mission>

<current_knowledge_map>
<key_entities>{entity_context}</key_entities>
<key_relationships>{relationship_context}</key_relationships>
<evidence_summary>{current_evidence_summary}</evidence_summary>
</current_knowledge_map>

<gap_identification_framework>
**Entity Gaps:**
- Missing key players or stakeholders
- Incomplete profiles of important entities
- Isolated entities with no connections

**Relationship Gaps:**
- Missing connections between important entities
- Unclear influence patterns or dependencies
- Unexplored competitive or collaborative dynamics

**Evidence Gaps:**
- Unsupported claims in current analysis
- Missing quantitative data for key assertions
- Lack of recent developments or trends

**Context Gaps:**
- Missing environmental factors or constraints
- Unclear temporal dynamics or causation
- Absent alternative perspectives or scenarios
</gap_identification_framework>

<instructions>
Identify 3-5 specific gaps that would significantly strengthen understanding of the research query. Focus on gaps that:
1. Directly impact ability to answer the original query
2. Could reveal important connections or insights
3. Are potentially addressable through targeted research
</instructions>
"""

    class GapBatch(BaseModel):
        gaps: List[ResearchGap]

    result = await ai_with_dynamic_params(
        system="You are a Knowledge Gap Analyst. Identify specific, actionable research gaps that would strengthen analysis quality.",
        user=prompt,
        schema=GapBatch,
        model=model,
        api_key=api_key,
    )

    return result.gaps


@app.reasoner()
async def generate_targeted_search_queries(
    gaps: List[ResearchGap],
    original_query: str,
    current_entities: List[str],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[GapFillingQuery]:
    """Generates specific search queries to address identified gaps."""

    high_priority_gaps = [g for g in gaps if g.gap_priority == "high"]
    gap_descriptions = [g.gap_description for g in high_priority_gaps[:5]]
    entity_context = ", ".join(current_entities[:10])

    prompt = f"""
<mission>
Generate targeted search queries to fill knowledge gaps.
Original research: "{original_query}"
</mission>

<gaps_to_address>
{chr(10).join([f"- {gap}" for gap in gap_descriptions])}
</gaps_to_address>

<search_context>
Known entities: {entity_context}
</search_context>

<query_generation_principles>
**Specificity:** Create precise queries that target exact information needs
**Discoverability:** Use terms likely to appear in relevant sources
**Diversity:** Generate queries that will find different types of sources
**Efficiency:** Each query should have high probability of filling gaps
</query_generation_principles>

<instructions>
Create 3-4 high-precision search queries that directly address the most critical gaps. Each query should:
1. Target specific missing information
2. Be likely to find authoritative sources
3. Fill gaps that strengthen overall analysis
</instructions>
"""

    class QueryBatch(BaseModel):
        queries: List[GapFillingQuery]

    result = await ai_with_dynamic_params(
        system="You are a Targeted Search Strategist. Design precise queries that efficiently fill critical knowledge gaps.",
        user=prompt,
        schema=QueryBatch,
        model=model,
        api_key=api_key,
    )

    return result.queries


@app.reasoner()
async def decide_iteration_continuation(
    quality_score: ResearchQualityScore,
    gaps_identified: List[ResearchGap],
    current_iteration: int,
    max_iterations: int,
    query: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LoopDecision:
    """Decides whether to continue with another research iteration."""

    high_priority_gaps = [g for g in gaps_identified if g.gap_priority == "high"]
    total_gaps = len(gaps_identified)
    critical_gaps = len(high_priority_gaps)

    prompt = f"""
<mission>
Decide whether to continue iterative research for: "{query}"
</mission>

<current_state>
- Iteration: {current_iteration}/{max_iterations}
- Research confidence: {quality_score.confidence_score}
- Evidence adequacy: {quality_score.evidence_adequacy}
- Critical gaps remain: {quality_score.critical_gaps_present}
- Total gaps identified: {total_gaps}
- High-priority gaps: {critical_gaps}
</current_state>

<continuation_criteria>
**Continue if:**
- Confidence < 0.75 AND iteration < max_iterations
- Critical gaps present AND addressable gaps identified
- Evidence inadequacy AND meaningful gaps to fill

**Stop if:**
- Confidence >= 0.8 OR evidence_adequacy == 'sufficient'
- No high-priority gaps remaining
- Max iterations reached
- Diminishing returns likely
</continuation_criteria>

<instructions>
Make continuation decision based on current research quality vs. requirements and remaining iteration budget.
</instructions>
"""

    return await ai_with_dynamic_params(
        system="You are a Research Strategy Decision Maker. Optimize research effort allocation based on quality requirements and remaining resources.",
        user=prompt,
        schema=LoopDecision,
        model=model,
        api_key=api_key,
    )


def format_hypothesis_as_probes(hypothesis: InvestmentHypothesis) -> List[InquiryProbe]:
    """Helper to convert hypothesis unknowns into standard 'InquiryProbe' objects."""
    probes = []
    for unknown in hypothesis.key_unknowns:
        probes.append(
            InquiryProbe(
                question=f"Investigate the key unknown: {unknown}",
                rationale="This was identified as a critical gap in the initial investment hypothesis.",
                suggested_method="Targeted web search for specific data points, financial reports, or technical documentation.",
            )
        )
    return probes


@app.reasoner()
async def generate_adaptive_search_streams(
    core_subject: str,
    key_question: str,
    query_type: str,
    num_parallel_streams: int = 2,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict]:
    """Generates adaptive search streams based on query type and domain."""

    temporal_context = get_temporal_context("search")

    prompt = f"""
<mission>
You are a Search Strategy Architect. Design {num_parallel_streams} parallel intelligence gathering streams that comprehensively map the research landscape for any domain or subject matter.
</mission>

{temporal_context}

<research_context>
<core_subject>{core_subject}</core_subject>
<key_research_question>{key_question}</key_research_question>
<query_classification>{query_type}</query_classification>
</research_context>

<universal_intelligence_streams>
**Stream Design Philosophy:**
Create complementary but non-overlapping streams that together provide 360-degree coverage of the research landscape.

**Core Intelligence Categories:**

**1. Primary/Direct Stream**:
- Focus: Direct information about the core subject
- Sources: Official sources, primary documentation, recent developments
- Goal: Establish baseline understanding and current status

**2. Contextual/Environment Stream**:
- Focus: Operating environment, external factors, broader landscape
- Sources: Industry reports, regulatory information, macro trends
- Goal: Understand forces and constraints shaping the subject

**3. Network/Ecosystem Stream**:
- Focus: Relationships, stakeholders, connected entities
- Sources: Partnership announcements, organizational information, network analysis
- Goal: Map the relationship ecosystem and influence patterns

**4. Performance/Capability Stream**:
- Focus: Capabilities, performance metrics, technical/operational assessment
- Sources: Performance data, technical documentation, capability assessments
- Goal: Evaluate strengths, weaknesses, and differentiation factors
</universal_intelligence_streams>

<adaptive_query_optimization>
**Search Query Design Principles:**
- Use specific, targeted terms that maximize relevant results
- Include temporal indicators (2024, 2025, recent, latest) for current information
- Combine subject with contextual terms for richer results
- Avoid generic terms that produce low-signal results
- Include both broad and specific queries for comprehensive coverage
- For instance, if a scientific use archive, if it is market use something, you know, you don't need to add everything just mention the site name and that should be fine, like 'carbon nano tube' ar5iv.labs.arxiv.org or something remember we are using Jina ai search so make it such that the query is searchable in public internet if you are not sure about good sources then dont add and it can come up on its own

**Domain Adaptation:**
- Business domain: Include financial, strategic, competitive terms
- Technology domain: Include technical specs, innovation, adoption terms
- Policy domain: Include regulatory, implementation, stakeholder terms
- Market domain: Include size, growth, segmentation, trend terms
</adaptive_query_optimization>

<instructions>
Design {num_parallel_streams} intelligence gathering streams for comprehensive research coverage:

1. **Stream Naming**: Create descriptive names that reflect the intelligence focus
2. **Query Generation**: Create 3-4 optimized search queries per stream that will find diverse, high-quality sources
3. **Analysis Focus**: Define the analytical perspective each stream should apply

Ensure streams are:
- **Comprehensive**: Together they cover all aspects needed to understand the subject
- **Complementary**: Each stream provides unique intelligence value
- **Optimized**: Search queries are crafted to find the best available sources
- **Adaptive**: Framework adapts to the specific domain and query type

Focus on maximizing information discovery and source diversity.
</instructions>
"""

    class SearchStream(BaseModel):
        stream_name: str
        search_queries: List[str]
        analysis_focus: str

    class AdaptiveSearchStreams(BaseModel):
        streams: List[SearchStream]

    result = await ai_with_dynamic_params(
        system="You are an Adaptive Search Strategy Architect who designs comprehensive, domain-agnostic intelligence gathering frameworks for any research subject.",
        user=prompt,
        schema=AdaptiveSearchStreams,
        model=model,
        api_key=api_key,
    )

    return [stream.dict() for stream in result.streams]


@app.reasoner()
async def execute_intelligence_stream_comprehensive(
    stream_name: str,
    search_queries: List[str],
    analysis_focus: str,
    subject: str,
    key_question: str,
    start_article_id: int,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> StreamOutput:
    """Comprehensive intelligence stream execution with full article/evidence collection."""

    # Show what specific angle we're exploring
    focus_preview = (
        analysis_focus[:50] + "..." if len(analysis_focus) > 50 else analysis_focus
    )
    app.note(
        f"Exploring {subject} from {stream_name.lower()} perspective: {focus_preview}"
    )

    # Execute searches in parallel
    search_tasks = [search_web_for_content(query) for query in search_queries]
    search_results_lists = await asyncio.gather(*search_tasks)

    # Process and deduplicate articles
    unique_articles_map = {
        res["url"]: res
        for res in [item for sublist in search_results_lists for item in sublist]
        if "url" in res and res.get("content", "").strip()
    }

    source_articles: List[Article] = []
    article_id_counter = start_article_id
    for url, result in list(unique_articles_map.items())[:MAX_ARTICLES_PER_TASK]:
        content = result.get("content", "")
        if content.strip():
            source_articles.append(
                Article(
                    id=article_id_counter,
                    title=result.get("title", "No Title"),
                    url=url,
                    content=content,
                    content_hash=create_content_hash(content),
                )
            )
            article_id_counter += 1

    # Parallel evidence extraction with enhanced prompts
    async def extract_evidence_comprehensive(
        article: Article,
    ) -> Optional[ArticleEvidence]:
        temporal_context = get_temporal_context("evidence")

        prompt = f"""
<task>
You are an Intelligence Analyst specializing in comprehensive evidence extraction. Your mission is to extract ALL relevant information that could inform the research objective.
</task>

{temporal_context}

<research_context>
<original_query>{key_question}</original_query>
<research_subject>{subject}</research_subject>
<intelligence_focus>{stream_name} - {analysis_focus}</intelligence_focus>
</research_context>

<article_source>
<title>{article.title}</title>
<url>{article.url}</url>
<content>{article.content[:4000]}</content>
</article_source>

<extraction_framework>
**Relevance Assessment:**
- How does this article inform understanding of the research subject?
- What specific aspects of the intelligence focus does it address?
- Why would this information matter for the overall research objective?

**Fact Extraction (Comprehensive):**
Extract ALL discrete, verifiable facts including:
- Quantitative data (numbers, percentages, dates, amounts)
- Qualitative indicators (assessments, evaluations, descriptions)
- Relationships and connections mentioned
- Strategic developments and changes
- Technical specifications or capabilities
- Risk factors and challenges identified
- Opportunities and advantages noted

**Quote Extraction (Selective):**
Extract impactful quotes that:
- Provide expert opinions or assessments
- Reveal strategic intentions or decisions
- Demonstrate market sentiment or reactions
- Support or challenge key claims
- Offer unique insights or perspectives

**Completeness Focus:**
- Extract 8-15 facts minimum if the article is relevant
- Don't leave important information unextracted
- Include context that makes facts meaningful
- Focus on information density and comprehensiveness
</extraction_framework>

<instructions>
Perform comprehensive evidence extraction optimized for the {stream_name} intelligence stream. Extract maximum relevant information while maintaining quality and accuracy. Base the relevance_summary on how this article specifically contributes to understanding {subject} in the context of {analysis_focus}.
</instructions>
"""

        class ComprehensiveEvidence(BaseModel):
            relevance_summary: str
            facts: List[str]
            quotes: List[str]

        try:
            ai_output = await ai_with_dynamic_params(
                system=f"You are a comprehensive Intelligence Analyst specializing in {analysis_focus}. Your goal is maximum information extraction while maintaining accuracy and relevance.",
                user=prompt,
                schema=ComprehensiveEvidence,
                model=model,
                api_key=api_key,
            )
            return ArticleEvidence(article_id=article.id, **ai_output.dict())
        except Exception as e:
            return None

    # Extract evidence from all articles in parallel
    extraction_tasks = [
        extract_evidence_comprehensive(article) for article in source_articles
    ]
    article_evidence_results = await run_in_batches(
        extraction_tasks, AI_CALL_CONCURRENCY_LIMIT
    )
    article_evidence: List[ArticleEvidence] = [
        ev for ev in article_evidence_results if ev
    ]

    # Show what we discovered in this comprehensive stream
    if article_evidence:
        sample_insights = []
        for ev in article_evidence[:2]:
            if ev.facts:
                sample_insights.append(ev.facts[0][:40] + "...")
        if sample_insights:
            insights_preview = " | ".join(sample_insights)
            app.note(f"Discovered insights about {subject}: {insights_preview}")
        else:
            app.note(f"Completed {stream_name} analysis of {subject}")
    else:
        app.note(f"Finished {stream_name} research on {subject}")

    # Return minimal synthesized intel (detailed synthesis happens later)
    return StreamOutput(
        stream_type=stream_name,
        synthesized_intel={
            "analysis_focus": analysis_focus,
            "evidence_count": len(article_evidence),
        },
        source_articles=source_articles,
        article_evidence=article_evidence,
    )


@app.reasoner()
async def prepare_research_package(
    query: str,
    mode: str = "general",
    research_focus: int = 3,
    research_scope: int = 3,
    max_research_loops: int = 3,
    num_parallel_streams: int = 2,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ModeAwareResearchResponse:
    """
    Iterative research orchestrator with multi-stream intelligence gathering.
    """
    start_time = time.time()
    app.note(f"Beginning research on: '{query[:60]}{'...' if len(query) > 60 else ''}'")

    # === INITIALIZATION ===
    classification = await classify_query_adaptive(query, model=model, api_key=api_key)
    app.note(
        f"Research classified as: {classification.query_type.replace('_', ' ').lower()}"
    )

    # Initialize research state
    all_source_articles: List[Article] = []
    all_article_evidence: List[ArticleEvidence] = []
    current_entities: List[Entity] = []
    current_relationships: List[Relationship] = []
    current_discoveries: List[str] = []
    iteration_summaries: List[str] = []

    # === ITERATIVE RESEARCH LOOP ===
    iteration = 1
    quality_score = None
    for iteration in range(1, max_research_loops + 1):
        iteration_start = time.time()
        app.note(
            f"Finding new gaps in our understanding and making higher order connections..."
        )

        # Determine search strategy for this iteration
        if iteration == 1:
            # Initial comprehensive search
            search_streams = await generate_adaptive_search_streams(
                classification.core_subject,
                classification.key_question,
                classification.query_type,
                num_parallel_streams=num_parallel_streams,
                model=model,
                api_key=api_key,
            )
            # Show what research angles we're exploring
            if search_streams:
                stream_names = [
                    stream.get("stream_name", "research").lower()
                    for stream in search_streams[:2]
                ]
                streams_preview = " and ".join(stream_names)
                if len(search_streams) > 2:
                    streams_preview += "…"
                app.note(
                    f"Analyzing {classification.core_subject} from multiple angles: {streams_preview}…"
                )
            else:
                app.note(
                    f"Beginning comprehensive research on {classification.core_subject}…"
                )
        else:
            # Gap-targeted search for subsequent iterations
            gaps = await identify_knowledge_gaps_batch(
                query,
                current_entities,
                current_relationships,
                (
                    " | ".join(current_discoveries[:3])
                    if current_discoveries
                    else "Initial research phase"
                ),
                f"Iteration {iteration} gap analysis",
                model=model,
                api_key=api_key,
            )

            gap_queries = await generate_targeted_search_queries(
                gaps,
                query,
                [e.name for e in current_entities[:10]],
                model=model,
                api_key=api_key,
            )

            # Convert gap queries to search streams format
            search_streams = [
                {
                    "stream_name": f"Gap_Filling_{i+1}",
                    "search_queries": [gq.search_query],
                    "analysis_focus": gq.expected_insights,
                }
                for i, gq in enumerate(gap_queries)
            ]

            if search_streams:
                app.note("Targeting specific knowledge gaps we've identified…")
            else:
                app.note("No significant gaps identified - preparing for completion")
                break

        # === PARALLEL STREAM EXECUTION ===
        stream_results: List[StreamOutput] = []
        article_id_offset = len(all_source_articles)

        for i, stream in enumerate(search_streams):
            stream_result = await execute_intelligence_stream_comprehensive(
                stream["stream_name"],
                stream.get("search_queries", []),
                stream["analysis_focus"],
                classification.core_subject,
                classification.key_question,
                article_id_offset + (i * 100),
                model,
                api_key,
            )
            stream_results.append(stream_result)

        # === EVIDENCE COLLECTION ===
        iteration_articles = []
        iteration_evidence = []

        for stream_result in stream_results:
            iteration_articles.extend(stream_result.source_articles)
            iteration_evidence.extend(stream_result.article_evidence)

        # Deduplicate new articles
        new_unique_articles = []
        existing_hashes = {a.content_hash for a in all_source_articles}

        for article in iteration_articles:
            if article.content_hash not in existing_hashes:
                new_unique_articles.append(article)
                existing_hashes.add(article.content_hash)

        all_source_articles.extend(new_unique_articles)
        all_article_evidence.extend(iteration_evidence)

        # Show meaningful progress to user
        if new_unique_articles and iteration_evidence:
            sample_source = (
                new_unique_articles[0].title[:50] + "..."
                if new_unique_articles[0].title
                else "new research"
            )
            app.note(f"Discovered {sample_source} and related insights")
        elif new_unique_articles:
            app.note(f"Found additional sources on {classification.core_subject}")
        elif iteration_evidence:
            app.note(f"Extracted new insights about {classification.core_subject}")
        else:
            app.note(f"Completed research cycle for {classification.core_subject}")

        # === NETWORK ANALYSIS ===
        iteration_entities = await extract_entities_from_evidence_comprehensive(
            all_article_evidence,
            classification.core_subject,
            query,
            existing_entities=current_entities if iteration > 1 else None,
            model=model,
            api_key=api_key,
        )

        iteration_relationships = await extract_relationships_comprehensive(
            all_article_evidence,
            iteration_entities,
            query,
            model=model,
            api_key=api_key,
        )

        # Track network growth
        entity_growth = len(iteration_entities) - len(current_entities)
        relationship_growth = len(iteration_relationships) - len(current_relationships)

        current_entities = iteration_entities
        current_relationships = iteration_relationships

        # === DISCOVERY SYNTHESIS ===
        iteration_discoveries = await synthesize_key_discoveries_meta(
            query,
            all_article_evidence,
            current_entities,
            current_relationships,
            model=model,
            api_key=api_key,
        )

        current_discoveries = iteration_discoveries

        # Record iteration summary
        iteration_time = time.time() - iteration_start
        iteration_summaries.append(
            f"Iteration {iteration}: +{entity_growth} entities, +{relationship_growth} relationships ({iteration_time:.1f}s)"
        )

        if entity_growth > 0 or relationship_growth > 0:
            app.note(
                f"Discovered new players and connections in the {classification.core_subject} ecosystem"
            )

        # === QUALITY ASSESSMENT & CONTINUATION DECISION ===
        hypothesis = await generate_adaptive_hypothesis(
            query,
            classification.query_type,
            current_discoveries,
            current_entities,
            current_relationships,
            model=model,
            api_key=api_key,
        )

        quality_score = await assess_research_completeness(
            query,
            current_entities,
            current_relationships,
            current_discoveries,
            f"Confidence based on {len(all_source_articles)} sources",
            model=model,
            api_key=api_key,
        )

        if quality_score:
            if quality_score.confidence_score >= 0.8:
                app.note(
                    f"High confidence in our analysis of {classification.core_subject}"
                )
            elif quality_score.confidence_score >= 0.6:
                app.note(
                    f"Strong foundation for our analysis of {classification.core_subject}"
                )
            else:
                app.note(
                    f"Initial insights gathered on {classification.core_subject} — continuing research"
                )
        else:
            app.note("Evaluating analysis quality…")

        # === CONTINUATION DECISION ===
        if iteration < max_research_loops:
            remaining_gaps = await identify_knowledge_gaps_batch(
                query,
                current_entities,
                current_relationships,
                " | ".join(current_discoveries[:3]),
                "Continuation assessment",
                model=model,
                api_key=api_key,
            )

            loop_decision = await decide_iteration_continuation(
                quality_score,
                remaining_gaps,
                iteration,
                max_research_loops,
                query,
                model=model,
                api_key=api_key,
            )

            if not loop_decision.should_continue:
                app.note(
                    f"Research complete — {loop_decision.termination_reason.lower()}"
                )
                break
            else:
                app.note(f"Expanding research to explore: {loop_decision.focus_areas}")

    # === FINAL PACKAGING ===
    final_hypothesis = await generate_adaptive_hypothesis(
        query,
        classification.query_type,
        current_discoveries,
        current_entities,
        current_relationships,
        model=model,
        api_key=api_key,
    )

    inquiry_probes = await generate_adaptive_inquiry_probes(
        query,
        final_hypothesis,
        current_entities,
        current_relationships,
        model=model,
        api_key=api_key,
    )

    final_package = UniversalResearchPackage(
        query=query,
        core_thesis=final_hypothesis.core_thesis,
        key_discoveries=current_discoveries,
        confidence_assessment=(
            f"Iterative analysis across {iteration} cycles. Network: {len(current_entities)} entities, {len(current_relationships)} relationships. Sources: {len(all_source_articles)}. Quality: {quality_score.confidence_score:.2f}"
            if quality_score
            else f"Iterative analysis across {iteration} cycles. Network: {len(current_entities)} entities, {len(current_relationships)} relationships. Sources: {len(all_source_articles)}."
        ),
        entities=current_entities,
        relationships=current_relationships,
        observed_causal_chains=final_hypothesis.supporting_strengths,
        hypothesized_implications=final_hypothesis.counter_risks,
        next_inquiry_probes=inquiry_probes,
        source_articles=all_source_articles,
        article_evidence=all_article_evidence,
    )

    execution_time = time.time() - start_time
    app.note("Analysis complete — ready to present findings")

    return ModeAwareResearchResponse(
        mode=mode,
        version="5.0-Iterative-Meta",
        research_package=final_package.dict(),
        metadata={
            "query": query,
            "created_at": datetime.datetime.now().isoformat(),
            "execution_time": execution_time,
            "iterations_completed": iteration,
            "iteration_summaries": iteration_summaries,
            "final_quality_score": (
                quality_score.confidence_score if quality_score else 0.0
            ),
            "total_entities": len(current_entities),
            "total_relationships": len(current_relationships),
            "total_sources": len(all_source_articles),
        },
    )


@app.reasoner()
async def continue_research(
    previous_package: dict,
    sub_query: str,
    mode: str = "general",
    research_focus: int = 3,
    research_scope: int = 3,
    max_research_loops: int = 2,
    num_parallel_streams: int = 2,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ModeAwareResearchResponse:
    """
    Continues research from a previous package with additional queries.
    """
    start_time = time.time()

    # Extract previous research state
    if "research_package" in previous_package:
        package_data = previous_package["research_package"]
    else:
        package_data = previous_package

    prev_pkg = UniversalResearchPackage(**package_data)
    app.note(
        f"Expanding research scope: '{sub_query[:50]}{'...' if len(sub_query) > 50 else ''}'"
    )

    # === CONTINUATION STRATEGY ===
    sub_classification = await classify_query_adaptive(
        sub_query, model=model, api_key=api_key
    )
    app.note(
        f"Expansion focus: {sub_classification.query_type.replace('_', ' ').lower()}"
    )

    # Initialize expanded research state from previous package
    all_source_articles = prev_pkg.source_articles[:]
    all_article_evidence = prev_pkg.article_evidence[:]
    current_entities = prev_pkg.entities[:]
    current_relationships = prev_pkg.relationships[:]
    current_discoveries = prev_pkg.key_discoveries[:]
    expansion_summaries: List[str] = []

    # === ITERATIVE EXPANSION LOOP ===
    for iteration in range(1, max_research_loops + 1):
        iteration_start = time.time()
        app.note("Deepening our analysis with additional research…")

        if iteration == 1:
            # Initial expansion based on sub-query
            expansion_streams = await generate_adaptive_search_streams(
                sub_classification.core_subject,
                sub_classification.key_question,
                sub_classification.query_type,
                num_parallel_streams=num_parallel_streams,
                model=model,
                api_key=api_key,
            )
            app.note("Exploring new research angles for deeper insights…")
        else:
            # Gap-targeted expansion
            expansion_gaps = await identify_knowledge_gaps_batch(
                f"Original: {prev_pkg.query} | Expansion: {sub_query}",
                current_entities,
                current_relationships,
                " | ".join(current_discoveries[:3]),
                f"Expansion iteration {iteration}",
                model=model,
                api_key=api_key,
            )

            gap_queries = await generate_targeted_search_queries(
                expansion_gaps,
                sub_query,
                [e.name for e in current_entities[:10]],
                model=model,
                api_key=api_key,
            )

            expansion_streams = [
                {
                    "stream_name": f"Expansion_Gap_{i+1}",
                    "search_queries": [gq.search_query],
                    "analysis_focus": gq.expected_insights,
                }
                for i, gq in enumerate(gap_queries)
            ]

            if not expansion_streams:
                app.note("Expansion research converged - no significant gaps remain")
                break

        # === EXECUTE EXPANSION STREAMS ===
        max_existing_id = max([a.id for a in all_source_articles], default=0)
        expansion_results: List[StreamOutput] = []

        for i, stream in enumerate(expansion_streams):
            stream_result = await execute_intelligence_stream_comprehensive(
                f"Expansion_{stream['stream_name']}",
                stream.get("search_queries", []),
                f"Expansion focus: {stream['analysis_focus']}",
                sub_classification.core_subject,
                sub_classification.key_question,
                max_existing_id + 1000 + (i * 100),
                model,
                api_key,
            )
            expansion_results.append(stream_result)

        # === COLLECT EXPANSION EVIDENCE ===
        new_articles = []
        new_evidence = []

        for result in expansion_results:
            new_articles.extend(result.source_articles)
            new_evidence.extend(result.article_evidence)

        # Deduplicate against existing
        existing_hashes = {a.content_hash for a in all_source_articles}
        unique_new_articles = [
            a for a in new_articles if a.content_hash not in existing_hashes
        ]

        all_source_articles.extend(unique_new_articles)
        all_article_evidence.extend(new_evidence)

        # Show meaningful expansion progress
        if unique_new_articles and new_evidence:
            sample_source = (
                unique_new_articles[0].title[:50] + "..."
                if unique_new_articles[0].title
                else "additional research"
            )
            app.note(f"Expanding with {sample_source} and related findings")
        elif unique_new_articles:
            app.note(f"Found additional sources for {sub_classification.core_subject}")
        elif new_evidence:
            app.note(f"Extracted new insights about {sub_classification.core_subject}")
        else:
            app.note(f"Completed expansion cycle for {sub_classification.core_subject}")

        # === NETWORK EXPANSION ===
        expanded_entities = await extract_entities_from_evidence_comprehensive(
            all_article_evidence,
            sub_classification.core_subject,
            f"Original: {prev_pkg.query} | Expansion: {sub_query}",
            existing_entities=current_entities,
            model=model,
            api_key=api_key,
        )

        expanded_relationships = await extract_relationships_comprehensive(
            all_article_evidence,
            expanded_entities,
            f"Original: {prev_pkg.query} | Expansion: {sub_query}",
            model=model,
            api_key=api_key,
        )

        # Track expansion
        entity_expansion = len(expanded_entities) - len(current_entities)
        relationship_expansion = len(expanded_relationships) - len(
            current_relationships
        )

        current_entities = expanded_entities
        current_relationships = expanded_relationships

        # === DISCOVERY EXPANSION ===
        expanded_discoveries = await synthesize_key_discoveries_meta(
            f"Original: {prev_pkg.query} | Expansion: {sub_query}",
            all_article_evidence,
            current_entities,
            current_relationships,
            model=model,
            api_key=api_key,
        )

        # Merge discoveries from previous and new research
        current_discoveries = prev_pkg.key_discoveries + [
            f"[EXPANSION] {discovery}"
            for discovery in expanded_discoveries
            if discovery not in prev_pkg.key_discoveries
        ]

        # Record expansion summary
        iteration_time = time.time() - iteration_start
        expansion_summaries.append(
            f"Expansion {iteration}: +{entity_expansion} entities, +{relationship_expansion} relationships ({iteration_time:.1f}s)"
        )

        if entity_expansion > 0 or relationship_expansion > 0:
            # Show network growth in user terms
            if entity_expansion > 0 and relationship_expansion > 0:
                app.note(
                    f"Discovered new players and connections in the {sub_classification.core_subject} ecosystem"
                )
            elif entity_expansion > 0:
                app.note(
                    f"Identified additional key players around {sub_classification.core_subject}"
                )
            elif relationship_expansion > 0:
                app.note(f"Uncovered new connections between existing players")

        # === EXPANSION QUALITY CHECK ===
        if iteration < max_research_loops:
            expansion_quality = await assess_research_completeness(
                sub_query,
                current_entities,
                current_relationships,
                current_discoveries,
                f"Expansion confidence based on {len(all_source_articles)} total sources",
                model=model,
                api_key=api_key,
            )

            remaining_gaps = await identify_knowledge_gaps_batch(
                sub_query,
                current_entities,
                current_relationships,
                " | ".join(expanded_discoveries[:3]),
                "Expansion continuation check",
                model=model,
                api_key=api_key,
            )

            continuation_decision = await decide_iteration_continuation(
                expansion_quality,
                remaining_gaps,
                iteration,
                max_research_loops,
                sub_query,
                model=model,
                api_key=api_key,
            )

            if not continuation_decision.should_continue:
                break

    # === FINAL EXPANSION PACKAGING ===
    enhanced_hypothesis = await generate_adaptive_hypothesis(
        f"Comprehensive: {prev_pkg.query} | Enhanced: {sub_query}",
        sub_classification.query_type,
        current_discoveries,
        current_entities,
        current_relationships,
        model=model,
        api_key=api_key,
    )

    enhanced_probes = await generate_adaptive_inquiry_probes(
        f"Original: {prev_pkg.query} | Expansion: {sub_query}",
        enhanced_hypothesis,
        current_entities,
        current_relationships,
        model=model,
        api_key=api_key,
    )

    enhanced_package = UniversalResearchPackage(
        query=prev_pkg.query,  # Keep original query
        core_thesis=enhanced_hypothesis.core_thesis,
        key_discoveries=current_discoveries,
        confidence_assessment=f"Comprehensive analysis of {sub_classification.core_subject} with extensive research coverage. Analysis strengthened through multiple research cycles with diverse sources.",
        entities=current_entities,
        relationships=current_relationships,
        observed_causal_chains=enhanced_hypothesis.supporting_strengths,
        hypothesized_implications=enhanced_hypothesis.counter_risks,
        next_inquiry_probes=enhanced_probes,
        source_articles=all_source_articles,
        article_evidence=all_article_evidence,
    )

    execution_time = time.time() - start_time

    # Calculate growth metrics
    original_entities = len(prev_pkg.entities)
    original_relationships = len(prev_pkg.relationships)
    final_entity_growth = len(current_entities) - original_entities
    final_relationship_growth = len(current_relationships) - original_relationships

    app.note("Research complete — comprehensive analysis ready")

    return ModeAwareResearchResponse(
        mode=mode,
        version="5.0-Iterative-Expansion",
        research_package=enhanced_package.dict(),
        metadata={
            "original_query": prev_pkg.query,
            "expansion_query": sub_query,
            "created_at": datetime.datetime.now().isoformat(),
            "execution_time": execution_time,
            "expansion_iterations": iteration,
            "expansion_summaries": expansion_summaries,
            "network_growth": {
                "entities_added": final_entity_growth,
                "relationships_added": final_relationship_growth,
                "total_entities": len(current_entities),
                "total_relationships": len(current_relationships),
                "total_sources": len(all_source_articles),
            },
        },
    )


@app.reasoner()
async def generate_adaptive_inquiry_probes(
    query: str,
    hypothesis: InvestmentHypothesis,
    entities: List[Entity],
    relationships: List[Relationship],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[InquiryProbe]:
    """Generates targeted inquiry probes based on hypothesis gaps and network analysis."""

    # Analyze network structure for probe opportunities
    entity_connections = {}
    for rel in relationships:
        entity_connections[rel.source_entity] = (
            entity_connections.get(rel.source_entity, 0) + 1
        )
        entity_connections[rel.target_entity] = (
            entity_connections.get(rel.target_entity, 0) + 1
        )

    # Find entities with high importance but low connections (investigation opportunities)
    isolated_entities = [
        e
        for e in entities
        if entity_connections.get(e.name, 0) < 2
        and len(e.summary) > 50  # Substantial entities with few connections
    ]

    # Find relationship gaps (entities that should be connected but aren't)
    high_importance_entities = [
        e.name for e in entities[:10]
    ]  # Top entities by importance

    entities_xml = "\n".join(
        [
            f'<entity name="{e.name}" type="{e.type}" connections="{entity_connections.get(e.name, 0)}">{e.summary}</entity>'
            for e in entities[:15]
        ]
    )

    relationships_xml = "\n".join(
        [
            f"<relationship>{r.source_entity} {r.description} {r.target_entity}</relationship>"
            for r in relationships[:12]
        ]
    )

    isolated_xml = "\n".join(
        [
            f'<isolated_entity name="{e.name}" type="{e.type}">{e.summary}</isolated_entity>'
            for e in isolated_entities[:5]
        ]
    )

    prompt = f"""
<mission>
You are a Strategic Inquiry Specialist. Your task is to generate targeted research probes that will strengthen understanding and resolve critical knowledge gaps.
</mission>

<research_context>
<original_query>{query}</original_query>
<current_hypothesis>
- **Thesis**: {hypothesis.core_thesis}
- **Strengths**: {hypothesis.supporting_strengths}
- **Risks**: {hypothesis.counter_risks}
- **Current Unknowns**: {hypothesis.key_unknowns}
</current_hypothesis>
</research_context>

<network_analysis>
<extracted_entities>
{entities_xml}
</extracted_entities>

<discovered_relationships>
{relationships_xml}
</discovered_relationships>

<isolated_entities>
{isolated_xml}
</isolated_entities>
</network_analysis>

<probe_generation_framework>
**Hypothesis Strengthening Probes:**
- Address the specific unknowns identified in the hypothesis
- Validate or challenge the supporting strengths
- Investigate and quantify the identified risks
- Test the core thesis through targeted investigation

**Network Completion Probes:**
- Investigate isolated entities to discover their hidden connections
- Explore missing relationships between important entities
- Map influence patterns and dependency chains not yet captured
- Understand system boundaries and external connections

**Strategic Validation Probes:**
- Verify critical assumptions underlying the hypothesis
- Investigate alternative explanations for observed patterns
- Explore potential future scenarios and their implications
- Validate the sustainability of identified advantages or positions

**Evidence Gap Probes:**
- Gather quantitative data to support qualitative assessments
- Find primary sources to validate secondary information
- Investigate contradictory evidence or alternative perspectives
- Explore temporal dynamics and trend validation
</probe_generation_framework>

<instructions>
Generate 4-7 targeted inquiry probes that will:

1. **Address Critical Unknowns**: Directly investigate the knowledge gaps in the hypothesis
2. **Strengthen Network Understanding**: Explore connections for isolated entities and missing relationships
3. **Validate Strategic Assumptions**: Test key assumptions underlying the core thesis
4. **Enhance Evidence Base**: Gather additional evidence for uncertain or controversial claims

Each probe should include:
- **Question**: A clear, specific, actionable research question
- **Rationale**: Why this investigation would strengthen understanding
- **Suggested Method**: How to approach answering this question

Prioritize probes that would most significantly impact confidence in the hypothesis or reveal systemic insights.
</instructions>
"""

    class AdaptiveInquiryProbes(BaseModel):
        probes: List[InquiryProbe]

    result = await ai_with_dynamic_params(
        system="You are a Strategic Inquiry Specialist who generates targeted research probes to strengthen understanding and resolve critical knowledge gaps in any domain.",
        user=prompt,
        schema=AdaptiveInquiryProbes,
        model=model,
        api_key=api_key,
    )

    return result.probes


@app.reasoner()
async def classify_query_adaptive(
    query: str, model: Optional[str] = None, api_key: Optional[str] = None
) -> VCQueryClassification:
    """Adaptive query classification that determines optimal research approach for any domain."""

    temporal_context = get_temporal_context("classification")

    prompt = f"""
<mission>
You are a Meta-Research Strategy Classifier. Your expertise is analyzing any research query and determining the most effective intelligence gathering approach, regardless of domain.
</mission>

{temporal_context}

<adaptive_classification_framework>
**Universal Query Types (Domain-Agnostic):**

**Market_Intelligence**: Broad questions about market dynamics, industry trends, sector analysis
- Examples: "AI market growth", "renewable energy landscape", "fintech adoption trends"

**Entity_Analysis**: Deep investigation of specific organizations, people, technologies, or concepts
- Examples: "Analysis of Tesla's strategy", "Assessment of GPT-4 capabilities", "Elizabeth Holmes leadership"

**Competitive_Analysis**: Comparative analysis between entities, options, or approaches
- Examples: "Tesla vs BYD comparison", "React vs Vue frameworks", "Traditional vs digital banking"

**Technology_Assessment**: Evaluation of technical capabilities, innovations, or implementations
- Examples: "Quantum computing readiness", "5G infrastructure maturity", "AI safety measures"

**Strategic_Assessment**: Analysis of strategies, decisions, positioning, or approaches
- Examples: "Netflix content strategy", "EU AI regulation approach", "Apple's privacy positioning"

**Trend_Analysis**: Investigation of patterns, movements, developments, or future directions
- Examples: "Remote work trends", "Cryptocurrency adoption patterns", "Climate tech developments"
</adaptive_classification_framework>

<user_query>{query}</user_query>

<classification_process>
**Step 1: Domain Identification**
Identify the primary domain (business, technology, policy, science, etc.) without forcing it into VC categories.

**Step 2: Intent Analysis**
Determine what the user wants to understand:
- Broad landscape understanding?
- Specific entity deep-dive?
- Comparative evaluation?
- Technical assessment?
- Strategic analysis?
- Trend investigation?

**Step 3: Subject Extraction**
Identify the core subject being investigated (company, technology, market, person, concept, etc.).

**Step 4: Question Refinement**
Transform the query into a clear, actionable research directive that maintains the user's intent while optimizing for comprehensive investigation.
</classification_process>

<instructions>
Analyze the user query and provide:
1. **query_type**: The classification that best matches the research intent (use the universal types above)
2. **core_subject**: The primary entity, concept, or domain being investigated
3. **key_question**: A refined, actionable research directive that captures the core intent for comprehensive investigation

Make the classification adaptive to any domain while maintaining analytical rigor.
</instructions>
"""

    return await ai_with_dynamic_params(
        system="You are a Meta-Research Strategy Classifier who determines optimal intelligence gathering approaches for any domain. You specialize in identifying research intent and designing comprehensive investigation strategies.",
        user=prompt,
        schema=VCQueryClassification,  # Reuse schema but treat it as universal
        model=model,
        api_key=api_key,
    )


@app.reasoner()
async def extract_entities_from_evidence_comprehensive(
    all_evidence: List[ArticleEvidence],
    subject: str,
    query: str,
    existing_entities: Optional[List[Entity]] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Entity]:
    """Meta-level entity extraction that adapts to any domain, not just VC."""

    app.note("Identifying key players and organizations…")

    # Process evidence in batches for comprehensive extraction
    batch_size = 8
    evidence_batches = [
        all_evidence[i : i + batch_size]
        for i in range(0, len(all_evidence), batch_size)
    ]

    # Convert existing entities for context
    existing_entities_xml = ""
    if existing_entities:
        existing_entities_xml = f"""
<existing_entities>
{chr(10).join([f'<entity name="{e.name}" type="{e.type}">{e.summary}</entity>' for e in existing_entities])}
</existing_entities>
"""

    async def extract_entities_from_batch(
        evidence_batch: List[ArticleEvidence],
    ) -> List[Entity]:
        evidence_xml = "\n".join(
            [
                f"<evidence id='{ev.article_id}'>\n<relevance>{ev.relevance_summary}</relevance>\n<facts>{'</fact><fact>'.join(ev.facts)}</facts>\n<quotes>{'</quote><quote>'.join(ev.quotes)}</quotes>\n</evidence>"
                for ev in evidence_batch
            ]
        )

        prompt = f"""
<mission>
You are a Network Intelligence Analyst extracting the STRUCTURAL FOUNDATION of any domain from evidence. Your goal is to identify entities that have CONSEQUENTIAL ROLES in the system being analyzed.
</mission>

<analysis_context>
<research_subject>{subject}</research_subject>
<original_query>{query}</original_query>
</analysis_context>

{existing_entities_xml}

<extraction_philosophy>
**Connectivity-First Principle**: Extract entities that are ACTIVE PARTICIPANTS in networks of relationships, not isolated mentions.

**Prioritize entities that:**
- Make decisions or drive outcomes in the domain
- Control resources or access to resources
- Influence other entities' behavior or success
- Create dependencies or provide critical services
- Compete or collaborate with other entities
- Represent key concepts that shape the domain
- Act as gatekeepers or bottlenecks in processes

**Entity Categories (Adaptive to Domain):**
- **Actors**: People, organizations, institutions that take actions
- **Resources**: Technologies, assets, capabilities that are utilized
- **Concepts**: Ideas, frameworks, trends that influence decisions
- **Constraints**: Regulations, limitations, requirements that shape behavior
- **Catalysts**: Events, innovations, changes that drive transformation

**Quality Over Quantity**: Extract 10-20 entities per batch maximum. Focus on entities with clear connections to others.
</extraction_philosophy>

<evidence_batch>
{evidence_xml}
</evidence_batch>

<instructions>
Extract entities that form the structural backbone of the domain being analyzed:

1. **For NEW entities**: Write summaries emphasizing their role and connections in the system
2. **For EXISTING entities**: Update summaries to integrate new information intelligently without duplication
3. **Connectivity Focus**: Every entity should have clear potential for relationships with other entities
4. **Domain Adaptation**: Adapt entity types to fit the specific domain being analyzed (not hardcoded to VC)
5. **Evidence Grounding**: Base all extractions on specific evidence provided

**Critical Question**: "If this entity disappeared, would it break important connections or change system dynamics?" If no, don't extract it.
</instructions>
"""

        class EntityBatchResult(BaseModel):
            entities: List[Entity]

        result = await ai_with_dynamic_params(
            system="You are a Network-Focused Entity Extraction Specialist who identifies consequential actors in any domain. You prioritize entities with clear systemic importance and relationship potential.",
            user=prompt,
            schema=EntityBatchResult,
            model=model,
            api_key=api_key,
        )

        return result.entities

    # Extract entities from all batches in parallel
    entity_extraction_tasks = [
        extract_entities_from_batch(batch) for batch in evidence_batches
    ]
    entity_batch_results = await run_in_batches(
        entity_extraction_tasks, AI_CALL_CONCURRENCY_LIMIT
    )

    all_entities = [
        entity for batch in entity_batch_results if batch for entity in batch
    ]

    # Deduplicate and merge entities
    final_entities = await process_entity_consolidation_parallel(
        all_entities, model=model, api_key=api_key
    )

    app.note("Mapped the key people, organizations, and concepts.")
    return final_entities


@app.reasoner()
async def extract_relationships_comprehensive(
    all_evidence: List[ArticleEvidence],
    entities: List[Entity],
    query: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Relationship]:
    """Meta-level relationship extraction with iterative discovery."""

    app.note("Connecting the dots between key players…")

    # Process in batches with iterative discovery like meta version
    batch_size = 8
    evidence_batches = [
        all_evidence[i : i + batch_size]
        for i in range(0, len(all_evidence), batch_size)
    ]

    async def extract_relationships_from_batch_iterative(
        evidence_batch: List[ArticleEvidence],
    ) -> List[Relationship]:
        """Iterative relationship discovery within a batch."""

        entities_list = [
            f"- {e.name} ({e.type})" for e in entities[:30]
        ]  # Manageable context

        all_relationships = []
        iteration = 1
        max_iterations = 4

        while iteration <= max_iterations:
            # Create context for this iteration
            evidence_xml = "\n".join(
                [
                    f"<evidence id='{ev.article_id}'>\n<relevance>{ev.relevance_summary}</relevance>\n<facts>{'</fact><fact>'.join(ev.facts[:5])}</facts>\n</evidence>"
                    for ev in evidence_batch
                ]
            )

            already_found_summary = []
            if all_relationships:
                already_found_summary = [
                    f"- {r.source_entity} → {r.target_entity}: {r.relationship_type}"
                    for r in all_relationships[-10:]  # Show recent ones
                ]

            prompt = f"""
<mission>
Find the TOP {8 + min(4, iteration)} most important relationships in this evidence batch (Iteration {iteration}/4).
</mission>

<analysis_context>
<original_query>{query}</original_query>
</analysis_context>

<available_entities>
{chr(10).join(entities_list)}
</available_entities>

<evidence_to_analyze>
{evidence_xml}
</evidence_to_analyze>

<already_identified_relationships>
{chr(10).join(already_found_summary) if already_found_summary else "None yet - this is the first iteration."}
</already_identified_relationships>

<discovery_strategy>
**Iteration {iteration} Focus:**
{"**PRIORITY**: Find the most obvious, clearly stated relationships first." if iteration == 1 else f"**DISCOVERY**: Look for subtler relationships, indirect connections, and implied relationships not yet captured."}

**Connection-First Mindset:**
- Look for EXPLICIT relationships (directly stated) and IMPLICIT relationships (strongly suggested)
- Consider all relationship types: hierarchical, competitive, collaborative, causal, dependency, influence
- Every entity should connect to at least one other entity where possible
- Focus on relationships that explain how the system/domain actually functions

**Relationship Discovery Framework:**
- **Power/Control**: Who controls, owns, governs, or directs whom
- **Dependencies**: Who needs, relies on, or is served by whom
- **Competition**: Who competes with, challenges, or threatens whom
- **Collaboration**: Who partners with, supports, or works with whom
- **Causation**: What causes, influences, or results in what
- **Information Flow**: Who informs, advises, or learns from whom

**Quality Focus:**
- Only include relationships supported by evidence
- Each relationship should be distinct and meaningful
- Use entity names EXACTLY as listed in available_entities
- Skip relationships already identified above
</discovery_strategy>

<instructions>
Find the most important relationships in this iteration:
1. Scan evidence systematically for entity connections
2. Select the most significant relationships you can clearly identify
3. Use exact entity names from available_entities list
4. Skip already identified relationships
5. Set has_more_relationships based on whether significant relationships remain undiscovered
</instructions>
"""

            class IterativeRelationshipResult(BaseModel):
                relationships: List[Relationship]
                has_more_relationships: bool
                confidence_in_completion: float
                iteration_notes: str

            try:
                result = await ai_with_dynamic_params(
                    system="You are a Relationship Discovery Specialist working iteratively to build comprehensive networks. Focus on finding meaningful connections that explain system dynamics.",
                    user=prompt,
                    schema=IterativeRelationshipResult,
                    model=model,
                    api_key=api_key,
                )

                new_relationships = result.relationships

                if new_relationships:
                    all_relationships.extend(new_relationships)
                    app.note(
                        f"Discovering additional connections like: {new_relationships[0].description[:60]}…"
                    )
                else:
                    break

                # Stopping conditions
                if (
                    not result.has_more_relationships
                    or result.confidence_in_completion > 0.9
                ):
                    break

                if iteration > 2 and len(new_relationships) < 3:
                    break

            except Exception as e:
                break

            iteration += 1

        return all_relationships

    # Extract relationships from all batches in parallel
    relationship_extraction_tasks = [
        extract_relationships_from_batch_iterative(batch) for batch in evidence_batches
    ]
    relationship_batch_results = await run_in_batches(
        relationship_extraction_tasks, AI_CALL_CONCURRENCY_LIMIT
    )

    all_relationships = [
        rel for batch in relationship_batch_results if batch for rel in batch
    ]

    # Deduplicate relationships
    final_relationships = await process_relationship_consolidation_parallel(
        all_relationships, model=model, api_key=api_key
    )

    app.note(
        f"Mapped {len(final_relationships)} key connections "
        f"(consolidated from {len(all_relationships)} total relationships)"
        + (
            f" like '{final_relationships[0].description[:40]}...'"
            if final_relationships
            else ""
        )
    )
    return final_relationships


@app.reasoner()
async def synthesize_key_discoveries_meta(
    query: str,
    all_evidence: List[ArticleEvidence],
    entities: List[Entity],
    relationships: List[Relationship],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[str]:
    """Meta-level discovery synthesis that adapts to any domain."""

    temporal_context = get_temporal_context("synthesis")

    # Create rich context from entities and relationships
    entities_xml = "\n".join(
        [
            f'<entity name="{e.name}" type="{e.type}">{e.summary}</entity>'
            for e in entities[:25]  # Manageable context
        ]
    )

    relationships_xml = "\n".join(
        [
            f"<relationship>{r.source_entity} {r.description} {r.target_entity} (type: {r.relationship_type})</relationship>"
            for r in relationships[:20]
        ]
    )

    # Sample evidence for pattern detection
    evidence_summary = []
    for ev in all_evidence[:15]:
        key_facts = " | ".join(ev.facts[:3])
        evidence_summary.append(f"Source {ev.article_id}: {key_facts}")

    prompt = f"""
<mission>
You are a Meta-Cognitive Discovery Synthesist. Your expertise is finding NON-OBVIOUS, CROSS-CUTTING insights by connecting disparate information points across any domain.
</mission>

{temporal_context}

<analysis_context>
<original_research_query>{query}</original_research_query>
<domain_subject>The research centers on understanding the ecosystem around this query</domain_subject>
</analysis_context>

<knowledge_landscape>
<extracted_entities>
{entities_xml}
</extracted_entities>

<discovered_relationships>
{relationships_xml}
</discovered_relationships>

<evidence_patterns>
{chr(10).join(evidence_summary)}
</evidence_patterns>
</knowledge_landscape>

<discovery_synthesis_framework>
**Meta-Level Pattern Recognition:**
Your goal is to identify insights that emerge from the INTERSECTION of entities, relationships, and evidence patterns. Look for:

**Cross-Domain Connections:**
- How do seemingly unrelated entities actually influence each other?
- What hidden bridges exist between different parts of the system?
- Where do feedback loops and system dynamics emerge?

**Emergent System Properties:**
- What behaviors emerge from the network of relationships?
- What constraints or enablers shape the entire system?
- What are the leverage points where small changes create big effects?

**Counter-Intuitive Insights:**
- Where does the evidence contradict common assumptions?
- What relationships exist that shouldn't theoretically exist?
- What entities are more/less important than they initially appear?

**Temporal and Causal Dynamics:**
- What cause-effect chains can you trace through the relationship network?
- How do different time horizons reveal different relationship patterns?
- What are the second and third-order effects of key relationships?

**Structural Insights:**
- What does the shape of the network reveal about power dynamics?
- Where are the bottlenecks, central nodes, and weak points?
- What alternative configurations could emerge from current trends?
</discovery_synthesis_framework>

<synthesis_quality_standards>
**Each discovery must be:**
1. **Non-Obvious**: Not a simple summary of individual facts
2. **Cross-Connected**: Links multiple entities or relationship patterns
3. **Systemically Important**: Reveals how the domain actually functions
4. **Evidence-Grounded**: Traceable to specific evidence patterns
5. **Actionable**: Provides insight that changes understanding

**Example Weak Discovery**: "Company X has experienced leadership."
**Example Strong Discovery**: "The convergence of Company X's regulatory expertise with Partner Y's technical capabilities creates a unique market position that traditional competitors cannot easily replicate, as evidenced by their joint patent filings and shared customer acquisition strategies."
</synthesis_quality_standards>

<instructions>
Generate 4-7 key discoveries that represent the most significant, non-obvious insights emerging from the complete knowledge landscape. Each discovery should:

1. **Connect Multiple Elements**: Link entities, relationships, and evidence patterns
2. **Reveal System Dynamics**: Show how the domain actually functions
3. **Challenge Assumptions**: Surface counter-intuitive or surprising findings
4. **Demonstrate Emergence**: Highlight insights that emerge from the network structure
5. **Maintain Evidence Links**: Ensure each discovery is grounded in the provided evidence

Focus on insights that would not be obvious from reading individual sources but emerge from the comprehensive analysis.
</instructions>
"""

    class MetaDiscoveries(BaseModel):
        key_discoveries: List[str] = Field(
            description="Cross-cutting insights that emerge from comprehensive analysis of entities, relationships, and evidence patterns"
        )

    result = await ai_with_dynamic_params(
        system="You are a Meta-Cognitive Discovery Synthesist who reveals non-obvious insights by connecting disparate information across any domain. You specialize in finding emergent patterns that individual sources cannot reveal.",
        user=prompt,
        schema=MetaDiscoveries,
        model=model,
        api_key=api_key,
    )

    return result.key_discoveries


@app.reasoner()
async def generate_research_briefing(
    package: dict,
    main_query: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ResearchBriefingResponse:
    """
    Generates an interactive research briefing from a research package.
    Uses parallel AI calls to generate briefing components.
    """
    start_time = time.time()

    if "research_package" in package:
        package_data = package["research_package"]
        mode = package.get("mode", "general")
    else:
        package_data = package
        mode = "general"

    pkg = UniversalResearchPackage(**package_data)
    app.note("Preparing executive summary…")

    # Define shared schema classes first
    class BriefingMetadata(BaseModel):
        """Metadata for the research briefing."""

        sources: int = Field(
            description="The total count of unique sources used for the briefing."
        )

    class Evidence(BaseModel):
        """A single piece of evidence supporting the main claim."""

        finding: str = Field(
            description="A crisp, factual statement of a key finding, written as a single sentence."
        )
        sourceNote: str = Field(
            description="A 3-6 word attribution for the finding (e.g., 'According to technical report', 'Based on market analysis')."
        )

    class NextProbe(BaseModel):
        """A suggested next step for further research, containing both user-facing and full internal data."""

        # User-facing short versions generated by the AI
        probe: str = Field(
            description="An action-oriented research question or term, 2-5 words long."
        )
        rationale: str = Field(
            description="A 3-6 word explanation of why this probe is valuable."
        )
        timeEstimate: str = Field(
            description="An estimated time to investigate, e.g., '1-2 min' or '3-5 min'."
        )
        isAutoOpened: bool = Field(
            description="Set to true for the single most important next probe, false otherwise."
        )

        # Full data from the original InquiryProbe for re-querying
        full_question: str = Field(
            description="The original, full research question from the research package."
        )
        full_rationale: str = Field(
            description="The original, detailed rationale for the question."
        )
        full_suggested_method: str = Field(
            description="The original, suggested method for answering the question."
        )

    # Define the parallelizable helper reasoners for briefing generation
    async def generate_vc_claim(pkg: UniversalResearchPackage) -> Any:
        class Claim(BaseModel):
            claim: str
            impact: str

        prompt = f"""
You are a VC Partner summarizing the key takeaway for an investment committee.
Based on the following thesis, distill it into a sharp, declarative 'claim' and a single 'impact' sentence.
**Thesis:** {pkg.core_thesis}
**Supporting Points:** {pkg.observed_causal_chains}
**Risks:** {pkg.hypothesized_implications}
"""
        return await ai_with_dynamic_params(
            system="You are a top-tier VC Partner.",
            user=prompt,
            schema=Claim,
            model=model,
            api_key=api_key,
        )

    async def extract_vc_evidence(pkg: UniversalResearchPackage) -> Any:
        class EvidenceList(BaseModel):
            evidence: List[Evidence]

        prompt = f"""
From the following key discoveries, extract the 3 most compelling and distinct facts as evidence points. For each, create a short 'sourceNote' (e.g., 'Market Analysis', 'Technical Diligence').
**Discoveries:** {pkg.key_discoveries}
"""
        return await ai_with_dynamic_params(
            system="You are a VC Analyst extracting key evidence.",
            user=prompt,
            schema=EvidenceList,
            model=model,
            api_key=api_key,
        )

    async def propose_diligence_questions(pkg: UniversalResearchPackage) -> Any:
        # Reusing the user's original briefing schemas for this part
        class NextProbeList(BaseModel):
            """A list of suggested next probes for investigation."""

            nextProbes: List[NextProbe]

        if not pkg.next_inquiry_probes:
            return NextProbeList(nextProbes=[])
        prompt = f"""
Convert these internal diligence questions into short, user-facing probes for the next round of investigation. The first one should be marked for auto-opening.
**Diligence Questions:** {[p.question for p in pkg.next_inquiry_probes]}
"""

        # This is a simplified call; a more robust version would use the full logic from your original code
        # to map back the full data, which is recommended.
        class TempProbe(BaseModel):
            probe: str
            rationale: str
            timeEstimate: str
            isAutoOpened: bool

        class TempProbeList(BaseModel):
            nextProbes: List[TempProbe]

        temp_probes = await ai_with_dynamic_params(
            system="You are a research translator.",
            user=prompt,
            schema=TempProbeList,
            model=model,
            api_key=api_key,
        )
        # Remap to full NextProbe, adding original data back in
        full_probes = []
        for i, tp in enumerate(temp_probes.nextProbes):
            if i < len(pkg.next_inquiry_probes):
                op = pkg.next_inquiry_probes[i]
                full_probes.append(
                    NextProbe(
                        **tp.model_dump(),
                        full_question=op.question,
                        full_rationale=op.rationale,
                        full_suggested_method=op.suggested_method,
                    )
                )
        return NextProbeList(nextProbes=full_probes)

    # Run all briefing tasks in parallel
    claim_task = generate_vc_claim(pkg)
    evidence_task = extract_vc_evidence(pkg)
    probes_task = propose_diligence_questions(pkg)

    claim_and_impact, evidence_list, probe_list = await asyncio.gather(
        claim_task, evidence_task, probes_task
    )

    briefing_id = hashlib.md5(claim_and_impact.claim.encode()).hexdigest()

    class ResearchBriefing(BaseModel):
        """A concise, user-facing summary of the research findings."""

        id: str = Field(
            description="A unique identifier for this briefing, generated as a hash of the claim."
        )
        claim: str
        impact: str
        evidence: List[Evidence]
        nextProbes: List[NextProbe]
        metadata: BriefingMetadata
        isCompleted: bool = Field(
            description="Indicates if the research has concluded (i.e., no more probes)."
        )

    briefing = ResearchBriefing(
        id=briefing_id,
        claim=claim_and_impact.claim,
        impact=claim_and_impact.impact,
        evidence=evidence_list.evidence,
        nextProbes=probe_list.nextProbes,
        metadata=BriefingMetadata(sources=len(pkg.source_articles)),
        isCompleted=not probe_list.nextProbes,
    )

    execution_time = time.time() - start_time
    return ResearchBriefingResponse(
        mode=mode,
        version="2.0",
        research_package=briefing.dict(),
        metadata={
            "query": main_query,
            "created_at": datetime.datetime.now().isoformat(),
            "execution_time": execution_time,
        },
    )


@app.reasoner()
async def generate_document_from_package(
    package: dict,
    main_query: str,
    tension_lens: str = "balanced",
    source_strictness: str = "mixed",
    evidence_style: str = "standard",
    analysis_depth: str = "ANALYTICAL_BRIEF",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> DocumentResponse:
    """
    Delegates to the original intelligent publishing pipeline with identical prompts and logic.
    """
    res: DocGenDocumentResponse = await generate_document_from_package_core(
        package,
        main_query,
        tension_lens=tension_lens,
        source_strictness=source_strictness,
        evidence_style=evidence_style,
        analysis_depth=analysis_depth,
        ai_call=lambda *args, **kwargs: ai_with_dynamic_params(*args, **kwargs),
        note=lambda msg: app.note(msg),
        ADJUDICATION_BATCH_SIZE=ADJUDICATION_BATCH_SIZE,
        AI_CALL_CONCURRENCY_LIMIT=AI_CALL_CONCURRENCY_LIMIT,
        model=model,
        api_key=api_key,
    )

    return DocumentResponse(
        mode=res.mode,
        version=res.version,
        research_package=res.research_package,
        metadata=res.metadata,
    )


# ==============================================================================
# END-TO-END RESEARCH ORCHESTRATION
# ==============================================================================


@app.reasoner()
async def execute_deep_research(
    query: str,
    mode: str = "general",
    research_focus: int = 3,
    research_scope: int = 3,
    max_research_loops: int = 3,
    num_parallel_streams: int = 2,
    tension_lens: str = "balanced",
    source_strictness: str = "mixed",
    evidence_style: str = "standard",
    analysis_depth: str = "ANALYTICAL_BRIEF",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> DocumentResponse:
    """
    End-to-end deep research pipeline that orchestrates the complete flow:
    1. Prepares a comprehensive research package (iterative multi-stream research)
    2. Generates a formatted document from the research findings

    This is the primary endpoint for executing a complete research workflow.

    Parameters:
    -----------
    query : str
        The research question or topic to investigate
    mode : str
        Research mode - "general" for broad research (default: "general")
    research_focus : int
        Depth of research on a scale of 1-5 (default: 3)
        - 1: Surface-level overview
        - 3: Balanced depth
        - 5: Maximum depth investigation
    research_scope : int
        Breadth of research on a scale of 1-5 (default: 3)
        - 1: Narrow, focused scope
        - 3: Balanced breadth
        - 5: Wide-ranging exploration
    max_research_loops : int
        Maximum number of iterative research cycles (default: 3)
        Each loop identifies gaps and performs targeted follow-up research
    num_parallel_streams : int
        Number of parallel research angles to explore simultaneously (default: 2)
    tension_lens : str
        Perspective lens for document generation (default: "balanced")
        - "balanced": Objective, multi-perspective analysis
        - "bull": Optimistic, opportunity-focused framing
        - "bear": Cautious, risk-focused framing
    source_strictness : str
        Source quality filtering level (default: "mixed")
        - "strict": Only high-quality, authoritative sources
        - "mixed": Balance of source types
        - "permissive": Include all relevant sources
    evidence_style : str
        Citation and evidence presentation style (default: "standard")
    analysis_depth : str
        Depth of analysis in generated document (default: "ANALYTICAL_BRIEF")
    model : str, optional
        Override the default LLM model for this request
    api_key : str, optional
        Override the default API key for this request

    Returns:
    --------
    DocumentResponse
        Contains:
        - mode: Research mode used
        - version: System version
        - research_package: Complete research data including:
            - document: Generated document content
            - entities: Extracted entities (companies, people, etc.)
            - relationships: Entity relationships
            - source_articles: All source materials
            - article_evidence: Extracted facts and quotes
        - metadata: Timing and performance statistics
    """
    start_time = time.time()
    app.note(f"Starting end-to-end deep research on: '{query[:80]}{'...' if len(query) > 80 else ''}'")

    # Phase 1: Prepare the research package
    app.note("Phase 1: Executing iterative research loops...")
    research_response = await prepare_research_package(
        query=query,
        mode=mode,
        research_focus=research_focus,
        research_scope=research_scope,
        max_research_loops=max_research_loops,
        num_parallel_streams=num_parallel_streams,
        model=model,
        api_key=api_key,
    )

    app.note("Phase 2: Generating document from research findings...")

    # Phase 2: Generate the document from the research package
    document_response = await generate_document_from_package(
        package=research_response.research_package,
        main_query=query,
        tension_lens=tension_lens,
        source_strictness=source_strictness,
        evidence_style=evidence_style,
        analysis_depth=analysis_depth,
        model=model,
        api_key=api_key,
    )

    total_time = time.time() - start_time
    app.note(f"Deep research complete in {total_time:.1f}s")

    # Merge metadata with total orchestration time
    merged_metadata = {
        **document_response.metadata,
        "total_orchestration_time_seconds": round(total_time, 2),
        "research_phase_metadata": research_response.metadata,
    }

    return DocumentResponse(
        mode=document_response.mode,
        version=document_response.version,
        research_package=document_response.research_package,
        metadata=merged_metadata,
    )


# ==============================================================================
# AGENT SERVER
# ==============================================================================

if __name__ == "__main__":
    app.serve(auto_port=True, host="0.0.0.0")
