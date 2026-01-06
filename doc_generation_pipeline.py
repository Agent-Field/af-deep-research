#!/usr/bin/env python3
"""
Document Generation Pipeline (extracted from old_deep.py)

This module contains the schemas and business logic for the intelligent
document generation pipeline. Prompts and logic are copied to match the
original behavior, but this module is self-contained and does not depend
on a global Agent. Callers must provide the AI call wrapper.
"""

import asyncio
import datetime
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Core helpers (copied)
# -----------------------------------------------------------------------------


async def ai_with_dynamic_params(
    *args, model: Optional[str] = None, api_key: Optional[str] = None, **kwargs
) -> Any:
    """
    Placeholder for injection. The caller should monkey-patch or pass a wrapper
    in orchestrator calls; this default raises to signal misconfiguration.
    """
    raise RuntimeError("ai_with_dynamic_params must be provided by the caller")


def create_content_hash(content: str) -> str:
    import hashlib

    return hashlib.md5(content.encode()).hexdigest()


async def run_in_batches(tasks: List, batch_size: int):
    results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        results.extend(batch_results)
    return [r for r in results if not isinstance(r, Exception)]


def _note(_: str) -> None:
    # default no-op logger; caller may pass a custom function
    return None


# -----------------------------------------------------------------------------
# Schemas required by the pipeline (copied)
# -----------------------------------------------------------------------------


class Article(BaseModel):
    id: int
    title: str
    url: str
    content: str
    content_hash: str


class ArticleEvidence(BaseModel):
    article_id: int
    relevance_summary: str
    facts: List[str]
    quotes: List[str]


class Entity(BaseModel):
    name: str
    type: str
    summary: str


class Relationship(BaseModel):
    source_entity: str
    target_entity: str
    description: str
    relationship_type: str


class InquiryProbe(BaseModel):
    question: str
    rationale: str
    suggested_method: str


class UniversalResearchPackage(BaseModel):
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


class DocumentResponse(BaseModel):
    mode: str
    version: str
    research_package: dict
    metadata: dict


class SourceNote(BaseModel):
    citation_id: int
    title: str
    domain: str
    url: str


class DocumentSection(BaseModel):
    title: str
    content: str


class FinalDocument(BaseModel):
    document_title: str
    executive_summary: str
    sections: List[DocumentSection]
    source_notes: List[SourceNote]
    disclaimers: List[str]


class FactForAdjudication(BaseModel):
    fact_id: str
    content: str
    source_type: str
    source_reliability_score: float


class AIAssessment(BaseModel):
    fact_id: str
    is_allowed: bool = Field(
        description="True if this fact is admissible based on the source strictness policy."
    )
    is_verified: bool = Field(
        description="True if the source is considered reputable and formally verified."
    )
    disagreement_score: float = Field(
        description="A score (0.0-1.0) indicating conflict with other facts."
    )


class AIAssessmentList(BaseModel):
    assessments: List[AIAssessment]


class AdjudicatedFact(BaseModel):
    fact_id: str
    content: str
    source_id: int
    source_title: str
    source_url: str
    source_domain: str
    is_quote: bool
    is_verified: bool
    disagreement_score: float


class SectionDirective(BaseModel):
    section_id: int
    section_title: str
    writing_instructions: str
    evidence_to_use: List[str]


class EditorialPlan(BaseModel):
    document_title: str
    plan: List[SectionDirective]


class FinalAssemblyInput(BaseModel):
    executive_summary: str


class DisclaimerList(BaseModel):
    disclaimers: List[str]


class SectionResult(BaseModel):
    section_id: int
    markdown_content: str


class EvidenceDigest(BaseModel):
    digest_id: str
    summary: str
    source_id: int


class Theme(BaseModel):
    theme_title: str
    planning_directive: str


class ThematicBlueprint(BaseModel):
    document_title: str
    themes: List[Theme]


# -----------------------------------------------------------------------------
# Agents (prompts and logic copied, with injectable AI and logger)
# -----------------------------------------------------------------------------


async def plan_document_themes(
    package: UniversalResearchPackage,
    analysis_depth: str,
    *,
    ai_call=ai_with_dynamic_params,
    note=_note,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ThematicBlueprint:
    depth_instructions = {
        "EXECUTIVE_SUMMARY": "The user wants a minimal overview. Define a single, focused knowledge cluster like 'Core Thesis and Key Evidence'.",
        "ANALYTICAL_BRIEF": "The user wants a structured exploration. Define 3-4 primary knowledge clusters covering the main subject, key relationships, and primary implications.",
        "DEEP_DIVE_REPORT": "The user wants an exhaustive knowledge map. Define multiple (5+) detailed clusters. You MUST include clusters for 'Second-Order Impacts', 'Identified Knowledge Gaps', and 'Contradictory Evidence'.",
    }

    research_digest = f"""
<core_thesis>{package.core_thesis}</core_thesis>
<key_discoveries>
    {chr(10).join([f"- {d}" for d in package.key_discoveries])}
</key_discoveries>
<entities>
    {chr(10).join([f"- {e.name} ({e.type}): {e.summary}" for e in package.entities])}
</entities>
<relationships>
    {chr(10).join([f"- {r.source_entity} {r.description} {r.target_entity}." for r in package.relationships])}
</relationships>
<causal_chains>
    {chr(10).join([f"- {c}" for c in package.observed_causal_chains])}
</causal_chains>
<implications>
    {chr(10).join([f"- {i}" for i in package.hypothesized_implications])}
</implications>
"""

    inquiry_context = ""
    if package.next_inquiry_probes:
        inquiry_context = f"""
<inquiry_exploration_history>
<what_we_have_explored>
The research has investigated the following areas through various inquiry probes:
{chr(10).join([f"- {probe.question}: {probe.rationale}" for probe in package.next_inquiry_probes])}
</what_we_have_explored>
</inquiry_exploration_history>
"""

    prompt = f"""
<task_description>
You are a **Knowledge Architect**. Your task is to organize a complex body of research into a logical, explorable structure by creating a `ThematicBlueprint`. Instead of a linear narrative, you will define distinct **'Knowledge Clusters'** that represent key domains of the research.

**CRITICAL**: Your themes must work toward answering the original research query. Ensure the document structure serves the user's original intent, not just organizing existing content.
</task_description>

<research_context>
<original_query>{package.query}</original_query>
<confidence_assessment>{package.confidence_assessment}</confidence_assessment>
{inquiry_context}
</research_context>

<strategic_controls>
 <analysis_depth>{analysis_depth}</analysis_depth>
</strategic_controls>

<instructions>
1.  **Understand the Original Intent**: Review the `original_query` to understand what the user originally wanted to know.
2.  **Assess Research Coverage**: Consider the `confidence_assessment` and `inquiry_exploration_history` to understand what has been thoroughly explored vs. what gaps remain.
3.  **Analyze the Research Digest**: Review the provided digest to understand the key entities, relationships, and findings.
4.  **Define Knowledge Clusters**: Based on the `analysis_depth` directive, group the information into logical clusters that work toward answering the original query. Each cluster is a self-contained but connected domain of knowledge.
  <depth_directive>{depth_instructions.get(analysis_depth.upper(), depth_instructions["ANALYTICAL_BRIEF"])}</depth_directive>
5.  **Prioritize Relevance**: Ensure themes directly address aspects of the original query. If research has explored tangential areas, organize them in a way that connects back to the main question.
6.  **Address Knowledge Gaps**: If the confidence assessment indicates areas of uncertainty, consider creating themes that acknowledge these gaps while presenting what is known.
7.  **Create Planning Directives**: For each cluster, write a concise `planning_directive` that instructs a section planner on the goal for that cluster, ensuring it contributes to answering the original query.
8.  **Propose a Title**: Create an insightful title for the entire knowledge map that reflects the original research intent.
</instructions>

<research_digest>
{research_digest}
</research_digest>
"""
    return await ai_call(
        system="You are an AI Knowledge Architect. You design explorable knowledge structures by organizing complex research into distinct, thematic clusters.",
        user=prompt,
        schema=ThematicBlueprint,
        model=model,
        api_key=api_key,
    )


async def plan_theme_sections(
    theme: Theme,
    research_digest: str,
    evidence_digests: List[EvidenceDigest],
    tension_lens: str,
    analysis_depth: str,
    original_query: str,
    *,
    ai_call=ai_with_dynamic_params,
    note=_note,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> EditorialPlan:
    evidence_xml_lines = []
    simple_id_map: Dict[int, str] = {}
    for idx, ed in enumerate(evidence_digests, start=1):
        simple_id_map[idx] = ed.digest_id
        evidence_xml_lines.append(
            f"<evidence id='{idx}' source='{ed.source_id}'>"
            + f"<summary>{ed.summary}</summary>"
            + "</evidence>"
        )
    evidence_xml = "\n".join(evidence_xml_lines)

    depth_instruction = (
        "Create 1-2 sections only — focus tightly on the core thesis and the strongest supporting evidence."
        if analysis_depth.upper() == "EXECUTIVE_SUMMARY"
        else (
            "Create 2-3 sections that balance the thesis, key relationships, and primary implications."
            if analysis_depth.upper() == "ANALYTICAL_BRIEF"
            else "Create 3 or more sections. Include structure for implications, knowledge gaps, and areas of disagreement if evidence suggests."
        )
    )

    prompt = f"""
<task_description>
You are a specialized Section Editor. Your job is to create a precise `EditorialPlan` for the assigned theme.

**CRITICAL**: Ensure your sections contribute to answering the original research query. Each section should advance understanding of what the user originally wanted to know.
</task_description>

<research_context>
<original_query>{original_query}</original_query>
</research_context>

<theme_assignment>
  <theme_title>{theme.theme_title}</theme_title>
  <planning_directive>{theme.planning_directive}</planning_directive>
</theme_assignment>

<strategic_controls>
  <tension_lens>{tension_lens}</tension_lens>
  <analysis_depth>{analysis_depth}</analysis_depth>
</strategic_controls>

<instructions>
1.  **Understand the Original Intent**: Review the `original_query` to understand what the user originally wanted to know.
2.  **Understand Your Theme**: Internalize the goal of your assigned theme and how it contributes to answering the original query.
3.  **Select Evidence**: From `<available_evidence>`, select the most relevant evidence for your theme by referencing its simple numeric `id`.
4.  **Design Sections**: Break your theme down into 1-3 logical sections that work toward answering the original query. Give each a clear title.
5.  **Write Smart Instructions**: For each section, write detailed `writing_instructions` for a journalist.
    -   Tell them the analytical goal and the key points to make in relation to the original research question.
    -   Make sure you include everything that is important for the theme that you are writing with all the details we have. Don't leave any information. You are suggesting as if to be presented to an investigative journal.
    -   **CRITICAL: Suggest Markdown formatting where appropriate to enhance clarity.** Analyze the evidence you're assigning and guide the writer on the best way to present it. For example:
        -   If the evidence involves **comparison** (e.g., stats, features, pros/cons), instruct the writer: **"Present this data in a Markdown table."**
        -   If the section's goal is to list **key findings or recommendations**, instruct the writer: **"Summarize these points in a bulleted list."**
        -   If a piece of evidence is a particularly **impactful statement or quote**, instruct the writer: **"Feature this quote using a blockquote."**
    -   {depth_instruction}
6.  **Assign Evidence**: For each section, list the simple numeric `id`s of the evidence the writer must use.
7.  **Ensure Relevance**: Make sure each section advances understanding of the original research query.
</instructions>

<full_research_context_digest>
{research_digest}
</full_research_context_digest>

<available_evidence>
{evidence_xml}
</available_evidence>
"""

    class PlanDirectiveSimple(BaseModel):
        section_title: str
        writing_instructions: str
        evidence_to_use: List[int]

    class PlanOnlySimple(BaseModel):
        plan: List[PlanDirectiveSimple]

    simple_plan = await ai_call(
        system="You are an AI section editor. You create detailed, evidence-backed content plans and suggest intelligent Markdown formatting to maximize clarity.",
        user=prompt,
        schema=PlanOnlySimple,
        model=model,
        api_key=api_key,
    )

    final_plan_directives: List[SectionDirective] = []
    for simple_directive in simple_plan.plan:
        remapped_evidence_ids = [
            simple_id_map[sid]
            for sid in simple_directive.evidence_to_use
            if sid in simple_id_map
        ]
        final_plan_directives.append(
            SectionDirective(
                section_id=0,
                section_title=simple_directive.section_title,
                writing_instructions=simple_directive.writing_instructions,
                evidence_to_use=remapped_evidence_ids,
            )
        )

    return EditorialPlan(document_title="", plan=final_plan_directives)


async def write_document_section(
    directive: SectionDirective,
    relevant_facts: List[AdjudicatedFact],
    citation_map: Dict[int, int],
    full_context_query: str,
    full_context_thesis: str,
    evidence_style: str,
    *,
    ai_call=ai_with_dynamic_params,
    note=_note,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> SectionResult:
    evidence_xml = "\n".join(
        [
            f'<evidence citation="[{citation_map.get(f.source_id)}]">{f.content}</evidence>'
            for f in relevant_facts
        ]
    )

    prompt = f"""
<task_description>
You are an expert writer and editor. Your task is to write a deeply analytical document section, adhering to the highest standards of professional and academic writing.
</task_description>

<full_research_context>
  <original_query>{full_context_query}</original_query>
  <core_thesis>{full_context_thesis}</core_thesis>
</full_research_context>

<section_directive>
  <title>{directive.section_title}</title>
  <instructions_from_editor>{directive.writing_instructions}</instructions_from_editor>
</section_directive>

<style_guide>
  <tone_and_style_mandate>
    **Goal**: To produce writing with the clarity, rigor, and professional tone of a premier publication like Wallstreet Journal/NYT or other top business or scientific journal. This is for a discerning audience (executives, academics, legal professionals).
    -   **Inspiration**: Emulate the high standards of publications like **The New York Times, The Economist, and respected academic journals.**
    -   **Formality**: Maintain a formal, objective, and analytical tone. **Avoid casual phrases, contractions (e.g., "don't"), rhetorical questions, and first-person pronouns (I, we, our).**
    -   **Precision**: Use precise, unambiguous language. Construct clear, grammatically correct sentences that convey complex information efficiently.
    -   **Synthesis**: Do not simply list facts back-to-back. Weave the evidence into a coherent analytical narrative. Your role is to connect the dots and explain the significance of the information.
    -  Make sure you include everything that is important for the theme that you are writing with all the details we have. Don't leave any information. You are suggesting as if to be presented to an investigative journal.
  </tone_and_style_mandate>

  <evidence_style>{evidence_style}</evidence_style>

  <markdown_usage_philosophy>
    **Use Markdown with purpose to improve clarity and impact. Do not overdo it.**
    -   **Use Tables for Structure**: When presenting structured comparisons (e.g., pros/cons, features of different products, quantitative data), render the information in a Markdown table for scannability.
    -   **Use Lists for Clarity**: For enumerating key findings, action items, or recommendations, always use bulleted or numbered lists.
    -   **Use Blockquotes for Impact**: Isolate powerful quotes, critical assumptions, or a key takeaway conclusion from a paragraph using blockquotes to make it stand out.
    -   **Use Sub-Headings for Organization**: For longer, more complex sections, use `###` Markdown sub-headings to break up the content into logical, digestible parts.
  </markdown_usage_philosophy>

</style_guide>

<instructions>
1.  **Adhere to the Tone**: Your primary goal is to follow the `<tone_and_style_mandate>`.
2.  **Follow the Editor's Lead**: Implement any specific formatting suggestions (like using a table or list) from the `<instructions_from_editor>`.
3.  **Apply the Style Guide**: Write the content for your section in rich Markdown, applying the principles from the `<markdown_usage_philosophy>`.
4.  **Cite Meticulously**: You **MUST** insert the corresponding `citation` marker (e.g., `[1]`, `[2]`) immediately after the sentence or clause it supports. Group citations where appropriate (e.g., `[1, 3, 4]`).
5.  **Formatting**: Ensure all Markdown is valid. Do not include the main section title (`##`), as it will be added programmatically.
</instructions>

<available_evidence>
{evidence_xml}
</available_evidence>
"""

    class AIWriterOutput(BaseModel):
        markdown_content: str

    ai_output = await ai_call(
        system="You are an AI writer and editor, trained in the rigorous style of premier academic and journalistic publications. You produce formal, analytical, and well-structured content.",
        user=prompt,
        schema=AIWriterOutput,
        model=model,
        api_key=api_key,
    )

    return SectionResult(
        section_id=directive.section_id,
        markdown_content=ai_output.markdown_content,
    )


async def adjudicate_evidence_ai(
    facts_for_adjudication: List[FactForAdjudication],
    source_strictness: str,
    *,
    ai_call=ai_with_dynamic_params,
    note=_note,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> AIAssessmentList:
    facts_xml = "\n".join(
        [
            f"<fact id='{f.fact_id}' source_type='{f.source_type}' reliability='{f.source_reliability_score}'>{f.content}</fact>"
            for f in facts_for_adjudication
        ]
    )

    prompt = f"""
<task_description>
You are a meticulous Source Adjudicator. Your task is to enforce the source strictness policy and analyze evidence for conflict.
</task_description>

<policy name="Source Strictness">
Current Policy Level: **{source_strictness}**

- **verified-only**: Allow only if `source_type` is peer_reviewed, gov, reputable_media, or primary_doc. `is_verified` should be true.
- **mixed**: Allow `verified-only` types plus well-argued `blog` and `corporate` sources. `is_verified` can be true or false.
- **exploratory**: Allow all source types, but be skeptical. `is_verified` is mostly false for non-primary sources.
</policy>

<instructions>
1.  **Assess Admissibility**: For each fact below, decide if it is allowed (`is_allowed`) under the current policy.
2.  **Assess Verification**: Determine if the source type is formally verifiable (`is_verified`).
3.  **Assess Conflict**: For each fact, compare it to all other facts. Assign a `disagreement_score` from 0.0 (total consensus) to 1.0 (direct contradiction with a high-reliability source). A high score indicates a point of tension.
</instructions>

<evidence_to_adjudicate>
{facts_xml}
</evidence_to_adjudicate>
"""
    return await ai_call(
        system="You are an AI data adjudicator. You assess evidence for admissibility, verification, and conflict based on a strict policy.",
        user=prompt,
        schema=AIAssessmentList,
        model=model,
        api_key=api_key,
    )


async def create_editorial_plan(
    query: str,
    core_thesis: str,
    adjudicated_facts: List[AdjudicatedFact],
    tension_lens: str,
    evidence_style: str,
    analysis_depth: str,
    *,
    ai_call=ai_with_dynamic_params,
    note=_note,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> EditorialPlan:
    facts_xml = "\n".join(
        [
            f"<fact id='{f.fact_id}' source_id='{f.source_id}' verified='{f.is_verified}' disagreement='{f.disagreement_score}'>{f.content}</fact>"
            for f in adjudicated_facts
        ]
    )

    depth_instructions = {
        "executive_summary": """
- **Scope**: Minimal. The plan should be very concise, likely resulting in 1-2 sections.
- **Content Focus**: Restrict the plan to cover ONLY the `core_thesis` and the top 2-3 most critical `key_discoveries`.
- **Exclusions**: Explicitly AVOID creating sections for secondary entities, hypothesized implications, knowledge gaps, or detailed source disagreements.
""",
        "analytical_brief": """
- **Scope**: Balanced. The plan should be well-rounded, likely resulting in 3-5 sections.
- **Content Focus**: The plan must cover the main thesis, the most important entities and their primary relationships, and the key causal chains.
- **Synthesis**: Instruct writers to synthesize the core adjudicated evidence into a clear narrative.
""",
        "deep_dive_report": """
- **Scope**: Exhaustive & Expansive. The plan should be comprehensive, likely resulting in 6-10+ sections. Your plan must aim to utilize the vast majority of the adjudicated evidence.
- **Content Focus**: Cover all major and most secondary topics present in the evidence.
- **Generative Sections**: You MUST create directives for new, forward-looking and analytical sections, including:
    1.  A "Future Outlook & Plausible Scenarios" section that expands on the research's hypothesized implications.
    2.  A "Knowledge Gaps & Research Frontiers" section that analyzes what remains unknown and suggests next steps.
    3.  A "Second-Order Connections" section that explores less obvious relationships between entities and implications.
""",
    }

    prompt = f"""
<task_description>
You are a Master Editor and Narrative Strategist. Your job is to design the strategic blueprint for a document that will present research in a compelling, logically structured way.
</task_description>

<context>
  <original_query>{query}</original_query>
  <core_thesis>{core_thesis}</core_thesis>
</context>

<strategic_controls>
  <tension_lens>{tension_lens}</tension_lens>
  <evidence_style>{evidence_style}</evidence_style>
  <analysis_depth>{analysis_depth}</analysis_depth>
</strategic_controls>

<instructions>
1.  **Set the Narrative Scope**: Use `analysis_depth` to determine the length and scope of the document.\n{depth_instructions.get(analysis_depth.lower(), depth_instructions['analytical_brief'])}
2.  **Assign Evidence Strategically**: Select the most critical fact_ids that support each section's goal.
3.  **Create Section Directives**: Devise a logical flow of sections. For each section:
    - Give it a clear, compelling title.
    - Assign the most relevant `fact_id`s from the available evidence.
    - Write detailed `writing_instructions` for a writer, guiding them on the analytical goal, tone, and suggested Markdown structures (like `###` sub-headings or tables).
4.  **Propose a Document Title**: Create a compelling, top-level title for the entire document This title should be inspired from the initial query and the kind of information we have and what we are trying to convey. Think about New York Journal or Wall Street post like title or top research article like title. Take inspiration from various places and we are going to write it like that. Write like a professional journalist or appropriate to the field and don't write it like AI. this should not contain things like "-" or ":" or \em dash etc..
</instructions>

<available_evidence>
{facts_xml}
</available_evidence>
"""
    return await ai_call(
        system="You are an AI editor that creates strategic document plans based on research findings and narrative controls like depth, tension, and style.",
        user=prompt,
        schema=EditorialPlan,
        model=model,
        api_key=api_key,
    )


async def assemble_final_summary_ai(
    section_contents: List[str],
    original_query: str,
    *,
    ai_call=ai_with_dynamic_params,
    note=_note,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> FinalAssemblyInput:
    full_text = "\n\n---\n\n".join(section_contents)
    prompt = f"""
<task_description>
You are the Final Editor. The document's sections have been written by a team of specialists. Your job is to provide the final piece: a compelling executive summary that ties back to the original research question.
</task_description>

<research_context>
<original_query>{original_query}</original_query>
</research_context>

<instructions>
Read the full text of the document. Write a single, concise paragraph that serves as the `executive_summary`. It should:
1. Capture the document's core thesis, key findings, and main conclusion
2. Explicitly reference how the research addresses the original query
3. Be self-contained and suitable for a busy reader
4. Demonstrate that the research journey has answered what the user originally wanted to know

**CRITICAL**: Ensure the summary connects back to the original research question and shows how the document fulfills the user's original intent.
</instructions>

<full_document_text>
{full_text}
</full_document_text>
"""
    return await ai_call(
        system="You are an AI editor that writes concise executive summaries for completed research documents.",
        user=prompt,
        schema=FinalAssemblyInput,
        model=model,
        api_key=api_key,
    )


async def generate_disclaimers_ai(
    source_strictness: str,
    num_adjudicated_facts: int,
    highest_disagreement_score: float,
    *,
    ai_call=ai_with_dynamic_params,
    note=_note,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> DisclaimerList:
    prompt = f"""
<task_description>
You are a Risk Assessment AI. Your job is to write 1-2 concise, one-sentence disclaimers for a research document based on the following metadata.
</task_description>

<metadata>
    <source_strictness_policy>{source_strictness}</source_strictness_policy>
    <total_facts_used>{num_adjudicated_facts}</total_facts_used>
    <peak_disagreement_score>{highest_disagreement_score:.2f}</peak_disagreement_score>
</metadata>

<instructions>
- If `source_strictness` is `exploratory`, you MUST include a warning about unverified sources.
- If `total_facts_used` is less than 10, you MUST include a warning about the limited evidence base.
- If `peak_disagreement_score` is greater than 0.7, you should consider a warning about the contentious nature of the topic.
- Keep disclaimers to a single, clear sentence.
- If no significant risks are present, return an empty list.
</instructions>
"""
    return await ai_call(
        system="You are an AI risk analyst that writes clear, concise disclaimers for research documents based on their metadata.",
        user=prompt,
        schema=DisclaimerList,
        model=model,
        api_key=api_key,
    )


# -----------------------------------------------------------------------------
# Main Orchestrator (copied, using injected helpers)
# -----------------------------------------------------------------------------


async def generate_document_from_package_core(
    package: dict,
    main_query: str,
    *,
    tension_lens: str = "balanced",
    source_strictness: str = "mixed",
    evidence_style: str = "standard",
    analysis_depth: str = "ANALYTICAL_BRIEF",
    ai_call=ai_with_dynamic_params,
    note=_note,
    ADJUDICATION_BATCH_SIZE: int = 50,
    AI_CALL_CONCURRENCY_LIMIT: int = 20,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> DocumentResponse:
    import time

    start_time = time.time()

    if isinstance(package, dict):
        if "research_package" in package:
            package_data = package["research_package"]
            pkg = UniversalResearchPackage(**package_data)
            mode = package.get("mode", "general")
        else:
            pkg = UniversalResearchPackage(**package)
            mode = "general"
    else:
        pkg = package  # assume UniversalResearchPackage-compatible
        mode = "general"

    note(f"🚀 Starting Intelligent Publishing Pipeline with Depth: {analysis_depth}...")

    note("Stage 1: Adjudicating evidence...")
    facts_to_assess: List[FactForAdjudication] = []
    fact_map: Dict[str, Dict] = {}

    for ev in pkg.article_evidence:
        source_article = next(
            (a for a in pkg.source_articles if a.id == ev.article_id), None
        )
        if not source_article:
            continue

        source_domain = (
            source_article.url.split("/")[2].replace("www.", "")
            if source_article.url
            else "Unknown"
        )
        source_type = (
            "reputable_media"
            if "reuters" in source_domain or "apnews" in source_domain
            else "blog"
        )
        reliability_score = 0.8 if source_type == "reputable_media" else 0.4

        for fact_content in ev.facts:
            fact_id = create_content_hash(f"{ev.article_id}-{fact_content}")
            facts_to_assess.append(
                FactForAdjudication(
                    fact_id=fact_id,
                    content=fact_content,
                    source_type=source_type,
                    source_reliability_score=reliability_score,
                )
            )
            fact_map[fact_id] = {
                "content": fact_content,
                "is_quote": False,
                "source_id": source_article.id,
                "source_title": source_article.title,
                "source_url": source_article.url,
                "source_domain": source_domain,
            }

    adjudication_tasks = []
    for i in range(0, len(facts_to_assess), ADJUDICATION_BATCH_SIZE):
        batch = facts_to_assess[i : i + ADJUDICATION_BATCH_SIZE]
        adjudication_tasks.append(
            adjudicate_evidence_ai(
                batch,
                source_strictness,
                ai_call=ai_call,
                note=note,
                model=model,
                api_key=api_key,
            )
        )

    batch_results: List[AIAssessmentList] = await run_in_batches(
        adjudication_tasks, AI_CALL_CONCURRENCY_LIMIT
    )

    ai_assessments_map: Dict[str, AIAssessment] = {
        assessment.fact_id: assessment
        for result_list in batch_results
        for assessment in result_list.assessments
    }

    adjudicated_facts: List[AdjudicatedFact] = []
    for fact_id, assessment in ai_assessments_map.items():
        if assessment.is_allowed and fact_id in fact_map:
            prog_data = fact_map[fact_id].copy()
            prog_data["content"] = re.sub(
                r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", prog_data["content"]
            ).strip()

            adjudicated_facts.append(
                AdjudicatedFact(
                    fact_id=fact_id,
                    **prog_data,
                    is_verified=assessment.is_verified,
                    disagreement_score=assessment.disagreement_score,
                )
            )
    # Fallback: if adjudication is over-restrictive, admit a small set of top facts
    if not adjudicated_facts and facts_to_assess:
        note(
            "No facts passed adjudication; applying permissive fallback on top evidence (no verification)."
        )
        for fact in facts_to_assess[: min(60, len(facts_to_assess))]:
            if fact.fact_id in fact_map:
                prog_data = fact_map[fact.fact_id].copy()
                prog_data["content"] = re.sub(
                    r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", prog_data["content"]
                ).strip()
                adjudicated_facts.append(
                    AdjudicatedFact(
                        fact_id=fact.fact_id,
                        **prog_data,
                        is_verified=False,
                        disagreement_score=0.0,
                    )
                )

    note(
        f"Adjudication complete and sanitized. {len(adjudicated_facts)} facts available for planning."
    )

    note("Pre-Stage 2: Creating research and evidence digests...")
    research_digest = f"""
<core_thesis>{pkg.core_thesis}</core_thesis>
<key_discoveries>
    {chr(10).join([f"- {d}" for d in pkg.key_discoveries])}
</key_discoveries>
<entities>
    {chr(10).join([f"- {e.name} ({e.type}): {e.summary}" for e in pkg.entities])}
</entities>
<relationships>
    {chr(10).join([f"- {r.source_entity} {r.description} {r.target_entity}." for r in pkg.relationships])}
</relationships>
<causal_chains>
    {chr(10).join([f"- {c}" for c in pkg.observed_causal_chains])}
</causal_chains>
<implications>
    {chr(10).join([f"- {i}" for i in pkg.hypothesized_implications])}
</implications>
"""
    evidence_digests = [
        EvidenceDigest(
            digest_id=fact.fact_id, summary=fact.content, source_id=fact.source_id
        )
        for fact in adjudicated_facts
    ]

    note("Stage 2a: Planning high-level themes with the Strategist agent...")
    thematic_blueprint = await plan_document_themes(
        pkg,
        analysis_depth,
        ai_call=ai_call,
        note=note,
        model=model,
        api_key=api_key,
    )
    note("Organizing research into key themes for analysis…")

    note("Planning detailed sections for each research theme…")
    planner_tasks = [
        plan_theme_sections(
            theme,
            research_digest,
            evidence_digests,
            tension_lens,
            analysis_depth,
            pkg.query,
            ai_call=ai_call,
            note=note,
            model=model,
            api_key=api_key,
        )
        for theme in thematic_blueprint.themes
    ]
    theme_plans: List[EditorialPlan] = await run_in_batches(
        planner_tasks, AI_CALL_CONCURRENCY_LIMIT
    )

    master_plan_directives: List[SectionDirective] = []
    section_counter = 1
    for plan in theme_plans:
        for directive in plan.plan:
            directive.section_id = section_counter
            master_plan_directives.append(directive)
            section_counter += 1

    editorial_plan = EditorialPlan(
        document_title=thematic_blueprint.document_title, plan=master_plan_directives
    )
    # Fallback: ensure at least one section exists
    if not editorial_plan.plan and adjudicated_facts:
        fallback_evidence_ids = [f.fact_id for f in adjudicated_facts[:6]]
        editorial_plan = EditorialPlan(
            document_title=thematic_blueprint.document_title or "Research Findings",
            plan=[
                SectionDirective(
                    section_id=1,
                    section_title="Key Findings and Context",
                    writing_instructions=(
                        "Synthesize the most important findings related to the original query. "
                        "Summarize context, drivers, and implications. Use lists for findings and quotes as blockquotes where impactful."
                    ),
                    evidence_to_use=fallback_evidence_ids,
                )
            ],
        )
        note(
            "No sections planned by AI; inserted a robust fallback section with top evidence."
        )
    else:
        note(
            f"All planners finished. Master plan has {len(editorial_plan.plan)} total sections."
        )

    note("Pre-computing robust global citation map...")
    fact_lookup = {f.fact_id: f for f in adjudicated_facts}
    source_article_map = {a.id: a for a in pkg.source_articles}
    all_source_ids_to_be_used = set(
        fact_lookup[fid].source_id
        for directive in editorial_plan.plan
        for fid in directive.evidence_to_use
        if fid in fact_lookup
    )
    unique_sources_by_url: Dict[str, Article] = {}
    for source_id in all_source_ids_to_be_used:
        if source_id in source_article_map:
            article = source_article_map[source_id]
            if article.url and article.url not in unique_sources_by_url:
                unique_sources_by_url[article.url] = article
    final_cited_articles = sorted(unique_sources_by_url.values(), key=lambda a: a.id)
    url_to_citation_id_map = {
        article.url: i + 1 for i, article in enumerate(final_cited_articles)
    }
    citation_map: Dict[int, int] = {}
    for source_id in all_source_ids_to_be_used:
        if source_id in source_article_map:
            article = source_article_map[source_id]
            if article.url in url_to_citation_id_map:
                citation_map[source_id] = url_to_citation_id_map[article.url]
    note("Organizing source citations for the final document…")

    note("Writing the detailed analysis sections…")
    writer_tasks = []
    for directive in editorial_plan.plan:
        relevant_facts = [
            fact_lookup[fid] for fid in directive.evidence_to_use if fid in fact_lookup
        ]
        task = write_document_section(
            directive=directive,
            relevant_facts=relevant_facts,
            citation_map=citation_map,
            full_context_query=pkg.query,
            full_context_thesis=pkg.core_thesis,
            evidence_style=evidence_style,
            ai_call=ai_call,
            note=note,
            model=model,
            api_key=api_key,
        )
        writer_tasks.append(task)

    section_results: List[SectionResult] = await run_in_batches(
        writer_tasks, AI_CALL_CONCURRENCY_LIMIT
    )
    results_by_id: Dict[int, SectionResult] = {r.section_id: r for r in section_results}
    section_results.sort(key=lambda r: r.section_id)
    missing_count = len(editorial_plan.plan) - len(section_results)
    if missing_count > 0:
        note(
            f"Warning: {missing_count} section(s) failed to generate; inserting placeholders."
        )
    else:
        note("All sections have been written.")

    note("Stage 4: Assembling final components...")

    highest_disagreement = max(
        [f.disagreement_score for f in adjudicated_facts], default=0.0
    )

    note("Generating summary and disclaimers concurrently...")
    summary_result, disclaimer_result = await asyncio.gather(
        assemble_final_summary_ai(
            [r.markdown_content for r in section_results],
            pkg.query,
            ai_call=ai_call,
            note=note,
            model=model,
            api_key=api_key,
        ),
        generate_disclaimers_ai(
            source_strictness,
            len(adjudicated_facts),
            highest_disagreement,
            ai_call=ai_call,
            note=note,
            model=model,
            api_key=api_key,
        ),
    )

    final_sections: List[DocumentSection] = []
    for d in editorial_plan.plan:
        result = results_by_id.get(d.section_id)
        if result is not None:
            content = result.markdown_content
        else:
            content = f"Note: This section ('{d.section_title}') could not be generated due to an upstream error. A placeholder is inserted to maintain structure."
        final_sections.append(DocumentSection(title=d.section_title, content=content))

    source_notes: List[SourceNote] = []
    for i, article in enumerate(final_cited_articles):
        source_notes.append(
            SourceNote(
                citation_id=i + 1,
                title=article.title,
                url=article.url,
                domain=(
                    article.url.split("/")[2].replace("www.", "")
                    if article.url
                    else "Unknown"
                ),
            )
        )

    final_document = FinalDocument(
        document_title=editorial_plan.document_title,
        executive_summary=summary_result.executive_summary,
        sections=final_sections,
        source_notes=source_notes,
        disclaimers=disclaimer_result.disclaimers,
    )

    note("✅ Publishing Pipeline Complete. Final document is ready.")

    execution_time = time.time() - start_time
    return DocumentResponse(
        mode=mode,
        version="1.0",
        research_package=final_document.dict(),
        metadata={
            "query": main_query,
            "created_at": datetime.datetime.now().isoformat(),
            "execution_time": execution_time,
        },
    )
