"""
Dynamic Research Models - Deep Research Agent

Foundational Pydantic models for the dynamic AI reasoning system.
Designed for simplicity to work effectively with LLM generation while supporting
the Brain platform's workflow DAGs and cross-agent memory features.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# ============================================================================
# Simple Enums for LLM-friendly generation
# ============================================================================

class ReasoningType(str, Enum):
    """Types of reasoning strategies"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    LOGICAL = "logical"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    PREDICTIVE = "predictive"
    SYNTHESIS = "synthesis"

class ContextType(str, Enum):
    """Types of research context"""
    FACTUAL = "factual"
    OPINION = "opinion"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    NEWS = "news"
    SOCIAL = "social"

class ConfidenceLevel(str, Enum):
    """Confidence levels for assessments"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class AdaptationTrigger(str, Enum):
    """Triggers for strategy adaptation"""
    QUALITY_THRESHOLD = "quality_threshold"
    TIME_CONSTRAINT = "time_constraint"
    CONTEXT_SHIFT = "context_shift"
    FEEDBACK_SIGNAL = "feedback_signal"


# ============================================================================
# Core Dynamic Reasoning Models
# ============================================================================

class ReasoningStrategy(BaseModel):
    """Simplified reasoning strategy for smaller AI models"""
    name: str = Field(description="Strategy name")
    type: ReasoningType = Field(description="Type of reasoning")
    description: str = Field(description="What this strategy does")
    confidence_threshold: float = Field(description="Minimum confidence needed", ge=0.0, le=1.0)

class PromptTemplate(BaseModel):
    """Simplified prompt template for smaller AI models"""
    template_id: str = Field(description="Unique template identifier")
    system_prompt: str = Field(description="System prompt template")
    user_prompt: str = Field(description="User prompt template")
    reasoning_type: ReasoningType = Field(description="Associated reasoning type")

class ResearchContext(BaseModel):
    """Current research context and state"""
    query: str = Field(description="Original research query")
    context_type: ContextType = Field(description="Type of research context")
    complexity: str = Field(description="simple, moderate, or complex")
    time_available: int = Field(description="Available time in minutes")
    quality_target: str = Field(description="high, medium, or low quality target")
    current_findings: int = Field(description="Number of findings so far")
    confidence_level: ConfidenceLevel = Field(description="Current confidence")

class DynamicPrompt(BaseModel):
    """Dynamically constructed prompt"""
    system_prompt: str = Field(description="Final system prompt")
    user_prompt: str = Field(description="Final user prompt")
    template_used: str = Field(description="Template ID used")
    variables_filled: Dict[str, str] = Field(description="Variables and their values")
    reasoning_type: ReasoningType = Field(description="Reasoning type applied")
    confidence_expected: float = Field(description="Expected confidence", ge=0.0, le=1.0)


# ============================================================================
# Memory and State Management Models
# ============================================================================

class MemoryState(BaseModel):
    """Research state stored in Brain memory"""
    workflow_id: str = Field(description="Workflow identifier")
    session_id: str = Field(description="Session identifier")
    current_strategy: str = Field(description="Current reasoning strategy")
    findings_count: int = Field(description="Number of findings")
    quality_score: float = Field(description="Current quality score", ge=0.0, le=1.0)
    time_elapsed: int = Field(description="Time elapsed in minutes")
    last_updated: str = Field(description="Last update timestamp")

class ContextMemory(BaseModel):
    """Context information stored in memory"""
    key: str = Field(description="Memory key")
    value: Any = Field(description="Memory value")
    scope: str = Field(description="Memory scope: workflow, session, actor, global")
    expires_at: Optional[str] = Field(description="Expiration timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ============================================================================
# Learning and Adaptation Models
# ============================================================================

class LearningInsight(BaseModel):
    """Simple learning insight from research experience"""
    insight: str = Field(description="The key insight learned")
    context: str = Field(description="When this insight applies")
    impact: str = Field(description="high, medium, or low impact")
    confidence: ConfidenceLevel = Field(description="Confidence in insight")
    evidence: str = Field(description="Supporting evidence")
    actionable_change: str = Field(description="What should change")

class AdaptationDecision(BaseModel):
    """Decision to adapt reasoning strategy"""
    trigger: AdaptationTrigger = Field(description="What triggered adaptation")
    current_strategy: str = Field(description="Current strategy name")
    new_strategy: str = Field(description="New strategy to use")
    reason: str = Field(description="Why adaptation is needed")
    expected_improvement: str = Field(description="Expected improvement")
    confidence: ConfidenceLevel = Field(description="Confidence in decision")

class QualityMetrics(BaseModel):
    """Simplified quality assessment for smaller AI models"""
    overall_quality: str = Field(description="excellent, good, fair, or poor")
    confidence_score: float = Field(description="Numeric confidence", ge=0.0, le=1.0)
    improvement_needed: str = Field(description="What needs improvement")


# ============================================================================
# Search Integration Models (from Jina AI)
# ============================================================================

class SearchResult(BaseModel):
    """Individual search result from Jina AI"""
    title: str = Field(description="Result title")
    url: str = Field(description="Result URL")
    content: str = Field(description="Result content")
    relevance: str = Field(description="high, medium, or low relevance")
    source_type: str = Field(description="Type of source")

class JinaSearchResponse(BaseModel):
    """Complete response from Jina AI search"""
    results: List[SearchResult] = Field(description="Search results")
    total_results: int = Field(description="Total number of results")
    query_used: str = Field(description="Query that was used")

class SearchStrategy(BaseModel):
    """Simplified search strategy for smaller AI models"""
    strategy_name: str = Field(description="Strategy name")
    query_patterns: List[str] = Field(description="Query patterns to use")
    max_results: int = Field(description="Maximum results per query")


# ============================================================================
# Workflow and Coordination Models
# ============================================================================

class WorkflowStep(BaseModel):
    """Simplified workflow step for smaller AI models"""
    step_id: str = Field(description="Step identifier")
    name: str = Field(description="Step name")
    description: str = Field(description="What this step does")
    reasoning_strategy: str = Field(description="Strategy to use")

class DynamicWorkflow(BaseModel):
    """Simplified research workflow for smaller AI models"""
    workflow_id: str = Field(description="Workflow identifier")
    query: str = Field(description="Research query")
    steps: List[WorkflowStep] = Field(description="Workflow steps")
    estimated_time: int = Field(description="Total estimated time in minutes")

class ExecutionStatus(BaseModel):
    """Simplified execution status for smaller AI models"""
    workflow_id: str = Field(description="Workflow identifier")
    current_step: str = Field(description="Current step ID")
    progress_percent: int = Field(description="Progress percentage", ge=0, le=100)
    quality_score: float = Field(description="Current quality", ge=0.0, le=1.0)


# ============================================================================
# Event and Communication Models
# ============================================================================

class MemoryEvent(BaseModel):
    """Memory change event for coordination"""
    event_id: str = Field(description="Event identifier")
    event_type: str = Field(description="Type of event")
    memory_key: str = Field(description="Memory key that changed")
    action: str = Field(description="set, delete, or update")
    data: Any = Field(description="New data value")
    scope: str = Field(description="Memory scope")
    timestamp: str = Field(description="Event timestamp")
    agent_id: str = Field(description="Agent that triggered event")

class CoordinationSignal(BaseModel):
    """Signal for cross-agent coordination"""
    signal_type: str = Field(description="Type of coordination signal")
    from_agent: str = Field(description="Sending agent")
    to_agents: List[str] = Field(description="Target agents")
    message: str = Field(description="Coordination message")
    data: Dict[str, Any] = Field(description="Signal data")
    priority: str = Field(description="high, medium, or low priority")
    timestamp: str = Field(description="Signal timestamp")


# ============================================================================
# Configuration and Control Models
# ============================================================================

class DynamicConfig(BaseModel):
    """Configuration for dynamic reasoning system"""
    max_strategies: int = Field(default=5, description="Maximum strategies to consider")
    adaptation_threshold: float = Field(default=0.7, description="Quality threshold for adaptation")
    memory_retention_hours: int = Field(default=24, description="How long to keep memory")
    parallel_limit: int = Field(default=3, description="Maximum parallel operations")
    quality_target: float = Field(default=0.8, description="Target quality score")
    time_budget_minutes: int = Field(default=30, description="Total time budget")

class StrategySelection(BaseModel):
    """Strategy selection result"""
    selected_strategy: str = Field(description="Selected strategy name")
    reasoning_type: ReasoningType = Field(description="Type of reasoning")
    confidence: ConfidenceLevel = Field(description="Confidence in selection")
    alternatives: List[str] = Field(description="Alternative strategies considered")
    selection_reason: str = Field(description="Why this strategy was selected")


# ============================================================================
# Simple Response Models for AI Generation
# ============================================================================

class SimpleResponse(BaseModel):
    """Simple response for basic operations"""
    success: bool = Field(description="Operation success")
    message: str = Field(description="Response message")
    data: Optional[Dict[str, Any]] = Field(description="Optional response data")

class StrategyList(BaseModel):
    """List of reasoning strategies"""
    strategies: List[ReasoningStrategy] = Field(description="Available strategies")

class PromptList(BaseModel):
    """List of prompt templates"""
    templates: List[PromptTemplate] = Field(description="Available templates")

class InsightList(BaseModel):
    """List of learning insights"""
    insights: List[LearningInsight] = Field(description="Learning insights")

class MetricsList(BaseModel):
    """List of quality metrics"""
    metrics: List[QualityMetrics] = Field(description="Quality assessments")


# ============================================================================
# Additional Research Models for app.call() Schema Wrapping
# ============================================================================

class QueryAnalysis(BaseModel):
    """Query analysis result"""
    query: str = Field(description="Original research query")
    analysis: str = Field(description="Detailed query analysis")
    key_concepts: List[str] = Field(description="Key concepts identified")
    context_type: str = Field(description="Type of research context")
    complexity: str = Field(description="Complexity level")
    research_intent: str = Field(description="Research intent (comprehensive, focused, etc.)")
    estimated_scope: str = Field(description="Estimated scope (broad, narrow)")
    confidence: float = Field(description="Confidence score", ge=0.0, le=1.0)

class DomainStrategy(BaseModel):
    """Domain-specific research strategy"""
    strategy_name: str = Field(description="Name of the strategy")
    query_patterns: List[str] = Field(description="Query patterns to use")
    max_results: int = Field(description="Maximum results per query")

class ResearchResults(BaseModel):
    """Research execution results"""
    success: bool = Field(description="Whether research was successful")
    strategy_used: str = Field(description="Strategy that was used")
    queries_executed: List[str] = Field(description="Queries that were executed")
    raw_results: int = Field(description="Number of raw results found")
    final_results: int = Field(description="Number of final filtered results")
    search_results: List[Dict[str, Any]] = Field(description="Search results data")
    analysis: str = Field(description="Analysis of the search results")
    quality_score: float = Field(description="Quality score", ge=0.0, le=1.0)
    execution_time: str = Field(description="Execution time estimate")
    error: Optional[str] = Field(default=None, description="Error message if failed")

class SynthesisData(BaseModel):
    """Research synthesis result"""
    synthesis_response: str = Field(description="Synthesized research response")
    key_insights: List[str] = Field(description="Key insights from synthesis")
    reasoning_type: str = Field(description="Type of reasoning used")
    sources_used: int = Field(description="Number of sources used")
    source_urls: List[str] = Field(description="URLs of sources used")
    confidence_level: float = Field(description="Confidence level as float", ge=0.0, le=1.0)
    synthesis_quality: str = Field(description="Quality of synthesis")
    limitations: str = Field(description="Limitations of the synthesis")

class ValidationData(BaseModel):
    """Research validation result"""
    validation_status: str = Field(description="Validation status (valid, needs_revision, invalid)")
    accuracy_score: int = Field(description="Accuracy score as percentage", ge=0, le=100)
    validation_report: str = Field(description="Detailed validation report")
    sources_checked: int = Field(description="Number of sources checked")
    consistency_level: str = Field(description="Consistency level (high, medium, low)")
    recommendations: List[str] = Field(description="Validation recommendations")
    confidence: float = Field(description="Confidence in validation", ge=0.0, le=1.0)
    error: Optional[str] = Field(default=None, description="Error message if validation failed")

class ExecutionSummary(BaseModel):
    """Execution summary from orchestrator"""
    strategy_selected: Optional[str] = Field(default=None, description="Selected strategy name")
    strategy_confidence: Optional[str] = Field(default=None, description="Strategy confidence level")
    sources_found: int = Field(default=0, description="Number of sources found")
    quality_achieved: str = Field(default="poor", description="Quality achieved")
    confidence_score: float = Field(default=0.0, description="Confidence score", ge=0.0, le=1.0)
    adaptation_recommended: bool = Field(default=False, description="Whether adaptation was recommended")
    execution_success: bool = Field(default=False, description="Whether execution was successful")

class SynthesisDataNested(BaseModel):
    """Synthesis data with synthesis_response attribute"""
    synthesis_response: str = Field(description="Synthesis response text")
    key_findings: List[str] = Field(default_factory=list, description="Key findings")
    confidence_level: str = Field(default="medium", description="Confidence level")
    sources_used: int = Field(default=0, description="Number of sources used")

class SearchResultsNested(BaseModel):
    """Search results with nested search_results attribute"""
    search_results: List[Dict[str, Any]] = Field(default_factory=list, description="List of search results")
    total_results: int = Field(default=0, description="Total number of results")
    query_used: str = Field(default="", description="Query that was used")

class ResearchResultsData(BaseModel):
    """Research results data from orchestrator"""
    query_analysis: Dict[str, Any] = Field(description="Query analysis data")
    domain_strategy: Dict[str, Any] = Field(description="Domain strategy data")
    search_results: SearchResultsNested = Field(description="Search results data")
    quality_metrics: Dict[str, Any] = Field(description="Quality metrics data")
    synthesis: SynthesisDataNested = Field(description="Synthesis data")
    validation: Optional[Dict[str, Any]] = Field(description="Validation data")

class OrchestrationResult(BaseModel):
    """Complete orchestration result - flexible to handle both success and error cases"""
    workflow_id: Optional[str] = Field(default=None, description="Workflow identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    query: Optional[str] = Field(default=None, description="Original query")
    context_type: Optional[str] = Field(default=None, description="Context type")
    execution_summary: Optional[ExecutionSummary] = Field(default=None, description="Execution summary")
    research_results: Optional[ResearchResultsData] = Field(default=None, description="Research results")
    learning_insights: Optional[Dict[str, Any]] = Field(default=None, description="Learning insights")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata")
    findings: Optional[str] = Field(default=None, description="Fallback findings field")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    
    def __init__(self, **data):
        # Handle error cases by providing defaults
        if 'error' in data and data.get('error'):
            # This is an error response, set defaults
            super().__init__(
                workflow_id=data.get('workflow_id'),
                session_id=data.get('session_id'),
                query=data.get('query'),
                error=data.get('error'),
                execution_summary=ExecutionSummary(),
                research_results=ResearchResultsData(
                    query_analysis={},
                    domain_strategy={},
                    search_results=SearchResultsNested(),
                    quality_metrics={},
                    synthesis=SynthesisDataNested(synthesis_response=""),
                    validation=None
                ),
                learning_insights={},
                metadata={}
            )
        else:
            # Normal case
            super().__init__(**data)