"""
Dynamic Reasoning Infrastructure - Deep Research Agent

Core infrastructure for the dynamic AI reasoning system using functional patterns
with proper @app.reasoner() decorators for all AI operations. Integrates with
Agent Field platform's workflow DAGs and cross-agent memory.

This module provides reasoner functions that should be imported and used in main.py
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from .dynamic_models import (
    AdaptationDecision,
    AdaptationTrigger,
    ConfidenceLevel,
    ContextMemory,
    ContextType,
    CoordinationSignal,
    DynamicConfig,
    DynamicPrompt,
    DynamicWorkflow,
    ExecutionStatus,
    JinaSearchResponse,
    LearningInsight,
    MemoryEvent,
    MemoryState,
    PromptList,
    PromptTemplate,
    QualityMetrics,
    ReasoningStrategy,
    ReasoningType,
    ResearchContext,
    SearchResult,
    SearchStrategy,
    SimpleResponse,
    StrategyList,
    StrategySelection,
    WorkflowStep,
)

# ============================================================================
# Dynamic Prompt Builder Reasoners
# ============================================================================


def create_prompt_reasoners(app):
    """Create prompt-related reasoner functions"""

    @app.reasoner()
    async def create_default_prompt_templates() -> PromptList:
        """Create default prompt templates for different reasoning types"""
        templates = await app.ai(
            system="""You are a prompt engineering expert. Create effective prompt templates
            for different types of research reasoning. Each template should be simple but effective.
            
            For each template provide:
            - template_id: unique identifier
            - name: descriptive name
            - system_prompt: system prompt with {variables} for dynamic content
            - user_prompt: user prompt with {variables} for dynamic content
            - reasoning_type: one of analytical, creative, logical, comparative, causal, predictive, synthesis
            - variables: list of variable names used in prompts
            - effectiveness_score: initial score between 0.0 and 1.0""",
            user="""Create 5 prompt templates for research reasoning:
            1. Analytical reasoning for factual research
            2. Creative reasoning for exploratory research  
            3. Comparative reasoning for comparison tasks
            4. Causal reasoning for cause-effect analysis
            5. Synthesis reasoning for combining information
            
            Each template needs system_prompt, user_prompt with {variables}, and metadata.""",
            schema=PromptList,
        )
        return templates

    @app.reasoner()
    async def select_best_template(
        context: ResearchContext, available_templates: List[str]
    ) -> StrategySelection:
        """Select the best prompt template for given research context"""
        selection = await app.ai(
            system="""You are a template selection expert. Choose the best prompt template
            for the given research context based on reasoning type and context characteristics.
            
            Consider:
            - Query complexity and type
            - Context type (factual, opinion, technical, etc.)
            - Quality requirements
            - Time constraints""",
            user=f"""Select best template for:
            Query: {context.query}
            Context Type: {context.context_type}
            Complexity: {context.complexity}
            Quality Target: {context.quality_target}
            Time Available: {context.time_available} minutes
            
            Available templates: {available_templates}
            
            Return the selected template name and reasoning for the choice.""",
            schema=StrategySelection,
        )
        return selection

    return {
        "create_default_prompt_templates": create_default_prompt_templates,
        "select_best_template": select_best_template,
    }


# ============================================================================
# Reasoning Strategy Registry Reasoners
# ============================================================================


def create_strategy_reasoners(app):
    """Create strategy-related reasoner functions"""

    @app.reasoner()
    async def create_default_reasoning_strategies() -> StrategyList:
        """Create default reasoning strategies for different research types"""
        strategies = await app.ai(
            system="""You are a reasoning strategy expert. Create effective reasoning strategies
            for different types of research tasks. Keep strategies simple but comprehensive.
            
            For each strategy provide:
            - name: unique strategy name
            - type: reasoning type (analytical, creative, logical, etc.)
            - description: what this strategy does
            - context_types: list of suitable context types
            - confidence_threshold: minimum confidence needed (0.0-1.0)
            - time_limit_minutes: time limit for this strategy
            - parallel_capable: whether it can run in parallel""",
            user="""Create 5 reasoning strategies for research:
            1. Analytical strategy for systematic fact-finding
            2. Creative strategy for exploratory research
            3. Comparative strategy for comparison tasks
            4. Causal strategy for cause-effect analysis
            5. Synthesis strategy for combining information
            
            Each strategy needs clear description, suitable contexts, and parameters.""",
            schema=StrategyList,
        )
        return strategies

    @app.reasoner()
    async def select_reasoning_strategy(
        context: ResearchContext,
        available_strategies: List[str],
        performance_data: Dict[str, float],
    ) -> StrategySelection:
        """Select the best reasoning strategy based on context and performance"""
        strategy_options = []
        for name in available_strategies:
            performance = performance_data.get(name, 0.5)
            strategy_options.append(f"{name} (avg performance: {performance:.2f})")

        selection = await app.ai(
            system="""You are a strategy selection expert. Choose the best reasoning strategy
            based on research context and historical performance.
            
            Consider:
            - Query type and complexity
            - Available time and quality requirements
            - Historical performance of strategies
            - Context suitability""",
            user=f"""Select best strategy for:
            Query: {context.query}
            Context Type: {context.context_type}
            Complexity: {context.complexity}
            Time Available: {context.time_available} minutes
            Quality Target: {context.quality_target}
            Current Findings: {context.current_findings}
            
            Available strategies with performance: {strategy_options}
            
            Return the selected strategy name and reasoning.""",
            schema=StrategySelection,
        )
        return selection

    return {
        "create_default_reasoning_strategies": create_default_reasoning_strategies,
        "select_reasoning_strategy": select_reasoning_strategy,
    }


# ============================================================================
# Learning and Adaptation Reasoners
# ============================================================================


def create_learning_reasoners(app):
    """Create learning and adaptation reasoner functions"""

    @app.reasoner()
    async def analyze_research_quality(
        findings_count: int,
        time_elapsed: int,
        confidence_scores: List[float],
        source_diversity: int,
    ) -> QualityMetrics:
        """Analyze the quality of current research progress"""
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.0
        )

        metrics = await app.ai(
            system="""You are a research quality analyst. Assess the quality of research progress
            based on quantitative metrics and provide improvement suggestions.
            
            Consider:
            - Number and quality of findings
            - Time efficiency
            - Confidence levels
            - Source diversity""",
            user=f"""Assess research quality:
            Findings Count: {findings_count}
            Time Elapsed: {time_elapsed} minutes
            Average Confidence: {avg_confidence:.2f}
            Source Diversity: {source_diversity} unique sources
            Individual Confidence Scores: {confidence_scores}
            
            Provide overall quality assessment and improvement suggestions.""",
            schema=QualityMetrics,
        )
        return metrics

    @app.reasoner()
    async def generate_learning_insight(
        context: str, evidence: str, outcome: str, performance_score: float
    ) -> LearningInsight:
        """Generate a learning insight from research experience"""
        insight = await app.ai(
            system="""You are a learning analyst. Extract actionable insights from research experiences
            that can improve future research performance.
            
            Focus on:
            - What worked well or poorly
            - Context-specific patterns
            - Actionable improvements
            - Confidence in the insight""",
            user=f"""Generate learning insight from:
            Context: {context}
            Evidence: {evidence}
            Outcome: {outcome}
            Performance Score: {performance_score}
            
            Extract a key insight that can improve future research.""",
            schema=LearningInsight,
        )
        return insight

    @app.reasoner()
    async def decide_strategy_adaptation(
        current_strategy: str,
        quality_metrics: QualityMetrics,
        time_remaining: int,
        available_strategies: List[str],
    ) -> AdaptationDecision:
        """Decide whether to adapt the current reasoning strategy"""
        decision = await app.ai(
            system="""You are an adaptation decision expert. Decide whether to change the current
            reasoning strategy based on quality metrics and constraints.
            
            Consider:
            - Current quality vs targets
            - Time remaining
            - Alternative strategies available
            - Cost of switching strategies""",
            user=f"""Decide on strategy adaptation:
            Current Strategy: {current_strategy}
            Quality Assessment: {quality_metrics.overall_quality}
            Confidence Score: {quality_metrics.confidence_score}
            Overall Quality: {quality_metrics.overall_quality}
            Time Remaining: {time_remaining} minutes
            Available Alternatives: {available_strategies}
            
            Should we adapt the strategy? If yes, which one and why?""",
            schema=AdaptationDecision,
        )
        return decision

    return {
        "analyze_research_quality": analyze_research_quality,
        "generate_learning_insight": generate_learning_insight,
        "decide_strategy_adaptation": decide_strategy_adaptation,
    }


# ============================================================================
# Search Strategy Reasoners
# ============================================================================


def create_search_reasoners(app):
    """Create search-related reasoner functions"""

    @app.reasoner()
    async def create_search_strategy(
        query: str, context_type: ContextType, time_budget: int
    ) -> SearchStrategy:
        """Create a dynamic search strategy for the research query"""
        strategy = await app.ai(
            system="""You are a search strategy expert. Create effective search strategies
            that maximize information gathering within time and quality constraints.
            
            Consider:
            - Query type and complexity
            - Context requirements
            - Time constraints
            - Search pattern effectiveness""",
            user=f"""Create search strategy for:
            Query: {query}
            Context Type: {context_type}
            Time Budget: {time_budget} minutes
            
            Design query patterns, result limits, and execution approach.""",
            schema=SearchStrategy,
        )
        return strategy

    @app.reasoner()
    async def optimize_search_queries(
        base_query: str, context_type: ContextType, max_queries: int = 5
    ) -> List[str]:
        """Generate optimized search query variations"""
        current_year = datetime.now().year

        queries = await app.ai(
            system=f"""You are a search query optimization expert. Generate effective search query
            variations that maximize information coverage for research.
            
            Current year: {current_year}
            
            Consider:
            - Temporal variations (recent, {current_year}, latest)
            - Perspective variations (analysis, research, study)
            - Specificity variations
            - Context-appropriate terms""",
            user=f"""Generate {max_queries} optimized search queries for:
            Base Query: {base_query}
            Context Type: {context_type}
            
            Return a simple list of query strings, each on a new line.""",
        )

        # Parse the response into a list
        query_list = [q.strip() for q in queries.split("\n") if q.strip()]
        return query_list[:max_queries]

    return {
        "create_search_strategy": create_search_strategy,
        "optimize_search_queries": optimize_search_queries,
    }


# ============================================================================
# Workflow Management Reasoners
# ============================================================================


def create_workflow_reasoners(app):
    """Create workflow management reasoner functions"""

    @app.reasoner()
    async def create_dynamic_workflow(
        query: str, context: ResearchContext, available_strategies: List[str]
    ) -> DynamicWorkflow:
        """Create a dynamic research workflow based on query and context"""
        workflow = await app.ai(
            system="""You are a workflow planning expert. Create efficient research workflows
            that coordinate multiple reasoning strategies and steps.
            
            Consider:
            - Query complexity and requirements
            - Available time and quality targets
            - Strategy capabilities and dependencies
            - Parallel execution opportunities""",
            user=f"""Create research workflow for:
            Query: {query}
            Context Type: {context.context_type}
            Complexity: {context.complexity}
            Time Available: {context.time_available} minutes
            Quality Target: {context.quality_target}
            Available Strategies: {available_strategies}
            
            Design workflow steps with dependencies and parallel execution groups.""",
            schema=DynamicWorkflow,
        )
        return workflow

    return {"create_dynamic_workflow": create_dynamic_workflow}


# ============================================================================
# Memory Management Functions (Non-AI operations)
# ============================================================================


async def initialize_research_memory(
    app, workflow_id: str, session_id: str, context: ResearchContext
) -> MemoryState:
    """Initialize research state in Brain memory"""
    print("🔍 DEBUG: Starting memory initialization...")
    
    try:
        # Store research context in workflow-scoped memory
        print("🔍 DEBUG: Preparing context data for serialization...")
        context_data = context.model_dump(mode='json')  # Ensure enums are serialized to strings
        print(f"🔍 DEBUG: Context data prepared: {type(context_data)}, keys: {list(context_data.keys()) if isinstance(context_data, dict) else 'not dict'}")
        print(f"🔍 DEBUG: Sample enum values: context_type={context_data.get('context_type')}, confidence_level={context_data.get('confidence_level')}")
        
        print("🔍 DEBUG: Calling app.memory.set for research_context...")
        await app.memory.set("research_context", context_data)
        print("✅ DEBUG: Successfully stored research_context")
        
        # Initialize memory state
        print("🔍 DEBUG: Creating memory state object...")
        memory_state = MemoryState(
            workflow_id=workflow_id,
            session_id=session_id,
            current_strategy="initializing",
            findings_count=0,
            quality_score=0.0,
            time_elapsed=0,
            last_updated=datetime.now().isoformat(),
        )
        print(f"✅ DEBUG: Memory state created: {memory_state}")
        
        print("🔍 DEBUG: Preparing memory state for serialization...")
        memory_data = memory_state.model_dump(mode='json')  # Ensure enums are serialized to strings
        print(f"🔍 DEBUG: Memory data prepared: {type(memory_data)}, keys: {list(memory_data.keys()) if isinstance(memory_data, dict) else 'not dict'}")
        
        print("🔍 DEBUG: Calling app.memory.set for memory_state...")
        await app.memory.set("memory_state", memory_data)
        print("✅ DEBUG: Successfully stored memory_state")
        
        print("✅ DEBUG: Memory initialization completed successfully")
        return memory_state
        
    except Exception as e:
        print(f"❌ DEBUG: Exception in memory initialization: {type(e).__name__}: {e}")
        import traceback
        print(f"❌ DEBUG: Traceback: {traceback.format_exc()}")
        raise


async def get_research_context(app) -> Optional[ResearchContext]:
    """Get current research context from Brain memory"""
    context_data = await app.memory.get("research_context")
    if context_data:
        return ResearchContext(**context_data)
    return None


async def update_research_context(app, updates: Dict[str, Any]) -> None:
    """Update research context in Brain memory"""
    current_context = await get_research_context(app)
    if current_context:
        context_dict = current_context.model_dump(mode='json')
        context_dict.update(updates)
        await app.memory.set("research_context", context_dict)


async def get_memory_state(app) -> Optional[MemoryState]:
    """Get current memory state from Brain memory"""
    state_data = await app.memory.get("memory_state")
    if state_data:
        return MemoryState(**state_data)
    return None


async def update_memory_state(app, updates: Dict[str, Any]) -> None:
    """Update memory state in Brain memory"""
    current_state = await get_memory_state(app)
    if current_state:
        state_dict = current_state.model_dump(mode='json')
        state_dict.update(updates)
        state_dict["last_updated"] = datetime.now().isoformat()
        await app.memory.set("memory_state", state_dict)


async def store_learning_insights(app, insights: List[LearningInsight]) -> None:
    """Store learning insights in global Brain memory"""
    insight_data = [insight.model_dump(mode='json') for insight in insights]
    await app.memory.global_scope.set("learning_insights", insight_data)


async def get_learning_insights(app) -> List[LearningInsight]:
    """Get learning insights from global Brain memory"""
    print("🔍 DEBUG: Starting get_learning_insights...")
    try:
        print("🔍 DEBUG: Calling app.memory.global_scope.get('learning_insights')...")
        insight_data = await app.memory.global_scope.get("learning_insights", default=[])
        print(f"🔍 DEBUG: Retrieved {len(insight_data)} learning insights")
        return [LearningInsight(**insight) for insight in insight_data]
    except Exception as e:
        print(f"❌ DEBUG: get_learning_insights failed: {str(e)}")
        print("🔍 DEBUG: Returning empty list as fallback")
        return []


async def store_strategy_performance(
    app, strategy_name: str, performance_scores: List[float]
) -> None:
    """Store strategy performance history in Brain memory"""
    print(f"🔍 DEBUG: Storing strategy performance for {strategy_name}: {performance_scores[-5:]}")
    try:
        await app.memory.set(
            f"strategy_performance_{strategy_name}", performance_scores[-20:]
        )
        print(f"✅ DEBUG: Successfully stored strategy performance for {strategy_name}")
    except Exception as e:
        print(f"❌ DEBUG: store_strategy_performance failed for {strategy_name}: {str(e)}")


async def get_strategy_performance(app, strategy_name: str) -> List[float]:
    """Get strategy performance history from Brain memory"""
    print(f"🔍 DEBUG: Getting strategy performance for {strategy_name}...")
    try:
        result = await app.memory.get(f"strategy_performance_{strategy_name}", default=[])
        print(f"🔍 DEBUG: Retrieved {len(result)} performance scores for {strategy_name}")
        return result
    except Exception as e:
        print(f"❌ DEBUG: get_strategy_performance failed for {strategy_name}: {str(e)}")
        print("🔍 DEBUG: Returning empty list as fallback")
        return []


# ============================================================================
# Memory Event Handler Setup Functions
# ============================================================================


def setup_memory_event_handlers(app):
    """Setup memory event handlers for real-time coordination (simplified for Brain SDK compatibility)"""

    # Note: Brain SDK doesn't support memory event handlers like app.memory.on_change()
    # Instead, we'll use polling-based coordination through memory checks in reasoners
    print("🔄 Memory event handlers configured (using polling-based coordination)")

    # Store coordination configuration in memory
    coordination_config = {
        "enabled": True,
        "polling_interval": 5,  # seconds
        "signal_retention": 300,  # 5 minutes
        "max_signals": 100,
    }

    # This will be used by reasoners to coordinate via memory polling
    # Each reasoner can check for coordination signals periodically
    return coordination_config

    # Memory event handlers removed - Brain SDK doesn't support app.memory.on_change()
    # Coordination will be handled through polling-based memory checks in reasoners
    pass


# ============================================================================
# Utility Functions
# ============================================================================


def build_dynamic_prompt(
    template: PromptTemplate, variables: Dict[str, str], context: ResearchContext
) -> DynamicPrompt:
    """Build a dynamic prompt from template and variables"""
    # Fill template variables
    system_prompt = template.system_prompt
    user_prompt = template.user_prompt

    for var, value in variables.items():
        system_prompt = system_prompt.replace(f"{{{var}}}", value)
        user_prompt = user_prompt.replace(f"{{{var}}}", value)

    return DynamicPrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        template_used=template.template_id,
        variables_filled=variables,
        reasoning_type=template.reasoning_type,
        confidence_expected=0.8,  # Default confidence for simplified model
    )


def update_execution_status(
    current_status: ExecutionStatus,
    completed_step: str,
    quality_score: float,
    time_elapsed: int,
) -> ExecutionStatus:
    """Update workflow execution status based on completed step (simplified for smaller AI models)"""
    # Simplified progress calculation
    progress_percent = min(current_status.progress_percent + 20, 100)  # Simple increment

    # Update status with simplified fields
    new_status = ExecutionStatus(
        workflow_id=current_status.workflow_id,
        current_step="next_step" if progress_percent < 100 else "completed",
        progress_percent=progress_percent,
        quality_score=quality_score,
    )

    return new_status


def get_average_performance(performance_scores: List[float]) -> float:
    """Calculate average performance from score history"""
    if not performance_scores:
        return 0.5  # Default neutral performance

    # Use last 10 scores for recent performance
    recent_scores = performance_scores[-10:]
    return sum(recent_scores) / len(recent_scores)


async def cleanup_expired_memory(app) -> None:
    """Clean up expired context memory entries"""
    # This would be called periodically to clean up expired entries
    # Implementation depends on specific memory patterns used
    pass


def create_workflow_id() -> str:
    """Create a unique workflow identifier"""
    return f"workflow_{uuid.uuid4().hex[:8]}"


def create_session_id() -> str:
    """Create a unique session identifier"""
    return f"session_{uuid.uuid4().hex[:8]}"


# ============================================================================
# Infrastructure Initialization Function
# ============================================================================


def initialize_dynamic_infrastructure(app):
    """Initialize all dynamic reasoning infrastructure with the app instance"""

    # Create all reasoner functions
    prompt_reasoners = create_prompt_reasoners(app)
    strategy_reasoners = create_strategy_reasoners(app)
    learning_reasoners = create_learning_reasoners(app)
    search_reasoners = create_search_reasoners(app)
    workflow_reasoners = create_workflow_reasoners(app)

    # Setup memory event handlers
    setup_memory_event_handlers(app)

    # Return all reasoner functions for easy access
    return {
        **prompt_reasoners,
        **strategy_reasoners,
        **learning_reasoners,
        **search_reasoners,
        **workflow_reasoners,
    }
