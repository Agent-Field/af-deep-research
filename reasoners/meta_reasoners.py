"""
Meta-Intelligence Layer Reasoners - Deep Research Agent

Core meta-reasoners that form the heart of the dynamic AI reasoning system.
These reasoners orchestrate research strategy decisions, adaptive reasoning,
dynamic prompt construction, and context-aware scheduling.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from .dynamic_models import (
    ReasoningStrategy, PromptTemplate, ResearchContext, DynamicPrompt,
    MemoryState, ContextMemory, LearningInsight, AdaptationDecision,
    QualityMetrics, SearchStrategy, WorkflowStep, DynamicWorkflow,
    ExecutionStatus, MemoryEvent, CoordinationSignal, DynamicConfig,
    StrategySelection, SimpleResponse, StrategyList, PromptList,
    ReasoningType, ContextType, ConfidenceLevel, AdaptationTrigger
)
from .dynamic_infrastructure import (
    get_research_context, update_research_context, get_memory_state,
    update_memory_state, get_learning_insights, store_learning_insights,
    get_strategy_performance, store_strategy_performance
)


def create_meta_reasoners(app):
    """Create all meta-intelligence layer reasoners"""
    
    print("🔍 DEBUG: Starting create_meta_reasoners...")
    
    # ============================================================================
    # Meta-Reasoning Controller - Central orchestrator for research strategy decisions
    # ============================================================================
    
    @app.reasoner()
    async def meta_reasoning_controller(
        query: str,
        context_type: Union[str, ContextType],
        time_budget: int,
        quality_target: str
    ) -> StrategySelection:
        """
        Central orchestrator that makes high-level research strategy decisions.
        Analyzes the research request and selects the optimal meta-strategy.
        """
        print("🔍 DEBUG: Starting meta_reasoning_controller...")
        print(f"🔍 DEBUG: Query: {query[:100]}...")
        print(f"🔍 DEBUG: Context: {context_type}, Budget: {time_budget}, Quality: {quality_target}")
        
        try:
            # Robust enum conversion with validation
            if isinstance(context_type, str):
                try:
                    context_type_enum = ContextType(context_type.lower())
                    print(f"🔍 DEBUG: Converted context_type string '{context_type}' to enum: {context_type_enum}")
                except ValueError as e:
                    print(f"⚠️ DEBUG: Invalid context_type '{context_type}', error: {e}, defaulting to FACTUAL")
                    context_type_enum = ContextType.FACTUAL
            elif isinstance(context_type, ContextType):
                context_type_enum = context_type
            else:
                print(f"⚠️ DEBUG: Unexpected context_type type {type(context_type)}, defaulting to FACTUAL")
                context_type_enum = ContextType.FACTUAL
            
            # Get current learning insights to inform decision with error handling
            print("🔍 DEBUG: Getting learning insights...")
            try:
                insights = await get_learning_insights(app)
                print(f"🔍 DEBUG: Retrieved {len(insights)} insights")
                insight_summary = "\n".join([f"- {insight.insight}" for insight in insights[-5:]])
            except Exception as e:
                print(f"⚠️ DEBUG: Error getting learning insights: {e}")
                insights = []
                insight_summary = "No insights available"
            
            # Get strategy performance data with error handling
            print("🔍 DEBUG: Getting strategy performance data...")
            strategy_names = ["analytical", "creative", "comparative", "causal", "synthesis"]
            performance_data = {}
            for strategy in strategy_names:
                try:
                    print(f"🔍 DEBUG: Getting performance for strategy: {strategy}")
                    scores = await get_strategy_performance(app, strategy)
                    performance_data[strategy] = sum(scores) / len(scores) if scores else 0.5
                    print(f"🔍 DEBUG: Strategy {strategy} performance: {performance_data[strategy]:.2f}")
                except Exception as e:
                    print(f"⚠️ DEBUG: Error getting performance for strategy {strategy}: {e}")
                    performance_data[strategy] = 0.5  # Default performance score
            
            strategy_selection = await app.ai(
                system="""You are the Meta-Reasoning Controller, the central intelligence that orchestrates
                all research strategy decisions. Your role is to analyze research requests and select the
                optimal high-level approach based on query characteristics, constraints, and learned experience.
                
                Consider:
                - Query complexity and domain requirements
                - Time and quality constraints
                - Historical performance of different strategies
                - Learning insights from previous research sessions
                - Resource availability and parallel execution opportunities
                
                Select the best meta-strategy that will guide the entire research process.""",
                user=f"""Analyze this research request and select the optimal meta-strategy:
                
                Query: {query}
                Context Type: {context_type_enum}
                Time Budget: {time_budget} minutes
                Quality Target: {quality_target}
                
                Strategy Performance History:
                {chr(10).join([f"- {name}: {score:.2f}" for name, score in performance_data.items()])}
                
                Recent Learning Insights:
                {insight_summary if insight_summary else "No previous insights available"}
                
                Select the best meta-strategy and provide detailed reasoning for your choice.""",
                schema=StrategySelection
            )
            
            # Store the decision in memory for coordination with error handling
            try:
                await update_memory_state(app, {
                    "current_strategy": strategy_selection.selected_strategy,
                    "strategy_confidence": strategy_selection.confidence.value if hasattr(strategy_selection.confidence, 'value') else str(strategy_selection.confidence)
                })
            except Exception as e:
                print(f"⚠️ DEBUG: Error updating memory state: {e}")
            
            print("✅ DEBUG: meta_reasoning_controller completed successfully")
            return strategy_selection
            
        except Exception as e:
            print(f"❌ DEBUG: Error in meta_reasoning_controller: {e}")
            # Return a fallback strategy selection
            return StrategySelection(
                selected_strategy="analytical",
                reasoning_type=ReasoningType.ANALYTICAL,
                confidence=ConfidenceLevel.MEDIUM,
                alternatives=["creative", "logical", "comparative"],
                selection_reason="Fallback strategy due to error in meta-reasoning controller"
            )
    
    print("✅ DEBUG: meta_reasoning_controller reasoner registered")
    
    # ============================================================================
    # Adaptive Reasoning Strategist - Dynamic strategy selection and configuration
    # ============================================================================
    
    @app.reasoner()
    async def adaptive_reasoning_strategist(
        current_context: ResearchContext,
        quality_metrics: QualityMetrics,
        time_remaining: int
    ) -> AdaptationDecision:
        """
        Dynamically adapts reasoning strategies based on real-time performance
        and changing context. Monitors research progress and triggers adaptations.
        """
        # Get current memory state
        memory_state = await get_memory_state(app)
        current_strategy = memory_state.current_strategy if memory_state else "unknown"
        
        # Get recent learning insights for adaptation guidance
        insights = await get_learning_insights(app)
        relevant_insights = [
            insight for insight in insights 
            if insight.context.lower() in current_context.query.lower() or
               insight.impact in ["high", "medium"]
        ]
        
        adaptation_decision = await app.ai(
            system="""You are the Adaptive Reasoning Strategist, responsible for dynamic strategy
            adaptation during research execution. You monitor research progress in real-time and
            decide when and how to adapt strategies for optimal performance.
            
            Your decisions are based on:
            - Current research quality and progress
            - Time constraints and efficiency
            - Learning insights from similar contexts
            - Strategy performance patterns
            - Resource utilization and bottlenecks
            
            Make adaptive decisions that maximize research effectiveness.""",
            user=f"""Analyze current research progress and decide on strategy adaptation:
            
            Current Strategy: {current_strategy}
            Research Query: {current_context.query}
            Context Type: {current_context.context_type}
            Time Remaining: {time_remaining} minutes
            
            Quality Assessment:
            - Overall Quality: {quality_metrics.overall_quality}
            - Overall Quality: {quality_metrics.overall_quality}
            - Improvement Needed: {quality_metrics.improvement_needed}
            - Confidence Score: {quality_metrics.confidence_score}
            
            Relevant Learning Insights:
            {chr(10).join([f"- {insight.insight} (Impact: {insight.impact})" for insight in relevant_insights[-3:]]) if relevant_insights else "No relevant insights available"}
            
            Should we adapt the current strategy? If yes, what changes are needed?""",
            schema=AdaptationDecision
        )
        
        # If adaptation is recommended, update memory state
        if adaptation_decision.new_strategy != adaptation_decision.current_strategy:
            await update_memory_state(app, {
                "current_strategy": adaptation_decision.new_strategy,
                "adaptation_reason": adaptation_decision.reason
            })
        
        return adaptation_decision
    
    # ============================================================================
    # Dynamic Prompt Constructor - Context-aware prompt building and evolution
    # ============================================================================
    
    @app.reasoner()
    async def dynamic_prompt_constructor(
        reasoning_type: ReasoningType,
        research_context: ResearchContext,
        domain_knowledge: str,
        previous_findings: str = ""
    ) -> DynamicPrompt:
        """
        Constructs context-aware prompts that evolve based on research progress
        and domain-specific requirements. Builds prompts optimized for specific reasoning types.
        """
        # Get learning insights to improve prompt construction
        insights = await get_learning_insights(app)
        prompt_insights = [
            insight for insight in insights 
            if "prompt" in insight.insight.lower() or "question" in insight.insight.lower()
        ]
        
        dynamic_prompt = await app.ai(
            system="""You are the Dynamic Prompt Constructor, an expert in creating context-aware
            prompts that maximize AI reasoning effectiveness. You build prompts that evolve based on
            research progress, domain requirements, and learned patterns.
            
            Your prompts should:
            - Be tailored to the specific reasoning type and context
            - Incorporate domain knowledge and previous findings
            - Use effective prompt engineering techniques
            - Adapt based on learning insights from previous research
            - Guide AI toward high-quality, relevant responses
            
            Create prompts that unlock the full potential of AI reasoning.""",
            user=f"""Construct an optimized prompt for this research context:
            
            Reasoning Type: {reasoning_type}
            Research Query: {research_context.query}
            Context Type: {research_context.context_type}
            Complexity: {research_context.complexity}
            Quality Target: {research_context.quality_target}
            
            Domain Knowledge Available:
            {domain_knowledge[:1000] if domain_knowledge else "No specific domain knowledge provided"}
            
            Previous Findings:
            {previous_findings[:800] if previous_findings else "No previous findings available"}
            
            Prompt Construction Insights:
            {chr(10).join([f"- {insight.insight}" for insight in prompt_insights[-3:]]) if prompt_insights else "No prompt-specific insights available"}
            
            Create a system prompt and user prompt that will guide AI reasoning effectively for this specific context.""",
            schema=DynamicPrompt
        )
        
        return dynamic_prompt
    
    # ============================================================================
    # Context-Aware Scheduler - Intelligent execution flow management
    # ============================================================================
    
    @app.reasoner()
    async def context_aware_scheduler(
        workflow: DynamicWorkflow,
        available_resources: Dict[str, Any],
        priority_constraints: List[str]
    ) -> ExecutionStatus:
        """
        Manages intelligent execution flow with context-aware scheduling.
        Optimizes resource allocation and parallel execution based on constraints.
        """
        # Get current memory state for scheduling context
        memory_state = await get_memory_state(app)
        
        execution_status = await app.ai(
            system="""You are the Context-Aware Scheduler, responsible for intelligent execution
            flow management. You optimize resource allocation, manage parallel execution, and
            ensure efficient workflow progression based on real-time constraints and priorities.
            
            Your scheduling decisions consider:
            - Resource availability and constraints
            - Task dependencies and parallel opportunities
            - Priority requirements and deadlines
            - Current system state and progress
            - Quality checkpoints and validation needs
            
            Create execution plans that maximize efficiency while maintaining quality.""",
            user=f"""Create an optimized execution schedule for this workflow:
            
            Workflow ID: {workflow.workflow_id}
            Research Query: {workflow.query}
            Total Steps: {len(workflow.steps)}
            Estimated Time: {workflow.estimated_time} minutes
            
            Workflow Steps:
            {chr(10).join([f"- {step.name} (strategy: {step.reasoning_strategy})" for step in workflow.steps])}
            
            Available Resources:
            {chr(10).join([f"- {key}: {value}" for key, value in available_resources.items()])}
            
            Priority Constraints:
            {chr(10).join([f"- {constraint}" for constraint in priority_constraints])}
            
            Current Progress:
            - Findings Count: {memory_state.findings_count if memory_state else 0}
            - Quality Score: {memory_state.quality_score if memory_state else 0.0}
            - Time Elapsed: {memory_state.time_elapsed if memory_state else 0} minutes
            
            Create an execution status that optimizes the workflow execution.""",
            schema=ExecutionStatus
        )
        
        # Update memory state with execution status
        await update_memory_state(app, {
            "current_step": execution_status.current_step,
            "progress_percent": execution_status.progress_percent,
            "quality_score": execution_status.quality_score
        })
        
        return execution_status
    
    # ============================================================================
    # Meta-Learning Coordinator - Cross-session learning and improvement
    # ============================================================================
    
    @app.reasoner()
    async def meta_learning_coordinator(
        session_results: Dict[str, Any],
        performance_metrics: QualityMetrics,
        strategy_used: str
    ) -> LearningInsight:
        """
        Coordinates learning across research sessions to improve future performance.
        Extracts insights and updates the system's knowledge base.
        """
        # Get existing insights for context
        existing_insights = await get_learning_insights(app)
        
        learning_insight = await app.ai(
            system="""You are the Meta-Learning Coordinator, responsible for extracting actionable
            insights from research sessions to improve future performance. You analyze patterns,
            identify successful strategies, and generate learning insights that enhance the system's
            intelligence over time.
            
            Focus on:
            - Strategy effectiveness patterns
            - Context-specific optimizations
            - Quality improvement opportunities
            - Efficiency enhancements
            - Failure mode prevention
            
            Generate insights that make the system progressively smarter.""",
            user=f"""Analyze this research session and extract key learning insights:
            
            Strategy Used: {strategy_used}
            
            Session Results:
            {chr(10).join([f"- {key}: {value}" for key, value in session_results.items()])}
            
            Performance Metrics:
            - Overall Quality: {performance_metrics.overall_quality}
            - Overall Quality: {performance_metrics.overall_quality}
            - Confidence Score: {performance_metrics.confidence_score}
            - Improvement Needed: {performance_metrics.improvement_needed}
            
            Existing Insights Count: {len(existing_insights)}
            
            What key insight can be extracted from this session to improve future research performance?""",
            schema=LearningInsight
        )
        
        # Store the new insight
        updated_insights = existing_insights + [learning_insight]
        await store_learning_insights(app, updated_insights[-50:])  # Keep last 50 insights
        
        # Update strategy performance
        current_scores = await get_strategy_performance(app, strategy_used)
        current_scores.append(performance_metrics.confidence_score)
        await store_strategy_performance(app, strategy_used, current_scores)
        
        return learning_insight
    
    return {
        'meta_reasoning_controller': meta_reasoning_controller,
        'adaptive_reasoning_strategist': adaptive_reasoning_strategist,
        'dynamic_prompt_constructor': dynamic_prompt_constructor,
        'context_aware_scheduler': context_aware_scheduler,
        'meta_learning_coordinator': meta_learning_coordinator
    }