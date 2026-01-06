"""
Dynamic Research Orchestrator - Deep Research Agent

Main orchestrator that coordinates all meta-reasoners and universal reasoners
to create adaptive research intelligence. Implements dynamic workflow management
using Agent Field DAG patterns with parallel execution and memory-based learning.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from . import universal_reasoners
from .dynamic_infrastructure import (
    get_learning_insights,
    get_memory_state,
    get_research_context,
    initialize_dynamic_infrastructure,
    initialize_research_memory,
    setup_memory_event_handlers,
    store_learning_insights,
    update_memory_state,
    update_research_context,
)
from .dynamic_models import (
    AdaptationDecision,
    AdaptationTrigger,
    ConfidenceLevel,
    ContextMemory,
    ContextType,
    CoordinationSignal,
    DomainStrategy,
    DynamicConfig,
    DynamicPrompt,
    DynamicWorkflow,
    ExecutionStatus,
    LearningInsight,
    MemoryEvent,
    MemoryState,
    PromptTemplate,
    QualityMetrics,
    QueryAnalysis,
    ReasoningStrategy,
    ReasoningType,
    ResearchContext,
    ResearchResults,
    SearchStrategy,
    SimpleResponse,
    StrategySelection,
    SynthesisData,
    ValidationData,
    WorkflowStep,
)
from .meta_reasoners import create_meta_reasoners


def create_research_orchestrator(app):
    """Create the main dynamic research orchestrator"""

    print("🔍 DEBUG: Starting create_research_orchestrator...")

    # Initialize infrastructure and reasoners
    print("🔍 DEBUG: Initializing dynamic infrastructure...")
    initialize_dynamic_infrastructure(app)
    print("🔍 DEBUG: Setting up memory event handlers...")
    setup_memory_event_handlers(app)

    print("🔍 DEBUG: Creating meta reasoners...")
    meta_reasoners = create_meta_reasoners(app)
    print(f"🔍 DEBUG: Meta reasoners created: {list(meta_reasoners.keys())}")

    print("🔍 DEBUG: Creating universal reasoners...")
    universal_reasoners.create_universal_reasoners(app)
    print("🔍 DEBUG: Universal reasoners registered with app")

    # ============================================================================
    # Dynamic Research Orchestrator - Main coordination reasoner
    # ============================================================================

    print("🔍 DEBUG: About to register dynamic_research_orchestrator...")

    @app.reasoner()
    async def dynamic_research_orchestrator(
        query: str,
        context_type: Union[str, ContextType] = "factual",
        complexity: str = "moderate",
        time_budget: int = 30,
        quality_target: str = "high",
        max_iterations: int = 3,
        quality_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Main orchestrator that coordinates all meta-reasoners and universal reasoners
        to execute comprehensive research with adaptive intelligence and Dynamic Hierarchy of Execution (DHE).

        DHE Features:
        - Complexity-based branching (simple/moderate/complex execution paths)
        - Quality-driven recursive loops with adaptive re-execution
        - Parallel execution for complex queries
        - Meta-learning feedback loops for continuous improvement
        """
        # Generate unique workflow and session IDs
        workflow_id = f"research_{uuid.uuid4().hex[:8]}"
        session_id = f"session_{uuid.uuid4().hex[:8]}"

        try:
            print(
                f"🚀 ORCHESTRATOR: Starting dynamic research orchestrator for: {query}"
            )

            # Robust enum conversion with validation at the start
            if isinstance(context_type, str):
                try:
                    context_type_enum = ContextType(context_type.lower())
                    print(
                        f"🔍 DEBUG: Converted context_type string '{context_type}' to enum: {context_type_enum}"
                    )
                except ValueError as e:
                    print(
                        f"⚠️ DEBUG: Invalid context_type '{context_type}', error: {e}, defaulting to FACTUAL"
                    )
                    context_type_enum = ContextType.FACTUAL
            elif isinstance(context_type, ContextType):
                context_type_enum = context_type
            else:
                print(
                    f"⚠️ DEBUG: Unexpected context_type type {type(context_type)}, defaulting to FACTUAL"
                )
                context_type_enum = ContextType.FACTUAL

            # ========================================================================
            # Phase 1: Initialize Research Context and Memory
            # ========================================================================

            print("📋 ORCHESTRATOR: Creating research context...")
            research_context = ResearchContext(
                query=query,
                context_type=context_type_enum,
                complexity=complexity,
                time_available=time_budget,
                quality_target=quality_target,
                current_findings=0,
                confidence_level=ConfidenceLevel.MEDIUM,
            )
            print(
                f"✅ ORCHESTRATOR: Research context created: {research_context.query[:50]}..."
            )

            # Initialize research memory
            print("🧠 ORCHESTRATOR: Initializing research memory...")
            memory_state = await initialize_research_memory(
                app, workflow_id, session_id, research_context
            )
            print(f"✅ ORCHESTRATOR: Memory initialized successfully")

            print(f"🚀 Starting research workflow: {workflow_id}")
            print(f"📋 Query: {query}")
            print(f"🎯 Context: {context_type_enum.value}, Quality: {quality_target}")

            # ========================================================================
            # DHE Phase 1: Complexity Analysis and Dynamic Path Selection
            # ========================================================================

            print(
                "🧠 DHE Phase 1: Analyzing query complexity for dynamic execution path..."
            )

            # Analyze query complexity to determine execution path
            complexity_analysis = await app.ai(
                system="""You are a complexity analysis expert. Analyze research queries to determine their complexity level and optimal execution strategy.
                
                Complexity Levels:
                - SIMPLE: Direct factual queries, single domain, clear answer expected
                - MODERATE: Multi-faceted queries, some analysis required, 2-3 domains
                - COMPLEX: Multi-domain research, synthesis required, comparative analysis, predictive elements
                
                Consider: query scope, domain breadth, analysis depth, synthesis requirements.""",
                user=f"""Analyze this query for complexity and execution requirements:
                
                Query: {query}
                Context: {context_type_enum.value}
                Stated Complexity: {complexity}
                
                Determine:
                1. Actual complexity level (simple/moderate/complex)
                2. Number of domains involved
                3. Whether parallel execution would be beneficial
                4. Estimated reasoner count needed (3-4 for simple, 6-8 for moderate, 10+ for complex)
                5. Key complexity factors
                
                Respond in JSON format:
                {{
                    "complexity_level": "simple|moderate|complex",
                    "domain_count": number,
                    "parallel_beneficial": boolean,
                    "estimated_reasoners": number,
                    "complexity_factors": ["factor1", "factor2"],
                    "execution_strategy": "linear|parallel|hybrid"
                }}""",
            )

            try:
                import json

                complexity_data = json.loads(complexity_analysis)
                actual_complexity = complexity_data.get("complexity_level", complexity)
                execution_strategy = complexity_data.get("execution_strategy", "linear")
                estimated_reasoners = complexity_data.get("estimated_reasoners", 6)
                parallel_beneficial = complexity_data.get("parallel_beneficial", False)

                print(
                    f"✅ Complexity Analysis: {actual_complexity} ({estimated_reasoners} reasoners)"
                )
                print(f"🔄 Execution Strategy: {execution_strategy}")
                print(f"⚡ Parallel Beneficial: {parallel_beneficial}")

            except Exception as e:
                print(f"⚠️ Complexity analysis parsing failed: {e}, using defaults")
                import json  # Ensure json is imported even in exception case

                complexity_data = {
                    "complexity_level": complexity,
                    "execution_strategy": "linear",
                    "estimated_reasoners": 6,
                    "parallel_beneficial": False,
                    "complexity_factors": ["fallback_analysis"],
                }
                actual_complexity = complexity
                execution_strategy = "linear"
                estimated_reasoners = 6
                parallel_beneficial = False

            # Update research context with complexity insights
            research_context.complexity = actual_complexity

            # ========================================================================
            # DHE Phase 2: Dynamic Execution Path Selection
            # ========================================================================

            print(
                f"🎯 DHE Phase 2: Selecting execution path for {actual_complexity} complexity..."
            )

            # Initialize iteration tracking for quality-driven loops
            iteration_count = 0
            quality_history = []
            strategy_performance = {}

            # Initialize adaptation_decision with default values
            adaptation_decision = AdaptationDecision(
                trigger=AdaptationTrigger.QUALITY_THRESHOLD,
                current_strategy="default",
                new_strategy="default",
                confidence=ConfidenceLevel.MEDIUM,
                reason="No adaptation needed",
                expected_improvement="Baseline performance",
            )

            # Determine if we should use parallel execution for complex queries
            if (
                actual_complexity == "complex"
                and parallel_beneficial
                and len(query.split()) > 10
            ):
                print("🚀 Complex query detected - initiating parallel execution path")

                # Generate sub-queries for parallel execution
                sub_query_generation = await app.ai(
                    system="""You are an expert at breaking down complex research queries into focused sub-queries for parallel execution.
                    Create 3-4 specific, focused sub-queries that together comprehensively address the main query.""",
                    user=f"""Break down this complex query into focused sub-queries for parallel research:
                    
                    Main Query: {query}
                    Context: {context_type_enum.value}
                    
                    Generate 3-4 specific sub-queries that:
                    1. Cover different aspects/domains of the main query
                    2. Can be researched independently
                    3. Together provide comprehensive coverage
                    4. Are specific enough for focused research
                    
                    Return as JSON array: ["sub_query_1", "sub_query_2", "sub_query_3", "sub_query_4"]""",
                )

                try:
                    sub_queries = json.loads(sub_query_generation)
                    if len(sub_queries) >= 2:
                        print(
                            f"🔀 Executing parallel research with {len(sub_queries)} sub-queries"
                        )

                        # Execute parallel research orchestrator
                        parallel_result = await app.call(
                            "deepresearchagent.parallel_research_orchestrator",
                            query=query,
                            sub_queries=sub_queries,
                            context_type=context_type_enum,
                            time_budget=time_budget,
                            quality_target=quality_target,
                        )

                        # Return parallel result with DHE metadata
                        parallel_result["dhe_metadata"] = {
                            "execution_path": "parallel",
                            "complexity_analysis": complexity_data,
                            "sub_queries_count": len(sub_queries),
                            "reasoners_utilized": estimated_reasoners
                            * len(sub_queries),
                        }

                        return parallel_result

                except Exception as e:
                    print(
                        f"⚠️ Parallel execution failed, falling back to enhanced linear: {e}"
                    )

            # Continue with enhanced linear execution for simple/moderate or fallback
            print(
                f"📈 Executing enhanced linear path with {estimated_reasoners} reasoners"
            )

            # ========================================================================
            # Phase 2: Meta-Strategy Selection
            # ========================================================================

            print("🧠 Phase 1: Meta-Strategy Selection")

            # Use meta-reasoning controller to select optimal strategy
            print("🔍 DEBUG: About to call meta_reasoning_controller...")
            print(
                f"🔍 DEBUG: Parameters - query: {query[:50]}..., context_type: {context_type}, time_budget: {time_budget}, quality_target: {quality_target}"
            )

            try:
                # Convert enum to string for API serialization
                context_type_str = context_type_enum.value
                print(
                    f"🔍 DEBUG: Converting context_type to string: {context_type_str}"
                )

                # Wrap app.call with StrategySelection schema
                try:
                    raw_strategy = await app.call(
                        "deepresearchagent.meta_reasoning_controller",
                        query=query,
                        context_type=context_type_str,
                        time_budget=time_budget,
                        quality_target=quality_target,
                    )
                    strategy_selection = StrategySelection(**raw_strategy)
                except Exception as schema_error:
                    print(
                        f"⚠️ Schema conversion failed for StrategySelection: {schema_error}"
                    )
                    # Create fallback strategy selection
                    strategy_selection = StrategySelection(
                        selected_strategy="comprehensive",
                        reasoning_type=ReasoningType.ANALYTICAL,
                        confidence=ConfidenceLevel.MEDIUM,
                        alternatives=["focused", "exploratory"],
                        selection_reason=f"Fallback due to schema error: {str(schema_error)}",
                    )

                print("✅ DEBUG: meta_reasoning_controller call completed successfully")

                print(f"✅ Selected strategy: {strategy_selection.selected_strategy}")
                print(f"🔍 Reasoning: {strategy_selection.selection_reason}")
            except Exception as e:
                print(f"❌ DEBUG: meta_reasoning_controller call failed: {str(e)}")
                print(f"❌ DEBUG: Exception type: {type(e).__name__}")
                raise e

            # ========================================================================
            # Phase 3: Query Analysis and Domain Strategy
            # ========================================================================

            print("🔍 Phase 2: Query Analysis and Domain Strategy")

            # Parallel execution of query analysis and domain strategy
            query_analysis_task = app.call(
                "deepresearchagent.query_analysis_reasoner",
                query=query,
                context_type=context_type,
                complexity=complexity,
            )

            # Wait for query analysis to complete before domain strategy
            try:
                raw_query_analysis = await query_analysis_task
                query_analysis = QueryAnalysis(**raw_query_analysis)
            except Exception as e:
                print(f"❌ Failed to parse query analysis: {e}")
                raise e

            # Create domain strategy based on analysis
            try:
                raw_domain_strategy = await app.call(
                    "deepresearchagent.domain_strategy_reasoner",
                    query_analysis=query_analysis.dict(),  # Convert Pydantic object to dict
                    time_budget=time_budget,
                    quality_target=quality_target,
                )
                domain_strategy = DomainStrategy(**raw_domain_strategy)
            except Exception as e:
                print(f"❌ Failed to create domain strategy: {e}")
                raise e

            print(f"✅ Query analyzed: {len(query_analysis.analysis)} chars")
            print(f"✅ Domain strategy: {domain_strategy.strategy_name}")

            # ========================================================================
            # Phase 4: Research Execution
            # ========================================================================

            print("🔬 Phase 3: Research Execution")

            # Execute research using the domain strategy
            try:
                raw_research_results = await app.call(
                    "deepresearchagent.research_execution_reasoner",
                    search_strategy=domain_strategy.dict(),  # Convert Pydantic object to dict
                    base_query=query,
                    max_results=20,
                )
                research_results = ResearchResults(**raw_research_results)
            except Exception as e:
                print(f"❌ Failed to execute research: {e}")
                raise e

            if research_results.success:
                print(
                    f"✅ Research executed: {research_results.final_results} sources found"
                )
            else:
                print(f"⚠️ Research execution had issues: {research_results.error}")

            # Update memory with findings
            await update_memory_state(
                app,
                {
                    "findings_count": research_results.final_results,
                    "time_elapsed": 10,  # Approximate time for phases 1-3
                },
            )

            # ========================================================================
            # Phase 5: Quality Control and Adaptation
            # ========================================================================

            print("🎯 Phase 4: Quality Control and Adaptation")

            # Assess research quality
            try:
                raw_quality_metrics = await app.call(
                    "deepresearchagent.quality_control_reasoner",
                    research_results=research_results.dict(),  # Convert Pydantic object to dict
                    original_query=query,
                    quality_target=quality_target,
                )
                quality_metrics = QualityMetrics(**raw_quality_metrics)
            except Exception as e:
                print(f"❌ Failed to assess quality metrics: {e}")
                raise e

            print(f"📊 Quality assessment: {quality_metrics.overall_quality}")
            print(f"🎯 Confidence score: {quality_metrics.confidence_score:.2f}")

            # Store quality metrics for tracking
            quality_history.append(
                {
                    "iteration": iteration_count,
                    "quality_score": quality_metrics.confidence_score,
                    "overall_quality": quality_metrics.overall_quality,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # ========================================================================
            # DHE Phase 3: Quality-Driven Recursive Loop Implementation
            # ========================================================================

            print(
                f"🔄 DHE Phase 3: Quality-driven recursive evaluation (iteration {iteration_count + 1})"
            )

            # Check if quality meets threshold and if we should iterate
            should_iterate = (
                quality_metrics.confidence_score < quality_threshold
                and iteration_count < max_iterations
                and research_results.success
                and research_results.final_results > 0
            )

            if should_iterate:
                print(
                    f"⚠️ Quality below threshold ({quality_metrics.confidence_score:.2f} < {quality_threshold})"
                )
                print(
                    f"🔄 Initiating recursive improvement loop (iteration {iteration_count + 1}/{max_iterations})"
                )

                # Get adaptive strategy for improvement
                try:
                    raw_adaptation_decision = await app.call(
                        "deepresearchagent.adaptive_reasoning_strategist",
                        current_context=research_context.dict(),
                        quality_metrics=quality_metrics.dict(),
                        time_remaining=max(0, time_budget - 15),
                        iteration_count=iteration_count,
                        quality_history=quality_history,
                    )
                    adaptation_decision = AdaptationDecision(**raw_adaptation_decision)
                except Exception as e:
                    print(f"❌ Failed to get adaptation decision: {e}")
                    adaptation_decision = AdaptationDecision(
                        trigger=AdaptationTrigger.QUALITY_THRESHOLD,
                        current_strategy="fallback",
                        new_strategy="fallback",
                        confidence=ConfidenceLevel.LOW,
                        reason=f"Error in adaptation analysis: {str(e)}",
                        expected_improvement="Unable to determine due to error",
                    )

                if (
                    adaptation_decision
                    and adaptation_decision.new_strategy
                    != adaptation_decision.current_strategy
                ):
                    print(
                        f"🔄 Adapting strategy: {adaptation_decision.current_strategy} → {adaptation_decision.new_strategy}"
                    )
                    print(f"� Reason: {adaptation_decision.reason}")

                    # Update strategy performance tracking
                    strategy_performance[adaptation_decision.current_strategy] = (
                        quality_metrics.confidence_score
                    )

                    # Create new domain strategy with adaptive insights
                    try:
                        enhanced_query_analysis = await app.ai(
                            system="""You are enhancing a query analysis based on quality feedback and adaptive strategy.
                            Improve the analysis to address quality gaps and incorporate new strategic insights.""",
                            user=f"""Enhance this query analysis for improved research quality:
                            
                            Original Query: {query}
                            Previous Analysis: {query_analysis.analysis}
                            Quality Issues: {quality_metrics.overall_quality}
                            New Strategy: {adaptation_decision.new_strategy}
                            Adaptation Reason: {adaptation_decision.reason}
                            
                            Provide enhanced analysis that addresses quality gaps and incorporates the new strategy.""",
                        )

                        # Update query analysis with enhanced version
                        query_analysis.analysis = enhanced_query_analysis

                        # Create improved domain strategy
                        raw_improved_strategy = await app.call(
                            "deepresearchagent.domain_strategy_reasoner",
                            query_analysis=query_analysis.dict(),
                            time_budget=max(10, time_budget - 20),
                            quality_target=quality_target,
                            previous_strategy=domain_strategy.dict(),
                            improvement_focus=adaptation_decision.reason,
                        )
                        improved_domain_strategy = DomainStrategy(
                            **raw_improved_strategy
                        )

                        print(
                            f"✅ Enhanced strategy: {improved_domain_strategy.strategy_name}"
                        )

                        # Re-execute research with improved strategy
                        raw_improved_results = await app.call(
                            "deepresearchagent.research_execution_reasoner",
                            search_strategy=improved_domain_strategy.dict(),
                            base_query=query,
                            max_results=25,  # Slightly more results for improvement
                            previous_results=research_results.dict(),
                            improvement_mode=True,
                        )
                        improved_research_results = ResearchResults(
                            **raw_improved_results
                        )

                        # Re-assess quality with improved results
                        raw_improved_quality = await app.call(
                            "deepresearchagent.quality_control_reasoner",
                            research_results=improved_research_results.dict(),
                            original_query=query,
                            quality_target=quality_target,
                            previous_quality=quality_metrics.dict(),
                        )
                        improved_quality_metrics = QualityMetrics(
                            **raw_improved_quality
                        )

                        print(
                            f"📈 Quality improvement: {quality_metrics.confidence_score:.2f} → {improved_quality_metrics.confidence_score:.2f}"
                        )

                        # Update results if improvement achieved
                        if (
                            improved_quality_metrics.confidence_score
                            > quality_metrics.confidence_score
                        ):
                            print("✅ Quality improved - using enhanced results")
                            research_results = improved_research_results
                            quality_metrics = improved_quality_metrics
                            domain_strategy = improved_domain_strategy

                            # Store successful strategy performance
                            strategy_performance[adaptation_decision.new_strategy] = (
                                improved_quality_metrics.confidence_score
                            )
                        else:
                            print("⚠️ No quality improvement - keeping original results")

                        iteration_count += 1

                        # Update memory with iteration progress
                        await update_memory_state(
                            app,
                            {
                                "iteration_count": iteration_count,
                                "quality_improvement": improved_quality_metrics.confidence_score
                                > quality_metrics.confidence_score,
                                "strategy_adapted": True,
                            },
                        )

                    except Exception as e:
                        print(f"❌ Recursive improvement failed: {e}")
                        print("📋 Continuing with original results")

            else:
                if quality_metrics.confidence_score >= quality_threshold:
                    print(
                        f"✅ Quality threshold met ({quality_metrics.confidence_score:.2f} >= {quality_threshold})"
                    )
                elif iteration_count >= max_iterations:
                    print(
                        f"⏰ Maximum iterations reached ({iteration_count}/{max_iterations})"
                    )
                else:
                    print("📋 Continuing with current results")

                # Final adaptation decision for metadata
                try:
                    raw_adaptation_decision = await app.call(
                        "deepresearchagent.adaptive_reasoning_strategist",
                        current_context=research_context.dict(),
                        quality_metrics=quality_metrics.dict(),
                        time_remaining=max(0, time_budget - 15),
                    )
                    adaptation_decision = AdaptationDecision(**raw_adaptation_decision)
                except Exception as e:
                    print(f"❌ Failed to get final adaptation decision: {e}")
                    adaptation_decision = AdaptationDecision(
                        trigger=AdaptationTrigger.QUALITY_THRESHOLD,
                        current_strategy="default",
                        new_strategy="default",
                        confidence=ConfidenceLevel.MEDIUM,
                        reason="Error in adaptation analysis",
                        expected_improvement="Unable to determine due to error",
                    )

            # ========================================================================
            # Phase 6: Synthesis and Validation
            # ========================================================================

            print("🔗 Phase 5: Synthesis and Validation")

            # Determine reasoning type for synthesis
            reasoning_type = ReasoningType.SYNTHESIS
            if "compare" in query.lower() or "versus" in query.lower():
                reasoning_type = ReasoningType.COMPARATIVE
            elif "cause" in query.lower() or "why" in query.lower():
                reasoning_type = ReasoningType.CAUSAL
            elif "predict" in query.lower() or "future" in query.lower():
                reasoning_type = ReasoningType.PREDICTIVE

            # Synthesize results
            try:
                raw_synthesis_data = await app.call(
                    "deepresearchagent.synthesis_reasoner",
                    research_results=research_results.dict(),  # Convert Pydantic object to dict
                    quality_metrics=quality_metrics.dict(),  # Convert Pydantic object to dict
                    original_query=query,
                    reasoning_type=reasoning_type,
                )
                synthesis_data = SynthesisData(**raw_synthesis_data)
            except Exception as e:
                print(f"❌ Failed to synthesize results: {e}")
                raise e

            print(
                f"✅ Synthesis completed: {len(synthesis_data.synthesis_response)} chars"
            )

            # Validate synthesis if high quality target
            validation_data = None
            if quality_target == "high" and research_results.success:
                try:
                    raw_validation_data = await app.call(
                        "deepresearchagent.research_validation_reasoner",
                        synthesis_data=synthesis_data.dict(),  # Convert Pydantic object to dict
                        original_sources=research_results.search_results,
                    )
                    validation_data = ValidationData(**raw_validation_data)
                    print("✅ Validation completed")
                except Exception as e:
                    print(f"❌ Failed to validate synthesis: {e}")
                    raise e

            # ========================================================================
            # DHE Phase 4: Meta-Learning and Strategy Performance Feedback
            # ========================================================================

            print("🧠 DHE Phase 4: Meta-Learning and Strategy Performance Feedback")

            # Store strategy performance for future learning
            final_strategy = strategy_selection.selected_strategy
            if adaptation_decision.new_strategy != adaptation_decision.current_strategy:
                final_strategy = adaptation_decision.new_strategy

            strategy_performance[final_strategy] = quality_metrics.confidence_score

            # Store performance data for meta-learning
            try:
                from .dynamic_infrastructure import store_strategy_performance

                await store_strategy_performance(
                    app, final_strategy, [quality_metrics.confidence_score]
                )
                print(
                    f"📊 Stored performance for strategy '{final_strategy}': {quality_metrics.confidence_score:.2f}"
                )
            except Exception as e:
                print(f"⚠️ Failed to store strategy performance: {e}")

            # Enhanced session results with DHE metadata
            session_results = {
                "strategy_used": strategy_selection.selected_strategy,
                "final_strategy": final_strategy,
                "sources_found": research_results.final_results,
                "quality_achieved": quality_metrics.overall_quality,
                "confidence_score": quality_metrics.confidence_score,
                "adaptation_triggered": adaptation_decision.new_strategy
                != adaptation_decision.current_strategy,
                "execution_success": research_results.success,
                "complexity_level": actual_complexity,
                "execution_strategy": execution_strategy,
                "iteration_count": iteration_count,
                "quality_history": quality_history,
                "strategy_performance": strategy_performance,
                "reasoners_utilized": estimated_reasoners,
            }

            # Generate enhanced learning insights with DHE context
            try:
                raw_learning_insight = await app.call(
                    "deepresearchagent.meta_learning_coordinator",
                    session_results=session_results,
                    performance_metrics=quality_metrics,
                    strategy_used=final_strategy,
                    complexity_analysis=complexity_data,
                    quality_progression=quality_history,
                )
                learning_insight = LearningInsight(**raw_learning_insight)
            except Exception as e:
                print(f"❌ Failed to generate learning insight: {e}")
                # Create fallback learning insight
                learning_insight = LearningInsight(
                    insight=f"DHE execution completed for {actual_complexity} query with {quality_metrics.confidence_score:.2f} quality score",
                    context=f"{actual_complexity} queries in {context_type_enum.value} context",
                    impact="medium",
                    confidence=ConfidenceLevel.MEDIUM,
                    evidence=f"Quality score: {quality_metrics.confidence_score:.2f}, Strategy: {final_strategy}, Iterations: {iteration_count}",
                    actionable_change=f"Consider {execution_strategy} execution for similar {actual_complexity} queries"
                )

            print(f"💡 Learning insight: {learning_insight.insight[:100]}...")

            # Store learning insights for future strategy selection
            try:
                from .dynamic_infrastructure import store_learning_insights

                await store_learning_insights(app, [learning_insight])
                print("✅ Learning insights stored for future strategy optimization")
            except Exception as e:
                print(f"⚠️ Failed to store learning insights: {e}")

            # ========================================================================
            # Phase 8: Final Results Compilation
            # ========================================================================

            # Update final memory state
            await update_memory_state(
                app,
                {
                    "progress_percent": 100,
                    "quality_score": quality_metrics.confidence_score,
                    "time_elapsed": min(time_budget, 25),  # Estimate total time
                },
            )

            # Compile comprehensive results
            orchestration_result = {
                "workflow_id": workflow_id,
                "session_id": session_id,
                "query": query,
                "context_type": context_type_enum.value,
                "execution_summary": {
                    "strategy_selected": strategy_selection.selected_strategy,
                    "strategy_confidence": strategy_selection.confidence.value,
                    "sources_found": research_results.final_results,
                    "quality_achieved": quality_metrics.overall_quality,
                    "confidence_score": quality_metrics.confidence_score,
                    "adaptation_recommended": adaptation_decision.new_strategy
                    != adaptation_decision.current_strategy,
                    "execution_success": research_results.success,
                },
                "research_results": {
                    "query_analysis": query_analysis.model_dump(mode="json") if hasattr(query_analysis, 'model_dump') else query_analysis,
                    "domain_strategy": domain_strategy.model_dump(mode="json"),
                    "search_results": research_results.model_dump(mode="json") if hasattr(research_results, 'model_dump') else research_results,
                    "quality_metrics": quality_metrics.model_dump(mode="json"),
                    "synthesis": synthesis_data.model_dump(mode="json") if hasattr(synthesis_data, 'model_dump') else synthesis_data,
                    "validation": (
                        validation_data.model_dump(mode="json")
                        if validation_data
                        else None
                    ),
                },
                "learning_insights": {
                    "session_insight": learning_insight.model_dump(mode="json"),
                    "adaptation_decision": adaptation_decision.model_dump(mode="json"),
                },
                "metadata": {
                    "workflow_start": memory_state.last_updated,
                    "workflow_end": datetime.now().isoformat(),
                    "total_time_minutes": min(time_budget, 25),
                    "reasoning_type": reasoning_type.value,
                },
                "dhe_metadata": {
                    "execution_path": "enhanced_linear",
                    "complexity_analysis": complexity_data,
                    "actual_complexity": actual_complexity,
                    "execution_strategy": execution_strategy,
                    "estimated_reasoners": estimated_reasoners,
                    "iteration_count": iteration_count,
                    "quality_history": quality_history,
                    "strategy_performance": strategy_performance,
                    "quality_threshold": quality_threshold,
                    "max_iterations": max_iterations,
                    "final_strategy": final_strategy,
                    "dhe_features_used": [
                        "complexity_based_branching",
                        "quality_driven_loops",
                        "meta_learning_feedback",
                        "adaptive_strategy_selection"
                    ]
                },
            }

            print(f"🎉 Research workflow completed successfully!")
            print(f"📊 Final quality score: {quality_metrics.confidence_score:.2f}")
            print(f"🔍 Sources analyzed: {research_results.final_results}")

            return orchestration_result

        except Exception as e:
            print(f"❌ Research workflow failed: {str(e)}")

            # Return error result
            error_result = {
                "workflow_id": workflow_id,
                "session_id": session_id,
                "query": query,
                "context_type": "factual",  # Default context type for error cases
                "error": str(e),
                "execution_summary": {
                    "strategy_selected": None,
                    "strategy_confidence": None,
                    "sources_found": 0,
                    "quality_achieved": "poor",
                    "confidence_score": 0.0,
                    "adaptation_recommended": False,
                    "execution_success": False,
                },
                "research_results": {
                    "query_analysis": {},
                    "domain_strategy": {},
                    "search_results": {
                        "search_results": [],
                        "total_results": 0,
                        "query_used": query
                    },
                    "quality_metrics": {},
                    "synthesis": {
                        "synthesis_response": "",
                        "key_findings": [],
                        "confidence_level": "low",
                        "sources_used": 0
                    },
                    "validation": None,
                },
                "learning_insights": {},
                "metadata": {
                    "workflow_start": datetime.now().isoformat(),
                    "workflow_end": datetime.now().isoformat(),
                    "total_time_minutes": 0,
                    "reasoning_type": "analytical",
                },
            }

            return error_result

    # ============================================================================
    # Parallel Research Orchestrator - For complex multi-faceted queries
    # ============================================================================

    @app.reasoner()
    async def parallel_research_orchestrator(
        query: str,
        sub_queries: List[str],
        context_type: Union[str, ContextType] = "factual",
        time_budget: int = 45,
        quality_target: str = "high",
    ) -> Dict[str, Any]:
        """
        Orchestrates parallel research execution for complex queries with multiple facets.
        Coordinates multiple research workflows and synthesizes results.
        """
        workflow_id = f"parallel_{uuid.uuid4().hex[:8]}"

        print(f"🚀 Starting parallel research workflow: {workflow_id}")
        print(f"📋 Main query: {query}")
        print(f"🔍 Sub-queries: {len(sub_queries)}")

        try:
            # Calculate time budget per sub-query
            time_per_query = max(15, time_budget // len(sub_queries))

            # Create parallel research tasks
            research_tasks = []
            for i, sub_query in enumerate(sub_queries):
                task = app.call(
                    "deepresearchagent.dynamic_research_orchestrator",
                    query=sub_query,
                    context_type=context_type,
                    complexity="moderate",
                    time_budget=time_per_query,
                    quality_target=quality_target,
                )
                research_tasks.append(task)

            # Execute all research tasks in parallel
            parallel_results = await asyncio.gather(
                *research_tasks, return_exceptions=True
            )

            # Process results and handle exceptions
            successful_results = []
            failed_results = []

            for i, result in enumerate(parallel_results):
                if isinstance(result, Exception):
                    failed_results.append(
                        {"sub_query": sub_queries[i], "error": str(result)}
                    )
                    print(f"❌ Sub-query {i+1} failed: {result}")
                else:
                    successful_results.append(result)
                    print(f"✅ Sub-query {i+1} completed")

            # Synthesize parallel results
            if successful_results:
                # Combine all synthesis responses
                combined_syntheses = []
                total_sources = 0
                avg_confidence = 0.0

                for result in successful_results:
                    synthesis = result.get("research_results", {}).get("synthesis", {})
                    if synthesis:
                        combined_syntheses.append(
                            synthesis.get("synthesis_response", "")
                        )

                    total_sources += result.get("execution_summary", {}).get(
                        "sources_found", 0
                    )
                    avg_confidence += result.get("execution_summary", {}).get(
                        "confidence_score", 0.0
                    )

                avg_confidence = (
                    avg_confidence / len(successful_results)
                    if successful_results
                    else 0.0
                )

                # Create final synthesis
                final_synthesis = await app.ai(
                    system="""You are synthesizing results from parallel research workflows.
                    Combine the individual syntheses into a comprehensive response that addresses
                    the main research query while integrating insights from all sub-queries.""",
                    user=f"""Synthesize these parallel research results into a comprehensive response:
                    
                    Main Query: {query}
                    
                    Individual Syntheses:
                    {chr(10).join([f"Sub-research {i+1}: {synthesis}" for i, synthesis in enumerate(combined_syntheses)])}
                    
                    Create a unified synthesis that addresses the main query comprehensively.""",
                )

                parallel_result = {
                    "workflow_id": workflow_id,
                    "main_query": query,
                    "sub_queries": sub_queries,
                    "execution_summary": {
                        "successful_sub_queries": len(successful_results),
                        "failed_sub_queries": len(failed_results),
                        "total_sources_found": total_sources,
                        "average_confidence": avg_confidence,
                        "execution_success": len(successful_results) > 0,
                    },
                    "parallel_results": successful_results,
                    "failed_results": failed_results,
                    "final_synthesis": final_synthesis,
                    "metadata": {
                        "workflow_type": "parallel",
                        "workflow_end": datetime.now().isoformat(),
                        "total_time_budget": time_budget,
                    },
                }

                print(f"🎉 Parallel research completed!")
                print(
                    f"✅ Successful sub-queries: {len(successful_results)}/{len(sub_queries)}"
                )
                print(f"📊 Total sources: {total_sources}")

                return parallel_result

            else:
                print("❌ All parallel research tasks failed")
                return {
                    "workflow_id": workflow_id,
                    "main_query": query,
                    "error": "All parallel research tasks failed",
                    "execution_summary": {"execution_success": False},
                    "failed_results": failed_results,
                }

        except Exception as e:
            print(f"❌ Parallel research orchestration failed: {str(e)}")
            return {
                "workflow_id": workflow_id,
                "main_query": query,
                "error": str(e),
                "execution_summary": {"execution_success": False},
            }

    # ============================================================================
    # Adaptive Research Monitor - Real-time monitoring and adjustment
    # ============================================================================

    @app.reasoner()
    async def adaptive_research_monitor(
        workflow_id: str, monitoring_interval: int = 5
    ) -> Dict[str, Any]:
        """
        Monitors research workflows in real-time and makes adaptive adjustments
        based on progress and quality metrics.
        """
        print(f"👁️ Starting adaptive monitoring for workflow: {workflow_id}")

        monitoring_data = {
            "workflow_id": workflow_id,
            "monitoring_start": datetime.now().isoformat(),
            "adjustments_made": [],
            "quality_progression": [],
            "monitoring_active": True,
        }

        try:
            while monitoring_data["monitoring_active"]:
                # Get current memory state
                memory_state = await get_memory_state(app)

                if memory_state and memory_state.workflow_id == workflow_id:
                    # Record quality progression
                    monitoring_data["quality_progression"].append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "quality_score": memory_state.quality_score,
                            "findings_count": memory_state.findings_count,
                            "progress_percent": getattr(
                                memory_state, "progress_percent", 0
                            ),
                        }
                    )

                    # Check if workflow is complete
                    if getattr(memory_state, "progress_percent", 0) >= 100:
                        monitoring_data["monitoring_active"] = False
                        print(f"✅ Workflow {workflow_id} completed - stopping monitor")
                        break

                    # Check for quality issues
                    if (
                        memory_state.quality_score < 0.5
                        and memory_state.findings_count > 0
                        and memory_state.time_elapsed > 10
                    ):

                        adjustment = {
                            "timestamp": datetime.now().isoformat(),
                            "trigger": "low_quality",
                            "action": "quality_boost_recommended",
                            "details": f"Quality score {memory_state.quality_score:.2f} below threshold",
                        }
                        monitoring_data["adjustments_made"].append(adjustment)
                        print(f"⚠️ Quality adjustment recommended for {workflow_id}")

                # Wait for next monitoring cycle
                await asyncio.sleep(monitoring_interval)

        except Exception as e:
            print(f"❌ Monitoring error for {workflow_id}: {str(e)}")
            monitoring_data["error"] = str(e)

        monitoring_data["monitoring_end"] = datetime.now().isoformat()
        return monitoring_data

    # Reasoners are already registered with @app.reasoner() decorators
    # No need to return anything from this setup function
    print("✅ All research orchestrator reasoners registered successfully")
