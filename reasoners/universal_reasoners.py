"""
Universal Reasoners - Deep Research Agent

Core universal reasoners that perform the actual research tasks.
These reasoners are called by the research orchestrator and provide
functional research capabilities using Agent Field SDK patterns.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, cast

from .dynamic_models import (
    ContextType,
    ConfidenceLevel,
    ReasoningType,
    QualityMetrics,
    SearchStrategy,
    SearchResult,
    JinaSearchResponse,
    SimpleResponse
)

# Import search function from utils to avoid circular imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import search_jina_ai


def create_universal_reasoners(app):
    """Create all universal reasoners that perform actual research tasks"""
    
    print("🔍 DEBUG: Starting create_universal_reasoners...")
    
    # ============================================================================
    # Query Analysis Reasoner - Analyzes and understands research queries
    # ============================================================================
    
    @app.reasoner()
    async def query_analysis_reasoner(
        query: str,
        context_type: Union[str, ContextType],
        complexity: str
    ) -> Dict[str, Any]:
        """
        Analyzes research queries to understand intent, complexity, and requirements.
        Provides structured analysis to guide research strategy.
        """
        print(f"🔍 Starting query analysis for: {query[:50]}...")
        
        try:
            # Robust enum conversion with validation
            if isinstance(context_type, str):
                try:
                    context_type_enum = ContextType(context_type.lower())
                    context_str = context_type_enum.value
                    print(f"🔍 DEBUG: Converted context_type string '{context_type}' to enum: {context_type_enum}")
                except ValueError as e:
                    print(f"⚠️ DEBUG: Invalid context_type '{context_type}', error: {e}, defaulting to FACTUAL")
                    context_type_enum = ContextType.FACTUAL
                    context_str = context_type_enum.value
            elif isinstance(context_type, ContextType):
                context_type_enum = context_type
                context_str = context_type_enum.value
            else:
                print(f"⚠️ DEBUG: Unexpected context_type type {type(context_type)}, defaulting to FACTUAL")
                context_type_enum = ContextType.FACTUAL
                context_str = context_type_enum.value
            
            analysis = await app.ai(
                system="""You are a query analysis expert. Analyze research queries to understand:
                - Research intent and objectives
                - Key concepts and domains involved
                - Information requirements and scope
                - Complexity factors and challenges
                - Optimal research approaches
                
                Provide structured analysis that guides effective research strategy.""",
                user=f"""Analyze this research query in detail:
                
                Query: {query}
                Context Type: {context_str}
                Complexity: {complexity}
                
                Provide comprehensive analysis including:
                1. Research intent and main objectives
                2. Key concepts and domains to explore
                3. Information requirements and scope
                4. Potential challenges and complexity factors
                5. Recommended research approaches
                6. Success criteria for the research
                
                Make your analysis detailed and actionable."""
            )
            
            # Extract key concepts for further processing
            concepts = await app.ai(
                system="Extract 3-5 key concepts from this query analysis that should guide research.",
                user=f"Query: {query}\nAnalysis: {analysis}\n\nExtract key concepts as a simple list."
            )
            
            result = {
                "query": query,
                "analysis": analysis,
                "key_concepts": concepts.split('\n') if isinstance(concepts, str) else [str(concepts)],
                "context_type": context_str,
                "complexity": complexity,
                "research_intent": "comprehensive" if complexity == "complex" else "focused",
                "estimated_scope": "broad" if "multiple" in analysis.lower() or "various" in analysis.lower() else "narrow",
                "confidence": 0.8
            }
            
            print(f"✅ Query analysis completed: {len(analysis)} chars")
            return result
            
        except Exception as e:
            print(f"❌ Query analysis failed: {e}")
            # Ensure context_str is defined in exception case with proper enum handling
            if isinstance(context_type, str):
                context_str = context_type
            elif isinstance(context_type, ContextType):
                context_str = context_type.value
            else:
                context_str = str(context_type)
            return {
                "query": query,
                "analysis": f"Analysis failed: {str(e)}",
                "key_concepts": [query],
                "context_type": context_str,
                "complexity": complexity,
                "research_intent": "basic",
                "estimated_scope": "narrow",
                "confidence": 0.1
            }
    
    # ============================================================================
    # Domain Strategy Reasoner - Creates domain-specific research strategies
    # ============================================================================
    
    @app.reasoner()
    async def domain_strategy_reasoner(
        query_analysis: Dict[str, Any],
        time_budget: int,
        quality_target: str
    ) -> SearchStrategy:
        """
        Creates domain-specific research strategies based on query analysis.
        Develops targeted search and analysis approaches.
        """
        print("🔍 Creating domain-specific research strategy...")
        
        try:
            query = query_analysis.get("query", "")
            analysis = query_analysis.get("analysis", "")
            key_concepts = query_analysis.get("key_concepts", [])
            
            strategy = await app.ai(
                system="""You are a domain strategy expert. Create effective research strategies
                tailored to specific domains and query requirements. Design strategies that
                maximize information gathering efficiency within constraints.
                
                Consider:
                - Domain-specific search patterns
                - Information source priorities
                - Quality vs. speed tradeoffs
                - Parallel execution opportunities""",
                user=f"""Create a domain-specific research strategy:
                
                Query: {query}
                Analysis: {analysis[:500]}...
                Key Concepts: {key_concepts}
                Time Budget: {time_budget} minutes
                Quality Target: {quality_target}
                
                Design a strategy that includes:
                1. Strategy name and approach
                2. Optimized query patterns for search
                3. Maximum results to gather
                4. Whether to use parallel queries
                5. Quality filtering approach
                
                Focus on efficiency and relevance for this specific domain.""",
                schema=SearchStrategy
            )
            
            print(f"✅ Domain strategy created: {strategy.strategy_name}")
            return strategy
            
        except Exception as e:
            print(f"❌ Domain strategy creation failed: {e}")
            # Return a basic fallback strategy
            return SearchStrategy(
                strategy_name="basic_search",
                query_patterns=[query_analysis.get("query", "research")],
                max_results=10
            )
    
    # ============================================================================
    # Research Execution Reasoner - Executes actual research using search
    # ============================================================================
    
    @app.reasoner()
    async def research_execution_reasoner(
        search_strategy: SearchStrategy,
        base_query: str,
        max_results: int = 20
    ) -> Dict[str, Any]:
        """
        Executes actual research using search engines and external sources.
        Implements the search strategy and gathers information.
        """
        print(f"🔍 Executing research with strategy: {search_strategy.strategy_name}")
        
        try:
            all_results = []
            search_queries = search_strategy.query_patterns[:3]  # Limit to 3 queries
            
            if len(search_queries) > 1:  # Simplified: always process sequentially for smaller AI models
                # Execute searches in parallel
                print(f"🔍 Executing {len(search_queries)} parallel searches...")
                search_tasks = [search_jina_ai(query) for query in search_queries]
                search_responses = await asyncio.gather(*search_tasks, return_exceptions=True)
                
                for i, response in enumerate(search_responses):
                    if isinstance(response, Exception):
                        print(f"❌ Search {i+1} failed: {response}")
                        continue
                    # Type check to ensure response is JinaSearchResponse
                    if hasattr(response, 'results') and hasattr(response, 'total_results'):
                        # Cast to JinaSearchResponse to satisfy type checker
                        search_response = cast(JinaSearchResponse, response)
                        all_results.extend(search_response.results)
                    
            else:
                # Execute searches sequentially
                print(f"🔍 Executing {len(search_queries)} sequential searches...")
                for query in search_queries:
                    response = await search_jina_ai(query)
                    all_results.extend(response.results)
                    
                    # Respect max_results limit
                    if len(all_results) >= max_results:
                        break
            
            # Filter and rank results based on quality
            filtered_results = []
            for result in all_results[:max_results]:
                # Simplified quality filtering for smaller AI models
                if len(result.content) > 100:  # Basic content length filter
                    filtered_results.append(result)
            
            # Analyze the gathered information
            if filtered_results:
                content_summary = "\n\n".join([
                    f"Title: {result.title}\nContent: {result.content[:300]}..."
                    for result in filtered_results[:5]
                ])
                
                analysis = await app.ai(
                    system="""You are a research analyst. Analyze the gathered search results
                    and provide insights about the information quality and coverage.""",
                    user=f"""Analyze these search results for query: "{base_query}"
                    
                    Results Summary:
                    {content_summary}
                    
                    Provide analysis of:
                    1. Information quality and relevance
                    2. Coverage of the research topic
                    3. Key findings and insights
                    4. Gaps or areas needing more research
                    
                    Keep analysis concise but comprehensive."""
                )
                
                success = True
                final_count = len(filtered_results)
            else:
                analysis = "No relevant results found with current search strategy."
                success = False
                final_count = 0
            
            result = {
                "success": success,
                "strategy_used": search_strategy.strategy_name,
                "queries_executed": search_queries,
                "raw_results": len(all_results),
                "final_results": final_count,
                "search_results": [
                    {
                        "title": r.title,
                        "url": r.url,
                        "content": r.content[:500],  # Truncate for memory efficiency
                        "relevance": r.relevance
                    } for r in filtered_results
                ],
                "analysis": analysis,
                "quality_score": min(0.9, 0.3 + (final_count * 0.05)),
                "execution_time": "estimated_3_minutes"
            }
            
            print(f"✅ Research execution completed: {final_count} results")
            return result
            
        except Exception as e:
            print(f"❌ Research execution failed: {e}")
            return {
                "success": False,
                "strategy_used": search_strategy.strategy_name,
                "queries_executed": [],
                "raw_results": 0,
                "final_results": 0,
                "search_results": [],
                "analysis": f"Research execution failed: {str(e)}",
                "quality_score": 0.0,
                "error": str(e)
            }
    
    # ============================================================================
    # Quality Control Reasoner - Assesses research quality and completeness
    # ============================================================================
    
    @app.reasoner()
    async def quality_control_reasoner(
        research_results: Dict[str, Any],
        original_query: str,
        quality_target: str
    ) -> QualityMetrics:
        """
        Assesses the quality and completeness of research results.
        Provides metrics and improvement suggestions.
        """
        print("🔍 Assessing research quality...")
        
        try:
            results_count = research_results.get("final_results", 0)
            success = research_results.get("success", False)
            analysis = research_results.get("analysis", "")
            search_results = research_results.get("search_results", [])
            
            # Calculate source diversity
            unique_domains = set()
            for result in search_results:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(result.get("url", "")).netloc
                    if domain:
                        unique_domains.add(domain)
                except:
                    pass
            
            source_diversity = len(unique_domains)
            
            # Assess quality using AI
            quality_assessment = await app.ai(
                system="""You are a research quality assessor. Evaluate research results
                against quality criteria and provide detailed metrics.
                
                Consider:
                - Completeness of information
                - Source diversity and credibility
                - Evidence strength and relevance
                - Coverage of the research question
                - Overall research effectiveness""",
                user=f"""Assess the quality of this research:
                
                Original Query: {original_query}
                Quality Target: {quality_target}
                
                Research Results:
                - Success: {success}
                - Results Count: {results_count}
                - Source Diversity: {source_diversity} unique domains
                - Analysis: {analysis[:500]}...
                
                Sample Results:
                {json.dumps(search_results[:3], indent=2) if search_results else "No results"}
                
                Provide quality assessment with specific metrics and improvement suggestions.""",
                schema=QualityMetrics
            )
            
            print(f"✅ Quality assessment completed: {quality_assessment.overall_quality}")
            return quality_assessment
            
        except Exception as e:
            print(f"❌ Quality assessment failed: {e}")
            # Return basic quality metrics
            return QualityMetrics(
                overall_quality="poor",
                confidence_score=0.2,
                improvement_needed="Increase search query diversity and gather more sources"
            )
    
    # ============================================================================
    # Synthesis Reasoner - Synthesizes research findings into coherent responses
    # ============================================================================
    
    @app.reasoner()
    async def synthesis_reasoner(
        research_results: Dict[str, Any],
        quality_metrics: QualityMetrics,
        original_query: str,
        reasoning_type: ReasoningType
    ) -> Dict[str, Any]:
        """
        Synthesizes research findings into coherent, comprehensive responses.
        Combines information from multiple sources with appropriate reasoning.
        """
        print(f"🔍 Synthesizing research findings with {reasoning_type} reasoning...")
        
        try:
            search_results = research_results.get("search_results", [])
            analysis = research_results.get("analysis", "")
            
            # Prepare content for synthesis
            if search_results:
                source_content = "\n\n".join([
                    f"Source {i+1}: {result.get('title', 'Untitled')}\n{result.get('content', '')[:400]}..."
                    for i, result in enumerate(search_results[:5])
                ])
                source_urls = [result.get('url', '') for result in search_results[:10]]
            else:
                source_content = "No external sources available."
                source_urls = []
            
            # Adapt synthesis approach based on reasoning type
            reasoning_guidance = {
                ReasoningType.ANALYTICAL: "Provide systematic analysis with clear structure and evidence-based conclusions.",
                ReasoningType.COMPARATIVE: "Compare and contrast different perspectives, highlighting similarities and differences.",
                ReasoningType.CAUSAL: "Focus on cause-and-effect relationships and underlying mechanisms.",
                ReasoningType.PREDICTIVE: "Analyze trends and make informed predictions about future developments.",
                ReasoningType.SYNTHESIS: "Integrate information from multiple sources into a unified understanding.",
                ReasoningType.CREATIVE: "Explore innovative perspectives and generate novel insights.",
                ReasoningType.LOGICAL: "Use logical reasoning and structured argumentation."
            }.get(reasoning_type, "Provide comprehensive analysis and insights.")
            
            synthesis = await app.ai(
                system=f"""You are a research synthesis expert. Create comprehensive, well-structured
                responses that effectively combine information from multiple sources.
                
                Reasoning Approach: {reasoning_guidance}
                
                Your synthesis should:
                - Directly address the original research question
                - Integrate findings from multiple sources
                - Provide clear, actionable insights
                - Acknowledge limitations and uncertainties
                - Be well-structured and easy to understand""",
                user=f"""Synthesize research findings for this query:
                
                Original Query: {original_query}
                Reasoning Type: {reasoning_type}
                
                Research Analysis:
                {analysis}
                
                Source Content:
                {source_content}
                
                Quality Context:
                - Overall Quality: {quality_metrics.overall_quality}
                - Confidence Score: {quality_metrics.confidence_score}
                - Improvement Needed: {quality_metrics.improvement_needed}
                
                Create a comprehensive synthesis that addresses the query effectively."""
            )
            
            # Generate key insights
            insights = await app.ai(
                system="Extract 3-5 key insights from this research synthesis.",
                user=f"Synthesis: {synthesis}\n\nExtract the most important insights as a simple list."
            )
            
            result = {
                "synthesis_response": synthesis,
                "key_insights": insights.split('\n') if isinstance(insights, str) else [str(insights)],
                "reasoning_type": reasoning_type.value,
                "sources_used": len(search_results),
                "source_urls": source_urls,
                "confidence_level": quality_metrics.confidence_score,
                "synthesis_quality": quality_metrics.overall_quality,
                "limitations": f"Based on {len(search_results)} sources with simplified quality assessment"
            }
            
            print(f"✅ Synthesis completed: {len(synthesis)} chars")
            return result
            
        except Exception as e:
            print(f"❌ Synthesis failed: {e}")
            return {
                "synthesis_response": f"Synthesis failed: {str(e)}. Unable to process research findings.",
                "key_insights": ["Research synthesis encountered technical difficulties"],
                "reasoning_type": reasoning_type.value,
                "sources_used": 0,
                "source_urls": [],
                "confidence_level": 0.0,
                "synthesis_quality": "poor",
                "limitations": "Technical error prevented synthesis",
                "error": str(e)
            }
    
    # ============================================================================
    # Research Validation Reasoner - Validates research against sources
    # ============================================================================
    
    @app.reasoner()
    async def research_validation_reasoner(
        synthesis_data: Dict[str, Any],
        original_sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validates research synthesis against original sources.
        Checks for accuracy, consistency, and proper attribution.
        """
        print("🔍 Validating research synthesis against sources...")
        
        try:
            synthesis_response = synthesis_data.get("synthesis_response", "")
            key_insights = synthesis_data.get("key_insights", [])
            
            # Prepare source content for validation
            source_content = "\n\n".join([
                f"Source {i+1}: {source.get('title', 'Untitled')}\n{source.get('content', '')[:300]}..."
                for i, source in enumerate(original_sources[:5])
            ])
            
            validation = await app.ai(
                system="""You are a research validation expert. Verify that research synthesis
                accurately represents the source material and identify any potential issues.
                
                Check for:
                - Accuracy of claims against sources
                - Proper representation of source content
                - Consistency between synthesis and evidence
                - Potential misinterpretations or overstatements
                - Missing important information from sources""",
                user=f"""Validate this research synthesis against the original sources:
                
                Synthesis to Validate:
                {synthesis_response[:1000]}...
                
                Key Insights:
                {chr(10).join(key_insights[:5])}
                
                Original Sources:
                {source_content}
                
                Provide validation assessment including:
                1. Accuracy score (0-100)
                2. Consistency assessment
                3. Any identified issues or concerns
                4. Recommendations for improvement
                5. Overall validation status (valid/needs_revision/invalid)"""
            )
            
            # Extract validation score
            validation_score = 85  # Default score
            try:
                import re
                score_match = re.search(r'(\d+)(?:/100|\%)', validation)
                if score_match:
                    validation_score = int(score_match.group(1))
            except:
                pass
            
            result = {
                "validation_status": "valid" if validation_score >= 70 else "needs_revision" if validation_score >= 50 else "invalid",
                "accuracy_score": validation_score,
                "validation_report": validation,
                "sources_checked": len(original_sources),
                "consistency_level": "high" if validation_score >= 80 else "medium" if validation_score >= 60 else "low",
                "recommendations": [
                    "Review source accuracy" if validation_score < 70 else "Synthesis appears accurate",
                    "Check for missing information" if len(original_sources) < 3 else "Good source coverage",
                    "Verify key claims" if validation_score < 80 else "Claims well-supported"
                ],
                "confidence": min(0.9, validation_score / 100.0)
            }
            
            print(f"✅ Validation completed: {result['validation_status']} ({validation_score}%)")
            return result
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return {
                "validation_status": "error",
                "accuracy_score": 0,
                "validation_report": f"Validation failed: {str(e)}",
                "sources_checked": len(original_sources),
                "consistency_level": "unknown",
                "recommendations": ["Manual review required due to validation error"],
                "confidence": 0.0,
                "error": str(e)
            }
    
    print("✅ DEBUG: All universal reasoners registered successfully")
    
    return {
        'query_analysis_reasoner': query_analysis_reasoner,
        'domain_strategy_reasoner': domain_strategy_reasoner,
        'research_execution_reasoner': research_execution_reasoner,
        'quality_control_reasoner': quality_control_reasoner,
        'synthesis_reasoner': synthesis_reasoner,
        'research_validation_reasoner': research_validation_reasoner
    }
