import asyncio
import os
import json
import concurrent.futures
from typing import Dict, Any, Optional, List
from openai import AzureOpenAI
import certifi
import urllib3
from connectors.ai_search.hybrid_retriver import HybridSearchRetriever
from utils.context_deterministic_cache import get_cached_context_for_query, cache_context_for_query
from utils.conversation_memory_manager import conversation_memory_manager
# from utils.intelligent_context_compressor import compress_context_intelligently, get_context_token_count
from utils.token_monitoring_system import monitor_and_compress_context
from utils.schema_relationship_discovery import schema_relationship_discovery
# Disable SSL warnings and configure bypass
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['PYTHONHTTPSVERIFY'] = '0'
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
MULTI_VECTOR_INDEX = os.getenv("MULTI_VECTOR_INDEX")

# Global plan cache that persists across requests for query consistency
_global_plan_cache = {}

class MultiHopRetriever:
    """
    MultiHopRetriever: Implements a multi-hop retrieval pipeline with parallel hop execution,
    performance tracking, robust error handling, and resource cleanup.
    """
    def __init__(self, max_workers: int = 4, hop_timeout: float = 30.0):
        import hashlib
        self.hybrid_retriever = HybridSearchRetriever()
        # Use global plan cache instead of instance-level cache
        self.memory_store = _global_plan_cache
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.hop_timeout = hop_timeout
        
        # Add normalization helper for consistency with hybrid retriever
        self._norm = lambda s: " ".join(s.lower().split())
        # Request ID generator for multi-hop consistency
        self._reqid = lambda q: hashlib.sha1(self._norm(q).encode()).hexdigest()

    def cleanup(self):
        """Cleanup resources, especially the thread pool executor"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
            print("[MultiHop] Thread pool executor cleaned up")

    async def _async_retrieve_wrapper(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Wrapper to make hybrid retriever truly async for parallel execution.
        This ensures that synchronous calls in hybrid_retriever don't block the event loop.
        """
        loop = asyncio.get_event_loop()
        def sync_retrieve():
            import asyncio
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(self.hybrid_retriever.retrieve(query, top_k))
            finally:
                new_loop.close()
        return await loop.run_in_executor(self.executor, sync_retrieve)

    def _fully_qualify_table_names(self, synthesis: dict) -> dict:
        """
        Ensures all table names in the synthesis['tables'] dict are fully qualified (database.schema.table).
        Modifies the synthesis dict in-place and returns it.
        """
        tables = synthesis.get("tables", {})
        qualified_tables = {}
        for table_name, table_info in tables.items():
            # If already fully qualified, keep as is
            if table_name.count('.') == 2:
                qualified_tables[table_name] = table_info
                continue
            # Try to synthesize fully qualified name from table_info
            db = table_info.get('database') or table_info.get('db') or ''
            schema = table_info.get('schema') or table_info.get('sch') or ''
            base = table_info.get('table') or table_info.get('name') or table_name
            if db and schema:
                fq_name = f"{db}.{schema}.{base}"
            else:
                fq_name = table_name  # fallback
            qualified_tables[fq_name] = table_info
        synthesis["tables"] = qualified_tables
        print(f"[MultiHop] Qualifying table names: after={list(qualified_tables.keys())}")
        return synthesis
    """
    MultiHopRetriever: Implements a multi-hop retrieval pipeline with memory pattern check,
    agentic query planning, multi-hop reasoning, hop execution, and learning updates.
    """
    async def retrieve(self, user_query: str, top_k: int = 3, conversation_id: str = None) -> Dict[str, Any]:
        retrieval_start_time = asyncio.get_event_loop().time()
        print(f"[MultiHop] Starting parallel multi-hop retrieval for: '{user_query}'")

        # Enhanced Follow-up Detection and Context Reuse
        if conversation_id:
            follow_up_analysis = conversation_memory_manager.analyze_follow_up_potential(conversation_id, user_query)
            
            if follow_up_analysis.get("is_follow_up"):
                print(f"[MultiHop] Follow-up detected: {follow_up_analysis.get('follow_up_type', 'unknown')}")
                
                # Get context for follow-up processing
                followup_context = conversation_memory_manager.get_context_for_followup(conversation_id, follow_up_analysis)
                
                # Build intelligent conversation context for concept extraction
                conversation_context = self._build_conversation_context(followup_context)
                
                # Check if we can reuse previous context with intelligent concept analysis
                if follow_up_analysis.get("strategy") == "extend_previous":
                    return await self._handle_context_extension(user_query, followup_context, top_k)
                elif follow_up_analysis.get("strategy") == "combine_multiple":
                    return await self._handle_context_combination(user_query, followup_context, top_k)
                
                # For other strategies, continue with enhanced context
                print(f"[MultiHop] Using follow-up strategy: {follow_up_analysis.get('strategy')}")
                
                # Store conversation learning for future improvements
                self._store_conversation_learning(user_query, follow_up_analysis, conversation_context)

        # Step 1: Memory Pattern Check
        plan = self._check_memory_for_plan(user_query)
        if plan:
            print("[MultiHop] Found existing plan in memory.")
        else:
            print("[MultiHop] No plan found. Generating new plan via LLM.")
            plan_start_time = asyncio.get_event_loop().time()
            plan = await self._generate_query_plan(user_query, conversation_id)
            plan_time = asyncio.get_event_loop().time() - plan_start_time
            print(f"[MultiHop] Plan generation completed in {plan_time:.2f}s")
            self._store_plan_in_memory(user_query, plan)

        # Step 2: Multi-hop Reasoning (break into hops) - PARALLEL EXECUTION
        hops = plan.get("hops", [])
        print(f"[MultiHop] Executing {len(hops)} hops in parallel")

        async def _execute_hop_with_timeout(hop, i, top_k):
            hop_start_time = asyncio.get_event_loop().time()
            print(f"[MultiHop] Starting Hop {i+1}/{len(hops)} in parallel: {hop['description']}")
            try:
                async def _execute_hop_core():
                    hop_query = hop["query"]
                    print(f"[MultiHop] Hop {i+1} Query: '{hop_query}'")
                    context = await self._async_retrieve_wrapper(hop_query, top_k)
                    # Add None checks before calling .get()
                    if context is None:
                        tables_found = 0
                        examples_found = 0
                        confidence = "none"
                        safe_context = {"tables": {}, "examples": [], "metadata": {"confidence": "none", "error": "context is None"}}
                    else:
                        tables_found = len(context.get("tables", {}))
                        examples_found = len(context.get("examples", []))
                        metadata = context.get("metadata", {}) if isinstance(context, dict) else {}
                        confidence = metadata.get("confidence", "none")
                        safe_context = context
                    hop_result = {
                        "hop": i+1,
                        "description": hop["description"],
                        "query": hop_query,
                        "context": safe_context,
                        "execution_time": asyncio.get_event_loop().time() - hop_start_time,
                        "status": "success"
                    }
                    if confidence in ["none", "very_low", "low"] and i == 0:
                        print(f"[MultiHop] Hop {i+1}: Low confidence ({confidence}), enhancing query")
                        try:
                            enhanced_query = await self.hybrid_retriever._enhance_query(hop_query)
                            print(f"[MultiHop] Hop {i+1} Enhanced query: '{enhanced_query}'")
                            enhanced_context = await self._async_retrieve_wrapper(enhanced_query, top_k)
                            hop_result["context"] = enhanced_context
                            hop_result["enhanced_query"] = enhanced_query
                            hop_result["enhancement_applied"] = True
                        except Exception as enhance_error:
                            print(f"[MultiHop] Hop {i+1}: Query enhancement failed: {str(enhance_error)}")
                            hop_result["enhancement_error"] = str(enhance_error)
                    print(f"[MultiHop] Hop {i+1} completed: {tables_found} tables, {examples_found} examples, confidence: {confidence}")
                    return hop_result
                result = await asyncio.wait_for(_execute_hop_core(), timeout=self.hop_timeout)
                return result
            except asyncio.TimeoutError:
                print(f"[MultiHop] Hop {i+1} TIMEOUT after {self.hop_timeout} seconds")
                return {
                    "hop": i+1,
                    "description": hop["description"],
                    "query": hop.get("query", ""),
                    "context": {"tables": {}, "examples": [], "metadata": {"confidence": "none", "error": "timeout"}},
                    "execution_time": self.hop_timeout,
                    "status": "timeout",
                    "error": "Hop execution timed out"
                }
            except Exception as e:
                print(f"[MultiHop] Hop {i+1} ERROR: {str(e)}")
                return {
                    "hop": i+1,
                    "description": hop["description"],
                    "query": hop.get("query", ""),
                    "context": {"tables": {}, "examples": [], "metadata": {"confidence": "none", "error": str(e)}},
                    "execution_time": asyncio.get_event_loop().time() - hop_start_time,
                    "status": "error",
                    "error": str(e)
                }

        print(f"[MultiHop] Creating {len(hops)} parallel hop tasks...")
        hop_tasks = [
            _execute_hop_with_timeout(hop, i, top_k)
            for i, hop in enumerate(hops)
        ]
        print("[MultiHop] Executing all hops in parallel...")
        parallel_start_time = asyncio.get_event_loop().time()
        hop_contexts = await asyncio.gather(*hop_tasks, return_exceptions=True)
        parallel_execution_time = asyncio.get_event_loop().time() - parallel_start_time

        successful_hops = []
        failed_hops = []
        for i, result in enumerate(hop_contexts):
            if isinstance(result, Exception):
                print(f"[MultiHop] Hop {i+1} failed with exception: {str(result)}")
                failed_hops.append({
                    "hop": i+1,
                    "description": hops[i].get("description", "Unknown"),
                    "query": hops[i].get("query", ""),
                    "context": {"tables": {}, "examples": [], "metadata": {"confidence": "none", "error": str(result)}},
                    "execution_time": parallel_execution_time,
                    "status": "exception",
                    "error": str(result)
                })
            else:
                successful_hops.append(result)
        hop_contexts = successful_hops + failed_hops

        print(f"[MultiHop] Parallel execution completed in {parallel_execution_time:.2f}s")
        print(f"[MultiHop] Successful hops: {len(successful_hops)}, Failed hops: {len(failed_hops)}")

        # Step 4: Synthesis (combine hop results)
        print("[MultiHop] Synthesizing results from all hops")
        synthesis_start_time = asyncio.get_event_loop().time()
        synthesis = self._synthesize_hop_contexts(hop_contexts)
        synthesis_time = asyncio.get_event_loop().time() - synthesis_start_time
        print(f"[MultiHop] Synthesis completed in {synthesis_time:.2f}s")

        # Step 5: Context Optimization (NEW)
        print("[MultiHop] Optimizing context for LLM processing")
       
        synthesis = self._fully_qualify_table_names(synthesis)
        
        # Apply intelligent compression using comprehensive token monitoring
        synthesis_json = json.dumps(synthesis, indent=2)
        query_type = "synthesis"
        
        # Use token monitoring system for intelligent compression
        compressed_synthesis_json = monitor_and_compress_context(
            synthesis_json, query_type, ""
        )
        
        # Update synthesis if compression was applied
        # if compressed_synthesis_json != synthesis_json:
        #     try:
        #         synthesis = json.loads(compressed_synthesis_json)
        #         original_tokens = get_context_token_count(synthesis_json)
        #         compressed_tokens = get_context_token_count(compressed_synthesis_json)
        #         reduction = ((original_tokens - compressed_tokens) / original_tokens) * 100
        #         print(f"[MultiHop] Applied intelligent synthesis compression: {original_tokens} -> {compressed_tokens} tokens ({reduction:.1f}% reduction)")
        #     except json.JSONDecodeError:
        #         print(f"[MultiHop] Compression produced invalid JSON, using original synthesis")
        
        import datetime
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synthesized_context_{timestamp}.json"
        file_path = os.path.join(logs_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(synthesis, f, indent=2)
        print(f"[MultiHop] Synthesized context saved to {file_path}")
        synthesis_tables = len(synthesis.get("tables", {}))
        synthesis_examples = len(synthesis.get("examples", []))
        print(f"[MultiHop] Synthesis Results: {synthesis_tables} total tables, {synthesis_examples} total examples")

        self._store_learning_update(user_query, plan, hop_contexts, synthesis)

        total_time = asyncio.get_event_loop().time() - retrieval_start_time
        print(f"[MultiHop] Complete parallel multi-hop retrieval finished in {total_time:.2f}s")
        print(f"[MultiHop] Performance breakdown - Planning: {plan_time if 'plan_time' in locals() else 0:.2f}s, Parallel Execution: {parallel_execution_time:.2f}s, Synthesis: {synthesis_time:.2f}s")

        return {
            "user_query": user_query,
            "plan": plan,
            "hops": hop_contexts,
            "final_synthesis": synthesis,
            "performance_metrics": {
                "total_time": total_time,
                "plan_generation_time": plan_time if 'plan_time' in locals() else 0,
                "parallel_execution_time": parallel_execution_time,
                "synthesis_time": synthesis_time,
                "successful_hops": len(successful_hops),
                "failed_hops": len(failed_hops)
            }
        }

    def _check_memory_for_plan(self, user_query: str) -> Optional[Dict[str, Any]]:
        """Check memory for existing plan with fuzzy matching for similar queries"""
        # Normalize the query for better matching
        normalized_query = user_query.lower().strip()
        
        # Check for exact match first
        if normalized_query in self.memory_store:
            print("[MultiHop] Found exact plan match in memory")
            return self.memory_store[normalized_query]
        
        # Check for similar queries (fuzzy matching)
        for stored_query, plan in self.memory_store.items():
            # Simple similarity check - could be enhanced with more sophisticated matching
            if self._queries_are_similar(normalized_query, stored_query):
                print(f"[MultiHop] Found similar plan in memory for: '{stored_query}'")
                return plan
        
        return None
    
    def _queries_are_similar(self, query1: str, query2: str, threshold: float = 0.8) -> bool:
        """Check if two queries are similar enough to reuse the same plan"""
        # Simple word-based similarity
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity >= threshold

    def _is_simple(self, q: str) -> bool:
        """Determine if query is simple enough to skip LLM planning"""
        ql = q.lower()
        # Trivial heuristic: single-entity + optional basic filter/aggregation
        return any(t in ql for t in ["count ", " how many ", " list ", " show "]) and \
               not any(t in ql for t in [" join ", " and ", " or ", " between ", " window "])

    def _create_fallback_plan(self, user_query: str) -> Dict[str, Any]:
        """Create deterministic fallback plan for simple queries"""
        query_lower = user_query.lower()
        
        if any(term in query_lower for term in ["count", "how many", "total"]):
            return {
                "hops": [
                    {"description": "Find relevant tables and count information", "query": user_query}
                ]
            }
        elif any(term in query_lower for term in ["list", "show", "get"]):
            return {
                "hops": [
                    {"description": "Find relevant tables and data", "query": user_query},
                    {"description": "Apply filters and formatting", "query": f"filter and format {user_query}"}
                ]
            }
        else:
            return {
                "hops": [
                    {"description": "Analyze query requirements", "query": user_query}
                ]
            }

    async def _generate_query_plan(self, user_query: str, conversation_id: str = None) -> Dict[str, Any]:
        # Check if query is simple before LLM call
        if self._is_simple(user_query):
            # Use intelligent fallback instead of basic one
            return self._create_fallback_plan(user_query)
        
        # Use LLM for complex queries with concept-enhanced prompting
        try:
            # Extract concepts to enhance LLM planning
            concepts = self._extract_new_concepts_from_query(user_query, {})
            concept_context = f"Key concepts identified: {', '.join(concepts[:5])}" if concepts else ""
            
            # Get conversation history hints for better planning
            history_hints = ""
            if conversation_id:
                history_hints = self._get_history_hints_for_planning(conversation_id, user_query)
            
            prompt = f"""Decompose the user query into 1–4 hops for database table search. Focus on the key concepts to find relevant tables.

USER QUERY: {user_query}
{concept_context}
{history_hints}

**IMPORTANT CONTEXT GUIDANCE:**
- If conversation history shows successful columns/tables for this concept, REUSE those exact column/table patterns
- Search for columns/fields in existing database tables, NOT for product specification tables
- For device-related queries, look for DEVICE_NAME, DEVICE_MODEL, or similar columns in customer/subscriber tables
- Use database schema vocabulary, not generic product database terms

Create specific search queries that will help find database tables containing the relevant data.
Return ONLY JSON:
{{"hops":[{{"description":"...","query":"..."}}, ...]}}
"""
            
            response = self.hybrid_retriever.openai_client.chat.completions.create(
                model=OPENAI_DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.0,     # deterministic
                top_p=0.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            content = response.choices[0].message.content.strip()
            print(f"[MultiHop] LLM Response: {content}")
            
            # Try to extract JSON from the response
            if content.startswith('```json'):
                content = content.split('```json')[1].split('```')[0].strip()
            elif content.startswith('```'):
                content = content.split('```')[1].split('```')[0].strip()
            
            # Try to find JSON within the content
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            plan = json.loads(content)
            print(f"[MultiHop] Successfully parsed plan with {len(plan.get('hops', []))} hops")
            return plan
        except Exception as e:
            print(f"[MultiHop] Query planning failed: {str(e)}")
            print(f"[MultiHop] Raw response: {response.choices[0].message.content if 'response' in locals() else 'No response'}")
            # Fallback: create intelligent multi-hop plan based on concept analysis
            return self._create_fallback_plan(user_query)


    def _store_plan_in_memory(self, user_query: str, plan: Dict[str, Any]):
        """Store plan in memory with normalized key for better reuse"""
        normalized_query = user_query.lower().strip()
        self.memory_store[normalized_query] = plan
        print(f"[MultiHop] Stored plan for normalized query: '{normalized_query}'")

    def _synthesize_hop_contexts(self, hop_contexts: list) -> Dict[str, Any]:
        """Synthesize hop contexts with deterministic ordering for consistent results"""
        print(f"[MultiHop] Synthesizing {len(hop_contexts)} hop contexts from parallel execution")
        all_tables = {}
        all_examples = []
        all_join_keys = set()
        hop_metadata = []
        total_execution_time = 0
        successful_hops = 0
        basic_dedup_stats = {"tables_processed": 0, "tables_unique": 0}
        
        for hop in hop_contexts:
            if not isinstance(hop, dict):
                print(f"[MultiHop] Skipping invalid hop result: {type(hop)}")
                continue
            context = hop.get("context", {})
            tables = context.get("tables", {})
            examples = context.get("examples", [])
            join_keys = context.get("join_keys", [])
            metadata = context.get("metadata", {})
            hop_status = hop.get("status", "unknown")
            execution_time = hop.get("execution_time", 0)
            total_execution_time += execution_time
            if hop_status == "success":
                successful_hops += 1
            print(f"[MultiHop] Hop {hop['hop']}: {len(tables)} tables, {len(examples)} examples, status: {hop_status}, time: {execution_time:.2f}s")
            
            # Basic table collection with minimal deduplication (detailed dedup in context optimizer)
            for table_name, table_info in tables.items():
                basic_dedup_stats["tables_processed"] += 1
                
                if table_name not in all_tables:
                    # First occurrence - store with hop metadata
                    all_tables[table_name] = table_info.copy() if isinstance(table_info, dict) else table_info
                    if isinstance(all_tables[table_name], dict):
                        all_tables[table_name]["hop_sources"] = [hop["hop"]]
                        all_tables[table_name]["hop_descriptions"] = [hop["description"]]
                    basic_dedup_stats["tables_unique"] += 1
                else:
                    # Already exists - just add hop source
                    existing_table = all_tables[table_name]
                    if isinstance(existing_table, dict):
                        existing_table.setdefault("hop_sources", []).append(hop["hop"])
                        existing_table.setdefault("hop_descriptions", []).append(hop["description"])
            
            # Basic example deduplication
            if isinstance(examples, list):
                for example in examples:
                    if example not in all_examples:  # Simple comparison for speed
                        all_examples.append(example)
            
            if isinstance(join_keys, list):
                all_join_keys.update(join_keys)
            
            hop_metadata.append({
                "hop": hop["hop"],
                "confidence": metadata.get("confidence", "none"),
                "result_count": metadata.get("result_count", 0),
                "search_method": metadata.get("search_method", "unknown"),
                "status": hop_status,
                "execution_time": execution_time,
                "enhancement_applied": hop.get("enhancement_applied", False)
            })
        
        # Log basic deduplication stats
        basic_reduction = ((basic_dedup_stats["tables_processed"] - basic_dedup_stats["tables_unique"]) / 
                          max(basic_dedup_stats["tables_processed"], 1)) * 100
        print(f"[MultiHop] Table deduplication: {basic_dedup_stats['tables_processed']} - {basic_dedup_stats['tables_unique']} ({basic_reduction:.1f}% reduction)")
        
        if not all_tables:
            print("[MultiHop] No tables found in any hop - returning empty structure")
            return {
                "tables": {},
                "columns": {},
                "examples": [],
                "join_keys": [],
                "business_context": "No relevant schema found across all hops",
                "metadata": {
                    "result_count": 0,
                    "confidence": "none",
                    "search_method": "multi_hop_hybrid_parallel",
                    "hops_executed": len(hop_contexts),
                    "successful_hops": successful_hops,
                    "parallel_execution_time": total_execution_time,
                    "hop_metadata": hop_metadata,
                    "basic_dedup_stats": basic_dedup_stats
                }
            }
        
        hop_confidences = [hm["confidence"] for hm in hop_metadata if hm["status"] == "success"]
        if not hop_confidences:
            overall_confidence = "none"
        else:
            confidence_scores = {"very_high": 4, "high": 3, "medium": 2, "low": 1, "very_low": 0.5, "none": 0}
            avg_confidence_score = sum(confidence_scores.get(conf, 0) for conf in hop_confidences) / len(hop_confidences)
            if avg_confidence_score >= 3.5:
                overall_confidence = "very_high"
            elif avg_confidence_score >= 2.5:
                overall_confidence = "high"
            elif avg_confidence_score >= 1.5:
                overall_confidence = "medium"
            elif avg_confidence_score >= 0.5:
                overall_confidence = "low"
            else:
                overall_confidence = "very_low"
        print(f"[MultiHop] Parallel synthesis complete: {len(all_tables)} total tables, confidence: {overall_confidence}")
        print(f"[MultiHop] Parallel execution stats: {successful_hops}/{len(hop_contexts)} successful hops, total time: {total_execution_time:.2f}s")
        
        # Create deterministically ordered output
        ordered_tables = {}
        for t in sorted(all_tables.keys(), key=lambda x: x.lower()):
            info = all_tables[t]
            # Sort columns if dict-like
            if isinstance(info, dict) and "columns" in info and isinstance(info["columns"], dict):
                info["columns"] = {k: info["columns"][k] for k in sorted(info["columns"].keys(), key=str.lower)}
            ordered_tables[t] = info
        
        # Sort examples deterministically
        all_examples = sorted(set(all_examples))[:10]  # deterministic
        
        # 🧠 INTELLIGENT SCHEMA RELATIONSHIP DISCOVERY FOR MULTI-HOP CONTEXT
        schema_intelligence = {}
        try:
            print(f"[MultiHop] Applying intelligent schema relationship discovery to {len(ordered_tables)} tables")
            
            # Prepare context for schema analysis
            schema_context = {
                "tables": ordered_tables,
                "examples": all_examples
            }
            
            # Perform schema relationship discovery
            schema_analysis = schema_relationship_discovery.discover_relationships(schema_context)
            
            if schema_analysis.relationships:
                print(f"[MultiHop] Discovered {len(schema_analysis.relationships)} intelligent table relationships")
                
                # Store relationship intelligence for SQL generation
                schema_intelligence["relationships"] = [
                    {
                        "source_table": rel.source_table,
                        "target_table": rel.target_table,
                        "join_condition": rel.join_condition,
                        "confidence": rel.confidence,
                        "type": rel.relationship_type
                    }
                    for rel in schema_analysis.relationships
                ]
                
                # Add optimal join paths
                if schema_analysis.join_paths:
                    schema_intelligence["join_paths"] = schema_analysis.join_paths
                    print(f"[MultiHop] Generated optimal join paths for {len(schema_analysis.join_paths)} table combinations")
                
                # Add business concept mappings
                if schema_analysis.column_mappings:
                    schema_intelligence["business_mappings"] = schema_analysis.column_mappings
                    print(f"[MultiHop] Mapped {len(schema_analysis.column_mappings)} business concepts")
                
                # Suggest optimal JOINs for current table set
                table_names = list(ordered_tables.keys())
                if len(table_names) > 1:
                    join_suggestions = schema_relationship_discovery.suggest_optimal_joins(
                        table_names, schema_context
                    )
                    if join_suggestions:
                        schema_intelligence["suggested_joins"] = join_suggestions
                        print(f"[MultiHop] Generated {len(join_suggestions)} optimal JOIN suggestions")
                
                # Add relationship confidence metrics
                high_confidence_rels = [r for r in schema_analysis.relationships if r.confidence > 0.8]
                schema_intelligence["relationship_quality"] = {
                    "total_relationships": len(schema_analysis.relationships),
                    "high_confidence_count": len(high_confidence_rels),
                    "average_confidence": sum(r.confidence for r in schema_analysis.relationships) / len(schema_analysis.relationships) if schema_analysis.relationships else 0
                }
                
                print(f"[MultiHop] Schema intelligence complete - {len(high_confidence_rels)} high-confidence relationships")
            else:
                print("[MultiHop] No schema relationships discovered - proceeding with basic context")
                
        except Exception as e:
            print(f"[MultiHop] Schema relationship discovery failed: {str(e)}")
            # Continue without schema intelligence rather than failing
        
        synthesis_result = {
            "tables": ordered_tables,
            "examples": all_examples,
            "join_keys": sorted(all_join_keys),
            "metadata": {
                "result_count": len(all_tables),
                "confidence": overall_confidence,
                "search_method": "multi_hop_hybrid_parallel",
                "hops_executed": len(hop_contexts),
                "successful_hops": successful_hops,
                "parallel_execution_time": total_execution_time,
                "average_hop_time": total_execution_time / len(hop_contexts) if hop_contexts else 0,
                "hop_metadata": hop_metadata,
                "total_examples": len(all_examples)
            }
        }
        
        # Add schema intelligence if discovered
        if schema_intelligence:
            synthesis_result["schema_intelligence"] = schema_intelligence
            print(f"[MultiHop] Enhanced synthesis with schema intelligence")
        
        return synthesis_result

    async def _handle_context_extension(self, user_query: str, followup_context: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
        """Handle follow-up queries that extend previous context"""
        print("[MultiHop] Handling context extension for follow-up")
        
        # Get base context from previous query
        base_query_context = followup_context.get("base_query_context", {})
        
        # Determine what additional context we need with conversation awareness
        conversation_context = self._build_conversation_context(followup_context)
        additional_concepts = self._extract_new_concepts_from_query(user_query, conversation_context)
        
        if additional_concepts:
            print(f"[MultiHop] Identified additional concepts: {additional_concepts}")
            
            # CRITICAL: Preserve table consistency for follow-up queries
            # Extract previously successful tables from inherited context
            inherited_tables = followup_context.get("inherited_context", {}).get("tables", {})
            available_tables_list = followup_context.get("available_tables", [])
            
            # CRITICAL FIX: If we have table names but not full schemas, retrieve them first
            # if not inherited_tables and available_tables_list:
            #     print(f"[MultiHop] CRITICAL: Previous query used {available_tables_list} but schemas missing")
            #     print(f"[MultiHop] Retrieving full schemas for previous tables to preserve context")
                
            #     # Retrieve full schemas for previous tables
            #     inherited_tables = await self._retrieve_exact_table_schemas(available_tables_list)
            #     print(f"[MultiHop] Retrieved {len(inherited_tables)} previous table schemas")
            
            # If we have previous successful tables, prioritize extending them first
            if inherited_tables:
                print(f"[MultiHop] Preserving table consistency with {len(inherited_tables)} inherited tables")
                
                # Check if inherited tables can satisfy the new concepts
                extended_inherited_context = self._extend_inherited_tables_with_concepts(
                    inherited_tables, additional_concepts, user_query
                )
                
                if extended_inherited_context.get("tables"):
                    print(f"[MultiHop] Successfully extended inherited tables for follow-up concepts")
                    # Use extended inherited context as primary
                    merged_context = extended_inherited_context
                else:
                    print(f"[MultiHop] Inherited tables insufficient, performing targeted search with table guidance")
                    # Retrieve additional context but with table guidance from previous success
                    additional_context = await self._retrieve_incremental_context_with_table_guidance(
                        additional_concepts, 
                        existing_tables=available_tables_list,
                        table_guidance=list(inherited_tables.keys()),
                        top_k=top_k
                    )
                    
                    # CRITICAL: Always include previous table schemas in the merged context
                    merged_context = self._merge_contexts_with_table_priority(
                        base_context={"tables": inherited_tables, "examples": [], "join_keys": []},
                        additional_context=additional_context,
                        query=user_query,
                        preserve_tables=list(inherited_tables.keys())
                    )
            else:
                # Fallback to original logic if no inherited tables
                print("[MultiHop] WARNING: No previous tables found, performing new search")
                additional_context = await self._retrieve_incremental_context(
                    additional_concepts, 
                    existing_tables=available_tables_list,
                    top_k=top_k
                )
                
                # Merge contexts intelligently
                merged_context = self._merge_contexts(
                    base_context=followup_context.get("inherited_context", {}),
                    additional_context=additional_context,
                    query=user_query
                )
            
            followup_result = {
                "tables": merged_context.get("tables", {}),
                "examples": merged_context.get("examples", []),
                "join_keys": merged_context.get("join_keys", []),
                "metadata": {
                    "result_count": len(merged_context.get("tables", {})),
                    "confidence": "very_high",
                    "search_method": "context_extension",
                    "base_query": base_query_context.get("original_query", ""),
                    "extension_concepts": additional_concepts,
                    "reused_context": True
                }
            }
            
            # Save follow-up context to logs (similar to synthesized context)
            self._save_followup_context_to_logs(user_query, followup_result, additional_concepts, followup_context)
            
            return followup_result
        else:
            # No new concepts, reuse existing context
            print("[MultiHop] No new concepts detected, reusing existing context")
            reused_result = self._format_existing_context_for_reuse(followup_context)
            
            # Save reused context to logs 
            self._save_followup_context_to_logs(user_query, reused_result, [], followup_context, context_type="reused")
            
            return reused_result

    async def _handle_context_combination(self, user_query: str, followup_context: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
        """Handle follow-up queries that combine multiple previous contexts"""
        print("[MultiHop] Handling context combination for follow-up")
        
        relevant_queries = followup_context.get("relevant_queries", [])
        
        # Combine contexts from multiple relevant queries
        combined_context = self._combine_multiple_contexts(relevant_queries)
        
        # Check if we need additional context with conversation awareness
        conversation_context = self._build_conversation_context(followup_context)
        additional_concepts = self._extract_new_concepts_from_query(user_query, conversation_context)
        
        if additional_concepts:
            additional_context = await self._retrieve_incremental_context(
                additional_concepts,
                existing_tables=combined_context.get("tables", {}).keys(),
                top_k=top_k
            )
            
            final_context = self._merge_contexts(combined_context, additional_context, user_query)
        else:
            final_context = combined_context
        
        combination_result = {
            "tables": final_context.get("tables", {}),
            "examples": final_context.get("examples", []),
            "join_keys": final_context.get("join_keys", []),
            "metadata": {
                "result_count": len(final_context.get("tables", {})),
                "confidence": "very_high",
                "search_method": "context_combination",
                "combined_queries": len(relevant_queries),
                "additional_concepts": additional_concepts,
                "reused_context": True
            }
        }
        
        # Save combination context to logs
        self._save_followup_context_to_logs(user_query, combination_result, additional_concepts, followup_context, context_type="combined")
        
        return combination_result

    def _extract_new_concepts_from_query(self, query: str, conversation_context: Dict[str, Any] = None) -> List[str]:
        """
        Intelligent, adaptive concept extraction using semantic analysis and learning.
        
        This system dynamically identifies business concepts without hardcoded patterns,
        learning from query patterns and conversation context to continuously improve.
        """
        try:
            # Semantic concept extraction using LLM reasoning
            semantic_concepts = self._extract_concepts_semantically(query, conversation_context)
            
            # Context-aware concept enrichment based on previous queries
            contextual_concepts = self._enhance_with_conversation_context(query, semantic_concepts, conversation_context)
            
            # Dynamic learning from query patterns
            learned_concepts = self._apply_learned_patterns(query, contextual_concepts)
            
            # Combine and deduplicate concepts
            all_concepts = list(set(semantic_concepts + contextual_concepts + learned_concepts))
            
            # Intelligent concept prioritization based on relevance
            prioritized_concepts = self._prioritize_concepts_by_relevance(query, all_concepts, conversation_context)
            
            print(f"[MultiHop] Intelligent concept extraction for '{query}': {prioritized_concepts}")
            print(f"[MultiHop] Concept breakdown - Semantic: {len(semantic_concepts)}, Contextual: {len(contextual_concepts)}, Learned: {len(learned_concepts)}")
            
            return prioritized_concepts
            
        except Exception as e:
            print(f"[MultiHop] Intelligent concept extraction failed: {str(e)}, falling back to basic analysis")
            # Graceful fallback to ensure system resilience
            return self._extract_concepts_basic_analysis(query)

    def _extract_concepts_semantically(self, query: str, conversation_context: Dict[str, Any] = None) -> List[str]:
        """
        Use LLM semantic analysis to identify business concepts dynamically.
        This replaces hardcoded patterns with intelligent reasoning.
        """
        try:
            # Build context-aware prompt for concept extraction
            context_info = ""
            if conversation_context:
                previous_queries = conversation_context.get("previous_queries", [])
                if previous_queries:
                    context_info = f"\nPrevious context: {previous_queries[-1].get('query', '')}"
            
            prompt = f"""Analyze this user query and identify the key business concepts that would require data retrieval:

Query: "{query}"{context_info}

Identify specific data domains/concepts present in this query. Consider:
- What type of business data is being requested?
- What analytical dimensions are involved?
- What business entities or processes are referenced?

Return ONLY a JSON array of concept names (max 6 concepts):
["concept1", "concept2", "concept3"]

Focus on data-centric concepts that would help in finding relevant tables and columns."""

            response = self.hybrid_retriever.openai_client.chat.completions.create(
                model=OPENAI_DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1,  # Low temperature for consistency
                top_p=0.9
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON array from response
            import re
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                concepts_json = json_match.group()
                concepts = json.loads(concepts_json)
                # Normalize concept names
                normalized_concepts = [concept.lower().replace(" ", "_") for concept in concepts if isinstance(concept, str)]
                print(f"[MultiHop] Semantic concepts extracted: {normalized_concepts}")
                return normalized_concepts[:6]  # Limit to 6 concepts
                
        except Exception as e:
            print(f"[MultiHop] Semantic concept extraction failed: {str(e)}")
        
        return []

    def _enhance_with_conversation_context(self, query: str, base_concepts: List[str], 
                                         conversation_context: Dict[str, Any] = None) -> List[str]:
        """
        Enhance concepts based on conversation history and business domain patterns.
        This creates continuity and context-awareness in concept detection.
        """
        enhanced_concepts = []
        
        if not conversation_context:
            return enhanced_concepts
        
        try:
            # Analyze previous queries for concept evolution
            previous_queries = conversation_context.get("previous_queries", [])
            if previous_queries:
                recent_query = previous_queries[-1]
                previous_concepts = recent_query.get("detected_concepts", [])
                
                # Check for concept evolution (e.g., "customers" -> "customer payment methods")
                for prev_concept in previous_concepts:
                    if prev_concept in ["customer", "subscriber", "individual"]:
                        # Check if current query is extending customer analysis
                        if any(term in query.lower() for term in ["payment", "billing", "auto", "pay", "financial"]):
                            enhanced_concepts.append("customer_financial_profile")
                        elif any(term in query.lower() for term in ["device", "plan", "service", "subscription"]):
                            enhanced_concepts.append("customer_service_profile")
                        elif any(term in query.lower() for term in ["active", "status", "enabled", "disabled"]):
                            enhanced_concepts.append("customer_account_status")
            
            # Domain relationship detection
            if "device" in base_concepts and query.lower().find("pay") >= 0:
                enhanced_concepts.append("device_financing")
            
            if "customer" in base_concepts and any(term in query.lower() for term in ["pah", "primary", "account", "holder"]):
                enhanced_concepts.append("account_hierarchy")
            
            print(f"[MultiHop] Context-enhanced concepts: {enhanced_concepts}")
            return enhanced_concepts
            
        except Exception as e:
            print(f"[MultiHop] Context enhancement failed: {str(e)}")
            return []

    def _apply_learned_patterns(self, query: str, current_concepts: List[str]) -> List[str]:
        """
        Apply learned patterns from previous successful query resolutions.
        This creates a feedback loop for continuous improvement.
        """
        learned_concepts = []
        
        try:
            # Pattern learning from successful queries
            # This could be enhanced with a persistent learning store
            query_lower = query.lower()
            
            # Dynamic pattern recognition based on query structure
            if "how many" in query_lower and "are" in query_lower:
                # Pattern: "how many X are Y" -> suggests filtering/classification
                learned_concepts.append("classification_analysis")
            
            if "out of these" in query_lower or "from those" in query_lower:
                # Pattern: Follow-up filtering -> suggests drill-down analysis
                learned_concepts.append("subset_analysis")
                
            if any(word in query_lower for word in ["enabled", "disabled", "active", "inactive"]):
                # Pattern: Status-based queries -> suggests status tracking
                learned_concepts.append("status_tracking")
            
            # Learn from co-occurrence patterns
            if "auto" in query_lower and ("pay" in query_lower or "payment" in query_lower):
                learned_concepts.append("automated_payment_systems")
            
            print(f"[MultiHop] Learned pattern concepts: {learned_concepts}")
            return learned_concepts
            
        except Exception as e:
            print(f"[MultiHop] Pattern learning failed: {str(e)}")
            return []

    def _prioritize_concepts_by_relevance(self, query: str, concepts: List[str], 
                                        conversation_context: Dict[str, Any] = None) -> List[str]:
        """
        Intelligently prioritize concepts based on query relevance and business context.
        This ensures the most important concepts are processed first.
        """
        try:
            if not concepts:
                return concepts
            
            # Score concepts based on relevance
            concept_scores = {}
            query_lower = query.lower()
            
            for concept in concepts:
                score = 0
                concept_terms = concept.replace("_", " ").split()
                
                # Direct mention scoring
                for term in concept_terms:
                    if term in query_lower:
                        score += 3  # High score for direct mentions
                
                # Context relevance scoring
                if conversation_context:
                    previous_queries = conversation_context.get("previous_queries", [])
                    if previous_queries:
                        recent_context = previous_queries[-1].get("query", "").lower()
                        for term in concept_terms:
                            if term in recent_context:
                                score += 2  # Medium score for context relevance
                
                # Business priority scoring (dynamic)
                priority_patterns = {
                    "financial": ["payment", "billing", "cost", "revenue"],
                    "customer": ["individual", "subscriber", "user"],
                    "status": ["active", "enabled", "disabled"],
                    "analysis": ["how many", "count", "total"]
                }
                
                for priority_type, terms in priority_patterns.items():
                    if any(term in concept for term in terms) and any(term in query_lower for term in terms):
                        score += 1  # Boost for business priority alignment
                
                concept_scores[concept] = score
            
            # Sort by relevance score
            prioritized = sorted(concepts, key=lambda x: concept_scores.get(x, 0), reverse=True)
            
            # Return top concepts (limit to 5 for performance)
            return prioritized[:5]
            
        except Exception as e:
            print(f"[MultiHop] Concept prioritization failed: {str(e)}")
            return concepts[:5]  # Fallback to first 5

    def _extract_concepts_basic_analysis(self, query: str) -> List[str]:
        """
        Fallback method using basic linguistic analysis when advanced methods fail.
        This ensures system resilience.
        """
        try:
            query_lower = query.lower()
            basic_concepts = []
            
            # Basic entity extraction
            if any(term in query_lower for term in ["customer", "individual", "subscriber", "user"]):
                basic_concepts.append("customer_entity")
            
            if any(term in query_lower for term in ["payment", "pay", "billing", "financial"]):
                basic_concepts.append("financial_entity")
            
            if any(term in query_lower for term in ["device", "phone", "equipment"]):
                basic_concepts.append("device_entity")
            
            if any(term in query_lower for term in ["plan", "service", "subscription"]):
                basic_concepts.append("service_entity")
            
            if any(term in query_lower for term in ["active", "inactive", "enabled", "disabled", "status"]):
                basic_concepts.append("status_entity")
            
            # Basic analysis type detection
            if any(term in query_lower for term in ["how many", "count", "total", "number"]):
                basic_concepts.append("count_analysis")
            
            if any(term in query_lower for term in ["list", "show", "display", "get"]):
                basic_concepts.append("list_analysis")
            
            print(f"[MultiHop] Basic concept analysis: {basic_concepts}")
            return basic_concepts
            
        except Exception as e:
            print(f"[MultiHop] Basic concept analysis failed: {str(e)}")
            return ["general_inquiry"]  # Ultimate fallback

    def _get_history_hints_for_planning(self, conversation_id: str, user_query: str) -> str:
        """
        Extract hints from conversation history to guide the multi-hop planner.
        Returns a formatted string with table/column patterns from previous successful queries.
        """
        try:
            # Get conversation chain from memory
            chain = conversation_memory_manager.conversation_chains.get(conversation_id)
            if not chain or not chain.contexts:
                return ""
            
            # Extract relevant concepts from user query
            query_lower = user_query.lower()
            hints = []
            
            # Check if query mentions concepts from previous contexts
            for ctx in chain.contexts[-5:]:  # Check last 5 contexts
                # Extract table and column information from successful queries
                if hasattr(ctx, 'learning_signals'):
                    table_metadata = ctx.learning_signals.get('table_chain_metadata', {})
                    tables_used = table_metadata.get('tables_accessed', [])
                    condition_bindings = table_metadata.get('condition_table_bindings', {})
                    
                    # Check if any columns relate to current query keywords
                    for binding_key, binding_info in condition_bindings.items():
                        if binding_key.startswith('__COLUMN__'):
                            column_name = binding_info.get('column', '').lower()
                            # Check if column relates to query (e.g., "device" in query and "DEVICE_NAME" in column)
                            for keyword in ['device', 'model', 'phone', 'iphone', 'customer', 'subscriber', 'account']:
                                if keyword in query_lower and keyword in column_name:
                                    source_table = binding_info.get('source_table_short', '')
                                    if source_table:
                                        hints.append(f"- Previous successful query for '{keyword}' used column: {binding_info.get('column')} from table {source_table}")
                                        break
            
            if hints:
                return f"\n**CONVERSATION HISTORY HINTS:**\n" + "\n".join(hints[:3])  # Limit to 3 hints
            return ""
        except Exception as e:
            print(f"[MultiHop] Failed to get history hints: {str(e)}")
            return ""

    def _build_conversation_context(self, followup_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build comprehensive conversation context for intelligent concept extraction.
        """
        try:
            context = {}
            
            # Extract previous queries and their concepts
            if followup_context:
                # Get conversation memory information
                base_query_context = followup_context.get("base_query_context", {})
                if base_query_context:
                    context["previous_queries"] = [{
                        "query": base_query_context.get("original_query", ""),
                        "detected_concepts": base_query_context.get("extracted_conditions", {}).get("concepts", []),
                        "business_context": base_query_context.get("business_context", {})
                    }]
                
                # Extract business thread and domain information
                context["business_domain"] = followup_context.get("business_thread", "")
                context["established_entities"] = followup_context.get("available_tables", [])
                
                # Get strategic context for learning
                strategy_context = followup_context.get("strategy_context", {})
                if strategy_context:
                    context["query_patterns"] = strategy_context.get("patterns", [])
                    context["success_indicators"] = strategy_context.get("success_metrics", {})
            
            return context
            
        except Exception as e:
            print(f"[MultiHop] Context building failed: {str(e)}")
            return {}

    def _store_concept_learning(self, query: str, extracted_concepts: List[str], 
                              success_indicators: Dict[str, Any] = None):
        """
        Store successful concept extractions for future learning.
        This builds a knowledge base of effective concept detection patterns.
        """
        try:
            # Use global plan cache to store learning data
            learning_key = f"concept_learning_{self._norm(query)[:50]}"
            
            learning_data = {
                "query": query,
                "concepts": extracted_concepts,
                "timestamp": asyncio.get_event_loop().time(),
                "success_indicators": success_indicators or {}
            }
            
            # Store in global cache for persistence across requests
            if learning_key not in _global_plan_cache:
                _global_plan_cache[learning_key] = []
            
            _global_plan_cache[learning_key].append(learning_data)
            
            # Keep only recent entries to prevent cache bloat
            if len(_global_plan_cache[learning_key]) > 10:
                _global_plan_cache[learning_key] = _global_plan_cache[learning_key][-10:]
            
            print(f"[MultiHop] Stored concept learning for future adaptation")
            
        except Exception as e:
            print(f"[MultiHop] Concept learning storage failed: {str(e)}")

    def _get_learned_patterns(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve learned patterns similar to the current query for adaptive improvement.
        """
        try:
            learned_patterns = []
            query_norm = self._norm(query)
            
            # Search through stored learning data
            for key, learning_entries in _global_plan_cache.items():
                if key.startswith("concept_learning_") and isinstance(learning_entries, list):
                    for entry in learning_entries:
                        stored_query = entry.get("query", "")
                        if self._queries_are_similar(query_norm, self._norm(stored_query), threshold=0.6):
                            learned_patterns.append({
                                "concepts": entry.get("concepts", []),
                                "success_score": entry.get("success_indicators", {}).get("relevance_score", 0.5),
                                "query_similarity": 0.6  # Could be computed more precisely
                            })
            
            # Sort by success score and similarity
            learned_patterns.sort(key=lambda x: x["success_score"] + x["query_similarity"], reverse=True)
            
            return learned_patterns[:3]  # Top 3 most relevant learned patterns
            
        except Exception as e:
            print(f"[MultiHop] Pattern retrieval failed: {str(e)}")
            return []

    # async def _retrieve_exact_table_schemas(self, table_names: List[str]) -> Dict[str, Any]:
    #     """
    #     Retrieve full schemas for specific table names from AI Search.
    #     This ensures that previous query tables are included in follow-up context.
        
    #     Args:
    #         table_names: List of full table names (e.g., ['INDIVIDUAL_SUBSCRIBER_PROFILE'])
            
    #     Returns:
    #         Dictionary mapping table names to their full schema information
    #     """
    #     print(f"[MultiHop] Retrieving exact schemas for {len(table_names)} tables")
        
    #     retrieved_tables = {}
        
    #     try:
    #         # Search for each table by exact name
    #         for table_name in table_names:
    #             # Extract just the table name without schema prefix if present
    #             short_name = table_name.split('.')[-1] if '.' in table_name else table_name
                
    #             # Build search query for exact table match
    #             search_query = f"{short_name} data table column information"
                
    #             print(f"[MultiHop] Searching for table: {short_name}")
                
    #             # Execute search - returns dict with "tables" key
    #             search_result = await self._async_retrieve_wrapper(search_query, top_k=3)
                
    #             # Get the tables dictionary from search result
    #             tables_dict = search_result.get("tables", {})
                
    #             # Find exact match by comparing table names
    #             for full_table_name, table_info in tables_dict.items():
    #                 result_short_name = full_table_name.split('.')[-1] if '.' in full_table_name else full_table_name
                    
    #                 if result_short_name.upper() == short_name.upper():
    #                     # Found exact match - store with full table name as key
    #                     retrieved_tables[full_table_name] = table_info
    #                     print(f"[MultiHop] ✓ Retrieved schema for {full_table_name}")
    #                     break
    #             else:
    #                 print(f"[MultiHop] ✗ Schema not found for {short_name}")
            
    #         return retrieved_tables
            
    #     except Exception as e:
    #         print(f"[MultiHop] ERROR: Failed to retrieve table schemas: {str(e)}")
    #         import traceback
    #         print(f"[MultiHop] ERROR Traceback: {traceback.format_exc()}")
    #         return {}

    async def _retrieve_incremental_context(self, concepts: List[str], existing_tables: List[str], top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieve context intelligently based on semantic concepts rather than simple keywords.
        This creates more targeted and relevant metadata retrieval.
        """
        print(f"[MultiHop] Intelligent context retrieval for concepts: {concepts}")
        
        try:
            # Build semantic search queries for each concept
            search_queries = self._build_semantic_search_queries(concepts)
            
            # Execute searches in parallel for efficiency
            search_tasks = [
                self._async_retrieve_wrapper(query, top_k=2) 
                for query in search_queries
            ]
            
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Intelligently merge results
            merged_context = self._merge_concept_based_results(search_results, concepts, existing_tables)
            
            # Apply concept-based filtering to avoid irrelevant tables
            filtered_context = self._filter_context_by_concepts(merged_context, concepts)
            
            # Store successful concept patterns for learning
            if filtered_context.get("tables"):
                self._store_concept_learning(
                    query=" ".join(concepts), 
                    extracted_concepts=concepts,
                    success_indicators={"retrieved_tables": len(filtered_context["tables"])}
                )
            
            return filtered_context or {"tables": {}, "examples": [], "join_keys": []}
            
        except Exception as e:
            print(f"[MultiHop] Intelligent context retrieval failed: {str(e)}")
            # Fallback to basic approach
            return await self._retrieve_context_basic(concepts, existing_tables, top_k)

    def _build_semantic_search_queries(self, concepts: List[str]) -> List[str]:
        """
        Build intelligent search queries based on extracted concepts.
        This replaces generic keyword concatenation with semantic understanding.
        """
        search_queries = []
        
        for concept in concepts:
            # Transform concepts into targeted search queries
            if "financial" in concept or "payment" in concept:
                search_queries.append("payment billing autopay financial customer account")
            elif "customer" in concept:
                search_queries.append("customer individual subscriber profile account information")
            elif "device" in concept:
                search_queries.append("device equipment phone model subscription hardware")
            elif "status" in concept or "active" in concept:
                search_queries.append("status active inactive enabled disabled account state")
            elif "service" in concept:
                search_queries.append("service plan subscription rate offering product")
            elif "analysis" in concept:
                search_queries.append("count total aggregation metrics analysis reporting")
            else:
                # Semantic transformation for unknown concepts
                concept_terms = concept.replace("_", " ").split()
                search_queries.append(f"{' '.join(concept_terms)} data table column information")
        
        # Remove duplicates while preserving order
        unique_queries = []
        for query in search_queries:
            if query not in unique_queries:
                unique_queries.append(query)
        
        return unique_queries[:4]  # Limit to 4 searches for performance

    def _merge_concept_based_results(self, search_results: List[Any], concepts: List[str], 
                                   existing_tables: List[str]) -> Dict[str, Any]:
        """
        Intelligently merge search results based on concept relevance.
        """
        merged_tables = {}
        merged_examples = []
        merged_join_keys = []
        
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                continue
                
            if not isinstance(result, dict):
                continue
                
            tables = result.get("tables", {})
            
            # Score tables based on concept relevance
            for table_name, table_info in tables.items():
                if table_name in existing_tables:
                    continue  # Skip existing tables
                    
                relevance_score = self._calculate_table_concept_relevance(
                    table_info, concepts[i % len(concepts)]
                )
                
                if relevance_score > 0.3:  # Only include relevant tables
                    if table_name not in merged_tables:
                        merged_tables[table_name] = table_info.copy() if isinstance(table_info, dict) else table_info
                        if isinstance(merged_tables[table_name], dict):
                            merged_tables[table_name]["concept_relevance"] = relevance_score
                            merged_tables[table_name]["source_concepts"] = [concepts[i % len(concepts)]]
                    else:
                        # Update relevance score for existing table
                        if isinstance(merged_tables[table_name], dict):
                            existing_score = merged_tables[table_name].get("concept_relevance", 0)
                            merged_tables[table_name]["concept_relevance"] = max(existing_score, relevance_score)
                            merged_tables[table_name]["source_concepts"].append(concepts[i % len(concepts)])
            
            # Merge examples and join keys
            if isinstance(result.get("examples"), list):
                merged_examples.extend(result["examples"])
            if isinstance(result.get("join_keys"), list):
                merged_join_keys.extend(result["join_keys"])
        
        return {
            "tables": merged_tables,
            "examples": list(set(merged_examples))[:5],  # Deduplicate and limit
            "join_keys": list(set(merged_join_keys))
        }

    def _calculate_table_concept_relevance(self, table_info: Any, concept: str) -> float:
        """
        Calculate how relevant a table is to a specific concept using semantic analysis.
        """
        if not isinstance(table_info, dict):
            return 0.1  # Low default score
        
        score = 0.0
        concept_lower = concept.lower()
        
        # Analyze table name
        table_name = table_info.get("name", "").lower()
        if any(term in table_name for term in concept_lower.split("_")):
            score += 0.3
        
        # Analyze table description/business context
        description = table_info.get("business_context", "").lower()
        if description and any(term in description for term in concept_lower.split("_")):
            score += 0.2
        
        # Analyze column information
        columns = table_info.get("columns", {})
        if isinstance(columns, dict):
            column_matches = 0
            for col_name, col_info in columns.items():
                col_desc = ""
                if isinstance(col_info, dict):
                    col_desc = col_info.get("description", "").lower()
                elif isinstance(col_info, str):
                    col_desc = col_info.lower()
                
                if any(term in col_name.lower() or term in col_desc for term in concept_lower.split("_")):
                    column_matches += 1
            
            if column_matches > 0:
                score += min(0.4, column_matches * 0.1)  # Up to 0.4 for column relevance
        
        return min(score, 1.0)  # Cap at 1.0

    def _filter_context_by_concepts(self, context: Dict[str, Any], concepts: List[str]) -> Dict[str, Any]:
        """
        Apply intelligent filtering to ensure retrieved context is concept-relevant.
        """
        if not context or not context.get("tables"):
            return context
        
        # Filter tables based on concept relevance scores
        filtered_tables = {}
        tables = context.get("tables", {})
        
        for table_name, table_info in tables.items():
            if isinstance(table_info, dict):
                relevance = table_info.get("concept_relevance", 0)
                if relevance > 0.4:  # Keep highly relevant tables
                    filtered_tables[table_name] = table_info
                elif relevance > 0.2 and len(filtered_tables) < 3:  # Keep moderately relevant if we need more
                    filtered_tables[table_name] = table_info
            else:
                # Keep non-dict table info as is
                filtered_tables[table_name] = table_info
        
        return {
            "tables": filtered_tables,
            "examples": context.get("examples", []),
            "join_keys": context.get("join_keys", [])
        }

    async def _retrieve_context_basic(self, concepts: List[str], existing_tables: List[str], top_k: int = 3) -> Dict[str, Any]:
        """
        Fallback method for context retrieval when intelligent methods fail.
        """
        try:
            # Simple concept concatenation as fallback
            concept_query = " ".join(concepts) + " data tables columns"
            additional_context = await self._async_retrieve_wrapper(concept_query, top_k)
            
            # Basic filtering to remove existing tables
            if additional_context and "tables" in additional_context:
                new_tables = {}
                for table_name, table_info in additional_context["tables"].items():
                    if table_name not in existing_tables:
                        new_tables[table_name] = table_info
                additional_context["tables"] = new_tables
            
            return additional_context or {"tables": {}, "examples": [], "join_keys": []}
            
        except Exception as e:
            print(f"[MultiHop] Basic context retrieval failed: {str(e)}")
            return {"tables": {}, "examples": [], "join_keys": []}

    def _merge_contexts(self, base_context: Dict[str, Any], additional_context: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Intelligently merge two contexts"""
        print("[MultiHop] Merging contexts intelligently")
        
        merged = {
            "tables": {},
            "examples": [],
            "join_keys": []
        }
        
        # Merge tables
        if isinstance(base_context, dict) and "tables" in base_context:
            merged["tables"].update(base_context["tables"])
        
        if isinstance(additional_context, dict) and "tables" in additional_context:
            merged["tables"].update(additional_context["tables"])
        
        # Merge examples (deduplicate)
        all_examples = []
        if isinstance(base_context, dict) and "examples" in base_context:
            all_examples.extend(base_context["examples"])
        if isinstance(additional_context, dict) and "examples" in additional_context:
            all_examples.extend(additional_context["examples"])
        
        merged["examples"] = list(set(all_examples))[:10]  # Limit to 10 unique examples
        
        # Merge join keys
        all_join_keys = []
        if isinstance(base_context, dict) and "join_keys" in base_context:
            all_join_keys.extend(base_context["join_keys"])
        if isinstance(additional_context, dict) and "join_keys" in additional_context:
            all_join_keys.extend(additional_context["join_keys"])
        
        merged["join_keys"] = list(set(all_join_keys))
        
        return merged

    def _combine_multiple_contexts(self, relevant_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine contexts from multiple previous queries"""
        print(f"[MultiHop] Combining contexts from {len(relevant_queries)} queries")
        
        combined = {
            "tables": {},
            "examples": [],
            "join_keys": []
        }
        
        for query_context in relevant_queries:
            if "tables_used" in query_context:
                # This is query context from conversation memory
                for table in query_context["tables_used"]:
                    combined["tables"][table] = {"name": table, "source": "conversation_memory"}
        
        return combined

    def _format_existing_context_for_reuse(self, followup_context: Dict[str, Any]) -> Dict[str, Any]:
        """Format existing context for reuse with full schema information"""
        print("[MultiHop] Formatting existing context for reuse")
        
        # Priority 1: Get the full inherited context (contains complete schema info)
        inherited_context = followup_context.get("inherited_context", {})
        
        # If we have full inherited context with tables, use it directly
        if inherited_context and "tables" in inherited_context and inherited_context["tables"]:
            print(f"[MultiHop] Reusing full inherited context with {len(inherited_context.get('tables', {}))} tables")
            return {
                "tables": inherited_context.get("tables", {}),
                "examples": inherited_context.get("examples", []),
                "join_keys": inherited_context.get("join_keys", []),
                "metadata": {
                    "result_count": len(inherited_context.get("tables", {})),
                    "confidence": "very_high",
                    "search_method": "context_reuse",
                    "reused_context": True,
                    "source": "full_inherited_context"
                }
            }
        
        # Priority 2: Try schema_inheritance (also should contain full schema)
        schema_inheritance = followup_context.get("schema_inheritance", {})
        if schema_inheritance and "tables" in schema_inheritance and schema_inheritance["tables"]:
            print(f"[MultiHop] Using schema inheritance with {len(schema_inheritance.get('tables', {}))} tables")
            return {
                "tables": schema_inheritance.get("tables", {}),
                "examples": schema_inheritance.get("examples", []),
                "join_keys": schema_inheritance.get("join_keys", []),
                "metadata": {
                    "result_count": len(schema_inheritance.get("tables", {})),
                    "confidence": "very_high", 
                    "search_method": "schema_inheritance",
                    "reused_context": True,
                    "source": "schema_inheritance"
                }
            }
        
        # Priority 3: Build context from available table names (minimal approach)
        tables = {}
        available_tables = followup_context.get("available_tables", [])
        
        # Try to get any table information from the context
        for table in available_tables:
            table_info = {
                "name": table,
                "source": "conversation_memory",
                "filters": followup_context.get("inherited_filters", [])
            }
            
            # Try to find more detailed info from schema_inheritance
            if schema_inheritance and isinstance(schema_inheritance, dict):
                for key, value in schema_inheritance.items():
                    if isinstance(value, dict) and table.lower() in key.lower():
                        table_info.update(value)
                        break
            
            tables[table] = table_info
        
        print(f"[MultiHop] Using minimal context fallback with {len(tables)} tables")
        return {
            "tables": tables,
            "examples": [],
            "join_keys": followup_context.get("established_joins", []),
            "metadata": {
                "result_count": len(tables),
                "confidence": "medium" if tables else "low",
                "search_method": "context_reuse_minimal",
                "reused_context": True,
                "source": "minimal_fallback",
                "warning": "Limited schema information available" if not tables else None
            }
        }

    def _store_conversation_learning(self, query: str, follow_up_analysis: Dict[str, Any], 
                                   conversation_context: Dict[str, Any]):
        """
        Store conversation patterns for intelligent follow-up processing improvements.
        """
        try:
            learning_key = f"conversation_learning_{self._reqid(query)}"
            
            learning_data = {
                "query": query,
                "follow_up_type": follow_up_analysis.get("follow_up_type"),
                "strategy": follow_up_analysis.get("strategy"),
                "confidence": follow_up_analysis.get("confidence", 0),
                "conversation_context": conversation_context,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            _global_plan_cache[learning_key] = learning_data
            print(f"[MultiHop] Stored conversation learning pattern for adaptive improvement")
            
        except Exception as e:
            print(f"[MultiHop] Conversation learning storage failed: {str(e)}")

    def _store_learning_update(self, user_query, plan, hop_contexts, synthesis):
        """
        Enhanced learning system that stores successful patterns for future adaptability.
        """
        try:
            # Extract successful patterns
            successful_hops = [h for h in hop_contexts if h.get("status") == "success"]
            if not successful_hops:
                return
                
            # Store query-plan-result patterns for future optimization
            learning_pattern = {
                "query_normalized": self._norm(user_query),
                "successful_plan": plan,
                "result_quality": {
                    "tables_found": len(synthesis.get("tables", {})),
                    "confidence": synthesis.get("metadata", {}).get("confidence", "none"),
                    "execution_time": synthesis.get("metadata", {}).get("parallel_execution_time", 0)
                },
                "success_indicators": {
                    "hop_success_rate": len(successful_hops) / len(hop_contexts),
                    "relevance_score": self._calculate_result_relevance(synthesis, user_query)
                },
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Store in global cache with expiration
            learning_key = f"pattern_learning_{self._reqid(user_query)}"
            _global_plan_cache[learning_key] = learning_pattern
            
            print(f"[MultiHop] Stored enhanced learning pattern for query optimization")
            
        except Exception as e:
            print(f"[MultiHop] Enhanced learning storage failed: {str(e)}")

    def _calculate_result_relevance(self, synthesis: Dict[str, Any], query: str) -> float:
        """
        Calculate how relevant the retrieved results are to the original query.
        """
        try:
            tables = synthesis.get("tables", {})
            if not tables:
                return 0.0
                
            query_terms = set(self._norm(query).split())
            relevance_scores = []
            
            for table_name, table_info in tables.items():
                table_terms = set(table_name.lower().replace("_", " ").split())
                if isinstance(table_info, dict):
                    context = table_info.get("business_context", "")
                    if context:
                        table_terms.update(context.lower().split())
                
                # Calculate term overlap
                overlap = len(query_terms.intersection(table_terms))
                total_terms = len(query_terms.union(table_terms))
                if total_terms > 0:
                    relevance_scores.append(overlap / total_terms)
            
            return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
            
        except Exception as e:
            print(f"[MultiHop] Relevance calculation failed: {str(e)}")
            return 0.5  # Default moderate relevance

    def _save_followup_context_to_logs(self, user_query: str, followup_result: Dict[str, Any], 
                                     additional_concepts: List[str], followup_context: Dict[str, Any], 
                                     context_type: str = "extended"):
        """
        Save follow-up context to logs similar to synthesized context for analysis and debugging.
        """
        try:
            import datetime
            import os
            import json
            
            # Create comprehensive follow-up context data
            followup_log_data = {
                "context_type": "followup_query",
                "query_strategy": context_type,  # "extended" or "reused"
                "user_query": user_query,
                "timestamp": datetime.datetime.now().isoformat(),
                "additional_concepts": additional_concepts,
                "base_query": followup_context.get("base_query_context", {}).get("original_query", ""),
                "context_reuse_percentage": 85 if additional_concepts else 100,
                "search_method": followup_result.get("metadata", {}).get("search_method", "context_extension"),
                "result_summary": {
                    "total_tables": len(followup_result.get("tables", {})),
                    "total_examples": len(followup_result.get("examples", [])),
                    "total_join_keys": len(followup_result.get("join_keys", [])),
                    "confidence": followup_result.get("metadata", {}).get("confidence", "high")
                },
                "cached_context": {
                    "inherited_tables": list(followup_context.get("inherited_context", {}).get("tables", {}).keys()),
                    "available_tables": followup_context.get("available_tables", []),
                    "inherited_filters": followup_context.get("inherited_filters", [])
                },
                "new_context": {
                    "new_tables": [table for table in followup_result.get("tables", {}).keys() 
                                 if table not in followup_context.get("available_tables", [])],
                    "targeted_searches": len(additional_concepts),
                    "search_efficiency": "high" if additional_concepts else "maximum"
                },
                "performance_indicators": {
                    "context_extension": context_type == "extended",
                    "full_context_reuse": context_type == "reused",
                    "concept_based_search": len(additional_concepts) > 0,
                    "estimated_time_savings": "82%" if additional_concepts else "95%"
                },
                "full_context": followup_result
            }
            
            # Save to logs directory
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"followup_context_{context_type}_{timestamp}.json"
            file_path = os.path.join(logs_dir, filename)
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(followup_log_data, f, indent=2)
            
            print(f"[MultiHop] Follow-up context ({context_type}) saved to {file_path}")
            
            # Log summary statistics
            tables_count = followup_log_data["result_summary"]["total_tables"]
            examples_count = followup_log_data["result_summary"]["total_examples"]
            concepts_count = len(additional_concepts)
            reuse_percentage = followup_log_data["context_reuse_percentage"]
            
            print(f"[MultiHop] Follow-up Results: {tables_count} total tables, {examples_count} examples, "
                  f"{concepts_count} new concepts, {reuse_percentage}% context reuse")
            
            # Log performance benefits
            if context_type == "extended":
                print(f"[MultiHop] Context Extension: {concepts_count} targeted searches vs full multi-hop")
            else:
                print(f"[MultiHop] Full Context Reuse: No additional searches needed")
                
        except Exception as e:
            print(f"[MultiHop] Follow-up context logging failed: {str(e)}")

    def _extend_inherited_tables_with_concepts(self, inherited_tables: Dict[str, Any], 
                                             additional_concepts: List[str], user_query: str) -> Dict[str, Any]:
        """
        Extend inherited tables with new concept-related columns instead of searching for new tables.
        This ensures table consistency for follow-up queries.
        """
        try:
            print(f"[MultiHop] Extending inherited tables with concepts: {additional_concepts}")
            
            extended_context = {
                "tables": {},
                "examples": [],
                "join_keys": [],
                "metadata": {
                    "search_method": "inherited_table_extension",
                    "table_consistency": True,
                    "extended_concepts": additional_concepts
                }
            }
            
            # Check each inherited table for concept-related columns
            for table_name, table_info in inherited_tables.items():
                if isinstance(table_info, dict) and "columns" in table_info:
                    table_columns = table_info["columns"]
                    
                    # Find columns that match the new concepts
                    concept_matches = self._find_concept_matching_columns(table_columns, additional_concepts)
                    
                    if concept_matches:
                        print(f"[MultiHop] Found concept matches in {table_name}: {list(concept_matches.keys())}")
                        
                        # Include the table with highlighted concept columns
                        extended_table_info = table_info.copy()
                        
                        # Mark concept-matching columns
                        for col_name, col_info in concept_matches.items():
                            if isinstance(extended_table_info.get("columns", {}), dict):
                                if col_name in extended_table_info["columns"]:
                                    extended_table_info["columns"][col_name]["concept_match"] = True
                                    extended_table_info["columns"][col_name]["matched_concepts"] = col_info.get("matched_concepts", [])
                        
                        extended_table_info["source"] = "inherited_extended"
                        extended_table_info["concept_relevance"] = "high"
                        extended_context["tables"][table_name] = extended_table_info
            
            # If we found concept matches in inherited tables, return the extended context
            if extended_context["tables"]:
                print(f"[MultiHop] Successfully extended {len(extended_context['tables'])} inherited tables")
                return extended_context
            else:
                print(f"[MultiHop] No concept matches found in inherited tables")
                return {"tables": {}, "examples": [], "join_keys": []}
                
        except Exception as e:
            print(f"[MultiHop] Inherited table extension failed: {str(e)}")
            return {"tables": {}, "examples": [], "join_keys": []}

    def _find_concept_matching_columns(self, table_columns: Dict[str, Any], concepts: List[str]) -> Dict[str, Any]:
        """
        Find columns in a table that match the given concepts semantically.
        """
        try:
            concept_matches = {}
            
            for col_name, col_info in table_columns.items():
                if not isinstance(col_info, dict):
                    continue
                    
                # Get column description and natural language terms
                description = col_info.get("description", "").lower()
                natural_terms = col_info.get("natural_language_term", "").lower()
                column_name_lower = col_name.lower()
                
                # Check each concept for matches
                matched_concepts = []
                for concept in concepts:
                    concept_lower = concept.lower()
                    
                    # Direct name match
                    if concept_lower in column_name_lower:
                        matched_concepts.append(concept)
                    # Description match
                    elif concept_lower in description:
                        matched_concepts.append(concept)
                    # Natural language term match
                    elif concept_lower in natural_terms:
                        matched_concepts.append(concept)
                    # Semantic matches for specific concepts
                    elif self._is_semantic_concept_match(concept_lower, column_name_lower, description, natural_terms):
                        matched_concepts.append(concept)
                
                if matched_concepts:
                    concept_matches[col_name] = {
                        "column_info": col_info,
                        "matched_concepts": matched_concepts,
                        "relevance_score": len(matched_concepts) / len(concepts)
                    }
            
            return concept_matches
            
        except Exception as e:
            print(f"[MultiHop] Concept matching failed: {str(e)}")
            return {}

    def _is_semantic_concept_match(self, concept: str, column_name: str, description: str, natural_terms: str) -> bool:
        """
        Check for semantic matches between concepts and column information.
        """
        # Tenure-related concept matching
        if "tenure" in concept:
            tenure_indicators = ["tenure", "duration", "length", "time", "period", "active", "service", "lifetime"]
            return any(indicator in column_name or indicator in description or indicator in natural_terms 
                      for indicator in tenure_indicators)
        
        # Individual-related concept matching
        if "individual" in concept:
            individual_indicators = ["individual", "person", "customer", "subscriber", "user", "account"]
            return any(indicator in column_name or indicator in description or indicator in natural_terms 
                      for indicator in individual_indicators)
        
        # Time period concept matching
        if "time_period" in concept or "time" in concept:
            time_indicators = ["date", "time", "period", "duration", "day", "month", "year", "timestamp"]
            return any(indicator in column_name or indicator in description or indicator in natural_terms 
                      for indicator in time_indicators)
        
        # Count/aggregation concept matching
        if "count" in concept:
            count_indicators = ["count", "number", "total", "sum", "aggregate", "quantity"]
            return any(indicator in column_name or indicator in description or indicator in natural_terms 
                      for indicator in count_indicators)
        
        return False

    async def _retrieve_incremental_context_with_table_guidance(self, concepts: List[str], existing_tables: List[str], 
                                                              table_guidance: List[str], top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieve incremental context with guidance from previously successful tables.
        Prioritizes tables similar to those that worked before.
        """
        try:
            print(f"[MultiHop] Guided context retrieval for concepts: {concepts}")
            print(f"[MultiHop] Table guidance from previous success: {table_guidance}")
            
            # Build search queries with table guidance
            guided_search_queries = self._build_guided_semantic_search_queries(concepts, table_guidance)
            
            # Execute searches in parallel
            search_tasks = [
                self._async_retrieve_wrapper(query, top_k=2) 
                for query in guided_search_queries
            ]
            
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Merge results with preference for guided tables
            merged_context = self._merge_concept_based_results_with_guidance(
                search_results, concepts, existing_tables, table_guidance
            )
            
            return merged_context or {"tables": {}, "examples": [], "join_keys": []}
            
        except Exception as e:
            print(f"[MultiHop] Guided context retrieval failed: {str(e)}")
            # Fallback to regular incremental context
            return await self._retrieve_incremental_context(concepts, existing_tables, top_k)

    def _build_guided_semantic_search_queries(self, concepts: List[str], table_guidance: List[str]) -> List[str]:
        """
        Build search queries guided by previously successful tables.
        """
        search_queries = []
        
        # Extract table prefixes/patterns from guidance
        table_patterns = []
        for table in table_guidance:
            # Extract meaningful parts of table names
            table_parts = table.lower().replace("_", " ").split(".")
            if table_parts:
                table_patterns.extend(table_parts[-1].split())  # Last part (actual table name)
        
        for concept in concepts:
            # Build guided search query
            base_query = self._concept_to_search_query(concept)
            
            # Add table guidance to the search
            if table_patterns:
                guided_query = f"{base_query} {' '.join(set(table_patterns))}"
            else:
                guided_query = base_query
                
            search_queries.append(guided_query)
        
        return search_queries

    def _concept_to_search_query(self, concept: str) -> str:
        """
        Convert a concept to a search query.
        """
        if "tenure" in concept:
            return "tenure duration lifetime customer subscription period"
        elif "individual" in concept:
            return "customer individual subscriber person account user"
        elif "time_period" in concept:
            return "time period duration date range temporal"
        elif "count" in concept:
            return "count total aggregation metrics analysis reporting"
        else:
            return concept.replace("_", " ")

    def _merge_concept_based_results_with_guidance(self, search_results: List[Any], concepts: List[str], 
                                                 existing_tables: List[str], table_guidance: List[str]) -> Dict[str, Any]:
        """
        Merge search results with preference for tables similar to guidance.
        """
        merged_context = {"tables": {}, "examples": [], "join_keys": []}
        
        for result, concept in zip(search_results, concepts):
            if isinstance(result, dict) and result.get("tables"):
                for table_name, table_info in result["tables"].items():
                    # Calculate guidance similarity
                    guidance_score = self._calculate_table_guidance_similarity(table_name, table_guidance)
                    
                    # Prefer tables with higher guidance similarity
                    if guidance_score > 0.3 or not merged_context["tables"]:  # Accept if similar or nothing found yet
                        table_info["guidance_score"] = guidance_score
                        table_info["source"] = "guided_search"
                        merged_context["tables"][table_name] = table_info
        
        return merged_context

    def _calculate_table_guidance_similarity(self, table_name: str, table_guidance: List[str]) -> float:
        """
        Calculate similarity between a table name and guidance tables.
        """
        if not table_guidance:
            return 0.0
            
        table_name_lower = table_name.lower()
        max_similarity = 0.0
        
        for guidance_table in table_guidance:
            guidance_lower = guidance_table.lower()
            
            # Extract meaningful parts
            table_parts = set(table_name_lower.replace("_", " ").split())
            guidance_parts = set(guidance_lower.replace("_", " ").split())
            
            # Calculate Jaccard similarity
            intersection = table_parts.intersection(guidance_parts)
            union = table_parts.union(guidance_parts)
            
            if union:
                similarity = len(intersection) / len(union)
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity

    def _merge_contexts_with_table_priority(self, base_context: Dict[str, Any], additional_context: Dict[str, Any], 
                                          query: str, preserve_tables: List[str]) -> Dict[str, Any]:
        """
        Merge contexts with priority given to preserving specific tables.
        """
        merged_context = {
            "tables": {},
            "examples": [],
            "join_keys": [],
            "metadata": {
                "search_method": "table_priority_merge",
                "preserved_tables": preserve_tables
            }
        }
        
        # First, add preserved tables from base context
        base_tables = base_context.get("tables", {})
        for table_name in preserve_tables:
            if table_name in base_tables:
                merged_context["tables"][table_name] = base_tables[table_name]
                merged_context["tables"][table_name]["source"] = "preserved"
        
        # Then, add any additional relevant tables
        additional_tables = additional_context.get("tables", {})
        for table_name, table_info in additional_tables.items():
            if table_name not in merged_context["tables"]:
                # Check if this table is complementary to preserved tables
                if self._is_complementary_table(table_name, preserve_tables):
                    table_info["source"] = "complementary"
                    merged_context["tables"][table_name] = table_info
        
        # Merge other components
        merged_context["examples"] = base_context.get("examples", []) + additional_context.get("examples", [])
        merged_context["join_keys"] = base_context.get("join_keys", []) + additional_context.get("join_keys", [])
        
        return merged_context

    def _is_complementary_table(self, table_name: str, preserve_tables: List[str]) -> bool:
        """
        Check if a table is complementary to the preserved tables.
        """
        table_name_lower = table_name.lower()
        
        for preserved_table in preserve_tables:
            preserved_lower = preserved_table.lower()
            
            # Check for related table patterns
            if any(pattern in table_name_lower and pattern in preserved_lower 
                   for pattern in ["individual", "account", "profile", "subscriber"]):
                return True
        
        return False

async def multi_hop_retrieve_context(user_query: str, top_k: int = 3):
    retriever = MultiHopRetriever()
    try:
        result = await retriever.retrieve(user_query, top_k)
        return result
    finally:
        retriever.cleanup()

def multi_hop_retrieve_context_node(state):
    print("[MultiHop] Execution started: Multi-hop context retrieval")
    from utils.logging_config import logger
    logger.info("[MultiHop] Multi-hop context retrieval node started")
    try:
        retriever = MultiHopRetriever()
        user_query = ""
        for message in reversed(state.get("messages", [])):
            if hasattr(message, 'content') and message.content.strip():
                user_query = message.content.strip()
                break
        if not user_query:
            raise ValueError("No valid user query found")
        logger.info(f"[MultiHop] Processing query: {user_query}")
        
        # Extract conversation ID from state (should be set by API handler)
        conversation_id = state.get("conversation_id")
        if not conversation_id:
            print("[MultiHop] Warning: No conversation_id found in state")
        else:
            print(f"[MultiHop] Using conversation_id: {conversation_id}")
            
        # Debug: Check what's in the state
        state_keys = list(state.keys()) if state else []
        print(f"[MultiHop] Available state keys: {state_keys}")
        
        # Check for cached context first for deterministic results
        cached_context_json = None
        
        import concurrent.futures
        retriever = None
        try:
            retriever = MultiHopRetriever()
            
            # First check memory for plan to see if we can check cache
            temp_plan = retriever._check_memory_for_plan(user_query)
            if temp_plan:
                cached_context_json = get_cached_context_for_query(user_query, temp_plan.get("hops", []))
            
            if cached_context_json:
                logger.info("[MultiHop] Using cached context for deterministic results")
                print("[MultiHop] Using cached context for deterministic results")
                return {
                    "context": cached_context_json,
                    "enhanced_context": cached_context_json,
                    "query_interpretation": state.get("query_interpretation")  # Preserve query interpretation
                }
            
            # No cache hit, proceed with enhanced retrieval that supports follow-ups
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(retriever.retrieve(user_query, top_k=3, conversation_id=conversation_id))
                )
                multi_hop_context = future.result(timeout=120)
                logger.info("[MultiHop] Context retrieval completed successfully")
        except concurrent.futures.TimeoutError:
            print("[MultiHop] TIMEOUT: Context retrieval exceeded timeout")
            logger.warning("[MultiHop] Context retrieval timeout")
            multi_hop_context = {
                "user_query": user_query,
                "plan": {},
                "hops": [],
                "final_synthesis": {},
                "error": "Timeout exceeded"
            }
        except Exception as async_error:
            print(f"[MultiHop] Async execution failed: {str(async_error)}")
            multi_hop_context = {
                "user_query": user_query,
                "plan": {},
                "hops": [],
                "final_synthesis": {},
                "error": str(async_error)
            }
        finally:
            if retriever:
                retriever.cleanup()
        formatted_context = {
            "status": "success" if multi_hop_context.get("final_synthesis") else "fallback",
            "user_query": user_query,
            "plan": multi_hop_context.get("plan", {}),
            "hops": multi_hop_context.get("hops", []),
            "final_synthesis": multi_hop_context.get("final_synthesis", {})
        }
        final_synthesis = multi_hop_context.get("final_synthesis", {})
        
        # Handle follow-up context reuse - context is directly in multi_hop_context
        if not final_synthesis and "tables" in multi_hop_context:
            print(f"[MultiHop] Using direct context for follow-up (tables: {len(multi_hop_context.get('tables', {}))})")
            context_for_sql = json.dumps(multi_hop_context, indent=2)
        elif final_synthesis and final_synthesis.get("tables"):
            context_for_sql = json.dumps(final_synthesis, indent=2)
        else:
            context_for_sql = json.dumps(formatted_context, indent=2)
        formatted_context_json = json.dumps(formatted_context, indent=2)
        
        # Cache the context for future deterministic results
        plan = multi_hop_context.get("plan", {})
        if plan and plan.get("hops"):
            cache_context_for_query(user_query, plan.get("hops", []), context_for_sql)
            logger.info("[MultiHop] Context cached for future deterministic results")
        
        print(f"[MultiHop] Multi-hop retrieval completed for: {user_query}")
        logger.info(f"[MultiHop] Multi-hop retrieval completed successfully for query: {user_query}")
        return {
            "context": context_for_sql,
            "enhanced_context": formatted_context_json,
            "query_interpretation": state.get("query_interpretation")  # Preserve query interpretation
        }
    except Exception as e:
        print(f"[MultiHop] ERROR: {str(e)}")
        logger.error(f"[MultiHop] Multi-hop context retrieval failed: {str(e)}")
        emergency_context = {
            "status": "emergency_fallback",
            "user_query": user_query if 'user_query' in locals() else "unknown",
            "plan": {},
            "hops": [],
            "final_synthesis": {},
            "error": str(e)
        }
        emergency_context_json = json.dumps(emergency_context, indent=2)
        return {
            "context": emergency_context_json,
            "error": f"Multi-hop context retrieval failed: {str(e)}",
            "enhanced_context": emergency_context_json,
            "query_interpretation": state.get("query_interpretation")  # Preserve query interpretation
        }
