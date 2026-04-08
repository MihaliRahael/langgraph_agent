import asyncio
import os
import requests
import json
from typing import Dict, List, Any, Optional
from openai import AzureOpenAI
import re
import certifi
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# SSL Bypass configuration
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
model_version = os.getenv("MODEL")
# Refer below class for multi hop retrieval 
class HybridSearchRetriever:
    async def _generate_single_embedding(self, text: str, prompt: str = "") -> List[float]:
        """
        Generate a single embedding using Azure OpenAI, with retries and validation.
        """
        max_retries = 2
        for attempt in range(max_retries):
            try:
                input_text = f"{prompt}\n\n{text}" if prompt else text
                response = self.openai_client.embeddings.create(
                    input=[input_text],
                    model=model_version  # Updated for consistency and better performance
                )
                if response and response.data and len(response.data) > 0:
                    embedding = response.data[0].embedding
                    # Ensure the embedding is a list of floats
                    if not isinstance(embedding, list):
                        if hasattr(embedding, 'tolist'):
                            embedding = embedding.tolist()
                        else:
                            raise ValueError("Embedding is not a list and cannot be converted.")
                    validated_embedding = []
                    for x in embedding:
                        try:
                            validated_embedding.append(float(x))
                        except (ValueError, TypeError):
                            validated_embedding.append(0.0)
                    return validated_embedding
                else:
                    raise ValueError("Invalid embedding response")
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[HybridSearch] Embedding retry {attempt+1}/{max_retries}: {str(e)}")
                    await asyncio.sleep(1)
                else:
                    raise
    """
    HybridSearchRetriever: Streamlined hybrid search retrieval system with multi-vector support
    """
    def __init__(self):
        import hashlib
        self.search_service_name = AZURE_SEARCH_SERVICE_NAME
        self.admin_key = AZURE_SEARCH_KEY
        self.index_name = MULTI_VECTOR_INDEX
        self.api_version = "2024-07-01"
        self.openai_client = AzureOpenAI(
            api_version=OPENAI_API_VERSION,
            azure_endpoint=OPENAI_ENDPOINT,
            api_key=OPENAI_API_KEY
        )
        self.base_url = f"https://{self.search_service_name}.search.windows.net"
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.admin_key
        }
        
        # Enhancement cache for deterministic query enhancement
        self._enhance_cache = {}
        
        # Request ID generator for stable request identification
        self._reqid = lambda q: hashlib.sha1(self._norm(q).encode()).hexdigest()
        
        # Static vector weights for deterministic scoring
        self._static_vector_weights = {
            "business_vector": 0.3,
            "technical_vector": 0.3,
            "semantic_vector": 0.4
        }
        self.database_name = os.getenv("SNOWFLAKE_DATABASE")
        self.schema_name = os.getenv("SNOWFLAKE_SCHEMA")
        self.scoring_profile = "relevance-boost"
        self.semantic_config = "enhanced-semantic"
    
    def _norm(self, s: str) -> str:
        """Normalize query string for consistent caching and comparison"""
        return " ".join(s.lower().split())

    async def retrieve(self, query: str, top_k: int = 2) -> Dict[str, Any]:
        print(f"[HybridSearch] Retrieving top {top_k} results for: '{query}'")
        try:
            embeddings = await self._generate_embeddings(query)
            search_results = await self._execute_hybrid_search(query, embeddings, top_k)
            formatted_results = self._format_results(search_results, query)
            print(f"[HybridSearch] Retrieved {len(formatted_results['tables'])} relevant tables")
            return formatted_results
        except Exception as e:
            print(f"[HybridSearch] Error during retrieval: {str(e)}")
            return {
                "query": query,
                "tables": {},
                "join_keys": [],
                "examples": [],
                "metadata": {
                    "result_count": 0,
                    "confidence": "none",
                    "search_method": "error",
                    "error": str(e)
                }
            }

    async def _generate_embeddings(self, query: str) -> Dict[str, List[float]]:
        embeddings = {}
        vector_types = [
            ("content_vector", "Generate an embedding for general content search"),
            ("semantic_vector", "Generate an embedding focused on semantic meaning and intent"),
            ("business_vector", "Generate an embedding focused on business concepts and metrics"),
            ("technical_vector", "Generate an embedding focused on database tables and columns")
        ]
        tasks = [self._generate_single_embedding(query, prompt) for vector_type, prompt in vector_types]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            vector_type = vector_types[i][0]
            if not isinstance(result, Exception):
                embeddings[vector_type] = result
        if not embeddings.get("content_vector") and embeddings:
            for vector_type, embedding in embeddings.items():
                embeddings["content_vector"] = embedding
                break
        print(f"[HybridSearch] Generated {len(embeddings)} vector embeddings")
    # No vectorQueries logic here; all vectorQueries logic is in _execute_hybrid_search

    async def _execute_hybrid_search(self, query: str, embeddings: Dict[str, List[float]], top_k: int) -> Dict[str, Any]:
        search_url = f"{self.base_url}/indexes/{self.index_name}/docs/search?api-version={self.api_version}"
        enhanced_query = await self._enhance_query(query)
        
        search_payload = {
            "search": enhanced_query,
            "queryType": "semantic",
            "semanticConfiguration": self.semantic_config,
            "scoringProfile": self.scoring_profile,
            "searchFields": "table,columns_text,content_combined,content_semantic,business_context_text",
            "select": "*",
            "top": top_k,       # Use top_k directly for deterministic results
            "count": True
        }
        
        # Build vector queries deterministically if embeddings exist
        if embeddings:
            vector_queries = []
            vector_weights = self._calculate_vector_weights(query)
            for vector_field, embedding in embeddings.items():
                if vector_field in vector_weights and embedding:
                    # Ensure vector values are properly formatted
                    vector_values = [float(v) if isinstance(v, (int, float)) else 0.0 for v in embedding]
                    
                    if len(vector_values) > 0:
                        vector_queries.append({
                            "kind": "vector",
                            "vector": vector_values,
                            "field": vector_field,
                            "k": top_k,                         # Use top_k consistently
                            "weight": float(vector_weights[vector_field])
                        })
            if vector_queries:
                search_payload["vectorQueries"] = vector_queries
        
        try:
            log_payload = self._get_loggable_payload(search_payload)
            print(f"[HybridSearch] Executing hybrid search: {json.dumps(log_payload, indent=2)}")
            
            # Add deterministic request ID header
            headers = {**self.headers, "x-ms-client-request-id": self._reqid(enhanced_query)}
            
            response = requests.post(
                search_url,
                headers=headers,
                json=search_payload,
                timeout=15,
                verify=False  # Keep False for dev, change to True in production
            )
            if response.status_code == 200:
                result = response.json()
                result_count = len(result.get('value', []))
                print(f"[HybridSearch] Hybrid search returned {result_count} results")
                if result_count > 0:
                    return self._post_process_results(result, query, top_k)
            else:
                print(f"[HybridSearch] Hybrid search failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"[HybridSearch] Hybrid search error: {str(e)}")
        # Fallback: semantic search only
        try:
            semantic_payload = {k: v for k, v in search_payload.items() if k != "vectorQueries"}
            response = requests.post(
                search_url,
                headers=self.headers,
                json=semantic_payload,
                timeout=15,
                verify=False
            )
            if response.status_code == 200:
                result = response.json()
                print(f"[HybridSearch] Semantic search returned {len(result.get('value', []))} results")
                return self._post_process_results(result, query, top_k)
            else:
                print(f"[HybridSearch] Semantic search failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"[HybridSearch] Semantic search error: {str(e)}")
        # Last resort: return empty
        return {"value": []}

    def _get_loggable_payload(self, payload: Dict) -> Dict:
        loggable = payload.copy()
        if "vectorQueries" in loggable:
            for vq in loggable["vectorQueries"]:
                if "vector" in vq:
                    vector_length = len(vq["vector"]) if isinstance(vq["vector"], list) else "unknown"
                    vq["vector"] = f"[{vector_length} dimensions]"
        return loggable

    def _calculate_vector_weights(self, query: str) -> Dict[str, float]:
        """Calculate deterministic vector weights for consistent scoring"""
        # Use static base weights for consistency
        weights = self._static_vector_weights.copy()
        
        # Add minimal, deterministic adjustments based on query patterns
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["count", "how many", "total", "average"]):
            weights["business_vector"] = min(weights["business_vector"] + 0.3, 1.0)
        if any(term in query_lower for term in ["table", "column", "schema", "database"]):
            weights["technical_vector"] = min(weights["technical_vector"] + 0.4, 1.0)
        if any(term in query_lower for term in ["customer", "account", "user", "subscription"]):
            weights["business_vector"] = min(weights["business_vector"] + 0.2, 1.0)
            weights["semantic_vector"] = min(weights["semantic_vector"] + 0.1, 1.0)
        if any(term in query_lower for term in ["join", "relate", "connection"]):
            weights["technical_vector"] = min(weights["technical_vector"] + 0.3, 1.0)
        
        # Ensure all weights are capped at 1.0 for consistency
        return {k: min(v, 1.0) for k, v in weights.items()}

    async def _enhance_query(self, query: str) -> str:
        """Enhanced query with caching for deterministic results"""
        key = self._norm(query)
        if key in self._enhance_cache:
            return self._enhance_cache[key]
        
        try:
            enhancement_prompt = f"""Enhance this query for database schema search with relevant terms.
QUERY: {query}
Return ONLY the enhanced query text."""
            
            response = self.openai_client.chat.completions.create(
                model=OPENAI_DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": enhancement_prompt}],
                max_tokens=80,
                temperature=0.0,        # deterministic
                top_p=0.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            enhanced = response.choices[0].message.content.strip().strip('"\'')
            self._enhance_cache[key] = enhanced
            print(f"[HybridSearch] Enhanced query (cached): '{enhanced}'")
            return enhanced
        except Exception as e:
            print(f"[HybridSearch] Query enhancement failed: {str(e)}")
            self._enhance_cache[key] = query
            return query

    def _post_process_results(self, results: Dict[str, Any], query: str, top_k: int) -> Dict[str, Any]:
        items = results.get("value", [])
        if not items:
            return results
            
        scored_items = []
        query_terms = query.lower().split()
        for item in items:
            base_score = item.get("@search.score", 0)
            relevance_score = self._calculate_relevance_score(item, query_terms, query)
            item["@hybrid_score"] = base_score + relevance_score
            scored_items.append(item)
        
        # Stable ranking with tie-breaker for deterministic ordering
        scored_items.sort(
            key=lambda x: (
                -(x.get("@hybrid_score", x.get("@search.score", 0)) or 0),
                x.get("table", "").lower()   # tie-breaker
            )
        )
        
        results["value"] = scored_items[:top_k]
        return results

    def _calculate_relevance_score(self, item: Dict[str, Any], query_terms: List[str], query: str) -> float:
        """
        Calculates a relevance score for a search result item based on its overlap with the query.
        If business entities or other metadata are not stored in the index, this function will
        only use available fields (e.g., table name, columns, example SQL, captions, highlights).
        """
        relevance = 0.0
        query_lower = query.lower()
        table_name = item.get("table", "").lower()
        table_terms = table_name.split('_')
        matches = set(query_terms).intersection(set(table_terms))
        relevance += len(matches) * 0.5

        # If business entities are not stored, skip this section
        # (No business_entities dict or lookup)

        columns = item.get("columns", [])
        if isinstance(columns, list):
            for col in columns:
                if isinstance(col, dict):
                    col_name = col.get("column_name", "").lower()
                    if any(term in col_name for term in query_terms):
                        relevance += 0.5
                    if "id" in col_name and "id" in query_lower:
                        relevance += 0.5
        examples = item.get("example_sql_queries", [])
        if isinstance(examples, list):
            for example in examples:
                if isinstance(example, dict):
                    sql = example.get("sql", "").lower()
                    questions = example.get("questions", [])
                    if "count" in query_lower and "count" in sql:
                        relevance += 0.7
                    if "group by" in sql and any(term in query_lower for term in ["group", "by"]):
                        relevance += 0.6
                    for question in questions:
                        if isinstance(question, str) and any(term in question.lower() for term in query_terms):
                            relevance += 0.5
                            break
        if "@search.captions" in item:
            captions = item.get("@search.captions", [])
            relevance += min(len(captions) * 0.3, 1.0)
        business_context = item.get("business_context_text", "").lower()
        if business_context and any(term in business_context for term in query_terms):
            relevance += 0.7
        if "@search.highlights" in item and item["@search.highlights"]:
            relevance += 0.8
        return relevance

    def _format_results(self, search_results: Dict[str, Any], query: str) -> Dict[str, Any]:
        results = search_results.get("value", [])
        response = {
            "query": query,
            "tables": {},
            "join_keys": [],
            "examples": [],
            "metadata": {
                "result_count": len(results),
                "confidence": self._get_confidence_level(results),
                "search_method": "hybrid"
            }
        }
        all_join_keys = set()
        for result in results:
            table_name = result.get("table", "")
            if not table_name:
                continue
            fully_qualified_name = f"{self.database_name}.{self.schema_name}.{table_name}" if self.database_name and self.schema_name else table_name
            columns = {}
            column_data = result.get("columns", [])
            if isinstance(column_data, list):
                for col in column_data:
                    if isinstance(col, dict) and "column_name" in col and "data_type" in col:
                        column_name = col.get("column_name")
                        columns[column_name] = {
                            "data_type": col.get("data_type", ""),
                            "description": col.get("description", ""),
                            "natural_language_term": col.get("natural_language_term", "")
                        }
            join_keys = result.get("join_keys", [])
            if isinstance(join_keys, list):
                for key in join_keys:
                    if key:
                        all_join_keys.add(key)
            table_examples = []
            examples_data = result.get("example_sql_queries", [])
            if isinstance(examples_data, list):
                for example in examples_data:
                    if isinstance(example, dict) and "sql" in example:
                        sql = example.get("sql", "")
                        sql = self._update_sql_with_qualified_names(sql, table_name, fully_qualified_name)
                        if sql:
                            table_examples.append(sql)
            high_priority_sql = result.get("high_priority_sql", "")
            if high_priority_sql:
                high_priority_sql = self._update_sql_with_qualified_names(
                    high_priority_sql, table_name, fully_qualified_name
                )
                if high_priority_sql:
                    table_examples.insert(0, high_priority_sql)
            response["tables"][fully_qualified_name] = {
                "columns": columns,
                "examples": table_examples[:2],
                "relevance_score": result.get("@hybrid_score", result.get("@search.score", 0)),
                "business_context": result.get("business_context_text", "")
            }
            response["examples"].extend(table_examples[:1])
        response["join_keys"] = list(all_join_keys)
        response["examples"] = response["examples"][:3]
        return response

    def _update_sql_with_qualified_names(self, sql_query: str, original_table_name: str, 
                                        fully_qualified_name: str) -> str:
        if not sql_query or not original_table_name:
            return sql_query
        patterns = [
            (rf'\bFROM\s+{re.escape(original_table_name)}\b', f'FROM {fully_qualified_name}'),
            (rf'\bJOIN\s+{re.escape(original_table_name)}\b', f'JOIN {fully_qualified_name}')
        ]
        updated_query = sql_query
        for pattern, replacement in patterns:
            updated_query = re.sub(pattern, replacement, updated_query, flags=re.IGNORECASE)
        return updated_query

    def _get_confidence_level(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "none"
        avg_score = sum(result.get("@hybrid_score", result.get("@search.score", 0)) 
                        for result in results) / len(results)
        if avg_score >= 10.0:
            return "very_high"
        elif avg_score >= 7.0:
            return "high"
        elif avg_score >= 4.0:
            return "medium"
        elif avg_score > 0.0:
            return "low"
        else:
            return "very_low"

async def retrieve_hybrid_context(query: str, top_k: int = 2):
    retriever = HybridSearchRetriever()
    return await retriever.retrieve(query, top_k)

def hybrid_retrieve_context_node(state):
    print("[HybridSearch] Execution started: Hybrid context retrieval")
    try:
        retriever = HybridSearchRetriever()
        user_query = ""
        for message in reversed(state.get("messages", [])):
            if hasattr(message, 'content') and message.content.strip():
                user_query = message.content.strip()
                break
        if not user_query:
            raise ValueError("No valid user query found")
        import concurrent.futures
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(retriever.retrieve(user_query, top_k=2))
                )
                hybrid_context = future.result(timeout=45)
        except concurrent.futures.TimeoutError:
            print("[HybridSearch] TIMEOUT: Context retrieval exceeded timeout")
            hybrid_context = {
                "query": user_query,
                "tables": {},
                "join_keys": [],
                "examples": [],
                "metadata": {
                    "result_count": 0,
                    "confidence": "none",
                    "search_method": "timeout",
                    "error": "Timeout exceeded"
                }
            }
        except Exception as async_error:
            print(f"[HybridSearch] Async execution failed: {str(async_error)}")
            hybrid_context = {
                "query": user_query,
                "tables": {},
                "join_keys": [],
                "examples": [],
                "metadata": {
                    "result_count": 0,
                    "confidence": "none",
                    "search_method": "error",
                    "error": str(async_error)
                }
            }
        formatted_context = {
            "status": "success" if hybrid_context["metadata"]["confidence"] not in ["none", "timeout", "error"] else "fallback",
            "user_query": user_query,
            "total_tables_found": len(hybrid_context["tables"]),
            "relevant_tables_count": len(hybrid_context["tables"]),
            "schema_context": {
                "tables": {},
                "relationships": {
                    "join_keys": hybrid_context["join_keys"]
                },
                "examples": {
                    "most_relevant": hybrid_context["examples"]
                }
            },
            "query_analysis": {
                "detected_patterns": {
                    "count_aggregation": "count" in user_query.lower() or "how many" in user_query.lower(),
                    "filter_condition": "where" in user_query.lower() or "filter" in user_query.lower(),
                    "autopay_related": "autopay" in user_query.lower()
                },
                "relevant_table_names": list(hybrid_context["tables"].keys())
            },
            "metadata": {
                "retrieval_timestamp": "",
                "search_confidence": hybrid_context["metadata"]["confidence"],
                "search_method": hybrid_context["metadata"]["search_method"]
            }
        }
        for table_name, table_info in hybrid_context["tables"].items():
            column_list = list(table_info["columns"].keys())
            column_types = {col_name: col_info["data_type"] for col_name, col_info in table_info["columns"].items()}
            formatted_context["schema_context"]["tables"][table_name] = {
                "columns": column_list,
                "schema": table_name,
                "examples": table_info["examples"],
                "column_types": column_types
            }
        import json
        formatted_context_json = json.dumps(formatted_context, indent=2)
        print(f"[HybridSearch] Retrieved {len(hybrid_context['tables'])} relevant tables")
        return {
            "context": formatted_context_json,
            "enhanced_context": formatted_context_json
        }
    except Exception as e:
        print(f"[HybridSearch] ERROR: {str(e)}")
        emergency_context = {
            "status": "emergency_fallback",
            "user_query": user_query if 'user_query' in locals() else "unknown",
            "total_tables_found": 0,
            "relevant_tables_count": 0,
            "error": str(e),
            "schema_context": {
                "tables": {},
                "relationships": {"join_keys": []},
                "examples": {"most_relevant": []}
            }
        }
        import json
        emergency_context_json = json.dumps(emergency_context, indent=2)
        return {
            "context": emergency_context_json,
            "error": f"Context retrieval failed: {str(e)}",
            "enhanced_context": emergency_context_json
        }
