"""
Module 2 - Inference Pipeline Nodes.

This module implements the runtime inference system for hierarchical
classification with:

1. Label-based retrieval for initial candidate recall
2. Multi-view scoring with on-demand embedding_effective computation
3. Top-down routing with asymmetric stopping (parent veto)
4. Explainability at each decision point

Spec Reference: Module 2 — Inference
"""

from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ============================================================================
# Helper: Missing embedding detection (avoid ambiguous numpy truth values)
# ============================================================================


def _is_missing_embedding(value: Any) -> bool:
    """Return True when an embedding cell is missing/NA (but not a vector)."""
    if value is None:
        return True
    if np.isscalar(value):
        return bool(pd.isna(value))
    return False


# ============================================================================
# Node 0: Load Queries (Batch Support)
# ============================================================================


def load_queries(query_input: Union[str, List[str], pd.DataFrame]) -> pd.DataFrame:
    """
    Load and normalize query input into DataFrame format.

    Supports three input formats:
    1. Single string: "I work as a teacher"
    2. List of strings: ["query 1", "query 2", ...]
    3. DataFrame: Already formatted with 'text' column

    Args:
        query_input: Query text(s) in any supported format

    Returns:
        DataFrame with columns:
        - query_id: int (0-indexed)
        - text: str (query text)

    Example:
        >>> load_queries("teacher")
        DataFrame({'query_id': [0], 'text': ['teacher']})

        >>> load_queries(["teacher", "developer"])
        DataFrame({'query_id': [0, 1], 'text': ['teacher', 'developer']})
    """
    if isinstance(query_input, str):
        # Single query
        queries_df = pd.DataFrame({
            "query_id": [0],
            "text": [query_input]
        })
        logger.info("Loaded 1 query (single string)")

    elif isinstance(query_input, list):
        # List of queries
        queries_df = pd.DataFrame({
            "query_id": list(range(len(query_input))),
            "text": query_input
        })
        logger.info(f"Loaded {len(query_input)} queries (list)")

    elif isinstance(query_input, pd.DataFrame):
        # Already a DataFrame
        if "text" not in query_input.columns:
            raise ValueError("DataFrame must have 'text' column")

        queries_df = query_input.copy()
        if "query_id" not in queries_df.columns:
            queries_df["query_id"] = range(len(queries_df))

        logger.info(f"Loaded {len(queries_df)} queries (DataFrame)")

    else:
        raise TypeError(
            f"query_input must be str, List[str], or DataFrame, "
            f"got {type(query_input)}"
        )

    return queries_df


# ============================================================================
# Node 1: Load Taxonomy Index
# ============================================================================


def load_taxonomy_index(
    taxonomy_index: Dict[str, Callable],
    taxonomy_key: str,
) -> pd.DataFrame:
    """
    Load taxonomy index from partitioned dataset.

    This node loads the pre-computed taxonomy index created by Module 1
    (build_taxonomy pipeline). The index contains:
    - Taxonomy structure (code, level, parentCode, label, etc.)
    - Embedding views (embedding_label, embedding_definition, embedding_examples)
    - Evidence state (evidence_centroid, evidence_count, evidence_last_updated)

    Args:
        taxonomy_index: Dictionary from catalog.load('taxonomy_index')
            Format: {partition_id: callable_that_returns_DataFrame}
        taxonomy_key: The partition key to retrieve (e.g., "ISCO", "ISIC")

    Returns:
        DataFrame with taxonomy index for the specified partition

    Raises:
        ValueError: If taxonomy_key not found in partitions

    Spec Reference:
        Module 2 — Inference, Step 1: Load taxonomy index
    """
    if taxonomy_key not in taxonomy_index:
        available_keys = list(taxonomy_index.keys())
        raise ValueError(
            f"Taxonomy key '{taxonomy_key}' not found in taxonomy_index. "
            f"Available keys: {available_keys}"
        )

    # Load the partition
    df = taxonomy_index[taxonomy_key]()

    logger.info(
        f"Loaded taxonomy index: taxonomy_key={taxonomy_key}, "
        f"{len(df)} nodes, {df['embedding_dim'].iloc[0]} dimensions"
    )

    # Validate required columns
    required_columns = [
        "code", "level", "parentCode", "label",
        "embedding_label", "embedding_definition", "embedding_examples",
        "evidence_centroid", "evidence_count",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in taxonomy_index: {missing_columns}"
        )

    return df


# ============================================================================
# Node 2: Build Taxonomy Graph
# ============================================================================


def load_taxonomy_graph(taxonomy_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build parent-child adjacency dictionary from taxonomy DataFrame.

    Creates a mapping from parent code to list of child codes for
    efficient sibling expansion during inference.

    Args:
        taxonomy_df: Taxonomy DataFrame with 'code' and 'parentCode' columns

    Returns:
        Dict mapping parent_code -> [child_code1, child_code2, ...]
        Root nodes (parentCode == "__root__") are stored under key "__root__"

    Spec Reference:
        Module 2 — Inference, Step 2: Build taxonomy graph
    """
    graph = defaultdict(list)

    for _, row in taxonomy_df.iterrows():
        parent_code = row["parentCode"]
        # Handle root nodes: parentCode == "__root__"
        if pd.isna(parent_code) or parent_code == "":
            parent_code = "__root__"
        child_code = row["code"]
        graph[parent_code].append(child_code)

    # Convert to regular dict for clarity
    graph_dict = dict(graph)

    # Count nodes per level
    level_counts = taxonomy_df["level"].value_counts().sort_index()
    logger.info(
        f"Built taxonomy graph: {len(graph_dict)} parent nodes, "
        f"level distribution: {level_counts.to_dict()}"
    )

    return graph_dict


# ============================================================================
# Node 3: Build Retrieval Index
# ============================================================================


def build_retrieval_index(taxonomy_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build label-based retrieval index for fast candidate recall.

    Uses ONLY label embeddings for retrieval (not multi-view). This ensures:
    1. Stable retrieval regardless of evidence accumulation
    2. Fast exact cosine similarity search (no approximation)
    3. Consistent recall across queries

    Multi-view scoring happens during routing, NOT during retrieval.

    Args:
        taxonomy_df: Taxonomy DataFrame with embedding_label column

    Returns:
        Dict with:
        - 'embeddings': np.ndarray of shape (n_nodes, embedding_dim)
        - 'codes': List[str] of node codes (aligned with embeddings)
        - 'code_to_pos': Dict[code -> index] for O(1) lookup
        - 'code_to_parent': Dict[code -> parent_code] for ancestor closure
        - 'method': 'label_only' (for tracking)

    Spec Reference:
        Module 2 — Inference, Step 3: Build retrieval index
        Design Decision: Label-based retrieval (revised plan)

    Failure Mode Prevention:
        - Retrieval stability: Label embeddings don't change with evidence
        - Consistent recall: Same query always retrieves same candidates
        - Performance: O(1) lookups instead of O(N) list.index()
    """
    # Extract label embeddings and codes
    embeddings = np.vstack(taxonomy_df["embedding_label"].values)
    codes = taxonomy_df["code"].tolist()

    # Build O(1) lookup maps
    code_to_pos = {code: i for i, code in enumerate(codes)}

    # Build parent mapping for ancestor closure
    code_to_parent = {}
    for _, row in taxonomy_df.iterrows():
        parent = row["parentCode"]
        if pd.isna(parent) or parent == "":
            parent = "__root__"
        code_to_parent[row["code"]] = parent

    # Verify L2 normalization (required for cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        logger.warning("Label embeddings not L2-normalized, normalizing now...")
        embeddings = embeddings / norms[:, np.newaxis]

    logger.info(
        f"Built retrieval index: {len(codes)} nodes, "
        f"embedding_dim={embeddings.shape[1]}, method=label_only"
    )

    return {
        "embeddings": embeddings,
        "codes": codes,
        "code_to_pos": code_to_pos,
        "code_to_parent": code_to_parent,
        "method": "label_only",
    }


# ============================================================================
# Node 4: Prepare Scoring Views
# ============================================================================


def prepare_scoring_views(taxonomy_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare multi-view embeddings for fast O(1) scoring.

    Extracts all embedding views (label, definition, examples, evidence)
    into dictionaries for efficient multi-view scoring during routing.

    Args:
        taxonomy_df: Taxonomy DataFrame with embedding columns:
            - embedding_label (required)
            - embedding_definition (optional, may be None for high-level nodes)
            - embedding_examples (optional, may be None)
            - evidence_centroid (optional, from incremental learning)
            - evidence_count (int, number of corrections)

    Returns:
        Dict with:
        - 'code_to_label_emb': Dict[code -> np.ndarray] label embeddings
        - 'code_to_def_emb': Dict[code -> np.ndarray] definition embeddings
        - 'code_to_ex_emb': Dict[code -> np.ndarray] examples embeddings
        - 'code_to_evidence': Dict[code -> tuple(centroid, count)]
        - 'taxonomy_df': Original DataFrame (for level/label lookup)

    Spec Reference:
        Module 2 — Inference, Step 4: Prepare scoring views
        Multi-view scoring: label + definition + examples + evidence

    Design Decision:
        O(1) dict lookups (not DataFrame.iloc) for performance during routing
    """
    code_to_label_emb = {}
    code_to_def_emb = {}
    code_to_ex_emb = {}
    code_to_evidence = {}

    nodes_with_def = 0
    nodes_with_ex = 0
    nodes_with_evidence = 0
    total_evidence_count = 0

    for _, row in taxonomy_df.iterrows():
        code = row["code"]

        # Label embedding (required)
        code_to_label_emb[code] = np.array(row["embedding_label"])

        # Definition embedding (optional)
        emb_def = row.get("embedding_definition")
        if not _is_missing_embedding(emb_def):
            code_to_def_emb[code] = np.array(emb_def)
            nodes_with_def += 1

        # Examples embedding (optional)
        emb_ex = row.get("embedding_examples")
        if not _is_missing_embedding(emb_ex):
            code_to_ex_emb[code] = np.array(emb_ex)
            nodes_with_ex += 1

        # Evidence (optional, from Module 3)
        evidence_centroid = row.get("evidence_centroid")
        evidence_count = int(row.get("evidence_count", 0))

        if evidence_count > 0 and not _is_missing_embedding(evidence_centroid):
            code_to_evidence[code] = (
                np.array(evidence_centroid),
                evidence_count
            )
            nodes_with_evidence += 1
            total_evidence_count += evidence_count

    logger.info(
        f"Prepared scoring views: {len(code_to_label_emb)} nodes, "
        f"{nodes_with_def} with definition, "
        f"{nodes_with_ex} with examples, "
        f"{nodes_with_evidence} with evidence "
        f"({total_evidence_count} corrections)"
    )

    return {
        "code_to_label_emb": code_to_label_emb,
        "code_to_def_emb": code_to_def_emb,
        "code_to_ex_emb": code_to_ex_emb,
        "code_to_evidence": code_to_evidence,
        "taxonomy_df": taxonomy_df,  # For level/label lookup
    }


# ============================================================================
# Helper Function: Multi-View Scoring with Evidence
# ============================================================================


def compute_multiview_score(
    query_embedding: np.ndarray,
    query_text: str,
    node_code: str,
    scoring_views: Dict[str, Any],
    evidence_tau: float = 10.0,
    evidence_max_beta: float = 0.8,
    short_query_tokens: int = 2,
) -> float:
    """
    Compute multi-view similarity score with per-node dynamic evidence blending.

    Scoring Strategy (Spec-Aligned):
    1. Blend label embedding with evidence (if available):
       - beta_n = min(k / (k + tau), max_beta) where k = evidence_count
       - E_label_eff = (1-beta_n) * E_label + beta_n * E_evidence
    2. Compute similarities: sim_label, sim_def, sim_ex, sim_emp
    3. Apply short-query rule (≤2 tokens): max(sim_label, sim_emp)
    4. Default: max(sim_label, sim_def, sim_ex, sim_emp)

    Using max() prevents definition dilution on short queries.

    Args:
        query_embedding: L2-normalized query embedding
        query_text: Original query text (for token count)
        node_code: Taxonomy code to score
        scoring_views: Dict from prepare_scoring_views with:
            - code_to_label_emb, code_to_def_emb, code_to_ex_emb,
              code_to_evidence
        evidence_tau: Confidence threshold for beta (default 10)
        evidence_max_beta: Cap on evidence weight (default 0.8)
        short_query_tokens: Token threshold for short-query rule (default 2)

    Returns:
        Multi-view similarity score (float, 0.0-1.0)

    Spec Reference:
        Module 2 — Inference, Multi-view scoring
        Evidence blending: Per-node dynamic beta
        Aggregation: max() to prevent dilution

    Failure Mode Prevention:
        - Definition dilution: max() + short-query rule
        - Evidence stability: Dynamic beta based on correction count
        - Semantic accuracy: Multi-view corrects label-only errors
    """
    # Extract embeddings
    E_label = scoring_views["code_to_label_emb"][node_code]
    E_def = scoring_views["code_to_def_emb"].get(node_code)
    E_ex = scoring_views["code_to_ex_emb"].get(node_code)
    evidence_tuple = scoring_views["code_to_evidence"].get(node_code)

    # Step 1: Compute E_label_effective (with evidence if available)
    if evidence_tuple is not None:
        E_evidence, evidence_count = evidence_tuple
        # Dynamic beta based on evidence count
        beta_n = min(
            evidence_count / (evidence_count + evidence_tau),
            evidence_max_beta
        )
        # Blend label + evidence
        E_label_eff = (1 - beta_n) * E_label + beta_n * E_evidence
        # Normalize
        E_label_eff = E_label_eff / np.linalg.norm(E_label_eff)

        # Pure evidence similarity (for short-query rule)
        E_emp = E_evidence / np.linalg.norm(E_evidence)
        sim_emp = float(np.dot(query_embedding, E_emp))
    else:
        # No evidence -> use pure label
        E_label_eff = E_label
        sim_emp = 0.0

    # Step 2: Compute view similarities
    sim_label = float(np.dot(query_embedding, E_label_eff))
    sim_def = float(np.dot(query_embedding, E_def)) if E_def is not None else 0.0
    sim_ex = float(np.dot(query_embedding, E_ex)) if E_ex is not None else 0.0

    # Step 3: Apply short-query rule (≤N tokens)
    tokens = query_text.split()
    if len(tokens) <= short_query_tokens:
        # Short query: only label + evidence (no definition dilution)
        return max(sim_label, sim_emp)

    # Step 4: Default: max across all views
    return max(sim_label, sim_def, sim_ex, sim_emp)


# ============================================================================
# Node 5b: Embed Queries (Batch)
# ============================================================================


def embed_queries(
    queries_df: pd.DataFrame,
    embedding_model: SentenceTransformer,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Embed multiple queries in batch for efficiency.

    Uses the same embedding model as taxonomy preparation to ensure
    semantic consistency. Batch processing is more efficient than
    one-by-one encoding.

    Args:
        queries_df: DataFrame with 'text' column
        embedding_model: Pre-loaded SentenceTransformer model
        batch_size: Batch size for encoding (default 32)

    Returns:
        DataFrame with added 'embedding' column (np.ndarray per row)

    Spec Reference:
        Module 2 — Inference, Step 5: Embed queries
        Architectural extension: Batch support
    """
    texts = queries_df["text"].tolist()

    # Encode all queries in batch
    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 10,  # Show progress for large batches
    )

    # Add embeddings to DataFrame (as list of arrays for proper storage)
    queries_df = queries_df.copy()
    queries_df["embedding"] = list(embeddings)

    logger.info(
        f"Embedded {len(queries_df)} queries: "
        f"embedding_dim={embeddings.shape[1]}, batch_size={batch_size}"
    )

    return queries_df


# ============================================================================
# Node 5c: Embed Query (Single - Backward Compatibility)
# ============================================================================


def embed_query(
    query_text: str,
    embedding_model: SentenceTransformer,
) -> np.ndarray:
    """
    Embed input query text for similarity comparison.

    Uses the same embedding model as taxonomy preparation to ensure
    semantic consistency.

    DEPRECATED: Use load_queries + embed_queries for batch support.
    This node is kept for backward compatibility with single-query mode.

    Args:
        query_text: Input text to classify
        embedding_model: Pre-loaded SentenceTransformer model

    Returns:
        L2-normalized query embedding vector

    Spec Reference:
        Module 2 — Inference, Step 5: Embed query
    """
    # Encode query
    query_embedding = embedding_model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    logger.info(
        f"Embedded query: text_length={len(query_text)}, "
        f"embedding_dim={query_embedding.shape[0]}"
    )

    return query_embedding


# ============================================================================
# Node 6: Retrieve Candidates
# ============================================================================


def retrieve_candidates(
    query_embedding: np.ndarray,
    retrieval_index: Dict[str, Any],
    retrieval_k: int = 10,
    beam_count: int = 2,
) -> Dict[str, Any]:
    """
    Retrieve and refine candidates using structural closure and beam selection.

    Process (Spec-Aligned):
    1. Retrieve top-K by label similarity (fast recall)
    2. Compute ancestor closure A (paths to roots)
    3. Form candidate set V = R ∪ A (for Variant 2: taxonomy-complete routing)
    4. Aggregate evidence by root (L1) for beam selection
    5. Select top-B root beams (prevents semantic dilution)

    Variant 2: Taxonomy-Complete Routing
    - Only requires ancestor closure (not sibling expansion)
    - Routing scores full taxonomy_graph[parent] at each step
    - Ensures path connectivity (retrieved nodes can always be reached from roots)

    Args:
        query_embedding: L2-normalized query embedding
        retrieval_index: Label-based index from build_retrieval_index
            (contains embeddings, codes, code_to_pos, code_to_parent)
        retrieval_k: Number of initial candidates to retrieve
        beam_count: Number of root beams to select (default 2)

    Returns:
        Dict with:
        - 'retrieved_codes': List[str] of top-K retrieved codes
        - 'retrieved_scores': Dict[code -> similarity_score]
        - 'V_codes': set of candidate codes (R ∪ A)
        - 'ancestors': set of ancestor codes (A)
        - 'beam_roots': List[str] of selected L1 root codes
        - 'root_evidence': Dict[root -> List[score]] for beam selection

    Spec Reference:
        Module 2 — Inference, Step 6: Retrieve candidates
        Structural refinement: R ∪ A (ancestor closure)
        Beam selection: Use retrieval evidence as prior

    Failure Mode Prevention:
        - Semantic dilution: Beam selection prevents mixing "Legislator" +
          "Teacher"
        - Path connectivity: Ancestor closure ensures routing can traverse
        - Performance: O(1) lookups (no list.index calls)
    """
    # Extract index components
    label_embeddings = retrieval_index["embeddings"]
    codes = retrieval_index["codes"]
    code_to_parent = retrieval_index["code_to_parent"]

    # Step 1: Retrieve top-K by label similarity (exact search)
    similarities = np.dot(label_embeddings, query_embedding)
    top_k_indices = np.argsort(similarities)[::-1][:retrieval_k]
    retrieved_codes = [codes[i] for i in top_k_indices]
    retrieved_scores = {
        codes[i]: float(similarities[i]) for i in top_k_indices
    }

    R = set(retrieved_codes)

    logger.info(
        f"Retrieved top-{retrieval_k} candidates: "
        f"best_score={max(retrieved_scores.values()):.3f}"
    )

    # Step 2: Compute ancestor closure A (path connectivity)
    A = set()
    for code in R:
        current = code
        while current and current != "__root__":
            parent = code_to_parent.get(current)
            if parent and parent != "__root__":
                A.add(parent)
            current = parent

    # Step 3: Form candidate set V = R ∪ A
    V = R | A

    logger.info(
        f"Structural closure: |R|={len(R)}, |A|={len(A)}, |V|={len(V)}"
    )

    # Step 4: Aggregate evidence by root (L1) for beam selection
    # Helper: Get L1 ancestor (root) for a code
    def get_root_ancestor(code: str) -> str:
        """Traverse to L1 root."""
        current = code
        path = [current]
        while current and current != "__root__":
            parent = code_to_parent.get(current)
            if parent == "__root__":
                break  # current is L1 root
            path.append(parent)
            current = parent
        # Return first node in path whose parent is __root__
        for node in reversed(path):
            if code_to_parent.get(node) == "__root__":
                return node
        return path[0]  # Fallback

    root_evidence = defaultdict(list)
    for code in retrieved_codes:
        root = get_root_ancestor(code)
        root_evidence[root].append(retrieved_scores[code])

    # Step 5: Select top-B roots by evidence mass (sum of scores)
    beam_roots_ranked = sorted(
        root_evidence.items(),
        key=lambda x: sum(x[1]),  # Sum of retrieval scores
        reverse=True
    )[:beam_count]

    beam_roots = [root for root, _ in beam_roots_ranked]

    logger.info(
        f"Beam selection: {len(root_evidence)} roots found, "
        f"selected top-{beam_count}: {beam_roots}"
    )

    return {
        "retrieved_codes": retrieved_codes,
        "retrieved_scores": retrieved_scores,
        "V_codes": V,
        "ancestors": A,
        "beam_roots": beam_roots,
        "root_evidence": dict(root_evidence),
    }


# ============================================================================
# Helper: Scoped Validation (HiRAG-style)
# ============================================================================


def _children_in_v(
    node_code: str,
    taxonomy_graph: Dict[str, List[str]],
    v_codes: set,
) -> List[str]:
    """Return children of node_code that are within candidate set V."""
    return [child for child in taxonomy_graph.get(node_code, []) if child in v_codes]


def _collect_subtree_nodes_in_v(
    root_code: str,
    taxonomy_graph: Dict[str, List[str]],
    v_codes: set,
) -> set:
    """Collect all nodes in the subtree of root_code, restricted to V."""
    if root_code not in v_codes:
        return set()
    visited = set()
    stack = [root_code]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for child in taxonomy_graph.get(node, []):
            if child in v_codes:
                stack.append(child)
    return visited


def _build_path_to_root(code: str, code_to_parent: Dict[str, str]) -> List[str]:
    """Build path from root to code (excluding __root__)."""
    path = []
    current = code
    while current and current != "__root__":
        path.append(current)
        current = code_to_parent.get(current)
    return list(reversed(path))


def validate_prediction_scoped(
    query_embedding: np.ndarray,
    query_text: str,
    routing_result: Dict[str, Any],
    candidates_dict: Dict[str, Any],
    scoring_views: Dict[str, Any],
    taxonomy_graph: Dict[str, List[str]],
    evidence_tau: float = 10.0,
    evidence_max_beta: float = 0.8,
    validation_threshold: float = 0.05,
    validation_override_margin: Optional[float] = None,
    short_query_tokens: int = 2,
) -> Dict[str, Any]:
    """
    Scoped validation (HiRAG-style) against candidate set V.

    Returns validation status and optional override decision.
    """
    if validation_override_margin is not None:
        validation_threshold = validation_override_margin

    v_codes = set(candidates_dict["V_codes"])
    td_code = routing_result["predicted_code"]

    # Identify leaves within V
    leaves_in_v = [
        code for code in v_codes
        if not _children_in_v(code, taxonomy_graph, v_codes)
    ]

    if not leaves_in_v:
        return {
            "final_code": td_code,
            "final_score": routing_result["score"],
            "validation_status": "INSUFFICIENT_NO_LEAVES",
            "validation_override_code": None,
            "validation_margin": None,
        }

    # Score leaves in V to find L*
    def score_node(code: str) -> float:
        return compute_multiview_score(
            query_embedding=query_embedding,
            query_text=query_text,
            node_code=code,
            scoring_views=scoring_views,
            evidence_tau=evidence_tau,
            evidence_max_beta=evidence_max_beta,
            short_query_tokens=short_query_tokens,
        )

    l_star = max(leaves_in_v, key=score_node)
    l_star_score = score_node(l_star)

    # If TD is root, L* is within subtree by definition
    if td_code == "__root__":
        return {
            "final_code": td_code,
            "final_score": routing_result["score"],
            "validation_status": "CONSISTENT",
            "validation_override_code": None,
            "validation_margin": None,
        }

    # Collect leaves under TD subtree within V
    subtree_nodes = _collect_subtree_nodes_in_v(
        td_code, taxonomy_graph, v_codes
    )
    if not subtree_nodes:
        return {
            "final_code": td_code,
            "final_score": routing_result["score"],
            "validation_status": "INSUFFICIENT_NO_SUBTREE",
            "validation_override_code": None,
            "validation_margin": None,
        }

    leaves_sub = [
        code for code in subtree_nodes
        if not _children_in_v(code, taxonomy_graph, v_codes)
    ]
    if not leaves_sub:
        return {
            "final_code": td_code,
            "final_score": routing_result["score"],
            "validation_status": "INSUFFICIENT_NO_SUB_LEAVES",
            "validation_override_code": None,
            "validation_margin": None,
        }

    l_sub = max(leaves_sub, key=score_node)
    l_sub_score = score_node(l_sub)

    if l_star == l_sub:
        return {
            "final_code": td_code,
            "final_score": routing_result["score"],
            "validation_status": "CONSISTENT",
            "validation_override_code": None,
            "validation_margin": None,
        }

    margin = l_star_score - l_sub_score
    if margin > validation_threshold:
        return {
            "final_code": l_star,
            "final_score": l_star_score,
            "validation_status": "OVERRIDE",
            "validation_override_code": l_star,
            "validation_margin": margin,
        }

    return {
        "final_code": td_code,
        "final_score": routing_result["score"],
        "validation_status": "CONFLICT",
        "validation_override_code": None,
        "validation_margin": margin,
    }


# ============================================================================
# Node 7: Route Query
# ============================================================================


def route_query_topdown(
    query_embedding: np.ndarray,
    query_text: str,
    candidates_dict: Dict[str, Any],
    scoring_views: Dict[str, Any],
    taxonomy_graph: Dict[str, List[str]],
    min_descent_gap: float = 0.05,
    parent_veto_margin: float = 0.05,
    evidence_tau: float = 10.0,
    evidence_max_beta: float = 0.8,
    short_query_tokens: int = 2,
    max_depth: Optional[int] = None,
    beam_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    True top-down routing using taxonomy edges, level by level.

    Spec-Aligned Variant 2: Taxonomy-Complete Sibling Comparison
    - Starts at root (or beam root if specified)
    - Scores ONLY children of current parent (level-by-level)
    - Cannot skip levels or jump to L4
    - Sibling comparison uses FULL taxonomy_graph[parent] (all children)
    - Uses multi-view scoring (label + definition + examples + evidence)

    Routing Logic:
    1. Start at beam_root (if specified) or __root__
    2. At each level:
       a. Get ALL children of current parent from taxonomy_graph
       b. Filter to candidates within V (candidate set with ancestor closure)
       c. Score children using compute_multiview_score()
       d. Check stopping criteria (asymmetric):
          - Sibling ambiguity: gap < min_descent_gap
          - Parent competitive: parent_score + margin >= child_score
       e. If not stopped, descend to best child
    3. Return final node with routing trace

    Args:
        query_embedding: L2-normalized query embedding
        query_text: Original query text (for short-query rule)
        candidates_dict: From retrieve_candidates, contains:
            - V_codes: candidate set (retrieved ∪ ancestors)
            - beam_roots: selected L1 roots
        scoring_views: Multi-view embeddings from prepare_scoring_views
        taxonomy_graph: Parent -> children adjacency
        min_descent_gap: Sibling separation threshold (default 0.05)
        parent_veto_margin: Parent competitiveness margin (default 0.05)
        evidence_tau: Evidence confidence threshold (default 10.0)
        evidence_max_beta: Evidence weight cap (default 0.8)
        max_depth: Optional max level to descend
        beam_root: If specified, start from this L1 root (not __root__)

    Returns:
        Dict with:
        - 'predicted_code': Final node code
        - 'score': Final node multi-view score
        - 'path': List[str] of codes from root to prediction
        - 'stopping_reason': str explaining why routing stopped
        - 'ambiguous': bool (True if sibling near-tie)
        - 'alternatives': List[str] of alternative codes (if ambiguous)
        - 'routing_trace': List[Dict] of decisions at each level

    Spec Reference:
        Module 2 — Inference, Step 7: Route query
        Variant 2: Taxonomy-complete routing
        Multi-view scoring: label + definition + examples + evidence

    Failure Mode Prevention:
        - Over-specification: Sibling ambiguity stops at parent
        - Wrong branch: Multi-view scoring corrects label-only errors
        - Semantic dilution: Beam selection isolates root branches
    """
    V_codes = set(candidates_dict["V_codes"])
    taxonomy_df = scoring_views["taxonomy_df"]
    routing_trace = []

    # Start from specified beam root or __root__
    if beam_root:
        current_parent = beam_root
        path = [beam_root]
        current_level = 1
    else:
        current_parent = "__root__"
        path = []
        current_level = 0

    stopping_reason = "unknown"
    ambiguous = False
    alternatives = []
    last_best_child = None

    logger.info(
        f"Starting top-down routing from {current_parent}, "
        f"|V|={len(V_codes)}"
    )

    # Iterative level-by-level descent
    while True:
        # Get ALL children of current parent (taxonomy-complete)
        all_children = taxonomy_graph.get(current_parent, [])

        if not all_children:
            # Leaf reached
            stopping_reason = "leaf_node_reached"
            break

        candidate_children = [c for c in all_children if c in V_codes]
        if not candidate_children:
            stopping_reason = "no_candidate_children"
            ambiguous = True
            alternatives = []
            routing_trace.append({
                "level": current_level + 1,
                "parent": current_parent,
                "all_children_count": len(all_children),
                "candidate_children_count": 0,
                "candidate_children": [],
                "scores": {},
                "best_child": None,
                "best_score": None,
                "sibling_gap": None,
                "decision": "stop_no_candidate_children",
            })
            break

        # Check max depth constraint
        if max_depth is not None and current_level >= max_depth:
            stopping_reason = f"max_depth_reached (depth={max_depth})"
            break

        # Score ALL taxonomy children using multi-view scoring
        child_scores = {
            child: compute_multiview_score(
                query_embedding=query_embedding,
                query_text=query_text,
                node_code=child,
                scoring_views=scoring_views,
                evidence_tau=evidence_tau,
                evidence_max_beta=evidence_max_beta,
                short_query_tokens=short_query_tokens,
            )
            for child in all_children
        }

        # Sort by score (descending)
        sorted_children = sorted(
            child_scores.items(), key=lambda x: x[1], reverse=True
        )
        best_child, best_score = sorted_children[0]
        last_best_child = best_child

        # Compute sibling separation
        if len(sorted_children) > 1:
            second_score = sorted_children[1][1]
            sibling_gap = best_score - second_score
        else:
            sibling_gap = float('inf')  # Only one child

        if best_child not in V_codes:
            stopping_reason = "best_child_not_in_candidates"
            ambiguous = True
            alternatives = [
                code for code, _ in sorted_children if code in V_codes
            ][:3]
            if not alternatives:
                alternatives = [best_child]
            routing_trace.append({
                "level": current_level + 1,
                "parent": current_parent,
                "all_children_count": len(all_children),
                "candidate_children_count": len(candidate_children),
                "candidate_children": candidate_children,
                "scores": child_scores,
                "best_child": best_child,
                "best_score": best_score,
                "sibling_gap": sibling_gap,
                "decision": "stop_best_child_not_in_candidates",
            })
            break

        # Asymmetric stopping: Check sibling ambiguity
        if sibling_gap < min_descent_gap:
            # Siblings too close - stop at parent (prevents over-specification)
            stopping_reason = "sibling_ambiguity"
            ambiguous = True
            alternatives = [code for code, _ in sorted_children[:3]]
            routing_trace.append({
                "level": current_level + 1,
                "parent": current_parent,
                "all_children_count": len(all_children),
                "candidate_children_count": len(candidate_children),
                "candidate_children": candidate_children,
                "scores": child_scores,
                "best_child": best_child,
                "best_score": best_score,
                "sibling_gap": sibling_gap,
                "decision": "stop_sibling_ambiguity",
            })
            break

        # Parent competitiveness veto (if not at root)
        if current_parent != "__root__":
            parent_score = compute_multiview_score(
                query_embedding=query_embedding,
                query_text=query_text,
                node_code=current_parent,
                scoring_views=scoring_views,
                evidence_tau=evidence_tau,
                evidence_max_beta=evidence_max_beta,
                short_query_tokens=short_query_tokens,
            )

            if parent_score + parent_veto_margin >= best_score:
                # Parent is competitive - don't force descent
                stopping_reason = "parent_competitive"
                ambiguous = True
                alternatives = [current_parent, best_child]
                scores_with_parent = dict(child_scores)
                scores_with_parent[current_parent] = parent_score
                routing_trace.append({
                    "level": current_level + 1,
                    "parent": current_parent,
                    "all_children_count": len(all_children),
                    "candidate_children_count": len(candidate_children),
                    "candidate_children": candidate_children,
                    "scores": scores_with_parent,
                    "best_child": best_child,
                    "best_score": best_score,
                    "sibling_gap": sibling_gap,
                    "decision": "stop_parent_competitive",
                })
                break

        # Record trace
        routing_trace.append({
            "level": current_level + 1,
            "parent": current_parent,
            "all_children_count": len(all_children),
            "candidate_children_count": len(candidate_children),
            "candidate_children": candidate_children,
            "scores": child_scores,
            "best_child": best_child,
            "best_score": best_score,
            "sibling_gap": sibling_gap,
            "decision": "descend",
        })

        # Descend to best child
        path.append(best_child)
        current_parent = best_child
        current_level += 1

        # Safety: prevent infinite loops
        if current_level > 10:
            stopping_reason = "max_iterations_exceeded"
            logger.warning(f"Max iterations exceeded at level {current_level}")
            break

    # Final prediction
    if not path:
        predicted_code = "__root__"
        predicted_level = 0
        final_score = routing_trace[-1]["best_score"] if routing_trace else 0.0
        path = ["__root__"]
    else:
        predicted_code = path[-1]
        final_score = compute_multiview_score(
            query_embedding=query_embedding,
            query_text=query_text,
            node_code=predicted_code,
            scoring_views=scoring_views,
            evidence_tau=evidence_tau,
            evidence_max_beta=evidence_max_beta,
            short_query_tokens=short_query_tokens,
        )

        # Get level
        pred_row = taxonomy_df[taxonomy_df["code"] == predicted_code].iloc[0]
        predicted_level = int(pred_row["level"])

    logger.info(
        f"Routing complete: predicted={predicted_code} (L{predicted_level}), "
        f"score={final_score:.3f}, reason={stopping_reason}, "
        f"ambiguous={ambiguous}"
    )

    return {
        "predicted_code": predicted_code,
        "predicted_level": predicted_level,
        "score": final_score,
        "path": path,
        "stopping_reason": stopping_reason,
        "ambiguous": ambiguous,
        "alternatives": alternatives,
        "routing_trace": routing_trace,
    }


# ============================================================================
# Node 8: Format Predictions
# ============================================================================


def format_predictions(
    routing_result: Dict[str, Any],
    taxonomy_df: pd.DataFrame,
    query_text: str,
) -> Dict[str, Any]:
    """
    Format final prediction output with explainability.

    Returns structured prediction with:
    - Primary prediction
    - Confidence/ambiguity indicators
    - Alternative predictions (if ambiguous)
    - Routing trace for explainability

    Args:
        routing_result: Output from route_query
        taxonomy_df: Taxonomy DataFrame (for label lookup)
        query_text: Original query text

    Returns:
        Dict with formatted prediction:
        {
            "query": str,
            "prediction": {
                "code": str,
                "label": str,
                "level": int,
                "score": float,
            },
            "ambiguous": bool,
            "alternatives": [{"code": str, "label": str, "score": float}, ...],
            "stopping_reason": str,
            "path": [str],  # codes from root to prediction
            "routing_trace": [...],  # detailed decisions
        }

    Spec Reference:
        Module 2 — Inference, Step 8: Format predictions
        Explainability: Include ambiguous flag and alternatives
    """
    code_to_label = {row["code"]: row["label"] for _, row in taxonomy_df.iterrows()}

    final_code = routing_result.get("final_code", routing_result["predicted_code"])
    final_score = routing_result.get("final_score", routing_result["score"])
    validation_status = routing_result.get("validation_status", "UNVALIDATED")
    validation_override_code = routing_result.get("validation_override_code")

    if final_code == "__root__":
        predicted_label = "__root__"
        predicted_level = 0
    else:
        predicted_label = code_to_label[final_code]
        pred_row = taxonomy_df[taxonomy_df["code"] == final_code].iloc[0]
        predicted_level = int(pred_row["level"])

    # Format alternatives
    alternatives_formatted = []
    last_trace = (
        routing_result["routing_trace"][-1]
        if routing_result["routing_trace"]
        else None
    )
    ambiguous = routing_result["ambiguous"]
    if validation_status in {"CONFLICT", "INSUFFICIENT_NO_LEAVES", "INSUFFICIENT_NO_SUBTREE", "INSUFFICIENT_NO_SUB_LEAVES"}:
        ambiguous = True
    if ambiguous and routing_result["alternatives"]:
        for alt_code in routing_result["alternatives"]:
            if alt_code != final_code:
                alternatives_formatted.append({
                    "code": alt_code,
                    "label": code_to_label.get(alt_code, "Unknown"),
                    "score": (
                        last_trace["scores"].get(alt_code, 0.0)
                        if last_trace
                        else 0.0
                    ),
                })

    formatted_output = {
        "query": query_text,
        "prediction": {
            "code": final_code,
            "label": predicted_label,
            "level": predicted_level,
            "score": final_score,
        },
        "ambiguous": ambiguous,
        "alternatives": alternatives_formatted,
        "stopping_reason": routing_result["stopping_reason"],
        "path": routing_result["path"],
        "routing_trace": routing_result["routing_trace"],
        "validation_status": validation_status,
        "validation_override_code": validation_override_code,
        "validation_margin": routing_result.get("validation_margin"),
    }

    logger.info(
        f"Formatted prediction: {final_code} ({predicted_label}), "
        f"ambiguous={ambiguous}, "
        f"{len(alternatives_formatted)} alternatives"
    )

    return formatted_output


# ============================================================================
# Batch Processing Nodes
# ============================================================================


def batch_inference(
    queries_df: pd.DataFrame,
    retrieval_index: Dict[str, Any],
    scoring_views: Dict[str, Any],
    taxonomy_graph: Dict[str, List[str]],
    taxonomy_df: pd.DataFrame,
    retrieval_k: int = 10,
    beam_count: int = 2,
    min_descent_gap: float = 0.05,
    parent_veto_margin: float = 0.05,
    evidence_tau: float = 10.0,
    evidence_max_beta: float = 0.8,
    short_query_tokens: int = 2,
    validation_threshold: float = 0.05,
    validation_override_margin: Optional[float] = None,
    max_depth: Optional[int] = None,
) -> pd.DataFrame:
    """
    Process multiple queries in batch through the full inference pipeline.

    This node combines retrieval, routing, and formatting for efficiency.
    It iterates over queries in the DataFrame and applies the full
    inference pipeline to each one.

    Args:
        queries_df: DataFrame with 'query_id', 'text', and 'embedding' columns
        retrieval_index: Label-based index from build_retrieval_index
        scoring_views: Multi-view handles from prepare_scoring_views
        taxonomy_graph: Parent-child adjacency
        taxonomy_df: Taxonomy DataFrame (for label lookup)
        retrieval_k: Number of candidates to retrieve
        min_descent_gap: Sibling separation threshold
        parent_veto_margin: Parent competitiveness margin
        beta: Evidence weight
        short_query_tokens: Token threshold for short-query rule
        validation_threshold: Override threshold for scoped validation
        max_depth: Optional maximum depth

    Returns:
        DataFrame with columns:
        - query_id: Original query ID
        - query: Original query text
        - predicted_code: Predicted taxonomy code
        - predicted_label: Predicted taxonomy label
        - predicted_level: Predicted taxonomy level
        - score: Final similarity score
        - ambiguous: Whether prediction is ambiguous
        - alternatives: JSON list of alternative predictions
        - stopping_reason: Explanation of stopping decision
        - path: JSON list of codes from root to prediction
        - validation_status: CONSISTENT/OVERRIDE/CONFLICT/INSUFFICIENT_*
        - validation_override_code: Code chosen by validation override (if any)
        - validation_margin: Score margin for override decision

    Spec Reference:
        Module 2 — Inference, Batch processing
        Architectural extension: Combines Steps 6-8 for multiple queries

    Design Decision:
        Batch processing in a single node allows:
        1. Efficient iteration without Kedro overhead
        2. Reuse of loaded model/taxonomy/indexes
        3. Progress tracking for large batches
        4. Simpler error handling per query
    """
    results = []

    logger.info(f"Starting batch inference for {len(queries_df)} queries")

    for idx, row in queries_df.iterrows():
        query_id = row["query_id"]
        query_text = row["text"]
        query_embedding = row["embedding"]

        try:
            # Step 1: Retrieve candidates with structural closure
            candidates_dict = retrieve_candidates(
                query_embedding=query_embedding,
                retrieval_index=retrieval_index,
                retrieval_k=retrieval_k,
                beam_count=beam_count,
            )

            # Step 2: Route query (true top-down with multi-view scoring)
            beam_roots = candidates_dict.get("beam_roots") or []
            if not beam_roots:
                beam_roots = [None]

            beam_results = []
            for root in beam_roots:
                beam_results.append(
                    route_query_topdown(
                        query_embedding=query_embedding,
                        query_text=query_text,
                        candidates_dict=candidates_dict,
                        scoring_views=scoring_views,
                        taxonomy_graph=taxonomy_graph,
                        min_descent_gap=min_descent_gap,
                        parent_veto_margin=parent_veto_margin,
                        evidence_tau=evidence_tau,
                        evidence_max_beta=evidence_max_beta,
                        short_query_tokens=short_query_tokens,
                        max_depth=max_depth,
                        beam_root=root,
                    )
                )

            routing_result = max(
                beam_results,
                key=lambda r: (r["score"], not r["ambiguous"], r["predicted_level"]),
            )

            # Step 3: Scoped validation (HiRAG-style)
            validation_result = validate_prediction_scoped(
                query_embedding=query_embedding,
                query_text=query_text,
                routing_result=routing_result,
                candidates_dict=candidates_dict,
                scoring_views=scoring_views,
                taxonomy_graph=taxonomy_graph,
                evidence_tau=evidence_tau,
                evidence_max_beta=evidence_max_beta,
                validation_threshold=validation_threshold,
                validation_override_margin=validation_override_margin,
                short_query_tokens=short_query_tokens,
            )

            routing_result.update(validation_result)

            if routing_result.get("validation_status") == "OVERRIDE":
                override_code = routing_result.get("validation_override_code")
                if override_code:
                    routing_result["path"] = _build_path_to_root(
                        override_code, retrieval_index["code_to_parent"]
                    )
                    routing_result["stopping_reason"] = (
                        f"{routing_result['stopping_reason']}|validation_override"
                    )

            # Step 4: Format prediction
            prediction = format_predictions(
                routing_result=routing_result,
                taxonomy_df=taxonomy_df,
                query_text=query_text,
            )

            # Flatten prediction to row format
            results.append({
                "query_id": query_id,
                "query": query_text,
                "predicted_code": prediction["prediction"]["code"],
                "predicted_label": prediction["prediction"]["label"],
                "predicted_level": prediction["prediction"]["level"],
                "score": prediction["prediction"]["score"],
                "ambiguous": prediction["ambiguous"],
                "alternatives": prediction["alternatives"],  # Will be serialized as JSON
                "stopping_reason": prediction["stopping_reason"],
                "path": prediction["path"],  # Will be serialized as JSON
                "validation_status": prediction["validation_status"],
                "validation_override_code": prediction["validation_override_code"],
                "validation_margin": prediction["validation_margin"],
            })

            if (idx + 1) % 10 == 0 or (idx + 1) == len(queries_df):
                logger.info(f"Processed {idx + 1}/{len(queries_df)} queries")

        except Exception as e:
            logger.error(
                f"Error processing query_id={query_id}: {e}",
                exc_info=True
            )
            # Add error row
            results.append({
                "query_id": query_id,
                "query": query_text,
                "predicted_code": None,
                "predicted_label": None,
                "predicted_level": None,
                "score": None,
                "ambiguous": None,
                "alternatives": None,
                "stopping_reason": f"error: {str(e)}",
                "path": None,
                "validation_status": None,
                "validation_override_code": None,
                "validation_margin": None,
            })

    results_df = pd.DataFrame(results)

    logger.info(
        f"Batch inference complete: {len(results_df)} predictions, "
        f"{len(results_df[results_df['predicted_code'].notna()])} successful"
    )

    return results_df
