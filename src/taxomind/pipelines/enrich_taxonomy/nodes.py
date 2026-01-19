"""
Nodes for the enrich_taxonomy pipeline.

This pipeline enriches taxonomy definitions using an LLM to:
- Clean definitions and examples (remove cross-references)
- Generate positive query examples
- Generate negative query examples (based on similar labels)
"""

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

from taxomind.pipelines.build_taxonomy import nodes as build_taxonomy_nodes
from taxomind.utils import embedding_utils
from taxomind.utils.taxonomy_utils import get_parent_chain

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Patterns to strip from definitions/examples
REFERENCE_PATTERNS = [
    r"\bSee class\b",
    r"\bSee group\b",
    r"\bSee division\b",
    r"\bSee section\b",
    r"\bSee\b.*\bclass\b",
    r"\bCheck parent\b",
    r"\bCheck\b.*\bparent\b",
    r"\bSee also\b",
]


def _is_missing(value: Any) -> bool:
    """Check if a value is missing (None or NaN)."""
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    if isinstance(missing, bool):
        return bool(missing)
    if hasattr(missing, "shape") and missing.shape == ():
        return bool(missing)
    return False


def _strip_reference_lines(text: Any) -> str:
    """Remove cross-reference lines from text."""
    if _is_missing(text):
        return ""
    lines = []
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(re.search(pat, line, flags=re.IGNORECASE) for pat in REFERENCE_PATTERNS):
            continue
        line = re.sub(r"^[\-*\d.\)\(]+\s*", "", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _basic_cleanup(text: Any) -> str:
    """Basic text cleanup: strip references and normalize whitespace."""
    if _is_missing(text):
        return ""
    cleaned = _strip_reference_lines(text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    if text is None:
        return None
    payload = str(text).strip()
    candidates = [payload]
    if "```" in payload:
        parts = payload.split("```")
        for idx in range(1, len(parts), 2):
            block = parts[idx].strip()
            if block.startswith("json"):
                block = block[4:].strip()
            candidates.append(block)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_multiline(value: Any, max_lines: Optional[int]) -> str:
    """Normalize multiline text, optionally limiting lines."""
    if value is None:
        return ""
    if isinstance(value, list):
        lines = [str(item).strip() for item in value if str(item).strip()]
    else:
        lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    cleaned = []
    for line in lines:
        line = re.sub(r"^[\-*\d.\)\(]+\s*", "", line).strip()
        line = line.replace("\u2022", "").strip()
        if line:
            cleaned.append(line)
    if max_lines is not None:
        cleaned = cleaned[:max_lines]
    return "\n".join(cleaned)


def _format_similar_labels(similar: Any) -> str:
    """Format similar labels for the LLM prompt."""
    if not similar:
        return "(none)"
    blocks = []
    for item in similar:
        label = item.get("label", "")
        definition = item.get("definition", "")
        examples = item.get("examples", "")
        blocks.append(
            f"- Label: {label}\n  Definition: {definition}\n  Examples: {examples}".strip()
        )
    return "\n".join(blocks)


def _build_prompt(row: pd.Series) -> str:
    """Build the LLM prompt for a taxonomy row."""
    parent_path = row.get("parent_path") or "(root)"
    definition = row.get("definition") or ""
    examples = row.get("examples") or ""
    similar_labels = _format_similar_labels(row.get("similar_labels") or [])

    return f"""You clean taxonomy text and craft short query examples.

Label: {row.get('label')}
Level: {row.get('level')}
Parent path labels: {parent_path}

Definition (raw):
{definition}

Examples (raw):
{examples}

Similar labels at the same level (use for negatives only):
{similar_labels}

Rules:
- Remove cross references like "See class" or "Check parent label"
- Keep meaning faithful to label and parent path
- Use the same language as the label/definition when present
- definition_clean: short description if empty or not descriptive enough
- examples_clean: 2-4 short examples if there are usable examples; if none, return empty string (do not hallucinate)
- positive_examples: 2 short query phrases (1-4 words) as if someone were casually describing it, newline separated
- negative_examples: 2 short query phrases as if someone were casually describing it but plausible and wrong, newline separated
- Base negative_examples on the similar labels when available
- Do not include the label or code in the examples
- If raw text is empty, infer cautiously from label and parent path only

Return JSON with keys: definition_clean, examples_clean, positive_examples, negative_examples.
"""


def _run_ollama(prompt: str, llm_config: Dict[str, Any]) -> str:
    """Call Ollama API."""
    url = llm_config.get("ollama_url", "http://localhost:11434")
    model = llm_config.get("model", "llama3")
    timeout = llm_config.get("timeout", 120)

    response = requests.post(
        f"{url}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    content = payload.get("message", {}).get("content") or payload.get("response", "")
    return content or ""


def _run_openai(prompt: str, llm_config: Dict[str, Any]) -> str:
    """Call OpenAI-compatible API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    base_url = llm_config.get("openai_base_url", "https://api.openai.com/v1")
    model = llm_config.get("model", "gpt-4.1-mini")
    timeout = llm_config.get("timeout", 120)

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )


def _run_llm(prompt: str, llm_config: Dict[str, Any]) -> str:
    """Run LLM based on provider configuration."""
    provider = llm_config.get("provider", "openai")
    if provider == "ollama":
        return _run_ollama(prompt, llm_config)
    if provider == "openai":
        return _run_openai(prompt, llm_config)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _llm_enrich_row(row: pd.Series, llm_config: Dict[str, Any]) -> Dict[str, str]:
    """Enrich a single taxonomy row using LLM."""
    prompt = _build_prompt(row)
    try:
        response = _run_llm(prompt, llm_config)
        parsed = _parse_json_block(response)
    except Exception as exc:
        logger.warning(f"LLM call failed for code {row.get('code')}: {exc}")
        parsed = None

    if not parsed:
        return {
            "definition_clean": _basic_cleanup(row.get("definition")),
            "examples_clean": _basic_cleanup(row.get("examples")),
            "positive_examples": "",
            "negative_examples": "",
        }
    return {
        "definition_clean": _basic_cleanup(parsed.get("definition_clean")),
        "examples_clean": _normalize_multiline(parsed.get("examples_clean"), max_lines=4),
        "positive_examples": _normalize_multiline(parsed.get("positive_examples"), max_lines=2),
        "negative_examples": _normalize_multiline(parsed.get("negative_examples"), max_lines=2),
    }


# =============================================================================
# Pipeline Node Functions
# =============================================================================


def load_and_prepare_taxonomy(
    taxonomy_definition: Dict[str, Callable[[], pd.DataFrame]],
    taxonomy_key: str,
) -> pd.DataFrame:
    """
    Load taxonomy from partitioned dataset and normalize columns.

    Args:
        taxonomy_definition: Partitioned dataset with taxonomy DataFrames
        taxonomy_key: Which taxonomy to load ("ISCO" or "ISIC")

    Returns:
        DataFrame with normalized columns: code, label, level, definition, examples, parentCode
    """
    logger.info(f"Loading taxonomy: {taxonomy_key}")

    df = build_taxonomy_nodes.load_taxonomy_from_partition(
        taxonomy_definition, taxonomy_key
    )
    df = build_taxonomy_nodes.normalize_prototype_views(df)

    required_cols = ["code", "label", "level", "parentCode"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    logger.info(f"Loaded {len(df)} taxonomy nodes")
    return df


def build_parent_paths(taxonomy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add parent_path and parent_label columns to taxonomy.

    Args:
        taxonomy_df: DataFrame with code, label, parentCode columns

    Returns:
        DataFrame with added parent_path and parent_label columns
    """
    logger.info("Building parent paths")

    code_to_label = taxonomy_df.set_index("code")["label"].to_dict()

    def _parent_path_labels(code: str) -> str:
        chain = get_parent_chain(taxonomy_df, code)
        labels = [code_to_label.get(parent_code, parent_code) for parent_code in chain]
        return " > ".join(labels)

    df = taxonomy_df.copy()
    df["parent_path"] = df["code"].apply(_parent_path_labels)
    df["parent_label"] = df["parentCode"].map(code_to_label).fillna("")

    return df


def build_similar_labels(
    taxonomy_df: pd.DataFrame,
    embedding_model: SentenceTransformer,
    k_similar_labels: int,
    embedding_prefix: Dict[str, str],
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Find similar labels at each level using embeddings.

    Args:
        taxonomy_df: DataFrame with code, label, level, definition, examples
        embedding_model: Pre-loaded SentenceTransformer model
        k_similar_labels: Number of similar labels to find per node
        embedding_prefix: Dict with 'document' prefix for embeddings

    Returns:
        DataFrame with added similar_labels column
    """
    logger.info(f"Building similar labels (k={k_similar_labels})")

    df = taxonomy_df.copy()
    prefix = embedding_prefix.get("document", "")

    if k_similar_labels <= 0:
        df["similar_labels"] = [[] for _ in range(len(df))]
        return df

    similar: Dict[str, List[Dict[str, str]]] = {}

    for level, group in df.groupby("level"):
        group_reset = group.reset_index(drop=True)
        labels = group_reset["label"].fillna("").astype(str).tolist()
        codes = group_reset["code"].tolist()

        if len(labels) < 2:
            for code in codes:
                similar[code] = []
            continue

        try:
            vectors, _ = embedding_utils.encode_texts(
                embedding_model,
                labels,
                embed_all=True,
                input_prefix=prefix,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.warning(f"Embedding failed for level {level}: {exc}")
            for code in codes:
                similar[code] = []
            continue

        # Compute cosine similarities
        sims = vectors @ vectors.T

        for idx, code in enumerate(codes):
            sims[idx, idx] = -1.0  # Exclude self
            top_idx = np.argsort(sims[idx])[::-1][:k_similar_labels]
            candidates = []
            for j in top_idx:
                candidates.append({
                    "label": labels[j],
                    "definition": _basic_cleanup(group_reset.at[j, "definition"]),
                    "examples": _basic_cleanup(group_reset.at[j, "examples"]),
                })
            similar[code] = candidates

    df["similar_labels"] = df["code"].map(similar).apply(lambda v: v or [])
    return df


def enrich_with_llm(
    taxonomy_df: pd.DataFrame,
    llm_config: Dict[str, Any],
    max_rows: Optional[int],
) -> pd.DataFrame:
    """
    Enrich taxonomy using LLM to clean definitions and generate examples.

    Args:
        taxonomy_df: DataFrame with taxonomy data and similar_labels
        llm_config: LLM configuration (provider, model, urls, timeout)
        max_rows: Limit rows for dry-run testing (None = all rows)

    Returns:
        DataFrame with added columns: definition_clean, examples_clean,
        positive_examples, negative_examples
    """
    logger.info(f"Enriching taxonomy with LLM (provider={llm_config.get('provider')})")

    df = taxonomy_df.copy()

    # Limit rows if max_rows is set (for dry-run testing)
    if max_rows is not None:
        df = df.head(max_rows)
        logger.info(f"Processing {len(df)} rows (max_rows={max_rows})")

    try:
        from tqdm.auto import tqdm
        iterator = tqdm(df.iterrows(), total=len(df), desc="LLM enrichment")
    except ImportError:
        iterator = df.iterrows()

    results = []
    for _, row in iterator:
        payload = _llm_enrich_row(row, llm_config)
        payload["code"] = row["code"]
        results.append(payload)

    results_df = pd.DataFrame(results).set_index("code")

    # Add new columns to the dataframe
    new_cols = ["definition_clean", "examples_clean", "positive_examples", "negative_examples"]
    for col in new_cols:
        df[col] = df["code"].map(results_df[col])

    logger.info(f"Enriched {len(results)} rows")
    return df


def finalize_enriched_taxonomy(
    taxonomy_df: pd.DataFrame,
    apply_cleaned: bool,
    taxonomy_key: str,
) -> Dict[str, pd.DataFrame]:
    """
    Finalize enriched taxonomy for output.

    Args:
        taxonomy_df: DataFrame with LLM-enriched columns
        apply_cleaned: Whether to overwrite definition/examples with cleaned versions
        taxonomy_key: Taxonomy key for partitioned output

    Returns:
        Dict mapping taxonomy_key to DataFrame (for PartitionedDataset)
    """
    logger.info(f"Finalizing enriched taxonomy (apply_cleaned={apply_cleaned})")

    df = taxonomy_df.copy()

    # Drop intermediate column
    if "similar_labels" in df.columns:
        df = df.drop(columns=["similar_labels"])

    # Optionally overwrite original fields with cleaned versions
    if apply_cleaned:
        if "definition_clean" in df.columns:
            df["definition"] = df["definition_clean"]
        if "examples_clean" in df.columns:
            df["examples"] = df["examples_clean"]

    logger.info(f"Output: {len(df)} rows for {taxonomy_key}")
    return {taxonomy_key: df}
