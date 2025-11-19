from typing import Iterable, List, Sequence, Any, Dict, List
from uuid import uuid4
import pandas as pd


UNKNOWN_LABEL = "Unknown"
UNKNOWN_DEFINITION = "This category reflects the inability to determine the appropriate classification."
UNKNOWN_EXAMPLES = ""


def compose_text(row: pd.Series) -> str:
    """Concatenate label, definition, and examples for embeddings."""

    label = row.get("label", "")
    definition = row.get("definition", "")
    examples = row.get("examples", "")
    segments: List[str] = []
    segments.append(label)
    segments.append(f"Definition: {definition}")
    segments.append(f"Examples: {examples}")

    return "\n".join(segments)


def get_level_nodes(
    taxonomy: pd.DataFrame, level: int, parent_code: str | None
) -> pd.DataFrame:
    """Return nodes for a given level filtered by parent code."""

    mask = taxonomy["level"] == level & taxonomy["parentCode"] == parent_code

    return taxonomy.loc[mask].copy()


def get_bottom_up_candidates(
    taxonomy: pd.DataFrame, route: Sequence[dict]
) -> pd.DataFrame:
    """Return candidate nodes for validation (leaf + siblings)."""

    if not route:
        return taxonomy.iloc[0:0].copy()

    target = route[-1]
    level = int(target.get("level", 0))
    parent_code = target.get("parentCode")
    code = target.get("code")

    siblings = get_level_nodes(taxonomy, level, parent_code)
    current = taxonomy[taxonomy["code"] == code]
    candidates = pd.concat([current, siblings], ignore_index=True)
    return candidates.drop_duplicates(subset="code")


def filter_by_codes(taxonomy: pd.DataFrame, codes: Iterable[str]) -> pd.DataFrame:
    """Convenience helper for selecting taxonomy entries by code."""

    code_list = list(codes)
    if not code_list:
        return taxonomy.iloc[0:0].copy()
    return taxonomy[taxonomy["code"].isin(code_list)].copy()


def build_taxonomy_context(taxonomy: pd.DataFrame, candidates: Sequence[dict]) -> str:
    """Prepare multilingual context strings for judge prompts."""

    codes = [candidate.get("code") for candidate in candidates if candidate.get("code")]
    subset = filter_by_codes(taxonomy, codes)
    lines: List[str] = []
    for _, row in subset.iterrows():
        definition = row.get("definition") or ""
        examples = row.get("examples") or ""
        snippet = f"{row['code']} | {row['label']}\nDefinition: {definition}\nExample: {examples}"
        lines.append(snippet.strip())
    return "\n\n".join(lines)


def row_to_candidate(row: pd.Series, score: float | None = None) -> dict:
    """Convert a taxonomy row into the candidate dictionary schema."""

    candidate = {
        "code": row.get("code"),
        "label": row.get("label"),
        "level": int(row.get("level", 0)),
        "parentCode": row.get("parentCode"),
        "isLeaf": bool(row.get("isLeaf")),
    }
    if score is not None:
        candidate["score"] = float(score)
    return candidate


def routes_match(route_a: Sequence[dict], route_b: Sequence[dict]) -> bool:
    """Compare two routes by their node codes."""

    if not route_a or not route_b:
        return False
    if len(route_a) != len(route_b):
        return False
    return all(a.get("code") == b.get("code") for a, b in zip(route_a, route_b))


def same_branch(code_a: str | None, code_b: str | None) -> bool:
    """Return True when the provided codes belong to the same hierarchical branch."""

    if not code_a or not code_b:
        return False
    return code_a.startswith(code_b) or code_b.startswith(code_a)


def compose_path_text(path_nodes: Sequence[dict]) -> str:
    """Concatenate the textual content of every node in a path."""

    segments: List[str] = []
    for idx, node in enumerate(path_nodes, 1):
        label = node.get("label") or ""
        definition = node.get("definition") or ""
        examples = node.get("examples") or ""
        parts = [f"Level {idx}: {label}".strip()]
        if definition:
            parts.append(f"Definition: {definition}")
        if examples:
            parts.append(f"Examples: {examples}")
        segments.append("\n".join(parts).strip())
    return "\n\n".join(segment for segment in segments if segment)


def row_to_node_dict(row: pd.Series) -> dict:
    return {
        "code": row.get("code"),
        "label": row.get("label"),
        "definition": row.get("definition"),
        "examples": row.get("examples"),
        "level": int(row.get("level", 0)),
        "parentCode": row.get("parentCode"),
        "isLeaf": bool(row.get("isLeaf")),
    }


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def unknown_code(level: int) -> str:
    if level <= 0:
        return ""
    return "-" + ("9" * level)


def normalize_parent(value: object) -> str:
    code = normalize_code(value)
    return str(code) if code is not None and not pd.isna(code) else "__root__"


def normalize_code(value: Any) -> str | None:
    text = normalize_text(value)
    return text or None


def infer_max_depth(max_depth: Any, nodes_raw: List[dict]) -> int:
    if max_depth is not None:
        try:
            value = int(max_depth)
            if value > 0:
                return value
        except (TypeError, ValueError):
            raise ValueError("taxonomy.maxDepth must be an integer") from None

    inferred = 0
    for node in nodes_raw:
        level = node.get("level")
        try:
            level_int = int(level)
        except (TypeError, ValueError):
            continue
        inferred = max(inferred, level_int)
    return inferred


def normalize_node(node: dict, taxonomy_key: str, max_depth: int) -> Dict[str, Any]:
    
    code = normalize_code(node.get("code"))
    if not code:
        raise ValueError("each node must include a code")

    try:
        level = int(node.get("level"))
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"invalid level for node {code}") from exc
    if level <= 0:
        raise ValueError(f"level must be positive for node {code}")
    if level > max_depth:
        raise ValueError(
            f"node {code} specifies level {level} which exceeds maxDepth {max_depth}"
        )

    parent_code = normalize_parent(node.get("parentCode"))

    label = normalize_text(node.get("label"))
    if not label:
        raise ValueError(f"node {code} is missing a label")

    definition = normalize_text(node.get("definition")) or ""
    examples = normalize_text(node.get("examples"))
    is_leaf = bool(node.get("isLeaf")) if node.get("isLeaf") is not None else level == max_depth

    return {
        "id": str(uuid4()),
        "code": code,
        "level": level,
        "label": label,
        "definition": definition,
        "examples": examples,
        "parentCode": parent_code,
        "isLeaf": is_leaf,
        "taxonomyKey": taxonomy_key,
    }
