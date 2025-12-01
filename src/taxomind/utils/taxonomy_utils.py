import re
from typing import Iterable, List, Sequence, Any, Dict, List
from uuid import uuid4
import pandas as pd
import uuid

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

    # Normalize parent_code to match taxonomy's parentCode format (None -> "__root__")
    normalized_parent = normalize_parent(parent_code)
    mask = (taxonomy["level"] == level) & (taxonomy["parentCode"] == normalized_parent)

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
        segment = compose_text(node)
        segments.append(segment)
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


def strip_parenthetical_markers(text: str) -> str:
    """
    Remove standalone parenthetical markers like '(a)', '(b)', etc., and tidy whitespace.
    """
    if text is None:
        return ""
    cleaned = re.sub(r"\s*\([A-Za-z]\)\s*", " ", str(text))
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


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


def get_parent_chain(taxonomy: pd.DataFrame, code: str) -> List[str]:
    """Trace parent relationships to return all ancestor codes from root to node."""
    parent_codes: List[str] = []
    visited: set[str] = set()
    current_code = code

    while current_code and current_code not in visited:
        visited.add(current_code)
        node_row = taxonomy[taxonomy["code"] == current_code]

        if node_row.empty:
            break

        parent = node_row.iloc[0].get("parentCode")

        # Stop at root (parentCode == "__root__")
        if parent == "__root__" or parent is None or pd.isna(parent):
            break

        parent_codes.insert(0, parent)
        current_code = parent

    return parent_codes


def identify_level(code):
    code = str(code)
    level = 1
    if len(code) in (1, 2) and code.isdigit():
        level = 2
    elif len(code) == 3 and code.isdigit():
        level = 3
    elif len(code) == 4 and code.isdigit():
        level = 4
    return level


def process_taxonomy_parent(df):
    """
    Process taxonomy dataframe to add id, parentCode, and isLeaf columns.

    The parentCode is determined by finding the closest preceding row with a lower level number
    (higher in hierarchy). For example:
    - Level 1 nodes have no parent (None)
    - Level 2 nodes have the most recent Level 1 node as parent
    - Level 3 nodes have the most recent Level 2 node as parent
    - Level 4 nodes have the most recent Level 3 node as parent

    Args:
        df: DataFrame with 'code' and 'level' columns

    Returns:
        DataFrame with added 'id', 'parentCode', and 'isLeaf' columns
    """
    # 1. Create unique ID for each row (UUID short form)
    df["id"] = [uuid.uuid4().hex[:8] for _ in range(len(df))]

    # 2. Generate parentCode by tracking the most recent parent at each level
    # Stack to keep track of the most recent code at each level
    parent_stack = {}  # level -> code
    parent_codes = []

    for _, row in df.iterrows():
        current_level = row["level"]
        current_code = row["code"]

        # For level 1, there is no parent
        if current_level == 1:
            parent_codes.append(None)
        else:
            # Find the parent: the most recent code with level = current_level - 1
            parent_level = current_level - 1
            parent_code = parent_stack.get(parent_level)
            parent_codes.append(parent_code)

        # Update the stack with the current code at its level
        parent_stack[current_level] = current_code

        # Clear all levels below this one (since we've moved to a new branch)
        levels_to_clear = [lvl for lvl in parent_stack.keys() if lvl > current_level]
        for lvl in levels_to_clear:
            del parent_stack[lvl]

    df["parentCode"] = parent_codes

    # 3. Compute isLeaf: 1 if code has >= 4 digits, else 0
    df["isLeaf"] = df["code"].apply(lambda x: 1 if len(x) >= 4 else 0)

    return df


def transform_taxonomy_to_training_format(taxonomy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform taxonomy definition into training data format.

    Creates rows with columns: text, label, label_code, level
    - One row for each definition
    - Additional rows for each example (split by newline)

    Args:
        taxonomy_df: DataFrame with columns: id, code, level, label, definition, examples, parentCode, isLeaf

    Returns:
        DataFrame with columns: text, label, label_code, level
    """
    records = []

    for _, row in taxonomy_df.iterrows():
        label = row.get("label")
        label_code = row.get("code")
        level = row.get("level")
        definition = row.get("definition")
        examples = row.get("examples")

        definition_text = (
            strip_parenthetical_markers(str(definition).strip())
            if definition and pd.notna(definition)
            else ""
        )

        # Add definition row if definition is not empty
        if definition_text:
            records.append({
                "text": definition_text,
                "label": label,
                "code": label_code,
                "level": level
            })

        # Add example rows if examples is not empty
        if examples and pd.notna(examples) and str(examples).strip():
            # Split by newline and create a row for each non-empty example
            example_lines = str(examples).split("\n")
            for example in example_lines:
                example_stripped = strip_parenthetical_markers(example.strip())
                if example_stripped:
                    records.append({
                        "text": example_stripped,
                        "label": label,
                        "code": label_code,
                        "level": level
                    })

    result_df = pd.DataFrame(records)
    return result_df


def jsonize_taxonomy(df_processed, taxonomy_key, level_names):

    nodes_payload = []
    for row in df_processed.to_dict('records'):
        nodes_payload.append(
            {
                "code": row.get("code"),
                "level": int(row.get("level", 0)),
                "label": normalize_text(row.get("label")) or "",
                "definition": normalize_text(row.get("definition")),
                "examples": normalize_text(row.get("examples")),
                "parentCode": row.get("parentCode"),
                "isLeaf": bool(int(row.get("isLeaf", 0))),
            }
        )

    taxonomy_request = {
        "action": "create",
        "taxonomy": {
            "key": taxonomy_key,
            "maxDepth": int(df_processed["level"].max()),
            "levelNames": level_names,
            "nodes": nodes_payload,
        },
    }
    return taxonomy_request


def pad_taxonomy_codes(
    df: pd.DataFrame,
    isco_col: str = "isco_level4",
    isic_col: str = "isic_level4",
    target_length: int = 4
) -> pd.DataFrame:
    """
    Pad taxonomy codes with leading zeros to ensure they have a consistent length.

    Args:
        df: DataFrame with taxonomy code columns
        isco_col: Name of the ISCO code column (default: "isco_level4")
        isic_col: Name of the ISIC code column (default: "isic_level4")
        target_length: Target length for codes (default: 4)

    Returns:
        DataFrame with padded code columns

    Example:
        >>> df = pd.DataFrame({
        ...     'isco_level4': ['251', '2512', '1'],
        ...     'isic_level4': ['62', '6201', 'A']
        ... })
        >>> padded_df = pad_taxonomy_codes(df)
        >>> padded_df['isco_level4'].tolist()
        ['0251', '2512', '0001']
    """
    result_df = df.copy()

    def pad_code(code):
        """Pad a code with leading zeros to reach target length."""
        if pd.isna(code) or code == "":
            return code

        code_str = str(code)

        # Only pad if the code is shorter than target length
        if len(code_str) < target_length:
            # Pad with zeros on the left
            return code_str.zfill(target_length)

        return code_str

    # Pad ISCO codes
    if isco_col in result_df.columns:
        result_df[isco_col] = result_df[isco_col].apply(pad_code)

    # Pad ISIC codes
    if isic_col in result_df.columns:
        result_df[isic_col] = result_df[isic_col].apply(pad_code)

    return result_df


def convert_to_training_json(
    df: pd.DataFrame,
    taxonomy_key: str,
    job_description_col: str = "field_job_description",
    industry_description_col: str = "field_industry_description",
    level_cols: Dict[str, List[str]] = None
) -> dict:
    """
    Convert enriched dataframe to training JSON format.

    Args:
        df: DataFrame with hierarchical taxonomy columns
        taxonomy_key: The taxonomy key (e.g., "ISCO" or "ISIC")
        job_description_col: Column name for job description
        industry_description_col: Column name for industry description
        level_cols: Dictionary mapping taxonomy keys to level column names.
                   If None, defaults to standard column names.
                   Example: {"ISCO": ["isco_level1", "isco_level2", "isco_level3", "isco_level4"],
                            "ISIC": ["isic_level1", "isic_level2", "isic_level3", "isic_level4"]}

    Returns:
        Dictionary in the training JSON format

    Example:
        >>> df = pd.DataFrame({
        ...     'field_job_description': ['Software engineer'],
        ...     'field_industry_description': ['Tech company'],
        ...     'isco_level1': ['2'],
        ...     'isco_level2': ['25'],
        ...     'isco_level3': ['251'],
        ...     'isco_level4': ['2512']
        ... })
        >>> result = convert_to_training_json(df, "ISCO")
    """
    # Default level columns if not provided
    if level_cols is None:
        level_cols = {
            "ISCO": ["isco_level1", "isco_level2", "isco_level3", "isco_level4"],
            "ISIC": ["isic_level1", "isic_level2", "isic_level3", "isic_level4"]
        }

    # Get the appropriate level columns for this taxonomy
    taxonomy_level_cols = level_cols.get(taxonomy_key, [])

    sentences = []

    for _, row in df.iterrows():
        # Generate a unique sentence ID
        sentence_id = uuid.uuid4().hex[:24]

        # Build fields
        fields = {
            "job_description": str(row.get(job_description_col, "")) if pd.notna(row.get(job_description_col)) else "",
            "industry_description": str(row.get(industry_description_col, "")) if pd.notna(row.get(industry_description_col)) else ""
        }

        # Build annotations from level columns
        annotations = []
        for level, col_name in enumerate(taxonomy_level_cols, start=1):
            if col_name in row.index:
                node_code = row.get(col_name)
                # Only add annotation if the code exists and is not empty
                if pd.notna(node_code) and str(node_code).strip():
                    annotations.append({
                        "level": level,
                        "nodeCode": str(node_code)
                    })

        sentence = {
            "sentenceId": sentence_id,
            "fields": fields,
            "annotations": annotations
        }

        sentences.append(sentence)

    result = {
        "taxonomyKey": taxonomy_key,
        "sentences": sentences
    }

    return result


def enrich_with_taxonomy_hierarchy(
    df: pd.DataFrame,
    taxonomy_definition: Dict[str, pd.DataFrame],
    isco_col: str = "isco_level4",
    isic_col: str = "isic_level4"
) -> pd.DataFrame:
    """
    Enrich a dataframe with hierarchical taxonomy levels.

    Given a dataframe with level 4 codes (leaf codes), this function adds columns
    for all parent levels (1, 2, 3) by looking up the hierarchy in the taxonomy definition.

    Args:
        df: DataFrame with columns containing level 4 codes
        taxonomy_definition: Dictionary with 'ISCO' and 'ISIC' keys, each containing
                           a taxonomy DataFrame with columns: code, level, parentCode, etc.
        isco_col: Name of the column containing ISCO level 4 codes (default: "isco_level4")
        isic_col: Name of the column containing ISIC level 4 codes (default: "isic_level4")

    Returns:
        DataFrame with additional columns:
        - isco_level1, isco_level2, isco_level3
        - isic_level1, isic_level2, isic_level3

    Example:
        >>> df = pd.DataFrame({
        ...     'field_job_description': ['Software engineer'],
        ...     'isco_level4': ['2512'],
        ...     'field_industry_description': ['Tech company'],
        ...     'isic_level4': ['6201']
        ... })
        >>> taxonomy_def = {
        ...     'ISCO': isco_taxonomy_df,
        ...     'ISIC': isic_taxonomy_df
        ... }
        >>> enriched_df = enrich_with_taxonomy_hierarchy(df, taxonomy_def)
        >>> enriched_df.columns
        ['field_job_description', 'isco_level4', 'field_industry_description',
         'isic_level4', 'isco_level1', 'isco_level2', 'isco_level3',
         'isic_level1', 'isic_level2', 'isic_level3']
    """
    result_df = df.copy()

    # Get taxonomy dataframes
    isco_taxonomy = taxonomy_definition.get("ISCO")()
    isic_taxonomy = taxonomy_definition.get("ISIC")()

    if isco_taxonomy is None or isic_taxonomy is None:
        raise ValueError("taxonomy_definition must contain both 'ISCO' and 'ISIC' keys")

    # Create lookup dictionaries for faster access
    # Map each code to its full hierarchy (level1, level2, level3)
    def build_hierarchy_lookup(taxonomy_df: pd.DataFrame) -> Dict[str, Dict[int, str]]:
        """Build a lookup dict mapping each code to its ancestor codes by level."""
        lookup = {}

        for _, row in taxonomy_df.iterrows():
            code = row.get("code")
            level = row.get("level")

            # Get all parent codes for this node
            parent_codes = get_parent_chain(taxonomy_df, code)

            # Build hierarchy dict for this code
            hierarchy = {}
            for parent_code in parent_codes:
                parent_row = taxonomy_df[taxonomy_df["code"] == parent_code]
                if not parent_row.empty:
                    parent_level = parent_row.iloc[0].get("level")
                    hierarchy[parent_level] = parent_code

            # Add the current code itself
            hierarchy[level] = code

            lookup[code] = hierarchy

        return lookup

    isco_lookup = build_hierarchy_lookup(isco_taxonomy)
    isic_lookup = build_hierarchy_lookup(isic_taxonomy)

    # Helper function to get level code from hierarchy
    def get_level_code(code: str, level: int, lookup: Dict[str, Dict[int, str]]) -> str:
        """Get the code at a specific level for a given leaf code."""
        if pd.isna(code) or code == "":
            return ""

        hierarchy = lookup.get(str(code), {})
        return hierarchy.get(level, "")

    # Enrich ISCO levels
    if isco_col in result_df.columns:
        result_df["isco_level1"] = result_df[isco_col].apply(
            lambda x: get_level_code(x, 1, isco_lookup)
        )
        result_df["isco_level2"] = result_df[isco_col].apply(
            lambda x: get_level_code(x, 2, isco_lookup)
        )
        result_df["isco_level3"] = result_df[isco_col].apply(
            lambda x: get_level_code(x, 3, isco_lookup)
        )

    # Enrich ISIC levels
    if isic_col in result_df.columns:
        result_df["isic_level1"] = result_df[isic_col].apply(
            lambda x: get_level_code(x, 1, isic_lookup)
        )
        result_df["isic_level2"] = result_df[isic_col].apply(
            lambda x: get_level_code(x, 2, isic_lookup)
        )
        result_df["isic_level3"] = result_df[isic_col].apply(
            lambda x: get_level_code(x, 3, isic_lookup)
        )

    return result_df
