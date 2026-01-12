def build_text_variable(
    fields: dict[str, str],
    *,
    separator: str = "\n",
    max_chars_per_field: int | None = 512,
    drop_empty: bool = True,
) -> str:
    """
    Build a canonical, deterministic text representation from an arbitrary
    {field_name: value} dictionary for embedding.

    Rules:
    - Order-invariant: fields are sorted by field name
    - Field-aware: field names are preserved in the text
    - Robust to missing / empty values
    - Optional per-field length cap to avoid domination by verbose fields
    """

    if not fields:
        return ""

    parts: list[str] = []

    for field_name in sorted(fields.keys()):
        value = fields.get(field_name)

        if value is None:
            continue

        value = str(value).strip()

        if drop_empty and not value:
            continue

        if max_chars_per_field is not None:
            value = value[:max_chars_per_field]

        parts.append(f"{field_name}: {value}")

    return separator.join(parts)