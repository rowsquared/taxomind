"""Judge utilities for multilingual ambiguity resolution."""

from __future__ import annotations

import json
from typing import Any, Dict, Sequence

import requests


def _fallback_output(
    candidates: Sequence[Dict[str, Any]],
    route: Sequence[Dict[str, Any]],
    reason: str | None = None,
) -> Dict[str, Any]:
    top_candidate = max(candidates, key=lambda item: item.get("score", 0.0))
    return {
        "decision": top_candidate,
        "reason": reason
        or "Fallback similarity decision while multilingual judge unavailable.",
        "route": list(route),
    }


def _build_prompt(input_text: str, candidates: Sequence[Dict[str, Any]], taxonomy_context: str) -> str:
    candidate_payload = json.dumps(candidates, ensure_ascii=False, indent=2)
    prompt = f"""
Input text (any language allowed):
{input_text}

Candidates (JSON with code, label, score, level, parentCode, isLeaf):
{candidate_payload}

Taxonomy context:
{taxonomy_context}

Task: Select the best candidate considering hierarchical coherence and cross-lingual similarity.
Respond only with JSON using:
{{"code": "<best_code>", "reason": "explanation"}}
"""
    return prompt.strip()


def _extract_json_block(response_text: str) -> Dict[str, Any] | None:
    text = response_text.strip()
    blocks = []
    if "```" in text:
        parts = text.split("```")
        for idx in range(1, len(parts), 2):
            block = parts[idx]
            if block.startswith("json"):
                blocks.append(block[4:].strip())
            else:
                blocks.append(block.strip())
    candidates = [text] + blocks
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def run_judge(
    input_text: str,
    candidates: Sequence[Dict[str, Any]],
    taxonomy_context: str,
    model_name: str,
    route: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Multilingual LLM judge resolving ambiguity via local Ollama models."""

    if not candidates:
        return {"decision": None, "reason": "No candidates available.", "route": list(route)}

    prompt = _build_prompt(input_text, candidates, taxonomy_context)
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_name,
                "stream": False,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a multilingual taxonomy judge. Use definitions in their"
                            " native languages, avoid translation unless absolutely required,"
                            " and respond in the language detected in the input text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload.get("message", {}).get("content") or payload.get("response", "")
        response_text = content or ""
        parsed = _extract_json_block(response_text)
        if not parsed or "code" not in parsed:
            return _fallback_output(candidates, route)
        selected_code = parsed["code"]
        matched = next(
            (candidate for candidate in candidates if candidate.get("code") == selected_code),
            None,
        )
        if not matched:
            return _fallback_output(
                candidates, route, reason=parsed.get("reason") or response_text
            )
        reason = parsed.get("reason") or response_text
        return {"decision": matched, "reason": reason, "route": list(route)}
    except Exception:
        return _fallback_output(candidates, route)
