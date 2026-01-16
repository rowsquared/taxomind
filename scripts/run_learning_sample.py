#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


def _pick_sample_row(df: pd.DataFrame, taxonomy_key: str) -> pd.Series:
    level_cols = [f"{taxonomy_key}_{level}" for level in range(1, 5)]
    available = [col for col in level_cols if col in df.columns]
    if not available:
        raise ValueError(
            f"No taxonomy columns found for {taxonomy_key}. "
            f"Expected one of: {level_cols}"
        )

    mask = df[available].notna().any(axis=1)
    if not mask.any():
        raise ValueError(f"No rows with {taxonomy_key} annotations found.")

    return df[mask].iloc[0]


def _build_annotations(row: pd.Series, taxonomy_key: str) -> list[dict]:
    annotations = []
    for level in range(1, 5):
        col = f"{taxonomy_key}_{level}"
        if col not in row.index:
            continue
        code = row.get(col)
        if pd.isna(code):
            continue
        annotations.append({"level": level, "nodeCode": str(code)})
    return annotations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Module 3 learning pipeline with a sample payload."
    )
    parser.add_argument(
        "--taxonomy-key",
        default="ISCO",
        help="Taxonomy key to use (default: ISCO).",
    )
    args = parser.parse_args()

    taxonomy_key = str(args.taxonomy_key).upper()

    project_path = Path(__file__).resolve().parents[1]
    bootstrap_project(project_path)

    with KedroSession.create(project_path=project_path) as session:
        context = session.load_context()
        df = context.catalog.load("classifai_validation_data")

        row = _pick_sample_row(df, taxonomy_key)
        annotations = _build_annotations(row, taxonomy_key)

        fields = {
            "job_description": row.get("field_job_description", ""),
            "occupation_description": row.get("field_occupation_description", ""),
        }

        payload = {
            "taxonomyKey": taxonomy_key,
            "sentences": [
                {
                    "sentenceId": row.get("id", "sample-1"),
                    "fields": fields,
                    "annotations": annotations,
                }
            ],
        }

        context.catalog.save("api_training_payload", {"sample": payload})

        session.run(pipeline_name="learning_pipe")
        summary = context.catalog.load("learning_update_summary")

    print("Learning update summary:")
    print(summary)


if __name__ == "__main__":
    main()
