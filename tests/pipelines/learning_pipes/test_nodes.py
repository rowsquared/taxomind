import numpy as np
import pandas as pd

from taxomind.pipelines.learning_pipes import nodes


def test_select_deepest_annotation():
    annotations = [
        {"level": 1, "nodeCode": "A"},
        {"level": 3, "nodeCode": "A11"},
        {"level": 2, "nodeCode": "A1"},
    ]

    deepest = nodes._select_deepest_annotation(annotations)

    assert deepest == (3, "A11")


def test_update_centroid_from_empty():
    new_embedding = np.array([3.0, 4.0], dtype=np.float32)
    updated = nodes._update_centroid(None, 0, new_embedding)

    assert np.allclose(updated, np.array([0.6, 0.8], dtype=np.float32))


def test_update_centroid_with_existing():
    existing = np.array([1.0, 0.0], dtype=np.float32)
    new_embedding = np.array([0.0, 2.0], dtype=np.float32)
    updated = nodes._update_centroid(existing, 1, new_embedding)

    expected = np.array([1.0, 0.0], dtype=np.float32)
    expected = (expected + np.array([0.0, 1.0], dtype=np.float32)) / 2.0
    expected = expected / np.linalg.norm(expected)

    assert np.allclose(updated, expected)


def test_apply_evidence_updates_only_target_node():
    taxonomy_df = pd.DataFrame(
        [
            {
                "code": "A",
                "taxonomyKey": "TEST",
                "evidence_centroid": np.array([1.0, 0.0], dtype=np.float32),
                "evidence_count": 1,
                "evidence_last_updated": None,
                "last_evidence_centroid": None,
                "last_evidence_count": 0,
                "last_evidence_last_updated": None,
            },
            {
                "code": "A1",
                "taxonomyKey": "TEST",
                "evidence_centroid": None,
                "evidence_count": 0,
                "evidence_last_updated": None,
                "last_evidence_centroid": None,
                "last_evidence_count": 0,
                "last_evidence_last_updated": None,
            },
        ]
    )

    updates_df = pd.DataFrame(
        [
            {
                "nodeCode": "A1",
                "embedding": np.array([0.0, 1.0], dtype=np.float32),
            }
        ]
    )

    updated_partitions, summary = nodes.apply_evidence_updates(
        updates_df, taxonomy_df, {}
    )

    updated_df = updated_partitions["TEST"]

    parent_row = updated_df.loc[updated_df["code"] == "A"].iloc[0]
    child_row = updated_df.loc[updated_df["code"] == "A1"].iloc[0]

    assert parent_row["evidence_count"] == 1
    assert np.allclose(
        parent_row["evidence_centroid"],
        np.array([1.0, 0.0], dtype=np.float32),
    )

    assert child_row["evidence_count"] == 1
    assert np.allclose(
        child_row["evidence_centroid"],
        np.array([0.0, 1.0], dtype=np.float32),
    )
    assert child_row["evidence_last_updated"] is not None
    assert child_row["last_evidence_centroid"] is None
    assert child_row["last_evidence_count"] == 0
    assert child_row["last_evidence_last_updated"] is None

    assert summary["updates_applied"] == 1
    assert summary["nodes_touched"] == 1
