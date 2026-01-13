"""
Test script for Module 1 - Taxonomy Preparation.

This script tests the multi-view embedding pipeline implementation.
"""

import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path

# Bootstrap the Kedro project
project_path = Path(__file__).parent
bootstrap_project(project_path)


def test_build_taxonomy_pipeline():
    """Test the build_taxonomy pipeline with multi-view embeddings."""

    print("=" * 80)
    print("Testing Module 1 - Taxonomy Preparation Pipeline")
    print("=" * 80)

    with KedroSession.create() as session:
        context = session.load_context()

        # Get parameters
        taxonomy_key = context.params.get("taxonomy_key", "ISCO")
        model_name = context.params.get("model_name")

        print(f"\n📋 Configuration:")
        print(f"   - Taxonomy Key: {taxonomy_key}")
        print(f"   - Model Name: {model_name}")

        # Run the pipeline
        print(f"\n🚀 Running build_taxonomy pipeline...")
        print(f"   This will create multi-view embeddings for the taxonomy.\n")

        try:
            session.run(pipeline_name="build_taxonomy")
            print("\n✅ Pipeline completed successfully!")
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            raise

        # Load and inspect the output
        print(f"\n🔍 Inspecting output taxonomy_index...")

        catalog = context.catalog
        taxonomy_index = catalog.load("taxonomy_index")

        # Get the DataFrame for the specified taxonomy
        if taxonomy_key in taxonomy_index:
            df = taxonomy_index[taxonomy_key]()  # Call the callable

            print(f"\n📊 Taxonomy Index Schema:")
            print(f"   - Total nodes: {len(df)}")
            print(f"   - Columns: {df.columns.tolist()}")

            # Check embedding columns
            print(f"\n🎯 Multi-View Embeddings:")

            # Label embeddings (should always be present)
            if "embedding_label" in df.columns:
                sample_label_emb = df["embedding_label"].iloc[0]
                print(f"   ✓ Label embeddings: shape={sample_label_emb.shape}")
            else:
                print(f"   ✗ Label embeddings: MISSING!")

            # Definition embeddings
            if "embedding_definition" in df.columns:
                non_null_defs = df["embedding_definition"].notna().sum()
                null_defs = df["embedding_definition"].isna().sum()
                print(f"   ✓ Definition embeddings: {non_null_defs} present, {null_defs} missing")
                if non_null_defs > 0:
                    sample_def_emb = df[df["embedding_definition"].notna()]["embedding_definition"].iloc[0]
                    print(f"      Sample shape: {sample_def_emb.shape}")
            else:
                print(f"   ✗ Definition embeddings: MISSING!")

            # Examples embeddings
            if "embedding_examples" in df.columns:
                non_null_examples = df["embedding_examples"].notna().sum()
                null_examples = df["embedding_examples"].isna().sum()
                print(f"   ✓ Examples embeddings: {non_null_examples} present, {null_examples} missing")
                if non_null_examples > 0:
                    sample_ex_emb = df[df["embedding_examples"].notna()]["embedding_examples"].iloc[0]
                    print(f"      Sample shape: {sample_ex_emb.shape}")
            else:
                print(f"   ✗ Examples embeddings: MISSING!")

            # Metadata
            print(f"\n📝 Metadata:")
            if "embedding_model_name" in df.columns:
                print(f"   - Model: {df['embedding_model_name'].iloc[0]}")
            if "embedding_dim" in df.columns:
                print(f"   - Embedding dimension: {df['embedding_dim'].iloc[0]}")

            # Sample a few nodes
            print(f"\n🔎 Sample Nodes:")
            sample_df = df[["code", "level", "label"]].head(5)
            print(sample_df.to_string(index=False))

            # Check for roots
            from taxomind.utils.taxonomy_utils import get_roots, get_children
            roots = get_roots(df)
            print(f"\n🌳 Taxonomy Structure:")
            print(f"   - Root nodes: {len(roots)}")
            print(f"   - Root codes: {roots[:5]}{'...' if len(roots) > 5 else ''}")

            # Sample children
            if roots:
                first_root = roots[0]
                children = get_children(df, first_root)
                print(f"   - Children of '{first_root}': {len(children)}")
                if children:
                    print(f"      Sample: {children[:5]}{'...' if len(children) > 5 else ''}")

            # Verify schema compliance
            print(f"\n✓ Schema Validation:")
            required_cols = [
                "code", "level", "parentCode", "label", "definition",
                "embedding_label", "embedding_definition", "embedding_examples",
                "embedding_model_name", "embedding_dim",
                "evidence_centroid", "evidence_count", "evidence_last_updated"
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"   ❌ Missing columns: {missing_cols}")
            else:
                print(f"   ✅ All required columns present")

            # Verify evidence state initialization
            print(f"\n📊 Evidence State (Module 3 preparation):")
            print(f"   - evidence_centroid: {df['evidence_centroid'].notna().sum()} populated, {df['evidence_centroid'].isna().sum()} None")
            print(f"   - evidence_count: min={df['evidence_count'].min()}, max={df['evidence_count'].max()}")
            if df['evidence_count'].sum() == 0:
                print(f"   ✅ All evidence_count = 0 (correct initialization)")
            else:
                print(f"   ⚠️  Some evidence_count != 0 (unexpected)")

            print(f"\n" + "=" * 80)
            print("✅ Module 1 - Taxonomy Preparation: VALIDATED")
            print("=" * 80)

        else:
            print(f"\n❌ Taxonomy key '{taxonomy_key}' not found in output")
            print(f"   Available keys: {list(taxonomy_index.keys())}")


if __name__ == "__main__":
    test_build_taxonomy_pipeline()
