"""
Test script for Module 2 - Inference Pipeline.

This script tests the hierarchical classification inference pipeline.
"""

import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path

# Bootstrap the Kedro project
project_path = Path(__file__).parent
bootstrap_project(project_path)


def test_inference_pipeline():
    """Test the inference pipeline with a sample query."""

    print("=" * 80)
    print("Testing Module 2 - Inference Pipeline")
    print("=" * 80)

    test_queries = [
        "I work as a software developer",
        "I teach at a university",
        "I work in education",
        "I manage a retail store",
    ]

    with KedroSession.create(
        runtime_params={"inference_query_input": test_queries}
    ) as session:
        context = session.load_context()

        # Get parameters
        model_name = context.params.get("model_name")
        taxonomy_key = context.params.get("taxonomy_key")

        print("\n📋 Configuration:")
        print(f"   - Taxonomy Key: {taxonomy_key}")
        print(f"   - Model Name: {model_name}")

        results = []

        try:
            # Run inference pipeline (processes all queries at once)
            num_queries = len(test_queries)
            print(
                f"\n🚀 Running inference pipeline for "
                f"{num_queries} queries..."
            )
            # Run full pipeline
            # The pipeline will load taxonomy_index (from disk) and
            # inference_query_input (from memory)
            # All intermediate MemoryDatasets will be created fresh
            session.run(pipeline_name="inference")

            # Get predictions DataFrame
            predictions_df = context.catalog.load("inference_predictions_df")

            print("\n✅ Batch inference completed!")
            print(f"   Processed: {len(predictions_df)} queries")

            # Display results for each query
            for idx, row in predictions_df.iterrows():
                print(f"\n{'='*80}")
                print(f"Query {row['query_id']}: {row['query']}")
                print(f"{'='*80}")

                if pd.notna(row.get('error')):
                    print(f"\n   ❌ Error: {row['error']}")
                    results.append({
                        "query": row['query'],
                        "success": False,
                        "error": row['error']
                    })
                else:
                    print("\n   ✅ Prediction:")
                    print(f"      Code: {row['predicted_code']}")
                    print(f"      Label: {row['predicted_label']}")
                    print(f"      Level: {row['predicted_level']}")
                    print(f"      Score: {row['score']:.3f}")
                    print(f"      Ambiguous: {row['ambiguous']}")
                    print(f"      Stopping Reason: {row['stopping_reason']}")

                    if pd.notna(row['path']) and row['path']:
                        print(f"      Path: {' → '.join(row['path'])}")

                    if (pd.notna(row['alternatives'])
                            and row['alternatives']):
                        num_alts = len(row['alternatives'])
                        print(f"\n      Alternatives ({num_alts}):")
                        for alt in row['alternatives'][:3]:
                            alt_code = alt['code']
                            alt_label = alt['label']
                            alt_score = alt['score']
                            print(
                                f"         - {alt_code}: {alt_label} "
                                f"(score={alt_score:.3f})"
                            )

                    results.append({
                        "query": row['query'],
                        "success": True,
                        "prediction": {
                            "code": row['predicted_code'],
                            "label": row['predicted_label'],
                            "level": row['predicted_level'],
                            "score": row['score'],
                            "stopping_reason": row['stopping_reason'],
                        }
                    })

        except Exception as e:
            print(f"\n❌ Batch inference failed with error: {e}")
            import traceback
            traceback.print_exc()
            # Mark all queries as failed
            for query in test_queries:
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })

        # Summary
        print(f"\n{'='*80}")
        print("Summary")
        print(f"{'='*80}")

        successful = sum(1 for r in results if r['success'])
        print(f"\nTotal queries: {len(test_queries)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(test_queries) - successful}")

        if successful == len(test_queries):
            print("\n✅ Module 2 - Inference Pipeline: ALL TESTS PASSED")
        else:
            print("\n⚠️  Module 2 - Inference Pipeline: SOME TESTS FAILED")

        # Stopping reason distribution
        print("\n📊 Stopping Reasons:")
        from collections import Counter
        stopping_reasons = [
            r['prediction']['stopping_reason']
            for r in results if r['success']
        ]
        for reason, count in Counter(stopping_reasons).items():
            print(f"   - {reason}: {count}")

        print(f"\n{'='*80}")


if __name__ == "__main__":
    test_inference_pipeline()
