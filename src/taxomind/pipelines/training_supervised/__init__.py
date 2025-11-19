"""
Supervised hierarchical classification pipeline using ModernBERT.

This pipeline trains level-specific classification models when labeled data is available.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
