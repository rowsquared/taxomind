#!/usr/bin/env python3
"""
Validation script for supervised training pipeline setup.

This script checks that all required components are in place before running
the supervised training pipeline.

Usage:
    python scripts/validate_supervised_setup.py
"""

import sys
from pathlib import Path
from typing import List, Tuple
import importlib.util


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def check_file_exists(filepath: Path, description: str) -> Tuple[bool, str]:
    """Check if a file exists."""
    if filepath.exists():
        return True, f"{Colors.GREEN}✓{Colors.END} {description}: {filepath}"
    else:
        return False, f"{Colors.RED}✗{Colors.END} {description}: {filepath} (NOT FOUND)"


def check_module_installed(module_name: str) -> Tuple[bool, str]:
    """Check if a Python module is installed."""
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        return True, f"{Colors.GREEN}✓{Colors.END} {module_name} installed"
    else:
        return False, f"{Colors.RED}✗{Colors.END} {module_name} NOT installed"


def check_labeled_dataset(filepath: Path) -> Tuple[bool, str]:
    """Check labeled dataset format."""
    if not filepath.exists():
        return False, f"{Colors.RED}✗{Colors.END} Labeled dataset not found: {filepath}"

    try:
        import pandas as pd

        df = pd.read_csv(filepath, encoding='utf-8')

        # Check required columns
        required_cols = ['text', 'leaf_code']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return False, (
                f"{Colors.RED}✗{Colors.END} Labeled dataset missing columns: "
                f"{', '.join(missing_cols)}"
            )

        # Check minimum samples
        num_samples = len(df)
        if num_samples < 10:
            return False, (
                f"{Colors.YELLOW}⚠{Colors.END} Labeled dataset has only {num_samples} samples "
                f"(recommended: 100+)"
            )

        # Count samples per level
        level_counts = {}
        for leaf_code in df['leaf_code']:
            for level in range(1, 6):
                code_parts = str(leaf_code).split('.')
                if len(code_parts) >= level:
                    level_code = '.'.join(code_parts[:level])
                    level_key = f"level_{level}"
                    if level_key not in level_counts:
                        level_counts[level_key] = set()
                    level_counts[level_key].add(level_code)

        msg = f"{Colors.GREEN}✓{Colors.END} Labeled dataset valid: {num_samples} samples\n"
        for level in range(1, 6):
            level_key = f"level_{level}"
            if level_key in level_counts:
                num_classes = len(level_counts[level_key])
                msg += f"    Level {level}: {num_classes} unique classes\n"

        return True, msg.rstrip()

    except Exception as e:
        return False, f"{Colors.RED}✗{Colors.END} Error reading labeled dataset: {e}"


def main():
    """Run all validation checks."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}=" * 60)
    print("Supervised Training Pipeline - Setup Validation")
    print("=" * 60 + Colors.END + "\n")

    project_root = Path(__file__).parent.parent
    all_passed = True
    results = []

    # 1. Check Python dependencies
    print(f"{Colors.BOLD}1. Checking Python Dependencies{Colors.END}")
    dependencies = [
        'transformers',
        'torch',
        'sklearn',
        'tqdm',
        'pandas',
        'kedro',
    ]

    for dep in dependencies:
        passed, msg = check_module_installed(dep)
        results.append((passed, msg))
        print(f"   {msg}")
        all_passed &= passed

    print()

    # 2. Check pipeline files
    print(f"{Colors.BOLD}2. Checking Pipeline Files{Colors.END}")
    pipeline_files = [
        (project_root / "src/taxomind/pipelines/training_supervised/__init__.py",
         "Pipeline __init__.py"),
        (project_root / "src/taxomind/pipelines/training_supervised/pipeline.py",
         "Pipeline definition"),
        (project_root / "src/taxomind/pipelines/training_supervised/nodes.py",
         "Training nodes"),
        (project_root / "src/taxomind/pipelines/training_supervised/inference.py",
         "Inference utilities"),
    ]

    for filepath, description in pipeline_files:
        passed, msg = check_file_exists(filepath, description)
        results.append((passed, msg))
        print(f"   {msg}")
        all_passed &= passed

    print()

    # 3. Check configuration files
    print(f"{Colors.BOLD}3. Checking Configuration Files{Colors.END}")
    config_files = [
        (project_root / "conf/base/parameters/supervised.yml",
         "Parameters config"),
        (project_root / "conf/base/catalog_supervised.yml",
         "Catalog config"),
    ]

    for filepath, description in config_files:
        passed, msg = check_file_exists(filepath, description)
        results.append((passed, msg))
        print(f"   {msg}")
        all_passed &= passed

    print()

    # 4. Check data directories
    print(f"{Colors.BOLD}4. Checking Data Directories{Colors.END}")
    data_dirs = [
        (project_root / "data/01_raw", "Raw data directory"),
        (project_root / "data/03_primary", "Primary data directory"),
        (project_root / "data/06_models", "Models directory"),
    ]

    for dirpath, description in data_dirs:
        dirpath.mkdir(parents=True, exist_ok=True)
        passed, msg = check_file_exists(dirpath, description)
        results.append((passed, msg))
        print(f"   {msg}")
        all_passed &= passed

    print()

    # 5. Check labeled dataset (optional)
    print(f"{Colors.BOLD}5. Checking Labeled Dataset (Optional){Colors.END}")
    labeled_dataset_paths = [
        project_root / "data/01_raw/labeled_dataset.csv",
        project_root / "data/01_raw/labeled_dataset_example.csv",
    ]

    dataset_found = False
    for dataset_path in labeled_dataset_paths:
        if dataset_path.exists():
            passed, msg = check_labeled_dataset(dataset_path)
            results.append((passed, msg))
            print(f"   {msg}")
            dataset_found = True
            break

    if not dataset_found:
        msg = (
            f"{Colors.YELLOW}⚠{Colors.END} No labeled dataset found. "
            f"Create data/01_raw/labeled_dataset.csv before training."
        )
        results.append((False, msg))
        print(f"   {msg}")

    print()

    # 6. Check documentation
    print(f"{Colors.BOLD}6. Checking Documentation{Colors.END}")
    doc_files = [
        (project_root / "src/taxomind/pipelines/training_supervised/README.md",
         "Pipeline README"),
        (project_root / "SUPERVISED_QUICKSTART.md",
         "Quick start guide"),
        (project_root / "SUPERVISED_PIPELINE_SUMMARY.md",
         "Pipeline summary"),
    ]

    for filepath, description in doc_files:
        passed, msg = check_file_exists(filepath, description)
        results.append((passed, msg))
        print(f"   {msg}")
        all_passed &= passed

    print()

    # Summary
    print(f"{Colors.BOLD}{Colors.BLUE}=" * 60)
    print("Validation Summary")
    print("=" * 60 + Colors.END)

    total_checks = len(results)
    passed_checks = sum(1 for passed, _ in results if passed)

    print(f"\nTotal checks: {total_checks}")
    print(f"Passed: {Colors.GREEN}{passed_checks}{Colors.END}")
    print(f"Failed: {Colors.RED}{total_checks - passed_checks}{Colors.END}\n")

    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All checks passed!{Colors.END}")
        print(f"\n{Colors.BOLD}Next steps:{Colors.END}")
        print(f"1. Prepare your labeled dataset (data/01_raw/labeled_dataset.csv)")
        print(f"2. Run: {Colors.BLUE}kedro run --pipeline=training_supervised{Colors.END}")
        print(f"3. Check models in: {Colors.BLUE}data/06_models/{Colors.END}\n")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some checks failed{Colors.END}")
        print(f"\n{Colors.BOLD}Please fix the issues above before running the pipeline.{Colors.END}\n")

        # Provide specific help for common issues
        if not all(passed for passed, _ in results[:len(dependencies)]):
            print(f"{Colors.BOLD}To install missing dependencies:{Colors.END}")
            print(f"  pip install transformers torch scikit-learn tqdm\n")

        return 1


if __name__ == "__main__":
    sys.exit(main())
