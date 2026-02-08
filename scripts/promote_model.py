"""
Promote a model version to Production in MLflow Model Registry (Step 15).

Usage:
    python scripts/promote_model.py --version 3
    python scripts/promote_model.py --version 3 --model-name credit-risk-model
    python scripts/promote_model.py --version 3 --archive-current
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import mlflow
from mlflow.tracking import MlflowClient


def main():
    parser = argparse.ArgumentParser(
        description="Promote a model version to Production in MLflow Model Registry"
    )
    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Model version number to promote (e.g. 3)",
    )
    parser.add_argument(
        "--model-name",
        default="credit-risk-model",
        help="Registered model name (default: credit-risk-model)",
    )
    parser.add_argument(
        "--archive-current",
        action="store_true",
        help="Transition current Production version to Archived",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: ./mlruns or MLFLOW_TRACKING_URI)",
    )
    args = parser.parse_args()

    tracking_uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or str(project_root / "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    print("=" * 80)
    print("MODEL PROMOTION (MLflow)")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Version: {args.version}")
    print(f"Target stage: Production")
    print("=" * 80)

    try:
        # Optionally archive current Production version
        if args.archive_current:
            try:
                current = client.get_latest_versions(args.model_name, stages=["Production"])
                for mv in current:
                    client.transition_model_version_stage(
                        name=args.model_name,
                        version=mv.version,
                        stage="Archived",
                    )
                    print(f"  Archived previous Production: v{mv.version}")
            except Exception as e:
                print(f"  (No current Production or error archiving: {e})")

        # Transition the selected version to Production
        client.transition_model_version_stage(
            name=args.model_name,
            version=str(args.version),
            stage="Production",
        )
        print(f"\n✅ Version {args.version} promoted to Production")

        # Show current Production
        prod = client.get_latest_versions(args.model_name, stages=["Production"])
        if prod:
            mv = prod[0]
            print(f"\nCurrent Production: v{mv.version} (run_id: {mv.run_id})")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
