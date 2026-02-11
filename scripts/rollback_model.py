"""
Rollback Production to a previous model version (Step 16 – Rollback & Governance).

Restores a given version to Production and archives the current Production version.
Explicit, auditable operation; no automatic rollback.

Usage:
    python scripts/rollback_model.py --version 4
    python scripts/rollback_model.py --version 4 --model-name credit-risk-model
    python scripts/rollback_model.py --version 4 --no-archive-current
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
        description="Rollback Production to a previous model version (MLflow Model Registry)"
    )
    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Model version to restore to Production (e.g. 4)",
    )
    parser.add_argument(
        "--model-name",
        default="credit-risk-model",
        help="Registered model name (default: credit-risk-model)",
    )
    parser.add_argument(
        "--no-archive-current",
        action="store_true",
        help="Do not transition current Production to Archived (default: archive it)",
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
    print("MODEL ROLLBACK (MLflow)")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Target version to restore: {args.version}")
    print("=" * 80)

    try:
        # Archive current Production (unless --no-archive-current)
        if not args.no_archive_current:
            try:
                current = client.get_latest_versions(args.model_name, stages=["Production"])
                for mv in current:
                    if int(mv.version) == args.version:
                        print(f"  Version {args.version} is already in Production.")
                        print("\n✅ No change needed.")
                        return
                    client.transition_model_version_stage(
                        name=args.model_name,
                        version=mv.version,
                        stage="Archived",
                    )
                    print(f"  Archived current Production: v{mv.version}")
            except Exception as e:
                print(f"  (No current Production or error archiving: {e})")

        # Transition the target version to Production
        client.transition_model_version_stage(
            name=args.model_name,
            version=str(args.version),
            stage="Production",
        )
        print(f"\n✅ Rollback complete: v{args.version} is now Production")

        prod = client.get_latest_versions(args.model_name, stages=["Production"])
        if prod:
            mv = prod[0]
            print(f"\nCurrent Production: v{mv.version} (run_id: {mv.run_id})")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
