"""
Script to promote a model from staging to production (Step 13).

Usage:
    python scripts/promote_model.py --model-name random_forest --version-id random_forest_20250126_120000
"""

import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from registry.artifact_registry import ArtifactRegistry


def main():
    """Promote a model to production."""
    parser = argparse.ArgumentParser(description="Promote a model to production")
    parser.add_argument(
        "--model-name",
        required=True,
        help="Name of the model (e.g., 'random_forest')"
    )
    parser.add_argument(
        "--version-id",
        required=True,
        help="Version ID of the model to promote"
    )
    parser.add_argument(
        "--stage",
        default="production",
        choices=["staging", "production", "archived"],
        help="Target stage (default: production)"
    )
    
    args = parser.parse_args()
    
    # Initialize registry
    registry_path = project_root / "artifacts" / "registry"
    registry = ArtifactRegistry(registry_path)
    
    print("="*80)
    print("MODEL PROMOTION")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Version: {args.version_id}")
    print(f"Target stage: {args.stage}")
    print("="*80)
    
    try:
        registry.promote_model(
            model_name=args.model_name,
            version_id=args.version_id,
            target_stage=args.stage
        )
        
        print(f"\n✅ Successfully promoted {args.version_id} to {args.stage}")
        
        # Show current production model
        prod_model = registry.get_production_model(args.model_name)
        if prod_model:
            print(f"\nCurrent production model:")
            print(f"  Version: {prod_model['version_id']}")
            print(f"  Registered: {prod_model['registered_at']}")
            if 'metrics' in prod_model:
                print(f"  Metrics: {prod_model['metrics']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
