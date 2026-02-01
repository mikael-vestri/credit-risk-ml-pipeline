"""
Artifact registry for model versioning and tracking.

Tracks models, datasets, training runs, and their metadata.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib
import shutil

logger = logging.getLogger(__name__)


class ArtifactRegistry:
    """
    Local artifact registry for tracking models and training runs.
    
    Maintains a registry.json file with metadata about:
    - Models (versions, paths, metrics, timestamps)
    - Training runs (config, datasets, results)
    - Dataset versions (hashes, paths, metadata)
    """
    
    def __init__(self, registry_path: Path):
        """
        Initialize artifact registry.
        
        Parameters
        ----------
        registry_path : Path
            Path to registry directory (will contain registry.json)
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "registry.json"
        self._registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}. Creating new registry.")
        
        return {
            "models": {},
            "training_runs": {},
            "datasets": {},
            "metadata": {
                "created": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
    
    def _save_registry(self):
        """Save registry to disk."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self._registry, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            raise
    
    def register_model(
        self,
        model_name: str,
        model_path: Path,
        metadata_path: Path,
        metrics: Optional[Dict[str, Any]] = None,
        training_run_id: Optional[str] = None,
        stage: str = "staging"
    ) -> str:
        """
        Register a model in the registry.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        model_path : Path
            Path to model file (.pkl)
        metadata_path : Path
            Path to metadata file (.json)
        metrics : Optional[Dict[str, Any]]
            Model performance metrics
        training_run_id : Optional[str]
            ID of training run that produced this model
        stage : str
            Model stage: "staging", "production", "archived"
            
        Returns
        -------
        str
            Model version ID
        """
        # Generate version ID (timestamp-based)
        version_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate file hash for integrity
        model_hash = self._calculate_file_hash(model_path)
        
        model_entry = {
            "version_id": version_id,
            "model_name": model_name,
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "model_hash": model_hash,
            "stage": stage,
            "registered_at": datetime.now().isoformat(),
            "metrics": metrics or {},
            "training_run_id": training_run_id
        }
        
        if model_name not in self._registry["models"]:
            self._registry["models"][model_name] = []
        
        self._registry["models"][model_name].append(model_entry)
        
        # Update latest version pointer
        self._registry["models"][model_name].sort(
            key=lambda x: x["registered_at"],
            reverse=True
        )
        
        self._save_registry()
        logger.info(f"Registered model: {version_id} (stage: {stage})")
        
        return version_id
    
    def register_training_run(
        self,
        run_id: str,
        config: Dict[str, Any],
        dataset_version: Optional[str] = None,
        results: Optional[Dict[str, Any]] = None
    ):
        """
        Register a training run.
        
        Parameters
        ----------
        run_id : str
            Unique training run ID
        config : Dict[str, Any]
            Training configuration (hyperparameters, etc.)
        dataset_version : Optional[str]
            Dataset version used for training
        results : Optional[Dict[str, Any]]
            Training results (metrics, duration, etc.)
        """
        run_entry = {
            "run_id": run_id,
            "started_at": datetime.now().isoformat(),
            "config": config,
            "dataset_version": dataset_version,
            "results": results or {}
        }
        
        self._registry["training_runs"][run_id] = run_entry
        self._save_registry()
        logger.info(f"Registered training run: {run_id}")
    
    def update_training_run(
        self,
        run_id: str,
        results: Dict[str, Any],
        completed: bool = True
    ):
        """Update training run with results."""
        if run_id not in self._registry["training_runs"]:
            raise ValueError(f"Training run {run_id} not found")
        
        self._registry["training_runs"][run_id]["results"].update(results)
        if completed:
            self._registry["training_runs"][run_id]["completed_at"] = datetime.now().isoformat()
        
        self._save_registry()
    
    def promote_model(self, model_name: str, version_id: str, target_stage: str = "production"):
        """
        Promote a model to a new stage (e.g., staging -> production).
        
        Parameters
        ----------
        model_name : str
            Name of the model
        version_id : str
            Version ID to promote
        target_stage : str
            Target stage (usually "production")
        """
        if model_name not in self._registry["models"]:
            raise ValueError(f"Model {model_name} not found")
        
        # Find the model version
        model_versions = self._registry["models"][model_name]
        model_entry = None
        for entry in model_versions:
            if entry["version_id"] == version_id:
                model_entry = entry
                break
        
        if model_entry is None:
            raise ValueError(f"Model version {version_id} not found")
        
        # Demote previous production model if exists
        if target_stage == "production":
            for entry in model_versions:
                if entry["stage"] == "production" and entry["version_id"] != version_id:
                    entry["stage"] = "archived"
                    logger.info(f"Demoted previous production model: {entry['version_id']}")
        
        # Promote the model
        model_entry["stage"] = target_stage
        model_entry["promoted_at"] = datetime.now().isoformat()
        
        self._save_registry()
        logger.info(f"Promoted model {version_id} to {target_stage}")
    
    def get_production_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get the current production model for a given model name."""
        if model_name not in self._registry["models"]:
            return None
        
        for entry in self._registry["models"][model_name]:
            if entry["stage"] == "production":
                return entry
        
        return None
    
    def list_models(self, model_name: Optional[str] = None, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        """List models, optionally filtered by name and stage."""
        all_models = []
        
        if model_name:
            if model_name in self._registry["models"]:
                all_models = self._registry["models"][model_name]
            else:
                return []
        else:
            for models in self._registry["models"].values():
                all_models.extend(models)
        
        if stage:
            all_models = [m for m in all_models if m["stage"] == stage]
        
        return sorted(all_models, key=lambda x: x["registered_at"], reverse=True)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of registry contents."""
        model_count = sum(len(versions) for versions in self._registry["models"].values())
        production_count = sum(
            1 for models in self._registry["models"].values()
            for m in models if m["stage"] == "production"
        )
        
        return {
            "total_models": model_count,
            "production_models": production_count,
            "training_runs": len(self._registry["training_runs"]),
            "model_names": list(self._registry["models"].keys()),
            "last_updated": datetime.now().isoformat()
        }
