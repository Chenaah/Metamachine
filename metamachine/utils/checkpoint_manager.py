"""
Checkpoint Management System for MetaMachine

This module provides utilities for managing model checkpoints including:
- Downloading models from URLs
- Caching models locally
- Model registry management
- Integrity verification (MD5 checksums)

Example usage:
    from metamachine.utils.checkpoint_manager import CheckpointManager
    
    # Download and load a model by name
    manager = CheckpointManager()
    model_path = manager.get_checkpoint("example_policy")
    
    # Or use a custom URL
    model_path = manager.download_from_url("https://example.com/model.pkl")
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import URLError, HTTPError


class CheckpointManager:
    """
    Manages model checkpoint downloading, caching, and verification.
    
    Attributes:
        cache_dir: Directory where downloaded models are stored
        registry: Dictionary of registered models with their metadata
    """
    
    # Default registry - can be extended by users
    DEFAULT_REGISTRY = {
        "three_modules_run_policy": {
            "url": "https://drive.google.com/uc?export=download&id=1AY3WMJMKrrmy7XIUuvt9Sx4T5IJTD3uX",
            "md5": "a1390cff3173f0fc48be92ef214be606",
            "config": "example_three_modules",
            "description": "Running policy for three-module configuration",
        },


        "quadruped_run_policy": {
            "url": "https://drive.google.com/uc?export=download&id=1FYNwe5PDMARQQ07_zBZRWKh4uGohdFC8",
            "md5": None,
            "config": "basic_quadruped",
            "description": "Running policy for basic quadruped",
        }
        # Add more models here following this pattern:
        # "model_name": {
        #     "url": "https://your-storage.com/model.pkl",
        #     "md5": "optional_md5_checksum",
        #     "config": "config_name",
        #     "description": "Model description",
        # },
    }
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        registry: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize the checkpoint manager.
        
        Args:
            cache_dir: Custom cache directory. If None, uses ~/.cache/metamachine/checkpoints
            registry: Custom model registry. If None, uses DEFAULT_REGISTRY
        """
        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "metamachine" / "checkpoints"
        else:
            self.cache_dir = Path(cache_dir)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load registry
        self.registry = registry if registry is not None else self.DEFAULT_REGISTRY.copy()
        
        # Try to load user registry if it exists
        self._load_user_registry()
    
    def _get_registry_path(self) -> Path:
        """Get path to user registry file."""
        return self.cache_dir / "registry.json"
    
    def _load_user_registry(self) -> None:
        """Load user-defined registry from cache directory."""
        registry_path = self._get_registry_path()
        if registry_path.exists():
            try:
                with open(registry_path, "r") as f:
                    user_registry = json.load(f)
                # Merge with default registry (user registry takes precedence)
                self.registry.update(user_registry)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load user registry: {e}")
    
    def save_registry(self) -> None:
        """Save current registry to disk."""
        registry_path = self._get_registry_path()
        try:
            with open(registry_path, "w") as f:
                json.dump(self.registry, f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to save registry: {e}")
    
    def register_model(
        self,
        name: str,
        url: str,
        config: Optional[str] = None,
        md5: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Register a new model in the registry.
        
        Args:
            name: Unique identifier for the model
            url: Download URL for the model
            config: Associated environment config name
            md5: Optional MD5 hash for verification
            description: Human-readable description of the model
        """
        self.registry[name] = {
            "url": url,
            "md5": md5,
            "config": config,
            "description": description or f"Model: {name}",
        }
        self.save_registry()
    
    def unregister_model(self, name: str, delete_file: bool = False) -> None:
        """
        Remove a model from the registry.
        
        Args:
            name: Name of the model to unregister
            delete_file: If True, also delete the cached file
        """
        if name in self.registry:
            if delete_file:
                model_path = self.cache_dir / f"{name}.pkl"
                if model_path.exists():
                    model_path.unlink()
                    print(f"Deleted cached file: {model_path}")
            
            del self.registry[name]
            self.save_registry()
            print(f"Unregistered model: {name}")
        else:
            print(f"Model not found in registry: {name}")
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered models.
        
        Returns:
            Dictionary of model names and their metadata
        """
        return self.registry.copy()
    
    def print_models(self) -> None:
        """Print all registered models in a readable format."""
        if not self.registry:
            print("No models registered.")
            return
        
        print("\nRegistered Models:")
        print("=" * 80)
        for name, info in self.registry.items():
            cached = (self.cache_dir / f"{name}.pkl").exists()
            status = "[CACHED]" if cached else "[NOT DOWNLOADED]"
            print(f"\n{name} {status}")
            print(f"  Description: {info.get('description', 'N/A')}")
            print(f"  Config: {info.get('config', 'N/A')}")
            print(f"  URL: {info['url']}")
            if info.get('md5'):
                print(f"  MD5: {info['md5']}")
        print("=" * 80)
    
    @staticmethod
    def compute_md5(file_path: Path) -> str:
        """
        Compute MD5 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MD5 hash as hexadecimal string
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def verify_checksum(self, file_path: Path, expected_md5: str) -> bool:
        """
        Verify file integrity using MD5 checksum.
        
        Args:
            file_path: Path to the file to verify
            expected_md5: Expected MD5 hash
            
        Returns:
            True if checksums match, False otherwise
        """
        actual_md5 = self.compute_md5(file_path)
        return actual_md5 == expected_md5
    
    def download_from_url(
        self,
        url: str,
        filename: Optional[str] = None,
        expected_md5: Optional[str] = None,
        force_download: bool = False,
        show_progress: bool = True,
    ) -> Path:
        """
        Download a checkpoint from a URL.
        
        Args:
            url: URL to download from
            filename: Custom filename to save as. If None, generates from URL hash
            expected_md5: Optional MD5 hash for verification
            force_download: If True, re-download even if file exists
            show_progress: If True, show download progress
            
        Returns:
            Path to the downloaded file
            
        Raises:
            URLError: If download fails
            ValueError: If MD5 verification fails
        """
        # Generate filename if not provided
        if filename is None:
            url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
            filename = f"checkpoint_{url_hash}.pkl"
        
        # Ensure .pkl extension
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        
        file_path = self.cache_dir / filename
        
        # Check if file already exists
        if file_path.exists() and not force_download:
            if expected_md5:
                if self.verify_checksum(file_path, expected_md5):
                    print(f"Using cached file: {file_path}")
                    return file_path
                else:
                    print("Cached file checksum mismatch. Re-downloading...")
                    file_path.unlink()
            else:
                print(f"Using cached file: {file_path}")
                return file_path
        
        # Download the file
        print(f"Downloading from: {url}")
        print(f"Saving to: {file_path}")
        
        try:
            if show_progress:
                def report_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    if total_size > 0:
                        percent = min(100, downloaded * 100 / total_size)
                        mb_downloaded = downloaded / 1024 / 1024
                        mb_total = total_size / 1024 / 1024
                        print(
                            f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                            end="",
                        )
                    else:
                        mb_downloaded = downloaded / 1024 / 1024
                        print(f"\rDownloaded: {mb_downloaded:.1f} MB", end="")
                
                urlretrieve(url, file_path, reporthook=report_progress)
                print()  # New line after progress
            else:
                urlretrieve(url, file_path)
            
            print("Download completed!")
            
            # Verify checksum if provided
            if expected_md5:
                print("Verifying checksum...")
                if self.verify_checksum(file_path, expected_md5):
                    print("Checksum verified!")
                else:
                    file_path.unlink()
                    raise ValueError(
                        f"Checksum verification failed! Expected: {expected_md5}"
                    )
            
            return file_path
            
        except (URLError, HTTPError) as e:
            if file_path.exists():
                file_path.unlink()
            raise URLError(f"Failed to download checkpoint: {e}")
    
    def get_checkpoint(
        self,
        name: str,
        force_download: bool = False,
        show_progress: bool = True,
    ) -> Path:
        """
        Get a checkpoint by name from the registry.
        
        Args:
            name: Name of the registered model
            force_download: If True, re-download even if cached
            show_progress: If True, show download progress
            
        Returns:
            Path to the checkpoint file
            
        Raises:
            ValueError: If model name not found in registry
        """
        if name not in self.registry:
            available = ", ".join(self.registry.keys())
            raise ValueError(
                f"Model '{name}' not found in registry. "
                f"Available models: {available}"
            )
        
        model_info = self.registry[name]
        print(f"\nModel: {name}")
        print(f"Description: {model_info.get('description', 'N/A')}")
        if model_info.get('config'):
            print(f"Recommended config: {model_info['config']}")
        
        return self.download_from_url(
            url=model_info["url"],
            filename=f"{name}.pkl",
            expected_md5=model_info.get("md5"),
            force_download=force_download,
            show_progress=show_progress,
        )
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a registered model.
        
        Args:
            name: Name of the model
            
        Returns:
            Dictionary of model metadata, or None if not found
        """
        return self.registry.get(name)
    
    def clear_cache(self, confirm: bool = False) -> None:
        """
        Clear all cached checkpoint files.
        
        Args:
            confirm: Must be True to actually delete files (safety check)
        """
        if not confirm:
            print("Warning: Set confirm=True to actually clear the cache")
            return
        
        count = 0
        for file_path in self.cache_dir.glob("*.pkl"):
            file_path.unlink()
            count += 1
        
        print(f"Cleared {count} cached checkpoint(s) from {self.cache_dir}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        pkl_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in pkl_files)
        
        return {
            "cache_dir": str(self.cache_dir),
            "num_files": len(pkl_files),
            "total_size_mb": total_size / 1024 / 1024,
            "files": [f.name for f in pkl_files],
        }
    
    def print_cache_info(self) -> None:
        """Print cache information in a readable format."""
        info = self.get_cache_info()
        print("\nCache Information:")
        print("=" * 80)
        print(f"Cache Directory: {info['cache_dir']}")
        print(f"Number of Files: {info['num_files']}")
        print(f"Total Size: {info['total_size_mb']:.2f} MB")
        if info['files']:
            print("\nCached Files:")
            for filename in info['files']:
                print(f"  - {filename}")
        print("=" * 80)


# Singleton instance for convenience
_default_manager = None


def get_default_manager() -> CheckpointManager:
    """
    Get the default global CheckpointManager instance.
    
    Returns:
        Global CheckpointManager instance
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = CheckpointManager()
    return _default_manager


# Convenience functions using the default manager
def get_checkpoint(name: str, **kwargs) -> Path:
    """Convenience function to get a checkpoint using the default manager."""
    return get_default_manager().get_checkpoint(name, **kwargs)


def download_from_url(url: str, **kwargs) -> Path:
    """Convenience function to download from URL using the default manager."""
    return get_default_manager().download_from_url(url, **kwargs)


def register_model(name: str, url: str, **kwargs) -> None:
    """Convenience function to register a model using the default manager."""
    get_default_manager().register_model(name, url, **kwargs)


def list_models() -> Dict[str, Dict[str, Any]]:
    """Convenience function to list models using the default manager."""
    return get_default_manager().list_models()


def print_models() -> None:
    """Convenience function to print models using the default manager."""
    get_default_manager().print_models()
