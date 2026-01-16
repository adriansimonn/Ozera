"""
Activation storage and management for Ozera models.

Stores intermediate activations from model forward passes for visualization
and analysis purposes.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid


class ActivationStore:
    """
    In-memory storage for model activations.

    Stores activations with metadata and provides retrieval and cleanup.
    """

    def __init__(self, max_entries: int = 100):
        """
        Initialize activation store.

        Args:
            max_entries: Maximum number of activation sets to store
        """
        self.max_entries = max_entries
        self._store: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}

    def store_activations(
        self,
        activations: Dict[str, Any],
        tokens: List[int],
        prompt: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store activations from a forward pass.

        Args:
            activations: Dictionary of activation tensors from model
            tokens: Token IDs that were processed
            prompt: Original prompt text
            model_name: Name of model used
            metadata: Optional additional metadata

        Returns:
            Activation ID for later retrieval
        """
        # Generate unique ID
        activation_id = str(uuid.uuid4())

        # Clean up old entries if needed
        if len(self._store) >= self.max_entries:
            self._cleanup_oldest()

        # Convert tensors to numpy for storage
        processed_activations = self._process_activations(activations)

        # Store activation data
        self._store[activation_id] = {
            'id': activation_id,
            'activations': processed_activations,
            'tokens': tokens,
            'prompt': prompt,
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        self._access_times[activation_id] = datetime.now()

        return activation_id

    def get_activations(self, activation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored activations by ID.

        Args:
            activation_id: ID of activations to retrieve

        Returns:
            Activation data dictionary or None if not found
        """
        if activation_id not in self._store:
            return None

        # Update access time
        self._access_times[activation_id] = datetime.now()

        return self._store[activation_id]

    def get_activation_summary(self, activation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata summary without full activation tensors.

        Args:
            activation_id: ID of activations

        Returns:
            Summary dictionary or None if not found
        """
        if activation_id not in self._store:
            return None

        data = self._store[activation_id]

        return {
            'id': data['id'],
            'prompt': data['prompt'],
            'model': data['model'],
            'timestamp': data['timestamp'],
            'num_tokens': len(data['tokens']),
            'num_layers': len(data['activations'].get('layers', [])),
            'metadata': data['metadata']
        }

    def list_activations(self) -> List[Dict[str, Any]]:
        """
        List all stored activations (summaries only).

        Returns:
            List of activation summaries
        """
        return [
            self.get_activation_summary(act_id)
            for act_id in self._store.keys()
        ]

    def delete_activations(self, activation_id: str) -> bool:
        """
        Delete stored activations.

        Args:
            activation_id: ID of activations to delete

        Returns:
            True if deleted, False if not found
        """
        if activation_id in self._store:
            del self._store[activation_id]
            del self._access_times[activation_id]
            return True
        return False

    def clear_all(self):
        """Clear all stored activations."""
        self._store.clear()
        self._access_times.clear()

    def _process_activations(self, activations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert activation tensors to numpy arrays and compute statistics.

        Args:
            activations: Raw activation tensors from model

        Returns:
            Processed activation dictionary with numpy arrays and stats
        """
        processed = {}

        # Process top-level tensors
        for key, value in activations.items():
            if key == 'layers':
                # Process layer-by-layer activations
                processed['layers'] = []
                for layer_idx, layer_act in enumerate(value):
                    layer_processed = {}
                    for act_name, act_tensor in layer_act.items():
                        layer_processed[act_name] = self._tensor_to_data(act_tensor)
                    processed['layers'].append(layer_processed)
            elif isinstance(value, torch.Tensor):
                processed[key] = self._tensor_to_data(value)
            else:
                processed[key] = value

        return processed

    def _tensor_to_data(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Convert a tensor to numpy array with statistics.

        Args:
            tensor: PyTorch tensor

        Returns:
            Dictionary with array and statistics
        """
        # Move to CPU and convert to numpy
        arr = tensor.cpu().numpy()

        return {
            'values': arr.tolist(),  # Convert to list for JSON serialization
            'shape': list(arr.shape),
            'dtype': str(arr.dtype),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr))
        }

    def _cleanup_oldest(self):
        """Remove the least recently accessed activation."""
        if not self._access_times:
            return

        # Find oldest access time
        oldest_id = min(self._access_times.items(), key=lambda x: x[1])[0]

        # Remove it
        del self._store[oldest_id]
        del self._access_times[oldest_id]


# Global activation store instance
_global_store = None


def get_activation_store() -> ActivationStore:
    """
    Get global activation store instance.

    Returns:
        Global ActivationStore instance
    """
    global _global_store
    if _global_store is None:
        _global_store = ActivationStore(max_entries=100)
    return _global_store
