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
        activations = data['activations']

        # Build layer info with shapes but no values
        layer_info = []
        for layer in activations.get('layers', []):
            layer_shapes = {}
            for key, value in layer.items():
                if isinstance(value, dict) and 'shape' in value:
                    layer_shapes[key] = {
                        'shape': value['shape'],
                        'mean': value.get('mean'),
                        'std': value.get('std'),
                        'min': value.get('min'),
                        'max': value.get('max'),
                    }
            layer_info.append(layer_shapes)

        # Build top-level tensor info
        tensor_info = {}
        for key in ['token_embeddings', 'positional_embeddings', 'combined_embeddings', 'final_layer_norm', 'logits']:
            value = activations.get(key)
            if isinstance(value, dict) and 'shape' in value:
                tensor_info[key] = {
                    'shape': value['shape'],
                    'mean': value.get('mean'),
                    'std': value.get('std'),
                    'min': value.get('min'),
                    'max': value.get('max'),
                }

        # Include top_k_logits summary if present
        top_k_logits = activations.get('top_k_logits')
        if isinstance(top_k_logits, dict) and 'k' in top_k_logits:
            tensor_info['top_k_logits'] = {
                'k': top_k_logits['k'],
                'seq_len': top_k_logits['seq_len'],
            }

        return {
            'id': data['id'],
            'prompt': data['prompt'],
            'model': data['model'],
            'timestamp': data['timestamp'],
            'num_tokens': len(data['tokens']),
            'num_layers': len(data['activations'].get('layers', [])),
            'metadata': data['metadata'],
            'layer_info': layer_info,
            'tensor_info': tensor_info,
        }

    def get_layer_activations(self, activation_id: str, layer_idx: int) -> Optional[Dict[str, Any]]:
        """
        Get activations for a specific layer.

        Args:
            activation_id: ID of activations
            layer_idx: Layer index to retrieve

        Returns:
            Layer activation data or None if not found
        """
        if activation_id not in self._store:
            return None

        data = self._store[activation_id]
        layers = data['activations'].get('layers', [])

        if layer_idx < 0 or layer_idx >= len(layers):
            return None

        # Update access time
        self._access_times[activation_id] = datetime.now()

        return {
            'layer_idx': layer_idx,
            'activations': layers[layer_idx]
        }

    def get_tensor_activation(self, activation_id: str, tensor_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific top-level tensor activation (embeddings, logits, etc).

        Args:
            activation_id: ID of activations
            tensor_name: Name of tensor to retrieve

        Returns:
            Tensor data or None if not found
        """
        if activation_id not in self._store:
            return None

        data = self._store[activation_id]
        activations = data['activations']

        valid_tensors = ['token_embeddings', 'positional_embeddings', 'combined_embeddings', 'final_layer_norm', 'logits', 'top_k_logits']
        if tensor_name not in valid_tensors:
            return None

        tensor_data = activations.get(tensor_name)
        if tensor_data is None:
            return None

        # Update access time
        self._access_times[activation_id] = datetime.now()

        return {
            'tensor_name': tensor_name,
            'data': tensor_data
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

        Handles both torch.Tensor inputs (from local inference) and
        already-serialized list inputs (from Modal inference).

        Args:
            activations: Raw activation tensors or serialized lists from model

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
                    for act_name, act_value in layer_act.items():
                        layer_processed[act_name] = self._value_to_data(act_value)
                    processed['layers'].append(layer_processed)
            elif isinstance(value, torch.Tensor):
                processed[key] = self._tensor_to_data(value)
            elif isinstance(value, list):
                # Already serialized from Modal - convert to data format
                processed[key] = self._list_to_data(value)
            else:
                processed[key] = value

        return processed

    def _value_to_data(self, value) -> Dict[str, Any]:
        """Convert a tensor or list to data format."""
        if isinstance(value, torch.Tensor):
            return self._tensor_to_data(value)
        elif isinstance(value, list):
            return self._list_to_data(value)
        else:
            return value

    def _list_to_data(self, data: list) -> Dict[str, Any]:
        """
        Convert an already-serialized list to data format with statistics.

        Args:
            data: List of values (from Modal serialization)

        Returns:
            Dictionary with array and statistics
        """
        arr = np.array(data)

        return {
            'values': data,  # Keep original list for JSON serialization
            'shape': list(arr.shape),
            'dtype': str(arr.dtype),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr))
        }

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
