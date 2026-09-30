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

from core.tensor_codec import encode_tensor, is_tensor_entry, slice_last_dim, to_wire


class ActivationStore:
    """
    In-memory storage for model activations.

    Stores activations with metadata and provides retrieval and cleanup.
    Tensors are kept in the compact encoding from core.tensor_codec and only
    expanded to the base64 float32 wire format when they are read.
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

        data = self._store[activation_id]
        activations = {
            key: [{k: to_wire(v) for k, v in layer.items()} for layer in value] if key == 'layers' else to_wire(value)
            for key, value in data['activations'].items()
        }
        return {**data, 'activations': activations}

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
            'tokens': data['tokens'],
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
            'activations': {k: to_wire(v) for k, v in layers[layer_idx].items()}
        }

    def get_flow_activations(self, activation_id: str, dims: int) -> Optional[Dict[str, Any]]:
        """
        Get the slice of activations the generation flow visualization reads.

        That is each layer's residual stream after the FFN (post_attn when post_ff
        is missing) and the combined (else token) embeddings, limited to the first
        `dims` hidden dimensions. Statistics still describe the full tensors, since
        the visualization normalizes by them.

        Args:
            activation_id: ID of activations
            dims: Number of leading hidden dimensions to include

        Returns:
            Dict with per-layer and embedding slices, or None if not found
        """
        if activation_id not in self._store:
            return None

        # Update access time
        self._access_times[activation_id] = datetime.now()

        activations = self._store[activation_id]['activations']

        def pick(source: Dict[str, Any], *keys: str) -> Dict[str, Any]:
            for key in keys:
                if is_tensor_entry(source.get(key)):
                    return {key: slice_last_dim(source[key], dims)}
            return {}

        return {
            'layers': [pick(layer, 'post_ff', 'post_attn') for layer in activations.get('layers', [])],
            **pick(activations, 'combined_embeddings', 'token_embeddings'),
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
            'data': to_wire(tensor_data)
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
        Convert activation tensors to the compact storage encoding.

        Handles torch.Tensor inputs (from local inference), already-encoded
        dicts (from Modal inference), and legacy serialized lists.

        Args:
            activations: Raw activation tensors or serialized values from model

        Returns:
            Processed activation dictionary with encoded tensors and stats
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
            else:
                processed[key] = self._value_to_data(value)

        return processed

    def _value_to_data(self, value) -> Dict[str, Any]:
        """Convert a tensor or list to the compact encoding; encoded dicts pass through."""
        if isinstance(value, torch.Tensor):
            return encode_tensor(value)
        elif isinstance(value, list):
            # Legacy list serialization
            return encode_tensor(torch.tensor(np.array(value, dtype=np.float32)))
        else:
            return value

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
