/**
 * Static default SAE configurations for Ozera models.
 * These are hardcoded so the UI renders instantly without waiting for Modal.
 * Only actual analysis/inference operations need to call Modal.
 */

import type { SAEListResponse } from '../api/client'

// nano: 6 layers, d_model=192, 8x expansion = 1536 features
// mini: 8 layers, d_model=512, 8x expansion = 4096 features

function buildSAEs(
  numLayers: number,
  dInput: number,
  dHidden: number,
): SAEListResponse['models'][string]['saes'] {
  const saes: SAEListResponse['models'][string]['saes'] = []
  for (let layer = 0; layer < numLayers; layer++) {
    for (const activationType of ['residual', 'mlp_output'] as const) {
      saes.push({
        layer,
        activation_type: activationType,
        path: '',
        d_input: dInput,
        d_hidden: dHidden,
        activation: 'relu',
      })
    }
  }
  return saes
}

export const DEFAULT_SAE_LIST: SAEListResponse = {
  models: {
    nano: {
      num_layers: 6,
      d_model: 192,
      sae_d_hidden: 1536,
      saes: buildSAEs(6, 192, 1536),
    },
    mini: {
      num_layers: 8,
      d_model: 512,
      sae_d_hidden: 4096,
      saes: buildSAEs(8, 512, 4096),
    },
  },
  total_saes: 28, // 6*2 + 8*2
}
