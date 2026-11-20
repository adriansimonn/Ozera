"""
Circuit Tracing: the attribution graph of one prompt and one target token, built from a
model's transcoders with the circuit-tracer library, and interventions on its feature nodes.

- registry.py: the transcoder sets each model can be traced through;
- budget.py: limits and time estimates;
- store.py: where transcoders and graphs live on the SAE volume;
- graph.py: a graph as Ozera stores it (nodes, edges, influence, scores);
- loading.py, trace.py, intervene.py: the circuits worker's side (services/modal_circuits.py),
  the only place circuit-tracer is installed.

The backend imports registry, budget, store and graph; none of them needs circuit-tracer.
"""
