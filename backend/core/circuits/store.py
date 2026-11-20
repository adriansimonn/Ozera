"""
Where Circuit Tracing keeps its files on the SAE volume (mounted at /saes on the workers; the
backend reads it through services.atlas_store):

    /saes/circuits/transcoders/<set directory>/   a transcoder set's config.yaml and layers
    /saes/circuits/graphs/<job id>/graph.json     a job's graph (core.circuits.graph)
"""

import re

CIRCUITS_ROOT = "/saes/circuits"
TRANSCODERS_ROOT = f"{CIRCUITS_ROOT}/transcoders"
GRAPHS_ROOT = f"{CIRCUITS_ROOT}/graphs"

_JOB_ID = re.compile(r"^circuit_\d{14}_[0-9a-f]{8}$")


def is_job_id(job_id: str) -> bool:
    """Whether a string has the form of a circuit job's ID (so it can be used in a path)."""
    return bool(_JOB_ID.match(job_id or ""))


def graph_directory(job_id: str) -> str:
    if not is_job_id(job_id):
        raise ValueError(f"Not a circuit job ID: {job_id}")
    return f"{GRAPHS_ROOT}/{job_id}"


def graph_path(job_id: str) -> str:
    return f"{graph_directory(job_id)}/graph.json"
