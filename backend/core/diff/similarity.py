"""
How alike two models' activations are at one residual stream position, from sums kept while
the tokens go by: no activations are stored.

For the same n tokens read by both models, A and B are their activations ([n, d] each). From
ΣaᵀB-style second moments (three d×d matrices per position) come the measures
core.geometry.subspace computes from A and B themselves:

- linear CKA, ‖AᵀB‖²_F / (‖AᵀA‖_F ‖BᵀB‖_F) on centered activations: 1 when one is a rotation
  and uniform scaling of the other, weighted toward the directions that vary most;
- Procrustes R², how much of B a scaled rotation of A explains: (Σσ)² / (‖A‖²_F ‖B‖²_F), σ the
  singular values of AᵀB;

and one that needs no alignment: the distance ‖a − b‖ between the two models' activations of
the same token, relative to how far activations spread around their mean (0 for identical
models, about √2 for unrelated activations of equal spread).

Tokens paired at random give a linear CKA of about tr(C_A) tr(C_B) / (n ‖C_A‖_F ‖C_B‖_F): the
number of directions the activations effectively use, over the number of tokens. It's
reported as the chance level; with few tokens and wide models it isn't small.

Residual streams have means much larger than their spread, so sums are taken around a
reference point (the first tokens' mean): float32 then keeps the covariances' digits.
"""

from typing import Optional, Sequence

import torch


def nuclear_norm(matrix: torch.Tensor) -> float:
    """The sum of a matrix's singular values."""
    try:
        return float(torch.linalg.svdvals(matrix.float()).sum())
    except RuntimeError:
        # A solver that didn't converge: the singular values are the square roots of MᵀM's eigenvalues
        gram = matrix.double().T @ matrix.double()
        return float(torch.linalg.eigvalsh(gram).clamp_min(0).sqrt().sum())


class StreamSimilarity:
    """
    Sums over the tokens both models have read, at every residual stream position, kept
    apart for the two halves of the texts.

    The d×d second moments are kept per half when `matrix_halves` (linear CKA and Procrustes
    can then be read for each half), else for all tokens together; the sums the distance
    needs are small and always kept per half.
    """

    def __init__(self, positions: int, dim: int, matrix_halves: bool, device: torch.device):
        self.positions = positions
        self.matrix_halves = matrix_halves
        groups = 2 if matrix_halves else 1
        self.s_ab = torch.zeros(groups, positions, dim, dim, device=device)
        self.s_aa = torch.zeros(groups, positions, dim, dim, device=device)
        self.s_bb = torch.zeros(groups, positions, dim, dim, device=device)
        # Per half: Σ(a − r), Σ‖a − r‖² (r the reference point), Σ‖a − b‖², Σcos(a, b), tokens
        self.sum_a = torch.zeros(2, positions, dim, dtype=torch.float64, device=device)
        self.sum_b = torch.zeros(2, positions, dim, dtype=torch.float64, device=device)
        self.square_a = torch.zeros(2, positions, dtype=torch.float64, device=device)
        self.square_b = torch.zeros(2, positions, dtype=torch.float64, device=device)
        self.square_diff = torch.zeros(2, positions, dtype=torch.float64, device=device)
        self.cosine = torch.zeros(2, positions, dtype=torch.float64, device=device)
        self.count = torch.zeros(2, positions, dtype=torch.float64, device=device)
        self.reference_a = torch.zeros(positions, dim, device=device)
        self.reference_b = torch.zeros(positions, dim, device=device)
        self._referenced = [False] * positions

    def add(self, half: int, position: int, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Add tokens' activations at one position.

        Args:
            half: 0 or 1, the half of the texts the tokens are in
            a, b: [m, d] the two models' activations of the same m tokens

        Returns:
            [m] each token's ‖a − b‖
        """
        a, b = a.float(), b.float()
        if not self._referenced[position]:
            self.reference_a[position] = a.mean(dim=0)
            self.reference_b[position] = b.mean(dim=0)
            self._referenced[position] = True
        square_diff = (a - b).pow(2).sum(dim=1)
        self.square_diff[half, position] += square_diff.double().sum()
        self.cosine[half, position] += torch.nn.functional.cosine_similarity(a, b, dim=1).double().sum()
        self.count[half, position] += a.shape[0]

        a = a - self.reference_a[position]
        b = b - self.reference_b[position]
        self.sum_a[half, position] += a.double().sum(dim=0)
        self.sum_b[half, position] += b.double().sum(dim=0)
        self.square_a[half, position] += a.pow(2).sum().double()
        self.square_b[half, position] += b.pow(2).sum().double()
        group = half if self.matrix_halves else 0
        self.s_ab[group, position] += a.T @ b
        self.s_aa[group, position] += a.T @ a
        self.s_bb[group, position] += b.T @ b
        return square_diff.sqrt()

    def _spread(self, halves: Sequence[int], position: int) -> tuple[float, float, float]:
        """(tokens, tr C_A, tr C_B) over some halves: the summed squared distances from each model's mean."""
        n = float(self.count[list(halves), position].sum())
        if n < 1:
            return 0.0, 0.0, 0.0
        out = []
        for sums, squares in ((self.sum_a, self.square_a), (self.sum_b, self.square_b)):
            total = sums[list(halves), position].sum(dim=0)
            out.append(max(float(squares[list(halves), position].sum() - total.pow(2).sum() / n), 0.0))
        return n, out[0], out[1]

    def scale(self, position: int) -> float:
        """
        What distances at a position are divided by: the root mean squared distance of a
        token's activation from its model's mean, over both models and all tokens.
        """
        n, spread_a, spread_b = self._spread((0, 1), position)
        return ((spread_a + spread_b) / (2 * n)) ** 0.5 if n else 0.0

    def distance(self, halves: Sequence[int]) -> list[Optional[float]]:
        """
        The relative distance between the two models' activations at every position, over
        some halves' tokens: the root mean squared ‖a − b‖ over those tokens' spread.
        """
        out: list[Optional[float]] = []
        for position in range(self.positions):
            n, spread_a, spread_b = self._spread(halves, position)
            spread = (spread_a + spread_b) / 2
            square = float(self.square_diff[list(halves), position].sum())
            out.append((square / spread) ** 0.5 if n and spread > 0 else (0.0 if n and square == 0 else None))
        return out

    def mean_cosine(self, halves: Sequence[int]) -> list[Optional[float]]:
        """The mean cosine between the two models' activations of a token, at every position."""
        out: list[Optional[float]] = []
        for position in range(self.positions):
            n = float(self.count[list(halves), position].sum())
            out.append(float(self.cosine[list(halves), position].sum()) / n if n else None)
        return out

    def alignment(self, halves: Optional[int] = None) -> dict[str, list[Optional[float]]]:
        """
        Linear CKA, its chance level and Procrustes R² at every position.

        Args:
            halves: A half (needs matrix_halves), or None for all tokens

        Returns:
            {"cka", "cka_chance", "procrustes": [positions]}, None where the activations of
            either model don't vary
        """
        if halves is not None and not self.matrix_halves:
            raise ValueError("The second moments weren't kept per half")
        groups = [halves] if halves is not None else list(range(self.s_ab.shape[0]))
        which = [halves] if halves is not None else [0, 1]
        out: dict[str, list[Optional[float]]] = {"cka": [], "cka_chance": [], "procrustes": []}
        for position in range(self.positions):
            n = float(self.count[which, position].sum())
            mean_a = self.sum_a[which, position].sum(dim=0)
            mean_b = self.sum_b[which, position].sum(dim=0)
            if n < 2:
                for values in out.values():
                    values.append(None)
                continue
            c_ab = self.s_ab[groups, position].sum(dim=0) - (torch.outer(mean_a, mean_b) / n).float()
            c_aa = self.s_aa[groups, position].sum(dim=0) - (torch.outer(mean_a, mean_a) / n).float()
            c_bb = self.s_bb[groups, position].sum(dim=0) - (torch.outer(mean_b, mean_b) / n).float()
            norm_a, norm_b = float(c_aa.norm()), float(c_bb.norm())
            trace_a, trace_b = float(c_aa.trace()), float(c_bb.trace())
            if norm_a <= 0 or norm_b <= 0 or trace_a <= 0 or trace_b <= 0:
                for values in out.values():
                    values.append(None)
                continue
            out["cka"].append(min(float(c_ab.pow(2).sum()) / (norm_a * norm_b), 1.0))
            out["cka_chance"].append(min(trace_a * trace_b / ((n - 1) * norm_a * norm_b), 1.0))
            out["procrustes"].append(min(nuclear_norm(c_ab) ** 2 / (trace_a * trace_b), 1.0))
        return out
