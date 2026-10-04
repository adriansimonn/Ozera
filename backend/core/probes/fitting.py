"""
Fitting linear probes: difference-in-means and L2-regularized logistic regression.

Each function fits a batch of independent problems at once (one per layer) over the same
examples, so features are [batch, n, d] and results are [batch, ...].
"""

from typing import Optional, Union

import torch
import torch.nn.functional as F

# The bias is fit as the weight of a constant feature with this value, so the L2 penalty on
# it is 1/INTERCEPT_SCALE**2 of a feature weight's (liblinear's intercept_scaling). A barely
# penalized bias keeps the decision threshold right on imbalanced data.
INTERCEPT_SCALE = 10.0


def fit_diff_means(X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Difference-in-means probes: the direction from the negative class mean to the positive one.

    Scores are projections onto it, with the threshold halfway between the class means.

    Args:
        X: [batch, n, d] raw activations
        y: [n] labels (0/1)

    Returns:
        (weights [batch, d], biases [batch]), so that scores are X @ w + b
    """
    pos = y.bool()
    mean_pos = X[:, pos].mean(dim=1)
    mean_neg = X[:, ~pos].mean(dim=1)
    w = mean_pos - mean_neg
    b = -(w * (mean_pos + mean_neg)).sum(dim=-1) / 2
    return w, b


def standardization(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-dimension mean and standard deviation of [batch, n, d] features, over the examples.

    Dimensions that are (nearly) constant get a standard deviation of 1, so standardizing
    leaves them at ~0 and the probe gives them no weight.
    """
    std, mean = torch.std_mean(X, dim=1, correction=0)
    floor = std.mean(dim=-1, keepdim=True) * 1e-6 + 1e-12
    return mean, torch.where(std > floor, std, torch.ones_like(std))


def _with_intercept(Z: torch.Tensor) -> torch.Tensor:
    return torch.cat([Z, Z.new_full((*Z.shape[:2], 1), INTERCEPT_SCALE)], dim=-1)


def _cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve A x = b for a batch of symmetric positive definite A ([batch, m, m]) and b ([batch, m])."""
    L, info = torch.linalg.cholesky_ex(A)
    if bool((info > 0).any()):
        # Rounding made some nearly singular; nudge their diagonals and retry
        jitter = A.diagonal(dim1=1, dim2=2).mean(dim=-1) * 1e-5
        A = A + torch.diag_embed(jitter[:, None].expand(-1, A.shape[-1]))
        L = torch.linalg.cholesky(A)
    return torch.cholesky_solve(b[..., None], L)[..., 0]


def fit_logistic(
    Z: torch.Tensor,
    y: torch.Tensor,
    l2: Union[float, torch.Tensor],
    init: Optional[torch.Tensor] = None,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    L2-regularized logistic regression by Newton's method, for a batch of problems at once.

    Minimizes mean_i log(1 + exp(-y_i' * z_i)) + l2/2 * |theta|^2 for each problem, where
    z = [Z, INTERCEPT_SCALE] @ theta and y' = ±1. Newton steps solve the Hessian with
    Cholesky: in d-space when there are more examples than features, otherwise in example
    space through the Woodbury identity. A backtracking line search keeps every step a
    descent step, so it converges however weak the regularization.

    Args:
        Z: [batch, n, d] standardized features
        y: [n] or [batch, n] labels (0/1)
        l2: Regularization strength, one for all problems or [batch]
        init: [batch, d + 1] starting point (e.g. the solution for a nearby l2), else zeros
        max_iter: Newton iterations at most
        tol: Stop once every problem's Newton decrement is below this

    Returns:
        theta [batch, d + 1]: feature weights, then the intercept feature's weight
        (the bias is INTERCEPT_SCALE * theta[:, -1])
    """
    batch, n, d = Z.shape
    Za = _with_intercept(Z)
    y = y.to(Z.dtype).expand(batch, n)
    l2 = torch.as_tensor(l2, dtype=Z.dtype, device=Z.device).expand(batch)
    theta = torch.zeros(batch, d + 1, dtype=Z.dtype, device=Z.device) if init is None else init.clone()

    primal = d + 1 <= n
    gram = None if primal else Za @ Za.transpose(1, 2)  # [batch, n, n]

    def objective(t: torch.Tensor) -> torch.Tensor:
        z = (Za @ t[..., None])[..., 0]
        return (F.softplus(z) - y * z).mean(dim=1) + l2 / 2 * (t * t).sum(dim=1)

    f = objective(theta)
    for _ in range(max_iter):
        z = (Za @ theta[..., None])[..., 0]
        p = torch.sigmoid(z)
        g = (Za.transpose(1, 2) @ ((p - y) / n)[..., None])[..., 0] + l2[:, None] * theta
        D = p * (1 - p) / n

        if primal:
            H = (Za * D[..., None]).transpose(1, 2) @ Za
            H.diagonal(dim1=1, dim2=2).add_(l2[:, None])
            step = _cholesky_solve(H, g)
        else:
            # H^-1 g = (g - Za^T S (l2 I + S K S)^-1 S Za g) / l2, with S = sqrt(D), K = Za Za^T
            S = D.sqrt()
            M = S[:, :, None] * gram * S[:, None, :]
            M.diagonal(dim1=1, dim2=2).add_(l2[:, None])
            u = S * (Za @ g[..., None])[..., 0]
            t = _cholesky_solve(M, u)
            step = (g - (Za.transpose(1, 2) @ (S * t)[..., None])[..., 0]) / l2[:, None]

        decrement = (g * step).sum(dim=1)
        active = decrement > tol
        if not bool(active.any()):
            break

        # Backtracking (Armijo) line search, per problem
        size = torch.ones(batch, dtype=Z.dtype, device=Z.device)
        accepted = ~active
        candidate, f_candidate = theta, f
        for _ in range(30):
            trial = theta - size[:, None] * step
            f_trial = objective(trial)
            ok = active & ~accepted & (f_trial <= f - 0.25 * size * decrement)
            candidate = torch.where(ok[:, None], trial, candidate)
            f_candidate = torch.where(ok, f_trial, f_candidate)
            accepted = accepted | ok
            if bool(accepted.all()):
                break
            size = torch.where(accepted, size, size / 2)
        theta, f = candidate, f_candidate

    return theta


def logistic_to_raw(
    theta: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fold the standardization into a logistic probe, giving weights on raw activations.

    Args:
        theta: [batch, d + 1] from fit_logistic on (X - mean) / std
        mean, std: [batch, d] the standardization

    Returns:
        (weights [batch, d], biases [batch]), so that X @ w + b equals the probe's logit
    """
    w = theta[:, :-1] / std
    b = INTERCEPT_SCALE * theta[:, -1] - (mean * w).sum(dim=-1)
    return w, b


def scores(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Probe scores [batch, n] of raw activations X [batch, n, d] under weights [batch, d] and biases [batch]."""
    return (X @ w[..., None])[..., 0] + b[:, None]
