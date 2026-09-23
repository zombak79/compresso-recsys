"""The architecture: modules holding the learned parameters."""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch import nn

from compresso import MaskedParam, SRPParam, SRPTensor
from compresso_recsys.models.elsa.config import (
    ELSACompressionConfig,
    SparseInferenceBackend,
)


def _normalize_srp(factors: SRPTensor) -> SRPTensor:
    return SRPTensor(
        cols=factors.cols,
        vals=F.normalize(factors.vals, p=2.0, dim=-1),
        shape=factors.shape,
        validate=False,
    )


def _srp_to_coo(factors: SRPTensor) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse invariant checks are implicitly disabled.*",
            category=UserWarning,
        )
        return factors.to_coo()


def _score_sparse_candidates(
    x: torch.Tensor,
    *,
    source_embeddings: SRPTensor,
    candidate_embeddings: SRPTensor,
    x_out: torch.Tensor | None,
    use_relu: bool,
) -> torch.Tensor:
    user_factors = torch.sparse.mm(
        _srp_to_coo(source_embeddings).transpose(0, 1),
        x.T,
    ).T
    scores = torch.sparse.mm(
        _srp_to_coo(candidate_embeddings),
        user_factors.T,
    ).T
    if x_out is not None:
        scores = scores - x_out
    return F.relu(scores) if use_relu else scores


def _score_candidates(
    x: torch.Tensor,
    *,
    embeddings: torch.Tensor,
    sources: torch.Tensor | None,
    candidates: torch.Tensor | None,
    x_out: torch.Tensor | None,
    use_relu: bool,
) -> torch.Tensor:
    if candidates is None:
        source_embeddings = embeddings if sources is None else embeddings[sources]
        candidate_embeddings = embeddings
    else:
        candidate_embeddings = embeddings[candidates]
        if x.shape[1] > candidate_embeddings.shape[0]:
            raise ValueError("the candidate prefix must contain every source item")
        source_embeddings = candidate_embeddings[: x.shape[1]]
    scores = (x @ source_embeddings) @ candidate_embeddings.T
    if x_out is not None:
        scores = scores - x_out
    return F.relu(scores) if use_relu else scores


class ELSA(nn.Module):
    """Scalable linear shallow autoencoder with normalized item embeddings."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        use_relu: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.use_relu = bool(use_relu)
        self.A = nn.Parameter(torch.empty(self.input_dim, self.latent_dim))
        nn.init.xavier_uniform_(self.A)

    def normalized_item_embeddings(self) -> torch.Tensor:
        """Return row-normalized item embeddings."""
        return F.normalize(self.A, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        sources: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
        x_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score items, using candidate rows as the source prefix when given."""
        return _score_candidates(
            x,
            embeddings=self.normalized_item_embeddings(),
            sources=sources,
            candidates=candidates,
            x_out=x_out,
            use_relu=self.use_relu,
        )


class CompressedELSA(nn.Module):
    """ELSA item factors compressed to fixed row-wise sparsity.

    The model starts with a dense :class:`compresso.MaskedParam`. After its
    mask schedule is complete, :meth:`convert_to_srp` replaces that parameter
    with an :class:`compresso.SRPParam` whose structure is fixed and whose
    values remain trainable.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        compression: ELSACompressionConfig,
        *,
        use_relu: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if compression.k_target > latent_dim:
            raise ValueError("compression.k_target must be <= latent_dim")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.compression = compression
        self.use_relu = bool(use_relu)
        weight = torch.empty(self.input_dim, self.latent_dim)
        nn.init.xavier_uniform_(weight)
        self.masked_A: MaskedParam | None = MaskedParam(
            weight=weight,
            k_target=compression.k_target,
            k_schedule=compression.k_schedule,
            num_stages=compression.num_stages,
            stability_window=compression.stability_window,
            change_threshold=compression.change_threshold,
            sparsity="row",
            score_mode=compression.score_mode,
            ste_alpha=compression.ste_alpha,
            post_norm_l1=False,
        )
        self.sparse_A: SRPParam | None = None
        self.phase = "mask_search"
        self._inference_srp: SRPTensor | None = None
        self._inference_csr: torch.Tensor | None = None
        self._inference_dense: torch.Tensor | None = None

    def _invalidate_inference_cache(self) -> None:
        self._inference_srp = None
        self._inference_csr = None
        self._inference_dense = None

    def _apply(self, fn):
        result = super()._apply(fn)
        self._invalidate_inference_cache()
        return result

    def train(self, mode: bool = True) -> CompressedELSA:
        result = super().train(mode)
        if mode:
            self._invalidate_inference_cache()
            if self.is_sparse:
                self.phase = "sparse_finetune"
        return result

    @property
    def is_sparse(self) -> bool:
        """Whether the final fixed SRP structure has been installed."""
        return self.sparse_A is not None

    def normalized_item_embeddings(
        self,
        rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return normalized dense factors, optionally for selected rows."""
        if self.masked_A is not None:
            factors = self.masked_A() if rows is None else self.masked_A[rows]
        elif self.sparse_A is not None:
            factors = self.normalized_item_srp(rows).to_dense()
            return factors
        else:  # pragma: no cover - defensive invariant
            raise RuntimeError("compressed ELSA has no item parameter")
        return F.normalize(factors, p=2.0, dim=-1)

    def normalized_item_srp(
        self,
        rows: torch.Tensor | None = None,
    ) -> SRPTensor:
        """Return normalized sparse factors, optionally for selected rows."""
        if self.sparse_A is None:
            raise RuntimeError(
                "SRP factors are unavailable until mask search completes"
            )
        factors = self.sparse_A() if rows is None else self.sparse_A[rows]
        return _normalize_srp(factors)

    @torch.no_grad()
    def convert_to_srp(self) -> None:
        """Install Compresso's final fixed SRP parameter."""
        if self.sparse_A is not None:
            return
        if self.masked_A is None or not self.masked_A.schedule_done:
            raise RuntimeError("mask search must complete before conversion to SRP")

        sparse_A = self.masked_A.to_srp_param()
        self.sparse_A = sparse_A
        self.masked_A = None
        self.phase = "sparse_finetune"
        self._invalidate_inference_cache()

    @torch.no_grad()
    def prepare_inference(
        self,
        backend: SparseInferenceBackend | None = None,
    ) -> None:
        """Cache normalized factors for the selected inference backend."""
        resolved_backend = (
            self.compression.sparse_inference_backend if backend is None else backend
        )
        if resolved_backend not in {"csr", "dense"}:
            raise ValueError("sparse inference backend must be 'csr' or 'dense'")
        if self._inference_srp is None:
            self._inference_srp = self.normalized_item_srp().detach()
        if resolved_backend == "csr" and self._inference_csr is None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Sparse CSR tensor support is in beta state.*",
                    category=UserWarning,
                )
                self._inference_csr = self._inference_srp.to_csr()
        elif resolved_backend == "dense" and self._inference_dense is None:
            self._inference_dense = self._inference_srp.to_dense()
        self.phase = "inference"

    @torch.no_grad()
    def export_item_embeddings(self) -> SRPTensor:
        """Return a detached copy of the normalized final item factors."""
        return self.normalized_item_srp().detach().clone()

    def forward(
        self,
        x: torch.Tensor,
        *,
        sources: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
        x_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score candidates during mask search or sparse-value training."""
        if (
            self.sparse_A is not None
            and self.compression.sparse_finetune_backend == "coo"
        ):
            candidate_embeddings = self.normalized_item_srp(candidates)
            if candidates is None:
                source_embeddings = (
                    candidate_embeddings
                    if sources is None
                    else candidate_embeddings[sources]
                )
            else:
                if x.shape[1] > candidate_embeddings.rows:
                    raise ValueError(
                        "the candidate prefix must contain every source item"
                    )
                source_embeddings = candidate_embeddings[: x.shape[1]]
            return _score_sparse_candidates(
                x,
                source_embeddings=source_embeddings,
                candidate_embeddings=candidate_embeddings,
                x_out=x_out,
                use_relu=self.use_relu,
            )
        if candidates is not None:
            candidate_embeddings = self.normalized_item_embeddings(candidates)
            if x.shape[1] > candidate_embeddings.shape[0]:
                raise ValueError("the candidate prefix must contain every source item")
            source_embeddings = candidate_embeddings[: x.shape[1]]
            scores = (x @ source_embeddings) @ candidate_embeddings.T
            if x_out is not None:
                scores = scores - x_out
            return F.relu(scores) if self.use_relu else scores
        return _score_candidates(
            x,
            embeddings=self.normalized_item_embeddings(),
            sources=sources,
            candidates=candidates,
            x_out=x_out,
            use_relu=self.use_relu,
        )

    @torch.no_grad()
    def score_all_items(
        self,
        x: torch.Tensor,
        *,
        sources: torch.Tensor,
        backend: SparseInferenceBackend | None = None,
    ) -> torch.Tensor:
        """Score the full catalog with cached sparse or dense factors."""
        resolved_backend = (
            self.compression.sparse_inference_backend if backend is None else backend
        )
        self.prepare_inference(resolved_backend)
        assert self._inference_srp is not None

        if resolved_backend == "csr":
            assert self._inference_csr is not None
            source_factors = self._inference_srp[sources].to_dense()
            user_factors = x @ source_factors
            scores = torch.sparse.mm(
                self._inference_csr,
                user_factors.T,
            ).T
        else:
            assert self._inference_dense is not None
            source_factors = self._inference_dense[sources]
            user_factors = x @ source_factors
            scores = user_factors @ self._inference_dense.T
        return F.relu(scores) if self.use_relu else scores
