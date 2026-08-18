"""Verifier-guided population search for GRAM rule hypotheses.

The filename is retained for compatibility with the first experiment draft,
but the algorithm in this module is deliberately **not** described as SMC or
as a Bayesian posterior approximation.  GRAM's public-support proposal is
already conditioned on the same evidence used by the verifier; without a
tractable proposal density, importance correction, and an invariant move
kernel, path-weight accumulation or resample/rejuvenate terminology would be
mathematically misleading.

Each call therefore performs a bounded, auditable population-search stage:

1. draw exactly ``W`` fresh final rule codes from GRAM, or ``W`` fresh iid
   uniform codes for the matched control;
2. optionally merge discrete codes carried from the preceding stage;
3. deduplicate the combined pool before assigning any score;
4. evaluate the complete public history once with an independent frozen
   verifier executor;
5. rank candidate codes lexicographically by MAP compatibility, MAP error
   cells, full-history energy, and code; and
6. retain at most ``carry_limit`` codes from the best compatibility stratum.

The normalized energy scores returned for the retained population are
heuristic ranking weights for active-query utilities.  They are computed once
from each candidate's *current final code*, ignore earlier-stage scores, and
must not be interpreted as posterior mass.

The verifier may be different from ``proposer.executor``.  This is important
when GRAM's checkpoint contains the executor used by its learned support
encoder, while a later active-calibrated executor is used only to test final
discrete hypotheses.  Supplying the latter never overwrites the former.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover - optional neural dependency.
    raise ImportError("prp_wm.gram_smc requires PyTorch") from error

from .causal_filter import score_hypothesis_bank
from .causal_rules import CausalMechanismInference
from .gram_causal_rules import GRAMFactorizedCausalK4
from .latent_rules import (
    RULE_FACTOR_CARDINALITY,
    RULE_FACTOR_COUNT,
    OracleFactorExecutor,
)
from .neural import RuleGridTensorBatch


@dataclass(frozen=True)
class GRAMPopulationMemory:
    """Discrete codes retained between complete-history search stages.

    ``ranking_weights`` describe only the current stage's deterministic
    energy ranking.  ``search`` carries the codes but intentionally discards
    these weights before the next complete-history rescore.
    """

    factor_ids: Tensor  # [B,C,3], padded rows are -1
    mask: Tensor  # bool [B,C]
    ranking_weights: Tensor  # [B,C], zero on padding; not posterior mass
    energies: Tensor  # [B,C], +inf on padding
    map_error_cells: Tensor  # long [B,C], max-int on padding
    map_exact: Tensor  # bool [B,C]
    generation: int

    @property
    def batch_size(self) -> int:
        return int(self.factor_ids.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.factor_ids.shape[1])

    @property
    def counts(self) -> Tensor:
        return self.mask.sum(dim=-1)

    @property
    def weights(self) -> Tensor:
        """Compatibility alias for ranking weights; these are not posterior."""

        return self.ranking_weights


@dataclass(frozen=True)
class GRAMPopulationSearchResult:
    """Fresh proposals, scored candidate pool, and retained population."""

    population: GRAMPopulationMemory
    proposed_factor_ids: Tensor  # [B,W,3], always fresh at this stage
    candidate_factor_ids: Tensor  # [B,W+C,3], padded rows are -1
    candidate_mask: Tensor  # bool [B,W+C]
    candidate_energies: Tensor  # [B,W+C], +inf on padding
    candidate_map_error_cells: Tensor  # long [B,W+C]
    candidate_map_exact: Tensor  # bool [B,W+C]
    candidate_selection_weights: Tensor  # current-stage ranking weights
    candidate_multiplicities: Tensor  # fresh plus carried occurrences
    candidate_proposal_multiplicities: Tensor  # occurrences among fresh W
    candidate_was_carried: Tensor  # bool [B,W+C]
    applied_energy_scales: Tensor  # detached [B]
    verifier_bank_evaluations: int
    proposal_mode: str

    @property
    def unique_factor_ids(self) -> Tensor:
        """Compatibility view of the retained, deduplicated population."""

        return self.population.factor_ids

    @property
    def unique_weights(self) -> Tensor:
        return self.population.ranking_weights

    @property
    def unique_support_energies(self) -> Tensor:
        return self.population.energies

    @property
    def unique_mask(self) -> Tensor:
        return self.population.mask

    @property
    def unique_counts(self) -> Tensor:
        return self.population.counts

    @property
    def belief(self) -> GRAMPopulationMemory:
        """Legacy name for ``population``; no Bayesian claim is implied."""

        return self.population


class GRAMVerifierPopulationSearch(nn.Module):
    """Bounded fresh-proposal search with one-shot full-history verification."""

    def __init__(
        self,
        proposer: GRAMFactorizedCausalK4,
        *,
        verifier_executor: OracleFactorExecutor | None = None,
        proposals: int = 32,
        recursive_steps: int | None = None,
        proposal_mode: str = "gram",
        carry_limit: int | None = None,
        ranking_inverse_temperature: float = 1.0,
        proper_weight: float = 1.0,
        balanced_weight: float = 0.0,
        energy_scale: float | None = None,
    ) -> None:
        super().__init__()
        if proposals <= 0:
            raise ValueError("proposals must be positive")
        if recursive_steps is not None and recursive_steps <= 0:
            raise ValueError("recursive_steps must be positive")
        if proposal_mode not in {"gram", "uniform"}:
            raise ValueError("proposal_mode must be 'gram' or 'uniform'")
        if carry_limit is not None and carry_limit <= 0:
            raise ValueError("carry_limit must be positive")
        if ranking_inverse_temperature < 0:
            raise ValueError("ranking_inverse_temperature cannot be negative")
        if proper_weight < 0 or balanced_weight < 0:
            raise ValueError("verifier cost weights cannot be negative")
        if proper_weight == 0 and balanced_weight == 0:
            raise ValueError("at least one verifier cost weight must be positive")
        if energy_scale is not None and energy_scale <= 0:
            raise ValueError("energy_scale must be positive when provided")

        verifier = proposer.executor if verifier_executor is None else verifier_executor
        if verifier.config != proposer.config:
            raise ValueError("proposer and verifier executor configs must match")

        self.proposer = proposer
        self.verifier_executor = verifier
        self.proposals = int(proposals)
        self.recursive_steps = int(
            proposer.recursive_steps
            if recursive_steps is None
            else recursive_steps
        )
        self.proposal_mode = proposal_mode
        self.carry_limit = int(proposals if carry_limit is None else carry_limit)
        self.ranking_inverse_temperature = float(ranking_inverse_temperature)
        self.proper_weight = float(proper_weight)
        self.balanced_weight = float(balanced_weight)
        self.energy_scale = None if energy_scale is None else float(energy_scale)

        # The proposal checkpoint owns its executor; an independently supplied
        # verifier is registered separately and never assigned into it.
        for parameter in self.proposer.executor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.verifier_executor.parameters():
            parameter.requires_grad_(False)
        self.proposer.executor.eval()
        self.verifier_executor.eval()

    @property
    def config(self):
        return self.proposer.config

    def train(self, mode: bool = True) -> "GRAMVerifierPopulationSearch":
        """Keep both executors frozen/eval even if the wrapper is trained."""

        super().train(mode)
        self.proposer.executor.eval()
        self.verifier_executor.eval()
        return self

    @staticmethod
    def _stable_ranking_weights(
        energies: Tensor,
        mask: Tensor,
        *,
        inverse_temperature: float,
    ) -> Tensor:
        """Normalize one-shot energy scores over a selected candidate mask."""

        if energies.ndim != 2 or mask.shape != energies.shape:
            raise ValueError("energies and mask must share [B,C]")
        if mask.dtype != torch.bool or not torch.all(mask.any(dim=-1)):
            raise ValueError("each ranking row needs a non-empty boolean mask")
        if inverse_temperature < 0:
            raise ValueError("inverse_temperature cannot be negative")
        masked = energies.masked_fill(~mask, torch.inf)
        minimum = masked.amin(dim=-1, keepdim=True)
        logits = -inverse_temperature * (energies - minimum)
        logits = logits.masked_fill(~mask, -torch.inf)
        return F.softmax(logits, dim=-1).masked_fill(~mask, 0.0)

    @staticmethod
    def _bank_indices(factor_ids: Tensor, bank: Tensor) -> Tensor:
        if factor_ids.ndim != 3 or factor_ids.shape[-1] != RULE_FACTOR_COUNT:
            raise ValueError("factor_ids must have [B,N,3] shape")
        if bank.ndim != 2 or bank.shape[-1] != RULE_FACTOR_COUNT:
            raise ValueError("factor bank must have [H,3] shape")
        matches = (factor_ids[:, :, None] == bank[None, None]).all(dim=-1)
        if not torch.all(matches.sum(dim=-1) == 1):
            raise ValueError("every proposed factor code must occur once in the bank")
        return matches.to(dtype=torch.long).argmax(dim=-1)

    def _validate_carried(
        self,
        carried: GRAMPopulationMemory,
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        if carried.batch_size != batch_size:
            raise ValueError("carried population batch size does not match evidence")
        if carried.capacity != self.carry_limit:
            raise ValueError("carried population capacity does not match carry_limit")
        if carried.factor_ids.device != device:
            raise ValueError("carried population and evidence must share a device")
        if carried.mask.dtype != torch.bool:
            raise TypeError("carried population mask must be boolean")
        if carried.factor_ids.shape != (
            batch_size,
            self.carry_limit,
            RULE_FACTOR_COUNT,
        ):
            raise ValueError("carried factor ids have incompatible shape")
        if torch.any(carried.factor_ids[carried.mask] < 0) or torch.any(
            carried.factor_ids[carried.mask] >= RULE_FACTOR_CARDINALITY
        ):
            raise ValueError("valid carried factor ids must lie in [0,4)")
        if carried.generation < 1:
            raise ValueError("carried population generation must be positive")

    def _fresh_proposals(
        self,
        public: RuleGridTensorBatch,
        *,
        recursive_steps: int,
        generator: torch.Generator | None,
        seed: int | None,
        temperature: float,
    ) -> Tensor:
        if self.proposal_mode == "gram":
            inference = self.proposer.sample_width_candidates(
                public,
                width=self.proposals,
                recursive_steps=recursive_steps,
                generator=generator,
                seed=seed,
                temperature=temperature,
                sample_noise=True,
            )
            return inference.factor_ids.detach()

        current_generator = self.proposer._make_generator(
            public.support_states,
            generator=generator,
            seed=seed,
        )
        return torch.randint(
            RULE_FACTOR_CARDINALITY,
            (public.batch_size, self.proposals, RULE_FACTOR_COUNT),
            dtype=torch.long,
            device=public.support_states.device,
            generator=current_generator,
        )

    def _score_full_history(
        self,
        public: RuleGridTensorBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Evaluate the verifier bank once and return detached score tables."""

        scores = score_hypothesis_bank(
            self.verifier_executor,
            public.support_states,
            public.support_actions,
            public.support_targets,
            public.support_mask,
            public.support_action_mask,
        )
        if self.energy_scale is None:
            height, width = public.support_states.shape[-2:]
            scales = (
                public.support_mask.sum(dim=1).to(scores.proper_nll_per_cell.dtype)
                * height
                * width
            )
        else:
            scales = torch.full(
                (public.batch_size,),
                self.energy_scale,
                dtype=scores.proper_nll_per_cell.dtype,
                device=scores.proper_nll_per_cell.device,
            )
        per_cell = (
            self.proper_weight * scores.proper_nll_per_cell
            + self.balanced_weight * scores.balanced_nll_per_cell
        )
        energies = (per_cell * scales[:, None]).detach()
        return (
            scores.factor_ids.detach(),
            energies,
            scores.map_error_cells.detach(),
            scales.detach(),
        )

    def _deduplicate_candidates(
        self,
        proposals: Tensor,
        carried: GRAMPopulationMemory | None,
        *,
        bank: Tensor,
        energy_table: Tensor,
        map_error_table: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Deduplicate then sort by compatibility, error, energy, and code."""

        batch_size = proposals.shape[0]
        capacity = self.proposals + self.carry_limit
        device = proposals.device
        ids = torch.full(
            (batch_size, capacity, RULE_FACTOR_COUNT),
            -1,
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros((batch_size, capacity), dtype=torch.bool, device=device)
        energies = torch.full(
            (batch_size, capacity),
            torch.inf,
            dtype=energy_table.dtype,
            device=device,
        )
        max_errors = torch.iinfo(map_error_table.dtype).max
        map_errors = torch.full(
            (batch_size, capacity),
            max_errors,
            dtype=map_error_table.dtype,
            device=device,
        )
        multiplicities = torch.zeros(
            (batch_size, capacity), dtype=torch.long, device=device
        )
        proposal_multiplicities = torch.zeros_like(multiplicities)
        was_carried = torch.zeros_like(mask)
        bank_rows = [
            tuple(int(value) for value in row)
            for row in bank.detach().cpu().tolist()
        ]
        bank_index = {code: index for index, code in enumerate(bank_rows)}

        for batch_index in range(batch_size):
            groups: dict[tuple[int, int, int], dict[str, int | bool]] = {}
            for row in proposals[batch_index].detach().cpu().tolist():
                code = tuple(int(value) for value in row)
                record = groups.setdefault(
                    code,
                    {"total": 0, "proposal": 0, "carried": False},
                )
                record["total"] = int(record["total"]) + 1
                record["proposal"] = int(record["proposal"]) + 1
            if carried is not None:
                valid_carried = carried.factor_ids[batch_index][
                    carried.mask[batch_index]
                ]
                for row in valid_carried.detach().cpu().tolist():
                    code = tuple(int(value) for value in row)
                    record = groups.setdefault(
                        code,
                        {"total": 0, "proposal": 0, "carried": False},
                    )
                    record["total"] = int(record["total"]) + 1
                    record["carried"] = True

            entries = []
            for code, record in groups.items():
                index = bank_index.get(code)
                if index is None:
                    raise ValueError("candidate code is outside the verifier bank")
                error = int(map_error_table[batch_index, index].detach().cpu())
                energy = energy_table[batch_index, index]
                entries.append((code, record, index, error, energy))
            entries.sort(
                key=lambda item: (
                    item[3] != 0,
                    item[3],
                    float(item[4].detach().cpu()),
                    item[0],
                )
            )
            for candidate_index, (code, record, index, error, energy) in enumerate(
                entries
            ):
                ids[batch_index, candidate_index] = torch.tensor(
                    code, dtype=torch.long, device=device
                )
                mask[batch_index, candidate_index] = True
                energies[batch_index, candidate_index] = energy
                map_errors[batch_index, candidate_index] = error
                multiplicities[batch_index, candidate_index] = int(record["total"])
                proposal_multiplicities[batch_index, candidate_index] = int(
                    record["proposal"]
                )
                was_carried[batch_index, candidate_index] = bool(record["carried"])

        map_exact = mask & map_errors.eq(0)
        return (
            ids,
            mask,
            energies,
            map_errors,
            map_exact,
            multiplicities,
            proposal_multiplicities,
            was_carried,
        )

    def _retain_population(
        self,
        candidate_ids: Tensor,
        candidate_mask: Tensor,
        candidate_energies: Tensor,
        candidate_map_errors: Tensor,
        *,
        generation: int,
    ) -> tuple[GRAMPopulationMemory, Tensor]:
        """Retain only the best non-empty MAP-compatibility stratum."""

        batch_size, candidate_capacity = candidate_mask.shape
        device = candidate_ids.device
        selected_candidate_mask = torch.zeros_like(candidate_mask)
        for batch_index in range(batch_size):
            valid = candidate_mask[batch_index]
            exact = valid & candidate_map_errors[batch_index].eq(0)
            if bool(exact.any()):
                eligible_indices = exact.nonzero(as_tuple=False).flatten()
            else:
                minimum_error = candidate_map_errors[batch_index, valid].amin()
                eligible_indices = (
                    valid & candidate_map_errors[batch_index].eq(minimum_error)
                ).nonzero(as_tuple=False).flatten()
            # Candidates are already energy-sorted inside each error stratum.
            eligible_indices = eligible_indices[: self.carry_limit]
            selected_candidate_mask[batch_index, eligible_indices] = True

        all_candidate_weights = self._stable_ranking_weights(
            candidate_energies,
            selected_candidate_mask,
            inverse_temperature=self.ranking_inverse_temperature,
        )
        ids = torch.full(
            (batch_size, self.carry_limit, RULE_FACTOR_COUNT),
            -1,
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros(
            (batch_size, self.carry_limit), dtype=torch.bool, device=device
        )
        ranking_weights = torch.zeros(
            (batch_size, self.carry_limit),
            dtype=candidate_energies.dtype,
            device=device,
        )
        energies = torch.full_like(ranking_weights, torch.inf)
        maximum_error = torch.iinfo(candidate_map_errors.dtype).max
        map_errors = torch.full(
            (batch_size, self.carry_limit),
            maximum_error,
            dtype=candidate_map_errors.dtype,
            device=device,
        )
        for batch_index in range(batch_size):
            selected = selected_candidate_mask[batch_index].nonzero(
                as_tuple=False
            ).flatten()
            count = int(selected.numel())
            ids[batch_index, :count] = candidate_ids[batch_index, selected]
            mask[batch_index, :count] = True
            ranking_weights[batch_index, :count] = all_candidate_weights[
                batch_index, selected
            ]
            energies[batch_index, :count] = candidate_energies[
                batch_index, selected
            ]
            map_errors[batch_index, :count] = candidate_map_errors[
                batch_index, selected
            ]
        memory = GRAMPopulationMemory(
            factor_ids=ids,
            mask=mask,
            ranking_weights=ranking_weights,
            energies=energies,
            map_error_cells=map_errors,
            map_exact=mask & map_errors.eq(0),
            generation=generation,
        )
        return memory, all_candidate_weights

    @torch.no_grad()
    def search(
        self,
        batch: RuleGridTensorBatch,
        carried: GRAMPopulationMemory | None = None,
        *,
        recursive_steps: int | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        temperature: float | None = None,
    ) -> GRAMPopulationSearchResult:
        """Run one fresh-proposal, one-shot full-history search stage."""

        if generator is not None and seed is not None:
            raise ValueError("pass either generator or seed, not both")
        steps = self.recursive_steps if recursive_steps is None else recursive_steps
        if steps <= 0:
            raise ValueError("recursive_steps must be positive")
        current_temperature = (
            self.proposer.temperature if temperature is None else temperature
        )
        if current_temperature <= 0:
            raise ValueError("temperature must be positive")

        # Hard leakage boundary: neither proposal nor verifier can observe
        # query labels or privileged behavior panels.
        public = self.proposer._support_only_batch(batch)
        public.validate(self.proposer.config)
        if carried is not None:
            self._validate_carried(
                carried,
                batch_size=public.batch_size,
                device=public.support_states.device,
            )

        proposed = self._fresh_proposals(
            public,
            recursive_steps=steps,
            generator=generator,
            seed=seed,
            temperature=current_temperature,
        ).detach()
        bank, energy_table, map_error_table, scales = self._score_full_history(public)
        candidates = self._deduplicate_candidates(
            proposed,
            carried,
            bank=bank,
            energy_table=energy_table,
            map_error_table=map_error_table,
        )
        generation = 1 if carried is None else carried.generation + 1
        population, candidate_weights = self._retain_population(
            candidates[0],
            candidates[1],
            candidates[2],
            candidates[3],
            generation=generation,
        )
        result = GRAMPopulationSearchResult(
            population=population,
            proposed_factor_ids=proposed,
            candidate_factor_ids=candidates[0],
            candidate_mask=candidates[1],
            candidate_energies=candidates[2],
            candidate_map_error_cells=candidates[3],
            candidate_map_exact=candidates[4],
            candidate_selection_weights=candidate_weights,
            candidate_multiplicities=candidates[5],
            candidate_proposal_multiplicities=candidates[6],
            candidate_was_carried=candidates[7],
            applied_energy_scales=scales,
            verifier_bank_evaluations=1,
            proposal_mode=self.proposal_mode,
        )
        tensors = (
            result.proposed_factor_ids,
            result.candidate_energies,
            result.population.factor_ids,
            result.population.ranking_weights,
        )
        if any(tensor.requires_grad for tensor in tensors):
            raise AssertionError("population-search outputs must be detached")
        self.proposer.executor.eval()
        self.verifier_executor.eval()
        return result

    def update(
        self,
        batch: RuleGridTensorBatch,
        belief: GRAMPopulationMemory | None = None,
        *,
        accumulate_weights: bool = False,
        recursive_steps: int | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        temperature: float | None = None,
    ) -> GRAMPopulationSearchResult:
        """Compatibility alias for ``search`` using complete histories only."""

        if accumulate_weights:
            raise ValueError(
                "population search never accumulates path weights; pass a complete "
                "history with accumulate_weights=False"
            )
        return self.search(
            batch,
            carried=belief,
            recursive_steps=recursive_steps,
            generator=generator,
            seed=seed,
            temperature=temperature,
        )

    def infer_support(
        self,
        batch: RuleGridTensorBatch,
        *,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        temperature: float | None = None,
    ) -> GRAMPopulationSearchResult:
        """Fresh search without carried candidates."""

        return self.search(
            batch,
            carried=None,
            generator=generator,
            seed=seed,
            temperature=temperature,
        )

    @torch.no_grad()
    def topk_inference(
        self,
        result: GRAMPopulationSearchResult,
        *,
        k: int = 4,
    ) -> CausalMechanismInference:
        """Expose retained codes through the established fixed-width API."""

        if k <= 0:
            raise ValueError("k must be positive")
        population = result.population
        ids = torch.empty(
            (population.batch_size, k, RULE_FACTOR_COUNT),
            dtype=torch.long,
            device=population.factor_ids.device,
        )
        for batch_index in range(population.batch_size):
            valid = population.factor_ids[batch_index][population.mask[batch_index]]
            if valid.shape[0] == 0:
                raise AssertionError("retained population cannot be empty")
            take = valid[:k]
            if take.shape[0] < k:
                take = torch.cat((take, take[-1:].expand(k - take.shape[0], -1)))
            ids[batch_index] = take
        codes = F.one_hot(ids, RULE_FACTOR_CARDINALITY).to(
            dtype=next(self.proposer.parameters()).dtype
        )
        logits = torch.where(
            codes.bool(),
            torch.zeros_like(codes),
            torch.full_like(codes, -30.0),
        )
        probabilities = F.softmax(logits, dim=-1)
        return CausalMechanismInference(
            factor_logits=logits,
            factor_probabilities=probabilities,
            factor_codes=codes,
            factor_ids=ids,
            rule_latents=self.proposer.rule_latents_from_codes(codes),
        )


# Backward-compatible import names for the historical filename.  They refer to
# population search objects and carry no SMC/posterior semantics.
GRAMSMCBelief = GRAMPopulationMemory
GRAMSMCResult = GRAMPopulationSearchResult
GRAMSMCBeliefUpdater = GRAMVerifierPopulationSearch


__all__ = [
    "GRAMPopulationMemory",
    "GRAMPopulationSearchResult",
    "GRAMSMCBelief",
    "GRAMSMCBeliefUpdater",
    "GRAMSMCResult",
    "GRAMVerifierPopulationSearch",
]
