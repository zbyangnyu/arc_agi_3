"""Minimal GRAM-style recursive proposer for factorized RuleGrid rules.

This module isolates the part of Generative Recursive Reasoning that is useful
for the causal K=4 ceiling: stochastic *high-level* guidance creates parallel
rule hypotheses, while one shared pair of high/low recurrent cores refines all
hypotheses for a configurable number of recursive steps.  It deliberately does
not turn the frozen executor into a stochastic decoder.

The variational boundary is explicit:

* the inference prior ``p(epsilon | support, recursive state)`` reads public
  support transitions only;
* the training posterior ``q(epsilon | support, unordered behavior set,
  recursive state)`` additionally reads privileged behavior-set supervision;
* both use diagonal Gaussians, reparameterized samples, and an analytic KL;
* every recursive step is supervised against the inherited detached costs for
  all 64 exact integer rule codes; and
* the same learned high/low initial state is repeated across width.  There is
  no trajectory embedding, so width can differ only through sampled guidance.

This is a proposer diagnostic, not a claim that its latent trajectories are
already identifiable causal variables.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover - optional neural dependency.
    raise ImportError("prp_wm.gram_causal_rules requires PyTorch") from error

from .causal_rules import CausalMechanismInference
from .discrete_causal_rules import ExpectedDiscreteCausalK4
from .latent_rules import (
    RULE_FACTOR_CARDINALITY,
    RULE_FACTOR_COUNT,
    OracleFactorExecutor,
    outcome_map,
)
from .neural import RuleGridTensorBatch


@dataclass(frozen=True)
class DiagonalGaussian:
    """Parameters of a diagonal Gaussian guidance distribution."""

    mean: Tensor  # [B,W,E]
    log_variance: Tensor  # [B,W,E]

    @property
    def standard_deviation(self) -> Tensor:
        return torch.exp(0.5 * self.log_variance)


@dataclass(frozen=True)
class GRAMRuleTrajectories:
    """All recursive states and rule predictions for one width sample.

    The leading dimension of per-step fields is recursive depth ``R``.  State
    tensors include the common initial state and therefore have leading size
    ``R + 1``.  Posterior fields are ``None`` for public-support inference.
    """

    high_states: Tensor  # [R+1,B,W,D]
    low_states: Tensor  # [R+1,B,W,D]
    factor_logits: Tensor  # [R,B,W,3,4]
    factor_probabilities: Tensor  # [R,B,W,3,4]
    factor_codes: Tensor  # straight-through hard one-hot [R,B,W,3,4]
    factor_ids: Tensor  # [R,B,W,3]
    rule_latents: Tensor  # [R,B,W,D]
    prior_means: Tensor  # [R,B,W,E]
    prior_log_variances: Tensor  # [R,B,W,E]
    standard_noises: Tensor  # [R,B,W,E]
    guidance_samples: Tensor  # [R,B,W,E]
    posterior_means: Tensor | None = None  # [R,B,W,E]
    posterior_log_variances: Tensor | None = None  # [R,B,W,E]

    @property
    def recursive_steps(self) -> int:
        return int(self.factor_logits.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.factor_logits.shape[1])

    @property
    def width(self) -> int:
        return int(self.factor_logits.shape[2])

    @property
    def used_training_posterior(self) -> bool:
        return self.posterior_means is not None

    @property
    def final_inference(self) -> CausalMechanismInference:
        """Expose final width candidates through the established inference API."""

        return CausalMechanismInference(
            factor_logits=self.factor_logits[-1],
            factor_probabilities=self.factor_probabilities[-1],
            factor_codes=self.factor_codes[-1],
            factor_ids=self.factor_ids[-1],
            rule_latents=self.rule_latents[-1],
        )


@dataclass(frozen=True)
class GRAMExpectedDiscreteLoss:
    """Deeply supervised exact-code objective plus variational guidance KL."""

    total: Tensor
    reconstruction: Tensor
    kl: Tensor
    set_cost: Tensor
    validity_cost: Tensor
    diversity_barrier: Tensor
    joint_entropy: Tensor
    mean_top_probability: Tensor
    step_objectives: Tensor  # [R]
    step_set_costs: Tensor  # [R]
    step_validity_costs: Tensor  # [R]
    step_diversity_barriers: Tensor  # [R]
    step_joint_entropies: Tensor  # [R]
    step_kls: Tensor  # [R]
    deep_supervision_weights: Tensor  # detached [R]
    trajectories: GRAMRuleTrajectories
    joint_probabilities: Tensor  # [R,B,K,64]
    behavior_costs: Tensor  # detached [B,64,4]
    support_costs: Tensor  # detached [B,64]

    @property
    def inference(self) -> CausalMechanismInference:
        return self.trajectories.final_inference

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().cpu()),
            "loss_reconstruction": float(self.reconstruction.detach().cpu()),
            "loss_kl": float(self.kl.detach().cpu()),
            "loss_set_cost": float(self.set_cost.detach().cpu()),
            "loss_validity_cost": float(self.validity_cost.detach().cpu()),
            "loss_diversity_barrier": float(
                self.diversity_barrier.detach().cpu()
            ),
            "joint_entropy_nats": float(self.joint_entropy.detach().cpu()),
            "mean_top_rule_probability": float(
                self.mean_top_probability.detach().cpu()
            ),
        }


@dataclass(frozen=True)
class GRAMPublicCoverageLoss:
    """Public-support-only full-version-space coverage objective.

    The compatible-code mask is computed by comparing the frozen executor's
    MAP outcome for every one of the 64 integer codes with the observed public
    support outcomes.  It is detached and contains no selected/true program.
    """

    total: Tensor
    coverage: Tensor
    axis_balance: Tensor
    invalid_mass: Tensor
    joint_entropy: Tensor
    mean_top_probability: Tensor
    step_objectives: Tensor  # [R]
    step_coverage: Tensor  # [R]
    step_axis_balance: Tensor  # [R]
    step_invalid_mass: Tensor  # [R]
    step_joint_entropies: Tensor  # [R]
    deep_supervision_weights: Tensor  # detached [R]
    trajectories: GRAMRuleTrajectories
    joint_probabilities: Tensor  # [R,B,4,64]
    compatible_mask: Tensor  # detached bool [B,64]
    compatible_indices: Tensor  # detached long [B,4]

    @property
    def inference(self) -> CausalMechanismInference:
        return self.trajectories.final_inference

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().cpu()),
            "loss_coverage": float(self.coverage.detach().cpu()),
            "loss_axis_balance": float(self.axis_balance.detach().cpu()),
            "loss_invalid_mass": float(self.invalid_mass.detach().cpu()),
            "joint_entropy_nats": float(self.joint_entropy.detach().cpu()),
            "mean_top_rule_probability": float(
                self.mean_top_probability.detach().cpu()
            ),
        }


class GRAMFactorizedCausalK4(ExpectedDiscreteCausalK4):
    """Recursive stochastic-width proposer over three four-way rule factors."""

    def __init__(
        self,
        executor: OracleFactorExecutor,
        *,
        recursive_steps: int = 3,
        guidance_dim: int | None = None,
        attention_layers: int = 2,
        temperature: float = 1.0,
        minimum_log_variance: float = -8.0,
        maximum_log_variance: float = 4.0,
        initial_log_variance: float = -2.0,
        truncate_between_steps: bool = True,
    ) -> None:
        if recursive_steps <= 0:
            raise ValueError("recursive_steps must be positive")
        if guidance_dim is not None and guidance_dim <= 0:
            raise ValueError("guidance_dim must be positive")
        if minimum_log_variance >= maximum_log_variance:
            raise ValueError(
                "minimum_log_variance must be below maximum_log_variance"
            )
        if not minimum_log_variance <= initial_log_variance <= maximum_log_variance:
            raise ValueError("initial_log_variance must lie within the clamp bounds")
        super().__init__(
            executor,
            attention_layers=attention_layers,
            temperature=temperature,
        )
        self.recursive_steps = int(recursive_steps)
        self.guidance_dim = int(guidance_dim or self.config.rule_dim)
        self.minimum_log_variance = float(minimum_log_variance)
        self.maximum_log_variance = float(maximum_log_variance)
        self.initial_log_variance = float(initial_log_variance)
        self.truncate_between_steps = bool(truncate_between_steps)

        # The parent's amortized K-slot cross-attention path is intentionally
        # replaced.  GRAM width starts from one repeated state rather than four
        # learned slot identities.  Its exact-code cost machinery is retained.
        del self.initial_slots
        del self.cross_layers
        del self.factor_heads

        dimension = self.config.rule_dim
        hidden = self.config.attention_ffn
        self.support_context = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dimension),
        )
        self.behavior_item_context = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dimension),
        )
        self.initial_high = nn.Parameter(torch.empty(dimension))
        self.initial_low = nn.Parameter(torch.empty(dimension))
        nn.init.normal_(self.initial_high, mean=0.0, std=0.02)
        nn.init.normal_(self.initial_low, mean=0.0, std=0.02)

        # One prior and one training posterior are shared across recursion.
        # Both condition on the deterministic high-level proposal and refined
        # low-level state.  The posterior additionally sees its target behavior.
        self.prior_head = self._gaussian_head(3 * dimension, hidden)
        self.posterior_head = self._gaussian_head(4 * dimension, hidden)
        self.posterior_behavior_to_guidance = nn.Linear(
            dimension,
            self.guidance_dim,
            bias=False,
        )
        nn.init.orthogonal_(self.posterior_behavior_to_guidance.weight)

        # Match GRAM's transition order: deterministically refine the low state,
        # form a deterministic high-level proposal, then add learned stochastic
        # guidance as a residual.  Noise never enters the low-level core.
        self.high_core = nn.GRUCell(2 * dimension, dimension)
        self.low_core = nn.GRUCell(2 * dimension, dimension)
        self.guidance_to_high = nn.Linear(
            self.guidance_dim,
            dimension,
            bias=False,
        )
        if self.guidance_dim == dimension:
            nn.init.eye_(self.guidance_to_high.weight)
        else:
            nn.init.orthogonal_(self.guidance_to_high.weight)
        self.readout = nn.Sequential(
            nn.LayerNorm(2 * dimension),
            nn.Linear(2 * dimension, dimension),
            nn.SiLU(),
        )
        self.factor_heads = nn.ModuleList(
            nn.Linear(dimension, RULE_FACTOR_CARDINALITY)
            for _ in range(RULE_FACTOR_COUNT)
        )

    def _gaussian_head(self, input_dimension: int, hidden: int) -> nn.Module:
        head = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.guidance_dim),
        )
        # Start near a shared, low-variance process rather than injecting unit
        # Gaussian noise before the recursive core has learned a useful scale.
        nn.init.normal_(head[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(head[-1].bias)
        with torch.no_grad():
            head[-1].bias[self.guidance_dim :].fill_(
                self.initial_log_variance
            )
        return head

    @staticmethod
    def _support_only_batch(batch: RuleGridTensorBatch) -> RuleGridTensorBatch:
        """Drop every query/privileged field before inference validation."""

        return RuleGridTensorBatch(
            support_states=batch.support_states,
            support_actions=batch.support_actions,
            support_targets=batch.support_targets,
            support_mask=batch.support_mask,
            support_action_mask=batch.support_action_mask,
        )

    @staticmethod
    def _masked_mean(tokens: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _support_summary(self, batch: RuleGridTensorBatch) -> Tensor:
        support = self._support_only_batch(batch)
        support.validate(self.config)
        tokens = self._transition_tokens(support)
        return self.support_context(self._masked_mean(tokens, support.support_mask))

    def _behavior_contexts(self, batch: RuleGridTensorBatch) -> Tensor:
        """Encode each unordered behavior as one target-conditioned trajectory.

        The returned axis is an unordered set axis: the same item encoder is
        applied to every class, and there is no positional or class embedding.
        A permutation of panels therefore only permutes these contexts.
        """

        if (
            batch.query_states is None
            or batch.query_actions is None
            or batch.behavior_targets is None
            or batch.behavior_mass is None
        ):
            raise ValueError(
                "training posterior requires query inputs and an unordered behavior set"
            )
        batch.validate(self.config)
        batch_size, classes, queries, height, width = (
            batch.behavior_targets.shape
        )
        if classes != self.config.particles:
            raise ValueError("training posterior requires exactly four behaviors")
        if not torch.all(batch.behavior_mass > 0):
            raise ValueError("all four behavior classes must be valid")

        states = (
            batch.query_states[:, None]
            .expand(-1, classes, -1, -1, -1)
            .reshape(batch_size * classes, queries, height, width)
        )
        if batch.query_actions.ndim == 3:
            actions = (
                batch.query_actions[:, None]
                .expand(-1, classes, -1, -1)
                .reshape(batch_size * classes, queries, -1)
            )
        else:
            atoms = batch.query_actions.shape[2]
            actions = (
                batch.query_actions[:, None]
                .expand(-1, classes, -1, -1, -1)
                .reshape(batch_size * classes, queries, atoms, -1)
            )
        action_mask = None
        if batch.query_action_mask is not None:
            atoms = batch.query_action_mask.shape[-1]
            action_mask = (
                batch.query_action_mask[:, None]
                .expand(-1, classes, -1, -1)
                .reshape(batch_size * classes, queries, atoms)
            )
        behavior_batch = RuleGridTensorBatch(
            support_states=states,
            support_actions=actions,
            support_targets=batch.behavior_targets.reshape(
                batch_size * classes, queries, height, width
            ),
            support_mask=torch.ones(
                (batch_size * classes, queries),
                dtype=torch.bool,
                device=states.device,
            ),
            support_action_mask=action_mask,
        )
        class_tokens = self._transition_tokens(behavior_batch).mean(dim=1)
        return self.behavior_item_context(class_tokens).reshape(
            batch_size, classes, self.config.rule_dim
        )

    def _distribution(
        self,
        head: nn.Module,
        inputs: Tensor,
    ) -> DiagonalGaussian:
        mean, log_variance = head(inputs).chunk(2, dim=-1)
        return DiagonalGaussian(
            mean=mean,
            log_variance=log_variance.clamp(
                min=self.minimum_log_variance,
                max=self.maximum_log_variance,
            ),
        )

    @staticmethod
    def analytic_gaussian_kl(
        posterior: DiagonalGaussian,
        prior: DiagonalGaussian,
    ) -> Tensor:
        """Return ``KL(q || p)`` summed over the final latent dimension."""

        if posterior.mean.shape != prior.mean.shape:
            raise ValueError("posterior and prior means must have equal shape")
        if posterior.log_variance.shape != posterior.mean.shape:
            raise ValueError("posterior log variance must match its mean")
        if prior.log_variance.shape != prior.mean.shape:
            raise ValueError("prior log variance must match its mean")
        variance_ratio = torch.exp(
            posterior.log_variance - prior.log_variance
        )
        squared_mean = (
            (posterior.mean - prior.mean).square()
            * torch.exp(-prior.log_variance)
        )
        return 0.5 * (
            prior.log_variance
            - posterior.log_variance
            + variance_ratio
            + squared_mean
            - 1.0
        ).sum(dim=-1)

    @staticmethod
    def _make_generator(
        reference: Tensor,
        *,
        generator: torch.Generator | None,
        seed: int | None,
    ) -> torch.Generator | None:
        if generator is not None and seed is not None:
            raise ValueError("pass either generator or seed, not both")
        if seed is None:
            return generator
        local = torch.Generator(device=reference.device)
        local.manual_seed(seed)
        return local

    def _initial_states(
        self,
        *,
        batch_size: int,
        width: int,
    ) -> tuple[Tensor, Tensor]:
        high = self.initial_high.view(1, 1, -1).expand(
            batch_size, width, -1
        )
        low = self.initial_low.view(1, 1, -1).expand(
            batch_size, width, -1
        )
        return high, low

    def _run_trajectories(
        self,
        support_summary: Tensor,
        *,
        width: int,
        recursive_steps: int,
        behavior_summary: Tensor | None,
        generator: torch.Generator | None,
        seed: int | None,
        temperature: float,
        sample_noise: bool,
    ) -> GRAMRuleTrajectories:
        if width <= 0:
            raise ValueError("width must be positive")
        if recursive_steps <= 0:
            raise ValueError("recursive_steps must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        batch_size, dimension = support_summary.shape
        current_generator = self._make_generator(
            support_summary,
            generator=generator,
            seed=seed,
        )
        support = support_summary[:, None].expand(-1, width, -1)
        behavior = None
        if behavior_summary is not None:
            if behavior_summary.shape != (batch_size, width, dimension):
                raise ValueError(
                    "behavior contexts must have [B,width,rule_dim] shape"
                )
            behavior = behavior_summary
        high, low = self._initial_states(batch_size=batch_size, width=width)

        high_states = [high]
        low_states = [low]
        logits_steps: list[Tensor] = []
        probability_steps: list[Tensor] = []
        code_steps: list[Tensor] = []
        id_steps: list[Tensor] = []
        latent_steps: list[Tensor] = []
        prior_means: list[Tensor] = []
        prior_log_variances: list[Tensor] = []
        posterior_means: list[Tensor] = []
        posterior_log_variances: list[Tensor] = []
        standard_noises: list[Tensor] = []
        guidance_samples: list[Tensor] = []

        for _ in range(recursive_steps):
            if self.truncate_between_steps and len(logits_steps) > 0:
                # GRAM trains consecutive supervision steps with a truncated
                # surrogate: each decoded step receives a local gradient while
                # the previous recursive state is treated as fixed context.
                high = high.detach()
                low = low.detach()

            flat_high = high.reshape(batch_size * width, dimension)
            flat_low = low.reshape(batch_size * width, dimension)
            low = self.low_core(
                torch.cat((support, high), dim=-1).reshape(
                    batch_size * width, -1
                ),
                flat_low,
            ).reshape(batch_size, width, dimension)
            deterministic_high = self.high_core(
                torch.cat((support, low), dim=-1).reshape(
                    batch_size * width, -1
                ),
                flat_high,
            ).reshape(batch_size, width, dimension)

            prior = self._distribution(
                self.prior_head,
                torch.cat((support, deterministic_high, low), dim=-1),
            )
            posterior = None
            if behavior is not None:
                posterior = self._distribution(
                    self.posterior_head,
                    torch.cat(
                        (support, behavior, deterministic_high, low),
                        dim=-1,
                    ),
                )
                # A direct target-to-guidance path prevents the small posterior
                # network from initially ignoring y.  It is still a learned q
                # parameterization: the public prior never sees this behavior
                # context and must cover it through the balanced trajectory KL.
                posterior = DiagonalGaussian(
                    mean=(
                        posterior.mean
                        + self.posterior_behavior_to_guidance(behavior)
                    ),
                    log_variance=posterior.log_variance,
                )
            proposal = posterior if posterior is not None else prior
            # Each posterior trajectory is attached to one unordered behavior
            # class.  A shared base draw makes the Monte-Carlo objective exactly
            # permutation equivariant: reordering behavior panels only reorders
            # q trajectories.  Every marginal remains a valid reparameterized
            # Gaussian sample.  Prior inference instead uses iid width draws.
            if sample_noise:
                noise_shape = (
                    (batch_size, 1, self.guidance_dim)
                    if posterior is not None
                    else proposal.mean.shape
                )
                standard_noise = torch.randn(
                    noise_shape,
                    dtype=proposal.mean.dtype,
                    device=proposal.mean.device,
                    generator=current_generator,
                )
                if posterior is not None:
                    standard_noise = standard_noise.expand(-1, width, -1)
            else:
                standard_noise = torch.zeros_like(proposal.mean)
            guidance = (
                proposal.mean
                + proposal.standard_deviation * standard_noise
            )
            high = deterministic_high + self.guidance_to_high(guidance)

            readout = self.readout(torch.cat((high, low), dim=-1))
            logits = torch.stack(
                [head(readout) for head in self.factor_heads],
                dim=2,
            )
            probabilities = F.softmax(logits / temperature, dim=-1)
            factor_ids = probabilities.argmax(dim=-1)
            hard = F.one_hot(
                factor_ids,
                RULE_FACTOR_CARDINALITY,
            ).to(dtype=probabilities.dtype)
            factor_codes = hard + probabilities - probabilities.detach()

            high_states.append(high)
            low_states.append(low)
            logits_steps.append(logits)
            probability_steps.append(probabilities)
            code_steps.append(factor_codes)
            id_steps.append(factor_ids)
            latent_steps.append(self.rule_latents_from_codes(factor_codes))
            prior_means.append(prior.mean)
            prior_log_variances.append(prior.log_variance)
            standard_noises.append(standard_noise)
            guidance_samples.append(guidance)
            if posterior is not None:
                posterior_means.append(posterior.mean)
                posterior_log_variances.append(posterior.log_variance)

        return GRAMRuleTrajectories(
            high_states=torch.stack(high_states),
            low_states=torch.stack(low_states),
            factor_logits=torch.stack(logits_steps),
            factor_probabilities=torch.stack(probability_steps),
            factor_codes=torch.stack(code_steps),
            factor_ids=torch.stack(id_steps),
            rule_latents=torch.stack(latent_steps),
            prior_means=torch.stack(prior_means),
            prior_log_variances=torch.stack(prior_log_variances),
            standard_noises=torch.stack(standard_noises),
            guidance_samples=torch.stack(guidance_samples),
            posterior_means=(
                torch.stack(posterior_means) if posterior_means else None
            ),
            posterior_log_variances=(
                torch.stack(posterior_log_variances)
                if posterior_log_variances
                else None
            ),
        )

    def sample_trajectories(
        self,
        batch: RuleGridTensorBatch,
        *,
        width: int | None = None,
        recursive_steps: int | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        sample_noise: bool = True,
    ) -> GRAMRuleTrajectories:
        """Sample public-support trajectories from the prior only.

        Query tensors and privileged behavior supervision are discarded before
        validation, so changing either cannot affect this inference path.
        """

        current_temperature = self.temperature if temperature is None else temperature
        return self._run_trajectories(
            self._support_summary(batch),
            width=self.config.particles if width is None else width,
            recursive_steps=(
                self.recursive_steps
                if recursive_steps is None
                else recursive_steps
            ),
            behavior_summary=None,
            generator=generator,
            seed=seed,
            temperature=current_temperature,
            sample_noise=sample_noise,
        )

    def sample_training_trajectories(
        self,
        batch: RuleGridTensorBatch,
        *,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        sample_noise: bool = True,
    ) -> GRAMRuleTrajectories:
        """Sample K privileged training trajectories from the posterior."""

        current_temperature = self.temperature if temperature is None else temperature
        return self._run_trajectories(
            self._support_summary(batch),
            width=self.config.particles,
            recursive_steps=self.recursive_steps,
            behavior_summary=self._behavior_contexts(batch),
            generator=generator,
            seed=seed,
            temperature=current_temperature,
            sample_noise=sample_noise,
        )

    def sample_width_candidates(
        self,
        batch: RuleGridTensorBatch,
        *,
        width: int | None = None,
        recursive_steps: int | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        sample_noise: bool = True,
    ) -> CausalMechanismInference:
        """Return final prior-sampled candidates, allowing inference-time width."""

        return self.sample_trajectories(
            batch,
            width=width,
            recursive_steps=recursive_steps,
            generator=generator,
            seed=seed,
            temperature=temperature,
            sample_noise=sample_noise,
        ).final_inference

    def infer_support(
        self,
        batch: RuleGridTensorBatch,
        *,
        temperature: float | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        sample_noise: bool = True,
    ) -> CausalMechanismInference:
        """Compatibility inference entry point; it is prior and support only."""

        return self.sample_width_candidates(
            batch,
            width=self.config.particles,
            generator=generator,
            seed=seed,
            temperature=temperature,
            sample_noise=sample_noise,
        )

    @staticmethod
    def _deep_supervision_weights(
        steps: int,
        decay: float,
        reference: Tensor,
    ) -> Tensor:
        if decay <= 0:
            raise ValueError("deep_supervision_decay must be positive")
        powers = torch.arange(
            steps - 1,
            -1,
            -1,
            dtype=reference.dtype,
            device=reference.device,
        )
        weights = decay**powers
        return (weights / weights.sum()).detach()

    def public_support_exact_mask(
        self,
        batch: RuleGridTensorBatch,
    ) -> Tensor:
        """Return the all-64 MAP-compatible mask from public support only.

        No NLL threshold is involved: a code is compatible iff its frozen
        executor MAP grid equals every observed support target cell.  Query
        tensors and privileged behavior fields are dropped before validation
        and prediction.
        """

        support = self._support_only_batch(batch)
        support.validate(self.config)
        batch_size, steps, height, width = support.support_states.shape
        with torch.no_grad():
            prediction = self._predict_all_support_codes(support)
            maps = outcome_map(prediction).reshape(
                batch_size,
                steps,
                self.factor_bank.shape[0],
                height,
                width,
            )
            valid = support.support_mask[:, :, None, None, None]
            wrong = maps.ne(support.support_targets[:, :, None]) & valid
            exact = ~wrong.any(dim=(1, 3, 4))
        return exact.detach()

    def coverage_losses(
        self,
        batch: RuleGridTensorBatch,
        *,
        coverage_weight: float = 1.0,
        axis_balance_weight: float = 0.10,
        validity_weight: float = 0.10,
        assignment_temperature: float = 0.05,
        deep_supervision_decay: float = 0.5,
        temperature: float | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        sample_noise: bool = True,
    ) -> GRAMPublicCoverageLoss:
        """Train the public prior to cover all four support-compatible codes.

        Four iid public-prior trajectories are matched one-to-one to the four
        exact compatible joint codes.  Matching enumerates all 4! assignments,
        making the loss invariant to both trajectory and compatible-code order.
        The target set is derived exclusively from public support MAP equality;
        the training posterior, behavior panels, and true program are unused.
        """

        if min(
            coverage_weight,
            axis_balance_weight,
            validity_weight,
            assignment_temperature,
        ) < 0:
            raise ValueError("loss weights and assignment temperature cannot be negative")
        support = self._support_only_batch(batch)
        compatible_mask = self.public_support_exact_mask(support)
        compatible_counts = compatible_mask.sum(dim=-1)
        expected_count = self.config.particles
        if expected_count != 4:
            raise ValueError("public coverage requires the configured K=4 interface")
        if not torch.all(compatible_counts == expected_count):
            counts = compatible_counts.detach().cpu().tolist()
            raise ValueError(
                "public t0 support must have exactly four MAP-compatible codes; "
                f"got {counts}"
            )
        bank_indices = torch.arange(
            self.factor_bank.shape[0],
            device=compatible_mask.device,
        )[None].expand(support.batch_size, -1)
        compatible_indices = bank_indices[compatible_mask].reshape(
            support.batch_size,
            expected_count,
        ).detach()
        trajectories = self.sample_trajectories(
            support,
            width=expected_count,
            recursive_steps=self.recursive_steps,
            generator=generator,
            seed=seed,
            temperature=temperature,
            sample_noise=sample_noise,
        )
        current_temperature = self.temperature if temperature is None else temperature
        target_axis_mass = torch.stack(
            [
                torch.einsum(
                    "br,rv->bv",
                    compatible_mask.to(dtype=trajectories.factor_logits.dtype),
                    F.one_hot(
                        self.factor_bank[:, axis],
                        RULE_FACTOR_CARDINALITY,
                    ).to(dtype=trajectories.factor_logits.dtype),
                )
                / expected_count
                for axis in range(RULE_FACTOR_COUNT)
            ],
            dim=1,
        ).detach()

        step_joint_probabilities: list[Tensor] = []
        step_coverage: list[Tensor] = []
        step_axis_balance: list[Tensor] = []
        step_invalid_mass: list[Tensor] = []
        step_entropies: list[Tensor] = []
        step_top_probabilities: list[Tensor] = []
        for logits in trajectories.factor_logits:
            axis_log_probabilities = F.log_softmax(
                logits / current_temperature,
                dim=-1,
            )
            selected = [
                axis_log_probabilities[:, :, axis, self.factor_bank[:, axis]]
                for axis in range(RULE_FACTOR_COUNT)
            ]
            joint_log_probabilities = torch.stack(selected).sum(dim=0)
            joint_probabilities = joint_log_probabilities.exp()
            compatible_log_probabilities = joint_log_probabilities.gather(
                dim=-1,
                index=compatible_indices[:, None].expand(-1, expected_count, -1),
            )
            step_joint_probabilities.append(joint_probabilities)
            step_coverage.append(
                self._soft_permutation_loss(
                    -compatible_log_probabilities,
                    assignment_temperature,
                )
            )

            mean_axis_probabilities = axis_log_probabilities.exp().mean(dim=1)
            positive_target = target_axis_mass > 0
            target_logs = torch.where(
                positive_target,
                target_axis_mass.clamp_min(
                    torch.finfo(target_axis_mass.dtype).tiny
                ).log(),
                torch.zeros_like(target_axis_mass),
            )
            axis_kl = torch.where(
                positive_target,
                target_axis_mass
                * (target_logs - mean_axis_probabilities.clamp_min(
                    torch.finfo(mean_axis_probabilities.dtype).tiny
                ).log()),
                torch.zeros_like(target_axis_mass),
            )
            step_axis_balance.append(axis_kl.sum(dim=-1).mean())
            compatible_mass = torch.einsum(
                "bkr,br->bk",
                joint_probabilities,
                compatible_mask.to(dtype=joint_probabilities.dtype),
            )
            step_invalid_mass.append((1.0 - compatible_mass).mean())
            step_entropies.append(
                -(joint_probabilities * joint_log_probabilities)
                .sum(dim=-1)
                .mean()
            )
            step_top_probabilities.append(
                joint_probabilities.amax(dim=-1).mean()
            )

        coverage = torch.stack(step_coverage)
        axis_balance = torch.stack(step_axis_balance)
        invalid_mass = torch.stack(step_invalid_mass)
        entropies = torch.stack(step_entropies)
        top_probabilities = torch.stack(step_top_probabilities)
        step_objectives = (
            coverage_weight * coverage
            + axis_balance_weight * axis_balance
            + validity_weight * invalid_mass
        )
        weights = self._deep_supervision_weights(
            self.recursive_steps,
            deep_supervision_decay,
            step_objectives,
        )
        return GRAMPublicCoverageLoss(
            total=(weights * step_objectives).sum(),
            coverage=(weights * coverage).sum(),
            axis_balance=(weights * axis_balance).sum(),
            invalid_mass=(weights * invalid_mass).sum(),
            joint_entropy=(weights * entropies).sum(),
            mean_top_probability=(weights * top_probabilities).sum(),
            step_objectives=step_objectives,
            step_coverage=coverage,
            step_axis_balance=axis_balance,
            step_invalid_mass=invalid_mass,
            step_joint_entropies=entropies,
            deep_supervision_weights=weights,
            trajectories=trajectories,
            joint_probabilities=torch.stack(step_joint_probabilities),
            compatible_mask=compatible_mask,
            compatible_indices=compatible_indices,
        )

    def losses(
        self,
        batch: RuleGridTensorBatch,
        *,
        kl_weight: float = 0.01,
        kl_balance: float = 0.8,
        validity_weight: float = 0.10,
        diversity_weight: float = 0.10,
        sharpening_weight: float = 0.0,
        proper_weight: float = 1.0,
        balanced_weight: float = 1.0,
        deep_supervision_decay: float = 1.0,
        temperature: float | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        sample_noise: bool = True,
    ) -> GRAMExpectedDiscreteLoss:
        """Train q with exact-code deep supervision and analytic ``KL(q||p)``."""

        if min(
            kl_weight,
            validity_weight,
            diversity_weight,
            sharpening_weight,
            proper_weight,
            balanced_weight,
        ) < 0:
            raise ValueError("loss weights cannot be negative")
        if not 0.0 <= kl_balance <= 1.0:
            raise ValueError("kl_balance must lie in [0,1]")
        trajectories = self.sample_training_trajectories(
            batch,
            generator=generator,
            seed=seed,
            temperature=temperature,
            sample_noise=sample_noise,
        )
        behavior_costs = self.discrete_behavior_costs(
            batch,
            proper_weight=proper_weight,
            balanced_weight=balanced_weight,
        )
        support_costs = self.discrete_support_costs(
            batch,
            proper_weight=proper_weight,
            balanced_weight=balanced_weight,
        )

        step_joint_probabilities: list[Tensor] = []
        step_set_costs: list[Tensor] = []
        step_validity_costs: list[Tensor] = []
        step_diversity: list[Tensor] = []
        step_entropies: list[Tensor] = []
        step_top_probabilities: list[Tensor] = []
        for logits in trajectories.factor_logits:
            axis_log_probabilities = F.log_softmax(
                logits / (self.temperature if temperature is None else temperature),
                dim=-1,
            )
            selected = [
                axis_log_probabilities[:, :, axis, self.factor_bank[:, axis]]
                for axis in range(RULE_FACTOR_COUNT)
            ]
            joint_log_probabilities = torch.stack(selected).sum(dim=0)
            joint_probabilities = joint_log_probabilities.exp()
            # During training, q trajectory k is conditioned on behavior k.
            # Direct target-aligned credit avoids a second assignment problem;
            # a behavior-set permutation simply permutes both aligned axes.
            expected_class_cost = torch.einsum(
                "bkr,brk->bk",
                joint_probabilities,
                behavior_costs,
            )
            step_joint_probabilities.append(joint_probabilities)
            step_set_costs.append(expected_class_cost.mean())
            step_validity_costs.append(
                torch.einsum(
                    "bkr,br->bk",
                    joint_probabilities,
                    support_costs,
                ).mean()
            )
            step_diversity.append(
                self._diversity_barrier(joint_probabilities)
            )
            step_entropies.append(
                -(joint_probabilities * joint_log_probabilities)
                .sum(dim=-1)
                .mean()
            )
            step_top_probabilities.append(
                joint_probabilities.amax(dim=-1).mean()
            )

        assert trajectories.posterior_means is not None
        assert trajectories.posterior_log_variances is not None
        posterior = DiagonalGaussian(
            trajectories.posterior_means,
            trajectories.posterior_log_variances,
        )
        prior = DiagonalGaussian(
            trajectories.prior_means,
            trajectories.prior_log_variances,
        )
        # Dreamer-style KL balancing, also used by GRAM, separates the pressure
        # to fit the inference prior to q from the weaker pressure regularizing
        # q toward p.  Detaching one side changes gradients, not the KL value.
        detached_posterior = DiagonalGaussian(
            posterior.mean.detach(),
            posterior.log_variance.detach(),
        )
        detached_prior = DiagonalGaussian(
            prior.mean.detach(),
            prior.log_variance.detach(),
        )
        prior_fitting_kl = self.analytic_gaussian_kl(
            detached_posterior,
            prior,
        ).mean(dim=(1, 2))
        posterior_regularization_kl = self.analytic_gaussian_kl(
            posterior,
            detached_prior,
        ).mean(dim=(1, 2))
        step_kls = (
            kl_balance * prior_fitting_kl
            + (1.0 - kl_balance) * posterior_regularization_kl
        )
        set_costs = torch.stack(step_set_costs)
        validity_costs = torch.stack(step_validity_costs)
        diversity_barriers = torch.stack(step_diversity)
        entropies = torch.stack(step_entropies)
        top_probabilities = torch.stack(step_top_probabilities)
        step_objectives = (
            set_costs
            + validity_weight * validity_costs
            + diversity_weight * diversity_barriers
            + sharpening_weight * entropies
        )
        weights = self._deep_supervision_weights(
            self.recursive_steps,
            deep_supervision_decay,
            step_objectives,
        )
        reconstruction = (weights * step_objectives).sum()
        kl = (weights * step_kls).sum()
        total = reconstruction + kl_weight * kl
        return GRAMExpectedDiscreteLoss(
            total=total,
            reconstruction=reconstruction,
            kl=kl,
            set_cost=(weights * set_costs).sum(),
            validity_cost=(weights * validity_costs).sum(),
            diversity_barrier=(weights * diversity_barriers).sum(),
            joint_entropy=(weights * entropies).sum(),
            mean_top_probability=(weights * top_probabilities).sum(),
            step_objectives=step_objectives,
            step_set_costs=set_costs,
            step_validity_costs=validity_costs,
            step_diversity_barriers=diversity_barriers,
            step_joint_entropies=entropies,
            step_kls=step_kls,
            deep_supervision_weights=weights,
            trajectories=trajectories,
            joint_probabilities=torch.stack(step_joint_probabilities),
            behavior_costs=behavior_costs,
            support_costs=support_costs,
        )


__all__ = [
    "DiagonalGaussian",
    "GRAMExpectedDiscreteLoss",
    "GRAMFactorizedCausalK4",
    "GRAMPublicCoverageLoss",
    "GRAMRuleTrajectories",
]
