from dataclasses import dataclass, field

from src.base.base_config import BaseConfig


@dataclass
class EnsembleBaselineConfig(BaseConfig):
    """Config for the simple ensemble-only baseline."""

    pairwise_merge_before_ensemble: bool = False
    pair_population: int = 10

    # align with BaseMethod expectations for workspace naming/logging
    N: int = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.N = self.pair_population if self.pairwise_merge_before_ensemble else len(self.pools)

    def validate(self):
        super().validate()
        if len(self.pools) < 2:
            raise ValueError("Ensemble baseline requires at least two pool adapters.")
        if self.pair_population < 1:
            raise ValueError("pair_population must be positive.")
