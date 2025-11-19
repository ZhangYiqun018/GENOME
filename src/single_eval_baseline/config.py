from dataclasses import dataclass, field

from src.base.base_config import BaseConfig


@dataclass
class SingleEvalBaselineConfig(BaseConfig):
    """Config for evaluating a single LoRA adapter without search/ensemble."""

    # align with BaseMethod expectations
    N: int = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        # only one adapter is evaluated
        self.N = 1

    def validate(self):
        super().validate()
        if len(self.pools) != 1:
            raise ValueError("SingleEvalBaseline requires exactly one LoRA path in pools.")
