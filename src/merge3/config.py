from dataclasses import dataclass, field
from typing import Tuple

from src.base.base_config import BaseConfig


@dataclass
class Merge3Config(BaseConfig):
    """Configuration for the Merge3-inspired NSGA-style search."""

    nsga_generations: int = 10
    nsga_population: int = 8
    genotype_dimension: int = 2
    variable_bounds: Tuple[float, float] = (0.0, 1.0)
    parent_sample_size: int = 2
    save_intermediate: bool = True

    def validate(self):
        super().validate()
        if self.nsga_generations < 1:
            raise ValueError("nsga_generations must be positive")
        if self.nsga_population < 1:
            raise ValueError("nsga_population must be positive")
        if self.genotype_dimension < 1:
            raise ValueError("genotype_dimension must be positive")
        if self.parent_sample_size < 2:
            raise ValueError("parent_sample_size must be at least 2")
        lo, hi = self.variable_bounds
        if not lo < hi:
            raise ValueError("variable_bounds must satisfy low < high")
