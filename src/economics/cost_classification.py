"""
Phase 3 Module 1: Cost Classification & Taxonomy Engine.
Tags every cargo cost component into one of five categories:
1. Sunk / Committed
2. Variable Upstream
3. Destination-Dependent
4. Avoidable
5. Financing / Working Capital
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class CostCategory(Enum):
    SUNK_COMMITTED = "SUNK_COMMITTED"
    VARIABLE_UPSTREAM = "VARIABLE_UPSTREAM"
    DESTINATION_DEPENDENT = "DESTINATION_DEPENDENT"
    AVOIDABLE = "AVOIDABLE"
    FINANCING = "FINANCING"


@dataclass(frozen=True)
class CostItem:
    """Represents an individual cost component with commercial taxonomy tagging."""
    name: str
    amount_usd: float
    category: CostCategory
    description: str

    @property
    def is_marginal_routing_relevant(self) -> bool:
        """Determines if the cost item varies when deciding between destination routes (NWE vs NEA)."""
        return self.category in (
            CostCategory.VARIABLE_UPSTREAM,
            CostCategory.DESTINATION_DEPENDENT,
            CostCategory.FINANCING,
        )

    @property
    def is_lift_shut_in_relevant(self) -> bool:
        """Determines if the cost item can be avoided by cancelling the cargo lift."""
        return self.category in (
            CostCategory.VARIABLE_UPSTREAM,
            CostCategory.DESTINATION_DEPENDENT,
            CostCategory.AVOIDABLE,
            CostCategory.FINANCING,
        )