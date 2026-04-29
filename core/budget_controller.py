"""In-memory API budget (no persistence)."""

import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List

from shared.schemas import BudgetStatus
import structlog

logger = structlog.get_logger(__name__)


class BudgetController:
    """Running totals + optional low-budget alert."""
    
    def __init__(
        self,
        total_budget: float,
        daily_budget: Optional[float] = None
    ):
        """total_budget USD; daily_budget optional."""
        self.total_budget = total_budget
        self.daily_budget = daily_budget

        self._current_status = BudgetStatus(
            total_budget=total_budget,
            daily_budget=daily_budget
        )

        self._lock = asyncio.Lock()

        self.alert_threshold = 0.2
        self._alert_triggered = False

    async def initialize(self):
        """Reset spend counters (still in-memory only)."""
        self._current_status = BudgetStatus(
            total_budget=self.total_budget,
            daily_budget=self.daily_budget
        )
        self._alert_triggered = False
        logger.info("budget_controller_initialized", total_budget=self.total_budget)
    
    async def record_expense(
        self,
        cost: float,
        provider: str,
        model: str,
        tokens_used: int = 0,
        operation_type: str = "verification",
        details: Optional[str] = None
    ) -> BudgetStatus:
        """Accumulate cost; fire one alert when remaining % drops below ``alert_threshold``."""
        async with self._lock:
            self._current_status.add_expense(cost)

            if (not self._alert_triggered and
                self._current_status.is_low_budget(self.alert_threshold)):
                self._alert_triggered = True
                await self._trigger_alert()

            logger.info("budget_expense_recorded",
                       cost=cost,
                       provider=provider,
                       model=model,
                       remaining_percent=self._current_status.remaining_percent)

            return self._current_status
    
    async def get_status(self) -> BudgetStatus:
        return self._current_status
    
    async def can_afford(self, estimated_cost: float) -> bool:
        return self._current_status.can_afford(estimated_cost)

    async def is_daily_limit_reached(self) -> bool:
        """No real daily ledger; heuristic on ``remaining_percent`` when ``daily_budget`` set."""
        if self.daily_budget is None:
            return False
        return self._current_status.remaining_percent < 0.1

    async def get_usage_history(
        self,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Stub: no history without storage."""
        return []

    async def get_usage_by_provider(self) -> Dict[str, float]:
        return {}

    async def reset_period(self):
        async with self._lock:
            self._current_status = BudgetStatus(
                total_budget=self.total_budget,
                daily_budget=self.daily_budget
            )
            self._alert_triggered = False
            logger.info("budget_period_reset")

    async def _trigger_alert(self):
        print(
            f"LOW BUDGET: {self._current_status.remaining_percent:.1%} left "
            f"(${self._current_status.remaining_budget:.2f} / ${self._current_status.total_budget:.2f})"
        )

    def get_adjusted_threshold(
        self,
        base_threshold: float,
        low_budget_threshold: float
    ) -> float:
        if self._current_status.is_low_budget(self.alert_threshold):
            return low_budget_threshold
        return base_threshold

    async def export_report(self, filepath: str):
        status = await self.get_status()
        
        report = f"""
Budget (in-memory)
{datetime.now().isoformat()}

Total: ${status.total_budget:.2f}
Used: ${status.used_budget:.2f}
Left: ${status.remaining_budget:.2f} ({status.remaining_percent:.1%})
"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info("budget_report_exported", filepath=filepath)

