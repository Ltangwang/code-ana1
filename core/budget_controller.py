"""Budget management and tracking for cloud API usage (in-memory only, no database)."""

import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List

from shared.schemas import BudgetStatus
import structlog

logger = structlog.get_logger(__name__)


class BudgetController:
    """Tracks and manages API budget usage."""
    
    def __init__(
        self,
        total_budget: float,
        daily_budget: Optional[float] = None
    ):
        """Initialize budget controller (in-memory only).
        
        Args:
            total_budget: Total budget in USD
            daily_budget: Daily budget limit (optional)
        """
        self.total_budget = total_budget
        self.daily_budget = daily_budget
        
        # In-memory state
        self._current_status = BudgetStatus(
            total_budget=total_budget,
            daily_budget=daily_budget
        )
        
        # Lock for thread-safe updates
        self._lock = asyncio.Lock()
        
        # Alert thresholds
        self.alert_threshold = 0.2  # Alert at 20% remaining
        self._alert_triggered = False
    
    async def initialize(self):
        """Initialize in-memory budget controller (no database)."""
        # Reset to clean state
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
        """Record an API expense (in-memory only).
        
        Args:
            cost: Cost in USD
            provider: Cloud provider name
            model: Model name
            tokens_used: Number of tokens consumed
            operation_type: Type of operation
            details: Additional details (JSON string)
        
        Returns:
            Updated budget status
        """
        async with self._lock:
            # Update in-memory status only
            self._current_status.add_expense(cost)
            
            # Check for alerts
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
        """Get current budget status.
        
        Returns:
            Current BudgetStatus
        """
        return self._current_status
    
    async def can_afford(self, estimated_cost: float) -> bool:
        """Check if budget can afford an operation.
        
        Args:
            estimated_cost: Estimated cost in USD
        
        Returns:
            True if affordable
        """
        return self._current_status.can_afford(estimated_cost)
    
    async def is_daily_limit_reached(self) -> bool:
        """Check if daily budget limit is reached (simplified in-memory).
        
        Returns:
            True if daily limit exceeded
        """
        if self.daily_budget is None:
            return False
        # Simplified: use total for now (no daily tracking without DB)
        return self._current_status.remaining_percent < 0.1
    
    async def get_usage_history(
        self,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get usage history (simplified in-memory, returns empty for now)."""
        # No persistent history without DB
        return []
    
    async def get_usage_by_provider(self) -> Dict[str, float]:
        """Get total usage grouped by provider (simplified)."""
        # No persistent data
        return {}
    
    async def reset_period(self):
        """Reset budget period (start new billing cycle)."""
        async with self._lock:
            # Reset in-memory status
            self._current_status = BudgetStatus(
                total_budget=self.total_budget,
                daily_budget=self.daily_budget
            )
            self._alert_triggered = False
            logger.info("budget_period_reset")
    
    async def _trigger_alert(self):
        """Trigger low budget alert."""
        # In a real system, this might send notifications
        print(f"⚠️  LOW BUDGET ALERT: Only {self._current_status.remaining_percent:.1%} remaining!")
        print(f"   Remaining: ${self._current_status.remaining_budget:.2f} / ${self._current_status.total_budget:.2f}")
    
    def get_adjusted_threshold(
        self,
        base_threshold: float,
        low_budget_threshold: float
    ) -> float:
        """Get adjusted cloud upload threshold based on budget.
        
        Args:
            base_threshold: Normal confidence threshold
            low_budget_threshold: Threshold when budget is low
        
        Returns:
            Adjusted threshold
        """
        if self._current_status.is_low_budget(self.alert_threshold):
            return low_budget_threshold
        return base_threshold
    
    async def export_report(self, filepath: str):
        """Export budget report to file (simplified)."""
        status = await self.get_status()
        
        report = f"""
Budget Report (In-Memory)
Generated: {datetime.now().isoformat()}

=== Current Status ===
Total Budget: ${status.total_budget:.2f}
Used: ${status.used_budget:.2f}
Remaining: ${status.remaining_budget:.2f} ({status.remaining_percent:.1%})
"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info("budget_report_exported", filepath=filepath)

