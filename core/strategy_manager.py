"""Strategy manager for deciding when to upload to cloud."""

from typing import List, Dict, Any, Optional
from datetime import datetime

from shared.schemas import (
    AnalysisDraft,
    BudgetStatus,
    UploadDecision,
    Severity
)


class StrategyManager:
    """Manages strategy for cloud verification decisions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy manager.
        
        Args:
            config: Configuration dict with strategy settings
        """
        self.base_threshold = config.get('base_cloud_threshold', 0.6)
        self.low_budget_threshold = config.get('low_budget_threshold', 0.3)
        self.min_reportable_confidence = config.get('min_reportable_confidence', 0.3)
        
        # Always upload certain severities
        self.always_upload_severities = config.get(
            'always_upload_severity',
            ['critical']
        )
        
        # Budget thresholds for adjustments
        self.budget_thresholds = {
            'critical': 0.1,
            'low': 0.2,
            'warning': 0.4
        }
        
        # Track decisions for analytics
        self._decision_history: List[UploadDecision] = []
    
    def should_upload(
        self,
        draft: AnalysisDraft,
        budget_status: BudgetStatus
    ) -> UploadDecision:
        """Decide whether to upload a draft to cloud for verification.
        
        Args:
            draft: Local analysis draft
            budget_status: Current budget status
        
        Returns:
            UploadDecision with reasoning
        """
        confidence = draft.confidence.score
        severity = draft.severity.value
        remaining_budget_pct = budget_status.remaining_percent
        
        # Decision logic
        should_upload = False
        reason = ""
        
        # Rule 1: Always upload critical issues (if budget allows)
        if severity in self.always_upload_severities:
            if budget_status.can_afford(0.1):  # Min estimated cost
                should_upload = True
                reason = f"Critical severity ({severity}) always requires verification"
            else:
                should_upload = False
                reason = "Critical issue but insufficient budget"
        
        # Rule 2: Check confidence threshold
        elif confidence < self._get_effective_threshold(budget_status):
            if budget_status.can_afford(0.1):
                should_upload = True
                reason = f"Low confidence ({confidence:.2f}) below threshold"
            else:
                should_upload = False
                reason = f"Low confidence but insufficient budget"
        
        # Rule 3: High confidence - no need for verification
        else:
            should_upload = False
            reason = f"High confidence ({confidence:.2f}) - local analysis sufficient"
        
        # Rule 4: Budget emergency - only upload critical
        if remaining_budget_pct < self.budget_thresholds['critical']:
            if severity not in ['critical']:
                should_upload = False
                reason = f"Budget critical ({remaining_budget_pct:.1%}) - only uploading critical issues"
        
        # Rule 5: Daily budget exceeded
        if hasattr(budget_status, 'daily_budget') and budget_status.daily_budget:
            # This would need daily usage check from BudgetController
            # For now, simplified
            pass
        
        # Create decision record
        decision = UploadDecision(
            fragment_location=draft.fragment.get_location(),
            should_upload=should_upload,
            reason=reason,
            confidence_score=confidence,
            budget_remaining_percent=remaining_budget_pct
        )
        
        # Track decision
        self._decision_history.append(decision)
        
        return decision
    
    def filter_drafts(
        self,
        drafts: List[AnalysisDraft],
        budget_status: BudgetStatus,
        max_uploads: Optional[int] = None
    ) -> tuple[List[AnalysisDraft], List[UploadDecision]]:
        """Filter drafts to determine which should be uploaded.
        
        Args:
            drafts: List of analysis drafts
            budget_status: Current budget status
            max_uploads: Maximum number to upload (None = no limit)
        
        Returns:
            (drafts_to_upload, all_decisions)
        """
        decisions = []
        to_upload = []
        
        # Make decision for each draft
        for draft in drafts:
            decision = self.should_upload(draft, budget_status)
            decisions.append(decision)
            
            if decision.should_upload:
                to_upload.append(draft)
        
        # Apply max_uploads limit if specified
        if max_uploads and len(to_upload) > max_uploads:
            # Prioritize by severity and confidence
            to_upload = self._prioritize_uploads(to_upload, max_uploads)
        
        return to_upload, decisions
    
    def _prioritize_uploads(
        self,
        drafts: List[AnalysisDraft],
        max_count: int
    ) -> List[AnalysisDraft]:
        """Prioritize which drafts to upload when limited.
        
        Args:
            drafts: Candidate drafts
            max_count: Maximum to select
        
        Returns:
            Top priority drafts
        """
        # Severity priority
        severity_priority = {
            'critical': 5,
            'high': 4,
            'medium': 3,
            'low': 2,
            'info': 1
        }
        
        # Calculate priority score for each draft
        scored = []
        for draft in drafts:
            severity_score = severity_priority.get(draft.severity.value, 0)
            # Lower confidence = higher priority (needs verification more)
            confidence_score = (1.0 - draft.confidence.score) * 3
            
            total_score = severity_score + confidence_score
            scored.append((total_score, draft))
        
        # Sort by priority (highest first)
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [draft for _, draft in scored[:max_count]]
    
    def _get_effective_threshold(self, budget_status: BudgetStatus) -> float:
        """Get effective confidence threshold based on budget.
        
        Args:
            budget_status: Current budget status
        
        Returns:
            Effective threshold
        """
        remaining = budget_status.remaining_percent
        
        if remaining < self.budget_thresholds['critical']:
            # Very low budget - only upload if confidence < 0.2
            return 0.2
        elif remaining < self.budget_thresholds['low']:
            # Low budget - use low threshold
            return self.low_budget_threshold
        elif remaining < self.budget_thresholds['warning']:
            # Warning level - slightly lower threshold
            return self.base_threshold * 0.8
        else:
            # Normal operation
            return self.base_threshold
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """Get statistics about upload decisions.
        
        Returns:
            Dict with decision statistics
        """
        if not self._decision_history:
            return {
                'total_decisions': 0,
                'upload_count': 0,
                'skip_count': 0,
                'upload_rate': 0.0
            }
        
        total = len(self._decision_history)
        uploaded = sum(1 for d in self._decision_history if d.should_upload)
        skipped = total - uploaded
        
        # Reasons breakdown
        reasons = {}
        for decision in self._decision_history:
            reason_key = decision.reason.split('-')[0].strip()
            reasons[reason_key] = reasons.get(reason_key, 0) + 1
        
        return {
            'total_decisions': total,
            'upload_count': uploaded,
            'skip_count': skipped,
            'upload_rate': uploaded / total if total > 0 else 0.0,
            'reasons_breakdown': reasons
        }
    
    def reset_history(self):
        """Clear decision history."""
        self._decision_history.clear()
    
    def should_report(self, draft: AnalysisDraft) -> bool:
        """Determine if a draft should be included in final report.
        
        Args:
            draft: Analysis draft
        
        Returns:
            True if should be reported
        """
        # Filter out very low confidence results
        if draft.confidence.score < self.min_reportable_confidence:
            return False
        
        # Filter out "no issue" results unless high confidence
        if draft.severity == Severity.INFO and draft.confidence.score < 0.6:
            return False
        
        return True
    
    def recommend_action(
        self,
        budget_status: BudgetStatus,
        recent_upload_rate: float
    ) -> str:
        """Recommend strategy adjustments.
        
        Args:
            budget_status: Current budget status
            recent_upload_rate: Recent upload rate (0.0-1.0)
        
        Returns:
            Recommendation message
        """
        remaining = budget_status.remaining_percent
        
        if remaining < self.budget_thresholds['critical']:
            return (
                "⚠️  CRITICAL: Budget critically low. "
                "Consider increasing threshold or adding budget."
            )
        
        elif remaining < self.budget_thresholds['low']:
            return (
                "⚠️  WARNING: Budget running low. "
                "Upload threshold automatically increased to conserve budget."
            )
        
        elif recent_upload_rate > 0.8:
            return (
                "ℹ️  INFO: High upload rate detected. "
                "Consider improving local model or adjusting confidence threshold."
            )
        
        elif recent_upload_rate < 0.1:
            return (
                "✓ SUCCESS: Low upload rate - local model performing well. "
                "Could consider lowering threshold slightly for better coverage."
            )
        
        else:
            return "✓ Strategy operating normally."
    
    def estimate_cost_for_batch(
        self,
        drafts: List[AnalysisDraft],
        budget_status: BudgetStatus,
        cost_per_call: float = 0.01
    ) -> Dict[str, Any]:
        """Estimate cost for processing a batch of drafts.
        
        Args:
            drafts: Drafts to process
            budget_status: Current budget
            cost_per_call: Estimated cost per cloud call
        
        Returns:
            Cost estimate and recommendations
        """
        to_upload, decisions = self.filter_drafts(drafts, budget_status)
        
        estimated_cost = len(to_upload) * cost_per_call
        can_afford = budget_status.can_afford(estimated_cost)
        
        return {
            'total_drafts': len(drafts),
            'planned_uploads': len(to_upload),
            'estimated_cost': estimated_cost,
            'can_afford': can_afford,
            'remaining_after': budget_status.remaining_budget - estimated_cost,
            'upload_rate': len(to_upload) / len(drafts) if drafts else 0.0
        }

