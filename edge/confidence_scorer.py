"""Confidence scoring for local analysis results."""

from typing import Dict, List, Optional
from dataclasses import dataclass

from shared.schemas import AnalysisDraft, ConfidenceScore


@dataclass
class HistoricalAccuracy:
    """Track historical accuracy for confidence calibration."""
    
    total_predictions: int = 0
    correct_predictions: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    @property
    def accuracy(self) -> float:
        """Calculate overall accuracy."""
        if self.total_predictions == 0:
            return 0.5  # Default
        return self.correct_predictions / self.total_predictions
    
    @property
    def precision(self) -> float:
        """Calculate precision."""
        total_positive = self.correct_predictions + self.false_positives
        if total_positive == 0:
            return 0.5
        return self.correct_predictions / total_positive


class ConfidenceScorer:
    """Enhanced confidence scoring with calibration."""
    
    def __init__(self):
        """Initialize confidence scorer."""
        self.history: Dict[str, HistoricalAccuracy] = {}
        self.issue_type_accuracy: Dict[str, HistoricalAccuracy] = {}
    
    def calibrate_confidence(
        self,
        draft: AnalysisDraft,
        historical_data: Optional[HistoricalAccuracy] = None
    ) -> ConfidenceScore:
        """Calibrate confidence score based on historical accuracy.
        
        Args:
            draft: Analysis draft with initial confidence
            historical_data: Historical accuracy data for calibration
        
        Returns:
            Calibrated confidence score
        """
        base_score = draft.confidence.score
        factors = dict(draft.confidence.factors)
        
        # Apply historical calibration
        if historical_data:
            calibration_factor = historical_data.accuracy
            factors['historical_accuracy'] = calibration_factor
            base_score *= (0.7 + 0.3 * calibration_factor)
        
        # Apply issue-type specific calibration
        issue_type = draft.issue_type.value
        if issue_type in self.issue_type_accuracy:
            type_accuracy = self.issue_type_accuracy[issue_type].accuracy
            factors['issue_type_accuracy'] = type_accuracy
            base_score *= (0.8 + 0.2 * type_accuracy)
        
        # Severity-based adjustment (higher severity needs higher confidence)
        severity_penalties = {
            'critical': 0.9,  # Penalize slightly (need more evidence)
            'high': 0.95,
            'medium': 1.0,
            'low': 1.05,     # Boost slightly (less critical)
            'info': 1.1
        }
        severity_factor = severity_penalties.get(draft.severity.value, 1.0)
        factors['severity_adjustment'] = severity_factor
        base_score *= severity_factor
        
        # Clamp to valid range
        final_score = max(0.0, min(1.0, base_score))
        
        # Update reasoning
        reasoning = draft.confidence.reasoning
        if historical_data and historical_data.accuracy < 0.6:
            reasoning += f" (Note: Historical accuracy {historical_data.accuracy:.1%} suggests caution)"
        
        return ConfidenceScore(
            score=final_score,
            reasoning=reasoning,
            factors=factors
        )
    
    def record_feedback(
        self,
        draft: AnalysisDraft,
        was_correct: bool
    ):
        """Record feedback for future calibration.
        
        Args:
            draft: Original analysis draft
            was_correct: Whether the analysis was correct
        """
        # Update overall history
        model_name = draft.model_name
        if model_name not in self.history:
            self.history[model_name] = HistoricalAccuracy()
        
        hist = self.history[model_name]
        hist.total_predictions += 1
        if was_correct:
            hist.correct_predictions += 1
        else:
            if draft.confidence.score > 0.5:
                hist.false_positives += 1
            else:
                hist.false_negatives += 1
        
        # Update issue-type specific history
        issue_type = draft.issue_type.value
        if issue_type not in self.issue_type_accuracy:
            self.issue_type_accuracy[issue_type] = HistoricalAccuracy()
        
        type_hist = self.issue_type_accuracy[issue_type]
        type_hist.total_predictions += 1
        if was_correct:
            type_hist.correct_predictions += 1
    
    def get_model_stats(self, model_name: str) -> Optional[HistoricalAccuracy]:
        """Get historical stats for a model."""
        return self.history.get(model_name)
    
    def get_issue_type_stats(self, issue_type: str) -> Optional[HistoricalAccuracy]:
        """Get historical stats for an issue type."""
        return self.issue_type_accuracy.get(issue_type)
    
    def should_trust(
        self,
        draft: AnalysisDraft,
        threshold: float = 0.7
    ) -> bool:
        """Determine if we should trust this analysis without verification.
        
        Args:
            draft: Analysis draft
            threshold: Confidence threshold
        
        Returns:
            True if confidence is high enough to trust
        """
        calibrated = self.calibrate_confidence(draft)
        return calibrated.score >= threshold
    
    def get_uncertainty_factors(self, draft: AnalysisDraft) -> List[str]:
        """Identify factors contributing to uncertainty.
        
        Args:
            draft: Analysis draft
        
        Returns:
            List of uncertainty factors
        """
        factors = []
        
        if draft.confidence.score < 0.4:
            factors.append("Low base confidence from model")
        
        if not draft.suggested_fixes:
            factors.append("No fix suggestions provided")
        
        if len(draft.description) < 20:
            factors.append("Minimal description provided")
        
        if draft.severity == 'critical' and draft.confidence.score < 0.8:
            factors.append("Critical severity but moderate confidence")
        
        code_length = len(draft.fragment.content.split('\n'))
        if code_length > 50:
            factors.append("Long code fragment increases complexity")
        
        return factors

