"""Tests for strategy manager."""

import pytest
from core.strategy_manager import StrategyManager
from shared.schemas import (
    AnalysisDraft,
    CodeFragment,
    ConfidenceScore,
    BudgetStatus,
    IssueType,
    Severity,
    CodeLanguage
)


@pytest.fixture
def strategy_manager():
    """Create strategy manager instance."""
    config = {
        'base_cloud_threshold': 0.6,
        'low_budget_threshold': 0.3,
        'always_upload_severity': ['critical']
    }
    return StrategyManager(config)


@pytest.fixture
def sample_fragment():
    """Create sample code fragment."""
    return CodeFragment(
        file_path="test.py",
        start_line=10,
        end_line=20,
        content="def test(): pass",
        language=CodeLanguage.PYTHON
    )


@pytest.fixture
def good_budget():
    """Create budget with good status."""
    return BudgetStatus(total_budget=100.0, used_budget=20.0)


@pytest.fixture
def low_budget():
    """Create budget with low status."""
    return BudgetStatus(total_budget=100.0, used_budget=85.0)


def test_should_upload_low_confidence(strategy_manager, sample_fragment, good_budget):
    """Test upload decision for low confidence."""
    draft = AnalysisDraft(
        fragment=sample_fragment,
        issue_type=IssueType.BUG,
        severity=Severity.MEDIUM,
        description="Potential bug",
        suggested_fixes=["Fix 1"],
        confidence=ConfidenceScore(score=0.3, reasoning="Low confidence"),
        model_name="test-model"
    )
    
    decision = strategy_manager.should_upload(draft, good_budget)
    assert decision.should_upload is True
    assert "confidence" in decision.reason.lower()


def test_should_not_upload_high_confidence(strategy_manager, sample_fragment, good_budget):
    """Test no upload for high confidence."""
    draft = AnalysisDraft(
        fragment=sample_fragment,
        issue_type=IssueType.BUG,
        severity=Severity.MEDIUM,
        description="Clear bug",
        suggested_fixes=["Fix 1"],
        confidence=ConfidenceScore(score=0.9, reasoning="High confidence"),
        model_name="test-model"
    )
    
    decision = strategy_manager.should_upload(draft, good_budget)
    assert decision.should_upload is False
    assert "high confidence" in decision.reason.lower()


def test_always_upload_critical(strategy_manager, sample_fragment, good_budget):
    """Test critical issues always uploaded."""
    draft = AnalysisDraft(
        fragment=sample_fragment,
        issue_type=IssueType.SECURITY,
        severity=Severity.CRITICAL,
        description="Critical security issue",
        suggested_fixes=["Fix 1"],
        confidence=ConfidenceScore(score=0.9, reasoning="High confidence"),
        model_name="test-model"
    )
    
    decision = strategy_manager.should_upload(draft, good_budget)
    assert decision.should_upload is True
    assert "critical" in decision.reason.lower()


def test_low_budget_blocks_upload(strategy_manager, sample_fragment, low_budget):
    """Test low budget blocks non-critical uploads."""
    draft = AnalysisDraft(
        fragment=sample_fragment,
        issue_type=IssueType.BUG,
        severity=Severity.MEDIUM,
        description="Bug",
        suggested_fixes=["Fix 1"],
        confidence=ConfidenceScore(score=0.3, reasoning="Low confidence"),
        model_name="test-model"
    )
    
    decision = strategy_manager.should_upload(draft, low_budget)
    # With very low budget, should not upload medium severity
    assert decision.should_upload is False or "budget" in decision.reason.lower()


def test_filter_drafts(strategy_manager, sample_fragment, good_budget):
    """Test filtering multiple drafts."""
    drafts = [
        AnalysisDraft(
            fragment=sample_fragment,
            issue_type=IssueType.BUG,
            severity=Severity.HIGH,
            description=f"Bug {i}",
            suggested_fixes=["Fix"],
            confidence=ConfidenceScore(score=0.3 + i * 0.1, reasoning="Test"),
            model_name="test-model"
        )
        for i in range(5)
    ]
    
    to_upload, decisions = strategy_manager.filter_drafts(drafts, good_budget)
    
    assert len(decisions) == 5
    assert len(to_upload) <= len(drafts)


def test_prioritize_uploads(strategy_manager, sample_fragment):
    """Test upload prioritization."""
    drafts = [
        AnalysisDraft(
            fragment=sample_fragment,
            issue_type=IssueType.BUG,
            severity=Severity.CRITICAL,
            description="Critical",
            suggested_fixes=["Fix"],
            confidence=ConfidenceScore(score=0.3, reasoning="Low"),
            model_name="test"
        ),
        AnalysisDraft(
            fragment=sample_fragment,
            issue_type=IssueType.BUG,
            severity=Severity.LOW,
            description="Low",
            suggested_fixes=["Fix"],
            confidence=ConfidenceScore(score=0.2, reasoning="Low"),
            model_name="test"
        ),
        AnalysisDraft(
            fragment=sample_fragment,
            issue_type=IssueType.BUG,
            severity=Severity.HIGH,
            description="High",
            suggested_fixes=["Fix"],
            confidence=ConfidenceScore(score=0.4, reasoning="Medium"),
            model_name="test"
        )
    ]
    
    prioritized = strategy_manager._prioritize_uploads(drafts, max_count=2)
    
    assert len(prioritized) == 2
    # Critical should be first
    assert prioritized[0].severity == Severity.CRITICAL


def test_should_report(strategy_manager, sample_fragment):
    """Test reportability filtering."""
    # High confidence - should report
    draft_high = AnalysisDraft(
        fragment=sample_fragment,
        issue_type=IssueType.BUG,
        severity=Severity.MEDIUM,
        description="Bug",
        suggested_fixes=["Fix"],
        confidence=ConfidenceScore(score=0.8, reasoning="High"),
        model_name="test"
    )
    assert strategy_manager.should_report(draft_high) is True
    
    # Very low confidence - should not report
    draft_low = AnalysisDraft(
        fragment=sample_fragment,
        issue_type=IssueType.BUG,
        severity=Severity.LOW,
        description="Maybe bug",
        suggested_fixes=["Fix"],
        confidence=ConfidenceScore(score=0.1, reasoning="Very low"),
        model_name="test"
    )
    assert strategy_manager.should_report(draft_low) is False

