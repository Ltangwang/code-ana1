"""Shared data schemas for Edge-Cloud code analysis."""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class CodeLanguage(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    CPP = "cpp"
    C = "c"


class CodeFragment(BaseModel):
    """Represents a code fragment for analysis."""
    
    file_path: str = Field(..., description="Path to the source file")
    start_line: int = Field(..., ge=1, description="Starting line number")
    end_line: int = Field(..., ge=1, description="Ending line number")
    content: str = Field(..., description="Code content")
    language: CodeLanguage = Field(..., description="Programming language")
    function_name: Optional[str] = Field(None, description="Function/method name if applicable")
    context: Optional[str] = Field(None, description="Minimal surrounding context")
    
    @validator('end_line')
    def validate_line_range(cls, v, values):
        """Ensure end_line >= start_line."""
        if 'start_line' in values and v < values['start_line']:
            raise ValueError('end_line must be >= start_line')
        return v
    
    def get_location(self) -> str:
        """Get human-readable location string."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


class ConfidenceScore(BaseModel):
    """Represents confidence in an analysis result."""
    
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    reasoning: str = Field(..., description="Explanation for the confidence score")
    factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual factors contributing to confidence"
    )
    
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if confidence exceeds threshold."""
        return self.score >= threshold
    
    def is_low_confidence(self, threshold: float = 0.4) -> bool:
        """Check if confidence is below threshold."""
        return self.score < threshold


class IssueType(str, Enum):
    """Types of code issues."""
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CODE_QUALITY = "code_quality"
    LOGIC_ERROR = "logic_error"


class Severity(str, Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AnalysisDraft(BaseModel):
    """Local analysis result from Edge model."""
    
    fragment: CodeFragment = Field(..., description="Analyzed code fragment")
    issue_type: IssueType = Field(..., description="Type of issue detected")
    severity: Severity = Field(..., description="Issue severity")
    description: str = Field(..., description="Issue description")
    suggested_fixes: List[str] = Field(
        default_factory=list,
        description="N candidate fixes (usually 3)"
    )
    confidence: ConfidenceScore = Field(..., description="Confidence in this analysis")
    detected_at: datetime = Field(
        default_factory=datetime.now,
        description="When analysis was performed"
    )
    model_name: str = Field(..., description="Local model used")
    
    class Config:
        """Pydantic configuration."""
        protected_namespaces = ()  # 允许使用 model_name 字段
    
    def needs_cloud_verification(self, threshold: float = 0.6) -> bool:
        """Determine if this draft needs cloud verification."""
        return self.confidence.score < threshold


class VerificationResult(BaseModel):
    """Cloud verification result."""
    
    draft_id: str = Field(..., description="ID of the original draft")
    verified: bool = Field(..., description="Whether the issue was confirmed")
    refined_description: Optional[str] = Field(None, description="Improved description")
    best_fix_index: Optional[int] = Field(
        None,
        description="Index of best fix from candidates (0-based)"
    )
    alternative_fix: Optional[str] = Field(
        None,
        description="Alternative fix if none of candidates are good"
    )
    confidence_boost: float = Field(
        0.0,
        description="How much confidence increased after verification"
    )
    cloud_model: str = Field(..., description="Cloud model used")
    tokens_used: int = Field(0, description="Tokens consumed")
    latency_ms: float = Field(..., description="Cloud API latency")
    verified_at: datetime = Field(default_factory=datetime.now)


class AnalysisResult(BaseModel):
    """Final analysis result combining draft and verification."""
    
    draft: AnalysisDraft
    verification: Optional[VerificationResult] = None
    final_confidence: float = Field(..., ge=0.0, le=1.0)
    final_description: str
    final_fix: Optional[str] = None
    
    @property
    def was_verified(self) -> bool:
        """Check if this result went through cloud verification."""
        return self.verification is not None
    
    @property
    def location(self) -> str:
        """Get location string."""
        return self.draft.fragment.get_location()


class BudgetStatus(BaseModel):
    """Tracks API budget usage."""
    
    total_budget: float = Field(..., gt=0, description="Total budget in USD")
    used_budget: float = Field(0.0, ge=0, description="Budget used so far")
    daily_budget: Optional[float] = Field(None, description="Daily budget limit")
    period_start: datetime = Field(default_factory=datetime.now)
    period_end: Optional[datetime] = None
    
    @property
    def remaining_budget(self) -> float:
        """Calculate remaining budget."""
        return max(0, self.total_budget - self.used_budget)
    
    @property
    def remaining_percent(self) -> float:
        """Calculate remaining budget as percentage."""
        return self.remaining_budget / self.total_budget if self.total_budget > 0 else 0.0
    
    def is_low_budget(self, threshold: float = 0.2) -> bool:
        """Check if budget is running low."""
        return self.remaining_percent < threshold
    
    def can_afford(self, estimated_cost: float) -> bool:
        """Check if budget can afford an operation."""
        return self.remaining_budget >= estimated_cost
    
    def add_expense(self, cost: float) -> None:
        """Add an expense to the budget."""
        self.used_budget += cost


class UploadDecision(BaseModel):
    """Records decision about uploading to cloud."""
    
    fragment_location: str
    should_upload: bool
    reason: str
    confidence_score: float
    budget_remaining_percent: float
    timestamp: datetime = Field(default_factory=datetime.now)


class AnalysisMetrics(BaseModel):
    """Metrics for analysis session."""
    
    total_fragments: int = 0
    local_only: int = 0
    cloud_verified: int = 0
    high_confidence_local: int = 0
    low_confidence_local: int = 0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    avg_local_latency_ms: float = 0.0
    avg_cloud_latency_ms: float = 0.0
    session_start: datetime = Field(default_factory=datetime.now)
    session_end: Optional[datetime] = None
    
    def calculate_cloud_ratio(self) -> float:
        """Calculate ratio of fragments sent to cloud."""
        if self.total_fragments == 0:
            return 0.0
        return self.cloud_verified / self.total_fragments
    
    def calculate_cost_per_fragment(self) -> float:
        """Calculate average cost per fragment."""
        if self.total_fragments == 0:
            return 0.0
        return self.total_cost / self.total_fragments

