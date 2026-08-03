"""Edge module for local code analysis."""

from .ast_analyzer import ASTAnalyzer, Hotspot
from .local_inference import OllamaInference
from .confidence_scorer import ConfidenceScorer

__all__ = [
    "ASTAnalyzer",
    "Hotspot",
    "OllamaInference",
    "ConfidenceScorer",
]

