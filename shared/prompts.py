"""Prompt templates for local and cloud LLMs."""

from typing import List, Dict, Any
from .schemas import CodeFragment, AnalysisDraft, CodeLanguage


class PromptTemplates:
    """Container for all prompt templates."""
    
    # Local small model prompt (optimized for 1.3B-7B models)
    LOCAL_BUG_DETECTION_PROMPT = """You are a code analyzer. Analyze the following {language} code for potential bugs.

Code Location: {location}
Function: {function_name}

```{language}
{code}
```

Task: Identify potential bugs, logic errors, or issues that could cause runtime failures.

Respond in this exact JSON format:
{{
  "has_issue": true/false,
  "issue_type": "bug|logic_error|security|other",
  "severity": "critical|high|medium|low",
  "description": "Brief description of the issue",
  "suggested_fixes": ["fix1", "fix2", "fix3"],
  "confidence": 0.0-1.0,
  "reasoning": "Why this confidence score"
}}

Be concise. Focus on real bugs, not style issues."""

    # Cloud verification prompt (for high-end models)
    CLOUD_VERIFICATION_PROMPT = """You are an expert code reviewer verifying a potential bug found by an automated analyzer.

Original Code Fragment:
```{language}
{code}
```

Detected Issue:
- Type: {issue_type}
- Severity: {severity}
- Description: {description}

Suggested Fixes (from local analyzer):
{fixes_list}

Your Task:
1. Verify if this is a real bug or false positive
2. If real, select the best fix (by index) or propose a better one
3. Provide refined analysis

Respond in JSON format:
{{
  "is_real_bug": true/false,
  "verification_confidence": 0.0-1.0,
  "best_fix_index": 0-2 or null,
  "alternative_fix": "better fix if needed" or null,
  "refined_description": "more accurate description",
  "additional_context": "any important details"
}}"""

    # Cloud refinement prompt (when local model has low confidence)
    CLOUD_REFINEMENT_PROMPT = """You are an expert {language} code analyzer. Perform deep analysis on this code fragment.

Code:
```{language}
{code}
```

Context: {context}

The local analyzer detected a potential {issue_type} but with low confidence ({confidence}).
Its analysis: {description}

Your task:
1. Perform thorough analysis
2. Determine if there's really an issue
3. If yes, provide detailed explanation and best fix
4. If no, explain why it's a false positive

Respond in JSON format:
{{
  "analysis_result": "confirmed_bug|false_positive|unclear",
  "confidence": 0.0-1.0,
  "detailed_description": "comprehensive analysis",
  "recommended_fix": "single best fix" or null,
  "severity": "critical|high|medium|low|info",
  "explanation": "detailed reasoning"
}}"""

    # Selection prompt (when multiple candidates exist)
    CLOUD_SELECTION_PROMPT = """You are selecting the best fix for a confirmed bug.

Bug: {description}

Code:
```{language}
{code}
```

Candidate Fixes:
{candidates}

Evaluate each candidate on:
- Correctness
- Safety
- Performance
- Code quality

Respond in JSON:
{{
  "selected_index": 0-N,
  "rationale": "why this fix is best",
  "improvements": "any suggested modifications to selected fix"
}}"""

    @classmethod
    def format_local_prompt(
        cls,
        fragment: CodeFragment,
        language: str = None
    ) -> str:
        """Format local bug detection prompt."""
        lang = language or fragment.language.value
        return cls.LOCAL_BUG_DETECTION_PROMPT.format(
            language=lang,
            location=fragment.get_location(),
            function_name=fragment.function_name or "N/A",
            code=fragment.content
        )
    
    @classmethod
    def format_verification_prompt(
        cls,
        draft: AnalysisDraft
    ) -> str:
        """Format cloud verification prompt."""
        fixes_list = "\n".join([
            f"{i}. {fix}"
            for i, fix in enumerate(draft.suggested_fixes)
        ])
        
        return cls.CLOUD_VERIFICATION_PROMPT.format(
            language=draft.fragment.language.value,
            code=draft.fragment.content,
            issue_type=draft.issue_type.value,
            severity=draft.severity.value,
            description=draft.description,
            fixes_list=fixes_list
        )
    
    @classmethod
    def format_refinement_prompt(
        cls,
        draft: AnalysisDraft
    ) -> str:
        """Format cloud refinement prompt for low-confidence cases."""
        return cls.CLOUD_REFINEMENT_PROMPT.format(
            language=draft.fragment.language.value,
            code=draft.fragment.content,
            context=draft.fragment.context or "No additional context",
            issue_type=draft.issue_type.value,
            confidence=draft.confidence.score,
            description=draft.description
        )
    
    @classmethod
    def format_selection_prompt(
        cls,
        fragment: CodeFragment,
        description: str,
        candidates: List[str]
    ) -> str:
        """Format candidate selection prompt."""
        candidates_text = "\n".join([
            f"Candidate {i}: {fix}"
            for i, fix in enumerate(candidates)
        ])
        
        return cls.CLOUD_SELECTION_PROMPT.format(
            description=description,
            language=fragment.language.value,
            code=fragment.content,
            candidates=candidates_text
        )
    
    @staticmethod
    def get_system_prompt(role: str = "analyzer") -> str:
        """Get system prompt for different roles."""
        prompts = {
            "analyzer": "You are an expert code analyzer specializing in bug detection and code quality.",
            "verifier": "You are a senior code reviewer verifying automated analysis results.",
            "fixer": "You are an expert programmer focused on providing safe, correct code fixes."
        }
        return prompts.get(role, prompts["analyzer"])


# Pre-defined confidence scoring guidelines
CONFIDENCE_GUIDELINES = {
    "high": {
        "threshold": 0.7,
        "description": "Clear bug with obvious fix",
        "examples": [
            "Division by zero without check",
            "Null pointer dereference",
            "Array index out of bounds"
        ]
    },
    "medium": {
        "threshold": 0.4,
        "description": "Likely bug but context needed",
        "examples": [
            "Potential memory leak",
            "Race condition possibility",
            "Type confusion"
        ]
    },
    "low": {
        "threshold": 0.0,
        "description": "Uncertain or complex issue",
        "examples": [
            "Complex algorithm correctness",
            "Performance optimization needed",
            "Design pattern usage"
        ]
    }
}


# Cost estimation constants (approximate, in USD)
COST_ESTIMATES = {
    "gpt-4-turbo": {
        "input_per_1k": 0.01,
        "output_per_1k": 0.03
    },
    "gpt-3.5-turbo": {
        "input_per_1k": 0.0005,
        "output_per_1k": 0.0015
    },
    "claude-3-opus": {
        "input_per_1k": 0.015,
        "output_per_1k": 0.075
    },
    "claude-3-sonnet": {
        "input_per_1k": 0.003,
        "output_per_1k": 0.015
    },
    "claude-3-haiku": {
        "input_per_1k": 0.00025,
        "output_per_1k": 0.00125
    }
}


def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 chars for English)."""
    return len(text) // 4


def estimate_cost(
    prompt: str,
    expected_response_tokens: int,
    model: str = "gpt-4-turbo"
) -> float:
    """Estimate cost for a cloud API call."""
    if model not in COST_ESTIMATES:
        model = "gpt-4-turbo"  # Default to most expensive for safety
    
    input_tokens = estimate_tokens(prompt)
    input_cost = (input_tokens / 1000) * COST_ESTIMATES[model]["input_per_1k"]
    output_cost = (expected_response_tokens / 1000) * COST_ESTIMATES[model]["output_per_1k"]
    
    return input_cost + output_cost

