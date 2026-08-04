"""Ollama HTTP client; ``generate_text`` used on the CSN eval path."""

import json
import asyncio
from typing import List, Optional, Dict, Any, Tuple

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from shared.schemas import (
    CodeFragment,
    AnalysisDraft,
    ConfidenceScore,
    IssueType,
    Severity
)
from shared.prompts import PromptTemplates


class OllamaInference:
    """POST /api/generate against a local Ollama server."""

    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model_name = config.get('model_name', 'codellama:7b')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 512)
        self.timeout = config.get('timeout', 30)
        
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
    
    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
    
    async def __aenter__(self):
        await self.ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def analyze_fragment(
        self,
        fragment: CodeFragment,
        n_fixes: int = 3
    ) -> AnalysisDraft:
        """Legacy bug-report path (not CSN rerank)."""
        prompt = PromptTemplates.format_local_prompt(fragment)
        response = await self._call_ollama(prompt)

        analysis = self._parse_response(response, fragment)

        while len(analysis.suggested_fixes) < n_fixes:
            analysis.suggested_fixes.append(f"Fix option {len(analysis.suggested_fixes) + 1}")

        return analysis
    
    async def analyze_batch(
        self,
        fragments: List[CodeFragment],
        batch_size: int = 5
    ) -> List[AnalysisDraft]:
        results = []

        for i in range(0, len(fragments), batch_size):
            batch = fragments[i:i + batch_size]
            tasks = [self.analyze_fragment(frag) for frag in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, AnalysisDraft):
                    results.append(result)
                elif isinstance(result, Exception):
                    print(f"Error in batch analysis: {result}")

        return results

    async def health_check(self, timeout_sec: float = 5.0) -> Tuple[bool, str]:
        """Probe Ollama HTTP API; return (ok, message)."""
        await self.ensure_session()
        url = f"{self.base_url.rstrip('/')}/api/tags"
        try:
            async with self._session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout_sec),
            ) as response:
                response.raise_for_status()
                return True, "ok"
        except Exception as e:
            return False, str(e)

    async def generate_text(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> str:
        """Single-turn generate; optional system prefix. Returns model text only."""
        parts: List[str] = []
        if system:
            parts.append(f"### System\n{system.strip()}\n\n### User\n")
        parts.append(prompt)
        full_prompt = "".join(parts)
        tok = max_tokens if max_tokens is not None else self.max_tokens
        t = timeout_sec if timeout_sec is not None else float(self.timeout)
        resp = await self._call_ollama_with_timeout(
            full_prompt, t, num_predict=tok
        )
        return (resp.get("response") or "").strip()

    async def _call_ollama_with_timeout(
        self,
        prompt: str,
        timeout_sec: float,
        *,
        num_predict: Optional[int] = None,
    ) -> Dict[str, Any]:
        await self.ensure_session()
        url = f"{self.base_url}/api/generate"
        npred = num_predict if num_predict is not None else self.max_tokens
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": npred,
            },
        }
        try:
            async with self._session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_sec),
            ) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Ollama API error: {e}")
    
    async def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        return await self._call_ollama_with_timeout(prompt, float(self.timeout))
    
    def _parse_response(
        self,
        response: Dict[str, Any],
        fragment: CodeFragment
    ) -> AnalysisDraft:
        text = response.get('response', '')

        try:
            json_match = self._extract_json(text)
            if json_match:
                data = json.loads(json_match)
            else:
                data = {
                    'has_issue': False,
                    'confidence': 0.2,
                    'reasoning': 'Failed to parse model response'
                }

            has_issue = data.get('has_issue', False)

            if not has_issue:
                return AnalysisDraft(
                    fragment=fragment,
                    issue_type=IssueType.BUG,
                    severity=Severity.INFO,
                    description="No significant issues detected",
                    suggested_fixes=[],
                    confidence=ConfidenceScore(
                        score=data.get('confidence', 0.3),
                        reasoning=data.get('reasoning', 'No issues found'),
                        factors={'model_output': data.get('confidence', 0.3)}
                    ),
                    model_name=self.model_name
                )
            
            issue_type_str = data.get('issue_type', 'bug')
            issue_type = self._parse_issue_type(issue_type_str)
            
            severity_str = data.get('severity', 'medium')
            severity = self._parse_severity(severity_str)
            
            description = data.get('description', 'Potential issue detected')
            
            fixes = data.get('suggested_fixes', [])
            if not isinstance(fixes, list):
                fixes = [str(fixes)]
            
            confidence_score = float(data.get('confidence', 0.5))
            reasoning = data.get('reasoning', 'Based on model analysis')
            
            # Calculate confidence with additional factors
            confidence = self._calculate_confidence(
                confidence_score,
                reasoning,
                fragment,
                response
            )

            return AnalysisDraft(
                fragment=fragment,
                issue_type=issue_type,
                severity=severity,
                description=description,
                suggested_fixes=fixes[:5],  # Max 5 fixes
                confidence=confidence,
                model_name=self.model_name
            )
        
        except Exception as e:
            return AnalysisDraft(
                fragment=fragment,
                issue_type=IssueType.BUG,
                severity=Severity.LOW,
                description=f"Analysis incomplete: {str(e)}",
                suggested_fixes=[],
                confidence=ConfidenceScore(
                    score=0.1,
                    reasoning=f"Parse error: {str(e)}",
                    factors={'error': 1.0}
                ),
                model_name=self.model_name
            )
    
    def _extract_json(self, text: str) -> Optional[str]:
        import re

        matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _parse_issue_type(self, type_str: str) -> IssueType:
        type_str = type_str.lower()
        
        if 'security' in type_str:
            return IssueType.SECURITY
        elif 'performance' in type_str:
            return IssueType.PERFORMANCE
        elif 'logic' in type_str:
            return IssueType.LOGIC_ERROR
        elif 'quality' in type_str:
            return IssueType.CODE_QUALITY
        else:
            return IssueType.BUG
    
    def _parse_severity(self, severity_str: str) -> Severity:
        severity_str = severity_str.lower()
        
        if 'critical' in severity_str:
            return Severity.CRITICAL
        elif 'high' in severity_str:
            return Severity.HIGH
        elif 'low' in severity_str:
            return Severity.LOW
        elif 'info' in severity_str:
            return Severity.INFO
        else:
            return Severity.MEDIUM
    
    def _calculate_confidence(
        self,
        base_score: float,
        reasoning: str,
        fragment: CodeFragment,
        response: Dict[str, Any]
    ) -> ConfidenceScore:
        """Heuristic mix of model score, reasoning length, LOC."""
        factors = {
            'model_score': base_score
        }

        reasoning_length = len(reasoning.split())
        factors['reasoning_length'] = min(reasoning_length / 50.0, 1.0) * 0.1

        code_lines = len(fragment.content.split('\n'))
        factors['code_simplicity'] = max(0, 1.0 - code_lines / 100.0) * 0.1

        final_score = base_score * 0.8
        final_score += factors.get('reasoning_length', 0)
        final_score += factors.get('code_simplicity', 0)

        final_score = max(0.0, min(1.0, final_score))
        
        return ConfidenceScore(
            score=final_score,
            reasoning=reasoning,
            factors=factors
        )

