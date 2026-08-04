"""OpenAI-compatible async client for cloud LLM calls."""

import json
import time
from typing import Dict, Any, Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from shared.schemas import (
    AnalysisDraft,
    VerificationResult,
    CodeFragment
)
from shared.prompts import PromptTemplates, estimate_cost


class CloudClient:
    """OpenAI-compatible chat API wrapper."""

    def __init__(self, config: Dict[str, Any], provider: str = "openai"):
        self.provider = provider
        self.config = config

        provider_config = config.get(provider, {})
        
        api_key = provider_config.get('api_key', '')
        base_url = provider_config.get('base_url')
        self.model = provider_config.get('model', 'gpt-4-turbo-preview')
        self.timeout = provider_config.get('timeout', 60)
        self.max_retries = provider_config.get('max_retries', 3)
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None,
            timeout=self.timeout,
            max_retries=self.max_retries
        )

        self._call_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def verify(
        self,
        draft: AnalysisDraft,
        mode: str = "verification"
    ) -> VerificationResult:
        """Run verification or refinement prompt; increment usage counters."""
        start_time = time.time()

        if mode == "refinement":
            prompt = PromptTemplates.format_refinement_prompt(draft)
        else:
            prompt = PromptTemplates.format_verification_prompt(draft)

        estimated_cost = estimate_cost(prompt, 300, self.model)
        response = await self._call_api(prompt)
        latency_ms = (time.time() - start_time) * 1000
        result = self._parse_verification_response(
            response,
            draft,
            latency_ms,
            mode
        )

        self._call_count += 1
        self._total_tokens += result.tokens_used
        self._total_cost += estimated_cost

        return result
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def select_best_fix(
        self,
        fragment: CodeFragment,
        description: str,
        candidates: list[str]
    ) -> tuple[int, str]:
        """Pick best fix index from JSON in the model reply."""
        prompt = PromptTemplates.format_selection_prompt(
            fragment,
            description,
            candidates
        )

        response = await self._call_api(prompt)

        try:
            data = json.loads(response['content'])
            selected_idx = data.get('selected_index', 0)
            rationale = data.get('rationale', 'Selected by cloud model')
            return (selected_idx, rationale)
        except:
            return (0, 'Failed to parse selection')
    
    async def _call_api(
        self,
        prompt: str,
        *,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
        json_response_format: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """chat.completions; optional system prompt and JSON response_format."""
        try:
            sys_content = (
                system_prompt
                if system_prompt is not None
                else PromptTemplates.get_system_prompt("verifier")
            )
            use_json = json_response_format
            if use_json is None:
                m = (self.model or "").lower()
                use_json = "gpt" in m or m.startswith("o1") or m.startswith("o3")
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
            }
            if use_json:
                kwargs["response_format"] = {"type": "json_object"}
            response = await self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            content = choice.message.content
            usage = response.usage
            total_tokens = usage.total_tokens if usage else 0
            return {
                "content": content,
                "tokens": total_tokens,
                "model": response.model,
            }
        except Exception as e:
            raise RuntimeError(f"Cloud API error ({self.provider}): {e}")
    
    def _parse_verification_response(
        self,
        response: Dict[str, Any],
        draft: AnalysisDraft,
        latency_ms: float,
        mode: str
    ) -> VerificationResult:
        content = response['content']
        tokens = response['tokens']
        
        try:
            data = json.loads(content)
            
            if mode == "refinement":
                analysis_result = data.get('analysis_result', 'unclear')
                verified = analysis_result == 'confirmed_bug'
                
                refined_description = data.get('detailed_description', draft.description)
                alternative_fix = data.get('recommended_fix')
                cloud_confidence = data.get('confidence', 0.5)
                
                confidence_boost = max(0, cloud_confidence - draft.confidence.score)
                
                return VerificationResult(
                    draft_id=f"{draft.fragment.file_path}:{draft.fragment.start_line}",
                    verified=verified,
                    refined_description=refined_description,
                    best_fix_index=None,
                    alternative_fix=alternative_fix,
                    confidence_boost=confidence_boost,
                    cloud_model=self.model,
                    tokens_used=tokens,
                    latency_ms=latency_ms
                )
            
            else:
                is_real = data.get('is_real_bug', True)
                refined_desc = data.get('refined_description', draft.description)
                best_idx = data.get('best_fix_index')
                alt_fix = data.get('alternative_fix')
                cloud_conf = data.get('verification_confidence', 0.7)
                
                confidence_boost = cloud_conf - draft.confidence.score
                
                return VerificationResult(
                    draft_id=f"{draft.fragment.file_path}:{draft.fragment.start_line}",
                    verified=is_real,
                    refined_description=refined_desc,
                    best_fix_index=best_idx,
                    alternative_fix=alt_fix,
                    confidence_boost=confidence_boost,
                    cloud_model=self.model,
                    tokens_used=tokens,
                    latency_ms=latency_ms
                )
        
        except json.JSONDecodeError:
            return VerificationResult(
                draft_id=f"{draft.fragment.file_path}:{draft.fragment.start_line}",
                verified=False,
                refined_description="Cloud verification failed to parse",
                best_fix_index=None,
                alternative_fix=None,
                confidence_boost=0.0,
                cloud_model=self.model,
                tokens_used=tokens,
                latency_ms=latency_ms
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """calls / tokens / spend snapshot."""
        return {
            'provider': self.provider,
            'model': self.model,
            'call_count': self._call_count,
            'total_tokens': self._total_tokens,
            'total_cost': self._total_cost,
            'avg_tokens_per_call': self._total_tokens / max(1, self._call_count)
        }
    
    async def health_check(self, verbose: bool = False) -> bool:
        """Tiny chat completion to see if credentials/route work."""
        try:
            await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            if verbose:
                print(f"  Error: {type(e).__name__}: {str(e)}")
            return False

