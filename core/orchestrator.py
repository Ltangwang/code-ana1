"""Main orchestrator for Edge-Cloud code analysis."""

import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
import structlog

from edge.ast_analyzer import ASTAnalyzer, Hotspot
from edge.local_inference import OllamaInference
from cloud.provider_factory import ProviderFactory
from shared.schemas import (
    CodeFragment,
    AnalysisDraft,
    AnalysisResult,
    BudgetStatus,
    AnalysisMetrics,
    CodeLanguage
)
from .budget_controller import BudgetController
from .strategy_manager import StrategyManager

logger = structlog.get_logger()


class Orchestrator:
    """Main orchestrator coordinating Edge and Cloud analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize orchestrator.
        
        Args:
            config: Full configuration dict
        """
        self.config = config
        
        # Initialize components
        self.ast_analyzer = ASTAnalyzer(config.get('ast', {}))
        # confidence_scorer not used in main flow; removed for simplicity
        
        # Local inference (will be created in async context)
        self.local_inference: Optional[OllamaInference] = None
        
        # Optional: Code encoder (e.g. UniXcoder Base) for code search eval
        self.code_encoder: Optional[Any] = None
        self.code_tokenizer: Optional[Any] = None
        # Optional: CodeSearchNet Retriever (for code search eval)
        self.csn_retriever: Optional[Any] = None
        
        # Cloud components
        self.cloud_factory = ProviderFactory(config.get('cloud', {}))
        
        # Budget management (in-memory)
        budget_config = config.get('budget', {})
        self.budget_controller = BudgetController(
            total_budget=budget_config.get('total_budget', 10.0),
            daily_budget=budget_config.get('daily_budget')
        )
        
        # Strategy manager
        self.strategy_manager = StrategyManager(config.get('strategy', {}))
        
        # Performance settings
        perf_config = config.get('performance', {})
        self.local_batch_size = perf_config.get('local_batch_size', 5)
        self.max_concurrent_cloud = config.get('strategy', {}).get(
            'max_concurrent_cloud_calls', 3
        )
        
        # Metrics
        self.metrics = AnalysisMetrics()
        
        # Initialized flag
        self._initialized = False
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        await self.shutdown()
    
    async def initialize(self):
        """Initialize async components."""
        if self._initialized:
            return
        
        # Initialize budget controller (in-memory)
        await self.budget_controller.initialize()
        
        # Create local inference engine
        ollama_config = self.config.get('ollama', {})
        self.local_inference = OllamaInference(ollama_config)
        await self.local_inference.ensure_session()
        
        self._initialized = True
        
        logger.info("orchestrator_initialized")
    
    async def analyze_file(
        self,
        file_path: str,
        language: Optional[CodeLanguage] = None
    ) -> List[AnalysisResult]:
        """Analyze a single file.
        
        Args:
            file_path: Path to source file
            language: Programming language (auto-detected if None)
        
        Returns:
            List of analysis results
        """
        await self.initialize()
        
        logger.info("analyze_file_start", file_path=file_path)
        
        try:
            # Step 1: AST Analysis - Identify Hotspots
            logger.info("ast_analysis_start", file_path=file_path)
            hotspots = self.ast_analyzer.analyze_file(file_path, language)
            logger.info("ast_analysis_complete", 
                       file_path=file_path, 
                       hotspot_count=len(hotspots))
            
            if not hotspots:
                logger.info("no_hotspots_found", file_path=file_path)
                return []
            
            # Step 2: Local LLM Analysis
            fragments = [h.fragment for h in hotspots]
            self.metrics.total_fragments += len(fragments)
            
            logger.info("local_analysis_start", fragment_count=len(fragments))
            
            async with self.local_inference:
                drafts = await self.local_inference.analyze_batch(
                    fragments,
                    batch_size=self.local_batch_size
                )
            
            logger.info("local_analysis_complete", draft_count=len(drafts))
            
            # Step 3: Strategy - Filter for Cloud Verification
            budget_status = await self.budget_controller.get_status()
            
            to_verify, decisions = self.strategy_manager.filter_drafts(
                drafts,
                budget_status,
                max_uploads=self.max_concurrent_cloud
            )
            
            upload_count = len(to_verify)
            logger.info("strategy_filtering_complete",
                       total_drafts=len(drafts),
                       to_verify=upload_count,
                       budget_remaining_pct=budget_status.remaining_percent)
            
            # Log upload decisions
            for decision in decisions:
                logger.info("upload_decision",
                           location=decision.fragment_location,
                           should_upload=decision.should_upload,
                           reason=decision.reason,
                           confidence=decision.confidence_score)
            
            # Step 4: Cloud Verification (async parallel)
            verification_results = {}
            if to_verify:
                logger.info("cloud_verification_start", count=upload_count)
                verification_results = await self._verify_in_cloud(to_verify)
                logger.info("cloud_verification_complete", count=len(verification_results))
            
            # Step 5: Combine Results
            results = self._combine_results(drafts, verification_results)
            
            # Update metrics
            self.metrics.local_only += len(drafts) - upload_count
            self.metrics.cloud_verified += upload_count
            
            logger.info("analyze_file_complete",
                       file_path=file_path,
                       total_results=len(results))
            
            return results
        
        except Exception as e:
            logger.error("analyze_file_error",
                        file_path=file_path,
                        error=str(e))
            raise
    
    async def analyze_directory(
        self,
        dir_path: str,
        pattern: str = "**/*.py",
        max_files: Optional[int] = None
    ) -> Dict[str, List[AnalysisResult]]:
        """Analyze all files in a directory.
        
        Args:
            dir_path: Directory path
            pattern: Glob pattern for files
            max_files: Maximum files to analyze
        
        Returns:
            Dict mapping file paths to results
        """
        await self.initialize()
        
        # Find matching files
        dir_path_obj = Path(dir_path)
        files = list(dir_path_obj.glob(pattern))
        
        if max_files:
            files = files[:max_files]
        
        logger.info("analyze_directory_start",
                   dir_path=dir_path,
                   file_count=len(files))
        
        # Analyze each file
        all_results = {}
        for file_path in files:
            try:
                results = await self.analyze_file(str(file_path))
                if results:
                    all_results[str(file_path)] = results
            except Exception as e:
                logger.error("file_analysis_failed",
                           file_path=str(file_path),
                           error=str(e))
        
        logger.info("analyze_directory_complete",
                   dir_path=dir_path,
                   files_analyzed=len(all_results))
        
        return all_results
    
    async def _verify_in_cloud(
        self,
        drafts: List[AnalysisDraft]
    ) -> Dict[str, Any]:
        """Verify drafts in cloud with controlled concurrency.
        
        Args:
            drafts: Drafts to verify
        
        Returns:
            Dict mapping draft IDs to verification results
        """
        # Get cloud client
        cloud_client = self.cloud_factory.get_client()
        
        # Create verification tasks with semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_cloud)
        
        async def verify_one(draft: AnalysisDraft):
            async with semaphore:
                try:
                    # Determine mode based on confidence
                    mode = "refinement" if draft.confidence.score < 0.3 else "verification"
                    
                    result = await cloud_client.verify(draft, mode)
                    
                    # Record expense
                    from shared.prompts import estimate_cost
                    estimated_cost = estimate_cost("", result.tokens_used, cloud_client.model)
                    
                    await self.budget_controller.record_expense(
                        cost=estimated_cost,
                        provider=cloud_client.provider,
                        model=cloud_client.model,
                        tokens_used=result.tokens_used,
                        operation_type=mode
                    )
                    
                    logger.info("cloud_verification",
                               draft_id=result.draft_id,
                               verified=result.verified,
                               latency_ms=result.latency_ms,
                               tokens=result.tokens_used)
                    
                    return result
                
                except Exception as e:
                    logger.error("cloud_verification_error",
                               draft_location=draft.fragment.get_location(),
                               error=str(e))
                    return None
        
        # Execute verifications in parallel
        tasks = [verify_one(draft) for draft in drafts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build result dict
        result_dict = {}
        for result in results:
            if result and hasattr(result, 'draft_id'):
                result_dict[result.draft_id] = result
        
        return result_dict
    
    def _combine_results(
        self,
        drafts: List[AnalysisDraft],
        verifications: Dict[str, Any]
    ) -> List[AnalysisResult]:
        """Combine local drafts with cloud verifications.
        
        Args:
            drafts: Local analysis drafts
            verifications: Cloud verification results
        
        Returns:
            List of final analysis results
        """
        results = []
        
        for draft in drafts:
            draft_id = f"{draft.fragment.file_path}:{draft.fragment.start_line}"
            verification = verifications.get(draft_id)
            
            # Calculate final confidence
            if verification:
                final_confidence = min(
                    1.0,
                    draft.confidence.score + verification.confidence_boost
                )
            else:
                final_confidence = draft.confidence.score
            
            # Determine final description
            if verification and verification.refined_description:
                final_description = verification.refined_description
            else:
                final_description = draft.description
            
            # Determine final fix
            final_fix = None
            if verification:
                if verification.alternative_fix:
                    final_fix = verification.alternative_fix
                elif verification.best_fix_index is not None and draft.suggested_fixes:
                    idx = verification.best_fix_index
                    if 0 <= idx < len(draft.suggested_fixes):
                        final_fix = draft.suggested_fixes[idx]
            elif draft.suggested_fixes:
                final_fix = draft.suggested_fixes[0]  # Use first suggestion
            
            # Create final result
            result = AnalysisResult(
                draft=draft,
                verification=verification,
                final_confidence=final_confidence,
                final_description=final_description,
                final_fix=final_fix
            )
            
            # Only include if passes reporting threshold
            if self.strategy_manager.should_report(draft):
                results.append(result)
        
        return results
    
    async def get_metrics(self) -> AnalysisMetrics:
        """Get analysis metrics.
        
        Returns:
            Current metrics
        """
        return self.metrics
    
    async def get_budget_status(self) -> BudgetStatus:
        """Get current budget status.
        
        Returns:
            Current budget status
        """
        return await self.budget_controller.get_status()
    
    async def reset_session(self):
        """Reset metrics for new session."""
        self.metrics = AnalysisMetrics()
        self.strategy_manager.reset_history()
        logger.info("session_reset")
    
    async def shutdown(self):
        """Cleanup and shutdown."""
        if self.local_inference is not None:
            await self.local_inference.close()
            self.local_inference = None
        await self.cloud_factory.close_all()
        self._initialized = False
        logger.info("orchestrator_shutdown")

