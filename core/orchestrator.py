"""Wires Ollama, cloud clients, and budget for CSN eval."""

from typing import Any, Dict, Optional

import structlog

from cloud.provider_factory import ProviderFactory
from edge.local_inference import OllamaInference

from .budget_controller import BudgetController

logger = structlog.get_logger()


class Orchestrator:
    """async context: init Ollama session + ``ProviderFactory``."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.local_inference: Optional[OllamaInference] = None
        self.code_encoder: Optional[Any] = None
        self.code_tokenizer: Optional[Any] = None
        self.csn_retriever: Optional[Any] = None

        budget_config = config.get("budget", {})
        self.budget_controller = BudgetController(
            total_budget=budget_config.get("total_budget", 10.0),
            daily_budget=budget_config.get("daily_budget"),
        )
        self.cloud_factory = ProviderFactory(config.get("cloud", {}))
        self._initialized = False

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.shutdown()

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self.budget_controller.initialize()
        ollama_config = self.config.get("ollama", {})
        self.local_inference = OllamaInference(ollama_config)
        await self.local_inference.ensure_session()
        self._initialized = True
        logger.info("orchestrator_initialized")

    async def shutdown(self) -> None:
        if self.local_inference is not None:
            await self.local_inference.close()
            self.local_inference = None
        await self.cloud_factory.close_all()
        self._initialized = False
        logger.info("orchestrator_shutdown")
