"""Builds and caches ``CloudClient`` instances from YAML ``cloud`` config."""

from typing import Dict, Any, Optional, List
from enum import Enum

from .client import CloudClient


class CloudProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


class ProviderFactory:
    """Lazy ``CloudClient`` pool keyed by provider name."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.default_provider = config.get('default_provider', 'openai')
        self._clients: Dict[str, CloudClient] = {}
        self._health_status: Dict[str, bool] = {}
    
    def get_client(
        self,
        provider: Optional[str] = None,
        fallback: bool = True
    ) -> CloudClient:
        if provider is None:
            provider = self.default_provider

        if provider in self._clients:
            return self._clients[provider]

        try:
            client = CloudClient(self.config, provider)
            self._clients[provider] = client
            return client
        except Exception as e:
            if fallback:
                return self._get_fallback_client(provider)
            raise RuntimeError(f"Failed to create client for {provider}: {e}")

    def _get_fallback_client(self, failed_provider: str) -> CloudClient:
        fallback_order = [
            CloudProvider.OPENAI,
            CloudProvider.ANTHROPIC,
            CloudProvider.CUSTOM
        ]
        
        fallback_order = [p for p in fallback_order if p.value != failed_provider]

        for provider in fallback_order:
            try:
                if provider.value in self.config:
                    client = CloudClient(self.config, provider.value)
                    self._clients[provider.value] = client
                    return client
            except:
                continue
        
        raise RuntimeError("All cloud providers failed")
    
    async def check_health(self, provider: str) -> bool:
        try:
            client = self.get_client(provider, fallback=False)
            is_healthy = await client.health_check()
            self._health_status[provider] = is_healthy
            return is_healthy
        except:
            self._health_status[provider] = False
            return False
    
    async def check_all_health(self) -> Dict[str, bool]:
        providers = [p for p in self.config.keys() if p != 'default_provider']
        
        for provider in providers:
            await self.check_health(provider)
        
        return dict(self._health_status)
    
    def get_available_providers(self) -> List[str]:
        return [
            provider
            for provider, is_healthy in self._health_status.items()
            if is_healthy
        ]
    
    def get_provider_info(self) -> Dict[str, Dict[str, Any]]:
        info = {}
        
        for provider, config in self.config.items():
            if provider == 'default_provider':
                continue
            
            info[provider] = {
                'model': config.get('model', 'unknown'),
                'base_url': config.get('base_url', 'default'),
                'is_healthy': self._health_status.get(provider, None),
                'has_api_key': bool(config.get('api_key'))
            }
        
        return info
    
    def switch_default_provider(self, provider: str):
        if provider not in self.config:
            raise ValueError(f"Provider {provider} not configured")
        
        self.default_provider = provider
    
    def get_metrics(self) -> Dict[str, Any]:
        metrics = {}
        
        for provider, client in self._clients.items():
            metrics[provider] = client.get_metrics()
        
        return metrics
    
    async def close_all(self):
        """Close all active clients and release HTTP connections."""
        for client in list(self._clients.values()):
            await client.aclose()
        self._clients.clear()

