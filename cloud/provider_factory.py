"""Factory for creating cloud provider clients."""

from typing import Dict, Any, Optional, List
from enum import Enum

from .client import CloudClient


class CloudProvider(str, Enum):
    """Supported cloud providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


class ProviderFactory:
    """Factory for creating and managing cloud provider clients."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize provider factory.
        
        Args:
            config: Full cloud configuration
        """
        self.config = config
        self.default_provider = config.get('default_provider', 'openai')
        self._clients: Dict[str, CloudClient] = {}
        self._health_status: Dict[str, bool] = {}
    
    def get_client(
        self,
        provider: Optional[str] = None,
        fallback: bool = True
    ) -> CloudClient:
        """Get or create a cloud client.
        
        Args:
            provider: Provider name (uses default if None)
            fallback: Whether to fallback to other providers on failure
        
        Returns:
            CloudClient instance
        """
        if provider is None:
            provider = self.default_provider
        
        # Return cached client if available
        if provider in self._clients:
            return self._clients[provider]
        
        # Create new client
        try:
            client = CloudClient(self.config, provider)
            self._clients[provider] = client
            return client
        except Exception as e:
            if fallback:
                # Try fallback providers
                return self._get_fallback_client(provider)
            raise RuntimeError(f"Failed to create client for {provider}: {e}")
    
    def _get_fallback_client(self, failed_provider: str) -> CloudClient:
        """Get a fallback provider when primary fails.
        
        Args:
            failed_provider: Provider that failed
        
        Returns:
            CloudClient for fallback provider
        """
        # Define fallback order
        fallback_order = [
            CloudProvider.OPENAI,
            CloudProvider.ANTHROPIC,
            CloudProvider.CUSTOM
        ]
        
        # Remove failed provider
        fallback_order = [p for p in fallback_order if p.value != failed_provider]
        
        # Try each fallback
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
        """Check health of a provider.
        
        Args:
            provider: Provider name
        
        Returns:
            True if healthy
        """
        try:
            client = self.get_client(provider, fallback=False)
            is_healthy = await client.health_check()
            self._health_status[provider] = is_healthy
            return is_healthy
        except:
            self._health_status[provider] = False
            return False
    
    async def check_all_health(self) -> Dict[str, bool]:
        """Check health of all configured providers.
        
        Returns:
            Dict mapping provider names to health status
        """
        providers = [p for p in self.config.keys() if p != 'default_provider']
        
        for provider in providers:
            await self.check_health(provider)
        
        return dict(self._health_status)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available (healthy) providers.
        
        Returns:
            List of provider names
        """
        return [
            provider
            for provider, is_healthy in self._health_status.items()
            if is_healthy
        ]
    
    def get_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all providers.
        
        Returns:
            Dict with provider configs and status
        """
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
        """Switch the default provider.
        
        Args:
            provider: New default provider name
        """
        if provider not in self.config:
            raise ValueError(f"Provider {provider} not configured")
        
        self.default_provider = provider
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from all active clients.
        
        Returns:
            Dict with metrics per provider
        """
        metrics = {}
        
        for provider, client in self._clients.items():
            metrics[provider] = client.get_metrics()
        
        return metrics
    
    async def close_all(self):
        """Close all active clients and release HTTP connections."""
        for client in list(self._clients.values()):
            await client.aclose()
        self._clients.clear()

