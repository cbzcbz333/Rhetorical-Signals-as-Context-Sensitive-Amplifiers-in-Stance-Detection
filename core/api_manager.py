# core/api_manager.py
from typing import Dict, Any, List, Optional
import asyncio
import time
import logging
from datetime import datetime
from collections import defaultdict

from utils.config_loader import ConfigLoader
from providers.provider_factory import ProviderFactory

logger = logging.getLogger(__name__)


class APIManager:
    """API管理器 - 核心调度类"""

    def __init__(self, config_dir: str = "config"):
        self.config_loader = ConfigLoader(config_dir)
        self.provider_factory = ProviderFactory()

        # 初始化所有提供商
        self._init_providers()

        # 加载全局策略
        self.global_strategy = self.config_loader.get_global_strategy()

        # 请求历史记录
        self.request_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000

        # 速率限制
        self.rate_limit_window = self.global_strategy.get("rate_limit_window", 60)
        self.rate_limit_max = self.global_strategy.get("rate_limit_max_requests", 100)
        self.request_timestamps: List[float] = []

        # 缓存
        self.cache_enabled = self.global_strategy.get("cache_enabled", False)
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = self.global_strategy.get("cache_ttl", 3600)

    def _init_providers(self):
        """初始化所有提供商"""
        provider_configs = self.config_loader.get_provider_configs()
        self.providers: Dict[str, List[BaseProvider]] = defaultdict(list)

        for provider_type, config_info in provider_configs.items():
            priority = config_info["priority"]
            for provider_config in config_info["configs"]:
                try:
                    provider = self.provider_factory.create_provider(
                        provider_type, provider_config
                    )
                    self.providers[provider_type].append(provider)
                    logger.info(f"Initialized provider: {provider.name} (type: {provider_type}, priority: {priority})")
                except Exception as e:
                    logger.error(f"Failed to initialize provider {provider_config['name']}: {e}")

    def _check_rate_limit(self):
        """检查速率限制"""
        now = time.time()

        # 清理过期的请求时间戳
        self.request_timestamps = [
            ts for ts in self.request_timestamps
            if now - ts < self.rate_limit_window
        ]

        if len(self.request_timestamps) >= self.rate_limit_max:
            wait_time = self.request_timestamps[0] + self.rate_limit_window - now
            if wait_time > 0:
                logger.warning(f"Rate limit exceeded. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

    def _get_cache_key(self, messages: list, model: str, **kwargs) -> str:
        """生成缓存键"""
        import hashlib
        import json

        cache_data = {
            "messages": messages,
            "model": model,
            "params": {k: v for k, v in kwargs.items() if k != "stream"}
        }

        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    async def chat_completion(
            self,
            messages: list,
            model: str,
            feature_name: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        智能聊天补全

        Args:
            messages: 消息列表
            model: 模型名称
            feature_name: 特征名称（用于选择专用配置）
            **kwargs: 额外参数
        """
        # 检查速率限制
        self._check_rate_limit()

        # 检查缓存
        cache_key = self._get_cache_key(messages, model, **kwargs)
        if self.cache_enabled and cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                logger.debug(f"Cache hit for {cache_key}")
                return cache_entry["response"]

        # 获取特征特定配置
        if feature_name:
            feature_config = self.config_loader.get_feature_config(feature_name)
            if feature_config:
                kwargs.setdefault("temperature", feature_config.get("temperature", 0.0))
                kwargs.setdefault("max_tokens", feature_config.get("max_tokens", 1000))

        # 选择提供商
        provider = self._select_provider(model, feature_name)

        if not provider:
            raise ValueError(f"No available provider for model {model}")

        # 记录请求开始时间
        self.request_timestamps.append(time.time())
        start_time = time.time()

        try:
            # 调用提供商
            response = await provider.chat_completion(messages, model, **kwargs)

            # 记录历史
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "provider": provider.name,
                "feature": feature_name,
                "tokens": response["tokens"],
                "latency": response["latency"],
                "success": True
            }
            self.request_history.append(history_entry)

            # 清理历史记录
            if len(self.request_history) > self.max_history_size:
                self.request_history = self.request_history[-self.max_history_size:]

            # 缓存结果
            if self.cache_enabled:
                self.cache[cache_key] = {
                    "response": response,
                    "timestamp": time.time()
                }

            logger.info(f"API call successful: model={model}, provider={provider.name}, "
                        f"tokens={response['tokens']}, latency={response['latency']:.2f}s")

            return response

        except Exception as e:
            latency = time.time() - start_time

            # 记录失败历史
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "provider": provider.name if provider else "unknown",
                "feature": feature_name,
                "error": str(e),
                "latency": latency,
                "success": False
            }
            self.request_history.append(history_entry)

            logger.error(f"API call failed: model={model}, provider={provider.name}, error={e}")

            # 如果启用了故障转移，尝试其他提供商
            if self.global_strategy.get("fallback_enabled", True):
                logger.info("Trying fallback providers...")
                return await self._try_fallback(messages, model, feature_name, **kwargs)

            raise

    def _select_provider(self, model: str, feature_name: Optional[str] = None) -> Optional[BaseProvider]:
        """选择最合适的提供商"""
        # 获取模型配置
        model_config = self.config_loader.get_model_config(model)
        if not model_config:
            logger.warning(f"No configuration found for model {model}")
            return None

        provider_type = model_config.provider_type

        # 获取该类型的所有提供商
        available_providers = [
            p for p in self.providers.get(provider_type, [])
            if p.enabled and p.health_check() and p.supports_model(model)
        ]

        if not available_providers:
            logger.warning(f"No available providers for {provider_type}")
            return None

        # 根据策略选择提供商
        strategy = self.global_strategy.get("load_balancing", "priority")

        if strategy == "round_robin":
            # 简单轮询
            self._last_provider_index = getattr(self, '_last_provider_index', {})
            last_idx = self._last_provider_index.get(provider_type, -1)
            next_idx = (last_idx + 1) % len(available_providers)
            self._last_provider_index[provider_type] = next_idx
            return available_providers[next_idx]

        elif strategy == "cost_effective":
            # 选择成本最低的
            return min(available_providers, key=lambda p: p.cost_per_1k_tokens)

        else:  # priority (默认)
            # 按配置的优先级排序
            provider_configs = self.config_loader.get_provider_configs()
            type_config = provider_configs.get(provider_type, {})

            # 这里可以添加更复杂的优先级逻辑
            # 暂时返回第一个可用的
            return available_providers[0] if available_providers else None

    async def _try_fallback(self, messages: list, model: str, feature_name: str, **kwargs):
        """尝试故障转移"""
        # 获取备选模型
        if feature_name:
            feature_config = self.config_loader.get_feature_config(feature_name)
            fallback_models = feature_config.get("fallback_models", [])

            for fallback_model in fallback_models:
                try:
                    logger.info(f"Trying fallback model: {fallback_model}")
                    return await self.chat_completion(
                        messages, fallback_model, feature_name, **kwargs
                    )
                except Exception as e:
                    logger.warning(f"Fallback model {fallback_model} also failed: {e}")
                    continue

        raise Exception("All providers and fallback models failed")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "total_requests": len(self.request_history),
            "success_rate": 0,
            "total_tokens": 0,
            "total_cost": 0,
            "avg_latency": 0,
            "providers": {}
        }

        # 计算总体统计
        successful_requests = [r for r in self.request_history if r["success"]]
        if self.request_history:
            stats["success_rate"] = len(successful_requests) / len(self.request_history)
            stats["avg_latency"] = sum(r.get("latency", 0) for r in successful_requests) / len(successful_requests)

        # 获取每个提供商的统计
        for provider_type, providers in self.providers.items():
            stats["providers"][provider_type] = [
                provider.get_stats() for provider in providers
            ]
            for provider in providers:
                stats["total_tokens"] += provider.total_tokens
                stats["total_cost"] += provider.total_cost

        return stats

    def disable_provider(self, provider_name: str):
        """禁用提供商"""
        for providers in self.providers.values():
            for provider in providers:
                if provider.name == provider_name:
                    provider.enabled = False
                    logger.info(f"Disabled provider: {provider_name}")
                    return

        logger.warning(f"Provider {provider_name} not found")

    def enable_provider(self, provider_name: str):
        """启用提供商"""
        for providers in self.providers.values():
            for provider in providers:
                if provider.name == provider_name:
                    provider.enabled = True
                    logger.info(f"Enabled provider: {provider_name}")
                    return

        logger.warning(f"Provider {provider_name} not found")