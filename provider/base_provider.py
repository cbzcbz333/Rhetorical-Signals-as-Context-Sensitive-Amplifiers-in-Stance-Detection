# providers/base_provider.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import logging

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """提供商基类"""

    def __init__(self, config: Dict[str, Any]):
        self.name = config.get("name", "unknown")
        self.base_url = config.get("base_url")
        self.api_key = config.get("api_key")
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)
        self.cost_per_1k_tokens = config.get("cost_per_1k_tokens", 0.0)

        # 统计信息
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.error_count = 0
        self.total_latency = 0.0

        self._enabled = config.get("enabled", True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.request_count if self.request_count > 0 else 0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.request_count if self.request_count > 0 else 0

    def record_request(self, tokens_used: int, latency: float, success: bool = True):
        """记录请求统计信息"""
        self.request_count += 1
        self.total_tokens += tokens_used
        self.total_latency += latency

        cost = (tokens_used / 1000) * self.cost_per_1k_tokens
        self.total_cost += cost

        if not success:
            self.error_count += 1

        logger.debug(f"Provider {self.name}: tokens={tokens_used}, latency={latency:.2f}s, "
                     f"cost=${cost:.4f}, success={success}")

    @abstractmethod
    async def chat_completion(self, messages: list, model: str, **kwargs) -> Dict[str, Any]:
        """聊天补全接口"""
        pass

    @abstractmethod
    def supports_model(self, model: str) -> bool:
        """是否支持指定模型"""
        pass

    def health_check(self) -> bool:
        """健康检查"""
        return self._enabled and self.error_rate < 0.5  # 错误率低于50%视为健康

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "name": self.name,
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "avg_latency": self.avg_latency,
            "enabled": self.enabled
        }