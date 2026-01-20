# providers/openai_provider.py
from typing import Dict, Any, Optional
import asyncio
from openai import OpenAI, AsyncOpenAI
from openai import APIError, RateLimitError, APITimeoutError
import logging

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI兼容提供商"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        self.sync_client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

        # 支持的模型前缀
        self.supported_prefixes = ["gpt-", "text-", "dall-e"]

    def supports_model(self, model: str) -> bool:
        """检查是否支持该模型"""
        return any(model.startswith(prefix) for prefix in self.supported_prefixes)

    async def chat_completion(self, messages: list, model: str, **kwargs) -> Dict[str, Any]:
        """异步聊天补全"""
        if not self.enabled:
            raise ValueError(f"Provider {self.name} is disabled")

        start_time = time.time()
        max_retries = kwargs.pop("max_retries", self.max_retries)

        for attempt in range(max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )

                latency = time.time() - start_time
                tokens_used = response.usage.total_tokens if response.usage else 0

                self.record_request(tokens_used, latency, success=True)

                return {
                    "content": response.choices[0].message.content,
                    "model": response.model,
                    "tokens": tokens_used,
                    "latency": latency,
                    "provider": self.name
                }

            except (APIError, RateLimitError, APITimeoutError) as e:
                latency = time.time() - start_time
                self.record_request(0, latency, success=False)

                if attempt < max_retries:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"All {max_retries + 1} attempts failed for provider {self.name}")
                    raise

    def chat_completion_sync(self, messages: list, model: str, **kwargs) -> Dict[str, Any]:
        """同步聊天补全（兼容旧代码）"""
        if not self.enabled:
            raise ValueError(f"Provider {self.name} is disabled")

        start_time = time.time()

        try:
            response = self.sync_client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )

            latency = time.time() - start_time
            tokens_used = response.usage.total_tokens if response.usage else 0

            self.record_request(tokens_used, latency, success=True)

            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "tokens": tokens_used,
                "latency": latency,
                "provider": self.name
            }

        except Exception as e:
            latency = time.time() - start_time
            self.record_request(0, latency, success=False)
            raise