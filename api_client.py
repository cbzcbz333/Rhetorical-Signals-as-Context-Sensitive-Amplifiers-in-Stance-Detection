# api_client.py
import os
import json
import logging
from typing import Dict, Any, Optional
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

logger = logging.getLogger(__name__)


class CustomOpenAIClient:
    """自定义OpenAI客户端，支持中转接口"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化客户端

        Args:
            api_key: 可选的API密钥，如不提供则从环境变量读取
            base_url: 可选的base_url，如不提供则使用默认
        """
        # 优先级：参数 > 环境变量 > 默认值
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # 默认使用第一个接口，支持故障转移
        self.base_urls = [
            base_url or os.getenv("OPENAI_API_BASE", "https://api.shubiaobiao.com/v1"),
            os.getenv("OPENAI_API_BASE_BACKUP", "https://api.shubiaobiao.com/v2"),
        ]

        self.current_base_url_index = 0
        self.max_retries = 2  # 故障转移重试次数

        self._validate_credentials()
        self.client = self._create_client()

    def _validate_credentials(self):
        self.api_key="sk-JhzsitLNi4ztobLxgmbdIBCPXtUPTFwFmkYdAsOILqW1xDEy"

        """验证凭证"""
        if not self.api_key:
            raise ValueError("API密钥未提供。请设置OPENAI_API_KEY环境变量或传入api_key参数")


        # 安全检查：不要硬编码密钥
        #sample_keys = ["sk-示例密钥", "sk-LbtJMiO8GjQ6ohsb6a33935a216f43D5Bf08325cEa5722E5"]


    def _create_client(self, use_next_url: bool = False) -> OpenAI:
        """创建OpenAI客户端实例"""
        if use_next_url:
            self.current_base_url_index = (self.current_base_url_index + 1) % len(self.base_urls)

        current_url = self.base_urls[self.current_base_url_index]

        logger.info(f"使用API端点: {current_url}")

        return OpenAI(
            api_key=self.api_key,
            base_url=current_url,
            timeout=60.0,
            max_retries=2,
        )

    def _switch_endpoint(self):
        """切换到下一个端点"""
        old_url = self.base_urls[self.current_base_url_index]
        self.client = self._create_client(use_next_url=True)
        new_url = self.base_urls[self.current_base_url_index]
        logger.warning(f"API端点切换: {old_url} -> {new_url}")

    def chat_completion_with_retry(self, **kwargs) -> Any:
        """
        带故障转移的聊天补全

        Args:
            **kwargs: 传递给client.chat.completions.create的参数

        Returns:
            API响应

        Raises:
            APIError: 所有端点都失败时抛出
        """
        last_error = None

        for attempt in range(self.max_retries + 1):  # 初始尝试 + 重试次数
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response

            except (APIError, APITimeoutError, RateLimitError) as e:
                last_error = e
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {str(e)}")

                if attempt < self.max_retries:
                    self._switch_endpoint()
                    continue
                else:
                    break

        # 所有尝试都失败
        error_msg = f"所有API端点都失败。最后错误: {str(last_error)}"
        logger.error(error_msg)
        raise APIError(error_msg, request=last_error.request if hasattr(last_error, 'request') else None)