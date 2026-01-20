# utils/config_loader.py
import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """单个API提供商配置"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    cost_per_1k_tokens: float = 0.0
    enabled: bool = True

    def __post_init__(self):
        """从环境变量加载API密钥"""
        if self.api_key_env and not self.api_key:
            self.api_key = os.getenv(self.api_key_env)
            if not self.api_key:
                logger.warning(f"环境变量 {self.api_key_env} 未设置")


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    provider_type: str
    context_window: int
    max_output_tokens: int
    supported_features: list = field(default_factory=list)


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._config_cache: Dict[str, Any] = {}

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if filename in self._config_cache:
            return self._config_cache[filename]

        filepath = self.config_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"配置文件不存在: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self._config_cache[filename] = config
        return config

    def get_provider_configs(self) -> Dict[str, Dict[str, ProviderConfig]]:
        """获取所有提供商配置"""
        api_config = self.load_yaml("api_config.yaml")
        providers_config = {}

        for provider_type, provider_info in api_config.get("providers", {}).items():
            if not provider_info.get("enabled", True):
                continue

            configs = []
            for config_data in provider_info.get("configs", []):
                config = ProviderConfig(
                    name=config_data["name"],
                    base_url=config_data["base_url"],
                    api_key_env=config_data.get("api_key_env"),
                    timeout=config_data.get("timeout", 60),
                    max_retries=config_data.get("max_retries", 3),
                    cost_per_1k_tokens=config_data.get("cost_per_1k_tokens", 0.0),
                    enabled=config_data.get("enabled", True)
                )
                configs.append(config)

            providers_config[provider_type] = {
                "priority": provider_info.get("priority", 99),
                "configs": configs
            }

        return providers_config

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """获取指定模型的配置"""
        model_config = self.load_yaml("model_config.yaml")

        for name, config_data in model_config.get("models", {}).items():
            if name == model_name:
                return ModelConfig(
                    name=name,
                    provider_type=config_data["provider_type"],
                    context_window=config_data["context_window"],
                    max_output_tokens=config_data["max_output_tokens"],
                    supported_features=config_data.get("supported_features", [])
                )

        return None

    def get_feature_config(self, feature_name: str) -> Dict[str, Any]:
        """获取特征提取配置"""
        model_config = self.load_yaml("model_config.yaml")
        return model_config.get("feature_extraction", {}).get(feature_name, {})

    def get_global_strategy(self) -> Dict[str, Any]:
        """获取全局策略配置"""
        api_config = self.load_yaml("api_config.yaml")
        return api_config.get("global_strategy", {})