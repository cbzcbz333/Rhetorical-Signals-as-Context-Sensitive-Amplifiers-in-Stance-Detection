# config.py
import os
from dotenv import load_dotenv
from typing import Optional

# 加载环境变量
load_dotenv()


class APIConfig:
    """安全的API配置管理器"""

    def __init__(self):
        # 从环境变量读取
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_urls = [
            os.getenv("OPENAI_API_BASE", "https://api.runapi.sbs/v1"),
            os.getenv("OPENAI_API_BASE_BACKUP", "https://api.runapi.cfd/v1"),
        ]

        # 验证配置
        self._validate_config()

    def _validate_config(self):
        """验证配置是否完整"""
        if not self.api_key or not self.api_key.startswith("sk-"):
            raise ValueError(
                "API密钥未设置或格式错误。请检查：\n"
                "1. 是否在.env文件中设置了OPENAI_API_KEY\n"
                "2. 密钥是否以'sk-'开头\n"
                "3. 是否已安装python-dotenv"
            )

        # 检查密钥是否太短（可能是示例密钥）
        if len(self.api_key) < 30:
            print("⚠️  警告：API密钥可能过短，请确认是否正确")

    def get_base_url(self, use_backup: bool = False) -> str:
        """获取基础URL"""
        return self.base_urls[1] if use_backup else self.base_urls[0]

    def get_client_config(self, use_backup: bool = False) -> dict:
        """获取客户端配置"""
        return {
            "api_key": self.api_key,
            "base_url": self.get_base_url(use_backup),
            "timeout": 60.0,
        }

    def rotate_base_url(self) -> bool:
        """轮换基础URL（故障转移）"""
        # 简单实现：交换URL顺序
        if len(self.base_urls) > 1:
            self.base_urls[0], self.base_urls[1] = self.base_urls[1], self.base_urls[0]
            return True
        return False


# 创建全局配置实例
config = APIConfig()