# test_simple.py
import os
import json
from openai import OpenAI


def test_basic_connection():
    """简单直接的API连接测试"""

    # 配置 - 直接使用你的中转接口
    API_KEY = "sk-JhzsitLNi4ztobLxgmbdIBCPXtUPTFwFmkYdAsOILqW1xDEy"  # 注意：这是你暴露的密钥，请尽快更换！
    BASE_URLS = [
        "https://api.shubiaobiao.com/v1",
        "https://api.shubiaobiao.com/v2",
    ]

    test_text = "这是一个测试句子。请问今天天气怎么样？难道你不知道吗？"

    for base_url in BASE_URLS:
        print(f"\n尝试连接: {base_url}")

        try:
            client = OpenAI(
                api_key=API_KEY,
                base_url=base_url,
                timeout=30
            )

            # 先尝试简单的聊天，不指定具体模型
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # 先尝试通用模型
                messages=[
                    {"role": "system", "content": "你是一个测试助手，请回复'test_ok'表示连接成功"},
                    {"role": "user", "content": "ping"}
                ],
                max_tokens=10,
                temperature=0
            )

            print(f"✅ 连接成功！")
            print(f"响应: {response.choices[0].message.content}")
            print(f"模型: {response.model}")
            print(f"使用token: {response.usage.total_tokens}")

            # 如果基本连接成功，再测试特征提取
            print(f"\n测试特征提取...")
            return test_feature_extraction(client, base_url, test_text)

        except Exception as e:
            print(f"❌ 连接失败: {e}")
            continue

    print("\n⚠️ 所有端点都失败，请检查：")
    print("1. API密钥是否正确且有效")
    print("2. 网络是否能访问这些域名")
    print("3. 模型名称是否正确")
    return None


def test_feature_extraction(client, base_url, text):
    """测试特征提取功能"""

    SYSTEM_PROMPT = """你是一个专业的修辞分析助手。请分析文本中的疑问句使用情况。

分析要求：
1. 识别所有疑问句（以问号结尾的句子）
2. 判断每个疑问句是否为反问句
3. 反问句的判定标准（满足任一即可）：
   - 答案已在问题中明显隐含
   - 不需要对方回答，用于强调观点
   - 包含强化语气的词："难道"、"岂"、"何尝"、"岂不是"

请返回严格的JSON格式：
{
  "question_count": 整数,
  "rhetorical_question_count": 整数,
  "questions": [
    {
      "text": "完整句子",
      "is_rhetorical": true/false,
      "reason": "简短判断理由"
    }
  ]
}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请分析以下文本：{text}"}
            ],
            response_format={"type": "json_object"}  # 要求返回JSON
        )

        content = response.choices[0].message.content
        print(f"✅ 特征提取成功！")
        print(f"原始响应: {content}")

        # 解析JSON
        features = json.loads(content)

        # 计算比例
        qc = features.get("question_count", 0)
        rc = features.get("rhetorical_question_count", 0)
        features["rhetorical_question_ratio"] = rc / qc if qc > 0 else 0.0

        print(f"\n分析结果:")
        print(f"疑问句总数: {qc}")
        print(f"反问句数量: {rc}")
        print(f"反问句比例: {features['rhetorical_question_ratio']:.2f}")

        if "questions" in features:
            for i, q in enumerate(features["questions"], 1):
                print(f"\n问题 {i}: {q.get('text', '')}")
                print(f"  是否反问: {q.get('is_rhetorical', False)}")
                print(f"  理由: {q.get('reason', '')}")

        return features

    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {e}")
        return None
    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        return None


def test_available_models(base_url, api_key):
    """测试可用的模型列表"""
    print(f"\n测试 {base_url} 可用的模型...")

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30
        )

        # 尝试获取模型列表（某些中转API支持）
        try:
            models = client.models.list()
            print("可用模型:")
            for model in models.data:
                print(f"  - {model.id}")
        except:
            # 如果不支持list接口，尝试几种常见模型
            test_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]

            for model_name in test_models:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": "hello"}],
                        max_tokens=5
                    )
                    print(f"✅ {model_name}: 可用")
                except Exception as e:
                    print(f"❌ {model_name}: 不可用 - {str(e)[:100]}")

    except Exception as e:
        print(f"❌ 模型测试失败: {e}")


if __name__ == "__main__":
    print("开始API连接测试...")

    # 重要提醒
    print("⚠️ 警告：你正在使用已暴露的API密钥，请立即更换！")

    # 测试基本连接
    features = test_basic_connection()

    # 测试可用模型
    if features:
        # 更换你的新密钥！
        NEW_API_KEY = "你的新密钥_在这里输入"
        if NEW_API_KEY != "你的新密钥_在这里输入":  # 如果已经更换
            test_available_models("https://api.shubiaobiao.com/v1", NEW_API_KEY)
            test_available_models("https://api.shubiaobiao.com/v2", NEW_API_KEY)