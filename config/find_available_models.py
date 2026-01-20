# final_model_test.py
import requests
import json
import time


def test_key_models():
    """测试最可能成功的几个模型"""
    api_key = "sk-JhzsitLNi4ztobLxgmbdIBCPXtUPTFwFmkYdAsOILqW1xDEy"
    endpoint = "https://api.shubiaobiao.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 基于你的模型列表，选择最可能成功的
    key_models = [
        "gpt-3.5-turbo",  # 最基本
        "gpt-3.5-turbo-0125",  # 较新版本
        "gpt-4o-mini",  # 最新轻量版
        "gpt-4o",  # 最新标准版
        "gpt-4-turbo",  # 较新版
        "gpt-4.1-mini",  # 最新mini版
        "chatgpt-4o-latest",  # 特殊名称
        "gpt-4.5-preview",  # 预览版
    ]

    print("测试关键模型...")
    print("=" * 60)

    working_models = []

    for model in key_models:
        print(f"\n测试: {model:<25}", end="")

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "简单测试"},
                {"role": "user", "content": "请回复'测试通过'"}
            ],
            "max_tokens": 5,
            "temperature": 0
        }

        try:
            start_time = time.time()
            response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if "choices" in data and data["choices"]:
                    content = data["choices"][0]["message"]["content"]
                    print(f" ✅ 成功 ({response_time:.2f}s) - '{content}'")
                    working_models.append({
                        "model": model,
                        "time": response_time,
                        "response": data
                    })
                else:
                    print(f" ⚠️  格式异常")
            else:
                error_text = response.text[:150] if response.text else ""
                print(f" ❌ {response.status_code}")
                if "无可用渠道" in error_text:
                    print(f"    错误: 分组配置问题 - 请联系提供商配置此模型")
                else:
                    print(f"    错误: {error_text}")

        except Exception as e:
            print(f" 💥 异常: {str(e)[:50]}")

        time.sleep(0.5)

    return working_models


def test_without_specified_model():
    """测试不指定模型（让API自动选择）"""
    print("\n" + "=" * 60)
    print("测试不指定模型（让API自动选择）...")
    print("=" * 60)

    api_key = "sk-JhzsitLNi4ztobLxgmbdIBCPXtUPTFwFmkYdAsOILqW1xDEy"
    endpoint = "https://api.shubiaobiao.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 尝试不指定模型
    payloads = [
        {"messages": [{"role": "user", "content": "测试"}]},  # 最简
        {"messages": [{"role": "user", "content": "测试"}], "max_tokens": 5},
    ]

    for i, payload in enumerate(payloads, 1):
        print(f"\n尝试 {i}: {payload}")

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
            print(f"  状态码: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ 成功!")
                print(f"     模型: {data.get('model', '未指定')}")
                if "choices" in data:
                    content = data["choices"][0]["message"]["content"]
                    print(f"     回复: '{content}'")
                return True, data
            else:
                print(f"  ❌ 失败: {response.text[:100]}")

        except Exception as e:
            print(f"  💥 异常: {e}")

    return False, None


if __name__ == "__main__":
    print("开始最终模型测试...")

    # 测试关键模型
    working = test_key_models()

    if working:
        print(f"\n🎉 找到 {len(working)} 个可用模型:")
        for item in working:
            print(f"  - {item['model']} ({item['time']:.2f}s)")
    else:
        print("\n😞 所有指定模型都失败")

        # 尝试不指定模型
        success, data = test_without_specified_model()

        if not success:
            print("\n⚠️ 所有测试都失败")
            print("\n问题分析:")
            print("1. 你的API密钥所在分组 'openai-1' 没有配置任何模型渠道")
            print("2. 你需要联系API提供商:")
            print("   - 登录管理面板")
            print("   - 检查分组配置")
            print("   - 为分组添加模型渠道")
            print("3. 或者让提供商将你的密钥移到有模型配置的分组")