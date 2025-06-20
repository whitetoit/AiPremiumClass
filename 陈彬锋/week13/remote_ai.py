from zhipuai import ZhipuAI

api_key = "36d7d32e72fc4e2498be47589aca3ca5.RsFlLAtDCiBTzYRZ"
client = ZhipuAI(api_key=api_key)

def test_zhipuai_parameters():
    """智谱AI API参数调试函数"""
    prompt = "解释量子力学的基本原理"

    # 测试不同temperature值
    print("\n===== temperature参数测试 =====")
    for temp in [0.1, 0.5, 1.0]:
        response = client.chat.completions.create(
            model="glm-z1-airx",  # 使用GLM-4模型
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=150
        )
        print(f"\ntemperature={temp}:")
        result = response.choices[0].message.content
        print(result[:200] + "..." if len(result) > 200 else result)

    # 测试不同top_p值
    print("\n===== top_p参数测试 =====")
    for top_p in [0.5, 0.7, 0.9]:
        response = client.chat.completions.create(
            model="glm-z1-airx",
            messages=[{"role": "user", "content": prompt}],
            top_p=top_p,
            max_tokens=150
        )
        print(f"\ntop_p={top_p}:")
        result = response.choices[0].message.content
        print(result[:200] + "..." if len(result) > 200 else result)

    # 测试不同max_tokens值
    print("\n===== max_tokens参数测试 =====")
    for tokens in [50, 100, 150]:
        response = client.chat.completions.create(
            model="glm-z1-airx",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=tokens
        )
        print(f"\nmax_tokens={tokens}:")
        print(response.choices[0].message.content)

if __name__ == "__main__":
    # 安装必要的库 (如果尚未安装)
    try:
        import zhipuai
        from zhipuai import ZhipuAI
    except ImportError:
        print("安装zhipuai库...")
        import os

        os.system("pip install zhipuai")
        import zhipuai
        from zhipuai import ZhipuAI

    # 执行API参数测试
    print("正在执行智谱AI API参数测试...")
    test_zhipuai_parameters()

    print("\n" + "=" * 50 + "\n")