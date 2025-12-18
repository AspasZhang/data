"""
问题改写器（Question Rewriter）
对输入问题进行轻微改写，保持故障本质，增加表达多样性
"""

import json
import openai
import random
from typing import Optional


class QuestionRewriter:
    """问题改写器 - 改变问法但保持故障本质"""
    
    def __init__(self, api_key: str, api_base: str = None, model: str = "gpt-4o-mini"):
        """
        初始化问题改写器
        
        Args:
            api_key: API密钥
            api_base: API基础URL
            model: 模型名称
        """
        self.api_key = api_key
        self.api_base = api_base or "http://10.12.208.86:8502"
        self.model = model
    
    def rewrite_question(self, question: str, mode: str = 'light') -> str:
        """
        改写问题
        
        Args:
            question: 原始问题
            mode: 改写模式
                - 'light': 轻微改写（改变疑问词、语序）
                - 'medium': 中等改写（改变句式、添加修饰）
                - 'none': 不改写（直接返回原问题）
        
        Returns:
            改写后的问题
        """
        if mode == 'none':
            return question
        
        prompt = self._generate_rewrite_prompt(question, mode)
        
        try:
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个网络故障诊断专家，擅长用不同方式表达相同的技术问题。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,  # 较高温度增加多样性
                max_tokens=200
            )
            
            rewritten = response.choices[0].message.content.strip()
            
            # 移除可能的引号
            rewritten = rewritten.strip('"\'')
            
            return rewritten
            
        except Exception as e:
            print(f"⚠️  问题改写失败: {e}，使用原问题")
            return question
    
    def _generate_rewrite_prompt(self, question: str, mode: str) -> str:
        """生成改写prompt"""
        
        if mode == 'light':
            prompt = f"""请对以下网络故障问题进行轻微改写，要求：

原问题：
{question}

改写要求：
1. **必须保持**：设备名、接口名、IP地址等关键参数完全不变
2. **必须保持**：故障类型（如丢包、连通性、性能等）不变
3. **可以改变**：
   - 疑问词（如何→怎么样、怎样、如何处理→怎么处理、应该如何→该如何）
   - 语序（稍微调整）
   - 连接词（发生→出现、该→应该）
4. 改写要自然、简洁
5. **只输出改写后的问题，不要有任何解释或其他文字**

改写后的问题："""

        else:  # medium
            prompt = f"""请对以下网络故障问题进行中等程度改写，要求：

原问题：
{question}

改写要求：
1. **必须保持**：设备名、接口名、IP地址等关键参数完全不变
2. **必须保持**：故障类型不变
3. **可以改变**：
   - 句式结构（疑问句→陈述句、主动→被动等）
   - 添加简单的状语（在...上、对于...来说）
   - 改变专业术语的表达方式（但不改变含义）
   - 问题的提问角度
4. 改写要自然流畅
5. **只输出改写后的问题，不要有任何解释或其他文字**

改写后的问题："""
        
        return prompt
    
    def rewrite_with_strategy(self, question: str, run_id: int, total_runs: int) -> str:
        """
        根据run_id使用不同的改写策略
        
        Args:
            question: 原始问题
            run_id: 当前运行ID (0-based)
            total_runs: 总运行次数
            
        Returns:
            改写后的问题
        """
        # 第一次运行使用原问题
        if run_id == 0:
            return question
        
        # 其他运行随机选择改写模式
        # 60% light, 30% medium, 10% none
        rand = random.random()
        if rand < 0.6:
            mode = 'light'
        elif rand < 0.9:
            mode = 'medium'
        else:
            mode = 'none'
        
        rewritten = self.rewrite_question(question, mode)
        
        # 验证改写是否成功（长度不能太短或太长）
        if len(rewritten) < len(question) * 0.5 or len(rewritten) > len(question) * 2:
            print(f"⚠️  改写结果长度异常，使用原问题")
            return question
        
        return rewritten


def test_question_rewriter():
    """测试问题改写器"""
    print("=" * 80)
    print("测试问题改写器")
    print("=" * 80 + "\n")
    
    rewriter = QuestionRewriter(
        api_key="kw-qIdb2KBfLLBkk6YEJ1clWKKOctnHgWMjtfRJwQ2yTLBCXjMv",
        api_base="http://10.12.208.86:8502"
    )
    
    # 测试问题
    test_questions = [
        "serverleaf01_1_16.135设备上10GE1/0/24接口发生丢包该如何处理？",
        "交换机192.168.1.1的eth0接口流量异常，速度很慢，怎么排查？",
        "网络设备端口频繁up/down，如何诊断？"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"测试{i}：")
        print(f"原问题: {question}")
        print("-" * 80)
        
        # 测试light模式
        rewritten_light = rewriter.rewrite_question(question, mode='light')
        print(f"Light改写: {rewritten_light}")
        
        # 测试medium模式
        rewritten_medium = rewriter.rewrite_question(question, mode='medium')
        print(f"Medium改写: {rewritten_medium}")
        
        print("=" * 80 + "\n")


if __name__ == '__main__':
    test_question_rewriter()
