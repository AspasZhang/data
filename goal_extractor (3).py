"""
目标提取器（Goal Extractor）
从问题描述中提取诊断目标和关键信息
"""

import json
import openai
from typing import Dict, List, Any, Optional


class GoalExtractor:
    """目标提取器 - 从问题中提取诊断目标"""
    
    def __init__(self, api_key: str, api_base: str = None, model: str = "gpt-4o-mini"):
        """
        初始化目标提取器
        
        Args:
            api_key: API密钥
            api_base: API基础URL
            model: 模型名称
        """
        self.api_key = api_key
        self.api_base = api_base or "http://10.12.208.86:8502"
        self.model = model
    
    def extract_goals(self, question: str, knowledge_base: Optional[Dict] = None) -> Dict[str, Any]:
        """
        从问题中提取诊断目标，并从knowledge_base的mock_data中提取参数信息
        
        Args:
            question: 问题描述
            knowledge_base: 知识库（可选），从mock_data中提取参数
            
        Returns:
            {
                "main_goal": "主要目标",
                "key_aspects": ["方面1", "方面2", ...],
                "entities": {"device": "...", "interface": "..."},
                "problem_type": "问题类型",
                "context_params": {...}  # 从mock_data提取的所有参数
            }
        """
        # 从knowledge_base的mock_data中提取参数信息
        context_params = None
        kb_entities = None
        
        if knowledge_base and knowledge_base.get('mock_data'):
            context_params = self._extract_params_from_mock_data(
                knowledge_base['mock_data']
            )
            # 同时提取简化的entities用于向后兼容
            kb_entities = {
                'device': context_params.get('device_name'),
                'interface': context_params.get('interface_name')
            }
            kb_entities = {k: v for k, v in kb_entities.items() if v}  # 去掉None
        
        prompt = self._generate_extraction_prompt(question, context_params)
        
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
                        "content": "你是一个网络故障诊断专家，擅长分析问题并提取关键信息。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            
            # 解析JSON
            result = self._parse_json_response(result_text)
            
            # 如果有从mock_data提取的参数，添加到结果中
            if context_params:
                result['context_params'] = context_params
                if kb_entities:
                    result['entities'] = kb_entities
                print(f"✅ 从mock_data提取参数: {context_params}")
            
            # 添加原始问题
            result['original_question'] = question
            
            return result
            
        except Exception as e:
            print(f"目标提取失败: {e}")
            # 返回默认结构
            default_goal = self._create_default_goal(question)
            if context_params:
                default_goal['context_params'] = context_params
                if kb_entities:
                    default_goal['entities'] = kb_entities
            return default_goal
    
    def _extract_params_from_mock_data(self, mock_data: List[Dict]) -> Dict[str, Any]:
        """
        从mock_data中提取所有参数信息
        
        Args:
            mock_data: mock数据列表
            
        Returns:
            参数字典，包含所有在tool_request中出现的参数
        """
        params = {}
        
        if not mock_data or len(mock_data) == 0:
            return params
        
        # 遍历所有mock_data，收集所有参数（可能有多个工具调用）
        seen_params = {}
        for mock in mock_data:
            tool_request = mock.get('tool_request', {})
            for key, value in tool_request.items():
                # 如果是第一次见到这个参数，或者是更重要的参数（如device_name, interface_name）
                if key not in seen_params:
                    seen_params[key] = value
                elif key in ['device_name', 'interface_name', 'device', 'interface']:
                    # 优先保留设备名和接口名
                    seen_params[key] = value
        
        # 标准化参数名（统一使用device_name和interface_name）
        if 'device_name' in seen_params:
            params['device_name'] = seen_params['device_name']
        elif 'device' in seen_params:
            params['device_name'] = seen_params['device']
        
        if 'interface_name' in seen_params:
            params['interface_name'] = seen_params['interface_name']
        elif 'interface' in seen_params:
            params['interface_name'] = seen_params['interface']
        
        # 保留其他参数
        for key, value in seen_params.items():
            if key not in ['device', 'interface', 'device_name', 'interface_name']:
                params[key] = value
        
        return params
    
    def _generate_extraction_prompt(self, question: str, context_params: Optional[Dict] = None) -> str:
        """
        生成目标提取的prompt
        
        Args:
            question: 问题描述
            context_params: 从mock_data提取的参数信息（如果有）
        """
        # 如果有context_params，添加提示信息
        params_hint = ""
        if context_params:
            params_hint = f"""
注意：根据知识库，以下是相关的参数信息，请在entities中使用这些值：
{json.dumps(context_params, ensure_ascii=False, indent=2)}
"""
        
        prompt = f"""分析以下网络故障问题，提取诊断所需的关键信息：

问题描述：
{question}
{params_hint}
请分析并提取以下信息：
1. 主要诊断目标（要找出什么问题）
2. 需要检查的关键方面（如：接口状态、流量情况、错误统计、配置等）
3. 涉及的实体（设备名、接口名等）
4. 问题类型（如：丢包、连通性、性能、配置等）

以JSON格式输出：
```json
{{
  "main_goal": "找出导致丢包的原因",
  "key_aspects": [
    "接口状态检查",
    "流量分析",
    "错误统计",
    "光模块状态",
    "配置检查"
  ],
  "entities": {{
    "device": "serverleaf01_1_16.135",
    "interface": "10GE1/0/24"
  }},
  "problem_type": "丢包"
}}
```

注意：
- main_goal要明确、具体
- key_aspects列出所有可能需要检查的方面
- entities尽可能提取完整的设备名和接口名（如果上面提供了参数信息，必须使用准确的值）
- problem_type用简短的词概括问题类型
"""
        return prompt
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析LLM返回的JSON"""
        try:
            # 提取JSON代码块
            if '```json' in response:
                json_start = response.find('```json') + 7
                json_end = response.find('```', json_start)
                json_str = response[json_start:json_end].strip()
            elif '```' in response:
                json_start = response.find('```') + 3
                json_end = response.find('```', json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except Exception as e:
            print(f"解析JSON失败: {e}")
            print(f"响应内容: {response}")
            return {}
    
    def _create_default_goal(self, question: str) -> Dict[str, Any]:
        """创建默认目标（当提取失败时）"""
        return {
            "main_goal": "诊断网络故障并找出根本原因",
            "key_aspects": [
                "接口状态检查",
                "流量分析",
                "错误统计",
                "日志查看"
            ],
            "entities": {},
            "problem_type": "未知",
            "original_question": question
        }


def test_goal_extractor():
    """测试目标提取器"""
    print("=" * 80)
    print("测试目标提取器")
    print("=" * 80)
    
    # 测试用例
    test_questions = [
        "serverleaf01_1_16.135设备上10GE1/0/24接口发生丢包该如何处理？",
        "网络设备eth0接口流量异常，速度很慢",
        "交换机端口频繁up/down，怎么排查？"
    ]
    
    extractor = GoalExtractor(
        api_key="kw-qIdb2KBfLLBkk6YEJ1clWKKOctnHgWMjtfRJwQ2yTLBCXjMv",
        api_base="http://10.12.208.86:8502"
    )
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n测试{i}：{question}")
        print("-" * 80)
        
        goal = extractor.extract_goals(question)
        
        print(f"主要目标: {goal.get('main_goal')}")
        print(f"问题类型: {goal.get('problem_type')}")
        print(f"关键方面: {goal.get('key_aspects')}")
        print(f"实体: {goal.get('entities')}")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == '__main__':
    test_goal_extractor()
