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
        从问题中提取诊断目标，优先使用knowledge_base中的entities
        
        Args:
            question: 问题描述
            knowledge_base: 知识库（可选），如果包含entities则优先使用
            
        Returns:
            {
                "main_goal": "主要目标",
                "key_aspects": ["方面1", "方面2", ...],
                "entities": {"设备": "...", "接口": "..."},
                "problem_type": "问题类型"
            }
        """
        # 检查knowledge_base中是否有entities信息
        kb_entities = None
        if knowledge_base:
            # 尝试从多个可能的位置获取entities
            kb_entities = knowledge_base.get('entities')
            
            # 如果没有直接的entities字段，尝试从mock_data中提取
            if not kb_entities and knowledge_base.get('mock_data'):
                kb_entities = self._extract_entities_from_mock_data(
                    knowledge_base['mock_data']
                )
        
        prompt = self._generate_extraction_prompt(question, kb_entities)
        
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
            
            # 如果有kb_entities，确保使用它们（覆盖LLM提取的）
            if kb_entities:
                result['entities'] = kb_entities
                print(f"✅ 使用knowledge base中的entities: {kb_entities}")
            
            # 添加原始问题
            result['original_question'] = question
            
            return result
            
        except Exception as e:
            print(f"目标提取失败: {e}")
            # 返回默认结构，如果有kb_entities则使用
            default_goal = self._create_default_goal(question)
            if kb_entities:
                default_goal['entities'] = kb_entities
            return default_goal
    
    def _extract_entities_from_mock_data(self, mock_data: List[Dict]) -> Dict[str, str]:
        """
        从mock_data中提取entities
        
        Args:
            mock_data: mock数据列表
            
        Returns:
            entities字典
        """
        entities = {}
        
        if not mock_data or len(mock_data) == 0:
            return entities
        
        # 从第一个mock_data的tool_request中提取
        first_mock = mock_data[0]
        tool_request = first_mock.get('tool_request', {})
        
        # 提取常见的实体字段
        if 'device_name' in tool_request:
            entities['device'] = tool_request['device_name']
        elif 'device' in tool_request:
            entities['device'] = tool_request['device']
            
        if 'interface_name' in tool_request:
            entities['interface'] = tool_request['interface_name']
        elif 'interface' in tool_request:
            entities['interface'] = tool_request['interface']
        
        return entities
    
    def _generate_extraction_prompt(self, question: str, kb_entities: Optional[Dict] = None) -> str:
        """
        生成目标提取的prompt
        
        Args:
            question: 问题描述
            kb_entities: 知识库中的entities（如果有）
        """
        # 如果有kb_entities，添加提示信息
        entities_hint = ""
        if kb_entities:
            entities_hint = f"""
注意：已知的实体信息如下，请在输出中使用这些准确的值：
{json.dumps(kb_entities, ensure_ascii=False, indent=2)}
"""
        
        prompt = f"""分析以下网络故障问题，提取诊断所需的关键信息：

问题描述：
{question}
{entities_hint}
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
- entities尽可能提取完整的设备名和接口名（如果上面提供了已知实体信息，必须使用准确的值）
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
