"""
增强世界模型（Enhanced World Model）
支持多样化的工具响应生成
"""

import json
import openai
import random
from typing import Dict, List, Any, Optional


class EnhancedWorldModel:
    """增强世界模型 - 支持多样化响应"""
    
    def __init__(self, api_key: str, 
                 knowledge_base: Optional[Dict[str, Any]] = None,
                 api_base: str = None,
                 model: str = "gpt-4o-mini",
                 diversity_mode: str = 'medium'):
        """
        初始化增强世界模型
        
        Args:
            api_key: API密钥
            knowledge_base: 知识库（用于few-shot）
            api_base: API基础URL
            model: 模型名称
            diversity_mode: 多样性模式 ('low', 'medium', 'high')
        """
        self.api_key = api_key
        self.api_base = api_base or "http://10.12.208.86:8502"
        self.model = model
        self.diversity_mode = diversity_mode
        self.knowledge_base = knowledge_base or {}
        
        # 从知识库提取few-shot示例
        self.few_shot_examples = self._extract_few_shot_examples()
        
        # 定义响应变体
        self.response_variants = self._initialize_response_variants()
    
    def _extract_few_shot_examples(self) -> List[Dict[str, Any]]:
        """从知识库中提取few-shot示例"""
        examples = []
        
        if not self.knowledge_base:
            return examples
        
        # 从chain_of_action中提取
        chain_of_action = self.knowledge_base.get('chain_of_action', {})
        for key, actions in chain_of_action.items():
            for action in actions:
                if action.get('type') == 'tool_call':
                    example = {
                        'tool_name': action.get('tool_name'),
                        'tool_request': action.get('tool_request'),
                        'tool_response': action.get('tool_response')
                    }
                    examples.append(example)
        
        # 从mock_data中提取
        mock_data = self.knowledge_base.get('mock_data', [])
        for mock in mock_data:
            example = {
                'tool_name': mock.get('tool_name'),
                'tool_request': mock.get('tool_request'),
                'tool_response': mock.get('tool_response')
            }
            examples.append(example)
        
        return examples
    
    def _initialize_response_variants(self) -> Dict[str, List[Dict]]:
        """初始化响应变体定义"""
        return {
            # 通用变体（适用于所有工具）
            "default": [
                {
                    "variant_id": 1,
                    "name": "正常状态",
                    "severity": "normal",
                    "instructions": "生成表示正常、健康状态的响应。数值应在正常范围内，无异常告警。"
                },
                {
                    "variant_id": 2,
                    "name": "轻微异常",
                    "severity": "mild",
                    "instructions": "生成表示轻微异常的响应。数值略微偏离正常范围，可能有轻微告警。"
                },
                {
                    "variant_id": 3,
                    "name": "中等异常",
                    "severity": "moderate",
                    "instructions": "生成表示中等程度异常的响应。数值明显偏离正常范围，有明确告警。"
                },
                {
                    "variant_id": 4,
                    "name": "严重异常",
                    "severity": "severe",
                    "instructions": "生成表示严重异常的响应。数值严重偏离正常范围，有紧急告警。"
                }
            ]
        }
    
    def execute_tool(self, tool_name: str, tool_request: Dict[str, Any], 
                    context: Dict[str, Any] = None, 
                    run_id: int = None,
                    force_anomaly: bool = False) -> Dict[str, Any]:
        """
        执行工具，生成响应
        
        Args:
            tool_name: 工具名称
            tool_request: 工具请求参数
            context: 上下文信息
            run_id: 运行ID（用于控制多样性）
            force_anomaly: 是否强制生成异常数据
            
        Returns:
            工具响应
        """
        # 1. 选择响应变体
        if force_anomaly:
            # 强制选择异常变体
            print(f"      ⚠️  强制生成异常数据")
            variant = self._select_anomaly_variant(tool_name)
        else:
            variant = self._select_response_variant(tool_name, run_id)
        
        # 2. 生成响应
        response = self._generate_tool_response(
            tool_name, 
            tool_request, 
            variant,
            context or {}
        )
        
        return response
    
    def _select_response_variant(self, tool_name: str, run_id: Optional[int]) -> Dict:
        """
        选择响应变体
        
        Args:
            tool_name: 工具名称
            run_id: 运行ID
            
        Returns:
            选中的变体
        """
        # 获取变体列表（先尝试特定工具的，否则用default）
        variants = self.response_variants.get(tool_name, self.response_variants['default'])
        
        # 根据diversity_mode和run_id选择变体
        if self.diversity_mode == 'low':
            # 低多样性：主要选择正常状态
            weights = [0.7, 0.2, 0.1, 0.0] if len(variants) >= 4 else [1.0]
            
        elif self.diversity_mode == 'medium':
            # 中多样性：均匀分布，但正常状态权重稍高
            weights = [0.4, 0.3, 0.2, 0.1] if len(variants) >= 4 else [1.0]
            
        else:  # high
            # 高多样性：完全均匀分布
            weights = [0.25, 0.25, 0.25, 0.25] if len(variants) >= 4 else [1.0]
        
        # 如果指定了run_id，使用确定性选择（同一run_id总是选同一类变体）
        if run_id is not None:
            random.seed(run_id)
            variant = random.choices(variants[:len(weights)], weights=weights, k=1)[0]
            random.seed()  # 重置随机种子
        else:
            # 随机选择
            variant = random.choices(variants[:len(weights)], weights=weights, k=1)[0]
        
        return variant
    
    def _select_anomaly_variant(self, tool_name: str) -> Dict:
        """
        强制选择异常变体
        
        Args:
            tool_name: 工具名称
            
        Returns:
            异常变体
        """
        # 获取变体列表
        variants = self.response_variants.get(tool_name, self.response_variants['default'])
        
        # 优先选择标记为异常的变体
        for variant in variants:
            status = variant.get('status', '').lower()
            if status in ['down', 'error', 'abnormal', '异常', 'warning']:
                return variant
        
        # 如果没有明确的异常变体，选择第二或第三个（通常是异常状态）
        if len(variants) >= 3:
            return variants[2]  # 第三个通常是异常
        elif len(variants) >= 2:
            return variants[1]  # 第二个
        else:
            return variants[0]  # 只有一个就用第一个
    
    def _generate_tool_response(self, tool_name: str, tool_request: Dict, 
                               variant: Dict, context: Dict) -> Dict[str, Any]:
        """
        生成工具响应
        
        Args:
            tool_name: 工具名称
            tool_request: 工具请求
            variant: 响应变体
            context: 上下文
            
        Returns:
            工具响应
        """
        # 构建prompt
        prompt = self._build_generation_prompt(tool_name, tool_request, variant, context)
        
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
                        "content": "你是一个网络故障诊断系统的工具执行模拟器，擅长生成合理的工具响应数据。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            
            # 解析JSON响应
            tool_response = self._parse_json_response(result_text)
            
            return tool_response
            
        except Exception as e:
            print(f"世界模型执行失败: {e}")
            return {"error": "执行失败", "details": str(e)}
    
    def _build_generation_prompt(self, tool_name: str, tool_request: Dict,
                                 variant: Dict, context: Dict) -> str:
        """构建生成prompt"""
        
        # 查找相似的few-shot示例
        similar_examples = self._find_similar_examples(tool_name, limit=2)
        
        # 格式化few-shot示例
        examples_text = ""
        if similar_examples:
            examples_text = "参考示例：\n"
            for i, example in enumerate(similar_examples, 1):
                examples_text += f"\n示例{i}:\n"
                examples_text += f"工具: {example.get('tool_name')}\n"
                examples_text += f"请求: {json.dumps(example.get('tool_request', {}), ensure_ascii=False)}\n"
                examples_text += f"响应: {json.dumps(example.get('tool_response', {}), ensure_ascii=False)}\n"
        
        prompt = f"""你需要模拟一个网络诊断工具的执行结果。

工具名称: {tool_name}
请求参数: {json.dumps(tool_request, ensure_ascii=False, indent=2)}

上下文信息: {json.dumps(context, ensure_ascii=False, indent=2)}

{examples_text}

响应要求:
- 状态类型: {variant['name']}
- 严重程度: {variant['severity']}
- 具体要求: {variant['instructions']}

请生成一个合理的工具响应。响应应该：
1. 符合工具的功能和参数
2. 与上下文信息一致
3. 符合"{variant['name']}"的特征
4. 如果有参考示例，保持相似的格式和字段
5. 数据要合理、具体、可信

以JSON格式输出（只输出JSON，不要其他文字）：
```json
{{
  "字段1": "值1",
  "字段2": "值2",
  ...
}}
```

注意：
- 如果是查询类工具，返回查询到的数据
- 如果是分析类工具，返回分析结果和结论
- 数值要符合网络设备的实际情况
- 根据"{variant['name']}"调整数值（正常/异常）
"""
        return prompt
    
    def _find_similar_examples(self, tool_name: str, limit: int = 2) -> List[Dict]:
        """查找相似的few-shot示例"""
        # 优先返回同名工具的示例
        exact_matches = [ex for ex in self.few_shot_examples 
                        if ex.get('tool_name') == tool_name]
        
        if exact_matches:
            return exact_matches[:limit]
        
        # 否则返回其他示例
        return self.few_shot_examples[:limit]
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        解析JSON响应，确保总是返回dict
        
        处理策略：
        1. 如果解析出dict，直接返回
        2. 如果解析出list：
           - 如果list只有1个dict元素，返回该dict
           - 否则包装成 {"data": list}
        3. 如果解析出其他类型，包装成 {"value": parsed_data}
        4. 如果解析失败，返回错误dict
        """
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
            
            # 解析JSON
            parsed = json.loads(json_str)
            
            # 类型检查和转换
            if isinstance(parsed, dict):
                # 已经是dict，直接返回
                return parsed
            elif isinstance(parsed, list):
                # 是list，需要转换
                if len(parsed) == 1 and isinstance(parsed[0], dict):
                    # list只有1个dict元素，返回该dict
                    print(f"⚠️  LLM返回了单元素list，自动提取: {list(parsed[0].keys())}")
                    return parsed[0]
                else:
                    # 多个元素或非dict元素，包装成dict
                    print(f"⚠️  LLM返回了list（{len(parsed)}个元素），包装为dict")
                    return {"data": parsed, "_type": "list"}
            else:
                # 其他类型（string, number等），包装成dict
                print(f"⚠️  LLM返回了{type(parsed).__name__}类型，包装为dict")
                return {"value": parsed, "_type": type(parsed).__name__}
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            print(f"   原始响应: {response[:200]}...")
            return {"error": "JSON解析失败", "raw": response[:500]}
        except Exception as e:
            print(f"❌ 解析异常: {e}")
            return {"error": "解析失败", "details": str(e), "raw": response[:500]}
    
    def set_diversity_mode(self, mode: str):
        """设置多样性模式"""
        if mode in ['low', 'medium', 'high']:
            self.diversity_mode = mode
        else:
            print(f"⚠️  无效的多样性模式: {mode}")


def test_enhanced_world_model():
    """测试增强世界模型"""
    print("=" * 80)
    print("测试增强世界模型")
    print("=" * 80)
    
    # 加载知识库
    try:
        with open('/mnt/user-data/uploads/workflow.json', 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
    except:
        knowledge_base = {}
    
    # 创建世界模型
    world_model = EnhancedWorldModel(
        api_key="kw-qIdb2KBfLLBkk6YEJ1clWKKOctnHgWMjtfRJwQ2yTLBCXjMv",
        knowledge_base=knowledge_base,
        api_base="http://10.12.208.86:8502",
        diversity_mode='medium'
    )
    
    # 测试工具
    tool_name = "query_interface_info"
    tool_request = {
        "device_name": "serverleaf01_1_16.135",
        "interface_name": "10GE1/0/24"
    }
    context = {}
    
    print("\n测试多样化响应（3次，不同run_id）：")
    print("-" * 80)
    
    for run_id in range(3):
        print(f"\nRun {run_id + 1}:")
        
        response = world_model.execute_tool(
            tool_name, 
            tool_request, 
            context,
            run_id=run_id
        )
        
        print(json.dumps(response, ensure_ascii=False, indent=2))
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == '__main__':
    test_enhanced_world_model()
