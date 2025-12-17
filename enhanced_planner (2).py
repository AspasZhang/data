"""
增强规划器（Enhanced Planner）
支持自由探索和多样化的工具选择策略
"""

import json
import openai
import random
from typing import Dict, List, Any, Optional
from tool_manager import ToolManager
from state_manager import StateManager


class EnhancedPlanner:
    """增强规划器 - 支持多样化探索"""
    
    def __init__(self, tool_manager: ToolManager, 
                 api_key: str, 
                 api_base: str = None,
                 model: str = "gpt-4o-mini",
                 exploration_mode: str = 'balanced'):
        """
        初始化增强规划器
        
        Args:
            tool_manager: 工具管理器
            api_key: API密钥
            api_base: API基础URL
            model: 模型名称
            exploration_mode: 探索模式 ('greedy', 'balanced', 'exploratory')
        """
        self.tool_manager = tool_manager
        self.api_key = api_key
        self.api_base = api_base or "http://10.12.208.86:8502"
        self.model = model
        self.exploration_mode = exploration_mode
    
    def select_next_tool(self, state: StateManager, goal: Dict[str, Any], 
                        temperature: float = 0.7) -> Dict[str, Any]:
        """
        选择下一步要使用的工具
        
        Args:
            state: 当前状态
            goal: 诊断目标
            temperature: 温度参数（控制随机性）
            
        Returns:
            {
                "tool_name": "工具名称",
                "tool_request": {...},
                "reasoning": "选择理由",
                "expected_info": "期望信息"
            }
        """
        # 1. 生成prompt
        prompt = self._generate_planning_prompt(state, goal)
        
        # 2. 调用LLM获取候选工具
        candidates = self._get_tool_candidates(prompt, temperature, top_k=3)
        
        if not candidates:
            return {"error": "未能生成工具选择"}
        
        # 3. 应用exploration策略选择最终工具
        selected = self._apply_exploration_strategy(candidates, state)
        
        return selected
    
    def _generate_planning_prompt(self, state: StateManager, goal: Dict[str, Any]) -> str:
        """生成规划prompt"""
        
        # 获取可用工具列表
        available_tools = self.tool_manager.get_tools_compact_list()
        
        # 格式化已执行的工具
        history = state.format_recent_history(n=5)
        
        # 格式化观察结果
        observations = state.format_observations()
        
        # 格式化发现
        findings = state.format_findings()
        
        # 已使用的工具（用于参考）
        used_tools = list(state.tool_usage_count.keys())
        
        # 提取context_params（从mock_data提取的参数）
        context_params = goal.get('context_params', {})
        
        # 如果有context_params，格式化展示
        params_section = ""
        if context_params:
            params_section = self._format_context_params(context_params)
        
        prompt = f"""你是一个网络故障诊断专家，正在进行故障排查。

【诊断目标】
主要目标: {goal.get('main_goal', '未知')}
问题类型: {goal.get('problem_type', '未知')}
需要关注的方面: {', '.join(goal.get('key_aspects', []))}

{params_section}

【已执行的步骤】（最近5步）
{history}

【当前观察结果】
{observations}

【当前发现】
{findings}

【已使用过的工具】
{', '.join(used_tools) if used_tools else '无'}

【可用工具列表】
{available_tools}

请根据当前诊断进度，选择下一步最合适的工具。

选择要求：
1. 工具必须从【可用工具列表】中选择
2. 可以重复使用已用过的工具（如需多次查询）
3. 优先选择能推进诊断、获取关键信息的工具
4. 考虑诊断的逻辑顺序和因果关系
5. 如果已有足够信息可以得出结论，选择能验证或补充结论的工具
6. **重要：填写tool_request参数时，请根据【相关参数】中的信息填写准确的设备名、接口名等**

以JSON格式输出（只输出JSON，不要有其他文字）：
```json
{{
  "tool_name": "从可用工具列表中选择的工具名称",
  "tool_request": {{
    "device_name": "设备名（参考上面的参数信息）",
    "interface_name": "接口名（参考上面的参数信息）",
    "其他参数": "根据工具需要填写"
  }},
  "reasoning": "为什么选择这个工具，期望解决什么问题",
  "expected_info": "期望从这个工具获得什么信息"
}}
```

注意：
- tool_name必须精确匹配工具列表中的名称
- tool_request中的device_name和interface_name等参数请使用【相关参数】中提供的准确值
- 如果工具需要其他参数，根据诊断上下文合理填写
- reasoning要简洁明了
"""
        return prompt
    
    def _format_context_params(self, context_params: Dict[str, Any]) -> str:
        """
        格式化context_params为易读的【相关参数】部分
        
        Args:
            context_params: 从mock_data提取的参数字典
            
        Returns:
            格式化的字符串
        """
        if not context_params:
            return ""
        
        # 参数名称映射（中文）
        param_names = {
            'device_name': '设备名',
            'device': '设备名',
            'interface_name': '接口名',
            'interface': '接口名',
            'port': '端口',
            'vlan': 'VLAN',
            'ip': 'IP地址',
            'hostname': '主机名',
            'start_time': '开始时间',
            'end_time': '结束时间',
            'filter_condition1': '过滤条件1',
            'filter_condition2': '过滤条件2',
        }
        
        lines = ["【相关参数】（工具调用时请使用这些参数）"]
        
        for key, value in context_params.items():
            # 获取中文名称，如果没有则使用原key
            display_name = param_names.get(key, key)
            lines.append(f"{display_name}: {value}")
        
        return '\n'.join(lines)
    
    def _format_entities(self, entities: Dict[str, Any]) -> str:
        """
        格式化entities信息
        
        Args:
            entities: 实体字典，如 {"device": "serverleaf01", "interface": "10GE1/0/24"}
            
        Returns:
            格式化的字符串
        """
        if not entities:
            return "无特定实体信息"
        
        lines = []
        
        # 常见的实体类型及其中文名称
        entity_names = {
            'device': '设备名',
            'device_name': '设备名',
            'interface': '接口名',
            'interface_name': '接口名',
            'port': '端口',
            'vlan': 'VLAN',
            'ip': 'IP地址',
            'hostname': '主机名',
        }
        
        for key, value in entities.items():
            # 获取中文名称，如果没有则使用原key
            display_name = entity_names.get(key, key)
            lines.append(f"{display_name}: {value}")
        
        return '\n'.join(lines) if lines else "无特定实体信息"
    
    def _get_tool_candidates(self, prompt: str, temperature: float, top_k: int = 3) -> List[Dict]:
        """
        调用LLM获取候选工具
        
        Args:
            prompt: 规划prompt
            temperature: 温度参数
            top_k: 返回top K个候选
            
        Returns:
            候选工具列表
        """
        candidates = []
        
        try:
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
            
            # 调用多次以获取多个候选
            for i in range(top_k):
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个网络故障诊断专家，擅长选择合适的诊断工具。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temperature + i * 0.1,  # 逐渐增加温度以获得多样性
                    max_tokens=500
                )
                
                result_text = response.choices[0].message.content
                
                # 解析JSON
                candidate = self._parse_json_response(result_text)
                
                if candidate and 'tool_name' in candidate:
                    # 验证工具是否有效
                    if self.tool_manager.is_valid_tool(candidate['tool_name']):
                        candidates.append(candidate)
                    else:
                        print(f"  ⚠️  LLM生成了无效工具: {candidate.get('tool_name')}")
            
            return candidates
            
        except Exception as e:
            print(f"规划器调用失败: {e}")
            return []
    
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
            return {}
    
    def _apply_exploration_strategy(self, candidates: List[Dict], state: StateManager) -> Dict:
        """
        应用exploration策略选择最终工具
        
        Args:
            candidates: 候选工具列表
            state: 当前状态
            
        Returns:
            选中的工具
        """
        if not candidates:
            return {"error": "无候选工具"}
        
        # 给每个候选打分
        for candidate in candidates:
            tool_name = candidate['tool_name']
            
            # 基础分数（假设LLM返回的顺序代表质量）
            base_score = len(candidates) - candidates.index(candidate)
            
            # Exploration bonus: 少用的工具加分
            usage_count = state.get_tool_usage_count(tool_name)
            exploration_bonus = 2.0 / (usage_count + 1)  # 用得越少，bonus越高
            
            # 总分
            candidate['score'] = base_score + exploration_bonus
        
        # 根据exploration_mode选择
        if self.exploration_mode == 'greedy':
            # 总是选择得分最高的
            candidates.sort(key=lambda x: x['score'], reverse=True)
            selected = candidates[0]
            
        elif self.exploration_mode == 'balanced':
            # 70%选top1, 30%从top2和top3中随机选
            candidates.sort(key=lambda x: x['score'], reverse=True)
            if random.random() < 0.7:
                selected = candidates[0]
            else:
                selected = random.choice(candidates[1:]) if len(candidates) > 1 else candidates[0]
        
        else:  # exploratory
            # 根据分数进行加权随机选择
            candidates.sort(key=lambda x: x['score'], reverse=True)
            weights = [c['score'] for c in candidates]
            selected = random.choices(candidates, weights=weights, k=1)[0]
        
        print(f"  📍 选择模式: {self.exploration_mode}")
        print(f"  🎯 选中工具: {selected['tool_name']}")
        print(f"  💭 理由: {selected.get('reasoning', '无')}")
        
        return selected
    
    def set_exploration_mode(self, mode: str):
        """
        设置探索模式
        
        Args:
            mode: 'greedy', 'balanced', 'exploratory'
        """
        if mode in ['greedy', 'balanced', 'exploratory']:
            self.exploration_mode = mode
        else:
            print(f"⚠️  无效的探索模式: {mode}，保持当前模式: {self.exploration_mode}")


def test_enhanced_planner():
    """测试增强规划器"""
    print("=" * 80)
    print("测试增强规划器")
    print("=" * 80)
    
    # 初始化组件
    from tool_manager import ToolManager
    
    tool_manager = ToolManager('/mnt/user-data/outputs/available_tools.txt')
    
    planner = EnhancedPlanner(
        tool_manager=tool_manager,
        api_key="kw-qIdb2KBfLLBkk6YEJ1clWKKOctnHgWMjtfRJwQ2yTLBCXjMv",
        api_base="http://10.12.208.86:8502",
        exploration_mode='balanced'
    )
    
    # 创建测试状态
    state = StateManager()
    
    # 创建测试目标
    goal = {
        "main_goal": "找出导致丢包的原因",
        "problem_type": "丢包",
        "key_aspects": ["接口状态", "流量分析", "错误统计"],
        "entities": {
            "device": "serverleaf01_1_16.135",
            "interface": "10GE1/0/24"
        }
    }
    
    # 测试选择工具
    print("\n测试工具选择（3次）：")
    print("-" * 80)
    
    for i in range(3):
        print(f"\n第 {i+1} 次选择:")
        
        plan = planner.select_next_tool(state, goal, temperature=0.7)
        
        if 'error' in plan:
            print(f"❌ 错误: {plan['error']}")
        else:
            print(f"✅ 工具: {plan['tool_name']}")
            print(f"   参数: {json.dumps(plan['tool_request'], ensure_ascii=False)}")
            
            # 模拟执行
            state.add_execution(
                plan['tool_name'],
                plan['tool_request'],
                {"mock": "response"},
                plan.get('reasoning', '')
            )
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == '__main__':
    test_enhanced_planner()
