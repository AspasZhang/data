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
                        temperature: float = 0.7, known_entities: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """
        选择下一步要使用的工具（增强版：支持重试和工具验证）
        
        Args:
            state: 当前状态
            goal: 诊断目标
            temperature: 温度参数（控制随机性）
            known_entities: 已知实体列表，如{'interfaces': ['eth0', 'eth1']}
            
        Returns:
            {
                "tool_name": "工具名称",
                "tool_request": {...},
                "reasoning": "选择理由",
                "expected_info": "期望信息"
            }
        """
        # 1. 生成prompt
        prompt = self._generate_planning_prompt(state, goal, known_entities)
        
        # 2. 多次尝试获取有效工具（最多重试3次）
        max_retries = 3
        for retry in range(max_retries):
            # 调用LLM获取候选工具
            candidates = self._get_tool_candidates(prompt, temperature + retry * 0.1, top_k=3)
            
            if candidates:
                # 3. 应用exploration策略选择最终工具
                selected = self._apply_exploration_strategy(candidates, state)
                
                # 验证选择的工具是否有效
                if 'tool_name' in selected and self.tool_manager.is_valid_tool(selected['tool_name']):
                    # 4. 验证并修正参数
                    selected = self._validate_and_fix_parameters(selected, state, goal)
                    return selected
                else:
                    if retry < max_retries - 1:
                        print(f"  ⚠️  第{retry + 1}次尝试：选择的工具 '{selected.get('tool_name', 'unknown')}' 无效，重试...")
            else:
                if retry < max_retries - 1:
                    print(f"  ⚠️  第{retry + 1}次尝试：未能生成有效候选，重试...")
        
        # 如果所有重试都失败，尝试工具名称纠正
        print(f"  🔄 所有重试失败，尝试智能纠正...")
        corrected_tool = self._try_tool_name_correction(state, goal, temperature)
        if corrected_tool:
            # 验证并修正参数
            corrected_tool = self._validate_and_fix_parameters(corrected_tool, state, goal)
            return corrected_tool
        
        # 最后的备选方案：选择一个默认工具
        print(f"  ⚠️  选择备用工具...")
        fallback = self._select_fallback_tool(state, goal)
        # 验证并修正参数
        fallback = self._validate_and_fix_parameters(fallback, state, goal)
        return fallback
    
    def _generate_planning_prompt(self, state, goal, known_entities=None):
        """Generate planning prompt with anti-repeat and comprehensive-call guidance."""

        diagnostic_context = state.get_diagnostic_context()
        diagnostic_chain   = state.format_diagnostic_chain()
        findings           = state.format_findings()
        context_params     = goal.get('context_params', {})
        params_section     = self._format_context_params(context_params) if context_params else ""

        # ── 已调用工具 + 参数明细（避免重复的核心）────────────────────────
        call_history_lines = []
        for ex in state.executed_tools:
            req_str = json.dumps(ex.get('tool_request', {}), ensure_ascii=False)
            call_history_lines.append(f"  - {ex['tool_name']}({req_str})")
        call_history = "\n".join(call_history_lines) if call_history_lines else "  （暂无）"

        # ── 已知实体 ─────────────────────────────────────────────────────
        entities_section = ""
        if known_entities:
            ent_lines = []
            for etype, ents in known_entities.items():
                if ents:
                    ent_lines.append(f"  {etype}: {', '.join(ents)}")
            if ent_lines:
                entities_section = "【已知实体】\n" + "\n".join(ent_lines)

        # ── 可用参数值 ───────────────────────────────────────────────────
        all_known = self._extract_known_parameters(state, goal)
        param_lines = []
        if all_known['device_name']:
            param_lines.append(f"  设备名: {', '.join(all_known['device_name'][:5])}")
        if all_known['interface_name']:
            param_lines.append(f"  接口名: {', '.join(all_known['interface_name'][:10])}")
        if all_known['ip']:
            param_lines.append(f"  IP地址: {', '.join(all_known['ip'][:5])}")
        available_params_section = ""
        if param_lines:
            available_params_section = (
                "【可用参数值】（填写 tool_request 时必须从此处精确复制）\n"
                + "\n".join(param_lines)
                + "\n⚠️ 禁止使用 \"设备\"、\"接口\" 等描述性占位符！"
            )

        # ── 上一步摘要 ───────────────────────────────────────────────────
        last_step_info = ""
        if state.executed_tools:
            lt = state.executed_tools[-1]
            last_step_info = (
                "【上一步结果】\n"
                f"工具: {lt['tool_name']}\n"
                f"观察: {json.dumps(lt['tool_response'], ensure_ascii=False)[:300]}"
            )

        available_tools_detailed = self.tool_manager.get_tools_with_parameters()
        all_tool_names = self.tool_manager.get_all_tool_names()
        tools_name_list = "\n".join(f"  - {n}" for n in all_tool_names)

        prompt = f"""你是网络故障诊断专家，正在进行故障排查。根据诊断进展选择下一步最合适的工具。

【诊断目标】
主要目标: {goal.get('main_goal', '未知')}
问题类型: {goal.get('problem_type', '未知')}
关注方面: {', '.join(goal.get('key_aspects', []))}

{params_section}

【诊断上下文】
{diagnostic_context}
{last_step_info}
{entities_section}
{available_params_section}

【诊断逻辑链】
{diagnostic_chain if diagnostic_chain != "暂无诊断逻辑链" else "第一步，开始诊断"}

【当前发现】
{findings}

【已调用工具明细】（以下组合禁止重复调用）
{call_history}

【所有可用工具精确名称】
{tools_name_list}

【可用工具及参数说明】
{available_tools_detailed}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【选择要求】

1. **禁止重复**：【已调用工具明细】中出现过的 工具+参数 组合，绝对不能再次选择。
   如需对同一设备进行不同维度检查，选择不同的工具。

2. **一次拿全**：调用一个工具时，尽量通过合理参数一次获取完整信息，避免同工具多次小批量调用。
   例如：查接口状态时，直接用已知的具体接口名，一次获取该接口所有指标。

3. **参数必须具体**：
   - ✅ device_name: "aggrleaf02_2_20.45"
   - ✅ interface_name: "40GE2/2/5"
   - ❌ device_name: "设备"（描述词）
   - 只能从【可用参数值】和【已知实体】中精确复制，不得自造。

4. **承接上一步**：说明这一步基于什么发现，期望验证什么假设。

5. **如需批量**：在 reasoning 中写明"逐一检查所有XXX"，系统会自动批量执行。

以 JSON 输出（只输出 JSON）：
```json
{{
  "tool_name": "从上面工具列表精确选择",
  "tool_request": {{"参数名": "具体参数值"}},
  "reasoning": "这一步的诊断思路（基于上一步XXX，现在需要检查YYY）",
  "expected_outcome": "若发现A则说明...；若发现B则说明...",
  "next_focus": "执行后下一步应关注什么"
}}
```"""
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
    
    def _try_tool_name_correction(self, state: StateManager, goal: Dict[str, Any], temperature: float) -> Dict[str, Any]:
        """
        尝试纠正LLM返回的工具名称
        
        当LLM返回的工具名称与实际工具名称相似时，尝试自动纠正
        
        Returns:
            纠正后的工具信息，如果无法纠正则返回None
        """
        # 获取所有可用工具名称
        all_tools = self.tool_manager.get_all_tool_names()
        
        # 使用更强的prompt，列出所有精确的工具名称
        tools_list = "\n".join([f"- {tool}" for tool in all_tools])
        
        strict_prompt = f"""你是网络故障诊断专家。下面是**所有可用工具的精确名称**：

{tools_list}

**重要：你必须从上述列表中精确选择一个工具名称，不要添加或修改任何字符。**

【诊断目标】
{goal.get('main_goal', '未知')}

【当前状态】
已执行步骤: {state.step_count}
已使用工具: {', '.join(state.tool_usage_count.keys()) if state.tool_usage_count else '无'}

请选择下一个工具，只输出JSON：
```json
{{
  "tool_name": "从上面列表中精确选择一个",
  "tool_request": {{}},
  "reasoning": "简短说明"
}}
```"""
        
        try:
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你必须从提供的工具列表中精确选择，不要编造工具名称。"},
                    {"role": "user", "content": strict_prompt}
                ],
                temperature=0.3,  # 降低温度以提高准确性
                max_tokens=300
            )
            
            result_text = response.choices[0].message.content
            candidate = self._parse_json_response(result_text)
            
            if candidate and 'tool_name' in candidate:
                if self.tool_manager.is_valid_tool(candidate['tool_name']):
                    print(f"  ✅ 成功纠正并选择工具: {candidate['tool_name']}")
                    return candidate
                else:
                    # 尝试模糊匹配
                    corrected_name = self._fuzzy_match_tool_name(candidate['tool_name'], all_tools)
                    if corrected_name:
                        print(f"  ✅ 工具名称纠正: '{candidate['tool_name']}' → '{corrected_name}'")
                        candidate['tool_name'] = corrected_name
                        return candidate
            
        except Exception as e:
            print(f"  ❌ 工具纠正失败: {e}")
        
        return None
    
    def _fuzzy_match_tool_name(self, invalid_name: str, valid_tools: List[str]) -> str:
        """
        模糊匹配工具名称
        
        Args:
            invalid_name: 无效的工具名称
            valid_tools: 有效工具名称列表
            
        Returns:
            最相似的有效工具名称，如果相似度太低则返回None
        """
        if not invalid_name or not valid_tools:
            return None
        
        invalid_lower = invalid_name.lower().strip()
        
        # 完全匹配（不区分大小写）
        for tool in valid_tools:
            if tool.lower() == invalid_lower:
                return tool
        
        # 包含匹配
        for tool in valid_tools:
            if invalid_lower in tool.lower() or tool.lower() in invalid_lower:
                return tool
        
        # Levenshtein距离匹配（简化版）
        def simple_similarity(s1, s2):
            """简单的相似度计算"""
            s1, s2 = s1.lower(), s2.lower()
            if len(s1) > len(s2):
                s1, s2 = s2, s1
            
            # 计算共同字符数
            common = sum(c in s2 for c in s1)
            return common / max(len(s1), len(s2))
        
        # 找最相似的
        best_match = None
        best_score = 0.0
        
        for tool in valid_tools:
            score = simple_similarity(invalid_name, tool)
            if score > best_score and score > 0.6:  # 相似度阈值
                best_score = score
                best_match = tool
        
        return best_match
    
    def _validate_and_fix_parameters(self, plan: Dict, state: StateManager, goal: Dict) -> Dict:
        """
        验证并修正工具参数
        
        Args:
            plan: 工具计划
            state: 状态管理器
            goal: 目标信息
            
        Returns:
            修正后的plan
        """
        tool_request = plan.get('tool_request', {})
        
        # 收集所有已知的有效参数值
        known_params = self._extract_known_parameters(state, goal)
        
        # 验证每个参数
        invalid_params = []
        for param_name, param_value in tool_request.items():
            if not self._is_valid_parameter_value(param_value):
                invalid_params.append((param_name, param_value))
        
        # 如果有无效参数，尝试修正
        if invalid_params:
            print(f"  ⚠️  发现无效参数: {invalid_params}")
            
            for param_name, invalid_value in invalid_params:
                # 尝试从已知参数中找到对应的值
                valid_value = self._find_valid_parameter(param_name, known_params)
                
                if valid_value:
                    print(f"  ✅ 参数纠正: {param_name}: '{invalid_value}' → '{valid_value}'")
                    tool_request[param_name] = valid_value
                else:
                    print(f"  ❌ 无法纠正参数: {param_name}={invalid_value}")
        
        plan['tool_request'] = tool_request
        return plan
    
    def _is_valid_parameter_value(self, value: str) -> bool:
        """
        检查参数值是否有效
        
        无效的参数值包括：
        - 中文描述：设备、接口、某个IP所在设备等
        - 占位符：device、interface等通用词
        - 空值
        
        Returns:
            True表示有效，False表示无效
        """
        if not value or not isinstance(value, str):
            return False
        
        value_lower = value.lower().strip()
        
        # 检查是否包含明显的占位符或描述性文字
        invalid_patterns = [
            '设备', '接口', '所在', '某个', '这个', '那个',
            'device', 'interface', 'ip', 'port', 'vlan',
            '未知', '待定', 'unknown', 'tbd',
            '...', 'xxx', 'yyy'
        ]
        
        # 如果值就是这些词，或者只包含这些词，则无效
        for pattern in invalid_patterns:
            if value_lower == pattern:
                return False
            # 如果值很短且包含这些描述词，也认为无效
            if len(value) < 15 and pattern in value_lower and '/' not in value and '.' not in value:
                return False
        
        return True
    
    def _extract_known_parameters(self, state: StateManager, goal: Dict) -> Dict[str, List[str]]:
        """
        从state和goal中提取所有已知的有效参数值
        
        Returns:
            {
                'device_name': ['aggrleaf02_2_20.45', 'spine01', ...],
                'interface_name': ['40GE2/2/5', '10GE1/0/24', ...],
                'ip': ['192.168.1.1', ...]
            }
        """
        known_params = {
            'device_name': [],
            'interface_name': [],
            'ip': [],
            'vlan': [],
            'hostname': []
        }
        
        # 从goal中提取
        goal_entities = goal.get('entities', {})
        context_params = goal.get('context_params', {})
        
        for key, value in {**goal_entities, **context_params}.items():
            if 'device' in key.lower():
                if value and self._is_valid_parameter_value(str(value)):
                    known_params['device_name'].append(str(value))
            elif 'interface' in key.lower():
                if value and self._is_valid_parameter_value(str(value)):
                    known_params['interface_name'].append(str(value))
            elif 'ip' in key.lower():
                if value and self._is_valid_parameter_value(str(value)):
                    known_params['ip'].append(str(value))
        
        # 从历史执行中提取
        for execution in state.executed_tools:
            tool_request = execution.get('tool_request', {})
            tool_response = execution.get('tool_response', {})
            
            # 从tool_request中提取
            for key, value in tool_request.items():
                if value and self._is_valid_parameter_value(str(value)):
                    if 'device' in key.lower():
                        known_params['device_name'].append(str(value))
                    elif 'interface' in key.lower():
                        known_params['interface_name'].append(str(value))
                    elif 'ip' in key.lower():
                        known_params['ip'].append(str(value))
            
            # 从tool_response中提取（可能返回了新的实体）
            if isinstance(tool_response, dict):
                for key, value in tool_response.items():
                    if value and self._is_valid_parameter_value(str(value)):
                        if key in ['设备', 'device', 'device_name']:
                            known_params['device_name'].append(str(value))
                        elif key in ['接口', 'interface', 'interface_name']:
                            known_params['interface_name'].append(str(value))
                        elif key in ['IP', 'IP地址', 'ip']:
                            known_params['ip'].append(str(value))
            
            elif isinstance(tool_response, list):
                for item in tool_response:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if value and self._is_valid_parameter_value(str(value)):
                                if key in ['设备', 'device', 'device_name']:
                                    known_params['device_name'].append(str(value))
                                elif key in ['接口', 'interface', 'interface_name']:
                                    known_params['interface_name'].append(str(value))
                                elif key in ['IP', 'IP地址', 'ip', 'IP地址/掩码']:
                                    # 提取IP（去掉掩码）
                                    ip_value = str(value).split('/')[0]
                                    if self._is_valid_parameter_value(ip_value):
                                        known_params['ip'].append(ip_value)
        
        # 去重
        for key in known_params:
            known_params[key] = list(set(known_params[key]))
        
        return known_params
    
    def _find_valid_parameter(self, param_name: str, known_params: Dict[str, List[str]]) -> str:
        """
        从已知参数中查找对应的有效值
        
        Args:
            param_name: 参数名称（如device_name, interface_name）
            known_params: 已知的有效参数字典
            
        Returns:
            有效的参数值，如果找不到返回空字符串
        """
        param_name_lower = param_name.lower()
        
        # 根据参数名称匹配对应的类型
        if 'device' in param_name_lower:
            if known_params['device_name']:
                return known_params['device_name'][0]
        elif 'interface' in param_name_lower:
            if known_params['interface_name']:
                return known_params['interface_name'][0]
        elif 'ip' in param_name_lower:
            if known_params['ip']:
                return known_params['ip'][0]
        elif 'vlan' in param_name_lower:
            if known_params['vlan']:
                return known_params['vlan'][0]
        elif 'hostname' in param_name_lower:
            if known_params['hostname']:
                return known_params['hostname'][0]
        
        return ""
    
    def _select_fallback_tool(self, state: StateManager, goal: Dict[str, Any]) -> Dict[str, Any]:
        """
        选择一个备用工具（当所有尝试都失败时）
        
        策略：
        1. 如果是第一步，选择一个通用的查询工具
        2. 否则，选择一个还没用过或用得最少的工具
        
        Returns:
            备用工具信息
        """
        all_tools = self.tool_manager.get_all_tool_names()
        
        if not all_tools:
            return {"error": "没有可用工具"}
        
        # 如果是第一步，选择一个通用查询工具
        if state.step_count == 0:
            # 优先选择查询类工具
            query_tools = [t for t in all_tools if 'query' in t.lower() or 'get' in t.lower() or 'show' in t.lower()]
            if query_tools:
                selected_tool = query_tools[0]
            else:
                selected_tool = all_tools[0]
        else:
            # 选择用得最少的工具
            unused_tools = [t for t in all_tools if t not in state.tool_usage_count]
            
            if unused_tools:
                selected_tool = unused_tools[0]
            else:
                # 所有工具都用过了，选择用得最少的
                min_usage = min(state.tool_usage_count.values())
                least_used = [t for t, count in state.tool_usage_count.items() if count == min_usage]
                selected_tool = least_used[0]
        
        print(f"  📌 选择备用工具: {selected_tool}")
        
        # 构建工具请求（从goal中提取参数）
        tool_request = {}
        context_params = goal.get('context_params', {})
        entities = goal.get('entities', {})
        
        # 合并参数
        all_params = {**context_params, **entities}
        
        # 根据工具类型填充参数
        if 'device' in all_params or 'device_name' in all_params:
            tool_request['device_name'] = all_params.get('device_name') or all_params.get('device')
        
        if 'interface' in all_params or 'interface_name' in all_params:
            tool_request['interface_name'] = all_params.get('interface_name') or all_params.get('interface')
        
        return {
            "tool_name": selected_tool,
            "tool_request": tool_request,
            "reasoning": f"备用工具选择（前{state.step_count}步选择遇到问题）",
            "expected_outcome": "获取基础信息",
            "next_focus": "根据结果决定下一步"
        }
    
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
