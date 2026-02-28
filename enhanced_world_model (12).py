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
                 diversity_mode: str = 'medium',
                 tool_manager = None):
        """
        初始化增强世界模型
        
        Args:
            api_key: API密钥
            knowledge_base: 知识库（用于few-shot）
            api_base: API基础URL
            model: 模型名称
            diversity_mode: 多样性模式 ('low', 'medium', 'high')
            tool_manager: 工具管理器（用于获取输出格式）
        """
        self.api_key = api_key
        self.api_base = api_base or "http://10.12.208.86:8502"
        self.model = model
        self.diversity_mode = diversity_mode
        self.knowledge_base = knowledge_base or {}
        self.tool_manager = tool_manager
        
        # 字段值记忆：保持不同工具返回的相同字段一致
        # 格式：{"实体:字段名": 值}，例如 {"40GE2/2/5:丢包率": 15}
        self.field_memory: Dict[str, Any] = {}
        
        # ── 异常调度表 ──────────────────────────────────────────────
        # 在生成开始前通过 schedule_anomaly() 预先设定异常出现的步骤。
        # world model 在执行到目标步骤时自动强制生成异常数据，
        # 而不是等到最后再补救，从而让故障信号自然地出现在诊断流中。
        self._anomaly_target_step: Optional[int] = None   # 目标步骤编号（1-based）
        self._anomaly_delivered: bool = False              # 是否已经生成过异常
        self._current_step: int = 0                        # 世界模型侧的步骤计数
        # ────────────────────────────────────────────────────────────
        
        # 从知识库提取few-shot示例
        self.few_shot_examples = self._extract_few_shot_examples()
        
        # 定义响应变体
        self.response_variants = self._initialize_response_variants()
    # ================================================================== #
    #  异常调度接口                                                         #
    # ================================================================== #

    def schedule_anomaly(self, total_steps: int, earliest: int = 2) -> int:
        """
        在生成开始前预先调度异常出现的步骤，让故障自然分布在诊断流中。

        策略：
        - 异常出现在 [earliest, ceil(total_steps * 0.65)] 之间均匀随机选取
        - 这样保证：① 不会在第 1 步就出现异常（缺乏铺垫）
                    ② 不会等到最后才出现（避免末步"保底补救"的假象）
                    ③ 有足够的后续步骤根据异常继续深入诊断

        Args:
            total_steps: 预计总步骤数（max_steps）
            earliest: 最早可以出现异常的步骤编号（默认 2）

        Returns:
            目标异常步骤编号（1-based）
        """
        import math
        latest = max(earliest, math.ceil(total_steps * 0.65))
        self._anomaly_target_step = random.randint(earliest, latest)
        self._anomaly_delivered = False
        self._current_step = 0
        print(f"   📅 异常调度：将在第 {self._anomaly_target_step}/{total_steps} 步注入异常数据")
        return self._anomaly_target_step

    def reset_schedule(self):
        """重置调度状态（新的一轮生成开始前调用）"""
        self._anomaly_target_step = None
        self._anomaly_delivered = False
        self._current_step = 0
        
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
                    force_anomaly: bool = False,
                    cot: str = None,
                    question: str = None) -> Dict[str, Any]:
        """
        执行工具，生成响应。

        异常注入优先级（从高到低）：
        1. force_anomaly=True（外部显式强制，保留兼容性）
        2. 调度表命中（schedule_anomaly 预先设定的目标步骤）
        3. 正常的多样性采样
        """
        # ── 步骤计数（世界模型侧）────────────────────────────────────
        self._current_step += 1

        # ── 判断是否需要注入异常 ──────────────────────────────────────
        should_inject = force_anomaly
        if not should_inject and not self._anomaly_delivered:
            if (self._anomaly_target_step is not None and
                    self._current_step >= self._anomaly_target_step):
                should_inject = True
                print(f"      📅 调度命中第 {self._current_step} 步，注入预定异常数据")

        # ── 选择响应变体 ──────────────────────────────────────────────
        if should_inject:
            print(f"      ⚠️  强制生成异常数据 (step={self._current_step})")
            variant = self._select_anomaly_variant(tool_name)
            self._anomaly_delivered = True
        else:
            variant = self._select_response_variant(tool_name, run_id)

        # ── 获取工具的输出格式 ────────────────────────────────────────
        output_format = None
        if self.tool_manager:
            output_format = self.tool_manager.tool_outputs.get(tool_name)

        # ── 生成响应 ──────────────────────────────────────────────────
        response = self._generate_tool_response(
            tool_name,
            tool_request,
            variant,
            context or {},
            cot=cot,
            question=question,
            output_format=output_format
        )

        # ── 更新字段记忆 ──────────────────────────────────────────────
        self._update_field_memory(response, tool_request)

        return response
    
    def _parse_output_format(self, output_format: str) -> Dict[str, str]:
        """
        解析Output格式，提取字段名和普通类型。
        enum 字段在此统一记为 'str'，实际可选值通过 _parse_enum_fields 获取。
        隐性字段（type、enum）被过滤，不会出现在 observation 中。

        Args:
            output_format: 格式字符串，如
                '{"状态": "enum[up,down,error]", "接口": "str", "丢包率": "int"}'

        Returns:
            {"状态": "str", "接口": "str", "丢包率": "int"}
        """
        if not output_format:
            return {}

        import re
        try:
            pattern = r'"([^"]+)":\s*"([^"]*)"' 
            matches = re.findall(pattern, output_format)
            result = {}
            for field, ftype in matches:
                # 过滤隐性系统字段：type 和 enum 均不进入 observation
                if field.lower() in ('type', 'enum'):
                    continue
                # enum 字段统一记为 str（枚举选项由 _parse_enum_fields 单独管理）
                if ftype.lower().startswith('enum'):
                    result[field] = 'str'
                else:
                    result[field] = ftype.strip()
            return result
        except:
            return {}

    def _parse_enum_fields(self, output_format: str) -> Dict[str, list]:
        """
        从Output格式中提取 enum 字段及其可选值。

        支持格式：
          - "状态": "enum[up,down,error]"
          - "状态": "enum(up|down|error)"
          - "状态": "enum: up, down, error"

        Args:
            output_format: 原始格式字符串

        Returns:
            {"状态": ["up", "down", "error"], ...}
            非 enum 字段不包含在返回值中。
        """
        if not output_format:
            return {}

        import re
        result = {}
        try:
            # 匹配所有字段和值
            pattern = r'"([^"]+)":\s*"([^"]*)"' 
            matches = re.findall(pattern, output_format)
            for field, ftype in matches:
                if field.lower() in ('type', 'enum'):
                    continue
                ftype_lower = ftype.lower().strip()
                if not ftype_lower.startswith('enum'):
                    continue
                # 提取括号 [] 或 () 内的选项
                bracket_match = re.search(r'[\[(]([^\])]+)[\])]', ftype)
                if bracket_match:
                    raw = bracket_match.group(1)
                elif ':' in ftype:
                    raw = ftype.split(':', 1)[1]
                else:
                    # 去掉 "enum" 前缀后剩余部分
                    raw = ftype[4:]
                # 分割选项（支持逗号或竖线分隔）
                options = [o.strip().strip('"\'\' ') for o in re.split(r'[,|]', raw) if o.strip()]
                if options:
                    result[field] = options
        except:
            pass
        return result

    def _enforce_output_schema(self, response, output_format, variant=None):
        """
        强制将 LLM 返回值裁剪到 output_format 定义的字段集：
          1. 移除未定义字段（包括 type、enum 等隐性字段）
          2. enum 字段：值不在选项中时，按 variant severity 自动选合法值
          3. int 字段：若值是字符串，强制转换
          4. str 字段缺失时，尝试模糊匹配补全
        """
        field_types = self._parse_output_format(output_format)
        enum_fields  = self._parse_enum_fields(output_format)
        if not field_types:
            return response

        severity = (variant or {}).get('severity', 'normal')
        result = {}

        for field, ftype in field_types.items():
            raw_val = response.get(field)

            if field in enum_fields:
                options = enum_fields[field]
                if isinstance(raw_val, str) and raw_val in options:
                    result[field] = raw_val
                else:
                    chosen = self._pick_enum_by_severity(options, severity)
                    if raw_val is not None and raw_val not in options:
                        print(f'      ⚠️  enum修正 [{field}]: "{raw_val}" → "{chosen}" (合法: {options})')
                    result[field] = chosen

            elif ftype == 'int':
                if isinstance(raw_val, int):
                    result[field] = raw_val
                elif raw_val is not None:
                    try:
                        result[field] = int(str(raw_val).replace(',', '').replace('%', '').strip())
                    except (ValueError, TypeError):
                        result[field] = 0
                else:
                    result[field] = 0

            else:
                if raw_val is not None:
                    result[field] = str(raw_val) if not isinstance(raw_val, str) else raw_val
                else:
                    fallback = None
                    for k, v in response.items():
                        if field.lower() in k.lower() or k.lower() in field.lower():
                            fallback = str(v)
                            break
                    result[field] = fallback or ""

        return result

    def _pick_enum_by_severity(self, options, severity):
        """根据 severity 从 enum 选项中选语义最近的合法值。"""
        if not options:
            return ""
        normal_kw  = {'up', 'active', 'normal', 'ok', 'success', 'enable', 'enabled', '正常', '上线'}
        anomaly_kw = {'down', 'error', 'fail', 'failed', 'fault', 'inactive', 'disable',
                      'disabled', 'degraded', 'warning', '异常', '故障', '告警', '离线'}
        if severity in ('normal', 'mild'):
            for opt in options:
                if opt.lower() in normal_kw:
                    return opt
            return options[0]
        else:
            for opt in options:
                if opt.lower() in anomaly_kw:
                    return opt
            return options[-1]

    def _update_field_memory(self, response: Dict[str, Any], tool_request: Dict[str, Any]):
        """
        更新字段记忆
        
        Args:
            response: 工具响应
            tool_request: 工具请求（用于提取实体）
        """
        # 提取主要实体（接口、设备、IP等）
        entity = None
        for key in ['interface', 'device', 'ip', '接口', '设备', 'IP']:
            if key in tool_request:
                entity = tool_request[key]
                break
            if key in response:
                entity = response[key]
                break
        
        if not entity:
            return
        
        # 记录每个字段的值
        for field, value in response.items():
            if field not in ['接口', '设备', 'IP', 'interface', 'device', 'ip']:
                memory_key = f"{entity}:{field}"
                self.field_memory[memory_key] = value
    
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
            # 低多样性：倾向正常状态，但保留30%异常概率
            weights = [0.5, 0.25, 0.15, 0.1] if len(variants) >= 4 else [1.0]
            
        elif self.diversity_mode == 'medium':
            # 中多样性：增加异常权重，确保有足够异常
            weights = [0.3, 0.3, 0.25, 0.15] if len(variants) >= 4 else [1.0]
            
        else:  # high
            # 高多样性：倾向异常，正常反而更少
            weights = [0.15, 0.25, 0.35, 0.25] if len(variants) >= 4 else [1.0]
        
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
        强制选择异常变体（增强版）
        
        Args:
            tool_name: 工具名称
            
        Returns:
            异常变体
        """
        # 获取变体列表
        variants = self.response_variants.get(tool_name, self.response_variants['default'])
        
        # 第1优先级：选择severity为high的变体
        high_severity = [v for v in variants if v.get('severity', '').lower() == 'high']
        if high_severity:
            print(f"      🔴 选择高严重性异常变体")
            return high_severity[0]
        
        # 第2优先级：选择status明确为异常的变体
        anomaly_status = ['down', 'error', 'abnormal', '异常', 'critical', 'failed']
        for variant in variants:
            status = variant.get('status', '').lower()
            name = variant.get('name', '').lower()
            if status in anomaly_status or any(a in name for a in ['异常', 'error', 'down']):
                print(f"      🟠 选择状态异常变体: {variant.get('name')}")
                return variant
        
        # 第3优先级：选择severity为medium的变体
        medium_severity = [v for v in variants if v.get('severity', '').lower() == 'medium']
        if medium_severity:
            print(f"      🟡 选择中等严重性变体")
            return medium_severity[0]
        
        # 第4优先级：排除正常状态，选择其他任何变体
        normal_keywords = ['normal', 'up', '正常', 'ok', 'healthy']
        non_normal = [v for v in variants 
                     if not any(k in v.get('name', '').lower() for k in normal_keywords)]
        if non_normal:
            print(f"      🟤 选择非正常变体")
            return non_normal[0]
        
        # 保底：选择列表中间或末尾的变体（通常不是正常状态）
        if len(variants) >= 3:
            print(f"      ⚪ 保底：选择第3个变体")
            return variants[2]
        elif len(variants) >= 2:
            print(f"      ⚪ 保底：选择第2个变体")
            return variants[1]
        else:
            print(f"      ⚠️  警告：只有1个变体，可能无法生成异常")
            return variants[0]
    
    def _generate_tool_response(self, tool_name, tool_request,
                               variant, context, cot=None,
                               question=None, output_format=None):
        """生成工具响应，并强制裁剪到 output_format 定义的字段集。"""
        prompt = self._build_generation_prompt(
            tool_name, tool_request, variant, context, cot, question, output_format, tool_request
        )
        try:
            client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content":
                        "你是网络故障诊断工具模拟器。严格按给定JSON骨架填值，不增加任何额外字段。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            result_text = response.choices[0].message.content
            tool_response = self._parse_json_response(result_text)

            # ── 程序强制校验：无论 LLM 返回什么，强制裁剪+修正 ──────────────
            if output_format:
                tool_response = self._enforce_output_schema(tool_response, output_format, variant)

            return tool_response
        except Exception as e:
            print(f"世界模型执行失败: {e}")
            return {"error": "执行失败", "details": str(e)}

    def _build_generation_prompt(self, tool_name, tool_request,
                                 variant, context, cot=None,
                                 question=None, output_format=None,
                                 request_params=None):
        """构建生成prompt：给LLM具体JSON骨架让它填值，而非描述格式。"""
        field_types = self._parse_output_format(output_format) if output_format else {}
        enum_fields  = self._parse_enum_fields(output_format)  if output_format else {}

        # 已记录字段值（保证同实体前后一致）
        entity = None
        for key in ['interface', 'device', 'ip', '接口', '设备', 'IP']:
            if key in tool_request:
                entity = tool_request[key]
                break
        remembered_fields = {}
        if entity:
            for field in field_types:
                mem_key = f"{entity}:{field}"
                if mem_key in self.field_memory:
                    remembered_fields[field] = self.field_memory[mem_key]

        # 构建 JSON 骨架
        skeleton_parts = []
        for field, ftype in field_types.items():
            if field in remembered_fields:
                val = remembered_fields[field]
                skeleton_parts.append(f'  "{field}": {json.dumps(val, ensure_ascii=False)}')
            elif field in enum_fields:
                opts = "/".join(enum_fields[field])
                skeleton_parts.append(f'  "{field}": "【必须从以下选项中选一个填入，删掉本括号和说明：{opts}】"')
            elif ftype == 'int':
                skeleton_parts.append(f'  "{field}": 0')
            else:
                skeleton_parts.append(f'  "{field}": ""')
        skeleton_json = "{\n" + ",\n".join(skeleton_parts) + "\n}"

        # 语义约束
        extra = []
        if question:
            if 'down' in question.lower() or '故障' in question or 'Down' in question:
                extra.append('- 状态类enum字段选异常值（如down/error/fault）')
            if '错包' in question or 'CRC' in question.lower():
                extra.append('- 错包/CRC字段填非零数值')
        extra.append('- str字段从请求参数/问题中提取实体名，无则编造（如"40GE2/2/5"）')
        extra.append('- int字段填合理数值，不要用0占位（除非真的是0）')

        prompt = f"""你是网络故障诊断工具模拟器。请填写以下JSON骨架中每个字段的值。

工具：{tool_name}
参数：{json.dumps(tool_request, ensure_ascii=False)}
状态：{variant['name']} ({variant['severity']})

【填写规则 - 严格遵守】
1. 只填骨架中已有的字段，禁止增加任何其他字段
2. enum字段：将【...】整个替换成一个合法选项的字符串值（不含括号和说明文字）
3. int字段：只填数字，不加引号
4. str字段：填字符串内容，加引号
{chr(10).join(extra)}

【需要填写的JSON骨架】
```json
{skeleton_json}
```

只输出填好后的JSON，不要有任何其他文字。"""
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
                # 移除隐性系统字段（tool 定义产物，禁止出现在 observation 中）
                parsed.pop('type', None)
                parsed.pop('enum', None)
                # 已经是dict，直接返回
                return parsed
            elif isinstance(parsed, list):
                # 是list，需要转换
                if len(parsed) == 1 and isinstance(parsed[0], dict):
                    # list只有1个dict元素，返回该dict
                    print(f"⚠️  LLM返回了单元素list，自动提取: {list(parsed[0].keys())}")
                    return parsed[0]
                else:
                    # 多个元素或非dict元素，包装成dict（不添加 _type 字段，避免污染数据）
                    print(f"⚠️  LLM返回了list（{len(parsed)}个元素），包装为dict")
                    return {"data": parsed}
            else:
                # 其他类型（string, number等），包装成dict（不添加 _type 字段）
                print(f"⚠️  LLM返回了{type(parsed).__name__}类型，包装为dict")
                return {"value": parsed}
                
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
