"""
异常判断器（Anomaly Judge）
在 World Model 和 Planner 之间工作，专门负责解读工具响应中的异常信号。

职责：
1. 接收 tool_response，判断其中是否包含故障迹象
2. 返回结构化的判断结果，供 StateManager 和 Planner 使用
3. 让 Planner 不再需要自己"猜"数据含义，直接消费判断结论
"""

import json
import openai
from typing import Dict, Any, Optional


class AnomalyJudge:
    """
    异常判断器 - World Model 与 Planner 之间的解释层

    调用时机：每次 world_model.execute_tool() 返回之后立即调用。
    输出结构：
    {
        "has_anomaly": bool,
        "severity": "none" | "low" | "medium" | "high",
        "anomaly_type": str,   # 如 "CRC错包过多", "接口Down", "丢包率超阈值" 等
        "evidence": str,       # 原始数据中支撑判断的字段和值
        "conclusion": str,     # 一句话结论，直接喂给 Planner 的 CoT
        "suggested_next": str  # 建议 Planner 下一步关注什么
    }
    """

    # 判定为异常的关键字（快速路径，避免不必要的 LLM 调用）
    ANOMALY_KEYWORDS = {
        # 状态类
        "down": "high", "Down": "high", "DOWN": "high",
        "error": "medium", "Error": "medium",
        "fail": "high", "failed": "high", "Failed": "high",
        "fault": "high", "Fault": "high",
        "异常": "medium", "故障": "high", "告警": "medium",
        "丢包": "high", "packet loss": "high",
        # 数值类关键词（需要阈值判断，这里只做快速初筛）
        "crc": "medium", "CRC": "medium",
        "drop": "medium", "drops": "medium",
        "congestion": "medium", "拥塞": "medium",
        "overload": "high", "过载": "high",
        "timeout": "medium", "超时": "medium",
        "unreachable": "high", "不可达": "high",
    }

    # 数值型字段的异常阈值
    NUMERIC_THRESHOLDS = {
        "丢包率": (">=", 1),        # 丢包率 >= 1%
        "packet_loss": (">=", 1),
        "crc_errors": (">", 100),
        "CRC错包数": (">", 100),
        "error_count": (">", 50),
        "错误帧数": (">", 50),
        "带宽利用率": (">=", 80),
        "cpu_usage": (">=", 85),
        "CPU使用率": (">=", 85),
        "内存使用率": (">=", 85),
        "output_drops": (">", 0),
        "input_drops": (">", 0),
        "input_errors": (">", 0),
        "output_errors": (">", 0),
        "光功率告警": ("==", True),
    }

    def __init__(self, api_key: str, api_base: str = None, model: str = "gpt-4o-mini",
                 use_llm: bool = True):
        """
        Args:
            api_key: LLM API 密钥
            api_base: API base URL
            model: 模型名称
            use_llm: 是否使用 LLM 进行深度判断（快速路径之后的二次分析）
                     设为 False 可以节省 token，只用规则判断
        """
        self.api_key = api_key
        self.api_base = api_base or "http://10.12.208.86:8502"
        self.model = model
        self.use_llm = use_llm

    # ------------------------------------------------------------------ #
    #  公开接口                                                             #
    # ------------------------------------------------------------------ #

    def judge(self,
              tool_name: str,
              tool_request: Dict,
              tool_response: Any,
              question: str = "",
              context: str = "") -> Dict[str, Any]:
        """
        判断工具响应中是否包含异常信号。

        Args:
            tool_name: 执行的工具名
            tool_request: 工具调用参数
            tool_response: 工具返回的原始数据
            question: 原始诊断问题（用于上下文）
            context: 当前诊断进展摘要（可选）

        Returns:
            判断结果字典，格式见类文档
        """
        # 1. 快速规则判断（无需 LLM）
        quick_result = self._quick_rule_check(tool_response)

        if quick_result["has_anomaly"] and not self.use_llm:
            # 不使用 LLM，直接用规则结果
            quick_result["conclusion"] = self._build_conclusion(tool_name, quick_result)
            quick_result["suggested_next"] = self._suggest_next(quick_result["anomaly_type"])
            return quick_result

        if not quick_result["has_anomaly"] and not self.use_llm:
            return self._no_anomaly_result()

        # 2. LLM 深度分析（快速路径发现异常 或 需要 LLM 验证时）
        # 只在以下情况调 LLM：快速路径发现了异常信号 OR 数据复杂需要语义理解
        should_use_llm = quick_result["has_anomaly"] or self._needs_semantic_analysis(tool_response)

        if should_use_llm and self.use_llm:
            try:
                llm_result = self._llm_judge(tool_name, tool_request, tool_response, question, context)
                # 合并：LLM 结果优先，但快速路径发现的 evidence 也保留
                if quick_result["has_anomaly"] and not llm_result.get("has_anomaly"):
                    # 快速路径发现了但 LLM 否定了 → 相信 LLM（规则可能误报）
                    pass
                return llm_result
            except Exception as e:
                print(f"   ⚠️  AnomalyJudge LLM 调用失败: {e}，使用规则判断结果")
                # 降级到规则结果
                quick_result["conclusion"] = self._build_conclusion(tool_name, quick_result)
                quick_result["suggested_next"] = self._suggest_next(quick_result["anomaly_type"])
                return quick_result

        return self._no_anomaly_result()

    # ------------------------------------------------------------------ #
    #  快速规则判断                                                         #
    # ------------------------------------------------------------------ #

    def _quick_rule_check(self, tool_response: Any) -> Dict[str, Any]:
        """基于关键字和阈值的快速规则判断"""
        if tool_response is None:
            return self._no_anomaly_result()

        # 展平响应为可搜索的 {key: value} 字典
        flat = self._flatten_response(tool_response)

        found_keywords = []
        found_numerics = []
        max_severity = "none"

        # 关键字匹配
        for key, val in flat.items():
            val_str = str(val).lower() if not isinstance(val, str) else val
            for keyword, severity in self.ANOMALY_KEYWORDS.items():
                if keyword.lower() in val_str or keyword.lower() in key.lower():
                    found_keywords.append(f"{key}={val}({keyword})")
                    if self._severity_rank(severity) > self._severity_rank(max_severity):
                        max_severity = severity
                    break

        # 数值阈值匹配
        for key, val in flat.items():
            if key in self.NUMERIC_THRESHOLDS:
                op, threshold = self.NUMERIC_THRESHOLDS[key]
                try:
                    num_val = float(str(val).replace("%", "").replace(",", ""))
                    violated = self._check_threshold(num_val, op, threshold)
                    if violated:
                        found_numerics.append(f"{key}={val}(阈值{op}{threshold})")
                        if self._severity_rank("medium") > self._severity_rank(max_severity):
                            max_severity = "medium"
                except (ValueError, TypeError):
                    pass

        has_anomaly = bool(found_keywords or found_numerics)
        evidence = "; ".join(found_keywords + found_numerics) if has_anomaly else ""

        # 推断异常类型
        anomaly_type = self._infer_anomaly_type(found_keywords, found_numerics)

        return {
            "has_anomaly": has_anomaly,
            "severity": max_severity if has_anomaly else "none",
            "anomaly_type": anomaly_type,
            "evidence": evidence,
            "conclusion": "",    # 由调用方填充
            "suggested_next": "" # 由调用方填充
        }

    def _flatten_response(self, response: Any, prefix: str = "") -> Dict[str, Any]:
        """将嵌套响应展平为 {key: value}"""
        result = {}
        if isinstance(response, dict):
            for k, v in response.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    result.update(self._flatten_response(v, full_key))
                else:
                    result[full_key] = v
        elif isinstance(response, list):
            for i, item in enumerate(response):
                result.update(self._flatten_response(item, f"{prefix}[{i}]"))
        else:
            result[prefix or "value"] = response
        return result

    def _severity_rank(self, severity: str) -> int:
        return {"none": 0, "low": 1, "medium": 2, "high": 3}.get(severity, 0)

    def _check_threshold(self, val: float, op: str, threshold) -> bool:
        if op == ">=": return val >= threshold
        if op == ">":  return val > threshold
        if op == "<=": return val <= threshold
        if op == "<":  return val < threshold
        if op == "==": return val == threshold
        return False

    def _infer_anomaly_type(self, keywords: list, numerics: list) -> str:
        if not keywords and not numerics:
            return ""
        evidence_str = " ".join(keywords + numerics).lower()
        if "down" in evidence_str:            return "接口Down"
        if "crc" in evidence_str:             return "CRC错包异常"
        if "drop" in evidence_str or "丢包" in evidence_str: return "丢包异常"
        if "cpu" in evidence_str:             return "CPU过载"
        if "内存" in evidence_str:            return "内存不足"
        if "带宽" in evidence_str:            return "带宽利用率过高"
        if "故障" in evidence_str or "fault" in evidence_str: return "设备故障"
        if "超时" in evidence_str or "timeout" in evidence_str: return "连接超时"
        if "unreachable" in evidence_str or "不可达" in evidence_str: return "目标不可达"
        return "数据异常"

    def _needs_semantic_analysis(self, tool_response: Any) -> bool:
        """判断是否需要 LLM 语义分析（针对复杂结构）"""
        if isinstance(tool_response, list) and len(tool_response) > 3:
            return True
        if isinstance(tool_response, dict) and len(tool_response) > 8:
            return True
        return False

    # ------------------------------------------------------------------ #
    #  LLM 深度分析                                                        #
    # ------------------------------------------------------------------ #

    def _llm_judge(self, tool_name: str, tool_request: Dict, tool_response: Any,
                   question: str, context: str) -> Dict[str, Any]:
        """使用 LLM 对工具响应进行深度异常判断"""

        response_str = json.dumps(tool_response, ensure_ascii=False)
        if len(response_str) > 1000:
            response_str = response_str[:1000] + "...(已截断)"

        prompt = f"""你是一个网络故障诊断专家。请分析以下工具响应，判断其中是否包含异常信号。

【原始问题】
{question or "网络故障诊断"}

【当前诊断进展】
{context or "暂无"}

【工具调用】
工具: {tool_name}
参数: {json.dumps(tool_request, ensure_ascii=False)}

【工具响应】
{response_str}

请判断上述响应中是否存在以下类型的异常（任何一种满足即可认定有异常）：
- 接口/设备状态异常（Down、Error、Failed 等）
- 性能指标超阈值（丢包率>1%、CRC错包>100、CPU>85%、带宽利用率>80% 等）
- 告警/故障信息
- 连通性问题（Ping不通、路由不可达等）
- 配置错误

以 JSON 格式输出（只输出 JSON，不要有其他文字）：
```json
{{
  "has_anomaly": true或false,
  "severity": "none" 或 "low" 或 "medium" 或 "high",
  "anomaly_type": "异常类型（如：CRC错包异常、接口Down、丢包率过高），无异常时填空字符串",
  "evidence": "数据中支撑判断的具体字段和值，无异常时填空字符串",
  "conclusion": "一句话总结（如：10GE1/0/24接口CRC错包数达1500，明显超过正常阈值，是导致丢包的根因），无异常时总结正常状态",
  "suggested_next": "建议下一步检查什么（如：检查光模块功率是否正常），无明确建议时填空字符串"
}}
```"""

        client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是网络故障分析专家，擅长从工具响应数据中识别异常信号。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 低温度保证判断一致性
            max_tokens=300
        )

        result_text = response.choices[0].message.content
        result = self._parse_json(result_text)

        # 确保必要字段存在
        result.setdefault("has_anomaly", False)
        result.setdefault("severity", "none")
        result.setdefault("anomaly_type", "")
        result.setdefault("evidence", "")
        result.setdefault("conclusion", "")
        result.setdefault("suggested_next", "")

        return result

    # ------------------------------------------------------------------ #
    #  辅助方法                                                             #
    # ------------------------------------------------------------------ #

    def _build_conclusion(self, tool_name: str, result: Dict) -> str:
        if not result["has_anomaly"]:
            return f"{tool_name} 响应正常，无异常信号"
        return f"[{result['anomaly_type']}] {result['evidence']}"

    def _suggest_next(self, anomaly_type: str) -> str:
        mapping = {
            "接口Down":       "检查接口物理连接和光模块状态",
            "CRC错包异常":     "检查光模块发送/接收功率是否正常，排查物理链路问题",
            "丢包异常":        "检查接口带宽利用率和QoS策略",
            "CPU过载":        "检查设备运行进程和路由表规模",
            "内存不足":        "检查设备内存使用详情和进程占用",
            "带宽利用率过高":  "检查接口流量构成和Top流量来源",
            "连接超时":        "检查路由表和ARP表项",
            "目标不可达":      "执行Traceroute定位断点",
        }
        return mapping.get(anomaly_type, "根据异常类型进一步深入排查")

    def _no_anomaly_result(self) -> Dict[str, Any]:
        return {
            "has_anomaly": False,
            "severity": "none",
            "anomaly_type": "",
            "evidence": "",
            "conclusion": "响应数据正常，无异常信号",
            "suggested_next": ""
        }

    def _parse_json(self, text: str) -> Dict:
        try:
            if "```json" in text:
                s = text.find("```json") + 7
                e = text.find("```", s)
                return json.loads(text[s:e].strip())
            elif "```" in text:
                s = text.find("```") + 3
                e = text.find("```", s)
                return json.loads(text[s:e].strip())
            else:
                return json.loads(text.strip())
        except Exception:
            return {}
