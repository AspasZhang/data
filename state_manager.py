"""
状态管理器（State Manager）
管理诊断过程中的状态、观察结果和判断是否继续
"""

from typing import Dict, List, Any, Tuple
from collections import defaultdict
import json


class StateManager:
    """状态管理器"""
    
    def __init__(self):
        """初始化状态管理器"""
        self.step_count = 0
        self.executed_tools = []  # [{tool_name, tool_request, tool_response, step}]
        self.observations = {}     # {key: observation_data}
        self.findings = []         # [{finding, severity, step}]
        self.tool_usage_count = defaultdict(int)  # {tool_name: count}
        
        # 新增：诊断逻辑链
        self.diagnostic_chain = []  # [{step, action, result, conclusion, next_focus}]
        self.current_focus = None   # 当前诊断焦点
        self.excluded_causes = []   # 已排除的原因
    
    def add_execution(self, tool_name: str, tool_request: Dict, tool_response: Dict, reasoning: str = ""):
        """
        添加一次工具执行记录
        
        Args:
            tool_name: 工具名称
            tool_request: 工具请求参数
            tool_response: 工具响应结果
            reasoning: 选择该工具的理由
        """
        self.step_count += 1
        
        record = {
            "step": self.step_count,
            "tool_name": tool_name,
            "tool_request": tool_request,
            "tool_response": tool_response,
            "reasoning": reasoning
        }
        
        self.executed_tools.append(record)
        self.tool_usage_count[tool_name] += 1
        
        # 提取关键观察
        self._extract_observations(tool_name, tool_response)
    
    def _extract_observations(self, tool_name: str, tool_response: Dict):
        """从工具响应中提取关键观察"""
        # 简单提取：将响应的关键字段存入observations
        for key, value in tool_response.items():
            if not key.startswith('_'):  # 跳过内部字段
                obs_key = f"{tool_name}_{key}"
                self.observations[obs_key] = value
    
    def add_finding(self, finding: str, severity: str = "medium"):
        """
        添加一个发现
        
        Args:
            finding: 发现的问题或结论
            severity: 严重程度 (low/medium/high)
        """
        self.findings.append({
            "finding": finding,
            "severity": severity,
            "step": self.step_count
        })
    
    def update_diagnostic_chain(self, action: str, result: str, conclusion: str, next_focus: str = None):
        """
        更新诊断逻辑链
        
        Args:
            action: 执行的动作（选择的工具和原因）
            result: 观察结果
            conclusion: 得出的结论
            next_focus: 下一步的诊断焦点
        """
        chain_item = {
            "step": self.step_count,
            "action": action,
            "result": result,
            "conclusion": conclusion
        }
        
        if next_focus:
            chain_item["next_focus"] = next_focus
            self.current_focus = next_focus
        
        self.diagnostic_chain.append(chain_item)
    
    def add_excluded_cause(self, cause: str):
        """
        添加已排除的原因
        
        Args:
            cause: 排除的原因
        """
        if cause not in self.excluded_causes:
            self.excluded_causes.append(cause)
    
    def set_current_focus(self, focus: str):
        """
        设置当前诊断焦点
        
        Args:
            focus: 当前焦点描述
        """
        self.current_focus = focus
    
    def format_diagnostic_chain(self) -> str:
        """
        格式化诊断逻辑链为易读的字符串
        
        Returns:
            格式化的诊断链
        """
        if not self.diagnostic_chain:
            return "暂无诊断逻辑链"
        
        lines = []
        for item in self.diagnostic_chain:
            lines.append(f"【Step {item['step']}】")
            lines.append(f"  动作: {item['action']}")
            lines.append(f"  结果: {item['result']}")
            lines.append(f"  结论: {item['conclusion']}")
            if 'next_focus' in item:
                lines.append(f"  → 下一步焦点: {item['next_focus']}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def get_diagnostic_context(self) -> str:
        """
        获取当前诊断上下文（用于Planner）
        
        Returns:
            包含当前焦点、已排除原因、诊断链的上下文
        """
        lines = []
        
        # 当前焦点
        if self.current_focus:
            lines.append(f"当前焦点: {self.current_focus}")
        else:
            lines.append("当前焦点: 初步诊断")
        
        # 已排除的原因
        if self.excluded_causes:
            lines.append(f"已排除: {', '.join(self.excluded_causes)}")
        
        # 最近3步的逻辑链
        if self.diagnostic_chain:
            recent_chain = self.diagnostic_chain[-3:]
            lines.append("\n最近诊断步骤:")
            for item in recent_chain:
                lines.append(f"  Step {item['step']}: {item['action']} → {item['conclusion']}")
        
        return '\n'.join(lines) if lines else "初始状态"
    
    
    def get_tool_usage_count(self, tool_name: str) -> int:
        """获取某个工具的使用次数"""
        return self.tool_usage_count.get(tool_name, 0)
    
    def get_recent_tools(self, n: int = 3) -> List[str]:
        """获取最近N步使用的工具"""
        recent = self.executed_tools[-n:]
        return [record['tool_name'] for record in recent]
    
    def has_used_tool(self, tool_name: str) -> bool:
        """判断是否已经使用过某个工具"""
        return tool_name in self.tool_usage_count
    
    def should_continue(self, max_steps: int = 20) -> Tuple[bool, str]:
        """
        判断是否应该继续诊断
        
        Args:
            max_steps: 最大步骤数
            
        Returns:
            (should_continue, reason)
        """
        # 条件1：达到最大步数
        if self.step_count >= max_steps:
            return False, "达到最大步数限制"
        
        # 条件2：找到高严重度的问题
        if self._has_critical_finding():
            return False, "找到关键问题"
        
        # 条件3：连续N步没有新发现
        if self._no_new_findings_recently(window=3):
            if self.step_count >= 3:  # 至少执行3步
                return False, "连续多步无新发现"
        
        # 条件4：已经检查了足够多的方面
        if self._sufficient_coverage():
            return False, "已完成全面检查"
        
        return True, "继续诊断"
    
    def _has_critical_finding(self) -> bool:
        """判断是否有关键发现"""
        for finding in self.findings:
            if finding['severity'] == 'high':
                return True
        return False
    
    def _no_new_findings_recently(self, window: int = 3) -> bool:
        """判断最近N步是否没有新发现"""
        if self.step_count < window:
            return False
        
        # 检查最近window步是否有新发现
        recent_steps = range(self.step_count - window + 1, self.step_count + 1)
        recent_findings = [f for f in self.findings if f['step'] in recent_steps]
        
        return len(recent_findings) == 0
    
    def _sufficient_coverage(self) -> bool:
        """判断是否已经有足够的检查覆盖"""
        # 简单策略：执行了8个以上不同的工具
        unique_tools = len(self.tool_usage_count)
        return unique_tools >= 8
    
    def get_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            "total_steps": self.step_count,
            "unique_tools_used": len(self.tool_usage_count),
            "total_findings": len(self.findings),
            "critical_findings": len([f for f in self.findings if f['severity'] == 'high'])
        }
    
    def get_execution_records(self) -> List[Dict[str, Any]]:
        """获取所有执行记录"""
        return self.executed_tools.copy()
    
    def format_recent_history(self, n: int = 5) -> str:
        """格式化最近N步的历史"""
        recent = self.executed_tools[-n:]
        
        if not recent:
            return "无执行历史"
        
        lines = []
        for record in recent:
            lines.append(f"Step {record['step']}: {record['tool_name']}")
            if record.get('reasoning'):
                lines.append(f"  理由: {record['reasoning']}")
        
        return "\n".join(lines)
    
    def format_observations(self) -> str:
        """格式化观察结果"""
        if not self.observations:
            return "暂无观察结果"
        
        # 只显示最重要的观察（最近的）
        recent_obs = dict(list(self.observations.items())[-10:])
        return json.dumps(recent_obs, ensure_ascii=False, indent=2)
    
    def format_findings(self) -> str:
        """格式化发现列表"""
        if not self.findings:
            return "暂无发现"
        
        lines = []
        for i, finding in enumerate(self.findings, 1):
            severity_symbol = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }.get(finding['severity'], '⚪')
            
            lines.append(f"{severity_symbol} {i}. {finding['finding']} (Step {finding['step']})")
        
        return "\n".join(lines)


def test_state_manager():
    """测试状态管理器"""
    print("=" * 80)
    print("测试状态管理器")
    print("=" * 80)
    
    state = StateManager()
    
    # 模拟执行
    print("\n模拟诊断过程：")
    print("-" * 80)
    
    # Step 1
    state.add_execution(
        "query_interface_info",
        {"device": "device1", "interface": "eth0"},
        {"status": "up", "speed": "1000Mbps", "errors": 0},
        "检查接口基本信息"
    )
    print(f"Step 1: 已使用工具数 = {len(state.tool_usage_count)}")
    
    # Step 2
    state.add_execution(
        "query_interface_traffic",
        {"device": "device1", "interface": "eth0"},
        {"rx_rate": "800Mbps", "tx_rate": "50Mbps"},
        "检查流量情况"
    )
    state.add_finding("接收流量较高，可能存在拥塞", "medium")
    print(f"Step 2: 发现数 = {len(state.findings)}")
    
    # Step 3
    state.add_execution(
        "query_interface_error_statistics",
        {"device": "device1", "interface": "eth0"},
        {"crc_errors": 1500, "collisions": 0},
        "检查错误统计"
    )
    state.add_finding("CRC错包数异常偏高", "high")
    print(f"Step 3: 发现严重问题")
    
    # 测试是否应该继续
    print("\n测试继续判断：")
    print("-" * 80)
    should_continue, reason = state.should_continue(max_steps=20)
    print(f"是否继续: {should_continue}")
    print(f"原因: {reason}")
    
    # 显示摘要
    print("\n状态摘要：")
    print("-" * 80)
    summary = state.get_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    
    # 显示发现
    print("\n发现列表：")
    print("-" * 80)
    print(state.format_findings())
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == '__main__':
    test_state_manager()
