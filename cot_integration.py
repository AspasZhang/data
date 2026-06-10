"""
CoT决策树集成模块
负责CoT匹配、节点执行和调用栈管理
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple


class CoTIntegration:
    """CoT决策树集成器"""

    def __init__(self, cot_trees_path: str = None):
        """
        初始化CoT集成器

        Args:
            cot_trees_path: CoT决策树JSON文件路径
        """
        self.cot_trees_path = cot_trees_path or "cot_compiler/cot_trees.json"
        self.cot_trees = {}
        self.dependency_graph = {}

        # 调用栈管理
        self.call_stack = []  # [(cot_name, step_id), ...]
        self.cot_call_count = {}  # {cot_name: count}

        # 限制
        self.MAX_DEPTH = 10
        self.MAX_SAME_COT_CALLS = 2

        # 当前执行状态
        self.current_cot = None
        self.current_step = None

        # ── 防重复：记录已执行过的 (cot_name, step_id) ──
        self.visited_nodes = set()

        self._load_cot_trees()

    def _load_cot_trees(self):
        """加载CoT决策树"""
        try:
            with open(self.cot_trees_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.cot_trees = data.get('cot_trees', {})
                self.dependency_graph = data.get('dependency_graph', {})
            print(f"✅ 加载了 {len(self.cot_trees)} 个CoT决策树")
        except FileNotFoundError:
            print(f"⚠️  CoT决策树文件未找到: {self.cot_trees_path}")
            print(f"   将使用自由探索模式")
        except Exception as e:
            print(f"⚠️  加载CoT决策树失败: {e}")
            print(f"   将使用自由探索模式")

    def match_cot(self, question: str, goal: Dict[str, Any]) -> Optional[str]:
        """
        匹配问题到对应的CoT

        策略：
        1. 关键词匹配（问题类型、关键方面）
        2. 示例问题相似度匹配

        Args:
            question: 用户问题
            goal: 提取的目标信息

        Returns:
            匹配的CoT名称，如果没有匹配则返回None
        """
        if not self.cot_trees:
            return None

        question_lower = question.lower()
        problem_type = goal.get('problem_type', '').lower()
        key_aspects = [aspect.lower() for aspect in goal.get('key_aspects', [])]

        # 关键词映射
        keyword_map = {
            '丢包': ['物理接口丢包异常', '接口流量异常变化'],
            '错包': ['物理接口错包异常', 'CRC错包超阈值'],
            'crc': ['CRC错包超阈值', '物理接口错包异常'],
            '流量': ['接口流量异常变化', '接口微突发分析'],
            '光功率': ['光功率超阈值', '光模块状态异常'],
            '光模块': ['光模块状态异常', '光功率超阈值'],
        }

        # 收集候选CoT
        candidates = []

        # 1. 基于关键词匹配
        for keyword, cot_names in keyword_map.items():
            if keyword in question_lower or keyword in problem_type:
                for cot_name in cot_names:
                    if cot_name in self.cot_trees:
                        candidates.append((cot_name, 2.0))  # 权重2.0

        # 2. 基于CoT的example字段匹配
        for cot_name, cot_data in self.cot_trees.items():
            examples = cot_data.get('example', [])
            for example in examples:
                if isinstance(example, str):
                    # 简单的相似度计算
                    common_words = sum(1 for word in example.split() if word in question)
                    if common_words > 2:
                        candidates.append((cot_name, 1.0 + common_words * 0.2))

        # 3. 基于问题类型匹配
        for cot_name, cot_data in self.cot_trees.items():
            cot_type = cot_data.get('type', '').lower()
            cot_subtype = cot_data.get('subtype', '').lower()

            if problem_type and (problem_type in cot_type or problem_type in cot_subtype):
                candidates.append((cot_name, 1.5))

        # 去重并按权重排序
        cot_scores = {}
        for cot_name, score in candidates:
            cot_scores[cot_name] = cot_scores.get(cot_name, 0) + score

        if not cot_scores:
            return None

        # 返回得分最高的CoT
        best_cot = max(cot_scores.items(), key=lambda x: x[1])
        print(f"   🎯 匹配到CoT: {best_cot[0]} (得分: {best_cot[1]:.1f})")

        return best_cot[0]

    def start_cot(self, cot_name: str) -> bool:
        """
        开始执行一个CoT

        Args:
            cot_name: CoT名称

        Returns:
            是否成功开始
        """
        if cot_name not in self.cot_trees:
            print(f"   ❌ CoT不存在: {cot_name}")
            return False

        # 检查调用深度
        if len(self.call_stack) >= self.MAX_DEPTH:
            print(f"   ❌ 调用深度超限 (max={self.MAX_DEPTH})")
            return False

        # 检查同一CoT的调用次数
        self.cot_call_count[cot_name] = self.cot_call_count.get(cot_name, 0) + 1
        if self.cot_call_count[cot_name] > self.MAX_SAME_COT_CALLS:
            print(f"   ❌ CoT调用次数超限: {cot_name} (max={self.MAX_SAME_COT_CALLS})")
            return False

        self.current_cot = cot_name
        self.current_step = "step_1"  # 总是从step_1开始

        # 压栈
        self.call_stack.append((cot_name, self.current_step))

        print(f"   📍 开始执行CoT: {cot_name}")
        return True

    def get_current_node(self) -> Optional[Dict[str, Any]]:
        """
        获取当前步骤的节点信息

        Returns:
            节点信息，包括type, intent, candidate_tools等
        """
        if not self.current_cot or not self.current_step:
            return None

        cot_data = self.cot_trees.get(self.current_cot)
        if not cot_data:
            return None

        tree = cot_data.get('tree', {})
        node = tree.get(self.current_step)

        if node:
            # 添加CoT和step信息
            node['cot_name'] = self.current_cot
            node['step_id'] = self.current_step

        return node

    def advance_to_next_step(self, condition_result: bool = True) -> bool:
        """
        前进到下一步

        Args:
            condition_result: 条件判断结果（对于条件节点）

        Returns:
            是否成功前进（False表示到达终点或出错）
        """
        node = self.get_current_node()
        if not node:
            return False

        node_type = node.get('type')

        # 终止节点
        if node_type == 'terminal':
            print(f"   ✅ CoT执行完成: {self.current_cot}")
            self._pop_call_stack()
            return False

        # 条件节点
        if node_type == 'condition':
            next_step = node.get('next_if_true') if condition_result else node.get('next_if_false')
        else:
            next_step = node.get('next')

        if not next_step:
            print(f"   ⚠️  节点 {self.current_step} 没有下一步，CoT结束")
            self._pop_call_stack()
            return False

        # 更新当前步骤
        self.current_step = next_step

        # 更新栈顶
        if self.call_stack:
            self.call_stack[-1] = (self.current_cot, self.current_step)

        # ── 防重复：如果新节点已访问过，自动跳过直到找到未访问的节点 ──
        max_skip = 20  # 防止无限循环
        skipped = 0
        while self._is_visited(self.current_cot, self.current_step) and skipped < max_skip:
            print(f"   ⏭️  跳过已访问节点: {self.current_cot}/{self.current_step}")
            skip_node = self.get_current_node()
            if not skip_node:
                break
            if skip_node.get('type') == 'terminal':
                print(f"   ✅ CoT执行完成: {self.current_cot}")
                self._pop_call_stack()
                return False
            # 根据节点类型决定下一步（修复：condition 类型需走 next_if_true/next_if_false）
            skip_type = skip_node.get('type', '')
            if skip_type == 'condition':
                # 已访问的 condition 节点无法再做判断，默认走 true 路径；
                # 如果 true 路径也为空则尝试 false 路径
                skip_next = skip_node.get('next_if_true') or skip_node.get('next_if_false')
            else:
                skip_next = skip_node.get('next')
            if not skip_next:
                self._pop_call_stack()
                return False
            self.current_step = skip_next
            if self.call_stack:
                self.call_stack[-1] = (self.current_cot, self.current_step)
            skipped += 1

        return True

    def _is_visited(self, cot_name: str, step_id: str) -> bool:
        """检查节点是否已访问过"""
        return (cot_name, step_id) in self.visited_nodes

    def mark_visited(self, cot_name: str = None, step_id: str = None):
        """标记当前节点为已访问"""
        cot = cot_name or self.current_cot
        step = step_id or self.current_step
        if cot and step:
            self.visited_nodes.add((cot, step))

    def handle_cot_call(self, target_cot: str) -> bool:
        """
        处理CoT调用（子程序调用）

        Args:
            target_cot: 目标CoT名称

        Returns:
            是否成功调用
        """
        # 开始新的CoT
        success = self.start_cot(target_cot)

        if success:
            print(f"   🔄 调用子CoT: {target_cot} (深度: {len(self.call_stack)})")

        return success

    def _pop_call_stack(self):
        """弹出调用栈，返回父CoT并自动前进到调用点的next节点"""
        if self.call_stack:
            self.call_stack.pop()

        # 恢复到上一层CoT
        if self.call_stack:
            self.current_cot, self.current_step = self.call_stack[-1]
            print(f"   ↩️  返回到CoT: {self.current_cot}, 步骤: {self.current_step}")

            # ── 修复：子CoT返回后自动前进到调用点的next ──────────────────
            # 避免停留在同一个 conditional_call/call_with_return 节点上，
            # 导致下一轮循环 should_call_subcot 再次触发同一子CoT调用。
            cot_data = self.cot_trees.get(self.current_cot)
            if cot_data:
                tree = cot_data.get('tree', {})
                node = tree.get(self.current_step)
                if node and node.get('type') in ('conditional_call', 'call_with_return', 'tail_call'):
                    next_step = node.get('next')
                    if next_step:
                        print(f"   ⏩ 子CoT完成，自动前进: {self.current_step} → {next_step}")
                        self.current_step = next_step
                        self.call_stack[-1] = (self.current_cot, self.current_step)
        else:
            self.current_cot = None
            self.current_step = None

    def is_active(self) -> bool:
        """是否有活跃的CoT执行"""
        return self.current_cot is not None

    def get_candidate_tools(self) -> List[str]:
        """
        获取当前节点的候选工具列表

        Returns:
            工具名称列表
        """
        node = self.get_current_node()
        if not node:
            return []

        return node.get('candidate_tools', [])

    def get_node_intent(self) -> str:
        """
        获取当前节点的意图描述

        Returns:
            意图描述文本
        """
        node = self.get_current_node()
        if not node:
            return ""

        return node.get('intent', '')

    def should_call_subcot(self) -> Tuple[bool, Optional[str]]:
        """
        检查当前节点是否需要调用子CoT

        Returns:
            (是否需要调用, 目标CoT名称)
        """
        node = self.get_current_node()
        if not node:
            return False, None

        node_type = node.get('type')

        if node_type in ['conditional_call', 'call_with_return', 'tail_call']:
            target = node.get('call_target')
            return True, target

        return False, None

    def reset(self):
        """重置执行状态（用于新的问题）"""
        self.call_stack = []
        self.cot_call_count = {}
        self.current_cot = None
        self.current_step = None

    def get_status_summary(self) -> str:
        """获取当前状态摘要"""
        if not self.is_active():
            return "无活跃CoT"

        return (f"CoT: {self.current_cot}, "
                f"步骤: {self.current_step}, "
                f"深度: {len(self.call_stack)}")


def test_cot_integration():
    """测试CoT集成"""
    print("=" * 80)
    print("测试CoT集成模块")
    print("=" * 80)

    # 初始化
    cot_integration = CoTIntegration()

    # 测试匹配
    print("\n测试1: CoT匹配")
    print("-" * 80)

    test_questions = [
        ("交换机Leaf-03的10GE1/0/9接口出现丢包", {"problem_type": "丢包", "key_aspects": ["接口状态"]}),
        ("设备的光模块收发光功率异常", {"problem_type": "光功率异常", "key_aspects": ["光模块"]}),
        ("接口CRC错包超阈值", {"problem_type": "CRC异常", "key_aspects": ["错包"]}),
    ]

    for question, goal in test_questions:
        print(f"\n问题: {question}")
        matched_cot = cot_integration.match_cot(question, goal)
        if matched_cot:
            print(f"✅ 匹配成功: {matched_cot}")
        else:
            print(f"❌ 未匹配到CoT")

    # 测试执行流程
    print("\n\n测试2: CoT执行流程")
    print("-" * 80)

    cot_integration.reset()

    if cot_integration.start_cot("物理接口丢包异常"):
        step_count = 0
        while cot_integration.is_active() and step_count < 10:
            step_count += 1

            node = cot_integration.get_current_node()
            if not node:
                break

            print(f"\n步骤 {step_count}:")
            print(f"  节点: {node['step_id']}")
            print(f"  类型: {node['type']}")
            print(f"  意图: {node['intent'][:50]}...")
            print(f"  候选工具: {node.get('candidate_tools', [])}")

            # 检查是否需要调用子CoT
            should_call, target = cot_integration.should_call_subcot()
            if should_call:
                print(f"  🔄 需要调用子CoT: {target}")
                cot_integration.handle_cot_call(target)
            else:
                # 前进到下一步
                if not cot_integration.advance_to_next_step():
                    break

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == '__main__':
    test_cot_integration()
