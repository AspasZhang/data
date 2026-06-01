"""
Agent生成器（Agent Generator）- 集成新格式和批量操作
整合所有组件，实现自由探索的故障诊断数据生成
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from goal_extractor import GoalExtractor
from anomaly_judge import AnomalyJudge
from state_manager import StateManager
from enhanced_planner import EnhancedPlanner
from enhanced_world_model import EnhancedWorldModel
from tool_manager import ToolManager
from cot_integration import CoTIntegration
from structured_output import (
    StructuredOutputGenerator,
    extract_entities_from_observation,
    should_batch_execute
)


class AgentGenerator:
    """自由探索的Agent数据生成器"""
    
    def __init__(self, tool_manager: ToolManager,
                 api_key: str,
                 api_base: str = None,
                 knowledge_base: Optional[Dict] = None,
                 max_steps: int = 20):
        """
        初始化Agent生成器
        
        Args:
            tool_manager: 工具管理器
            api_key: API密钥
            api_base: API基础URL
            knowledge_base: 知识库
            max_steps: 最大步骤数
        """
        self.tool_manager = tool_manager
        self.api_key = api_key
        self.api_base = api_base or "http://localhost:20015/v1"
        self.knowledge_base = knowledge_base
        self.max_steps = max_steps
        
        # 初始化组件
        self.goal_extractor = GoalExtractor(api_key, api_base)

        # AnomalyJudge：world model 和 planner 之间的异常解读层
        # use_llm=True 时每次工具调用后做深度 LLM 分析，False 时只用规则（更快更省 token）
        self.anomaly_judge = AnomalyJudge(
            api_key=api_key,
            api_base=api_base or "http://10.12.208.86:8502",
            use_llm=True
        )

        # CoT决策树集成
        self.cot_integration = CoTIntegration()

        print("✅ Agent生成器初始化完成")
    
    def generate(self, 
                question: str, 
                run_config: Optional[Dict] = None,
                rewrite_question: bool = False) -> Dict[str, Any]:
        """
        生成一次诊断数据（新格式：query -> response[step{cot, coa}]）
        
        Args:
            question: 问题描述
            run_config: 运行配置
            rewrite_question: 是否改写问题
            
        Returns:
            {
                "query": "问题",
                "response": [
                    {
                        "step1": {
                            "cot": "推理",
                            "coa": [{"action": {...}, "observation": ...}]
                        }
                    }
                ]
            }
        """
        # 默认配置
        if run_config is None:
            run_config = {
                "run_id": 0,
                "exploration_mode": "balanced",
                "diversity_mode": "medium",
                "temperature": 0.7
            }
        
        run_id = run_config.get('run_id', 0)
        original_question = question
        
        print(f"\n{'='*80}")
        print(f"🚀 开始运行 #{run_id + 1}")
        print(f"{'='*80}")
        
        # 0. 问题改写（如果启用）
        if rewrite_question and run_id > 0:
            print(f"📝 步骤0: 改写问题以增加多样性...")
            from question_rewriter import QuestionRewriter
            
            if not hasattr(self, 'question_rewriter'):
                self.question_rewriter = QuestionRewriter(
                    api_key=self.api_key,
                    api_base=self.api_base
                )
            
            question = self.question_rewriter.rewrite_with_strategy(
                original_question,
                run_id=run_id,
                total_runs=run_config.get('total_runs', 10)
            )
            
            if question != original_question:
                print(f"   原始: {original_question}")
                print(f"   改写: {question}")
            else:
                print(f"   保持原问题")
            print()
        else:
            question = original_question
            if run_id == 0:
                print(f"问题: {question}")
            else:
                print(f"问题: {question} (未改写)")
        
        print(f"配置: exploration={run_config.get('exploration_mode')}, "
              f"diversity={run_config.get('diversity_mode')}, "
              f"temp={run_config.get('temperature')}")
        print(f"{'='*80}\n")

        # ============ 初始化新的结构化输出生成器 ============
        output_generator = StructuredOutputGenerator()

        # 1. 提取目标
        print("📍 步骤1: 提取诊断目标...")
        goal = self.goal_extractor.extract_goals(question, knowledge_base=self.knowledge_base)
        print(f"   主要目标: {goal.get('main_goal')}")
        print(f"   问题类型: {goal.get('problem_type')}")
        print(f"   关键方面: {', '.join(goal.get('key_aspects', []))}")
        if goal.get('context_params'):
            print(f"   相关参数: {goal.get('context_params')}")
        elif goal.get('entities'):
            print(f"   实体信息: {goal.get('entities')}")
        print()

        # ============ CoT匹配和初始化 ============
        self.cot_integration.reset()
        matched_cot = self.cot_integration.match_cot(question, goal)

        if matched_cot:
            print(f"📋 CoT决策树: 使用 '{matched_cot}' 引导诊断流程")
            self.cot_integration.start_cot(matched_cot)
        else:
            print(f"📋 CoT决策树: 未匹配到合适的CoT，使用自由探索模式")
        print()
        
        
        # 2. 初始化规划器和世界模型
        planner = EnhancedPlanner(
            tool_manager=self.tool_manager,
            api_key=self.api_key,
            api_base=self.api_base,
            exploration_mode=run_config.get('exploration_mode', 'balanced')
        )
        
        world_model = EnhancedWorldModel(
            api_key=self.api_key,
            knowledge_base=self.knowledge_base,
            api_base=self.api_base,
            diversity_mode=run_config.get('diversity_mode', 'medium'),
            tool_manager=self.tool_manager
        )
        
        # ── 预调度异常步骤 ──────────────────────────────────────────
        # 在生成开始前确定异常出现在哪一步，避免末尾"保底补救"的假象。
        world_model.schedule_anomaly(total_steps=self.max_steps, earliest=2)
        # ────────────────────────────────────────────────────────────
        
        # 3. 初始化状态
        state = StateManager()
        
        # 异常检测标志（由 AnomalyJudge 驱动）
        has_anomaly = False
        anomaly_steps = []          # 记录包含异常的步骤
        anomaly_found_step = None   # 首次发现异常的步骤编号

        # 发现异常后至少再执行多少步才允许停止（保证充分的深入排查数据）
        # 取 max_steps 的 30%~50%，并保证至少 3 步，最多 8 步
        import random as _rand
        _min_followup = _rand.randint(
            max(3, int(self.max_steps * 0.30)),
            max(4, int(self.max_steps * 0.50))
        )
        
        # ── CoT 完成跟踪（思路2：CoT 主导 + 异常驱动的去重尾部探索）──────────────
        # cot_was_active: 本次生成是否曾命中并启动过 CoT（决定停止策略）
        # cot_finished_step: CoT 走完（失活）发生在第几步；None 表示尚未结束
        # tail_budget: CoT 结束后允许的"尾部探索"步数上限（仅在刚发现异常、排查不足时使用）
        cot_was_active = self.cot_integration.is_active()
        cot_finished_step = None
        tail_budget = max(2, int(self.max_steps * 0.20))  # 尾部探索预算，硬上限
        in_tail_exploration = False

        # 4. 迭代探索
        print("🔍 步骤2: 开始迭代探索...\n")
        step_num = 0  # 初始化，避免循环顶部 CoT 完成判断引用未定义变量
        while True:
            # 检查是否应该继续
            should_continue, reason = state.should_continue(self.max_steps)

            # ── 思路2 核心：CoT 主导的终止控制 ──────────────────────────────
            # 一旦曾启动过 CoT 且当前 CoT 已走完（失活），不再无条件回退到自由探索，
            # 避免 Planner 在状态未变的情况下重新选出 CoT 起点工具，造成"路径走完又从头走一遍"。
            if cot_was_active and not self.cot_integration.is_active():
                if cot_finished_step is None:
                    cot_finished_step = step_num
                    print(f"\n   🏁 CoT 决策路径已走完（步骤 {step_num}）")

                # 判定是否需要"异常驱动的尾部探索"：
                # 仅当 CoT 走完时刚发现异常、且后续排查步数不足时，才允许有限补充排查。
                need_tail = (
                    has_anomaly
                    and anomaly_found_step is not None
                    and (cot_finished_step - anomaly_found_step) < _min_followup
                    and (step_num - cot_finished_step) < tail_budget
                )
                if need_tail:
                    if not in_tail_exploration:
                        in_tail_exploration = True
                        print(f"   🔎 进入异常驱动的尾部探索（最多 {tail_budget} 步，已用工具将被去重避让）")
                    # 继续循环做尾部补充排查（Planner 的 anti-repeat 会避让已执行工具）
                else:
                    print(f"\n🛑 停止探索: CoT 路径完成，无需进一步排查\n")
                    break

            elif not should_continue:
                # 仍在 CoT 引导中（或从未启动 CoT）的常规停止判断。
                # 发现异常后必须保证足够的后续排查步骤，否则强制继续。
                if (has_anomaly and anomaly_found_step is not None and
                        (step_num - anomaly_found_step) < _min_followup):
                    remaining = _min_followup - (step_num - anomaly_found_step)
                    print(f"   ⚡ 发现异常后续排查不足，强制继续 (还需至少 {remaining} 步)")
                else:
                    print(f"\n🛑 停止探索: {reason}\n")
                    break
            
            step_num = state.step_count + 1
            print(f"{'─'*80}")
            print(f"Step {step_num}:")
            
            # ── 异常调度由 world_model 内部处理，此处不再手动干预 ────────────────
            # world_model.schedule_anomaly() 在 generate() 开头已设定好目标步骤，
            # execute_tool() 到达目标步骤时会自动注入异常，无需外部逐步轮询。
            if has_anomaly and anomaly_steps:
                print(f"   ✅ 已在第 {anomaly_steps[-1]} 步发现异常，继续深入排查")
            
            # ============ 获取已知实体列表 ============
            known_entities_dict = {
                'interfaces': output_generator.get_known_entities('interfaces'),
                'devices': output_generator.get_known_entities('devices')
            }
            # 过滤空列表
            known_entities_dict = {k: v for k, v in known_entities_dict.items() if v}

            # ============ 获取CoT引导信息 ============
            candidate_tools = []
            cot_intent = ""

            if self.cot_integration.is_active():
                node = self.cot_integration.get_current_node()
                if node:
                    candidate_tools = self.cot_integration.get_candidate_tools()
                    cot_intent = self.cot_integration.get_node_intent()

                    print(f"   🎯 CoT节点: {node['step_id']} ({node['type']})")
                    if cot_intent:
                        print(f"   💡 诊断意图: {cot_intent[:80]}{'...' if len(cot_intent) > 80 else ''}")
                    if candidate_tools:
                        print(f"   🔧 推荐工具: {', '.join(candidate_tools)}")

            # 4.1 规划下一步（传入已知实体和CoT引导）
            plan = planner.select_next_tool(
                state,
                goal,
                temperature=run_config.get('temperature', 0.7),
                known_entities=known_entities_dict if known_entities_dict else None,
                candidate_tools=candidate_tools if candidate_tools else None,
                cot_intent=cot_intent if cot_intent else None
            )
            
            if 'error' in plan:
                print(f"   ❌ 规划失败: {plan['error']}")
                break
            
            # 获取reasoning（CoT）
            reasoning = plan.get('reasoning', '')
            print(f"   💭 CoT: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")
            
            # ============ 开始新的step ============
            output_generator.start_step(reasoning)
            
            # 是否临近步数上限（尚未发现异常时才需要强制注入）
            approaching_limit = (step_num >= self.max_steps - 3)

            # ============ 检查是否需要批量操作 ============
            relevant_entities = self._get_relevant_entities(
                output_generator, 
                plan['tool_name']
            )
            
            if relevant_entities and len(relevant_entities) > 1 and should_batch_execute(reasoning, relevant_entities):
                # 批量操作
                print(f"   🔄 批量操作: 对 {len(relevant_entities)} 个实体执行 {plan['tool_name']}")
                
                for entity in relevant_entities:
                    # 更新参数
                    tool_request = self._update_tool_request_for_entity(
                        plan['tool_request'].copy(),
                        entity,
                        plan['tool_name']
                    )
                    
                    # 执行工具（传递CoT和question帮助World Model生成更准确的响应）
                    tool_response = world_model.execute_tool(
                        plan['tool_name'],
                        tool_request,
                        context=goal.get('entities', {}),
                        run_id=run_id,
                        force_anomaly=approaching_limit and not has_anomaly,
                        cot=reasoning,  # 传递规划器的思考过程
                        question=question  # 传递原始问题，保持一致性
                    )
                    
                    # 添加到输出
                    output_generator.add_action_observation(
                        plan['tool_name'],
                        tool_request,
                        tool_response,
                        batch=True
                    )
                    
                    # 更新状态（只添加第一个，避免重复）
                    if entity == relevant_entities[0]:
                        state.add_execution(
                            plan['tool_name'],
                            tool_request,
                            tool_response,
                            reasoning
                        )
                    
                    print(f"      ✓ 处理实体: {entity}")
            
            else:
                # 单次操作
                print(f"   🔧 执行工具: {plan['tool_name']}")
                
                tool_response = world_model.execute_tool(
                    plan['tool_name'],
                    plan['tool_request'],
                    context=goal.get('entities', {}),
                    run_id=run_id,
                    force_anomaly=approaching_limit and not has_anomaly,
                    cot=reasoning,  # 传递规划器的思考过程
                    question=question  # 传递原始问题，保持一致性
                )
                
                # 添加到输出
                output_generator.add_action_observation(
                    plan['tool_name'],
                    plan['tool_request'],
                    tool_response
                )
                
                # 更新状态
                state.add_execution(
                    plan['tool_name'],
                    plan['tool_request'],
                    tool_response,
                    reasoning
                )
                
                # ============ 提取新的实体 ============
                # 尝试提取接口
                interfaces = extract_entities_from_observation(tool_response, 'interface')
                if interfaces:
                    output_generator.update_known_entities('interfaces', interfaces)
                    print(f"   📋 发现接口: {len(interfaces)} 个 - {interfaces[:3]}{'...' if len(interfaces) > 3 else ''}")
                
                # 尝试提取设备
                devices = extract_entities_from_observation(tool_response, 'device')
                if devices:
                    output_generator.update_known_entities('devices', devices)
                    print(f"   📋 发现设备: {len(devices)} 个 - {devices[:3]}{'...' if len(devices) > 3 else ''}")
            
            # 4.4 分析结果并更新诊断链
            # ── 4.4  AnomalyJudge：解读工具响应，更新诊断链 ─────────────────
            # AnomalyJudge 统一承担"响应里有没有故障信号"的判断工作，
            # Planner 通过 diagnostic_chain 中的 conclusion 字段直接消费结论，
            # 无需自己在 prompt 里猜测数据含义。
            judge_result = self.anomaly_judge.judge(
                tool_name=plan['tool_name'],
                tool_request=plan['tool_request'],
                tool_response=tool_response,
                question=question,
                context=state.get_diagnostic_context()
            )
            
            if judge_result['has_anomaly']:
                severity_map = {"low": "low", "medium": "medium", "high": "high"}
                severity = severity_map.get(judge_result['severity'], 'medium')
                finding_text = judge_result['conclusion'] or judge_result['anomaly_type']
                state.add_finding(finding_text, severity)
                print(f"   🔴 AnomalyJudge 判定异常: [{judge_result['anomaly_type']}] {judge_result['evidence'][:80]}")
                if judge_result['suggested_next']:
                    print(f"   💡 建议下一步: {judge_result['suggested_next']}")
                has_anomaly = True
                anomaly_steps.append(step_num)
                if anomaly_found_step is None:
                    anomaly_found_step = step_num
                    print(f"   📌 首次发现异常（步骤 {step_num}），将继续深入排查至少 {_min_followup} 步")
            else:
                print(f"   ✅ AnomalyJudge: 本步响应正常")
            
            # 更新诊断链（conclusion 使用 judge 的结论，让 Planner 直接读取）
            state.update_diagnostic_chain(
                action=f"{plan['tool_name']} - {reasoning[:50]}{'...' if len(reasoning) > 50 else ''}",
                result=self._summarize_tool_result(tool_response),
                conclusion=judge_result['conclusion'] or self._generate_conclusion(tool_response, None),
                next_focus=judge_result['suggested_next'] or plan.get('next_focus', '')
            )

            # ============ CoT决策树推进 ============
            if self.cot_integration.is_active():
                # 标记当前节点为已访问（防止重复执行）
                self.cot_integration.mark_visited()

                # 检查是否需要调用子CoT
                should_call, target_cot = self.cot_integration.should_call_subcot()

                if should_call and target_cot:
                    # 检查条件是否满足（简化版：如果发现异常则调用）
                    node = self.cot_integration.get_current_node()
                    node_type = node.get('type') if node else None

                    if node_type == 'conditional_call':
                        # 条件调用：依据"本步判定出的异常类型"是否与该分支条件匹配来决定，
                        # 而非沿用全局 has_anomaly（后者会让任意一次异常后所有条件分支都被触发，导致子CoT反复拉起）。
                        node_intent = node.get('intent', '') if node else ''
                        condition_met = self._conditional_branch_matches(
                            node_intent=node_intent,
                            target_cot=target_cot,
                            judge_result=judge_result,
                        )
                        if condition_met:
                            print(f"   🔄 条件满足（异常类型匹配分支），调用子CoT: {target_cot}")
                            self.cot_integration.handle_cot_call(target_cot)
                        else:
                            print(f"   ⏭️  条件不满足（本步异常类型不匹配该分支），跳过子CoT调用")
                            self.cot_integration.advance_to_next_step(condition_result=False)
                    else:
                        # 无条件调用
                        print(f"   🔄 调用子CoT: {target_cot}")
                        self.cot_integration.handle_cot_call(target_cot)
                else:
                    # 前进到下一步
                    if not self.cot_integration.advance_to_next_step():
                        print(f"   ✅ CoT执行完成")

            print()
        # ============ 添加总结步骤 ============
        print(f"{'─'*80}")
        print(f"📝 生成总结和处置建议...")
        
        # 只传递包含异常的步骤用于总结
        if has_anomaly:
            print(f"   📊 使用包含异常的步骤: {anomaly_steps}")
        else:
            print(f"   ℹ️  未发现异常，将生成正常总结")
        
        summary_cot, summary_coa_list = self._generate_summary(
            question=question,
            all_steps=output_generator.steps,
            state=state,
            anomaly_steps=anomaly_steps if has_anomaly else None
        )
        
        # 添加总结步骤
        output_generator.start_step(summary_cot)
        
        # 添加所有检测节点的结果
        for coa_item in summary_coa_list:
            output_generator.add_action_observation(
                coa_item['action']['name'],
                coa_item['action']['args'],
                coa_item['observation']
            )
        
        print(f"   ✅ 总结完成 (包含 {len(summary_coa_list)} 个检测节点)")
        print()
        
        # ============ 生成最终输出（新格式） ============
        result = output_generator.generate_output(question)
        
        print(f"{'='*80}")
        print(f"✅ 完成! 总共 {len(result['response'])} 步 (含总结)")
        print(f"{'='*80}\n")
        
        return result

    def _conditional_branch_matches(self, node_intent: str, target_cot: str,
                                    judge_result: Dict[str, Any]) -> bool:
        """
        判断 conditional_call 分支条件是否成立（思路2 修复）。

        旧逻辑用全局 has_anomaly，会让任意一次异常之后所有条件分支都触发，导致子CoT被反复拉起。
        新逻辑：要求本步 AnomalyJudge 确实判定了异常，且该异常的类型/证据与当前分支描述
        （call_target 或节点 intent）存在关键词重叠，才认为该分支条件成立。

        Args:
            node_intent: 当前 conditional_call 节点的意图描述文本（含条件，如"如伴随错包，..."）
            target_cot: 该分支要调用的子 CoT 名称（如"物理接口错包异常"）
            judge_result: 本步 AnomalyJudge 的判定结果

        Returns:
            分支条件是否成立
        """
        # 本步没有判定出异常 → 条件不成立（避免无异常时盲目跳转子CoT）
        if not judge_result or not judge_result.get('has_anomaly'):
            return False

        anomaly_type = (judge_result.get('anomaly_type') or '').strip()
        evidence = (judge_result.get('evidence') or '').strip()
        signal_text = f"{anomaly_type} {evidence}"
        branch_text = f"{target_cot or ''} {node_intent or ''}"

        if not anomaly_type and not evidence:
            # 判定为异常但没有具体类型/证据信息 → 保守地认为匹配（回退到旧的"有异常即调用"行为，
            # 但仅在确实有异常时，比全局 has_anomaly 已收紧）
            return True

        # 关键词重叠匹配：从分支描述中抽取中文关键词，检查是否出现在异常信号文本中
        import re as _re
        # 网络诊断领域常见关键特征词；命中任一即视为该分支相关
        feature_keywords = [
            'down', 'up', '丢包', '错包', 'crc', '光功率', '光模块', '带宽', '流量',
            '聚合', '抖动', '频繁', '超阈值', '错误', '中断', '链路', '协议', '速率',
            '半双工', '双工', '环路', '广播', '风暴',
        ]
        signal_low = signal_text.lower()
        branch_low = branch_text.lower()

        # 1) 分支描述与异常信号共同命中的领域特征词
        for kw in feature_keywords:
            if kw in branch_low and kw in signal_low:
                return True

        # 2) 兜底：抽取分支描述里的中文双字以上词，看是否直接出现在异常信号中
        for token in _re.findall(r'[\u4e00-\u9fff]{2,}', branch_text):
            if token in signal_text:
                return True

        return False

    def _get_relevant_entities(self, generator: StructuredOutputGenerator, tool_name: str) -> List[str]:
        """获取与当前工具相关的实体列表"""
        tool_name_lower = tool_name.lower()
        
        if 'interface' in tool_name_lower:
            return generator.get_known_entities('interfaces')
        elif 'device' in tool_name_lower:
            return generator.get_known_entities('devices')
        
        return []
    
    def _update_tool_request_for_entity(self, request: Dict, entity: str, tool_name: str) -> Dict:
        """更新工具请求参数中的实体"""
        tool_name_lower = tool_name.lower()
        
        if 'interface' in tool_name_lower:
            request['interface_name'] = entity
        elif 'device' in tool_name_lower:
            request['device_name'] = entity
        
        return request
    
    def _analyze_tool_response(self, tool_name: str, response: Any) -> Optional[Dict]:
        """分析工具响应，提取关键发现"""
        if not response:
            return None
        
        finding = None
        
        # 根据不同工具类型分析
        if isinstance(response, dict):
            # 检查状态异常
            if response.get('status') == 'down' or response.get('状态') == 'down':
                finding = {
                    'type': 'anomaly',
                    'content': f"发现异常: 接口状态为down"
                }
            # 检查错误统计
            elif 'errors' in response or '错包' in str(response):
                finding = {
                    'type': 'anomaly',
                    'content': f"发现错包或错误统计异常"
                }
            # 正常情况
            else:
                finding = {
                    'type': 'normal',
                    'content': "数据获取成功，未发现明显异常"
                }
        
        elif isinstance(response, list):
            finding = {
                'type': 'info',
                'content': f"成功获取{len(response)}条记录"
            }
        
        return finding
    
    def _summarize_tool_result(self, result: Any) -> str:
        """总结工具结果（用于诊断链）"""
        if isinstance(result, list):
            return f"返回{len(result)}条记录"
        elif isinstance(result, dict):
            # 提取关键字段
            key_fields = []
            for key in ['status', '状态', 'errors', '错包', 'interface', '接口']:
                if key in result:
                    key_fields.append(f"{key}={result[key]}")
            if key_fields:
                return ", ".join(key_fields[:3])
            return "数据获取成功"
        return str(result)[:50]
    
    def _generate_conclusion(self, response: Any, finding: Optional[Dict]) -> str:
        """生成结论"""
        if finding:
            if finding['type'] == 'anomaly':
                return f"发现异常: {finding['content']}"
            elif finding['type'] == 'normal':
                return "正常，无异常"
            else:
                return finding['content']
        return "已执行"
    
    def _generate_summary(self, question: str, all_steps: List[Dict], state: StateManager,
                         anomaly_steps: List[int] = None) -> tuple:
        """
        生成诊断总结和处置建议（改进版：传递完整内容给LLM分析）
        
        Args:
            question: 原始问题
            all_steps: 所有执行的步骤
            state: 状态管理器
            anomaly_steps: 包含异常的步骤编号列表（从1开始）
            
        Returns:
            (cot, summary_coa): CoT描述和总结内容列表
        """
        # 在方法开头导入需要的模块
        import json
        import openai
        
        # 构建完整的诊断过程描述
        diagnostic_process = []
        
        for i, step_dict in enumerate(all_steps, 1):
            step_key = f"step{i}"
            if step_key in step_dict:
                step_data = step_dict[step_key]
                cot = step_data.get('cot', '')
                coa = step_data.get('coa', [])
                
                step_info = {
                    'step_num': i,
                    'cot': cot,
                    'actions': []
                }
                
                for action_obs in coa:
                    action = action_obs.get('action', {})
                    observation = action_obs.get('observation', {})
                    
                    step_info['actions'].append({
                        'tool': action.get('name', ''),
                        'args': action.get('args', {}),
                        'observation': observation
                    })
                
                diagnostic_process.append(step_info)
        
        # 构建详细的诊断过程文本
        process_text = ""
        for step_info in diagnostic_process:
            process_text += f"\n【Step {step_info['step_num']}】\n"
            process_text += f"思考: {step_info['cot']}\n"
            
            for idx, action_info in enumerate(step_info['actions'], 1):
                process_text += f"  操作{idx}: {action_info['tool']}\n"
                process_text += f"    参数: {json.dumps(action_info['args'], ensure_ascii=False)}\n"
                process_text += f"    观察: {json.dumps(action_info['observation'], ensure_ascii=False)}\n"
        
        # 构建prompt - 让LLM分析完整流程
        prompt = f"""你是一个网络故障诊断专家。请仔细分析以下完整的故障诊断流程，并生成**总体诊断结论**。

【原始问题】
{question}

【完整诊断流程】
{process_text}

【任务要求】
1. **仔细阅读**上述诊断流程，识别是否存在故障或异常
2. **异常判断标准**：
   - 接口状态为down、error、异常
   - 存在错包、丢包、CRC错误
   - 设备不可达、连接失败
   - 任何明显的性能问题或配置错误

3. **输出格式**：
   - **只生成一个总体结论节点**，不要逐个节点分析
   - 总结整个诊断过程发现的主要问题
   - 提供针对性的整体修复建议

4. **输出JSON格式**（单个对象，不是数组）：
```json
{{
  "节点名称": "总体诊断结论",
  "检测项": "整体诊断总结",
  "状态": "发现异常" 或 "诊断完成",
  "原因": "简要总结诊断过程中发现的主要问题，包括具体的设备/接口名称和问题类型",
  "修复建议": "针对发现问题的整体修复建议"
}}
```

**重要提示：**
- 异常信息必须基于上述诊断流程中的**实际观察数据**，不要编造
- 如果observation显示状态为"down"、有"错包"等，就是异常
- 如果observation显示状态为"up"、"正常"，就是正常
- 原因要**简洁明了**，直接说明发现了什么问题（如"发现接口40GE2/2/5状态down，存在12543个CRC错包"）
- 修复建议要**针对性强**，直接给出解决方案
- **只输出一个总体结论节点，不要列出每个检测节点**

只输出JSON对象，不要有其他文字。"""
        
        try:
            # 调用LLM生成总结
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base if self.api_base.rstrip('/').endswith('/v1') else self.api_base + "/v1"
            )
            
            response = client.chat.completions.create(
                model="Qwen2.5-72B",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            summary_text = response.choices[0].message.content.strip()
            
            # 清理可能的markdown标记
            if summary_text.startswith('```json'):
                summary_text = summary_text.split('```json')[1]
            if summary_text.startswith('```'):
                summary_text = summary_text.split('```')[1]
            if summary_text.endswith('```'):
                summary_text = summary_text.rsplit('```', 1)[0]
            summary_text = summary_text.strip()
            
            # 解析JSON对象（现在是单个对象，不是数组）
            summary_node = json.loads(summary_text)
            
            # 如果返回的是数组，取第一个（兼容旧格式）
            if isinstance(summary_node, list):
                if len(summary_node) > 0:
                    # 找总体结论节点
                    for node in summary_node:
                        if '总体' in node.get('节点名称', '') or '结论' in node.get('节点名称', ''):
                            summary_node = node
                            break
                    else:
                        summary_node = summary_node[0]
                else:
                    summary_node = {
                        "节点名称": "总体诊断结论",
                        "状态": "诊断完成",
                        "原因": "诊断流程完成",
                        "修复建议": "请根据诊断结果采取相应措施"
                    }
            
            # 验证总结质量
            status = summary_node.get('状态', '').lower()
            has_anomaly_in_summary = any(k in status for k in ['异常', 'down', 'error', 'abnormal', '发现'])
            
            # 如果有异常步骤但总结说没有，打印警告
            if anomaly_steps and not has_anomaly_in_summary:
                print(f"   ⚠️  警告：检测到异常步骤{anomaly_steps}，但总结未识别异常")
                print(f"   💡 总结内容：{summary_node}")
            
            # 转换为coa格式（只有一个节点）
            summary_coa = [{
                "action": {
                    "name": "node_check",
                    "args": {
                        "node": summary_node.get("节点名称", "总体诊断结论"),
                        "check_item": summary_node.get("检测项", "整体诊断总结")
                    }
                },
                "observation": {
                    "状态": summary_node.get("状态", "未知"),
                    "原因": summary_node.get("原因", ""),
                    "修复建议": summary_node.get("修复建议", "")
                }
            }]
            
        except Exception as e:
            print(f"   ⚠️ 总结生成失败，使用默认模板: {e}")
            
            # 如果有异常步骤，生成基本的异常总结
            if anomaly_steps:
                summary_coa = [
                    {
                        "action": {
                            "name": "node_check",
                            "args": {
                                "node": f"Step {anomaly_steps} 发现异常",
                                "check_item": "异常检测"
                            }
                        },
                        "observation": {
                            "状态": "异常",
                            "原因": f"在第{anomaly_steps}步检测到异常",
                            "修复建议": "请根据诊断结果排查问题"
                        }
                    },
                    {
                        "action": {
                            "name": "node_check",
                            "args": {
                                "node": "总体结论",
                                "check_item": "整体诊断总结"
                            }
                        },
                        "observation": {
                            "状态": "发现异常",
                            "原因": "诊断流程中发现故障",
                            "修复建议": "请根据具体异常采取相应措施"
                        }
                    }
                ]
            else:
                # 使用默认模板
                summary_coa = [
                    {
                        "action": {
                            "name": "node_check",
                            "args": {
                                "node": "总体结论",
                                "check_item": "整体诊断总结"
                            }
                        },
                        "observation": {
                            "状态": "诊断完成",
                            "原因": f"已完成{len(all_steps)}步诊断流程",
                            "修复建议": "请根据诊断结果采取相应措施"
                        }
                    }
                ]
        
        cot = "总结分析报告，并给出处置建议"
        
        return cot, summary_coa
    
    def _extract_entity_name(self, observation: Any, action: Dict) -> str:
        """
        从observation和action中提取实体名称
        
        Returns:
            实体名称，如"设备XXX的接口YYY"或"设备XXX"
        """
        if isinstance(observation, dict):
            # 尝试提取接口信息
            device = observation.get('设备') or observation.get('device_name') or action.get('args', {}).get('device_name')
            interface = observation.get('接口') or observation.get('interface_name') or action.get('args', {}).get('interface_name')
            
            if device and interface:
                return f"设备{device}的接口{interface}"
            elif interface:
                return f"接口{interface}"
            elif device:
                return f"设备{device}"
            
            # 尝试提取IP
            ip = observation.get('IP地址') or observation.get('ip')
            if ip:
                return f"IP {ip}"
        
        return ""



    def generate_batch(self,
                      question: str,
                      n_runs: int = 10,
                      output_dir: str = '/mnt/user-data/outputs/batch_runs',
                      rewrite_question: bool = False) -> List[Dict[str, Any]]:
        """
        批量生成多条数据
        
        Args:
            question: 问题描述
            n_runs: 运行次数
            output_dir: 输出目录
            rewrite_question: 是否改写问题
            
        Returns:
            所有运行的结果列表
        """
        print(f"\n{'='*80}")
        print(f"🎯 批量生成: {n_runs} 条数据")
        if rewrite_question:
            print(f"📝 启用问题改写以增加多样性（第一次运行使用原问题）")
        print(f"{'='*80}\n")
        
        results = []
        
        for i in range(n_runs):
            # 生成运行配置
            config = self._generate_run_config(i, n_runs)
            config['total_runs'] = n_runs
            
            # 执行生成
            result = self.generate(
                question,
                config,
                rewrite_question=rewrite_question
            )
            results.append(result)
            
            # 保存单个结果
            output_file = f"{output_dir}/run_{i+1:03d}.json"
            self.save_result(result, output_file)
            
            # 短暂延迟
            time.sleep(1)
        
        # 保存汇总
        self._save_batch_summary(results, question, output_dir)
        
        return results
    
    def _generate_run_config(self, run_id: int, total_runs: int) -> Dict:
        """
        为每次运行生成不同的配置
        
        策略：
        - 前30%: greedy + low diversity
        - 中40%: balanced + medium diversity
        - 后30%: exploratory + high diversity
        """
        ratio = run_id / total_runs
        
        if ratio < 0.3:
            return {
                "run_id": run_id,
                "exploration_mode": "greedy",
                "diversity_mode": "low",
                "temperature": 0.5
            }
        elif ratio < 0.7:
            return {
                "run_id": run_id,
                "exploration_mode": "balanced",
                "diversity_mode": "medium",
                "temperature": 0.7
            }
        else:
            return {
                "run_id": run_id,
                "exploration_mode": "exploratory",
                "diversity_mode": "high",
                "temperature": 0.9
            }
    
    def save_result(self, result: Dict, output_file: str):
        """保存单个结果"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"💾 已保存: {output_file}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    def _save_batch_summary(self, results: List[Dict], question: str, output_dir: str):
        """保存批量运行的汇总（适配新格式）"""
        
        # 从新格式中提取统计信息
        def extract_statistics(result: Dict) -> Dict:
            """从新格式中提取统计信息"""
            steps = result.get('response', [])
            total_steps = len(steps)
            
            # 统计工具使用
            all_tools = []
            for step_dict in steps:
                for step_key, step_data in step_dict.items():
                    coa = step_data.get('coa', [])
                    for action_obs in coa:
                        tool_name = action_obs.get('action', {}).get('name')
                        if tool_name:
                            all_tools.append(tool_name)
            
            return {
                'total_steps': total_steps,
                'total_tools': len(all_tools),
                'diagnostic_path': all_tools
            }
        
        # 为每个结果添加统计信息
        processed_results = []
        for i, result in enumerate(results):
            stats = extract_statistics(result)
            processed_results.append({
                'run_id': i,
                'result': result,
                'statistics': stats,
                'summary': {
                    'diagnostic_path': stats['diagnostic_path']
                }
            })
        
        # 生成汇总
        summary = {
            "question": question,
            "total_runs": len(results),
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "avg_steps": sum(r['statistics']['total_steps'] for r in processed_results) / len(results),
                "avg_tools": sum(r['statistics']['total_tools'] for r in processed_results) / len(results),
                "step_distribution": [r['statistics']['total_steps'] for r in processed_results],
                "unique_paths": len(set(
                    tuple(r['summary']['diagnostic_path']) for r in processed_results
                ))
            },
            "runs": [
                {
                    "run_id": r['run_id'],
                    "steps": r['statistics']['total_steps'],
                    "tools": r['statistics']['total_tools'],
                    "path": r['summary']['diagnostic_path']
                }
                for r in processed_results
            ]
        }
        
        output_file = f"{output_dir}/batch_summary.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n💾 批量汇总已保存: {output_file}")
            
            # 打印汇总统计
            print(f"\n{'='*80}")
            print(f"📊 批量运行统计")
            print(f"{'='*80}")
            print(f"总运行数: {summary['total_runs']}")
            print(f"平均步骤: {summary['statistics']['avg_steps']:.1f}")
            print(f"平均工具调用: {summary['statistics']['avg_tools']:.1f}")
            print(f"唯一路径: {summary['statistics']['unique_paths']}")
            print(f"步骤分布: {summary['statistics']['step_distribution']}")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"❌ 保存汇总失败: {e}")


if __name__ == '__main__':
    print("Agent Generator with new format and batch operations")
    print("请使用 batch_generate.py 调用")
