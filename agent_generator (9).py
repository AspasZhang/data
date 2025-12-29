"""
Agent生成器（Agent Generator）- 集成新格式和批量操作
整合所有组件，实现自由探索的故障诊断数据生成
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from goal_extractor import GoalExtractor
from state_manager import StateManager
from enhanced_planner import EnhancedPlanner
from enhanced_world_model import EnhancedWorldModel
from tool_manager import ToolManager
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
        self.api_base = api_base or "http://10.12.208.86:8502"
        self.knowledge_base = knowledge_base
        self.max_steps = max_steps
        
        # 初始化组件
        self.goal_extractor = GoalExtractor(api_key, api_base)
        
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
            diversity_mode=run_config.get('diversity_mode', 'medium')
        )
        
        # 3. 初始化状态
        state = StateManager()
        
        # 异常检测标志
        has_anomaly = False
        anomaly_steps = []  # 记录包含异常的步骤
        
        # 4. 迭代探索
        print("🔍 步骤2: 开始迭代探索...\n")
        
        while True:
            # 检查是否应该继续
            should_continue, reason = state.should_continue(self.max_steps)
            
            if not should_continue:
                print(f"\n🛑 停止探索: {reason}\n")
                break
            
            step_num = state.step_count + 1
            print(f"{'─'*80}")
            print(f"Step {step_num}:")
            
            # 检查是否接近最大步骤且没有异常
            approaching_limit = step_num >= self.max_steps - 2
            if approaching_limit and not has_anomaly:
                print(f"   ⚠️  接近最大步骤({self.max_steps})且未发现异常，将强制生成异常数据")
            
            # ============ 获取已知实体列表 ============
            known_entities_dict = {
                'interfaces': output_generator.get_known_entities('interfaces'),
                'devices': output_generator.get_known_entities('devices')
            }
            # 过滤空列表
            known_entities_dict = {k: v for k, v in known_entities_dict.items() if v}
            
            # 4.1 规划下一步（传入已知实体）
            plan = planner.select_next_tool(
                state, 
                goal, 
                temperature=run_config.get('temperature', 0.7),
                known_entities=known_entities_dict if known_entities_dict else None
            )
            
            if 'error' in plan:
                print(f"   ❌ 规划失败: {plan['error']}")
                break
            
            # 获取reasoning（CoT）
            reasoning = plan.get('reasoning', '')
            print(f"   💭 CoT: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")
            
            # ============ 开始新的step ============
            output_generator.start_step(reasoning)
            
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
                    
                    # 执行工具
                    tool_response = world_model.execute_tool(
                        plan['tool_name'],
                        tool_request,
                        context=goal.get('entities', {}),
                        run_id=run_id,
                        force_anomaly=approaching_limit and not has_anomaly
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
                    force_anomaly=approaching_limit and not has_anomaly
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
            finding = self._analyze_tool_response(
                plan['tool_name'],
                tool_response
            )
            
            if finding:
                state.add_finding(finding['type'], finding['content'])
                print(f"   📌 发现: {finding['content'][:80]}{'...' if len(finding['content']) > 80 else ''}")
                
                # 检查是否发现异常
                if finding['type'] == 'anomaly':
                    has_anomaly = True
                    anomaly_steps.append(step_num)
                    print(f"   🔴 发现异常！记录步骤 {step_num}")
            
            # 更新诊断链
            state.update_diagnostic_chain(
                action=f"{plan['tool_name']} - {reasoning[:50]}{'...' if len(reasoning) > 50 else ''}",
                result=self._summarize_tool_result(tool_response),
                conclusion=self._generate_conclusion(tool_response, finding),
                next_focus=plan.get('next_focus', '')
            )
            
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
        生成诊断总结和处置建议（只总结重要节点）
        
        Args:
            question: 原始问题
            all_steps: 所有执行的步骤
            state: 状态管理器
            anomaly_steps: 包含异常的步骤编号列表（从1开始）
            
        Returns:
            (cot, summary_coa): CoT描述和总结内容列表
        """
        # 如果没有异常步骤，生成"未发现异常"的总结
        if not anomaly_steps:
            print("   ℹ️  未发现异常，生成正常总结")
            cot = "总结分析报告"
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
                        "原因": f"已完成{len(all_steps)}步诊断流程，所有检测项目均正常，未发现异常",
                        "修复建议": "系统运行正常，无需处理，建议定期监控"
                    }
                }
            ]
            return cot, summary_coa
        
        # 只从异常相关的步骤中提取信息
        print(f"   📊 从步骤 {anomaly_steps} 中提取异常信息")
        
        # 收集诊断过程中的关键信息（只从异常步骤）
        important_findings = []
        all_entities = set()
        
        # 从state中获取发现的异常
        findings = state.findings
        
        # 只分析包含异常的步骤
        for i, step_dict in enumerate(all_steps, 1):
            # 跳过非异常步骤
            if i not in anomaly_steps:
                continue
            
            step_key = f"step{i}"
            if step_key in step_dict:
                step_data = step_dict[step_key]
                coa = step_data.get('coa', [])
                
                for action_obs in coa:
                    observation = action_obs.get('observation', {})
                    
                    # 提取实体（设备、接口等）
                    if isinstance(observation, dict):
                        # 检查是否有异常状态
                        status = observation.get('状态') or observation.get('status')
                        if status and status.lower() in ['down', 'error', '异常', 'abnormal']:
                            # 这是一个异常节点，记录下来
                            entity_name = self._extract_entity_name(observation, action_obs.get('action', {}))
                            if entity_name:
                                important_findings.append({
                                    'entity': entity_name,
                                    'observation': observation,
                                    'step': i
                                })
                        
                        # 记录所有实体
                        entity = self._extract_entity_name(observation, action_obs.get('action', {}))
                        if entity:
                            all_entities.add(entity)
        
        # 构建诊断过程摘要
        execution_summary = f"诊断过程共{len(all_steps)}步，从第{anomaly_steps}步发现异常"
        
        # 构建发现摘要（findings是一个list，不是dict）
        findings_summary = ""
        if findings:  # findings是list，直接判断是否为空
            # 提取异常类型的发现
            anomalies = [f for f in findings if f.get('type') == 'anomaly']
            if anomalies:
                findings_summary += f"\n发现{len(anomalies)}个异常："
                for finding in anomalies[:3]:  # 最多列3个
                    findings_summary += f"\n  - {finding.get('content', '')}"
        
        # 构建prompt（只要求总结重要的节点）
        prompt = f"""你是一个网络故障诊断专家。请基于以下诊断结果，生成一份**简洁**的总结报告。

【原始问题】
{question}

【诊断摘要】
{execution_summary}
{findings_summary}

【重要发现】
{chr(10).join([f"- {item['entity']}: {item['observation']}" for item in important_findings[:5]])}

**要求：**
1. **只总结有问题或重要的**设备/接口/IP，正常的不需要列出
2. 内容简洁，每个节点的描述不超过50字
3. 如果没有异常，就只返回总体结论

输出JSON数组格式：
```json
[
  {{
    "节点名称": "有问题的设备/接口/IP",
    "检测项": "检测内容",
    "状态": "异常",
    "原因": "简短说明原因",
    "修复建议": "简短的修复建议"
  }},
  {{
    "节点名称": "总体结论",
    "检测项": "整体诊断总结",
    "状态": "诊断完成",
    "原因": "主要问题总结",
    "修复建议": "整体建议"
  }}
]
```

**如果所有节点都正常，只返回总体结论一个节点。**
只输出JSON数组，不要有其他文字。"""
        
        try:
            # 调用LLM生成总结
            import openai
            
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base + "/v1"
            )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
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
            
            # 解析JSON数组
            import json
            summary_nodes = json.loads(summary_text)
            
            # 限制节点数量（最多5个，包含总体结论）
            if len(summary_nodes) > 5:
                # 保留前4个重要节点 + 总体结论
                important_nodes = [n for n in summary_nodes if '总体结论' not in n.get('节点名称', '')][:4]
                conclusion_nodes = [n for n in summary_nodes if '总体结论' in n.get('节点名称', '')]
                summary_nodes = important_nodes + conclusion_nodes
            
            # 转换为coa格式
            summary_coa = []
            for node in summary_nodes:
                summary_coa.append({
                    "action": {
                        "name": "node_check",
                        "args": {
                            "node": node.get("节点名称", "未知节点"),
                            "check_item": node.get("检测项", "")
                        }
                    },
                    "observation": {
                        "状态": node.get("状态", "未知"),
                        "原因": node.get("原因", ""),
                        "修复建议": node.get("修复建议", "")
                    }
                })
            
        except Exception as e:
            print(f"   ⚠️ 总结生成失败，使用默认模板: {e}")
            # 使用默认模板（只有总体结论）
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
