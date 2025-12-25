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
                        run_id=run_id
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
                    run_id=run_id
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
            
            # 更新诊断链
            state.update_diagnostic_chain(
                action=f"{plan['tool_name']} - {reasoning[:50]}{'...' if len(reasoning) > 50 else ''}",
                result=self._summarize_tool_result(tool_response),
                conclusion=self._generate_conclusion(tool_response, finding),
                next_focus=plan.get('next_focus', '')
            )
            
            print()
        
        # ============ 生成最终输出（新格式） ============
        result = output_generator.generate_output(question)
        
        print(f"{'='*80}")
        print(f"✅ 完成! 总共 {len(result['response'])} 步")
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


if __name__ == '__main__':
    print("Agent Generator with new format and batch operations")
    print("请使用 batch_generate.py 调用")
