"""
增强的Agent Generator - 支持批量操作和结构化输出
"""

import json
import sys
sys.path.insert(0, '/mnt/user-data/outputs')

from typing import Dict, List, Any, Optional
from tool_manager import ToolManager
from state_manager import StateManager
from enhanced_planner import EnhancedPlanner
from world_model import WorldModel
from structured_output import (
    StructuredOutputGenerator, 
    extract_entities_from_observation,
    should_batch_execute
)


class BatchAwareAgentGenerator:
    """支持批量操作的Agent生成器"""
    
    def __init__(self, 
                 tool_manager: ToolManager,
                 world_model: WorldModel,
                 api_key: str,
                 api_base: str = None,
                 model: str = "gpt-4o-mini",
                 max_steps: int = 20):
        """
        初始化生成器
        
        Args:
            tool_manager: 工具管理器
            world_model: 世界模型
            api_key: API密钥
            api_base: API基础URL
            model: 模型名称
            max_steps: 最大步骤数
        """
        self.tool_manager = tool_manager
        self.world_model = world_model
        self.api_key = api_key
        self.api_base = api_base or "http://10.12.208.86:8502"
        self.model = model
        self.max_steps = max_steps
    
    def generate(self, question: str, knowledge_base: Dict, run_config: Dict = None) -> Dict:
        """
        生成一次诊断数据（支持批量操作）
        
        Args:
            question: 问题描述
            knowledge_base: 知识库
            run_config: 运行配置
            
        Returns:
            结构化输出
        """
        print(f"\n{'='*80}")
        print(f"开始生成：{question}")
        print(f"{'='*80}\n")
        
        # 初始化
        output_generator = StructuredOutputGenerator()
        state = StateManager()
        planner = EnhancedPlanner(
            self.tool_manager,
            self.api_key,
            self.api_base,
            self.model
        )
        
        # 提取目标
        goal = self._extract_goal(question, knowledge_base)
        
        # 主循环
        while state.step_count < self.max_steps:
            current_step = state.step_count + 1
            print(f"{'─'*80}")
            print(f"Step {current_step}:")
            
            # 规划下一步（传入已知实体）
            known_entities_dict = {
                'interfaces': output_generator.get_known_entities('interfaces'),
                'devices': output_generator.get_known_entities('devices')
            }
            # 过滤空列表
            known_entities_dict = {k: v for k, v in known_entities_dict.items() if v}
            
            plan = planner.select_next_tool(
                state,
                goal,
                temperature=run_config.get('temperature', 0.7) if run_config else 0.7,
                known_entities=known_entities_dict if known_entities_dict else None
            )
            
            if 'error' in plan:
                print(f"   ❌ 规划失败: {plan['error']}")
                break
            
            # 获取reasoning
            reasoning = plan.get('reasoning', '')
            print(f"   💭 CoT: {reasoning[:100]}...")
            
            # 开始新的step
            output_generator.start_step(reasoning)
            
            # 检查是否需要批量操作
            known_entities = self._get_relevant_entities(output_generator, plan['tool_name'])
            
            if known_entities and len(known_entities) > 1 and should_batch_execute(reasoning, known_entities):
                # 批量操作：对每个实体执行相同的工具
                print(f"   🔄 批量操作: 对 {len(known_entities)} 个实体执行 {plan['tool_name']}")
                
                for entity in known_entities:
                    # 更新请求参数中的实体
                    tool_request = self._update_tool_request_for_entity(
                        plan['tool_request'],
                        entity,
                        plan['tool_name']
                    )
                    
                    # 执行工具
                    tool_response = self.world_model.execute_tool(
                        plan['tool_name'],
                        tool_request,
                        context=goal.get('entities', {}),
                        run_id=f"run_{current_step}_{entity}"
                    )
                    
                    # 添加到输出
                    output_generator.add_action_observation(
                        plan['tool_name'],
                        tool_request,
                        tool_response,
                        batch=True
                    )
                    
                    print(f"      ✓ 处理实体: {entity}")
            
            else:
                # 单次操作
                print(f"   🔧 执行工具: {plan['tool_name']}")
                
                tool_response = self.world_model.execute_tool(
                    plan['tool_name'],
                    plan['tool_request'],
                    context=goal.get('entities', {}),
                    run_id=f"run_{current_step}"
                )
                
                # 添加到输出
                output_generator.add_action_observation(
                    plan['tool_name'],
                    plan['tool_request'],
                    tool_response
                )
                
                # 提取新的实体
                entities = extract_entities_from_observation(tool_response, 'interface')
                if entities:
                    output_generator.update_known_entities('interfaces', entities)
                    print(f"   📋 发现实体: {len(entities)} 个 - {entities}")
            
            # 更新状态
            state.add_execution(
                plan['tool_name'],
                plan['tool_request'],
                tool_response,
                reasoning
            )
            
            state.update_diagnostic_chain(
                action=f"{plan['tool_name']} - {reasoning[:50]}...",
                result=self._summarize_result(tool_response),
                conclusion=self._analyze_result(tool_response),
                next_focus=plan.get('next_focus', '')
            )
            
            # 检查是否应该继续
            should_continue, reason = state.should_continue(self.max_steps)
            if not should_continue:
                print(f"\n🛑 停止: {reason}")
                break
        
        # 生成最终输出
        result = output_generator.generate_output(question)
        
        print(f"\n{'='*80}")
        print(f"✅ 完成! 总共 {len(result['response'])} 步")
        print(f"{'='*80}\n")
        
        return result
    
    def _extract_goal(self, question: str, knowledge_base: Dict) -> Dict:
        """从问题和知识库中提取目标"""
        # 简化实现
        return {
            "main_goal": question,
            "problem_type": "故障诊断",
            "key_aspects": ["接口状态", "故障定位"],
            "entities": {},
            "context_params": {}
        }
    
    def _get_relevant_entities(self, generator: StructuredOutputGenerator, tool_name: str) -> List[str]:
        """获取与当前工具相关的实体列表"""
        # 根据工具名判断需要什么类型的实体
        if 'interface' in tool_name.lower():
            return generator.get_known_entities('interfaces')
        elif 'device' in tool_name.lower():
            return generator.get_known_entities('devices')
        return []
    
    def _update_tool_request_for_entity(self, original_request: Dict, entity: str, tool_name: str) -> Dict:
        """更新工具请求参数中的实体"""
        request = original_request.copy()
        
        # 根据工具名判断应该更新哪个参数
        if 'interface' in tool_name.lower():
            request['interface_name'] = entity
        elif 'device' in tool_name.lower():
            request['device_name'] = entity
        
        return request
    
    def _summarize_result(self, result: Any) -> str:
        """总结结果"""
        if isinstance(result, list):
            return f"返回{len(result)}条记录"
        elif isinstance(result, dict):
            keys = list(result.keys())[:3]
            return f"包含字段: {', '.join(keys)}"
        return str(result)[:50]
    
    def _analyze_result(self, result: Any) -> str:
        """分析结果"""
        if isinstance(result, list):
            return f"成功获取{len(result)}条数据"
        elif isinstance(result, dict):
            if result.get('status') == 'down' or result.get('状态') == 'down':
                return "发现异常：接口状态为down"
            return "数据获取成功"
        return "已执行"


if __name__ == '__main__':
    print("测试批量操作Agent Generator")
    print("="*80 + "\n")
    
    # 模拟测试
    print("注意：这只是结构测试，实际运行需要完整的环境")
