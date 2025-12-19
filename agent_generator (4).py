"""
Agent生成器（Agent Generator）
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
        生成一次诊断数据
        
        Args:
            question: 问题描述
            run_config: 运行配置 {
                "run_id": 1,
                "exploration_mode": "balanced",
                "diversity_mode": "medium",
                "temperature": 0.7,
                "total_runs": 10
            }
            rewrite_question: 是否改写问题以增加多样性（第一次运行始终使用原问题）
            
        Returns:
            生成的诊断数据
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
        original_question = question  # 保存原始问题
        
        print(f"\n{'='*80}")
        print(f"🚀 开始运行 #{run_id + 1}")
        print(f"{'='*80}")
        
        # 0. 问题改写（如果启用）
        if rewrite_question and run_id > 0:  # 第一次运行使用原问题
            print(f"📝 步骤0: 改写问题以增加多样性...")
            from question_rewriter import QuestionRewriter
            
            # 初始化改写器（如果还没有）
            if not hasattr(self, 'question_rewriter'):
                self.question_rewriter = QuestionRewriter(
                    api_key=self.api_key,
                    api_base=self.api_base
                )
            
            # 使用策略改写
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
            
            # 4.1 规划下一步
            plan = planner.select_next_tool(
                state, 
                goal, 
                temperature=run_config.get('temperature', 0.7)
            )
            
            if 'error' in plan:
                print(f"   ❌ 规划失败: {plan['error']}")
                break
            
            # 4.2 执行工具
            print(f"   🔧 执行工具: {plan['tool_name']}")
            
            tool_response = world_model.execute_tool(
                plan['tool_name'],
                plan['tool_request'],
                context=goal.get('entities', {}),
                run_id=run_id
            )
            
            # 4.3 更新状态
            state.add_execution(
                plan['tool_name'],
                plan['tool_request'],
                tool_response,
                plan.get('reasoning', '')
            )
            
            # 4.4 分析结果并更新诊断链
            finding = self._analyze_tool_response(
                plan['tool_name'],
                tool_response
            )
            
            # 构建诊断链条目
            action = f"{plan['tool_name']} - {plan.get('reasoning', '未说明原因')}"
            result_summary = self._summarize_tool_result(tool_response)
            
            if finding:
                conclusion = f"发现: {finding['description']}"
                state.add_finding(finding['description'], finding['severity'])
                print(f"   🔍 发现: {finding['description']} (严重度: {finding['severity']})")
                
                # 如果是严重问题，可能需要排除某些原因或聚焦新方向
                if finding['severity'] == 'high':
                    next_focus = plan.get('next_focus', '继续深入分析此问题')
                else:
                    next_focus = plan.get('next_focus')
            else:
                conclusion = f"正常 - {result_summary}"
                next_focus = plan.get('next_focus', '继续诊断')
            
            # 更新诊断链
            state.update_diagnostic_chain(
                action=action,
                result=result_summary,
                conclusion=conclusion,
                next_focus=next_focus
            )
        
        # 5. 生成总结
        print("📝 步骤3: 生成诊断总结...")
        summary = self._generate_summary(state, goal)
        
        # 6. 构建结果
        result = {
            "run_id": run_id,
            "question": question,
            "goal": goal,
            "configuration": run_config,
            "execution_records": state.get_execution_records(),
            "findings": state.findings,
            "summary": summary,
            "statistics": state.get_summary(),
            "timestamp": datetime.now().isoformat()
        }
        
        # 如果进行了问题改写，添加相关信息
        if rewrite_question and question != original_question:
            result["question_rewrite"] = {
                "original": original_question,
                "rewritten": question
            }
        
        # 显示统计
        print(f"\n{'='*80}")
        print(f"✅ 运行完成")
        print(f"{'='*80}")
        print(f"总步骤: {result['statistics']['total_steps']}")
        print(f"使用工具: {result['statistics']['unique_tools_used']}")
        print(f"发现问题: {result['statistics']['total_findings']}")
        print(f"关键发现: {result['statistics']['critical_findings']}")
        print(f"{'='*80}\n")
        
        return result
    
    def _analyze_tool_response(self, tool_name: str, tool_response: Dict) -> Optional[Dict]:
        """
        分析工具响应，判断是否发现问题
        
        Returns:
            {
                "description": "发现描述",
                "severity": "low/medium/high"
            } 或 None
        """
        if 'error' in tool_response:
            return None
        
        # 简单的启发式规则
        findings = []
        
        # 检查响应中的异常信号
        response_str = json.dumps(tool_response, ensure_ascii=False).lower()
        
        # 关键词检测
        if any(keyword in response_str for keyword in ['异常', '错误', '超阈值', '告警', 'error', 'alarm']):
            findings.append({
                "description": f"{tool_name}检测到异常信号",
                "severity": "medium"
            })
        
        if any(keyword in response_str for keyword in ['严重', '紧急', 'critical', 'severe']):
            findings.append({
                "description": f"{tool_name}检测到严重问题",
                "severity": "high"
            })
        
        # 返回第一个发现（如果有）
        return findings[0] if findings else None
    
    def _summarize_tool_result(self, tool_response: Dict) -> str:
        """
        总结工具执行结果为简短描述
        
        Args:
            tool_response: 工具响应
            
        Returns:
            简短的结果摘要
        """
        if not tool_response or 'error' in tool_response:
            return "执行失败或无结果"
        
        # 提取关键信息
        key_fields = []
        for key, value in tool_response.items():
            if key in ['error', 'status_code']:
                continue
            # 只保留前3个字段或重要字段
            if len(key_fields) < 3 or key in ['status', 'state', 'result']:
                key_fields.append(f"{key}={value}")
        
        if key_fields:
            summary = ", ".join(key_fields[:3])
            if len(summary) > 100:
                summary = summary[:97] + "..."
            return summary
        
        return "已执行"
    
    def _generate_summary(self, state: StateManager, goal: Dict) -> Dict[str, Any]:
        """生成诊断总结"""
        return {
            "goal_achieved": len(state.findings) > 0,
            "main_findings": [f['finding'] for f in state.findings[:3]],  # 前3个发现
            "diagnostic_path": [tool['tool_name'] for tool in state.executed_tools],
            "diagnostic_chain": state.diagnostic_chain,  # 新增：完整的诊断逻辑链
            "current_focus": state.current_focus,  # 新增：最终焦点
            "excluded_causes": state.excluded_causes,  # 新增：已排除原因
            "conclusion": self._generate_conclusion(state, goal)
        }
    
    def _generate_conclusion(self, state: StateManager, goal: Dict) -> str:
        """生成结论"""
        if not state.findings:
            return "未发现明显异常，系统状态正常。"
        
        critical_findings = [f for f in state.findings if f['severity'] == 'high']
        
        if critical_findings:
            return f"发现{len(critical_findings)}个关键问题，需要立即处理。"
        else:
            return f"发现{len(state.findings)}个需要关注的问题。"
    
    def generate_batch(self, 
                      question: str, 
                      n_runs: int = 10, 
                      output_dir: str = "/mnt/user-data/outputs",
                      rewrite_question: bool = False) -> List[Dict]:
        """
        批量生成多条数据
        
        Args:
            question: 问题描述
            n_runs: 运行次数
            output_dir: 输出目录
            rewrite_question: 是否对每次运行改写问题（第一次运行始终使用原问题）
            
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
            config['total_runs'] = n_runs  # 添加总运行数
            
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
        - 前30%: greedy + low diversity (倾向选最优工具，响应偏正常)
        - 中40%: balanced + medium diversity (平衡探索，响应多样)
        - 后30%: exploratory + high diversity (强探索，响应更异常)
        """
        ratio = run_id / total_runs
        
        if ratio < 0.3:
            # 前30%
            return {
                "run_id": run_id,
                "exploration_mode": "greedy",
                "diversity_mode": "low",
                "temperature": 0.5
            }
        elif ratio < 0.7:
            # 中40%
            return {
                "run_id": run_id,
                "exploration_mode": "balanced",
                "diversity_mode": "medium",
                "temperature": 0.7
            }
        else:
            # 后30%
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
        """保存批量运行的汇总"""
        summary = {
            "question": question,
            "total_runs": len(results),
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "avg_steps": sum(r['statistics']['total_steps'] for r in results) / len(results),
                "avg_findings": sum(r['statistics']['total_findings'] for r in results) / len(results),
                "step_distribution": [r['statistics']['total_steps'] for r in results],
                "unique_paths": len(set(
                    tuple(r['summary']['diagnostic_path']) for r in results
                ))
            },
            "runs": [
                {
                    "run_id": r['run_id'],
                    "steps": r['statistics']['total_steps'],
                    "findings": r['statistics']['total_findings'],
                    "path": r['summary']['diagnostic_path']
                }
                for r in results
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
            print(f"平均发现: {summary['statistics']['avg_findings']:.1f}")
            print(f"唯一路径: {summary['statistics']['unique_paths']}")
            print(f"步骤分布: {summary['statistics']['step_distribution']}")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"❌ 保存汇总失败: {e}")


def test_agent_generator():
    """测试Agent生成器"""
    print("=" * 80)
    print("测试Agent生成器")
    print("=" * 80)
    
    # 初始化工具管理器
    tool_manager = ToolManager('/mnt/user-data/outputs/available_tools.txt')
    
    # 加载知识库（如果有）
    try:
        with open('/mnt/user-data/uploads/workflow.json', 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
    except:
        knowledge_base = {}
    
    # 创建生成器
    generator = AgentGenerator(
        tool_manager=tool_manager,
        api_key="kw-qIdb2KBfLLBkk6YEJ1clWKKOctnHgWMjtfRJwQ2yTLBCXjMv",
        api_base="http://10.12.208.86:8502",
        knowledge_base=knowledge_base,
        max_steps=10  # 测试时减少步骤
    )
    
    # 测试问题
    question = "serverleaf01_1_16.135设备上10GE1/0/24接口发生丢包该如何处理？"
    
    # 单次生成测试
    print("\n测试单次生成:")
    result = generator.generate(question)
    
    print(f"\n生成的数据:")
    print(f"- 步骤数: {len(result['execution_records'])}")
    print(f"- 工具序列: {[r['tool_name'] for r in result['execution_records']]}")
    
    # 保存结果
    generator.save_result(result, "/mnt/user-data/outputs/test_single_run.json")


if __name__ == '__main__':
    test_agent_generator()
