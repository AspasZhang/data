"""
简单批量数据生成脚本
直接运行即可批量生成数据，无需额外配置
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any
from agent_generator import AgentGenerator
from tool_manager import ToolManager


class Tee:
    """同时输出到控制台和文件"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def batch_generate(
    questions: List[str],
    runs_per_question: int = 3,
    output_dir: str = "batch_output",
    api_key: str = "",
    api_base: str = "http://localhost:20015/v1",
    tools_file: str = "available_tools.txt",
    knowledge_base_file: str = "test_kb.json",
    max_steps: int = 20
):
    """
    批量生成数据

    Args:
        questions: 问题列表
        runs_per_question: 每个问题生成多少次
        output_dir: 输出目录
        api_key: API密钥
        api_base: API基础URL
        tools_file: 工具文件路径
        knowledge_base_file: 知识库文件路径
        max_steps: 最大步骤数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("批量数据生成系统")
    print("=" * 80)
    print(f"问题数量: {len(questions)}")
    print(f"每个问题运行: {runs_per_question} 次")
    print(f"总运行次数: {len(questions) * runs_per_question}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)
    print()

    # 初始化工具管理器
    print("📋 加载工具...")
    tool_manager = ToolManager(tools_file)
    print(f"✅ 加载了 {len(tool_manager.get_all_tool_names())} 个工具")
    print()

    # 加载知识库
    knowledge_base = None
    if os.path.exists(knowledge_base_file):
        try:
            with open(knowledge_base_file, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            print(f"✅ 加载知识库: {knowledge_base_file}")
        except Exception as e:
            print(f"⚠️  加载知识库失败: {e}")
    print()

    # 初始化生成器
    print("🤖 初始化Agent生成器...")
    generator = AgentGenerator(
        tool_manager=tool_manager,
        api_key=api_key,
        api_base=api_base,
        knowledge_base=knowledge_base,
        max_steps=max_steps
    )
    print()

    # 统计信息
    stats = {
        'total_questions': len(questions),
        'runs_per_question': runs_per_question,
        'total_runs': len(questions) * runs_per_question,
        'successful': 0,
        'failed': 0,
        'start_time': datetime.now().isoformat(),
        'results': []
    }

    all_results = []

    # 配置循环（增加多样性）
    exploration_modes = ['balanced', 'exploratory', 'greedy']
    diversity_modes = ['medium', 'high', 'low']
    temperatures = [0.7, 0.8, 0.6]

    # 遍历每个问题
    for q_idx, question in enumerate(questions, 1):
        print(f"\n{'#'*80}")
        print(f"问题 {q_idx}/{len(questions)}")
        print(f"{'#'*80}")
        print(f"{question}")
        print(f"{'#'*80}\n")

        question_results = []

        # 每个问题生成多次
        for run_idx in range(runs_per_question):
            # 循环使用不同配置
            exploration_mode = exploration_modes[run_idx % len(exploration_modes)]
            diversity_mode = diversity_modes[run_idx % len(diversity_modes)]
            temperature = temperatures[run_idx % len(temperatures)]

            run_config = {
                'run_id': run_idx,
                'total_runs': runs_per_question,
                'exploration_mode': exploration_mode,
                'diversity_mode': diversity_mode,
                'temperature': temperature
            }

            print(f"{'─'*80}")
            print(f"运行 {run_idx + 1}/{runs_per_question}")
            print(f"配置: exploration={exploration_mode}, diversity={diversity_mode}, temp={temperature}")
            print(f"{'─'*80}\n")

            try:
                start_time = time.time()

                # 生成数据
                result = generator.generate(
                    question=question,
                    run_config=run_config,
                    rewrite_question=(run_idx > 0)  # 第一次不改写，后续改写增加多样性
                )

                elapsed_time = time.time() - start_time

                # 添加元数据
                result['metadata'] = {
                    'question_index': q_idx,
                    'run_index': run_idx,
                    'config': run_config,
                    'elapsed_time': elapsed_time,
                    'timestamp': datetime.now().isoformat()
                }

                question_results.append(result)
                all_results.append(result)
                stats['successful'] += 1

                print(f"\n✅ 运行 {run_idx + 1} 完成 (耗时: {elapsed_time:.1f}秒)")
                print(f"   生成步骤数: {len(result['response'])}")

                # 保存单个结果
                filename = f"q{q_idx:02d}_run{run_idx + 1:02d}.json"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"   💾 已保存: {filename}")

            except Exception as e:
                print(f"\n❌ 运行 {run_idx + 1} 失败: {e}")
                stats['failed'] += 1

                # 保存错误信息
                error_info = {
                    'question': question,
                    'run_index': run_idx,
                    'config': run_config,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }

                error_filename = f"error_q{q_idx:02d}_run{run_idx + 1:02d}.json"
                error_filepath = os.path.join(output_dir, error_filename)
                with open(error_filepath, 'w', encoding='utf-8') as f:
                    json.dump(error_info, f, ensure_ascii=False, indent=2)

            # 短暂休息，避免API限流
            if run_idx < runs_per_question - 1:
                time.sleep(1)

        # 保存该问题的所有结果
        if question_results:
            combined_filename = f"q{q_idx:02d}_all_runs.json"
            combined_filepath = os.path.join(output_dir, combined_filename)
            with open(combined_filepath, 'w', encoding='utf-8') as f:
                json.dump(question_results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 问题 {q_idx} 的所有结果已保存: {combined_filename}")

        stats['results'].append({
            'question': question,
            'successful_runs': len(question_results),
            'failed_runs': runs_per_question - len(question_results)
        })

    # 保存统计信息
    stats['end_time'] = datetime.now().isoformat()
    stats['success_rate'] = stats['successful'] / stats['total_runs'] if stats['total_runs'] > 0 else 0

    stats_filename = f"batch_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    stats_filepath = os.path.join(output_dir, stats_filename)
    with open(stats_filepath, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 保存所有结果
    if all_results:
        all_results_filename = f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        all_results_filepath = os.path.join(output_dir, all_results_filename)
        with open(all_results_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 打印总结
    print(f"\n{'='*80}")
    print(f"批量生成完成")
    print(f"{'='*80}")
    print(f"总运行次数: {stats['total_runs']}")
    print(f"成功: {stats['successful']}")
    print(f"失败: {stats['failed']}")
    print(f"成功率: {stats['success_rate']:.1%}")
    print(f"输出目录: {output_dir}")
    print(f"统计文件: {stats_filename}")
    print(f"合并结果: {all_results_filename}")
    print(f"{'='*80}\n")

    return stats


def main():
    """主函数 - 直接运行即可"""

    # ============ 配置区（根据需要修改） ============

    # 问题列表
    QUESTIONS = [
        "交换机serverleaf01_1_16.135的10GE1/0/24接口出现丢包，请帮忙排查",
        "设备aggrleaf02_2_20.45的40GE2/2/5接口CRC错包超阈值",
        "核心交换机的光模块收发光功率异常，需要诊断",
        "接口流量突然下降，怀疑链路质量问题",
        "设备10.1.20.9的GE0/0/7端口持续出现错包"
    ]

    # 生成配置
    RUNS_PER_QUESTION = 3  # 每个问题生成3次
    OUTPUT_DIR = "/home/shijie/share_data/data_gen"  # 输出目录
    MAX_STEPS = 20  # 最大步骤数

    # API配置
    API_KEY = ""
    API_BASE = "http://localhost:20015/v1"

    # 文件路径
    TOOLS_FILE = "/home/shijie/share_data/workflow/case_data/tool_data6_normalized.txt"
    KNOWLEDGE_BASE_FILE = "test_kb.json"

    # ============ 执行批量生成 ============

    log_filename = f"batch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(OUTPUT_DIR, log_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tee = Tee(log_path)
    sys.stdout = tee

    stats = batch_generate(
        questions=QUESTIONS,
        runs_per_question=RUNS_PER_QUESTION,
        output_dir=OUTPUT_DIR,
        api_key=API_KEY,
        api_base=API_BASE,
        tools_file=TOOLS_FILE,
        knowledge_base_file=KNOWLEDGE_BASE_FILE,
        max_steps=MAX_STEPS
    )

    print("\n🎉 批量生成任务完成!")
    print(f"查看结果: {OUTPUT_DIR}/")

    tee.close()
    print(f"日志已保存: {log_path}")


if __name__ == '__main__':
    main()
