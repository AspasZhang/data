"""
简单批量数据生成脚本
支持多线程并发，直接运行即可批量生成数据
"""

import json
import glob as glob_mod
import os
import random
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any
from agent_generator import AgentGenerator
from tool_manager import ToolManager


# ============ 线程安全的 Tee ============

class Tee:
    """同时输出到控制台和文件（线程安全）"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
        self._lock = threading.Lock()

    def write(self, message):
        with self._lock:
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

    def flush(self):
        with self._lock:
            self.terminal.flush()
            self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def load_cot_samples(cot_dir: str = "CoT") -> List[str]:
    """从CoT目录加载所有example和question作为参考样例"""
    samples = []
    for filepath in glob_mod.glob(os.path.join(cot_dir, "*.json")):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            samples.extend(data.get('example', []))
            samples.extend(data.get('question', []))
        except Exception:
            pass
    return samples


def randomize_question(question: str, api_key: str, api_base: str,
                       cot_samples: List[str] = None) -> str:
    """用大模型改写问题，参考CoT中的真实问法，替换设备名、接口名，保持故障语义不变"""
    import openai

    # 从CoT样例中随机抽取几条作为参考
    ref_text = ""
    if cot_samples:
        picked = random.sample(cot_samples, min(8, len(cot_samples)))
        ref_text = "\n".join(f"- {s}" for s in picked)

    prompt = f"""请改写以下网络故障诊断问题。要求：
1. 替换成不同的设备名和接口名，保持故障类型和语义不变
2. 可以适当调整说法和语气，但不要改变故障本质
3. 设备名和接口名的风格请参考下面的真实样例
4. 只输出改写后的一句话问题，不要有任何解释

【真实问法参考】
{ref_text}

【原始问题】
{question}"""

    try:
        base_url = api_base if api_base.rstrip('/').endswith('/v1') else api_base.rstrip('/') + '/v1'
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model="Qwen2.5-72B",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=200
        )
        result = response.choices[0].message.content.strip()
        # 去掉可能的引号包裹
        if (result.startswith('"') and result.endswith('"')) or \
           (result.startswith('「') and result.endswith('」')):
            result = result[1:-1]
        return result
    except Exception as e:
        print(f"   ⚠️ 问题改写失败，使用原始问题: {e}")
        return question


# ============ 单个任务的 worker 函数 ============

def run_single_task(task: Dict, tool_manager: ToolManager,
                    api_key: str, api_base: str, max_steps: int,
                    cot_samples: List[str]) -> Dict:
    """
    单个生成任务的 worker（每个线程独立运行）

    Args:
        task: 任务描述字典
        tool_manager: 工具管理器（只读，线程间共享）
        其余为配置参数

    Returns:
        {"success": bool, "result": ..., "task": task, "error": str|None, "elapsed": float}
    """
    q_idx = task['q_idx']
    run_idx = task['run_idx']
    question = task['question']
    kb_file = task['kb_file']
    run_config = task['run_config']
    q_output_dir = task['q_output_dir']
    tag = f"[Q{q_idx}-R{run_idx + 1}]"

    start_time = time.time()

    try:
        # 每个 worker 创建独立的 generator（避免线程间状态冲突）
        knowledge_base = None
        if kb_file and os.path.exists(kb_file):
            with open(kb_file, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)

        generator = AgentGenerator(
            tool_manager=tool_manager,
            api_key=api_key,
            api_base=api_base,
            knowledge_base=knowledge_base,
            max_steps=max_steps
        )

        # 用大模型改写问题
        actual_question = randomize_question(question, api_key, api_base, cot_samples)
        print(f"\n{tag} 📝 实际问题: {actual_question}")

        # 生成数据
        result = generator.generate(
            question=actual_question,
            run_config=run_config,
            rewrite_question=(run_idx > 0)
        )

        elapsed = time.time() - start_time

        # 添加元数据
        result['metadata'] = {
            'question_index': q_idx,
            'run_index': run_idx,
            'config': run_config,
            'elapsed_time': elapsed,
            'timestamp': datetime.now().isoformat()
        }

        # 保存结果
        filepath = os.path.join(q_output_dir, f"run{run_idx + 1:02d}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n{tag} ✅ 完成 (耗时: {elapsed:.1f}秒, 步骤数: {len(result['response'])})")
        return {"success": True, "result": result, "task": task, "error": None, "elapsed": elapsed}

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n{tag} ❌ 失败: {e}")

        # 保存错误信息
        error_info = {
            'question': question,
            'run_index': run_idx,
            'config': run_config,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        error_filepath = os.path.join(q_output_dir, f"error_run{run_idx + 1:02d}.json")
        with open(error_filepath, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, ensure_ascii=False, indent=2)

        return {"success": False, "result": None, "task": task, "error": str(e), "elapsed": elapsed}


# ============ 主批量生成函数 ============

def batch_generate(
    questions: List[Dict[str, str]],
    runs_per_question: int = 3,
    output_dir: str = "batch_output",
    api_key: str = "",
    api_base: str = "http://localhost:20015/v1",
    tools_file: str = "available_tools.txt",
    max_steps: int = 20,
    cot_dir: str = "CoT",
    max_workers: int = 20
):
    """
    批量生成数据（多线程）

    Args:
        questions: 问题列表，每项为 {"question": "...", "kb_file": "..."}
        runs_per_question: 每个问题生成多少次
        output_dir: 输出目录
        api_key: API密钥
        api_base: API基础URL
        tools_file: 工具文件路径
        max_steps: 最大步骤数
        cot_dir: CoT参考样例目录
        max_workers: 最大并发线程数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载CoT参考样例
    cot_samples = load_cot_samples(cot_dir)
    if cot_samples:
        print(f"✅ 加载了 {len(cot_samples)} 条CoT参考样例（用于问题改写）")
    else:
        print(f"⚠️  未找到CoT参考样例（目录: {cot_dir}），改写将不使用参考")
    print()

    # 初始化工具管理器（只读，所有线程共享）
    print("📋 加载工具...")
    tool_manager = ToolManager(tools_file)
    print(f"✅ 加载了 {len(tool_manager.get_all_tool_names())} 个工具")
    print()

    total_tasks = len(questions) * runs_per_question
    print("=" * 80)
    print("批量数据生成系统（多线程）")
    print("=" * 80)
    print(f"问题数量: {len(questions)}")
    print(f"每个问题运行: {runs_per_question} 次")
    print(f"总任务数: {total_tasks}")
    print(f"并发线程数: {min(max_workers, total_tasks)}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)
    print()

    # ============ 构建所有任务 ============
    exploration_modes = ['balanced', 'exploratory', 'greedy']
    diversity_modes = ['medium', 'high', 'low']
    temperatures = [0.7, 0.8, 0.6]

    tasks = []
    for q_idx, q_item in enumerate(questions, 1):
        question = q_item["question"]
        kb_file = q_item.get("kb_file", "")

        q_output_dir = os.path.join(output_dir, f"Q{q_idx}")
        os.makedirs(q_output_dir, exist_ok=True)

        print(f"📌 Q{q_idx}: {question}")
        if kb_file:
            print(f"   知识库: {kb_file}")

        for run_idx in range(runs_per_question):
            run_config = {
                'run_id': run_idx,
                'total_runs': runs_per_question,
                'exploration_mode': exploration_modes[run_idx % len(exploration_modes)],
                'diversity_mode': diversity_modes[run_idx % len(diversity_modes)],
                'temperature': temperatures[run_idx % len(temperatures)]
            }
            tasks.append({
                'q_idx': q_idx,
                'run_idx': run_idx,
                'question': question,
                'kb_file': kb_file,
                'run_config': run_config,
                'q_output_dir': q_output_dir
            })

    print(f"\n🚀 开始并发执行 {len(tasks)} 个任务...\n")
    batch_start = time.time()

    # ============ 多线程执行 ============
    all_results = []
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single_task, task, tool_manager,
                api_key, api_base, max_steps, cot_samples
            ): task
            for task in tasks
        }

        for future in as_completed(futures):
            outcome = future.result()
            if outcome["success"]:
                successful += 1
                all_results.append(outcome["result"])
            else:
                failed += 1

            done = successful + failed
            print(f"   📊 进度: {done}/{len(tasks)} "
                  f"(成功: {successful}, 失败: {failed})")

    batch_elapsed = time.time() - batch_start

    # ============ 按问题合并 all_runs.json ============
    for q_idx, q_item in enumerate(questions, 1):
        q_output_dir = os.path.join(output_dir, f"Q{q_idx}")
        q_results = []
        for run_idx in range(runs_per_question):
            run_file = os.path.join(q_output_dir, f"run{run_idx + 1:02d}.json")
            if os.path.exists(run_file):
                with open(run_file, 'r', encoding='utf-8') as f:
                    q_results.append(json.load(f))
        if q_results:
            with open(os.path.join(q_output_dir, "all_runs.json"), 'w', encoding='utf-8') as f:
                json.dump(q_results, f, ensure_ascii=False, indent=2)

    # ============ 保存统计信息 ============
    stats = {
        'total_questions': len(questions),
        'runs_per_question': runs_per_question,
        'total_runs': total_tasks,
        'successful': successful,
        'failed': failed,
        'success_rate': successful / total_tasks if total_tasks > 0 else 0,
        'max_workers': max_workers,
        'total_elapsed_seconds': batch_elapsed,
        'start_time': datetime.now().isoformat(),
        'results': [
            {
                'question': q_item["question"],
                'kb_file': q_item.get("kb_file", "")
            }
            for q_item in questions
        ]
    }

    stats_filename = f"batch_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join(output_dir, stats_filename), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 保存所有结果合并文件
    if all_results:
        all_results_filename = f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join(output_dir, all_results_filename), 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 打印总结
    print(f"\n{'='*80}")
    print(f"批量生成完成")
    print(f"{'='*80}")
    print(f"总任务数: {total_tasks}")
    print(f"成功: {successful}")
    print(f"失败: {failed}")
    print(f"成功率: {stats['success_rate']:.1%}")
    print(f"总耗时: {batch_elapsed:.1f}秒")
    print(f"并发线程: {max_workers}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")

    return stats


def main():
    """主函数 - 直接运行即可"""

    # ============ 配置区（根据需要修改） ============

    # 问题列表（每项包含 question 和对应的 kb_file）
    QUESTIONS = [
        {"question": "交换机serverleaf01_1_16.135的10GE1/0/24接口出现丢包，请帮忙排查", "kb_file": "kb/kb_01.json"},
        {"question": "设备aggrleaf02_2_20.45的40GE2/2/5接口CRC错包超阈值", "kb_file": "kb/kb_02.json"},
        {"question": "核心交换机的光模块收发光功率异常，需要诊断", "kb_file": "kb/kb_03.json"},
        {"question": "接口流量突然下降，怀疑链路质量问题", "kb_file": "kb/kb_04.json"},
        {"question": "设备10.1.20.9的GE0/0/7端口持续出现错包", "kb_file": "kb/kb_05.json"},
        {"question": "聚合接口Eth-Trunk5状态Down，多条成员链路失效", "kb_file": "kb/kb_06.json"},
        {"question": "设备spine01_1_10.1的100GE1/0/1接口带宽利用率超阈值", "kb_file": "kb/kb_07.json"},
        {"question": "交换机coreleaf02_2_30.20的25GE1/0/6接口频繁Down", "kb_file": "kb/kb_08.json"},
    ]

    # 生成配置
    RUNS_PER_QUESTION = 3   # 每个问题生成3次
    OUTPUT_DIR = "/home/shijie/share_data/data_gen"  # 输出目录
    MAX_STEPS = 20           # 最大步骤数
    MAX_WORKERS = 20         # 最大并发线程数

    # API配置
    API_KEY = ""
    API_BASE = "http://localhost:20015/v1"

    # 文件路径
    TOOLS_FILE = "/home/shijie/share_data/workflow/case_data/tool_data6_normalized.txt"
    COT_DIR = "CoT"

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
        max_steps=MAX_STEPS,
        cot_dir=COT_DIR,
        max_workers=MAX_WORKERS
    )

    print("\n🎉 批量生成任务完成!")
    print(f"查看结果: {OUTPUT_DIR}/")

    tee.close()
    print(f"日志已保存: {log_path}")


if __name__ == '__main__':
    main()
