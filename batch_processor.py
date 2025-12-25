"""
批量处理多个问题的脚本
只需修改questions和knowledge_bases两个列表即可
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/mnt/user-data/outputs')

from tool_manager import ToolManager
from agent_generator import AgentGenerator


# ============================================================
# 📝 在这里修改你的问题和知识库列表
# ============================================================

# 问题列表
QUESTIONS = [
    "请分析设备aggrleaf02_2_20.45的接口是否有异常状态，并定位故障原因？",
    "serverleaf01_1_16.135设备上10GE1/0/24接口发生丢包该如何处理？",
    "设备spine01的BGP邻居关系异常，请帮我排查问题。",
    # 在这里添加更多问题...
]

# 知识库列表（必须和问题列表一一对应，或者只有一个表示所有问题共用）
KNOWLEDGE_BASES = [
    "/mnt/user-data/uploads/workflow.json",
    "/mnt/user-data/uploads/workflow.json",
    "/mnt/user-data/uploads/workflow.json",
    # 在这里添加更多知识库路径...
]

# ============================================================
# 运行配置（可选修改）
# ============================================================

# 每个问题生成多少条数据
N_RUNS = 10

# 最大步骤数
MAX_STEPS = 20

# 是否改写问题以增加多样性
REWRITE_QUESTION = True

# 工具文件路径
TOOLS_FILE = '/mnt/user-data/outputs/available_tools_with_params.txt'

# 输出基础目录
OUTPUT_BASE_DIR = '/mnt/user-data/outputs/batch_results'

# API配置
API_KEY = "kw-qIdb2KBfLLBkk6YEJ1clWKKOctnHgWMjtfRJwQ2yTLBCXjMv"
API_BASE = "http://10.12.208.86:8502"

# ============================================================
# 主处理逻辑（不需要修改）
# ============================================================


def validate_inputs():
    """验证输入配置"""
    if not QUESTIONS:
        raise ValueError("❌ QUESTIONS列表不能为空！")
    
    if not KNOWLEDGE_BASES:
        raise ValueError("❌ KNOWLEDGE_BASES列表不能为空！")
    
    # 检查数量匹配
    if len(KNOWLEDGE_BASES) == 1:
        print(f"ℹ️  所有 {len(QUESTIONS)} 个问题将使用同一个知识库")
        return [(q, KNOWLEDGE_BASES[0]) for q in QUESTIONS]
    elif len(QUESTIONS) == len(KNOWLEDGE_BASES):
        print(f"ℹ️  {len(QUESTIONS)} 个问题将分别使用对应的知识库")
        return list(zip(QUESTIONS, KNOWLEDGE_BASES))
    else:
        raise ValueError(
            f"❌ 问题数量({len(QUESTIONS)})和知识库数量({len(KNOWLEDGE_BASES)})不匹配！"
            f"\n   知识库列表必须是1个（共用）或与问题数量相同"
        )


def load_knowledge_base(kb_path: str) -> dict:
    """加载知识库"""
    try:
        with open(kb_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"   ⚠️  知识库加载失败: {e}")
        print(f"   ℹ️  将使用默认配置")
        return {}


def process_question(question: str, kb_path: str, question_idx: int, 
                     tool_manager: ToolManager, output_base: Path):
    """
    处理单个问题
    
    Args:
        question: 问题描述
        kb_path: 知识库路径
        question_idx: 问题索引（从1开始）
        tool_manager: 工具管理器
        output_base: 输出基础目录
    """
    print("\n" + "=" * 80)
    print(f"📄 处理问题 {question_idx}/{len(QUESTIONS)}")
    print("=" * 80)
    print(f"问题: {question[:80]}{'...' if len(question) > 80 else ''}")
    print(f"知识库: {kb_path}")
    print("=" * 80 + "\n")
    
    # 创建输出目录
    question_dir = output_base / f"question_{question_idx:03d}"
    question_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存问题信息
    question_info = {
        "question_id": question_idx,
        "question": question,
        "knowledge_base": kb_path,
        "timestamp": datetime.now().isoformat(),
        "n_runs": N_RUNS,
        "max_steps": MAX_STEPS
    }
    
    with open(question_dir / "question_info.json", 'w', encoding='utf-8') as f:
        json.dump(question_info, f, ensure_ascii=False, indent=2)
    
    # 加载知识库
    print("📚 加载知识库...")
    knowledge_base = load_knowledge_base(kb_path)
    print()
    
    # 创建生成器
    print("🤖 初始化Agent生成器...")
    generator = AgentGenerator(
        tool_manager=tool_manager,
        api_key=API_KEY,
        api_base=API_BASE,
        knowledge_base=knowledge_base,
        max_steps=MAX_STEPS
    )
    print()
    
    # 批量生成
    print(f"🔄 开始生成 {N_RUNS} 条数据...")
    print("-" * 80 + "\n")
    
    try:
        results = generator.generate_batch(
            question=question,
            n_runs=N_RUNS,
            output_dir=str(question_dir),
            rewrite_question=REWRITE_QUESTION
        )
        
        # 显示统计
        print("\n" + "=" * 80)
        print(f"✅ 问题 {question_idx} 生成完成！")
        print("=" * 80)
        print(f"输出目录: {question_dir}")
        print(f"生成文件:")
        print(f"  - 单次运行: run_*.json ({N_RUNS}个)")
        print(f"  - 批量汇总: batch_summary.json")
        print(f"  - 问题信息: question_info.json")
        
        # 路径多样性分析
        def extract_path(result):
            """从新格式中提取路径"""
            steps = result.get('response', [])
            path = []
            for step_dict in steps:
                for step_key, step_data in step_dict.items():
                    coa = step_data.get('coa', [])
                    for action_obs in coa:
                        tool_name = action_obs.get('action', {}).get('name')
                        if tool_name:
                            path.append(tool_name)
            return tuple(path)
        
        paths = [extract_path(r) for r in results]
        unique_paths = len(set(paths))
        
        print(f"\n📊 路径多样性:")
        print(f"  总运行数: {len(results)}")
        print(f"  唯一路径: {unique_paths}")
        print(f"  多样性比例: {unique_paths/len(results)*100:.1f}%")
        
        # 显示前3条路径
        if len(results) > 0:
            print(f"\n前3条路径示例:")
            for i, result in enumerate(results[:3], 1):
                path = list(extract_path(result))
                steps = len(path)
                print(f"  Run {i} ({steps}步): {' → '.join(path[:5])}" + 
                      (f" → ..." if steps > 5 else ""))
        
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 问题 {question_idx} 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 80)
    print("🚀 批量问题处理系统")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    # 验证输入
    print("📋 验证配置...")
    try:
        question_kb_pairs = validate_inputs()
    except ValueError as e:
        print(str(e))
        return
    
    print(f"✅ 配置验证通过")
    print(f"   问题总数: {len(QUESTIONS)}")
    print(f"   每问题运行: {N_RUNS} 次")
    print(f"   总数据条数: {len(QUESTIONS) * N_RUNS}")
    print(f"   输出目录: {OUTPUT_BASE_DIR}")
    print()
    
    # 创建输出目录
    output_base = Path(OUTPUT_BASE_DIR)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 初始化工具管理器（所有问题共享）
    print("📋 步骤1: 加载工具列表...")
    tool_manager = ToolManager(TOOLS_FILE)
    print(f"   ✅ 已加载 {len(tool_manager.tools)} 个工具\n")
    
    # 处理每个问题
    success_count = 0
    failed_questions = []
    
    for idx, (question, kb_path) in enumerate(question_kb_pairs, 1):
        success = process_question(
            question=question,
            kb_path=kb_path,
            question_idx=idx,
            tool_manager=tool_manager,
            output_base=output_base
        )
        
        if success:
            success_count += 1
        else:
            failed_questions.append(idx)
    
    # 生成总体汇总
    print("\n\n" + "=" * 80)
    print("🎉 批量处理完成！")
    print("=" * 80)
    print(f"总问题数: {len(QUESTIONS)}")
    print(f"成功处理: {success_count}")
    print(f"失败处理: {len(failed_questions)}")
    if failed_questions:
        print(f"失败问题索引: {failed_questions}")
    print(f"每问题生成: {N_RUNS} 条")
    print(f"总数据条数: {success_count * N_RUNS}")
    print(f"输出目录: {output_base}")
    
    # 保存总体汇总
    summary_file = output_base / "all_questions_summary.json"
    summary_data = {
        "total_questions": len(QUESTIONS),
        "successful": success_count,
        "failed": len(failed_questions),
        "failed_indices": failed_questions,
        "n_runs_per_question": N_RUNS,
        "max_steps": MAX_STEPS,
        "timestamp": datetime.now().isoformat(),
        "questions": [
            {
                "question_id": i,
                "question": q,
                "knowledge_base": kb,
                "output_dir": str(output_base / f"question_{i:03d}")
            }
            for i, (q, kb) in enumerate(question_kb_pairs, 1)
        ]
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 总体汇总已保存: {summary_file}")
    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
