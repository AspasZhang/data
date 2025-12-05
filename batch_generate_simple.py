"""
batch_generate.py 的简化配置版本
直接在文件顶部配置，然后运行即可
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, '/mnt/user-data/outputs')

from tool_manager import ToolManager
from agent_generator import AgentGenerator

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⚙️ 配置区域 - 修改这里的参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 问题列表
QUESTIONS = [
    "serverleaf01_1_16.135设备上10GE1/0/24接口发生丢包该如何处理？",
    "网络设备eth0接口流量异常，速度很慢，怎么排查？",
    "交换机端口频繁up/down，如何诊断？",
]

# 知识库列表（可以是单个或多个）
# 如果只有一个，所有问题共享这个知识库
KNOWLEDGE_BASES = [
    "/mnt/user-data/uploads/workflow.json",
    # 如果要每个问题独立知识库，添加更多：
    # "/mnt/user-data/uploads/workflow2.json",
    # "/mnt/user-data/uploads/workflow3.json",
]

# 生成参数
N_RUNS = 5  # 每个问题生成几条数据
MAX_STEPS = 15  # 最大步骤数
OUTPUT_DIR = "/mnt/user-data/outputs/configured_runs"

# 工具文件
TOOLS_FILE = "/mnt/user-data/outputs/available_tools.txt"

# API配置
API_KEY = "kw-qIdb2KBfLLBkk6YEJ1clWKKOctnHgWMjtfRJwQ2yTLBCXjMv"
API_BASE = "http://10.12.208.86:8502"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主程序 - 无需修改
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # 准备文档列表
    if len(KNOWLEDGE_BASES) == 1:
        # 所有问题共享一个知识库
        documents = [{"question": q, "knowledge_base": KNOWLEDGE_BASES[0]} 
                    for q in QUESTIONS]
    elif len(QUESTIONS) == len(KNOWLEDGE_BASES):
        # 一对一映射
        documents = [{"question": q, "knowledge_base": kb} 
                    for q, kb in zip(QUESTIONS, KNOWLEDGE_BASES)]
    else:
        print(f"❌ 错误: 问题数量({len(QUESTIONS)})和知识库数量({len(KNOWLEDGE_BASES)})不匹配")
        print("   知识库要么是1个（共享），要么与问题数量相同")
        return
    
    # 创建输出目录
    base_output_dir = Path(OUTPUT_DIR)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🚀 批量数据生成系统（配置模式）")
    print("=" * 80)
    print(f"文档数量: {len(documents)}")
    print(f"每文档生成: {N_RUNS} 条")
    print(f"最大步骤: {MAX_STEPS}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 80 + "\n")
    
    # 加载工具
    print("📋 加载工具列表...")
    tool_manager = ToolManager(TOOLS_FILE)
    print(f"   ✅ 已加载 {len(tool_manager.tools)} 个工具\n")
    
    all_results = []
    
    # 处理每个文档
    for doc_idx, doc in enumerate(documents, 1):
        print("\n" + "=" * 80)
        print(f"📄 处理文档 {doc_idx}/{len(documents)}")
        print("=" * 80)
        print(f"问题: {doc['question'][:80]}...")
        print("=" * 80 + "\n")
        
        # 创建文档输出目录
        if len(documents) > 1:
            doc_output_dir = base_output_dir / f"doc_{doc_idx:03d}"
        else:
            doc_output_dir = base_output_dir
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载知识库
        print("📚 加载知识库...")
        try:
            with open(doc['knowledge_base'], 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            print("   ✅ 已加载知识库\n")
        except Exception as e:
            print(f"   ⚠️  知识库加载失败: {e}")
            print("   ℹ️  将使用默认配置\n")
            knowledge_base = {}
        
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
                question=doc['question'],
                n_runs=N_RUNS,
                output_dir=str(doc_output_dir)
            )
            
            all_results.append({
                "doc_id": doc_idx,
                "question": doc['question'],
                "output_dir": str(doc_output_dir),
                "results": results
            })
            
            print("\n" + "=" * 80)
            print(f"✅ 文档 {doc_idx} 完成！")
            print("=" * 80)
            
            paths = [tuple(r['summary']['diagnostic_path']) for r in results]
            unique_paths = len(set(paths))
            print(f"路径多样性: {unique_paths}/{len(results)} ({unique_paths/len(results)*100:.1f}%)")
            
        except Exception as e:
            print(f"\n❌ 文档 {doc_idx} 生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print("\n" + "=" * 80)
    print("🎉 所有文档处理完成！")
    print("=" * 80)
    print(f"总文档数: {len(documents)}")
    print(f"成功处理: {len(all_results)}")
    print(f"总数据条数: {len(all_results) * N_RUNS}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
