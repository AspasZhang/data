"""
批量运行主脚本
用于批量生成多样化的故障诊断数据
支持单个文档或多个文档批量处理
"""

import sys
import json
import argparse
from pathlib import Path

# 添加路径
sys.path.insert(0, '/mnt/user-data/outputs')

from tool_manager import ToolManager
from agent_generator import AgentGenerator


def load_documents_config(questions_input, knowledge_bases_input):
    """
    加载文档配置
    
    Args:
        questions_input: 问题字符串、问题列表文件路径或JSON字符串
        knowledge_bases_input: 知识库文件路径、列表文件路径或JSON字符串
        
    Returns:
        List[Dict]: [{"question": "...", "knowledge_base": "..."}]
    """
    documents = []
    
    # 处理questions
    if questions_input.endswith('.json'):
        # 从JSON文件加载问题列表
        with open(questions_input, 'r', encoding='utf-8') as f:
            questions = json.load(f)
    elif questions_input.startswith('['):
        # 直接解析JSON字符串
        questions = json.loads(questions_input)
    else:
        # 单个问题字符串
        questions = [questions_input]
    
    # 处理knowledge_bases
    if knowledge_bases_input.endswith('.json') and Path(knowledge_bases_input).exists():
        # 从JSON文件加载知识库列表
        try:
            with open(knowledge_bases_input, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith('['):
                    # 是一个列表文件
                    knowledge_bases = json.loads(content)
                else:
                    # 是单个知识库文件
                    knowledge_bases = [knowledge_bases_input]
        except:
            knowledge_bases = [knowledge_bases_input]
    elif knowledge_bases_input.startswith('['):
        # 直接解析JSON字符串
        knowledge_bases = json.loads(knowledge_bases_input)
    else:
        # 单个知识库文件路径
        knowledge_bases = [knowledge_bases_input]
    
    # 组合成文档列表
    if len(questions) == len(knowledge_bases):
        # 一对一映射
        for q, kb in zip(questions, knowledge_bases):
            documents.append({"question": q, "knowledge_base": kb})
    elif len(knowledge_bases) == 1:
        # 所有问题使用同一个知识库
        for q in questions:
            documents.append({"question": q, "knowledge_base": knowledge_bases[0]})
    elif len(questions) == 1:
        # 一个问题使用多个知识库（不太常见但支持）
        for kb in knowledge_bases:
            documents.append({"question": questions[0], "knowledge_base": kb})
    else:
        raise ValueError(f"问题数量({len(questions)})和知识库数量({len(knowledge_bases)})不匹配")
    
    return documents


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='批量生成故障诊断数据 - 支持单个或多个文档',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 单个文档:
   python3 batch_generate.py --question "问题描述" --knowledge_base workflow.json

2. 多个文档（从文件加载）:
   python3 batch_generate.py --questions questions.json --knowledge_bases kb_list.json

3. 多个文档（命令行指定）:
   python3 batch_generate.py \\
     --questions '["问题1", "问题2", "问题3"]' \\
     --knowledge_bases '["kb1.json", "kb2.json", "kb3.json"]'

4. 多个问题共享一个知识库:
   python3 batch_generate.py \\
     --questions '["问题1", "问题2"]' \\
     --knowledge_base workflow.json
        """
    )
    
    # 单个文档参数（向后兼容）
    parser.add_argument('--question', type=str, 
                       default=None,
                       help='单个问题描述')
    parser.add_argument('--knowledge_base', type=str,
                       default=None,
                       help='单个知识库文件路径')
    
    # 多个文档参数
    parser.add_argument('--questions', type=str,
                       default=None,
                       help='问题列表: JSON文件路径或JSON字符串 ["问题1", "问题2"]')
    parser.add_argument('--knowledge_bases', type=str,
                       default=None,
                       help='知识库列表: JSON文件路径或JSON字符串 ["kb1.json", "kb2.json"]')
    
    # 通用参数
    parser.add_argument('--n_runs', type=int, default=10,
                       help='每个文档生成的数据条数')
    parser.add_argument('--max_steps', type=int, default=20,
                       help='每次运行的最大步骤数')
    parser.add_argument('--output_dir', type=str, 
                       default='/mnt/user-data/outputs/batch_runs',
                       help='基础输出目录')
    parser.add_argument('--tools_file', type=str,
                       default='/mnt/user-data/outputs/available_tools.txt',
                       help='工具列表文件路径')
    parser.add_argument('--rewrite_question', action='store_true',
                       help='是否改写问题以增加多样性（默认False）')
    
    args = parser.parse_args()
    
    # 确定使用哪种模式
    if args.questions or (args.question and args.questions):
        # 多文档模式
        questions_input = args.questions or args.question
        knowledge_bases_input = args.knowledge_bases or args.knowledge_base or '/mnt/user-data/uploads/workflow.json'
    else:
        # 单文档模式（向后兼容）
        questions_input = args.question or "serverleaf01_1_16.135设备上10GE1/0/24接口发生丢包该如何处理？"
        knowledge_bases_input = args.knowledge_base or '/mnt/user-data/uploads/workflow.json'
    
    # 加载文档配置
    try:
        documents = load_documents_config(questions_input, knowledge_bases_input)
    except Exception as e:
        print(f"❌ 文档配置加载失败: {e}")
        return
    
    # 创建基础输出目录
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🚀 批量数据生成系统")
    print("=" * 80)
    print(f"文档数量: {len(documents)}")
    print(f"每文档生成: {args.n_runs} 条")
    print(f"最大步骤: {args.max_steps}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80 + "\n")
    
    # 初始化工具管理器（所有文档共享）
    print("📋 步骤1: 加载工具列表...")
    tool_manager = ToolManager(args.tools_file)
    print(f"   ✅ 已加载 {len(tool_manager.tools)} 个工具\n")
    
    # 处理每个文档
    all_results = []
    
    for doc_idx, doc in enumerate(documents, 1):
        print("\n" + "=" * 80)
        print(f"📄 处理文档 {doc_idx}/{len(documents)}")
        print("=" * 80)
        print(f"问题: {doc['question'][:80]}{'...' if len(doc['question']) > 80 else ''}")
        print(f"知识库: {doc['knowledge_base']}")
        print("=" * 80 + "\n")
        
        # 为当前文档创建输出目录
        if len(documents) > 1:
            doc_output_dir = base_output_dir / f"doc_{doc_idx:03d}"
        else:
            doc_output_dir = base_output_dir
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载知识库
        print(f"📚 加载知识库...")
        try:
            with open(doc['knowledge_base'], 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            print(f"   ✅ 已加载知识库: {doc['knowledge_base']}\n")
        except Exception as e:
            print(f"   ⚠️  知识库加载失败: {e}")
            print(f"   ℹ️  将使用默认配置\n")
            knowledge_base = {}
        
        # 创建生成器
        print("🤖 初始化Agent生成器...")
        generator = AgentGenerator(
            tool_manager=tool_manager,
            api_key="kw-qIdb2KBfLLBkk6YEJ1clWKKOctnHgWMjtfRJwQ2yTLBCXjMv",
            api_base="http://10.12.208.86:8502",
            knowledge_base=knowledge_base,
            max_steps=args.max_steps
        )
        print()
        
        # 批量生成
        print(f"🔄 开始生成 {args.n_runs} 条数据...")
        print("-" * 80 + "\n")
        
        try:
            results = generator.generate_batch(
                question=doc['question'],
                n_runs=args.n_runs,
                output_dir=str(doc_output_dir),
                rewrite_question=args.rewrite_question
            )
            
            all_results.append({
                "doc_id": doc_idx,
                "question": doc['question'],
                "knowledge_base": doc['knowledge_base'],
                "output_dir": str(doc_output_dir),
                "results": results
            })
            
            # 显示当前文档统计
            print("\n" + "=" * 80)
            print(f"✅ 文档 {doc_idx} 生成完成！")
            print("=" * 80)
            print(f"输出目录: {doc_output_dir}")
            print(f"生成文件:")
            print(f"  - 单次运行: run_*.json ({args.n_runs}个)")
            print(f"  - 批量汇总: batch_summary.json")
            
            # 路径多样性分析
            paths = [tuple(r['summary']['diagnostic_path']) for r in results]
            unique_paths = len(set(paths))
            
            print(f"\n📊 路径多样性:")
            print(f"  总运行数: {len(results)}")
            print(f"  唯一路径: {unique_paths}")
            print(f"  多样性比例: {unique_paths/len(results)*100:.1f}%")
            
            # 显示前3条路径
            if len(results) > 0:
                print(f"\n前3条路径示例:")
                for i, result in enumerate(results[:3], 1):
                    path = result['summary']['diagnostic_path']
                    steps = len(path)
                    print(f"  Run {i} ({steps}步): {' → '.join(path[:5])}" + 
                          (f" → ..." if steps > 5 else ""))
            
            print("=" * 80)
            
        except Exception as e:
            print(f"\n❌ 文档 {doc_idx} 生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 最终汇总
    print("\n\n" + "=" * 80)
    print("🎉 所有文档处理完成！")
    print("=" * 80)
    print(f"总文档数: {len(documents)}")
    print(f"成功处理: {len(all_results)}")
    print(f"每文档生成: {args.n_runs} 条")
    print(f"总数据条数: {len(all_results) * args.n_runs}")
    
    # 保存总体汇总
    if len(documents) > 1:
        summary_file = base_output_dir / "all_documents_summary.json"
        summary_data = {
            "total_documents": len(documents),
            "n_runs_per_doc": args.n_runs,
            "max_steps": args.max_steps,
            "documents": [
                {
                    "doc_id": r["doc_id"],
                    "question": r["question"],
                    "output_dir": r["output_dir"],
                    "total_runs": len(r["results"]),
                    "avg_steps": sum(res['statistics']['total_steps'] for res in r["results"]) / len(r["results"]),
                    "unique_paths": len(set(tuple(res['summary']['diagnostic_path']) for res in r["results"]))
                }
                for r in all_results
            ]
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 总体汇总已保存: {summary_file}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
