"""
工具管理器（Tool Manager）模块
管理可用的故障诊断工具集
"""

import json
from typing import Dict, List, Any, Optional


class ToolManager:
    """工具管理器"""
    
    def __init__(self, tools_file_path: str):
        """
        初始化工具管理器
        
        Args:
            tools_file_path: 工具列表文件路径
        """
        self.tools_file_path = tools_file_path
        self.tools: Dict[str, str] = {}  # {工具名: 工具描述}
        self.tool_list: List[Dict[str, Any]] = []  # [{name: ..., description: ..., parameters: ...}]
        self.tool_parameters: Dict[str, str] = {}  # {工具名: parameters字符串}
        self._load_tools()
    
    def _load_tools(self):
        """从文件加载工具列表"""
        try:
            with open(self.tools_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            current_tool = None
            current_description = []
            current_parameters = None
            
            for line in lines:
                line = line.strip()
                
                if not line:  # 空行，跳过
                    continue
                
                # 检查是否是工具名称行（格式：数字.工具名）
                if line[0].isdigit() and '.' in line:
                    # 保存上一个工具
                    if current_tool:
                        description = ' '.join(current_description).strip()
                        self.tools[current_tool] = description
                        
                        tool_info = {
                            'name': current_tool,
                            'description': description
                        }
                        
                        if current_parameters:
                            tool_info['parameters'] = current_parameters
                            self.tool_parameters[current_tool] = current_parameters
                        
                        self.tool_list.append(tool_info)
                    
                    # 解析新工具名
                    parts = line.split('.', 1)
                    if len(parts) == 2:
                        current_tool = parts[1].strip()
                        current_description = []
                        current_parameters = None
                
                # 检查是否是Parameters行（直接存储为字符串）
                elif line.startswith('Parameters:'):
                    # 直接存储整行作为字符串
                    current_parameters = line.split('Parameters:', 1)[1].strip()
                
                else:
                    # 描述行
                    if current_tool:
                        current_description.append(line)
            
            # 保存最后一个工具
            if current_tool:
                description = ' '.join(current_description).strip()
                self.tools[current_tool] = description
                
                tool_info = {
                    'name': current_tool,
                    'description': description
                }
                
                if current_parameters:
                    tool_info['parameters'] = current_parameters
                    self.tool_parameters[current_tool] = current_parameters
                
                self.tool_list.append(tool_info)
            
            tools_with_params = len(self.tool_parameters)
            print(f"✅ 成功加载 {len(self.tools)} 个工具 (其中 {tools_with_params} 个包含参数定义)")
            
        except Exception as e:
            print(f"❌ 加载工具列表失败: {e}")
            self.tools = {}
            self.tool_list = []
            self.tool_parameters = {}
    
    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """
        获取工具描述
        
        Args:
            tool_name: 工具名称
            
        Returns:
            工具描述，如果不存在返回None
        """
        return self.tools.get(tool_name)
    
    def get_tool_parameters(self, tool_name: str) -> Optional[str]:
        """
        获取工具的参数定义（字符串格式）
        
        Args:
            tool_name: 工具名称
            
        Returns:
            参数字符串，如果不存在返回None
        """
        return self.tool_parameters.get(tool_name)
    
    def format_tool_parameters(self, tool_name: str) -> str:
        """
        格式化工具参数为易读的字符串
        
        Args:
            tool_name: 工具名称
            
        Returns:
            格式化的参数说明
        """
        params_str = self.get_tool_parameters(tool_name)
        if not params_str:
            return "  (无参数定义)"
        
        # 直接返回参数字符串，让LLM自己理解
        return f"  Parameters: {params_str}"
    
    def is_valid_tool(self, tool_name: str) -> bool:
        """
        检查工具是否有效
        
        Args:
            tool_name: 工具名称
            
        Returns:
            如果工具存在返回True，否则返回False
        """
        # 特殊工具：finish_diagnosis 总是有效
        if tool_name in ['finish_diagnosis', 'finish']:
            return True
        
        return tool_name in self.tools
    
    def get_all_tools(self) -> List[Dict[str, str]]:
        """
        获取所有工具列表
        
        Returns:
            工具列表 [{name: ..., description: ...}, ...]
        """
        return self.tool_list.copy()
    
    def get_tools_summary(self) -> str:
        """
        获取工具列表的摘要文本（用于LLM prompt）
        
        Returns:
            格式化的工具列表文本
        """
        if not self.tool_list:
            return "无可用工具"
        
        lines = ["可用工具列表："]
        for i, tool in enumerate(self.tool_list, 1):
            lines.append(f"{i}. {tool['name']}")
            lines.append(f"   描述：{tool['description']}")
        
        return "\n".join(lines)
    
    def get_tools_compact_list(self) -> str:
        """
        获取工具列表的紧凑格式（用于LLM prompt，节省token）
        
        Returns:
            紧凑格式的工具列表
        """
        if not self.tool_list:
            return "无可用工具"
        
        lines = []
        for tool in self.tool_list:
            lines.append(f"- {tool['name']}: {tool['description']}")
        
        return "\n".join(lines)
    
    def get_tools_with_parameters(self) -> str:
        """
        获取工具列表的详细格式（包含参数信息）
        
        Returns:
            包含参数的详细工具列表
        """
        if not self.tool_list:
            return "无可用工具"
        
        lines = []
        for i, tool in enumerate(self.tool_list, 1):
            lines.append(f"{i}. {tool['name']}")
            lines.append(f"   描述: {tool['description']}")
            
            # 添加参数信息（直接展示原始字符串）
            if 'parameters' in tool:
                lines.append(f"   Parameters: {tool['parameters']}")
            
            lines.append("")  # 空行分隔
        
        return "\n".join(lines)
    
    def search_tools(self, keyword: str) -> List[Dict[str, str]]:
        """
        根据关键词搜索工具
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            匹配的工具列表
        """
        keyword_lower = keyword.lower()
        results = []
        
        for tool in self.tool_list:
            name_lower = tool['name'].lower()
            desc_lower = tool['description'].lower()
            
            if keyword_lower in name_lower or keyword_lower in desc_lower:
                results.append(tool)
        
        return results
    
    def suggest_tool(self, task_description: str, top_n: int = 5) -> List[str]:
        """
        根据任务描述推荐工具（改进版 - 总是返回推荐）
        
        Args:
            task_description: 任务描述
            top_n: 返回最多多少个推荐
            
        Returns:
            推荐的工具名称列表
        """
        if not task_description:
            # 如果没有描述，返回最常用的通用工具
            return [
                'query_interface_info',
                'query_interface_traffic', 
                'query_device_logs'
            ][:top_n]
        
        task_lower = task_description.lower()
        scores = {}  # {工具名: 得分}
        
        # 1. 精确关键词匹配（高分）
        keywords_map = {
            'ping': ['query_ping_tool'],
            '连通': ['query_ping_tool', 'execute_traceroute'],
            '接口': ['query_interface_info', 'query_interface_configuration'],
            '流量': ['query_interface_traffic', 'query_interface_history_traffic'],
            '光': ['query_optical_module_power'],
            '光模块': ['query_optical_module_power'],
            '错包': ['query_interface_error_statistics'],
            'crc': ['query_interface_error_statistics'],
            '丢包': ['query_interface_drop_cache', 'query_physical_interface_car_drop'],
            '日志': ['query_device_logs'],
            '路由': ['query_route_table', 'execute_traceroute'],
            'arp': ['query_arp_table'],
            'mac': ['query_mac_address_table'],
            'qos': ['query_interface_qos_statistics'],
            'bgp': ['query_bgp_session_status'],
            'ospf': ['query_ospf_neighbor_status'],
            'vlan': ['query_vlan_information'],
            'cpu': ['query_device_cpu_memory'],
            '内存': ['query_device_cpu_memory'],
            '带宽': ['query_interface_bandwidth_utilization'],
            '配置': ['query_interface_configuration'],
            '历史': ['query_interface_history_traffic', 'query_device_logs'],
            '统计': ['query_interface_error_statistics', 'query_interface_qos_statistics'],
            '查询': ['query_interface_info', 'query_device_logs'],
            '检查': ['query_interface_info', 'query_ping_tool'],
            '分析': ['query_interface_error_statistics', 'auto_coding_tool'],
            '计算': ['auto_coding_tool'],
            '代码': ['auto_coding_tool'],
            '隔离': ['execute_port_isolation'],
            '复位': ['execute_interface_reset'],
            '重启': ['execute_interface_reset'],
            '生成树': ['query_spanning_tree_status'],
        }
        
        # 精确匹配加分
        for keyword, tools in keywords_map.items():
            if keyword in task_lower:
                for tool in tools:
                    scores[tool] = scores.get(tool, 0) + 10
        
        # 2. 模糊匹配（中分）- 在工具描述中搜索
        for tool in self.tool_list:
            tool_name = tool['name'].lower()
            tool_desc = tool['description'].lower()
            
            # 工具名称包含任务中的词
            task_words = task_lower.split()
            for word in task_words:
                if len(word) >= 2:  # 忽略单字
                    if word in tool_name:
                        scores[tool['name']] = scores.get(tool['name'], 0) + 5
                    if word in tool_desc:
                        scores[tool['name']] = scores.get(tool['name'], 0) + 3
        
        # 3. 字符串相似度匹配（低分）
        for tool in self.tool_list:
            tool_name = tool['name'].lower()
            tool_desc = tool['description'].lower()
            
            # 简单的字符串相似度（基于公共子串）
            similarity = self._calculate_similarity(task_lower, tool_desc)
            if similarity > 0.1:  # 相似度阈值
                scores[tool['name']] = scores.get(tool['name'], 0) + int(similarity * 10)
        
        # 4. 按得分排序
        if scores:
            sorted_tools = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = [tool for tool, score in sorted_tools[:top_n]]
        else:
            # 5. 兜底：如果还是没有任何匹配，返回最通用的工具
            recommendations = [
                'query_interface_info',
                'query_device_logs',
                'query_interface_traffic',
                'query_interface_error_statistics',
                'auto_coding_tool'
            ][:top_n]
        
        return recommendations
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度（简单版本）
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数 (0-1)
        """
        # 分词
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # 计算交集
        common = words1.intersection(words2)
        
        # Jaccard相似度
        similarity = len(common) / len(words1.union(words2))
        
        return similarity
    
    def export_to_json(self, output_path: str):
        """
        导出工具列表到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.tool_list, f, ensure_ascii=False, indent=2)
            print(f"✅ 工具列表已导出到: {output_path}")
        except Exception as e:
            print(f"❌ 导出失败: {e}")


def test_tool_manager():
    """测试工具管理器"""
    print("=" * 80)
    print("测试工具管理器")
    print("=" * 80)
    
    # 创建工具管理器
    manager = ToolManager('/mnt/user-data/outputs/available_tools.txt')
    
    print(f"\n加载的工具数量: {len(manager.tools)}")
    
    # 测试获取工具描述
    print("\n测试1：获取工具描述")
    print("-" * 80)
    tool_name = "query_ping_tool"
    description = manager.get_tool_description(tool_name)
    print(f"工具: {tool_name}")
    print(f"描述: {description}")
    
    # 测试验证工具
    print("\n测试2：验证工具")
    print("-" * 80)
    test_tools = ["query_ping_tool", "invalid_tool", "finish_diagnosis"]
    for tool in test_tools:
        is_valid = manager.is_valid_tool(tool)
        print(f"{tool}: {'✅ 有效' if is_valid else '❌ 无效'}")
    
    # 测试搜索工具
    print("\n测试3：搜索工具")
    print("-" * 80)
    keyword = "流量"
    results = manager.search_tools(keyword)
    print(f"搜索关键词: {keyword}")
    print(f"找到 {len(results)} 个工具:")
    for tool in results:
        print(f"  - {tool['name']}: {tool['description']}")
    
    # 测试推荐工具
    print("\n测试4：推荐工具")
    print("-" * 80)
    task = "查询接口流量统计"
    suggestions = manager.suggest_tool(task)
    print(f"任务: {task}")
    print(f"推荐工具: {suggestions}")
    
    # 测试获取紧凑列表
    print("\n测试5：紧凑工具列表（前5个）")
    print("-" * 80)
    compact = manager.get_tools_compact_list()
    lines = compact.split('\n')[:5]
    print('\n'.join(lines))
    print("...")
    
    # 导出到JSON
    print("\n测试6：导出到JSON")
    print("-" * 80)
    manager.export_to_json('/mnt/user-data/outputs/available_tools.json')
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == '__main__':
    test_tool_manager()
