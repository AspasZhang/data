"""
结构化数据生成器
按照 query -> response[step{cot, coa[{action, observation}]}] 格式生成
"""

import json
from typing import Dict, List, Any, Optional


class StructuredOutputGenerator:
    """结构化输出生成器"""
    
    def __init__(self):
        self.current_step = 0
        self.steps = []
        self.known_entities = {}  # 存储已知的实体列表，如{interface: [eth0, eth1, eth2]}
    
    def start_step(self, reasoning: str):
        """
        开始新的一步
        
        Args:
            reasoning: 这一步的CoT推理（做什么，为什么）
        """
        self.current_step += 1
        step_data = {
            f"step{self.current_step}": {
                "cot": reasoning,
                "coa": []
            }
        }
        self.steps.append(step_data)
    
    def add_action_observation(self, tool_name: str, tool_args: Dict, 
                               observation: Any, batch: bool = False):
        """
        在当前step的coa中添加action-observation对
        
        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            observation: 观察结果
            batch: 是否是批量操作的一部分
        """
        if not self.steps:
            raise ValueError("必须先调用start_step()开始新的步骤")
        
        current_step_key = f"step{self.current_step}"
        current_step = self.steps[-1][current_step_key]
        
        action_obs = {
            "action": {
                "name": tool_name,
                "args": tool_args
            },
            "observation": observation
        }
        
        current_step["coa"].append(action_obs)
    
    def update_known_entities(self, entity_type: str, entities: List[str]):
        """
        更新已知的实体列表
        
        Args:
            entity_type: 实体类型（如'interfaces', 'devices'）
            entities: 实体列表
        """
        self.known_entities[entity_type] = entities
    
    def get_known_entities(self, entity_type: str) -> List[str]:
        """获取已知的实体列表"""
        return self.known_entities.get(entity_type, [])
    
    def generate_output(self, query: str) -> Dict:
        """
        生成最终的结构化输出
        
        Args:
            query: 原始问题
            
        Returns:
            结构化的输出字典
        """
        return {
            "query": query,
            "response": self.steps
        }
    
    def to_json(self, query: str, indent: int = 2) -> str:
        """转换为JSON字符串"""
        output = self.generate_output(query)
        return json.dumps(output, ensure_ascii=False, indent=indent)


def extract_entities_from_observation(observation: Any, entity_type: str) -> List[str]:
    """
    从observation中提取实体列表
    
    Args:
        observation: 工具返回的结果
        entity_type: 要提取的实体类型（如'interface', 'device'）
        
    Returns:
        提取到的实体列表
    """
    entities = []
    
    if isinstance(observation, list):
        for item in observation:
            if isinstance(item, dict):
                # 尝试常见的键名
                for key in ['接口', 'interface', 'interface_name', '设备', 'device', 'device_name']:
                    if key in item:
                        entities.append(item[key])
                        break
    
    elif isinstance(observation, dict):
        # 单个结果
        for key in ['接口', 'interface', 'interface_name', '设备', 'device', 'device_name']:
            if key in observation:
                entities.append(observation[key])
                break
    
    return entities


def should_batch_execute(reasoning: str, entities: List[str]) -> bool:
    """
    判断是否应该批量执行
    
    Args:
        reasoning: 当前步骤的推理
        entities: 已知的实体列表
        
    Returns:
        是否应该批量执行
    """
    # 如果推理中提到"逐一"、"每个"、"所有"等词，且有多个实体
    batch_keywords = ['逐一', '每个', '所有', '批量', '遍历', '检查所有']
    
    if len(entities) > 1:
        for keyword in batch_keywords:
            if keyword in reasoning:
                return True
    
    return False


if __name__ == '__main__':
    # 测试
    print("="*80)
    print("结构化输出生成器测试")
    print("="*80 + "\n")
    
    generator = StructuredOutputGenerator()
    
    # Step 1: 查询所有接口
    generator.start_step("查询设备aggrleaf02_2_20.45的所有接口信息，确认接口基础配置")
    
    interfaces_result = [
        {"接口": "100GE2/1/1", "IP地址/掩码": "192.168.10.1/30"},
        {"接口": "40GE2/2/5", "IP地址/掩码": "10.200.1.1/24"},
        {"接口": "40GE2/2/6", "IP地址/掩码": "10.200.2.1/24"}
    ]
    
    generator.add_action_observation(
        "query_all_device_interfaces",
        {"device_name": "aggrleaf02_2_20.45"},
        interfaces_result
    )
    
    # 提取接口列表
    interfaces = extract_entities_from_observation(interfaces_result, 'interface')
    generator.update_known_entities('interfaces', interfaces)
    print(f"✅ Step 1完成，发现 {len(interfaces)} 个接口: {interfaces}\n")
    
    # Step 2: 批量检查所有接口状态
    generator.start_step("逐一检查接口状态，筛选出异常接口")
    
    # 对每个接口调用工具（在同一个step中）
    for interface in interfaces:
        result = {
            "接口": interface,
            "状态": "up" if interface != "40GE2/2/5" else "down",
            "速率": "100G" if "100GE" in interface else "40G"
        }
        
        generator.add_action_observation(
            "query_interface_public_info",
            {"device_name": "aggrleaf02_2_20.45", "interface_name": interface},
            result,
            batch=True
        )
    
    print(f"✅ Step 2完成，批量检查了 {len(interfaces)} 个接口\n")
    
    # 生成最终输出
    output = generator.generate_output("请分析设备aggrleaf02_2_20.45的接口是否有异常状态，并定位故障原因？")
    
    print("="*80)
    print("生成的结构化输出：")
    print("="*80)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    
    print("\n" + "="*80)
    print("✅ 测试完成！")
    print("="*80)
