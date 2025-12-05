def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        解析JSON响应，确保总是返回dict
        
        处理策略：
        1. 如果解析出dict，直接返回
        2. 如果解析出list：
           - 如果list只有1个dict元素，返回该dict
           - 否则包装成 {"data": list}
        3. 如果解析出其他类型，包装成 {"value": parsed_data}
        4. 如果解析失败，返回错误dict
        """
        try:
            # 提取JSON代码块
            if '```json' in response:
                json_start = response.find('```json') + 7
                json_end = response.find('```', json_start)
                json_str = response[json_start:json_end].strip()
            elif '```' in response:
                json_start = response.find('```') + 3
                json_end = response.find('```', json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            # 解析JSON
            parsed = json.loads(json_str)
            
            # 类型检查和转换
            if isinstance(parsed, dict):
                # 已经是dict，直接返回
                return parsed
            elif isinstance(parsed, list):
                # 是list，需要转换
                if len(parsed) == 1 and isinstance(parsed[0], dict):
                    # list只有1个dict元素，返回该dict
                    print(f"⚠️  LLM返回了单元素list，自动提取: {list(parsed[0].keys())}")
                    return parsed[0]
                else:
                    # 多个元素或非dict元素，包装成dict
                    print(f"⚠️  LLM返回了list（{len(parsed)}个元素），包装为dict")
                    return {"data": parsed, "_type": "list"}
            else:
                # 其他类型（string, number等），包装成dict
                print(f"⚠️  LLM返回了{type(parsed).__name__}类型，包装为dict")
                return {"value": parsed, "_type": type(parsed).__name__}
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            print(f"   原始响应: {response[:200]}...")
            return {"error": "JSON解析失败", "raw": response[:500]}
        except Exception as e:
            print(f"❌ 解析异常: {e}")
            return {"error": "解析失败", "details": str(e), "raw": response[:500]}
