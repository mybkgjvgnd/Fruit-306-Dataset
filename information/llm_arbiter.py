"""
LLM Arbiter: 基于Qwen-VL-Plus的多模态仲裁器
从4.py提取的LLM仲裁核心代码，支持CoT推理和水果描述增强
"""

import os
import base64
import re
import time
import logging
from typing import List, Tuple, Dict, Optional

# 阿里云百炼 SDK
import dashscope
from dashscope import MultiModalConversation

# 配置日志
logger = logging.getLogger(__name__)


class FruitLLMArbiter:
    """
    水果分类LLM仲裁器
    使用Qwen-VL-Plus进行多模态思维链推理
    """
    
    def __init__(self, api_key: str, fruit_descriptions: Dict[str, str] = None):
        """
        Args:
            api_key: DashScope API密钥
            fruit_descriptions: 水果描述字典 {class_name: description}
        """
        self.api_key = api_key
        dashscope.api_key = api_key
        self.fruit_descriptions = fruit_descriptions or {}
        self.default_desc = "未知特征，请基于图像分析颜色、形状、纹理、斑点等视觉细节"
        
    def _build_prompt(self, top3_classes: List[Tuple[str, float]]) -> str:
        """
        构建增强的CoT提示词
        
        Args:
            top3_classes: Top-3候选类别列表 [(class_name, probability), ...]
        
        Returns:
            system_prompt: 系统提示词
            user_prompt: 用户提示词（不含图片）
        """
        # 构建选项文本，添加关键特征描述
        options_text = ""
        for i, (cls_name, prob) in enumerate(top3_classes):
            desc = self.fruit_descriptions.get(cls_name, self.default_desc)
            options_text += f"{chr(65+i)}. {cls_name} (模型置信度：{prob:.2%}) - 关键特征：{desc}\n"
        
        # 专家级 Prompt 设计
        system_prompt = """你是一位拥有20年经验的水果分类学家，擅长通过微观特征区分相似品种。
你的任务是从给定的三个候选项中，找出图片中真正的水果种类。

请严格遵循以下【三步推理法】，不要跳过任何步骤：

1. 【视觉特征深度分析】
   - 仔细观察果柄：长度、粗细、连接处凹陷程度、颜色。
   - 仔细观察表皮：纹理（光滑/粗糙/网状）、斑点（形状/分布/颜色）、光泽度、底色。
   - 仔细观察形态：整体形状（圆/椭圆/不规则）、顶部/底部特征。

2. 【排他性论证】
   - 针对选项A：指出图片中哪一个特征证明它【不是】A（必须具体，与描述对比）。
   - 针对选项B：指出图片中哪一个特征证明它【不是】B（必须具体，与描述对比）。
   - 针对选项C：指出图片中哪两个特征强有力地支持它是C（必须具体，与描述匹配）。

3. 【最终结论】
   - 综合上述分析，给出唯一的正确选项字母。

注意：即使模型给出的置信度很高，如果视觉特征明显不符，也要敢于纠正。
优先基于关键特征描述与图像匹配度判断。

输出格式要求：
最后一行必须严格遵循格式："FINAL_ANSWER: [字母]"，例如 "FINAL_ANSWER: B"。
"""

        user_content = f"""
候选类别列表（包含关键特征描述）：
{options_text}

请开始你的专业鉴定：
"""
        
        return system_prompt, user_content
    
    def arbitrate(self, image_path: str, top3_classes: List[Tuple[str, float]], 
                  timeout: int = 60) -> Tuple[str, str, str]:
        """
        调用Qwen-VL进行仲裁
        
        Args:
            image_path: 图片路径
            top3_classes: Top-3候选类别列表 [(class_name, probability), ...]
            timeout: API超时时间（秒）
        
        Returns:
            choice_letter: 选择的选项字母 ('A', 'B', 或 'C')
            class_name: 选择的类别名称
            reasoning_text: 完整的推理文本
        """
        try:
            # 读取并编码图片
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # 构建提示词
            system_prompt, user_content = self._build_prompt(top3_classes)
            
            messages = [
                {
                    "role": "system",
                    "content": [{"text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {"image": f"data:image/jpeg;base64,{base64_image}"},
                        {"text": user_content}
                    ]
                }
            ]
            
            logger.info(f"[LLM] Sending request to Qwen-VL-Plus...")
            start_time = time.time()
            
            response = MultiModalConversation.call(
                model="qwen-vl-plus",
                messages=messages,
                stream=False,
                request_timeout=timeout
            )
            
            elapsed = time.time() - start_time
            logger.info(f"[LLM] Request completed in {elapsed:.2f}s")
            
            if response.status_code != 200:
                raise Exception(f"API call failed: Code={response.code}, Msg={response.message}")
            
            raw_text = response.output.choices[0].message.content[0].text
            logger.info(f"[LLM] Raw response preview: {raw_text[:200]}...")
            
            # 解析最终答案
            choice = self._parse_choice(raw_text, len(top3_classes))
            
            # 获取对应的类别名称
            idx = ord(choice) - ord('A')
            if 0 <= idx < len(top3_classes):
                class_name = top3_classes[idx][0]
            else:
                class_name = top3_classes[0][0]
                logger.warning(f"[LLM] Parsed index {idx} out of range, fallback to top1")
            
            return choice, class_name, raw_text
            
        except Exception as e:
            logger.error(f"[LLM] Arbitration failed: {str(e)}")
            raise
    
    def _parse_choice(self, text: str, num_options: int = 3) -> str:
        """
        从LLM响应中解析选择的选项字母
        
        Args:
            text: LLM响应文本
            num_options: 选项数量（通常为3）
        
        Returns:
            选项字母 ('A', 'B', 'C')
        """
        # 尝试匹配 FINAL_ANSWER: X 格式
        match = re.search(r"FINAL_ANSWER:\s*([A-C])", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # 尝试备用模式：答案.*?X
        match = re.search(r"答案.*?([A-C])", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # 尝试查找单独的字母
        valid_letters = [chr(65 + i) for i in range(num_options)]
        for letter in valid_letters:
            if re.search(rf"\b{letter}\b", text):
                return letter
        
        # 默认返回A
        logger.warning(f"[LLM] Could not parse choice from text, defaulting to A")
        return 'A'
    
    def batch_arbitrate(self, image_paths: List[str], candidates_list: List[List[Tuple[str, float]]],
                        max_retries: int = 1) -> List[Tuple[str, str, str]]:
        """
        批量仲裁（逐张调用，避免超限）
        
        Args:
            image_paths: 图片路径列表
            candidates_list: 每个图片对应的Top-3候选列表
            max_retries: 失败重试次数
        
        Returns:
            results: 列表，每个元素为 (choice_letter, class_name, reasoning_text)
        """
        results = []
        
        for img_path, candidates in zip(image_paths, candidates_list):
            for attempt in range(max_retries + 1):
                try:
                    result = self.arbitrate(img_path, candidates)
                    results.append(result)
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt+1} failed for {img_path}: {e}")
                    if attempt == max_retries:
                        # 最后一次失败，使用默认值
                        results.append(('A', candidates[0][0], f"LLM failed: {e}"))
                    time.sleep(1)  # 重试前等待
        
        return results


# ==================== 与集成系统集成的辅助函数 ====================

def create_llm_arbiter(api_key: str, class_mapping_path: str = None) -> FruitLLMArbiter:
    """
    创建LLM仲裁器实例，可选加载水果描述
    
    Args:
        api_key: DashScope API密钥
        class_mapping_path: 类别映射CSV路径（包含description列）
    
    Returns:
        FruitLLMArbiter实例
    """
    descriptions = {}
    
    if class_mapping_path and os.path.exists(class_mapping_path):
        try:
            import pandas as pd
            df = pd.read_csv(class_mapping_path)
            
            # 自动检测class_name列
            if 'class_name' in df.columns:
                col_name = 'class_name'
            elif 'class' in df.columns:
                col_name = 'class'
            else:
                col_name = df.columns[0]
            
            if 'description' in df.columns:
                for _, row in df.iterrows():
                    cls_name = row[col_name]
                    if pd.notna(row['description']):
                        descriptions[cls_name] = row['description']
                logger.info(f"[LLM] Loaded {len(descriptions)} fruit descriptions")
        except Exception as e:
            logger.warning(f"[LLM] Failed to load descriptions: {e}")
    
    return FruitLLMArbiter(api_key, descriptions)


# ==================== 决策集成函数 ====================

def should_trigger_llm(ensemble_confidence: float, threshold: float = 0.60) -> bool:
    """
    判断是否应该触发LLM仲裁
    
    Args:
        ensemble_confidence: 集成模型最高置信度
        threshold: 触发阈值
    
    Returns:
        True表示应该触发LLM
    """
    return ensemble_confidence < threshold


def integrate_with_ensemble(ensemble_result: dict, llm_arbiter: FruitLLMArbiter,
                           image_path: str, threshold: float = 0.60) -> dict:
    """
    将LLM仲裁与集成模型结果整合
    
    Args:
        ensemble_result: 集成模型预测结果
        llm_arbiter: LLM仲裁器实例
        image_path: 图片路径
        threshold: LLM触发阈值
    
    Returns:
        整合后的结果字典
    """
    max_confidence = ensemble_result.get('max_confidence', 0)
    top3_classes = ensemble_result.get('stage1_top3', [])
    
    # 判断是否触发LLM
    if should_trigger_llm(max_confidence, threshold) and top3_classes:
        logger.info(f"[Decision] Confidence {max_confidence:.2f} < {threshold}, triggering LLM")
        
        try:
            choice_letter, final_class, reasoning = llm_arbiter.arbitrate(image_path, top3_classes)
            
            ensemble_result['used_llm'] = True
            ensemble_result['final_class'] = final_class
            ensemble_result['llm_choice'] = choice_letter
            ensemble_result['decision_path'] = 'llm_arbitration'
            ensemble_result['llm_reasoning'] = reasoning
            
        except Exception as e:
            logger.error(f"[Decision] LLM arbitration failed: {e}")
            ensemble_result['used_llm'] = True
            ensemble_result['final_class'] = top3_classes[0][0]
            ensemble_result['decision_path'] = 'llm_failed_fallback'
            ensemble_result['arbitration_error'] = str(e)
    else:
        ensemble_result['used_llm'] = False
        ensemble_result['final_class'] = top3_classes[0][0]
        ensemble_result['decision_path'] = 'high_confidence_direct'
    
    return ensemble_result


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 示例：创建LLM仲裁器
    arbiter = create_llm_arbiter(
        api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxx",
        class_mapping_path="./fruit_name_mapping.txt"
    )
    
    # 示例：模拟集成结果
    ensemble_result = {
        'max_confidence': 0.55,
        'stage1_top3': [
            ('red_fuji', 0.55),
            ('gala', 0.30),
            ('green_apple', 0.15)
        ]
    }
    
    # 示例：整合决策
    final_result = integrate_with_ensemble(
        ensemble_result=ensemble_result,
        llm_arbiter=arbiter,
        image_path="test_image.jpg",
        threshold=0.60
    )
    
    print(f"Final decision: {final_result.get('final_class')}")
    print(f"Decision path: {final_result.get('decision_path')}")