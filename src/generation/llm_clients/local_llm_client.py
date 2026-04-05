# src/generation/llm_clients/local_llm_client.py

import logging
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

class LocalLLMClient:
    """
    本地大语言模型客户端
    支持 Qwen, ChatGLM, Baichuan 等 HuggingFace 模型
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct", device: str = "cuda"):
        """
        Args:
            model_name: 模型名称或本地路径
            device: 设备 ('cuda', 'cpu', 'mps') - 仅用于日志和 dtype 选择，不直接传给 pipeline
        """
        self.model_name = model_name
        self.device_str = device  # 重命名以避免混淆
        
        logger.info(f"🚀 正在加载本地 LLM: {model_name} (目标设备: {self.device_str})...")

        # 1. 确定数据类型
        if self.device_str == "cuda":
            torch_dtype = torch.float16
        elif self.device_str == "mps":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # 2. 加载 Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            # 确保 pad_token 存在
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"加载 Tokenizer 失败: {e}")
            raise e

        # 3. 加载模型
        # 关键点：使用 device_map="auto" 让 accelerate 自动分配设备
        # 这样就不需要手动调用 model.to(device)，也不会和 pipeline 的 device 参数冲突
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",  # ✅ 关键修改：自动管理设备
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            logger.info(f"✅ 模型加载完成。实际运行设备: {next(self.model.parameters()).device}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise e

        # 4. 创建 Pipeline
        # 关键点：传入 device=-1 (或者 device=None，取决于 transformers 版本)
        # 这告诉 pipeline：不要尝试移动模型，模型已经加载好了
        try:
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=None, 
                return_full_text=False,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            logger.info("✅ Pipeline 初始化成功")
        except Exception as e:
            logger.error(f"创建 Pipeline 失败: {e}")
            raise e

    def generate(self, prompt: str) -> str:
        """
        生成回答
        
        Args:
            prompt: 完整的提示词（包含 System Prompt 和 Context）
            
        Returns:
            生成的文本
        """
        try:
            # 执行生成
            # messages 格式兼容 Chat 模型 (如 Qwen-Instruct)
            # 如果模型是纯 Base 模型，可能需要直接传 prompt 字符串
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            outputs = self.pipe(text)
            
            if outputs and len(outputs) > 0:
                return outputs[0]["generated_text"].strip()
            else:
                return "未生成有效回答。"
                
        except Exception as e:
            logger.error(f"LLM 生成出错: {e}")
            return f"系统内部错误：{str(e)}"