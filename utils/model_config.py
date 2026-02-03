"""
模型配置工具 - 基于Paper2Poster项目的wei_utils.py实现

提供统一的模型配置接口，支持多种模型平台
"""

import os
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig, QwenConfig, VLLMConfig, OpenRouterConfig, GeminiConfig


def get_agent_config(model_type: str) -> dict:
    """
    获取模型配置 - 基于Paper2Poster项目的wei_utils.py

    Args:
        model_type: 模型类型，支持以下类型：
            - qwen: Qwen2.5-72B
            - gemini: Gemini 2 Flash
            - phi4: Phi-4 Multimodal
            - llama-4-scout-17b-16e-instruct: Llama 4 Scout
            - qwen-2.5-vl-72b: Qwen2.5-VL-72B
            - gemma: Gemma 3 4B
            - llava: LLaVA OneVision Qwen2 7B
            - molmo-o: Molmo 7B
            - qwen-2-vl-7b: Qwen2-VL-7B
            - vllm_phi4: Phi-4 Multimodal (VLLM)
            - o3-mini: OpenAI o3-mini
            - gpt-4.1: OpenAI GPT-4.1
            - gpt-4.1-mini: OpenAI GPT-4.1-mini
            - 4o: OpenAI GPT-4o
            - 4o-mini: OpenAI GPT-4o-mini
            - o1: OpenAI o1
            - o3: OpenAI o3
            - gpt-5: OpenAI GPT-5
            - vllm_qwen_vl: Qwen2.5-VL-7B (VLLM)
            - vllm_qwen: Qwen2.5-7B (VLLM)
            - openrouter_qwen_72b: Qwen2.5-72B (OpenRouter)
            - openrouter_qwen_vl_72b: Qwen2.5-VL-72B (OpenRouter)
            - openrouter_qwen_vl_7b: Qwen2.5-VL-7B (OpenRouter)
            - openrouter_qwen_7b: Qwen2.5-7B (OpenRouter)

    Returns:
        dict: 模型配置字典，包含model_type, model_config, model_platform等信息
    """
    agent_config = {}

    if model_type == 'qwen':
        agent_config = {
            "model_type": ModelType.DEEPINFRA_QWEN_2_5_72B,
            "model_config": QwenConfig().as_dict(),
            "model_platform": ModelPlatformType.DEEPINFRA,
        }
    elif model_type == 'gemini':
        agent_config = {
            "model_type": ModelType.DEEPINFRA_GEMINI_2_FLASH,
            "model_config": GeminiConfig().as_dict(),
            "model_platform": ModelPlatformType.DEEPINFRA,
            'max_images': 99
        }
    elif model_type == 'phi4':
        agent_config = {
            "model_type": ModelType.DEEPINFRA_PHI_4_MULTIMODAL,
            "model_config": QwenConfig().as_dict(),
            "model_platform": ModelPlatformType.DEEPINFRA,
        }
    elif model_type == 'llama-4-scout-17b-16e-instruct':
        agent_config = {
            'model_type': ModelType.ALIYUN_LLAMA4_SCOUT_17B_16E,
            'model_config': QwenConfig().as_dict(),
            'model_platform': ModelPlatformType.QWEN,
            'max_images': 99
        }
    elif model_type == 'qwen-2.5-vl-72b':
        agent_config = {
            'model_type': ModelType.QWEN_2_5_VL_72B,
            'model_config': QwenConfig().as_dict(),
            'model_platform': ModelPlatformType.QWEN,
            'max_images': 99
        }
    elif model_type == 'gemma':
        agent_config = {
            "model_type": "google/gemma-3-4b-it",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:5555/v1',
            'max_images': 99
        }
    elif model_type == 'llava':
        agent_config = {
            "model_type": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:8000/v1',
            'max_images': 99
        }
    elif model_type == 'molmo-o':
        agent_config = {
            "model_type": "allenai/Molmo-7B-O-0924",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:8000/v1',
            'max_images': 99
        }
    elif model_type == 'qwen-2-vl-7b':
        agent_config = {
            "model_type": "Qwen/Qwen2-VL-7B-Instruct",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:8000/v1',
            'max_images': 99
        }
    elif model_type == 'vllm_phi4':
        agent_config = {
            "model_type": "microsoft/Phi-4-multimodal-instruct",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:8000/v1',
            'max_images': 99
        }
    elif model_type == 'o3-mini':
        agent_config = {
            "model_type": ModelType.O3_MINI,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == 'gpt-4.1':
        agent_config = {
            "model_type": ModelType.GPT_4_1,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == 'gpt-4.1-mini':
        agent_config = {
            "model_type": ModelType.GPT_4_1_MINI,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == '4o':
        agent_config = {
            "model_type": ModelType.GPT_4O,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == '4o-mini':
        agent_config = {
            "model_type": ModelType.GPT_4O_MINI,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == 'o1':
        agent_config = {
            "model_type": ModelType.O1,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == 'o3':
        agent_config = {
            "model_type": ModelType.O3,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == 'gpt-5':
        agent_config = {
            "model_type": ModelType.GPT_5,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == 'vllm_qwen_vl':
        agent_config = {
            "model_type": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:7000/v1'
        }
    elif model_type == 'vllm_qwen':
        agent_config = {
            "model_type": "Qwen/Qwen2.5-7B-Instruct",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:8000/v1',
        }
    elif model_type == 'openrouter_qwen_72b':
        agent_config = {
            'model_type': ModelType.OPENROUTER_QWEN_2_5_72B,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    elif model_type == 'openrouter_qwen_vl_72b':
        agent_config = {
            'model_type': ModelType.OPENROUTER_QWEN_2_5_VL_72B,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    elif model_type == 'openrouter_qwen_vl_7b':
        agent_config = {
            'model_type': ModelType.OPENROUTER_QWEN_2_5_VL_7B,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    elif model_type == 'openrouter_qwen_7b':
        agent_config = {
            'model_type': ModelType.OPENROUTER_QWEN_2_5_7B,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    elif model_type == 'gpt5_mini':
        agent_config = {
            'model_type': ModelType.OPENROUTER_GPT_5_MINI,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    else:
        # 默认配置 - OpenAI兼容模型
        agent_config = {
            'model_type': model_type,
            'model_platform': ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            'model_config': None
        }

    return agent_config


def get_supported_models() -> dict:
    """
    获取支持的模型列表

    Returns:
        dict: 模型类型到描述的映射
    """
    return {
        'qwen': 'Qwen2.5-72B (DeepInfra)',
        'gemini': 'Gemini 2 Flash (DeepInfra)',
        'phi4': 'Phi-4 Multimodal (DeepInfra)',
        'llama-4-scout-17b-16e-instruct': 'Llama 4 Scout 17B (Qwen)',
        'qwen-2.5-vl-72b': 'Qwen2.5-VL-72B (Qwen)',
        'gemma': 'Gemma 3 4B (VLLM)',
        'llava': 'LLaVA OneVision Qwen2 7B (VLLM)',
        'molmo-o': 'Molmo 7B (VLLM)',
        'qwen-2-vl-7b': 'Qwen2-VL-7B (VLLM)',
        'vllm_phi4': 'Phi-4 Multimodal (VLLM)',
        'o3-mini': 'OpenAI o3-mini',
        'gpt-4.1': 'OpenAI GPT-4.1',
        'gpt-4.1-mini': 'OpenAI GPT-4.1-mini',
        '4o': 'OpenAI GPT-4o',
        '4o-mini': 'OpenAI GPT-4o-mini',
        'o1': 'OpenAI o1',
        'o3': 'OpenAI o3',
        'gpt-5': 'OpenAI GPT-5',
        'vllm_qwen_vl': 'Qwen2.5-VL-7B (VLLM)',
        'vllm_qwen': 'Qwen2.5-7B (VLLM)',
        'openrouter_qwen_72b': 'Qwen2.5-72B (OpenRouter)',
        'openrouter_qwen_vl_72b': 'Qwen2.5-VL-72B (OpenRouter)',
        'openrouter_qwen_vl_7b': 'Qwen2.5-VL-7B (OpenRouter)',
        'openrouter_qwen_7b': 'Qwen2.5-7B (OpenRouter)',
    }


if __name__ == "__main__":
    # 测试代码
    print("支持的模型列表:")
    models = get_supported_models()
    for model_type, description in models.items():
        config = get_agent_config(model_type)
        print(f"  {model_type:30} -> {description}")
        print(f"    平台: {config['model_platform']}")
        print(f"    类型: {config['model_type']}")
        print()
