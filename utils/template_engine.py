"""
YAML模板引擎 - 基于Jinja2的提示词模板系统
参考Paper2Poster项目的Role类实现
"""

import yaml
import os
from typing import Dict, Any, Optional
from jinja2 import Environment, Template, StrictUndefined
from dataclasses import dataclass


@dataclass
class TemplateConfig:
    """模板配置类"""
    system_prompt: str
    template: str
    jinja_args: list
    return_json: bool = False


class YAMLTemplateEngine:
    """
    YAML模板引擎类

    基于Jinja2模板引擎，用于处理YAML配置文件中的提示词模板
    支持参数注入和模板渲染
    """

    def __init__(self, templates_dir: str = "utils/prompt_templates"):
        """
        初始化模板引擎

        Args:
            templates_dir: 模板文件目录路径
        """
        self.templates_dir = templates_dir
        self._jinja_env = Environment(undefined=StrictUndefined)

    def load_template(self, template_name: str) -> TemplateConfig:
        """
        加载YAML模板配置文件

        Args:
            template_name: 模板名称（不含.yaml扩展名）

        Returns:
            TemplateConfig: 模板配置对象

        Raises:
            FileNotFoundError: 当模板文件不存在时
            yaml.YAMLError: 当YAML格式错误时
        """
        template_path = os.path.join(self.templates_dir, f"{template_name}.yaml")

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"模板文件不存在: {template_path}")

        with open(template_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return TemplateConfig(
            system_prompt=config_dict.get('system_prompt', ''),
            template=config_dict.get('template', ''),
            jinja_args=config_dict.get('jinja_args', []),
            return_json=config_dict.get('return_json', False)
        )

    def render_template(self,
                       template_name: str,
                       **kwargs) -> str:
        """
        渲染模板

        Args:
            template_name: 模板名称
            **kwargs: 模板参数

        Returns:
            str: 渲染后的提示词

        Raises:
            ValueError: 当提供的参数与模板要求不匹配时
        """
        config = self.load_template(template_name)

        # 检查参数是否完整
        expected_args = set(config.jinja_args)
        provided_args = set(kwargs.keys())

        if expected_args != provided_args:
            missing_args = expected_args - provided_args
            extra_args = provided_args - expected_args
            error_msg = []
            if missing_args:
                error_msg.append(f"缺少参数: {missing_args}")
            if extra_args:
                error_msg.append(f"多余参数: {extra_args}")
            raise ValueError("参数不匹配: " + ", ".join(error_msg))

        # 渲染模板
        template = self._jinja_env.from_string(config.template)
        return template.render(**kwargs)

    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """
        获取模板信息

        Args:
            template_name: 模板名称

        Returns:
            Dict: 包含模板配置信息的字典
        """
        config = self.load_template(template_name)

        return {
            'system_prompt': config.system_prompt,
            'template': config.template,
            'jinja_args': config.jinja_args,
            'return_json': config.return_json
        }

    def list_templates(self) -> list:
        """
        列出所有可用的模板

        Returns:
            list: 模板名称列表
        """
        if not os.path.exists(self.templates_dir):
            return []

        templates = []
        for file in os.listdir(self.templates_dir):
            if file.endswith('.yaml'):
                templates.append(file[:-5])  # 移除.yaml扩展名

        return sorted(templates)


# 创建全局模板引擎实例
template_engine = YAMLTemplateEngine()


def render_prompt(template_name: str, **kwargs) -> str:
    """
    便捷函数：快速渲染模板

    Args:
        template_name: 模板名称
        **kwargs: 模板参数

    Returns:
        str: 渲染后的提示词
    """
    return template_engine.render_template(template_name, **kwargs)


def get_prompt_template_info(template_name: str) -> Dict[str, Any]:
    """
    便捷函数：获取模板信息

    Args:
        template_name: 模板名称

    Returns:
        Dict: 模板配置信息
    """
    return template_engine.get_template_info(template_name)


if __name__ == "__main__":
    # 测试代码
    try:
        # 测试列出模板
        templates = template_engine.list_templates()
        print(f"可用的模板: {templates}")

        if 'image_table_filter_agent' in templates:
            # 测试加载模板信息
            info = template_engine.get_template_info('image_table_filter_agent')
            print("\n模板信息:")
            for key, value in info.items():
                print(f"  {key}: {value}")

            # 测试渲染模板（需要提供必要的参数）
            # 这里只是示例，实际使用时需要提供相应的参数
            # result = render_prompt('image_table_filter_agent',
            #                       image_information={},
            #                       table_information={},
            #                       json_content={})

    except Exception as e:
        print(f"测试失败: {e}")
