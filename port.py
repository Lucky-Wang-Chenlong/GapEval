#!/usr/bin/env python3
"""

Usage / 使用方法:
1) Import and build tasks for a single data point:
   导入并构建单个数据点任务：
     from port import BenchmarkPort
     port = BenchmarkPort()
     tasks = port.build_tasks("/home/cdp/cl/data/data_1")
     print(tasks)

2) Build tasks for all data points:
   构建所有数据点任务：
     port = BenchmarkPort()
     all_tasks = port.build_all_tasks()

Notes / 说明:
- For category == "Multi Hop": image task is image generation (no input image).
  对于 "Multi Hop": 图像任务为图像生成（不需要输入图）。
- For other categories: image task is image editing and requires an input image from image/.
  对于其他类别: 图像任务为图像编辑，需要 image/ 目录中的输入图。
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# path to benchmark data fold
DATA_ROOT_DEFAULT = Path("/home/cdp/cl/data")


@dataclass
class TextTask:
    """Text understanding task / 文本理解任务"""
    task_type: str
    prompt: str


@dataclass
class ImageGenerationTask:
    """Image generation task / 图像生成任务"""
    task_type: str
    prompt: str


@dataclass
class ImageEditingTask:
    """Image editing task / 图像编辑任务"""
    task_type: str
    prompt: str
    image_path: str


@dataclass
class DataPointTasks:
    """All tasks for a data point / 单个数据点的全部任务"""
    data_id: str
    category: str
    text_task: TextTask
    image_task: object


class BenchmarkPort:
    """Unified interface to parse benchmark tasks / 解析基准任务的统一接口"""
    def __init__(self, data_root: Path = DATA_ROOT_DEFAULT):
        self.data_root = Path(data_root)

    def list_data_points(self) -> List[Path]:
        """List data_* directories / 列出所有 data_* 目录"""
        return sorted([p for p in self.data_root.iterdir() if p.is_dir() and p.name.startswith("data_")])

    def load_prompt(self, data_point_dir: Path) -> Dict:
        """Load prompt.json / 读取 prompt.json"""
        prompt_path = Path(data_point_dir) / "prompt.json"
        with open(prompt_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            if not data:
                raise ValueError(f"prompt.json is empty in {data_point_dir}")
            data = data[0]
        if not isinstance(data, dict):
            raise ValueError(f"prompt.json must be object in {data_point_dir}")
        return data

    def find_image(self, data_point_dir: Path) -> Optional[str]:
        """Find first image in image/ / 查找 image/ 下的首张图片"""
        image_dir = Path(data_point_dir) / "image"
        if not image_dir.exists():
            return None
        images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        return str(images[0]) if images else None

    def build_tasks(self, data_point_dir: Path) -> DataPointTasks:
        """Build tasks for one data point / 构建单个数据点任务"""
        data_point_dir = Path(data_point_dir)
        prompt_data = self.load_prompt(data_point_dir)
        category = prompt_data.get("category", "")
        und_prompt = prompt_data.get("und_prompt", "")
        gen_prompt = prompt_data.get("gen_prompt", "")

        if not und_prompt:
            raise ValueError(f"missing und_prompt in {data_point_dir}")
        if not gen_prompt:
            raise ValueError(f"missing gen_prompt in {data_point_dir}")

        text_task = TextTask(task_type="text_understanding", prompt=und_prompt)

        if category == "Multi Hop":
            image_task = ImageGenerationTask(task_type="image_generation", prompt=gen_prompt)
        else:
            image_path = self.find_image(data_point_dir)
            if not image_path:
                raise FileNotFoundError(f"missing image in {data_point_dir}/image")
            image_task = ImageEditingTask(
                task_type="image_editing",
                prompt=gen_prompt,
                image_path=image_path,
            )

        return DataPointTasks(
            data_id=data_point_dir.name,
            category=category,
            text_task=text_task,
            image_task=image_task,
        )

    def build_all_tasks(self) -> List[DataPointTasks]:
        """Build tasks for all data points / 构建所有数据点任务"""
        tasks = []
        for data_point_dir in self.list_data_points():
            tasks.append(self.build_tasks(data_point_dir))
        return tasks


if __name__ == "__main__":
    port = BenchmarkPort()
    for item in port.build_all_tasks():
        print(item)
