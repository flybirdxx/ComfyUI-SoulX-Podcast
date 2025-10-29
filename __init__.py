"""
ComfyUI-SoulX-Podcast 自定义节点
将 SoulX-Podcast 的核心功能封装为 ComfyUI 节点
"""

import os
import sys

# 将当前目录添加到 Python 路径，以便 soulxpodcast 的绝对导入可以正常工作
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ComfyUI 自定义类型定义（用于类型传递）
class SOULX_MODEL:
    pass

class PODCAST_INPUT:
    pass

# 导入节点实现
from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

# 导出节点映射（ComfyUI 要求）
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

