"""LiDAR Lab: open-source Higgsfield-style depth effects for ComfyUI."""
from .lidarlab.nodes import LidarLabExtension

WEB_DIRECTORY = None


async def comfy_entrypoint() -> LidarLabExtension:
    return LidarLabExtension()
