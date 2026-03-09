"""头像生成模块 — 调用通义万相文生图 + Pillow 压缩."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

AVATAR_SIZE = 256
AVATAR_QUALITY = 85

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
DASHSCOPE_TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks/{}"
DASHSCOPE_MODEL = "wanx2.0-t2i-turbo"
DASHSCOPE_SIZE = "768*768"

# 生成任务轮询参数
POLL_INTERVAL = 5  # 秒
POLL_MAX_ATTEMPTS = 60  # 最多等 5 分钟

PHOTO_SUFFIX = (
    "。Professional headshot photo. Neutral light grey background, "
    "soft studio lighting, head and shoulders framing, "
    "looking at camera with natural expression. "
    "High quality, photorealistic, corporate profile photo style."
)


def _get_pillow():
    """延迟导入 Pillow，未安装时返回 None."""
    try:
        from PIL import Image

        return Image
    except ImportError:
        return None


def _get_api_key() -> str:
    """读取 DashScope API Key."""
    return os.environ.get("DASHSCOPE_API_KEY", "")


def build_avatar_prompt(
    display_name: str = "",
    character_name: str = "",
    description: str = "",
    avatar_prompt: str = "",
) -> str:
    """构建头像生成 prompt.

    优先使用 avatar_prompt，否则自动推断。
    """
    if avatar_prompt:
        return avatar_prompt + PHOTO_SUFFIX

    name = character_name or display_name or "a professional"
    role = description or "employee"
    return f"{name}，{role}" + PHOTO_SUFFIX


def generate_avatar(
    display_name: str = "",
    character_name: str = "",
    description: str = "",
    avatar_prompt: str = "",
    output_dir: Path | None = None,
) -> Path | None:
    """调用通义万相生成头像.

    需要环境变量 DASHSCOPE_API_KEY。

    Returns:
        生成的图片路径（avatar_raw.png），失败返回 None
    """
    api_key = _get_api_key()
    if not api_key:
        logger.error("DASHSCOPE_API_KEY 未设置")
        return None

    if output_dir is None:
        output_dir = Path.cwd()

    prompt = build_avatar_prompt(
        display_name=display_name,
        character_name=character_name,
        description=description,
        avatar_prompt=avatar_prompt,
    )

    # 提交异步任务
    payload = json.dumps(
        {
            "model": DASHSCOPE_MODEL,
            "input": {"prompt": prompt},
            "parameters": {"size": DASHSCOPE_SIZE, "n": 1},
        }
    ).encode()

    req = urllib.request.Request(
        DASHSCOPE_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        logger.error("提交生成任务失败: %s", e)
        return None

    task_id = result.get("output", {}).get("task_id")
    if not task_id:
        logger.error("未获取到 task_id: %s", result)
        return None

    logger.info("已提交生成任务: %s", task_id)

    # 轮询等待完成
    for attempt in range(1, POLL_MAX_ATTEMPTS + 1):
        time.sleep(POLL_INTERVAL)

        poll_req = urllib.request.Request(
            DASHSCOPE_TASK_URL.format(task_id),
            headers={"Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(poll_req, timeout=15) as poll_resp:
                data = json.loads(poll_resp.read())
        except Exception as e:
            logger.warning("轮询任务失败 (attempt %d): %s", attempt, e)
            continue

        status = data.get("output", {}).get("task_status", "")

        if status == "SUCCEEDED":
            results = data.get("output", {}).get("results", [])
            if not results:
                logger.error("任务成功但无结果")
                return None
            img_url = results[0].get("url")
            if not img_url:
                logger.error("任务成功但无图片 URL")
                return None

            # 下载图片（仅允许 HTTPS）
            output_path = output_dir / "avatar_raw.png"
            try:
                if not img_url.startswith("https://"):
                    logger.error("头像 URL 必须使用 HTTPS: %s", img_url)
                    return None
                urllib.request.urlretrieve(img_url, str(output_path))
                if not output_path.exists() or output_path.stat().st_size == 0:
                    logger.error("下载的图片文件为空")
                    output_path.unlink(missing_ok=True)
                    return None
                logger.info(
                    "头像已下载: %s (%d KB)", output_path, output_path.stat().st_size // 1024
                )
                return output_path
            except Exception as e:
                logger.error("下载图片失败: %s", e)
                output_path.unlink(missing_ok=True)
                return None

        elif status == "FAILED":
            msg = data.get("output", {}).get("message", "未知错误")
            logger.error("生成失败: %s", msg)
            return None

        # PENDING / RUNNING → 继续等待

    logger.error("生成超时（%d 次轮询）", POLL_MAX_ATTEMPTS)
    return None


def compress_avatar(input_path: Path, output_path: Path | None = None) -> Path | None:
    """压缩头像到 web 规格（256x256 webp）.

    Args:
        input_path: 原始图片路径
        output_path: 输出路径（默认同目录 avatar.webp）

    Returns:
        压缩后的图片路径，失败返回 None
    """
    Image = _get_pillow()
    if Image is None:
        logger.error("Pillow 未安装，请执行: pip install Pillow")
        return None

    if output_path is None:
        output_path = input_path.parent / "avatar.webp"

    try:
        img = Image.open(input_path)

        # RGBA/P → RGB
        if img.mode in ("RGBA", "LA", "P"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[-1] if "A" in img.mode else None)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # 中心裁剪为正方形
        w, h = img.size
        if w != h:
            size = min(w, h)
            left = (w - size) // 2
            top = (h - size) // 2
            img = img.crop((left, top, left + size, top + size))

        # 缩放到目标尺寸
        if img.size[0] != AVATAR_SIZE:
            img = img.resize((AVATAR_SIZE, AVATAR_SIZE), Image.LANCZOS)

        # 保存为 webp
        img.save(output_path, "WEBP", quality=AVATAR_QUALITY, method=4)

        # 删除原图（如果不同文件）
        if input_path != output_path and input_path.suffix != ".webp":
            input_path.unlink(missing_ok=True)

        return output_path

    except Exception as e:
        logger.error("头像压缩失败: %s", e)
        return None
