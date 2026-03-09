"""定时任务：从轨迹中提炼记忆（通过管线写入）."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


def load_trajectories_for_date(
    trajectories_dir: Path,
    target_date: date,
) -> list[dict]:
    """加载指定日期的轨迹.

    Args:
        trajectories_dir: 轨迹存储目录
        target_date: 目标日期

    Returns:
        轨迹列表
    """
    trajectories = []
    trajectories_file = trajectories_dir / "trajectories.jsonl"

    if not trajectories_file.exists():
        logger.info("轨迹文件不存在: %s", trajectories_file)
        return trajectories

    with open(trajectories_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                traj = json.loads(line)

                # 检查日期（从 metadata.timestamp 或 steps[0].timestamp）
                timestamp_str = None
                if "metadata" in traj and "timestamp" in traj["metadata"]:
                    timestamp_str = traj["metadata"]["timestamp"]
                elif "steps" in traj and traj["steps"]:
                    timestamp_str = traj["steps"][0].get("timestamp")

                if timestamp_str:
                    traj_date = datetime.fromisoformat(timestamp_str).date()
                    if traj_date == target_date:
                        trajectories.append(traj)

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.debug("跳过无效轨迹: %s", e)
                continue

    logger.info("加载 %s 的轨迹: %d 条", target_date, len(trajectories))
    return trajectories


def process_trajectories_batch(
    trajectories: list[dict],
    extractor,
    store,
    batch_size: int = 10,
    batch_delay: float = 2.0,
) -> dict[str, int]:
    """批量处理轨迹.

    Args:
        trajectories: 轨迹列表
        extractor: TrajectoryExtractor 实例
        store: MemoryStoreDB 实例
        batch_size: 每批处理数量
        batch_delay: 批次间延迟（秒）

    Returns:
        统计信息: {
            "total": 总数,
            "analyzed": 已分析,
            "extracted": 已提取,
            "memories_stored": 通过管线写入的记忆数,
            "errors": 错误数
        }
    """
    from ensoul.memory_pipeline import process_memory

    stats = {
        "total": len(trajectories),
        "analyzed": 0,
        "extracted": 0,
        "memories_stored": 0,
        "errors": 0,
    }

    for i in range(0, len(trajectories), batch_size):
        batch = trajectories[i : i + batch_size]
        logger.info(
            "处理批次 %d-%d / %d", i + 1, min(i + batch_size, len(trajectories)), len(trajectories)
        )

        for traj in batch:
            try:
                # 1. 分析价值
                analysis = extractor.analyze_trajectory(traj)
                stats["analyzed"] += 1

                if not analysis.get("should_extract", False):
                    logger.debug(
                        "轨迹价值不足，跳过: task_id=%s score=%.2f",
                        traj.get("task_id", "unknown"),
                        analysis.get("value_score", 0.0),
                    )
                    continue

                # 2. 提取记忆
                memories = extractor.extract_memories(traj)
                if not memories:
                    logger.debug("未提取到记忆: task_id=%s", traj.get("task_id", "unknown"))
                    continue

                stats["extracted"] += 1

                # 3. 通过管线写入记忆（skip_reflect: 已结构化数据）
                for mem in memories:
                    entry = process_memory(
                        raw_text=mem["content"],
                        employee=mem["employee"],
                        store=store,
                        skip_reflect=True,
                        category=mem["category"],
                        tags=mem.get("tags", []),
                        confidence=mem.get("confidence", 1.0),
                        source_session=mem.get("source_trajectory_id", ""),
                        keywords=mem.get("keywords"),
                        importance=mem.get("importance", 3),
                        trigger_condition=mem.get("trigger_condition", ""),
                        applicability=mem.get("applicability"),
                    )
                    if entry:
                        stats["memories_stored"] += 1

                logger.info(
                    "提取记忆: task_id=%s memories=%d",
                    traj.get("task_id", "unknown"),
                    len(memories),
                )

            except Exception as e:
                logger.error("处理轨迹失败: task_id=%s error=%s", traj.get("task_id", "unknown"), e)
                stats["errors"] += 1

        # 批次间延迟
        if i + batch_size < len(trajectories):
            logger.debug("批次延迟 %.1f 秒", batch_delay)
            time.sleep(batch_delay)

    return stats


def main():
    """主函数."""
    parser = argparse.ArgumentParser(description="从轨迹中提炼记忆并通过管线写入")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="手动触发模式（用于测试和紧急处理）",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="指定处理日期（YYYY-MM-DD），默认为昨天",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="每批处理数量（默认 10）",
    )
    parser.add_argument(
        "--batch-delay",
        type=float,
        default=2.0,
        help="批次间延迟秒数（默认 2.0）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="价值评分阈值（默认 0.7）",
    )

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 确定处理日期
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            logger.error("日期格式错误，应为 YYYY-MM-DD: %s", args.date)
            sys.exit(1)
    else:
        target_date = date.today() - timedelta(days=1)

    logger.info("开始提炼记忆: date=%s manual=%s", target_date, args.manual)

    # 初始化组件
    try:
        from ensoul.memory import get_memory_store
        from ensoul.trajectory_extractor import TrajectoryExtractor

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY 环境变量未设置")
            sys.exit(1)

        extractor = TrajectoryExtractor(
            api_key=api_key,
            value_threshold=args.threshold,
        )
        store = get_memory_store()

        # 加载轨迹
        trajectories_dir = Path(".ensoul/trajectories")
        trajectories = load_trajectories_for_date(trajectories_dir, target_date)

        if not trajectories:
            logger.info("没有需要处理的轨迹")
            sys.exit(0)

        # 批量处理
        stats = process_trajectories_batch(
            trajectories,
            extractor,
            store,
            batch_size=args.batch_size,
            batch_delay=args.batch_delay,
        )

        # 输出统计
        logger.info(
            "提炼完成: total=%d analyzed=%d extracted=%d stored=%d errors=%d",
            stats["total"],
            stats["analyzed"],
            stats["extracted"],
            stats["memories_stored"],
            stats["errors"],
        )

    except Exception as e:
        logger.exception("提炼任务失败: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
