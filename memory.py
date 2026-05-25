"""
dialogue_to_memory.py

将 DialogueLoader 加载的对话场景，通过 LLM 提取并分类记忆，
保存为结构化 JSON 文件。

记忆类型：
  - relation    : 角色之间的人际关系
  - preference  : 偏好、喜好、厌恶
  - experience  : 经历、事件、故事
  - style       : 说话风格、表达习惯

每条记忆格式：
  {
    "type"       : "relation" | "preference" | "experience" | "style",
    "text"       : "...",
    "importance" : 1-10  (LLM 打分)
  }
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from config import Config
from dialogue_loader import DialogueLoader, DialogueScene

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─── LLM client ───────────────────────────────────────────────────────────────

def _make_client() -> OpenAI:
    return OpenAI(api_key=Config.api_key, base_url=Config.url)


# ─── Prompt construction ──────────────────────────────────────────────────────

_SYSTEM_PROMPT = """你是一个角色记忆分析专家。
给你一段对话场景，请从中提取关于主角（{agent_name}）的记忆，分为以下四类：
- relation   : 与其他角色之间的人际关系、态度、互动模式
- preference : 明确或隐含的偏好、喜好、厌恶、价值观
- experience : 发生过的具体事件、经历、故事
- style      : 说话风格、语言习惯、表达特点、幽默感等

输出严格 JSON，格式如下（数组，可以为空）：
[
  {
    "type": "<relation|preference|experience|style>",
    "text": "<简洁的记忆描述，不超过 100 字>",
    "importance": <1-10之间的整数，10 最重要>
  },
  ...
]

评分参考：
  1-3  : 非常次要，一般性互动或口头禅
  4-6  : 中等重要，体现角色特征或关系发展
  7-9  : 重要，明确揭示性格、观点或关键关系
  10   : 极重要，核心性格特质或关键情节

只输出 JSON 数组，不要任何解释文字。"""


def _normalize_importance(value: object, default: int = 5) -> int:
    """接受 1-10 整数或 0-1 浮点分值，统一转成 1-10 整数。"""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default

    if 0.0 <= score <= 1.0:
        return max(1, min(10, round(score * 10)))
    return max(1, min(10, round(score)))


def _build_user_message(scene: DialogueScene, agent_name: str) -> str:
    """将场景序列化为文本，供 LLM 分析。"""
    lines = []
    lines.append(f"【情绪】{scene.emotion}")
    lines.append(f"【地点】{scene.location}")
    lines.append(f"【话题】{', '.join(scene.topic)}")
    lines.append(f"【场景描述】{' '.join(scene.setting)}")
    lines.append("【对话】")
    for turn in scene.dialogue:
        prefix = "💬" if turn.is_speaking else "💭"
        lines.append(f"  {prefix} {turn.role} {turn.action}: {turn.content}")
    return "\n".join(lines)


# ─── Single-scene extraction ──────────────────────────────────────────────────

def extract_memories_from_scene(
    client: OpenAI,
    scene: DialogueScene,
    agent_name: str,
    *,
    retries: int = 2,
    retry_delay: float = 2.0,
) -> list[dict]:
    """
    调用 LLM 从单个场景中提取记忆列表。

    Returns
    -------
    list of dicts with keys: type, text, importance, source
    """
    system_msg = _SYSTEM_PROMPT.replace("{agent_name}", agent_name)
    user_msg = _build_user_message(scene, agent_name)

    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=Config.model_name,
                max_tokens=Config.max_tokens,
                temperature=Config.temperature,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw_text = resp.choices[0].message.content.strip()

            # 兼容 LLM 有时会用 ```json ... ``` 包裹
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            memories: list[dict] = json.loads(raw_text)

            # 校验 & 注入 source
            valid = []
            for m in memories:
                if not isinstance(m, dict):
                    continue
                m_type = str(m.get("type", "")).strip().lower()
                if m_type not in {"relation", "preference", "experience", "style"}:
                    logger.warning("Skipping unknown memory type: %s", m_type)
                    continue
                valid.append({
                    "type": m_type,
                    "text": str(m.get("text", "")).strip(),
                    "importance": _normalize_importance(m.get("importance", 5)),
                    "source": scene.source,         # 记录来源场景
                })
            return valid

        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Parse error on attempt %d/%d: %s", attempt + 1, retries + 1, exc)
            if attempt < retries:
                time.sleep(retry_delay)
        except Exception as exc:
            logger.error("LLM call failed on attempt %d/%d: %s", attempt + 1, retries + 1, exc)
            if attempt < retries:
                time.sleep(retry_delay)
            else:
                raise

    return []


# ─── Main function ────────────────────────────────────────────────────────────

def dialogue_to_memory(
    agent_name: str,
    *,
    data_dir: str | Path = ".",
    output_dir: Optional[str | Path] = None,
    min_importance: int = 1,
    deduplicate: bool = True,
    scene_limit: Optional[int] = None,
) -> list[dict]:
    """
    加载 agent 对话文件，逐场景提取记忆，保存为 JSON 文件。

    Parameters
    ----------
    agent_name    : 与文件名对应的 agent 名，如 "Aria-Hartley"
    data_dir      : JSON 对话文件所在目录（默认当前目录）
    output_dir    : 记忆 JSON 输出目录；None 时使用 Config.data_path
    min_importance: 过滤低于此分数的记忆（1=不过滤）
    deduplicate   : 对相似记忆文本做简单去重
    scene_limit   : 只处理前 N 个场景（调试用）

    Returns
    -------
    全部提取到的记忆列表（同时已写入磁盘）
    """
    # ── 输出路径 ──────────────────────────────────────────────────────────────
    out_dir = Path(output_dir) if output_dir else Path(Config.data_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"memory_{agent_name}.json"

    # ── 加载对话 ──────────────────────────────────────────────────────────────
    loader = DialogueLoader(data_dir)
    scenes = loader.load(agent_name)
    if scene_limit:
        scenes = scenes[:scene_limit]
    logger.info("Loaded %d scenes for agent '%s'", len(scenes), agent_name)

    # ── LLM 客户端 ────────────────────────────────────────────────────────────
    client = _make_client()

    # ── 逐场景提取 ────────────────────────────────────────────────────────────
    all_memories: list[dict] = []
    for i, scene in enumerate(scenes):
        logger.info("Processing scene %d/%d  [%s]", i + 1, len(scenes), scene.source)
        mems = extract_memories_from_scene(client, scene, agent_name)
        all_memories.extend(mems)
        logger.info("  → extracted %d memories", len(mems))

    # ── 过滤 ──────────────────────────────────────────────────────────────────
    if min_importance > 0:
        before = len(all_memories)
        all_memories = [m for m in all_memories if m["importance"] >= min_importance]
        logger.info("Filtered by importance >= %d: %d → %d", min_importance, before, len(all_memories))

    # ── 去重（按 text 字段完全匹配） ──────────────────────────────────────────
    if deduplicate:
        seen: set[str] = set()
        unique = []
        for m in all_memories:
            key = m["type"] + "|" + m["text"]
            if key not in seen:
                seen.add(key)
                unique.append(m)
        logger.info("Deduplicated: %d → %d", len(all_memories), len(unique))
        all_memories = unique

    # ── 按 type 分组整理 ──────────────────────────────────────────────────────
    grouped: dict[str, list[dict]] = {
        "relation": [],
        "preference": [],
        "experience": [],
        "style": [],
    }
    for m in all_memories:
        grouped[m["type"]].append(m)

    output = {
        "agent": agent_name,
        "total": len(all_memories),
        "memories": all_memories,        # 扁平列表（便于向量检索）
        "by_type": grouped,              # 按类型分组（便于按需查询）
    }

    # ── 写入磁盘 ──────────────────────────────────────────────────────────────
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    logger.info("Memory saved → %s  (total %d entries)", out_file, len(all_memories))

    return all_memories


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract memories from agent dialogue JSON")
    parser.add_argument("name", help="Agent name, e.g. Aria-Hartley")
    parser.add_argument("--data-dir", default=".", help="Directory containing dialogue JSON files")
    parser.add_argument("--output-dir", default=None, help="Directory to save memory JSON (default: Config.data_path)")
    parser.add_argument("--min-importance", type=int, default=1, help="Filter memories below this score (1=keep all)")
    parser.add_argument("--no-dedup", action="store_true", help="Disable text deduplication")
    parser.add_argument("--scene-limit", type=int, default=None, help="Only process first N scenes (debug)")
    args = parser.parse_args()

    memories = dialogue_to_memory(
        args.name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_importance=args.min_importance,
        deduplicate=not args.no_dedup,
        scene_limit=args.scene_limit,
    )
    print(f"Done. Extracted {len(memories)} memories.")
