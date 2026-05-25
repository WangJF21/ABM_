"""
build_dpo_dataset.py

从 generated_agent_dialogue_{name}.json 文件构建 DPO 微调数据集。

数据结构：
  每条样本 = {
      "system"  : <角色 persona>,
      "prompt"  : <最近 2 轮对话（chosen 发言前的上下文）>,
      "chosen"  : <{name} 本人的 speaking 回复>,
      "rejected": <从其他场景 / 其他角色随机采样的 speaking 回复>
  }

输出格式兼容 LlamaFactory / TRL DPO trainer（sharegpt-style messages list）：
  {
    "system"  : "...",
    "chosen"  : [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
    "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
  }
  其中 prompt（上文）已合并进 chosen/rejected 的 user 侧。
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

from dialogue_loader import DialogueLoader, DialogueScene, DialogueTurn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── Persona loader ───────────────────────────────────────────────────────────

def _load_persona(name: str, data_dir: str | Path) -> str:
    """
    读取 data/individual_simulation_data/characters/wiki_{name}.txt
    找不到时返回空字符串并记 warning。
    """
    base = Path(data_dir)
    # 优先在 data_dir 下按规范路径查找
    candidates = [
        base.parent / "characters" / f"wiki_{name}.txt",
        base / "data" / "individual_simulation_data" / "characters" / f"wiki_{name}.txt",
        base / f"wiki_{name}.txt",
    ]
    for p in candidates:
        if p.exists():
            logger.info("Persona loaded from: %s", p)
            return p.read_text(encoding="utf-8").strip()
    logger.warning("Persona file not found for '%s', tried: %s", name, candidates)
    return ""


# ─── Context builder ──────────────────────────────────────────────────────────

def _format_turn(turn: DialogueTurn) -> str:
    """单轮格式化：[角色] 内容"""
    return f"[{turn.role}]: {turn.content}"


def _normalize_role(name: str) -> str:
    return name.replace("-", " ").strip().lower()


def _normalize_topic(topic: str) -> str:
    return topic.strip().lower()


def _build_prompt_context(scene: DialogueScene, target_idx: int) -> str:
    """
    返回 target_idx 位置 chosen 发言前最多 2 轮 speaking 对话的文本。
    只取 speaking（排除 thinking），以免泄露内心独白。
    """
    # 取 target_idx 之前所有 speaking 轮
    prior_speaking = [
        t for t in scene.dialogue[:target_idx]
        if t.is_speaking
    ]
    # 取最后 2 轮
    context_turns = prior_speaking[-2:] if len(prior_speaking) >= 2 else prior_speaking
    return "\n".join(_format_turn(t) for t in context_turns)


# ─── Rejected sampler ─────────────────────────────────────────────────────────

def _collect_candidate_turns(
    scenes: list[DialogueScene],
    agent_name: str,
    *,
    min_turn_len: int = 1,
) -> list[dict]:
    """
    收集 speaking 轮次及其场景元信息，用于构建更难的 rejected 候选池。
    """
    agent_normalized = _normalize_role(agent_name)
    candidates: list[dict] = []
    for scene in scenes:
        for turn in scene.spoken_turns():
            text = turn.content.strip()
            if len(text) < min_turn_len:
                continue
            candidates.append({
                "text": text,
                "role": turn.role,
                "is_target_agent": turn.role.lower() == agent_normalized,
                "source": scene.source,
                "emotion": scene.emotion,
                "location": scene.location,
                "topic": [_normalize_topic(t) for t in scene.topic],
            })
    return candidates


def _shares_topic(candidate: dict, scene: DialogueScene) -> bool:
    candidate_topics = set(candidate["topic"])
    scene_topics = {_normalize_topic(t) for t in scene.topic}
    return bool(candidate_topics & scene_topics)


def _tier_rejected_candidate(candidate: dict, scene: DialogueScene, chosen_text: str) -> Optional[int]:
    """
    分层定义 rejected 难度：
    1. 同角色、不同场景、同情绪且同话题
    2. 同角色、不同场景、同地点
    3. 同角色、不同场景、同情绪或同话题
    4. 同角色、不同场景
    5. 其他角色、同场景
    6. 其他角色、同情绪且同话题
    7. 其他角色、同情绪或同地点或同话题
    8. 其他角色、任意场景
    """
    if candidate["text"] == chosen_text:
        return None

    same_source = candidate["source"] == scene.source
    same_emotion = candidate["emotion"].lower() == scene.emotion.lower()
    same_location = candidate["location"].lower() == scene.location.lower()
    same_topic = _shares_topic(candidate, scene)

    if candidate["is_target_agent"]:
        if same_source:
            return None
        if same_emotion and same_topic:
            return 1
        if same_location:
            return 2
        if same_emotion or same_topic:
            return 3
        return 4

    if same_source:
        return 5
    if same_emotion and same_topic:
        return 6
    if same_emotion or same_location or same_topic:
        return 7
    return 8


def _sample_rejected_text(
    rng: random.Random,
    candidates: list[dict],
    scene: DialogueScene,
    chosen_text: str,
) -> str:
    """
    优先采样更难的负样本；同层内偏好长度相近的候选，避免过于容易区分。
    """
    ranked: dict[int, list[dict]] = {}
    chosen_len = len(chosen_text)

    for candidate in candidates:
        tier = _tier_rejected_candidate(candidate, scene, chosen_text)
        if tier is None:
            continue
        ranked.setdefault(tier, []).append(candidate)

    if not ranked:
        raise ValueError(f"No rejected candidates available for scene {scene.source!r}")

    best_tier = min(ranked)
    tier_candidates = ranked[best_tier]
    tier_candidates.sort(key=lambda item: abs(len(item["text"]) - chosen_len))
    shortlist = tier_candidates[: min(5, len(tier_candidates))]
    return rng.choice(shortlist)["text"]


# ─── Core builder ────────────────────────────────────────────────────────────

def build_dpo_dataset(
    agent_name: str,
    *,
    data_dir: str | Path = ".",
    output_dir: Optional[str | Path] = None,
    output_filename: Optional[str] = None,
    seed: int = 42,
    min_chosen_len: int = 10,
    format: str = "sharegpt",   # "sharegpt" | "raw"
) -> list[dict]:
    """
    为指定 agent 构建 DPO 微调数据集。

    Parameters
    ----------
    agent_name      : 对应文件名中的名称，如 "Aria-Hartley"
    data_dir        : 对话 JSON 文件所在目录
    output_dir      : 输出目录（默认 data/individual_simulation_data）
    output_filename : 输出文件名（默认 dpo_{agent_name}.json）
    seed            : 随机种子（rejected 采样用）
    min_chosen_len  : chosen 文本最短字符数，过滤过短回复
    format          : "sharegpt" 兼容 LlamaFactory/TRL；"raw" 保留原始字段

    Returns
    -------
    list[dict]  同时写入磁盘
    """
    rng = random.Random(seed)
    data_dir = Path(data_dir)

    # ── 加载 ──────────────────────────────────────────────────────────────────
    loader = DialogueLoader(data_dir)
    all_scenes = loader.load(agent_name)
    logger.info("Loaded %d scenes for agent '%s'", len(all_scenes), agent_name)

    persona = _load_persona(agent_name, data_dir)

    # 构建全局 rejected 候选池（优先使用同角色异场景的 harder negatives）
    rejected_candidates = _collect_candidate_turns(
        all_scenes,
        agent_name,
        min_turn_len=min_chosen_len,
    )
    if not rejected_candidates:
        raise ValueError("No rejected candidates found. Check agent_name and data_dir.")
    logger.info("Rejected candidate pool size: %d", len(rejected_candidates))

    # ── 遍历每个场景，找 chosen ───────────────────────────────────────────────
    dataset: list[dict] = []

    for scene in all_scenes:
        dialogue = scene.dialogue

        for idx, turn in enumerate(dialogue):
            # chosen 条件：{name} 本人 + speaking + 内容足够长
            agent_normalized = agent_name.replace("-", " ").lower()
            if turn.role.lower() != agent_normalized:
                continue
            if not turn.is_speaking:
                continue
            if len(turn.content.strip()) < min_chosen_len:
                continue

            chosen_text = turn.content.strip()

            # 上下文：chosen 前最多 2 轮 speaking
            context_str = _build_prompt_context(scene, idx)

            # rejected：优先采样同角色异场景、元信息更接近的 harder negative
            rejected_text = _sample_rejected_text(
                rng,
                rejected_candidates,
                scene,
                chosen_text,
            )

            # ── 组装样本 ─────────────────────────────────────────────────────
            if format == "sharegpt":
                # LlamaFactory DPO sharegpt 格式
                # prompt 部分嵌入 user message，assistant 分别填 chosen/rejected
                user_content = context_str if context_str else "(对话开始)"
                sample = {
                    "system": persona,
                    "chosen": [
                        {"role": "user",      "content": user_content},
                        {"role": "assistant", "content": chosen_text},
                    ],
                    "rejected": [
                        {"role": "user",      "content": user_content},
                        {"role": "assistant", "content": rejected_text},
                    ],
                    "_meta": {
                        "source": scene.source,
                        "emotion": scene.emotion,
                        "location": scene.location,
                        "chosen_role": turn.role,
                        "turn_idx": idx,
                    },
                }
            else:
                # raw 格式，保留完整字段
                sample = {
                    "system":   persona,
                    "prompt":   context_str,
                    "chosen":   chosen_text,
                    "rejected": rejected_text,
                    "_meta": {
                        "source":      scene.source,
                        "emotion":     scene.emotion,
                        "location":    scene.location,
                        "chosen_role": turn.role,
                        "turn_idx":    idx,
                    },
                }

            dataset.append(sample)

    logger.info("Built %d DPO samples", len(dataset))

    # ── 写入 ─────────────────────────────────────────────────────────────────
    out_dir = Path(output_dir) if output_dir else data_dir / "data" / "individual_simulation_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = output_filename or f"dpo_{agent_name}.json"
    out_file = out_dir / fname

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, indent=2)

    logger.info("DPO dataset saved → %s  (%d samples)", out_file, len(dataset))
    return dataset


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build DPO dataset from agent dialogue JSON")
    parser.add_argument("name",             help="Agent name, e.g. Aria-Hartley")
    parser.add_argument("--data-dir",       default=".",    help="Directory with dialogue JSON files")
    parser.add_argument("--output-dir",     default=None,   help="Output directory (default: data_dir/data/individual_simulation_data)")
    parser.add_argument("--output-filename",default=None,   help="Output filename (default: dpo_{name}.json)")
    parser.add_argument("--seed",           type=int, default=42, help="Random seed")
    parser.add_argument("--min-chosen-len", type=int, default=10, help="Min chars for chosen text")
    parser.add_argument("--format",         choices=["sharegpt","raw"], default="sharegpt",
                        help="sharegpt: LlamaFactory/TRL compatible; raw: plain prompt/chosen/rejected")
    args = parser.parse_args()

    samples = build_dpo_dataset(
        args.name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        seed=args.seed,
        min_chosen_len=args.min_chosen_len,
        format=args.format,
    )
    print(f"Done. {len(samples)} DPO samples written.")
