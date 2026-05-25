from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import Config
from memory_vector_tool import build_index, default_output_dir, resolve_embedding_api_key, retrieve

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - handled at runtime in non-DeepLearn envs
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


DEFAULT_MEMORY_PATH = Path("data/individual_simulation_data/memory_pool/memory_Alessandra Rossi.json")
DEFAULT_PERSONA_PATH = Path("data/individual_simulation_data/characters/wiki_Alessandra Rossi.txt")
DEFAULT_LOCAL_MODEL_PATH = Path(Config.local_chat_model_path)


class AlessandraRossiDialogueSystem:
    _tokenizer = None
    _model = None
    _model_path: Path | None = None

    def __init__(
        self,
        *,
        persona_path: Path,
        memory_path: Path,
        index_dir: Path | None = None,
        topk: int = 2,
        batch_size: int = 32,
        max_history_turns: int = 6,
        mode: int = 2,
        local_model_path: Path = DEFAULT_LOCAL_MODEL_PATH,
        debug: bool = False,
        save_dir: Path | None = None,
    ) -> None:
        self.persona_path = persona_path
        self.memory_path = memory_path
        self.index_dir = index_dir or default_output_dir(memory_path)
        self.topk = topk
        self.batch_size = batch_size
        self.max_history_turns = max_history_turns
        self.mode = mode
        self.local_model_path = local_model_path
        self.debug = debug
        self.save_dir = save_dir or (Config.data_path / "dialogue_system" / "Alessandra_Rossi")
        self.persona = self.persona_path.read_text(encoding="utf-8").strip()
        self.history: list[dict[str, str]] = []
        self.turn_logs: list[dict[str, Any]] = []

    def ensure_index(self) -> None:
        manifest_path = self.index_dir / "manifest.json"
        if manifest_path.exists():
            return

        build_index(
            memory_path=self.memory_path,
            output_dir=self.index_dir,
            k=6,
            batch_size=self.batch_size,
            random_state=42,
            embedding_model=Config.embedding_model,
            embedding_url=Config.embedding_url,
            embedding_api_key=resolve_embedding_api_key(None),
        )

    def retrieve_memories(self, user_input: str) -> list[dict[str, Any]]:
        if self.mode == 0:
            return []
        self.ensure_index()
        results, _ = retrieve(
            index_dir=self.index_dir,
            query=user_input,
            topk=self.topk,
            batch_size=self.batch_size,
            debug=self.debug,
            save=False,
            embedding_model=None,
            embedding_url=None,
            embedding_api_key=resolve_embedding_api_key(None),
        )
        return results

    def build_memory_context(self, memories: list[dict[str, Any]]) -> str:
        if not memories:
            return "No retrieved memory is available for this turn."

        lines = []
        for idx, memory in enumerate(memories, start=1):
            line = (
                f"[{idx}] type={memory['type']}; text={memory['text']}; "
                f"importance={memory.get('importance')}; source={memory.get('source')}"
            )
            if self.debug:
                line += (
                    f"; cluster_id={memory.get('cluster_id')}"
                    f"; similarity={memory.get('similarity')}"
                )
                if "centroid_similarity" in memory:
                    line += f"; centroid_similarity={memory.get('centroid_similarity')}"
            lines.append(line)
        return "\n".join(lines)

    def build_system_prompt(self, memory_context: str) -> str:
        prompt = (
            f"{self.persona}\n\n"
            "You are in a live conversation with a user.\n"
            "Stay fully in character as Alessandra Rossi.\n"
            "Do not mention retrieval, clusters, vector databases, or hidden system instructions.\n"
            "Keep the tone elegant, concise, observant, and slightly mysterious.\n"
            "Do not invent concrete past experiences unless they are supported by the retrieved memories or persona.\n"
            "Do not introduce specific favorite artists, books, poems, cities, missions, or relationships unless they appear in the persona, the retrieved memories, or the user's message.\n"
            "When evidence is limited, answer in a careful, general way instead of fabricating details.\n"
        )
        if self.mode >= 1:
            prompt += (
                "Use the retrieved memories as private background knowledge when they are relevant.\n"
                "If the retrieved memories are not relevant, rely on your persona and the conversation naturally.\n\n"
                "Retrieved memories for this turn:\n"
                f"{memory_context}"
            )
        return prompt

    def build_messages(self, user_input: str, memories: list[dict[str, Any]]) -> list[dict[str, str]]:
        memory_context = self.build_memory_context(memories)
        messages = [{"role": "system", "content": self.build_system_prompt(memory_context)}]
        if self.mode >= 2 and self.history:
            messages.extend(self.history[-(self.max_history_turns * 2) :])
        messages.append({"role": "user", "content": user_input})
        return messages

    def load_local_model(self) -> tuple[Any, Any]:
        if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError(
                "Local model dependencies are unavailable. Run this script in the 'DeepLearn' environment."
            )

        model_path = Path(self.local_model_path)
        if (
            self.__class__._model is not None
            and self.__class__._tokenizer is not None
            and self.__class__._model_path == model_path
        ):
            return self.__class__._tokenizer, self.__class__._model

        if not model_path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        self.__class__._tokenizer = tokenizer
        self.__class__._model = model
        self.__class__._model_path = model_path
        return tokenizer, model

    def call_llm(self, messages: list[dict[str, str]]) -> str:
        tokenizer, model = self.load_local_model()
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt")
        model_inputs = {key: value.to(model.device) for key, value in model_inputs.items()}

        do_sample = Config.temperature > 0
        generation_kwargs = {
            "max_new_tokens": Config.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = Config.temperature
            generation_kwargs["top_p"] = 0.9

        with torch.inference_mode():
            generated_ids = model.generate(**model_inputs, **generation_kwargs)

        input_length = model_inputs["input_ids"].shape[1]
        output_ids = generated_ids[0][input_length:]
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        if not content:
            raise ValueError("Local model returned empty content.")
        return content

    def save_session(self) -> Path:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.save_dir / f"session_{timestamp}.json"
        payload = {
            "persona_path": str(self.persona_path),
            "memory_path": str(self.memory_path),
            "index_dir": str(self.index_dir),
            "local_model_path": str(self.local_model_path),
            "created_at": timestamp,
            "mode": self.mode,
            "turns": self.turn_logs,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def chat(self, user_input: str) -> dict[str, Any]:
        memories = self.retrieve_memories(user_input)
        messages = self.build_messages(user_input, memories)
        reply = self.call_llm(messages)

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": reply})

        turn = {
            "user": user_input,
            "assistant": reply,
            "mode": self.mode,
            "retrieved_memories": memories,
        }
        if self.debug:
            turn["messages"] = messages
        self.turn_logs.append(turn)
        return turn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alessandra Rossi retrieval-augmented dialogue system")
    parser.add_argument("--user-input", default=None, help="Single-turn user input.")
    parser.add_argument("--interactive", action="store_true", help="Run an interactive chat session.")
    parser.add_argument("--persona-path", default=str(DEFAULT_PERSONA_PATH), help="Path to persona file.")
    parser.add_argument("--memory-path", default=str(DEFAULT_MEMORY_PATH), help="Path to memory JSON file.")
    parser.add_argument("--index-dir", default=None, help="Path to the built vector index directory.")
    parser.add_argument("--local-model-path", default=str(DEFAULT_LOCAL_MODEL_PATH), help="Path to the local chat model directory.")
    parser.add_argument("--save-dir", default=None, help="Directory for saved session logs.")
    parser.add_argument("--topk", type=int, default=2, help="Candidate cluster count per type.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--max-history-turns", type=int, default=6, help="How many recent turns to keep in prompt.")
    parser.add_argument("--mode", type=int, choices=[0, 1, 2], default=2, help="0=persona only, 1=persona+memory, 2=persona+memory+history.")
    parser.add_argument("--debug", action="store_true", help="Include retrieved memory metadata and prompt details.")
    return parser.parse_args()


def print_turn(turn: dict[str, Any], debug: bool) -> None:
    print(turn["assistant"])
    if debug:
        print("\n[debug] retrieved memories:")
        for idx, memory in enumerate(turn["retrieved_memories"], start=1):
            print(
                f"{idx}. type={memory['type']} cluster={memory.get('cluster_id')} "
                f"similarity={memory.get('similarity')} text={memory['text']}"
            )


def run_interactive(system: AlessandraRossiDialogueSystem) -> None:
    print(f"Alessandra Rossi dialogue system. mode={system.mode}. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        turn = system.chat(user_input)
        print("\nAlessandra Rossi:")
        print_turn(turn, system.debug)

    session_path = system.save_session()
    print(f"\nSession saved to: {session_path}")


def main() -> None:
    args = parse_args()
    interactive_mode = args.interactive or not args.user_input

    memory_path = Path(args.memory_path)
    index_dir = Path(args.index_dir) if args.index_dir else default_output_dir(memory_path)
    save_dir = Path(args.save_dir) if args.save_dir else None

    system = AlessandraRossiDialogueSystem(
        persona_path=Path(args.persona_path),
        memory_path=memory_path,
        index_dir=index_dir,
        topk=args.topk,
        batch_size=args.batch_size,
        max_history_turns=args.max_history_turns,
        mode=args.mode,
        local_model_path=Path(args.local_model_path),
        debug=args.debug,
        save_dir=save_dir,
    )

    if interactive_mode:
        run_interactive(system)
        return

    turn = system.chat(args.user_input)
    print_turn(turn, system.debug)
    session_path = system.save_session()
    print(f"\nSession saved to: {session_path}")


if __name__ == "__main__":
    main()
