from pathlib import Path
import argparse
import logging

from memory import dialogue_to_memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, help="1-based inclusive start index")
    parser.add_argument("--end", type=int, default=None, help="1-based inclusive end index")
    parser.add_argument("--force", action="store_true", help="Rebuild files even if they already exist")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.WARNING)

    data_dir = "data/individual_simulation_data/dialogue"
    out_dir = Path("data/individual_simulation_data/memory_pool")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_dir)
    names = sorted(
        p.stem.removeprefix("generated_agent_dialogue_")
        for p in data_path.glob("generated_agent_dialogue_*.json")
    )
    total = len(names)
    end = args.end if args.end is not None else total
    subset = names[args.start - 1 : end]

    for idx, name in enumerate(subset, args.start):
        out_file = out_dir / f"memory_{name}.json"
        if out_file.exists() and not args.force:
            print(f"[{idx}/{total}] SKIP {name}", flush=True)
            continue

        print(f"[{idx}/{total}] START {name}", flush=True)
        memories = dialogue_to_memory(
            name,
            data_dir=data_dir,
            output_dir=out_dir,
            min_importance=1,
            deduplicate=True,
            scene_limit=None,
        )
        print(f"[{idx}/{total}] OK {name} -> {len(memories)} memories", flush=True)


if __name__ == "__main__":
    main()
