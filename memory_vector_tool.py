from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

from config import Config


MEMORY_TYPES = ("relation", "preference", "experience", "style")


@dataclass
class RetrievedItem:
    memory_type: str
    cluster_id: int
    cluster_rank: int
    centroid_similarity: float
    item_similarity: float
    record: dict[str, Any]

    def to_dict(self, debug: bool) -> dict[str, Any]:
        payload = {
            "type": self.memory_type,
            "cluster_id": self.cluster_id,
            "text": self.record["text"],
            "importance": self.record.get("importance"),
            "source": self.record.get("source"),
            "global_id": self.record.get("global_id"),
            "local_id": self.record.get("local_id"),
            "similarity": round(self.item_similarity, 6),
        }
        if debug:
            payload.update(
                {
                    "cluster_rank": self.cluster_rank,
                    "centroid_similarity": round(self.centroid_similarity, 6),
                }
            )
        return payload


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(text: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", text.strip(), flags=re.UNICODE)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "query"


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vector_norm = np.linalg.norm(vector)
    if vector_norm == 0.0:
        raise ValueError("Query embedding has zero norm.")
    matrix_norms = np.linalg.norm(matrix, axis=1)
    matrix_norms = np.where(matrix_norms == 0.0, 1.0, matrix_norms)
    return (matrix @ vector) / (matrix_norms * vector_norm)


def run_kmeans(
    matrix: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> np.ndarray:
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")
    if len(matrix) < n_clusters:
        raise ValueError("n_clusters cannot exceed sample count.")

    rng = np.random.default_rng(random_state)
    centroids = matrix[rng.choice(len(matrix), size=n_clusters, replace=False)].copy()
    labels = np.zeros(len(matrix), dtype=int)

    for _ in range(max_iter):
        similarities = matrix @ centroids.T
        new_labels = np.argmax(similarities, axis=1)
        new_centroids = centroids.copy()

        for cluster_id in range(n_clusters):
            members = matrix[new_labels == cluster_id]
            if len(members) == 0:
                new_centroids[cluster_id] = matrix[rng.integers(0, len(matrix))]
                continue
            new_centroids[cluster_id] = normalize_rows(members.mean(axis=0, keepdims=True))[0]

        centroid_shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        labels = new_labels
        if centroid_shift <= tol:
            break

    return labels


def ensure_json_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, list):
        return [ensure_json_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: ensure_json_serializable(item) for key, item in value.items()}
    return value


class EmbeddingClient:
    def __init__(
        self,
        *,
        model: str,
        api_url: str,
        api_key: str,
        batch_size: int = 32,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.batch_size = batch_size
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, Config.embedding_dim), dtype=np.float32)

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.session.post(
                self.api_url,
                json={"model": self.model, "input": batch},
                timeout=self.timeout,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                detail = response.text.strip()
                if len(detail) > 300:
                    detail = detail[:300] + "..."
                raise RuntimeError(
                    f"Embedding API request failed with status {response.status_code}: {detail}"
                ) from exc
            payload = response.json()
            data = payload.get("data")
            if not isinstance(data, list):
                raise ValueError(f"Unexpected embedding payload: {payload}")
            ordered = sorted(data, key=lambda item: item["index"])
            all_vectors.extend(item["embedding"] for item in ordered)
        vectors = np.asarray(all_vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D embedding matrix, got shape {vectors.shape}")
        return vectors

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


def load_memory_records(memory_path: Path) -> tuple[str, list[dict[str, Any]]]:
    data = json.loads(memory_path.read_text(encoding="utf-8"))
    agent = data.get("agent") or memory_path.stem
    memories = data.get("memories")
    if not isinstance(memories, list):
        raise ValueError(f"'memories' must be a list in {memory_path}")

    records: list[dict[str, Any]] = []
    for global_id, item in enumerate(memories):
        memory_type = str(item.get("type", "")).strip().lower()
        if memory_type not in MEMORY_TYPES:
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        record = {
            "global_id": global_id,
            "type": memory_type,
            "text": text,
            "importance": item.get("importance"),
            "source": item.get("source"),
        }
        records.append(record)

    return agent, records


def group_records_by_type(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {memory_type: [] for memory_type in MEMORY_TYPES}
    for record in records:
        grouped[record["type"]].append(record)
    return grouped


def build_type_database(
    memory_type: str,
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    requested_k: int,
    random_state: int,
) -> dict[str, Any]:
    if len(records) != len(embeddings):
        raise ValueError(f"Record count and embedding count mismatch for {memory_type}.")
    if not records:
        raise ValueError(f"No records available for type '{memory_type}'.")

    normalized = normalize_rows(embeddings.astype(np.float32))
    cluster_count = min(requested_k, len(records))

    if cluster_count == 1:
        labels = np.zeros(len(records), dtype=int)
    else:
        labels = run_kmeans(
            normalized,
            n_clusters=cluster_count,
            random_state=random_state,
        )

    clusters: list[dict[str, Any]] = []
    db_records: list[dict[str, Any]] = []

    for local_id, (record, vector, cluster_id) in enumerate(zip(records, normalized, labels, strict=True)):
        db_records.append(
            {
                **record,
                "local_id": local_id,
                "cluster_id": int(cluster_id),
                "embedding": vector.tolist(),
            }
        )

    for cluster_id in range(cluster_count):
        member_indices = [idx for idx, label in enumerate(labels) if int(label) == cluster_id]
        member_vectors = normalized[member_indices]
        centroid = normalize_rows(member_vectors.mean(axis=0, keepdims=True))[0]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "size": len(member_indices),
                "member_local_ids": member_indices,
                "centroid": centroid.tolist(),
            }
        )

    return {
        "type": memory_type,
        "item_count": len(db_records),
        "cluster_count": cluster_count,
        "records": db_records,
        "clusters": clusters,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ensure_json_serializable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_index(
    *,
    memory_path: Path,
    output_dir: Path,
    k: int,
    batch_size: int,
    random_state: int,
    embedding_model: str,
    embedding_url: str,
    embedding_api_key: str,
) -> Path:
    agent, records = load_memory_records(memory_path)
    grouped = group_records_by_type(records)

    client = EmbeddingClient(
        model=embedding_model,
        api_url=embedding_url,
        api_key=embedding_api_key,
        batch_size=batch_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    db_files: dict[str, str] = {}
    stats: dict[str, dict[str, int]] = {}

    for memory_type in MEMORY_TYPES:
        type_records = grouped[memory_type]
        if not type_records:
            continue
        texts = [record["text"] for record in type_records]
        embeddings = client.embed_texts(texts)
        db = build_type_database(
            memory_type,
            type_records,
            embeddings,
            requested_k=k,
            random_state=random_state,
        )
        db_path = output_dir / f"db_{memory_type}.json"
        write_json(db_path, db)
        db_files[memory_type] = db_path.name
        stats[memory_type] = {
            "item_count": db["item_count"],
            "cluster_count": db["cluster_count"],
        }

    manifest = {
        "agent": agent,
        "source_memory_path": str(memory_path),
        "created_at": utc_now(),
        "embedding": {
            "model": embedding_model,
            "url": embedding_url,
            "dim": Config.embedding_dim,
            "batch_size": batch_size,
        },
        "clustering": {
            "requested_k": k,
            "random_state": random_state,
        },
        "db_files": db_files,
        "stats": stats,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def load_index(index_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    databases: dict[str, dict[str, Any]] = {}
    for memory_type, filename in manifest.get("db_files", {}).items():
        databases[memory_type] = json.loads((index_dir / filename).read_text(encoding="utf-8"))
    return manifest, databases


def retrieve(
    *,
    index_dir: Path,
    query: str,
    topk: int,
    batch_size: int,
    debug: bool,
    save: bool,
    embedding_model: str | None,
    embedding_url: str | None,
    embedding_api_key: str,
) -> tuple[list[dict[str, Any]], Path | None]:
    manifest, databases = load_index(index_dir)
    client = EmbeddingClient(
        model=embedding_model or manifest["embedding"]["model"],
        api_url=embedding_url or manifest["embedding"]["url"],
        api_key=embedding_api_key,
        batch_size=batch_size,
    )
    query_vector = normalize_rows(client.embed_texts([query]))[0]

    retrieved: list[RetrievedItem] = []
    by_type_results: dict[str, list[dict[str, Any]]] = {}

    for memory_type in MEMORY_TYPES:
        db = databases.get(memory_type)
        if not db:
            continue

        centroids = np.asarray([cluster["centroid"] for cluster in db["clusters"]], dtype=np.float32)
        centroid_scores = cosine_similarity(query_vector, centroids)
        candidate_indices = np.argsort(-centroid_scores)[: min(topk, len(centroid_scores))]

        records = db["records"]
        per_type_results: list[dict[str, Any]] = []
        for rank, cluster_idx in enumerate(candidate_indices, start=1):
            cluster_id = int(db["clusters"][int(cluster_idx)]["cluster_id"])
            cluster_records = [record for record in records if int(record["cluster_id"]) == cluster_id]
            cluster_vectors = np.asarray([record["embedding"] for record in cluster_records], dtype=np.float32)
            item_scores = cosine_similarity(query_vector, cluster_vectors)
            best_idx = int(np.argmax(item_scores))
            best_record = cluster_records[best_idx]
            item = RetrievedItem(
                memory_type=memory_type,
                cluster_id=cluster_id,
                cluster_rank=rank,
                centroid_similarity=float(centroid_scores[int(cluster_idx)]),
                item_similarity=float(item_scores[best_idx]),
                record=best_record,
            )
            retrieved.append(item)
            per_type_results.append(item.to_dict(debug=debug))
        by_type_results[memory_type] = per_type_results

    results = [item.to_dict(debug=debug) for item in retrieved]

    payload = {
        "query": query,
        "created_at": utc_now(),
        "index_dir": str(index_dir),
        "agent": manifest.get("agent"),
        "topk_per_type": topk,
        "result_count": len(results),
        "results": results,
        "by_type": by_type_results,
    }

    output_path: Path | None = None
    if save:
        query_dir = index_dir / "queries"
        output_path = query_dir / f"query_{slugify(query)[:80]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        write_json(output_path, payload)

    return results, output_path


def print_build_summary(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Built index for: {manifest['agent']}")
    print(f"Manifest: {manifest_path}")
    for memory_type in MEMORY_TYPES:
        stats = manifest.get("stats", {}).get(memory_type)
        if not stats:
            continue
        print(
            f"- {memory_type}: items={stats['item_count']}, clusters={stats['cluster_count']}, "
            f"file={manifest['db_files'][memory_type]}"
        )


def print_query_summary(results: list[dict[str, Any]], output_path: Path | None, debug: bool) -> None:
    print(f"Retrieved {len(results)} results.")
    for idx, result in enumerate(results, start=1):
        line = (
            f"{idx}. type={result['type']} cluster={result['cluster_id']} "
            f"similarity={result['similarity']:.6f} text={result['text'][:80]}"
        )
        if debug and "centroid_similarity" in result:
            line += f" centroid={result['centroid_similarity']:.6f} rank={result['cluster_rank']}"
        print(line)
    if output_path is not None:
        print(f"Saved query results to: {output_path}")


def default_output_dir(memory_path: Path) -> Path:
    return Config.data_path / "vector_index" / memory_path.stem


def resolve_embedding_api_key(cli_value: str | None) -> str:
    return (
        cli_value
        or os.getenv("SILICONFLOW_API_KEY")
        or os.getenv("EMBEDDING_API_KEY")
        or Config.embedding_api_key
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clustered vector databases for memory JSON files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build 4 type-specific vector databases.")
    build_parser.add_argument(
        "--memory-path",
        default="data/individual_simulation_data/memory_pool/memory_Alessandra Rossi.json",
        help="Path to the memory JSON file.",
    )
    build_parser.add_argument("--output-dir", default=None, help="Directory to save the vector index.")
    build_parser.add_argument("--k", type=int, default=6, help="Cluster count per type.")
    build_parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    build_parser.add_argument("--random-state", type=int, default=42, help="Random seed for clustering.")
    build_parser.add_argument("--embedding-model", default=Config.embedding_model, help="Embedding model name.")
    build_parser.add_argument("--embedding-url", default=Config.embedding_url, help="Embedding API URL.")
    build_parser.add_argument("--embedding-api-key", default=None, help="Embedding API key. Defaults to env or config.")

    query_parser = subparsers.add_parser("query", help="Retrieve memories with cluster-aware recall.")
    query_parser.add_argument("--index-dir", required=True, help="Directory containing manifest.json and db files.")
    query_parser.add_argument("--query", required=True, help="Query text to retrieve against the vector index.")
    query_parser.add_argument("--topk", type=int, default=2, help="Candidate clusters per type.")
    query_parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    query_parser.add_argument("--debug", action="store_true", help="Include type and cluster debug metadata.")
    query_parser.add_argument("--no-save", action="store_true", help="Do not write query results to disk.")
    query_parser.add_argument("--embedding-model", default=None, help="Override the embedding model used for query encoding.")
    query_parser.add_argument("--embedding-url", default=None, help="Override the embedding API URL.")
    query_parser.add_argument("--embedding-api-key", default=None, help="Embedding API key. Defaults to env or config.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        memory_path = Path(args.memory_path)
        output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(memory_path)
        manifest_path = build_index(
            memory_path=memory_path,
            output_dir=output_dir,
            k=args.k,
            batch_size=args.batch_size,
            random_state=args.random_state,
            embedding_model=args.embedding_model,
            embedding_url=args.embedding_url,
            embedding_api_key=resolve_embedding_api_key(args.embedding_api_key),
        )
        print_build_summary(manifest_path)
        return

    if args.command == "query":
        results, output_path = retrieve(
            index_dir=Path(args.index_dir),
            query=args.query,
            topk=args.topk,
            batch_size=args.batch_size,
            debug=args.debug,
            save=not args.no_save,
            embedding_model=args.embedding_model,
            embedding_url=args.embedding_url,
            embedding_api_key=resolve_embedding_api_key(args.embedding_api_key),
        )
        print_query_summary(results, output_path, args.debug)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
