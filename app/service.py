"""
CelebA Face AI & Intelligence System - Shared Application Service Layer
Provides centralized, reusable data loading, database queries, and MCP JSON-RPC
tools shared across Streamlit, Google AI Studio, CLI, and integration testing.

Author: Alok
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_gallery_data() -> List[Dict[str, Any]]:
    """
    Loads curated gallery dataset containing CelebA face images, predicted attributes,
    cluster embeddings, and ground truth identities.

    Returns:
        List of gallery item dictionaries.
    """
    json_path = PROJECT_ROOT / "frontend" / "gallery_data.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def load_benchmark_metrics() -> Optional[pd.DataFrame]:
    """
    Loads the canonical multi-architecture model benchmark evaluation results.

    Returns:
        Pandas DataFrame of benchmark results or None if uninitialized.
    """
    # Priority order: production_benchmark -> model_benchmark_results -> ci_smoke_test_benchmark
    candidates = [
        PROJECT_ROOT / "reports" / "benchmarks" / "production_benchmark.csv",
        PROJECT_ROOT / "reports" / "benchmarks" / "model_benchmark_results.csv",
        PROJECT_ROOT / "reports" / "benchmarks" / "ci_smoke_test_benchmark.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                continue
    return None


def query_mysql(
    sql: str, params: Optional[Tuple[Any, ...]] = None
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Executes a parameterized SQL query via the secure database connection layer.
    Rejects insecure defaults, adheres to strict environment-driven credentials,
    and returns a pandas DataFrame along with any error message.

    Args:
        sql: Parameterized SQL query string (e.g., using %s placeholders).
        params: Tuple of parameters matching the query placeholders.

    Returns:
        Tuple of (Optional[pd.DataFrame], Optional[str] error_message).
    """
    try:
        from src.database.connection import execute_parameterized_query

        rows, err = execute_parameterized_query(sql, params)
        if err:
            return None, err
        return pd.DataFrame(rows) if rows is not None else pd.DataFrame(), None
    except Exception as e:
        return None, str(e)


def execute_mcp_face_identification(
    image_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Executes a standard Model Context Protocol (MCP) JSON-RPC 2.0 tool invocation
    targeting the MySQL biometric database for facial identity cross-referencing.

    Args:
        image_id: Identifier of the face image (e.g. '001553.jpg').

    Returns:
        Tuple of (jsonrpc_request, jsonrpc_response, structured_record).
    """
    clean_id = (
        image_id.replace(".png", ".jpg").replace(".jpeg", ".jpg")
        if image_id.endswith((".png", ".jpeg"))
        else image_id
    )
    req_id = int(time.time() * 1000) % 100000

    mcp_request: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {
            "name": "mysql_identify_face",
            "arguments": {"image_id": clean_id, "database": "celeba_db"},
        },
    }

    t0 = time.perf_counter()
    sql = (
        "SELECT image_id, person_id, person_name, cluster_id, smiling, eyeglasses, male, young "
        "FROM celeba_identities WHERE image_id = %s"
    )
    df, err = query_mysql(sql, (clean_id,))
    t_elapsed = round((time.perf_counter() - t0) * 1000, 2)

    if err is None and df is not None and not df.empty:
        rec = df.iloc[0].to_dict()
        name = rec.get("person_name") or rec.get("detected_person_name") or f"Identity #{rec.get('person_id', 'Unknown')}"
        cluster_id = rec.get("identity_cluster_id", rec.get("cluster_id", -1))
        structured = {
            "source": "Live MySQL Database (celeba_db)",
            "image_id": clean_id,
            "person_id": int(rec.get("person_id", -1)) if str(rec.get("person_id", "")).isdigit() else -1,
            "person_name": name,
            "identity_cluster_id": cluster_id,
            "cluster_id": cluster_id,
            "is_fallback": False,
            "smiling": bool(rec.get("smiling", 0)),
            "eyeglasses": bool(rec.get("eyeglasses", 0)),
            "male": bool(rec.get("male", 0)),
            "young": bool(rec.get("young", 0)),
        }
        text_table = (
            f"| Attribute | Value |\n"
            f"|---|---|\n"
            f"| Image ID | {clean_id} |\n"
            f"| Identity | {name} (ID: {rec.get('person_id', cluster_id)}) |\n"
            f"| Gender | {'Male' if rec.get('male') else 'Female'} |\n"
            f"| Expression | {'Smiling' if rec.get('smiling') else 'Neutral'} |\n"
            f"| Eyeglasses | {'Yes' if rec.get('eyeglasses') else 'No'} |\n"
            f"| Query Latency | {t_elapsed} ms (Live MySQL) |"
        )
        mcp_response: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": text_table}],
                "structured": structured,
                "source": "Live MySQL Database (celeba_db)",
                "execution_ms": t_elapsed,
            },
        }
        return mcp_request, mcp_response, structured

    # Fallback to curated gallery index if MySQL service is offline/containerized
    gallery = load_gallery_data()
    stem = Path(clean_id).stem
    matched = next((g for g in gallery if Path(g["filename"]).stem == stem), None)

    if matched:
        name = matched.get("person_name") or f"Identity #{matched.get('ground_truth_id', 'Unknown')}"
        cid = matched.get("ground_truth_id", matched.get("cluster_id", -1))
        structured_fb = {
            "source": "Curated Ground-Truth Reference (Database Offline - Not Live Model Inference)",
            "image_id": clean_id,
            "person_id": matched.get("ground_truth_id", "N/A"),
            "person_name": name,
            "identity_cluster_id": cid,
            "cluster_id": matched.get("cluster_id", -1),
            "is_fallback": True,
            "smiling": bool(matched.get("smiling", False)),
            "eyeglasses": bool(matched.get("eyeglasses", False)),
            "male": bool(matched.get("male", False)),
            "young": bool(matched.get("young", False)),
        }
        text_table = (
            f"| Attribute | Value |\n"
            f"|---|---|\n"
            f"| Image ID | {clean_id} |\n"
            f"| Identity | {name} |\n"
            f"| Cluster | Cluster #{matched.get('cluster_id', 'N/A')} |\n"
            f"| Gender | {'Male' if matched.get('male') else 'Female'} |\n"
            f"| Expression | {'Smiling' if matched.get('smiling') else 'Neutral'} |\n"
            f"| Eyeglasses | {'Yes' if matched.get('eyeglasses') else 'No'} |\n"
            f"| Data Source | Curated Ground-Truth Reference (Database Offline - Not Live Model Inference) |"
        )
        mcp_response = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": text_table}],
                "structured": structured_fb,
                "source": "Curated Ground-Truth Reference (Database Offline - Not Live Model Inference)",
                "execution_ms": t_elapsed,
            },
        }
        return mcp_request, mcp_response, structured_fb

    # Image not found in either live DB or fallback index
    mcp_error: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
            "code": -32602,
            "message": f"Image ID '{clean_id}' not found in celeba_db or curated gallery index",
            "details": err or "No record matched query criteria",
        },
    }
    return mcp_request, mcp_error, None
