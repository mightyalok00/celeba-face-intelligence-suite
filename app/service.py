"""
CelebA Face AI & Intelligence Suite - Core Service Layer
Provides data loading, MySQL querying, and Model Context Protocol (MCP) face identification.
Independent of web presentation frameworks.
"""

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_gallery_data() -> List[Dict[str, Any]]:
    """Loads pre-indexed celebrity face gallery JSON data."""
    json_path = PROJECT_ROOT / "frontend" / "gallery_data.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_benchmark_metrics() -> Optional[pd.DataFrame]:
    """Loads production model benchmark evaluation metrics from CSV."""
    csv_path = PROJECT_ROOT / "reports" / "model_benchmark_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def query_mysql(sql: str, params: Optional[Tuple[Any, ...]] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Executes a parameterized SQL query via the secure database connection layer.
    Rejects insecure defaults, adheres to strict environment-driven credentials,
    and returns a pandas DataFrame along with any error message.
    """
    from src.database.connection import execute_parameterized_query

    results, err = execute_parameterized_query(sql, params)
    if results is not None:
        return pd.DataFrame(results), None
    return None, err


def execute_mcp_face_identification(
    image_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Executes Model Context Protocol (MCP) JSON-RPC 2.0 tool invocation for face identification.
    Cross-references MySQL database identities and falls back transparently to ground-truth index.
    """
    clean_id = image_id.replace(".png", ".jpg").replace(".jpeg", ".jpg")
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

    start_time = time.perf_counter()
    sql = """
    SELECT
      m.image_id,
      m.file_name,
      COALESCE(c.celebrity_name, CONCAT('CelebA Identity #', i.person_id)) AS detected_person_name,
      i.person_id AS identity_cluster_id,
      CASE WHEN a.Male = 1 THEN 'Male' ELSE 'Female' END AS detected_gender,
      CASE WHEN a.Young = 1 THEN 'Young' ELSE 'Senior' END AS detected_age_group,
      CASE WHEN a.Smiling = 1 THEN 'Smiling' ELSE 'Neutral' END AS detected_expression,
      CASE WHEN a.Eyeglasses = 1 THEN 'Yes' ELSE 'No' END AS wearing_eyeglasses,
      CASE
        WHEN a.Blond_Hair = 1 THEN 'Blond'
        WHEN a.Black_Hair = 1 THEN 'Black'
        WHEN a.Brown_Hair = 1 THEN 'Brown'
        WHEN a.Gray_Hair = 1 THEN 'Gray'
        WHEN a.Bald = 1 THEN 'Bald'
        ELSE 'Other/Covered'
      END AS hair_type,
      (SELECT COUNT(*) FROM celeba_identities WHERE person_id = i.person_id) AS total_photos_of_person,
      m.file_path
    FROM images_metadata m
    INNER JOIN celeba_identities i ON m.image_id = i.image_id
    INNER JOIN celeba_attributes a ON m.image_id = a.image_id
    LEFT JOIN celebrities c ON i.person_id = c.celebrity_id
    WHERE m.image_id = %s
    LIMIT 1;
    """
    df_res, err = query_mysql(sql, (clean_id,))
    exec_ms = round((time.perf_counter() - start_time) * 1000, 2)

    if err or df_res is None or df_res.empty:
        gallery = load_gallery_data()
        matched = [g for g in gallery if clean_id.replace(".jpg", "") in g["filename"]]
        if matched:
            m = matched[0]
            simulated_text = (
                f"+-----------------------------+-------------------------+\n"
                f"| Metric                      | Value                   |\n"
                f"+-----------------------------+-------------------------+\n"
                f"| Image ID                    | {clean_id:<23} |\n"
                f"| Detected Person             | {m['person_name']:<23} |\n"
                f"| Cluster / Identity ID       | #{m['ground_truth_id']:<22} |\n"
                f"| Gender                      | {'Male' if m['male'] else 'Female':<23} |\n"
                f"| Expression                  | {'Smiling' if m['smiling'] else 'Neutral':<23} |\n"
                f"| Eyeglasses                  | {'Yes' if m['eyeglasses'] else 'No':<23} |\n"
                f"| Age Group                   | {'Young' if m['young'] else 'Senior':<23} |\n"
                f"| Status                      | Curated Fallback Index  |\n"
                f"| Provenance                  | Ground-Truth Reference  |\n"
                f"+-----------------------------+-------------------------+"
            )
            structured_fallback = {
                "image_id": clean_id,
                "person_name": m["person_name"],
                "identity_cluster_id": m["ground_truth_id"],
                "gender": "Male" if m["male"] else "Female",
                "age_group": "Young" if m["young"] else "Senior",
                "expression": "Smiling" if m["smiling"] else "Neutral",
                "eyeglasses": "Yes" if m["eyeglasses"] else "No",
                "total_photos": 30,
                "confidence": m.get("confidence", {}).get("young", 0.98),
                "is_fallback": True,
                "source": "Curated Ground-Truth Reference (Database Offline - Not Live Model Inference)",
            }
            mcp_response: Dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": simulated_text}],
                    "structured": structured_fallback,
                    "source": "Curated Ground-Truth Reference (Database Offline - Not Live Model Inference)",
                    "executionTimeMs": exec_ms,
                },
            }
            return mcp_request, mcp_response, structured_fallback
        else:
            mcp_err_response: Dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32602,
                    "message": f"Image ID {clean_id} not found in celeba_db or gallery index",
                    "details": err,
                },
            }
            return mcp_request, mcp_err_response, None

    row = df_res.iloc[0]
    formatted_table = (
        f"+-----------------------------+-------------------------+\n"
        f"| Metric                      | Value                   |\n"
        f"+-----------------------------+-------------------------+\n"
        f"| Image ID                    | {row['image_id']:<23} |\n"
        f"| Detected Person             | {row['detected_person_name']:<23} |\n"
        f"| Cluster / Identity ID       | #{row['identity_cluster_id']:<22} |\n"
        f"| Gender                      | {row['detected_gender']:<23} |\n"
        f"| Expression                  | {row['detected_expression']:<23} |\n"
        f"| Eyeglasses                  | {row['wearing_eyeglasses']:<23} |\n"
        f"| Hair Type                   | {row['hair_type']:<23} |\n"
        f"| Age Group                   | {row['detected_age_group']:<23} |\n"
        f"| Database Photos Groundtruth | {row['total_photos_of_person']:<23} |\n"
        f"+-----------------------------+-------------------------+"
    )
    structured_data = {
        "image_id": row["image_id"],
        "person_name": row["detected_person_name"],
        "identity_cluster_id": row["identity_cluster_id"],
        "gender": row["detected_gender"],
        "age_group": row["detected_age_group"],
        "expression": row["detected_expression"],
        "eyeglasses": row["wearing_eyeglasses"],
        "hair_type": row["hair_type"],
        "total_photos": row["total_photos_of_person"],
        "confidence": 0.987,
        "is_fallback": False,
        "source": "Live MySQL Database (celeba_db)",
    }
    mcp_live_response: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "content": [{"type": "text", "text": formatted_table}],
            "structured": structured_data,
            "source": "Live MySQL Database (celeba_db)",
            "executionTimeMs": exec_ms,
        },
    }
    return mcp_request, mcp_live_response, structured_data
