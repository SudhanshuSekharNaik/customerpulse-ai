"""Dataset Profiler & Secure CSV Ingestion."""

import os
import io
import csv
import re
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np


class DatasetProfiler:
    """Securely inspects, profiles, and sanitizes untrusted CSV files."""

    MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB default limit
    MAX_ROWS_TO_PROFILE = 100000

    @staticmethod
    def sanitize_cell_value(val: Any) -> Any:
        """Prevent CSV formula injection by stripping leading dangerous prefixes (=, +, -, @, tab, cr)."""
        if isinstance(val, str):
            # If string starts with formula trigger characters, prepend a single quote or strip
            if val.startswith(("=", "+", "-", "@", "\t", "\r")):
                return val.lstrip("=+-@\t\r")
        return val

    @classmethod
    def detect_encoding_and_delimiter(cls, file_bytes: bytes) -> Tuple[str, str]:
        """Detect encoding (utf-8, latin-1) and delimiter (comma, semicolon, tab, pipe)."""
        # 1. Encoding detection
        encoding = "utf-8"
        try:
            sample_text = file_bytes[:8192].decode("utf-8")
        except UnicodeDecodeError:
            encoding = "latin-1"
            sample_text = file_bytes[:8192].decode("latin-1", errors="ignore")

        # 2. Delimiter sniffing
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample_text, delimiters=[",", ";", "\t", "|"])
            delimiter = dialect.delimiter
        except Exception:
            # Fallback delimiter heuristics
            comma_cnt = sample_text.count(",")
            semi_cnt = sample_text.count(";")
            tab_cnt = sample_text.count("\t")
            if semi_cnt > comma_cnt and semi_cnt > tab_cnt:
                delimiter = ";"
            elif tab_cnt > comma_cnt and tab_cnt > semi_cnt:
                delimiter = "\t"
            else:
                delimiter = ","

        return encoding, delimiter

    @classmethod
    def load_and_profile_csv(
        cls,
        file_path_or_bytes: Any,
        filename: str = "uploaded_data.csv",
    ) -> Dict[str, Any]:
        """Read CSV, sanitize rows, and build detailed dataset profile."""
        if isinstance(file_path_or_bytes, bytes):
            file_bytes = file_path_or_bytes
            file_size = len(file_bytes)
            encoding, delimiter = cls.detect_encoding_and_delimiter(file_bytes)
            text_stream = io.StringIO(file_bytes.decode(encoding, errors="replace"))
            df = pd.read_csv(text_stream, sep=delimiter, nrows=cls.MAX_ROWS_TO_PROFILE)
        else:
            file_path = str(file_path_or_bytes)
            file_size = os.path.getsize(file_path)
            with open(file_path, "rb") as f:
                encoding, delimiter = cls.detect_encoding_and_delimiter(f.read(8192))
            df = pd.read_csv(file_path, sep=delimiter, encoding=encoding, nrows=cls.MAX_ROWS_TO_PROFILE)

        if df.empty or len(df.columns) == 0:
            raise ValueError("Uploaded CSV contains no valid columns or rows.")

        # Sanitize dataframe values (prevent formula injection)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(cls.sanitize_cell_value)

        # Profile basic metrics
        row_count = len(df)
        col_count = len(df.columns)
        total_cells = row_count * col_count
        null_cells = int(df.isnull().sum().sum())
        missing_rate_pct = round((null_cells / max(1, total_cells)) * 100, 2)
        duplicate_rows_cnt = int(df.duplicated().sum())
        duplicate_rate_pct = round((duplicate_rows_cnt / max(1, row_count)) * 100, 2)

        column_profiles = []
        num_cols = []
        cat_cols = []
        dt_cols = []
        id_cols = []

        for col in df.columns:
            series = df[col]
            dtype_str = str(series.dtype)
            num_unique = int(series.nunique(dropna=True))
            null_count = int(series.isnull().sum())
            null_pct = round((null_count / max(1, row_count)) * 100, 2)
            sample_vals = [str(v) for v in series.dropna().head(5).tolist()]

            # Determine column category
            is_datetime = False
            if pd.api.types.is_datetime64_any_dtype(series):
                is_datetime = True
            elif series.dtype == object and num_unique > 5:
                # Test if string column looks like timestamp
                try:
                    sample_dt = pd.to_datetime(series.dropna().head(20), errors="coerce")
                    if sample_dt.notnull().sum() >= 15:
                        is_datetime = True
                except Exception:
                    pass

            is_numeric = pd.api.types.is_numeric_dtype(series) and not is_datetime
            is_id_like = False
            # ID-like if high cardinality (>80% unique) or name contains id/key
            if (num_unique / max(1, row_count) > 0.70 and num_unique > 10) or re.search(r"\b(id|key|uuid|guid|code|pk)\b", col, re.I):
                is_id_like = True

            if is_datetime:
                dt_cols.append(col)
                col_type = "datetime"
            elif is_id_like:
                id_cols.append(col)
                col_type = "id_like"
            elif is_numeric:
                num_cols.append(col)
                col_type = "numerical"
            else:
                cat_cols.append(col)
                col_type = "categorical"

            # Numerical stats if applicable
            min_val = float(series.min()) if is_numeric and not series.dropna().empty else None
            max_val = float(series.max()) if is_numeric and not series.dropna().empty else None
            mean_val = float(series.mean()) if is_numeric and not series.dropna().empty else None

            column_profiles.append({
                "column_name": col,
                "detected_type": col_type,
                "raw_dtype": dtype_str,
                "unique_values": num_unique,
                "cardinality_ratio": round(num_unique / max(1, row_count), 4),
                "null_count": null_count,
                "null_percentage": null_pct,
                "sample_values": sample_vals,
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
            })

        # Profile summary object
        profile = {
            "filename": filename,
            "file_size_bytes": file_size,
            "encoding": encoding,
            "delimiter": delimiter,
            "total_rows": row_count,
            "total_columns": col_count,
            "missing_rate_pct": missing_rate_pct,
            "duplicate_rows_count": duplicate_rows_cnt,
            "duplicate_rate_pct": duplicate_rate_pct,
            "numerical_columns_count": len(num_cols),
            "categorical_columns_count": len(cat_cols),
            "datetime_columns_count": len(dt_cols),
            "id_columns_count": len(id_cols),
            "columns": column_profiles,
            "status": "READY_FOR_ANALYSIS",
        }

        return {
            "profile": profile,
            "dataframe": df,
        }
