"""Safe Read-Only SQL Tool for CustomerPulse Analyst Agent.
Strictly blocks destructive SQL statements and returns 'Unsafe SQL operation rejected.'
"""

import re
from typing import Dict, Any, List
from sqlalchemy import text
from backend.app.database.session import SessionLocal


class ReadOnlySQLTool:
    """Executes safe read-only SQL queries against the analytical database."""

    FORBIDDEN_KEYWORDS = [
        r"\bINSERT\b",
        r"\bUPDATE\b",
        r"\bDELETE\b",
        r"\bDROP\b",
        r"\bALTER\b",
        r"\bTRUNCATE\b",
        r"\bCREATE\b",
        r"\bGRANT\b",
        r"\bREVOKE\b",
        r"\bREPLACE\b",
        r"\bATTACH\b",
        r"\bDETACH\b",
        r"\bPRAGMA\b",
        r"\bEXEC\b",
    ]

    @classmethod
    def is_safe_query(cls, query: str) -> bool:
        """Inspect query to ensure it only performs safe SELECT operations."""
        cleaned_query = query.strip()
        
        # Must start with SELECT or WITH
        if not re.match(r"^(SELECT|WITH)\b", cleaned_query, re.IGNORECASE):
            return False

        # Must not contain forbidden modification keywords
        for pattern in cls.FORBIDDEN_KEYWORDS:
            if re.search(pattern, cleaned_query, re.IGNORECASE):
                return False

        # Must not have multiple semicolon-separated statements that could chain attacks
        statements = [s.strip() for s in cleaned_query.split(";") if s.strip()]
        if len(statements) > 1:
            return False

        return True

    @classmethod
    def execute_query(cls, query: str, limit: int = 50) -> Dict[str, Any]:
        """Validate and execute SQL query."""
        if not cls.is_safe_query(query):
            return {
                "success": False,
                "security_policy_violation": True,
                "error": "SECURITY POLICY: Query rejected. Reason: Destructive SQL operation detected. Database mutation: DISABLED. Allowed operations: SELECT / WITH only.",
                "policy_rule": "Read-Only Analytical Database Isolation (Zero Mutation Allowed)",
                "rows": [],
                "columns": [],
                "row_count": 0,
            }

        db = SessionLocal()
        try:
            # Enforce limit if not present
            clean_q = query.strip().rstrip(";")
            if not re.search(r"\bLIMIT\b", clean_q, re.IGNORECASE):
                clean_q = f"{clean_q} LIMIT {limit}"

            result = db.execute(text(clean_q))
            columns = list(result.keys()) if result.returns_rows else []
            rows = [dict(zip(columns, row)) for row in result.fetchall()] if result.returns_rows else []

            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"SQL execution error: {str(e)}",
                "rows": [],
                "columns": [],
                "row_count": 0,
            }
        finally:
            db.close()
