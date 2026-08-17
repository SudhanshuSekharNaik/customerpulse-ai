"""Tests for SQL safety, AST validation, and rejection of destructive queries."""

import pytest
from ai.tools.sql_tool import ReadOnlySQLTool


def test_safe_select_query():
    query = "SELECT customer_id, total_revenue FROM customers LIMIT 5;"
    assert ReadOnlySQLTool.is_safe_query(query) is True


def test_reject_drop_table():
    query = "DROP TABLE customers;"
    assert ReadOnlySQLTool.is_safe_query(query) is False
    res = ReadOnlySQLTool.execute_query(query)
    assert res["success"] is False
    assert "Unsafe SQL operation rejected" in res["error"]


def test_reject_delete_query():
    query = "DELETE FROM events WHERE id > 0;"
    assert ReadOnlySQLTool.is_safe_query(query) is False
    res = ReadOnlySQLTool.execute_query(query)
    assert res["success"] is False
    assert "Unsafe SQL operation rejected" in res["error"]


def test_reject_update_query():
    query = "UPDATE customers SET total_revenue = 999999 WHERE customer_id = 'cust_1';"
    assert ReadOnlySQLTool.is_safe_query(query) is False
    res = ReadOnlySQLTool.execute_query(query)
    assert res["success"] is False
    assert "Unsafe SQL operation rejected" in res["error"]


def test_reject_truncate():
    query = "TRUNCATE TABLE model_runs;"
    assert ReadOnlySQLTool.is_safe_query(query) is False


def test_reject_chained_query_injection():
    query = "SELECT * FROM customers; DROP TABLE events;"
    assert ReadOnlySQLTool.is_safe_query(query) is False
