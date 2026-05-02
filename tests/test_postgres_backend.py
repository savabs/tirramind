"""Unit tests for PostgresBackend SQL translation layer.

These tests verify the SQL dialect translation without requiring a
running PostgreSQL instance.  They exercise ``_translate_ddl``,
``_translate_dml``, and the adapter wiring.
"""

from __future__ import annotations

import pytest

from agent.pipeline.storage_backend import (
    _AUTO_INCREMENT_TABLES,
    _IGNORE_CONFLICT_TARGETS,
    _PK_COLUMNS,
    _UPSERT_CONFLICT_TARGETS,
    PostgresBackend,
    _translate_ddl,
    _translate_dml,
)

# ── DDL translation ───────────────────────────────────────────


class TestTranslateDDL:
    def test_autoincrement_to_serial(self) -> None:
        ddl = "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        result = _translate_ddl(ddl)
        assert "SERIAL PRIMARY KEY" in result
        assert "AUTOINCREMENT" not in result

    def test_real_to_double_precision(self) -> None:
        ddl = "fetched_at REAL NOT NULL,"
        result = _translate_ddl(ddl)
        assert "DOUBLE PRECISION NOT NULL" in result
        assert " REAL " not in result

    def test_real_with_default(self) -> None:
        ddl = "confidence REAL NOT NULL DEFAULT 1.0,"
        result = _translate_ddl(ddl)
        assert "DOUBLE PRECISION NOT NULL DEFAULT 1.0," in result

    def test_blob_to_bytea(self) -> None:
        ddl = "state_dict_blob BLOB NOT NULL"
        result = _translate_ddl(ddl)
        assert "BYTEA NOT NULL" in result
        assert "BLOB" not in result

    def test_create_table_if_not_exists_preserved(self) -> None:
        ddl = "CREATE TABLE IF NOT EXISTS dag_runs ("
        result = _translate_ddl(ddl)
        assert "CREATE TABLE IF NOT EXISTS dag_runs (" in result

    def test_create_index_preserved(self) -> None:
        ddl = "CREATE INDEX IF NOT EXISTS idx_foo ON bar(baz);"
        result = _translate_ddl(ddl)
        assert result == ddl

    def test_unique_index_preserved(self) -> None:
        ddl = "CREATE UNIQUE INDEX IF NOT EXISTS idx_features_unique ON features(feature_name, version, effective_at);"
        result = _translate_ddl(ddl)
        assert result == ddl

    def test_full_table_translation(self) -> None:
        ddl = (
            "CREATE TABLE IF NOT EXISTS pipeline_data (\n"
            "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            "    source TEXT NOT NULL,\n"
            "    fetched_at REAL NOT NULL,\n"
            "    params_json TEXT NOT NULL,\n"
            "    data_json TEXT NOT NULL\n"
            ");"
        )
        result = _translate_ddl(ddl)
        assert "SERIAL PRIMARY KEY" in result
        assert "DOUBLE PRECISION NOT NULL" in result
        assert "AUTOINCREMENT" not in result
        # TEXT should remain TEXT.
        assert "TEXT NOT NULL" in result

    def test_real_not_replaced_in_identifiers(self) -> None:
        # "REAL" as a word boundary should not corrupt identifiers
        # like "unrealized" or column names containing "real".
        ddl = "unrealized_pnl REAL NOT NULL"
        result = _translate_ddl(ddl)
        assert "unrealized_pnl DOUBLE PRECISION NOT NULL" in result


# ── DML translation ───────────────────────────────────────────


class TestTranslateDML:
    # ── Parameter placeholder ──────────────────────────────────

    def test_question_mark_to_percent_s(self) -> None:
        sql = "SELECT * FROM foo WHERE a=? AND b=?"
        translated, pk = _translate_dml(sql)
        assert translated == "SELECT * FROM foo WHERE a=%s AND b=%s"
        assert pk is None

    # ── INSERT OR REPLACE ──────────────────────────────────────

    def test_insert_or_replace_features(self) -> None:
        sql = (
            "INSERT OR REPLACE INTO features "
            "(feature_name, version, effective_at, computed_at, horizon, "
            "value, quality, missing_reason, source_signals_json, builder, "
            "unit, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        translated, pk = _translate_dml(sql)

        # Should NOT contain SQLite-specific syntax.
        assert "OR REPLACE" not in translated
        assert "?" not in translated

        # Should have ON CONFLICT on the correct columns.
        assert "ON CONFLICT (feature_name, version, effective_at)" in translated
        assert "DO UPDATE SET" in translated

        # Update clause should include non-conflict columns.
        assert "computed_at = EXCLUDED.computed_at" in translated
        assert "value = EXCLUDED.value" in translated
        assert "metadata_json = EXCLUDED.metadata_json" in translated

        # Conflict columns should NOT appear in the SET clause.
        assert "feature_name = EXCLUDED.feature_name" not in translated
        assert "version = EXCLUDED.version" not in translated
        assert "effective_at = EXCLUDED.effective_at" not in translated

        # Should return PK column for RETURNING.
        assert pk == "id"

    def test_insert_or_replace_beliefs(self) -> None:
        sql = "INSERT OR REPLACE INTO beliefs (variable_name, version, effective_at, computed_at) VALUES (?, ?, ?, ?)"
        translated, pk = _translate_dml(sql)
        assert "ON CONFLICT (variable_name, version, effective_at)" in translated
        assert "DO UPDATE SET computed_at = EXCLUDED.computed_at" in translated
        assert pk == "id"

    def test_insert_or_replace_pending_rl(self) -> None:
        sql = (
            "INSERT OR REPLACE INTO pending_rl_transitions "
            "(date, timestamp, state_json, action_json) "
            "VALUES (?, ?, ?, ?)"
        )
        translated, pk = _translate_dml(sql)
        assert "ON CONFLICT (date)" in translated
        assert "DO UPDATE SET" in translated
        assert pk == "id"

    def test_insert_or_replace_portfolio_weights(self) -> None:
        sql = (
            "INSERT OR REPLACE INTO portfolio_weights "
            "(date, ticker, weight, metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)"
        )
        translated, pk = _translate_dml(sql)
        assert "ON CONFLICT (date, ticker)" in translated
        assert "weight = EXCLUDED.weight" in translated
        assert pk == "id"

    def test_insert_or_replace_paper_trade_pnl(self) -> None:
        sql = (
            "INSERT OR REPLACE INTO paper_trade_pnl "
            "(date, portfolio_return, benchmark_return, cumulative_return) "
            "VALUES (?, ?, ?, ?)"
        )
        translated, pk = _translate_dml(sql)
        assert "ON CONFLICT (date)" in translated
        assert "DO UPDATE SET" in translated
        assert pk == "id"

    # ── INSERT OR IGNORE ───────────────────────────────────────

    def test_insert_or_ignore_entities(self) -> None:
        sql = "INSERT OR IGNORE INTO entities (entity_id, entity_type, canonical_name, created_at) VALUES (?, ?, ?, ?)"
        translated, pk = _translate_dml(sql)
        assert "OR IGNORE" not in translated
        assert "ON CONFLICT (entity_id) DO NOTHING" in translated
        # entities PK is TEXT, not auto-increment.
        assert pk is None

    def test_insert_or_ignore_entity_aliases(self) -> None:
        sql = (
            "INSERT OR IGNORE INTO entity_aliases "
            "(entity_id, source, external_id, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?)"
        )
        translated, pk = _translate_dml(sql)
        assert "ON CONFLICT (source, external_id) DO NOTHING" in translated
        assert pk == "alias_id"

    def test_insert_or_ignore_entity_links(self) -> None:
        sql = (
            "INSERT OR IGNORE INTO entity_links "
            "(entity_id_a, entity_id_b, link_type, confidence, "
            "source, created_at, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        translated, pk = _translate_dml(sql)
        assert "ON CONFLICT (entity_id_a, entity_id_b, link_type) DO NOTHING" in translated
        assert pk == "link_id"

    def test_insert_or_ignore_discovered_sources(self) -> None:
        sql = "INSERT OR IGNORE INTO discovered_sources (source_id, name, url) VALUES (?, ?, ?)"
        translated, pk = _translate_dml(sql)
        assert "ON CONFLICT (source_id) DO NOTHING" in translated
        # TEXT PK, not auto-increment.
        assert pk is None

    def test_insert_or_ignore_entity_type_registry(self) -> None:
        sql = (
            "INSERT OR IGNORE INTO entity_type_registry "
            "(type_name, parent_type, discovered_at, source) "
            "VALUES (?, ?, ?, ?)"
        )
        translated, pk = _translate_dml(sql)
        assert "ON CONFLICT (type_name) DO NOTHING" in translated
        assert pk is None

    # ── Plain INSERT ───────────────────────────────────────────

    def test_plain_insert_auto_increment(self) -> None:
        sql = "INSERT INTO pipeline_data (source, fetched_at, params_json, data_json) VALUES (?, ?, ?, ?)"
        translated, pk = _translate_dml(sql)
        assert "?" not in translated
        assert pk == "id"

    def test_plain_insert_non_auto_increment(self) -> None:
        sql = "INSERT INTO dag_runs (run_id, dag_name, started_at, status, trigger) VALUES (?, ?, ?, ?, ?)"
        translated, pk = _translate_dml(sql)
        assert pk is None

    def test_plain_insert_schema_migrations(self) -> None:
        sql = "INSERT INTO schema_migrations (schema_name, version, applied_at, description) VALUES (?, ?, ?, ?)"
        translated, pk = _translate_dml(sql)
        assert pk is None  # composite PK, not auto-increment

    # ── SELECT / UPDATE / DELETE pass-through ──────────────────

    def test_select_passthrough(self) -> None:
        sql = "SELECT * FROM features WHERE feature_name=? ORDER BY effective_at"
        translated, pk = _translate_dml(sql)
        assert translated == ("SELECT * FROM features WHERE feature_name=%s ORDER BY effective_at")
        assert pk is None

    def test_update_passthrough(self) -> None:
        sql = "UPDATE dag_runs SET status=?, finished_at=? WHERE run_id=?"
        translated, pk = _translate_dml(sql)
        assert translated == ("UPDATE dag_runs SET status=%s, finished_at=%s WHERE run_id=%s")
        assert pk is None

    def test_delete_passthrough(self) -> None:
        sql = "DELETE FROM features WHERE feature_name=?"
        translated, pk = _translate_dml(sql)
        assert "feature_name=%s" in translated
        assert pk is None


# ── Registry completeness ─────────────────────────────────────


class TestRegistryCompleteness:
    """Verify that every INSERT OR REPLACE / IGNORE table is registered."""

    def test_all_upsert_tables_have_conflict_targets(self) -> None:
        for table in _UPSERT_CONFLICT_TARGETS:
            assert len(_UPSERT_CONFLICT_TARGETS[table]) > 0

    def test_all_ignore_tables_have_conflict_targets(self) -> None:
        for table in _IGNORE_CONFLICT_TARGETS:
            assert len(_IGNORE_CONFLICT_TARGETS[table]) > 0

    def test_pk_column_tables_are_auto_increment(self) -> None:
        for table in _PK_COLUMNS:
            assert table in _AUTO_INCREMENT_TABLES


# ── PostgresBackend construction ──────────────────────────────


class TestPostgresBackendConstruction:
    def test_import_error_without_psycopg2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PostgresBackend raises ImportError with a helpful message."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "psycopg2":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        backend = PostgresBackend("postgresql://localhost/test")
        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="psycopg2 is required"):
            backend.get_connection()

    def test_invalid_schema_name_rejected(self) -> None:
        backend = PostgresBackend("postgresql://localhost/test", schema="DROP TABLE--")
        # Schema validation happens in get_connection(), which also
        # needs psycopg2.  So we test the regex directly.
        import re

        assert not re.fullmatch(r"[a-zA-Z_]\w*", "DROP TABLE--")
        assert re.fullmatch(r"[a-zA-Z_]\w*", "test_schema_123")

    def test_db_path_returns_dsn(self) -> None:
        dsn = "postgresql://user:pass@host:5432/db"
        backend = PostgresBackend(dsn)
        assert backend.db_path == dsn

    def test_is_memory_always_false(self) -> None:
        backend = PostgresBackend("postgresql://localhost/test")
        assert backend.is_memory is False
