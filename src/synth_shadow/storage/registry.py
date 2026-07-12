"""SQLite registry for prompts, forecasts, and scores."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from synth_shadow.storage.files import ensure_dir
from synth_shadow.utils.time import utc_now

LOG = logging.getLogger(__name__)


class ForecastRegistry:
    """Minimal durable state for the shadow workflow."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self._init_schema()

    def upsert_prompt(self, start_time: str, source: str = "synth") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                insert into prompts(start_time, source, created_at)
                values (?, ?, ?)
                on conflict(start_time) do update set source=excluded.source
                """,
                (start_time, source, utc_now().isoformat()),
            )

    def upsert_prompts(self, start_times: list[str], source: str = "synth") -> None:
        created_at = utc_now().isoformat()
        with self._connect() as conn:
            conn.executemany(
                """
                insert into prompts(start_time, source, created_at)
                values (?, ?, ?)
                on conflict(start_time) do update set source=excluded.source
                """,
                [(start_time, source, created_at) for start_time in start_times],
            )
        LOG.debug("Registry upserted prompts count=%s source=%s", len(start_times), source)

    def list_prompts(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("select * from prompts order by start_time desc").fetchall()
        return [dict(row) for row in rows]

    def register_forecast(
        self,
        forecast_dir: str,
        metadata: dict[str, Any],
        status: str = "pending",
    ) -> None:
        prompt_start_time = metadata.get("prompt_start_time") or metadata["data_cutoff"]
        with self._connect() as conn:
            conn.execute(
                """
                insert into forecasts(
                    forecast_dir, prompt_start_time, generated_at, data_cutoff,
                    model_version, asset, status, metadata_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(forecast_dir) do update set
                    prompt_start_time=excluded.prompt_start_time,
                    generated_at=excluded.generated_at,
                    data_cutoff=excluded.data_cutoff,
                    model_version=excluded.model_version,
                    asset=excluded.asset,
                    status=excluded.status,
                    metadata_json=excluded.metadata_json
                """,
                (
                    forecast_dir,
                    str(prompt_start_time),
                    metadata["generated_at"],
                    metadata["data_cutoff"],
                    metadata["model_version"],
                    metadata["asset"],
                    status,
                    json.dumps(metadata, default=str),
                ),
            )
        LOG.debug("Registry registered forecast dir=%s status=%s", forecast_dir, status)

    def update_forecast_status(self, forecast_dir: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "update forecasts set status=? where forecast_dir=?",
                (status, forecast_dir),
            )

    def list_forecasts(self, status: str | None = None, asset: str | None = None) -> list[dict[str, Any]]:
        query = "select * from forecasts"
        clauses = []
        params: list[Any] = []
        if status:
            clauses.append("status=?")
            params.append(status)
        if asset:
            clauses.append("asset=?")
            params.append(asset.upper())
        if clauses:
            query += " where " + " and ".join(clauses)
        query += " order by generated_at desc"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def latest_forecast(self, asset: str | None = None) -> dict[str, Any] | None:
        query = "select * from forecasts"
        params: tuple[Any, ...] = ()
        if asset:
            query += " where asset=?"
            params = (asset.upper(),)
        query += " order by generated_at desc limit 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return dict(row) if row else None

    def register_score(
        self,
        forecast_dir: str,
        score: dict[str, Any],
        realized_path_file: str,
        comparison_json: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                insert into scores(
                    forecast_dir, scored_at, raw_crps, realized_path_file,
                    comparison_json, score_json
                )
                values (?, ?, ?, ?, ?, ?)
                on conflict(forecast_dir) do update set
                    scored_at=excluded.scored_at,
                    raw_crps=excluded.raw_crps,
                    realized_path_file=excluded.realized_path_file,
                    comparison_json=excluded.comparison_json,
                    score_json=excluded.score_json
                """,
                (
                    forecast_dir,
                    utc_now().isoformat(),
                    float(score["raw_crps"]),
                    realized_path_file,
                    json.dumps(comparison_json or {}, default=str),
                    json.dumps(score, default=str),
                ),
            )
            conn.execute("update forecasts set status='scored' where forecast_dir=?", (forecast_dir,))
        LOG.debug("Registry registered score forecast_dir=%s raw_crps=%s", forecast_dir, score["raw_crps"])

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                create table if not exists prompts (
                    start_time text primary key,
                    source text not null,
                    created_at text not null
                );

                create table if not exists forecasts (
                    forecast_dir text primary key,
                    prompt_start_time text not null,
                    generated_at text not null,
                    data_cutoff text not null,
                    model_version text not null,
                    asset text not null,
                    status text not null,
                    metadata_json text not null
                );

                create table if not exists scores (
                    forecast_dir text primary key,
                    scored_at text not null,
                    raw_crps real not null,
                    realized_path_file text not null,
                    comparison_json text not null,
                    score_json text not null,
                    foreign key(forecast_dir) references forecasts(forecast_dir)
                );
                """
            )
