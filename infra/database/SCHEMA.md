# Database Schema — Bitcoin Time-Series + Vector Metadata

These notes sketch how to organize the PostgreSQL + TimescaleDB + `pgvector` schema that will be deployed inside `infra/database/`.

## 1. Core Hypertables

### `market_data.window_5m`
Primary anchor for all unstructured records. Every other table stores the corresponding `window_start`, yielding a clean parent-child relationship.

| Column | Type | Notes |
| --- | --- | --- |
| `window_start` | `TIMESTAMPTZ NOT NULL` | Floor to 5-minute boundary. |
| `window_end` | `TIMESTAMPTZ` | `GENERATED ALWAYS AS (window_start + INTERVAL '5 minutes') STORED`. |
| `exchange` | `TEXT NOT NULL` | Venue identifier; use `ALL` for aggregated feeds. |
| `raw_payload` | `JSONB` | Optional unstructured data blob for that window (order book, news). |
| `status` | `TEXT DEFAULT 'open'` | Workflow flag (`open`, `sealed`, `replayed`). |
| `ingested_at` | `TIMESTAMPTZ DEFAULT now()` | Metadata. |

- Hypertable: `create_hypertable('market_data.window_5m','window_start', chunk_time_interval => INTERVAL '30 days');`
- Primary key `(window_start, exchange)`.

### `market_data.btc_ticks`
Child table storing every 100 ms trade, linked to its parent window.

| Column | Type | Notes |
| --- | --- | --- |
| `ts` | `TIMESTAMPTZ NOT NULL` | 100 ms aligned event time. |
| `window_start` | `TIMESTAMPTZ NOT NULL` | FK → `market_data.window_5m(window_start)`. |
| `exchange` | `TEXT NOT NULL` | Venue. |
| `source_seq` | `BIGINT` | Exchange sequence / trade id. |
| `price` | `NUMERIC(18,8)` | Trade price (USD). |
| `volume` | `NUMERIC(24,12)` | Trade size (BTC). |
| `side` | `SMALLINT` | +1 buy, −1 sell, 0 unknown. |
| `microstructure_flags` | `JSONB` | Latency/outlier annotations. |
| `ingested_at` | `TIMESTAMPTZ DEFAULT now()` | Insert time. |

- Hypertable: `create_hypertable('market_data.btc_ticks','ts', chunk_time_interval => INTERVAL '1 day');`
- Indexes: `(window_start, ts)`, `(exchange, ts DESC)`, unique `(exchange, source_seq)`.
- Compression after 7 days (`segmentby => 'exchange'`, `orderby => 'ts DESC'`).

### `market_data.btc_ohlcv_100ms`
Pre-aggregated candles derived from `btc_ticks`.

| Column | Type | Notes |
| --- | --- | --- |
| `bucket` | `TIMESTAMPTZ NOT NULL` | 100 ms bucket start. |
| `window_start` | `TIMESTAMPTZ NOT NULL` | Parent 5-minute window. |
| `exchange` | `TEXT NOT NULL` | Venue. |
| `open`, `high`, `low`, `close` | `NUMERIC(18,8)` | Aggregated prices. |
| `volume` | `NUMERIC(24,12)` | Sum of BTC volume. |
| `vwap` | `NUMERIC(18,8)` | Volume-weighted price. |
| `trade_count` | `INTEGER` | Number of ticks. |
| `features` | `JSONB` | Derived stats (spread, realized vol). |

- Hypertable with chunk interval `INTERVAL '7 days'`.
- Continuous aggregate definition:
  ```sql
  CREATE MATERIALIZED VIEW market_data.btc_ohlcv_100ms
  WITH (timescaledb.continuous) AS
  SELECT time_bucket('100 milliseconds', ts) AS bucket, exchange,
         first(price, ts) AS open,
         max(price) AS high,
         min(price) AS low,
         last(price, ts) AS close,
         sum(volume) AS volume,
         sum(price * volume) / NULLIF(sum(volume), 0) AS vwap,
         count(*) AS trade_count
  FROM market_data.btc_ticks
  GROUP BY 1,2;
  ```

## 2. Vector Store Tables

### `signals.btc_embeddings`
Stores per-bucket feature vectors for downstream ML/retrieval.

| Column | Type | Notes |
| --- | --- | --- |
| `bucket` | `TIMESTAMPTZ PRIMARY KEY` | Matches `btc_ohlcv_100ms.bucket`. |
| `window_start` | `TIMESTAMPTZ NOT NULL` | Same 5-minute parent as ticks/candles. |
| `exchange` | `TEXT NOT NULL` | Source venue. |
| `feature_vector` | `vector(256)` | Embedding produced by feature pipeline (dimension configurable). |
| `label_snapshot` | `JSONB` | Optional labels (future returns, vol). |
| `created_at` | `TIMESTAMPTZ DEFAULT now()` | Insertion timestamp. |

- Index: `CREATE INDEX ON signals.btc_embeddings USING ivfflat (feature_vector) WITH (lists = 200);`
- Use `ANALYZE` after bulk loads to optimize vector search.

## 3. Metadata Tables

### `ingestion.stream_offsets`
Tracks last processed offset per data source.

| Column | Type | Notes |
| --- | --- | --- |
| `source` | `TEXT PRIMARY KEY` | e.g., `binance_ws`, `historical_csv`. |
| `last_sequence` | `BIGINT` | Highest processed sequence. |
| `last_ts` | `TIMESTAMPTZ` | Timestamp of the sequence. |
| `state` | `JSONB` | Arbitrary decoder state. |
| `updated_at` | `TIMESTAMPTZ DEFAULT now()` | Maintenance timestamp. |

### `ops.retention_jobs`
Defines compression/backfill policies.

| Column | Type | Notes |
| --- | --- | --- |
| `job_name` | `TEXT PRIMARY KEY` | Human-readable id. |
| `hypertable` | `TEXT NOT NULL` | Target hypertable name. |
| `policy` | `JSONB NOT NULL` | e.g., `{ "compress_after": "7 days", "drop_after": "90 days" }`. |
| `enabled` | `BOOLEAN DEFAULT TRUE` | Toggle. |
| `created_at` | `TIMESTAMPTZ DEFAULT now()` | Metadata. |

## 4. Schema Relationships
- `market_data.window_5m` is the root entity; all tables contain the same `window_start` to keep 5-minute buckets co-located.
- `market_data.btc_ticks` feeds the continuous aggregate `market_data.btc_ohlcv_100ms`.
- `signals.btc_embeddings.bucket` references the candle table (`FOREIGN KEY bucket REFERENCES market_data.btc_ohlcv_100ms(bucket) ON DELETE CASCADE`).
- `ingestion.stream_offsets` ties ingestion workers to specific sequences, enabling idempotent replays.

## 5. Extension Requirements
```sql
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS timescaledb_toolkit;
CREATE EXTENSION IF NOT EXISTS vector;
```

## 6. Storage Notes
- Set `timescaledb.compress` on `btc_ticks` and `btc_ohlcv_100ms` with segment-by `exchange`.
- WAL archiving should cover all schemas; consider logical replication slots if downstream analytics need live data.

These definitions can be codified as migration files (Sqitch/Flyway) under `infra/database/migrations/` once the base schema is finalized.
