# Database Layer Overview

This folder documents the PostgreSQL + TimescaleDB + `pgvector` deployment that stores normalized Bitcoin tick data (100 ms resolution) and downstream embeddings.

## Data Flow
1. **Raw ingestion**  
   - Exchange feeds (websocket dumps or CSV replays) land in S3.  
   - An ETL job inside the VPC fetches batches via the S3 gateway endpoint, decodes trades, normalizes fields (timestamp, price, volume, side), and writes into Timescale using `COPY` or batched inserts.

2. **5-minute window anchor (`market_data.window_5m`)**  
   - Incoming events snap to their parent 5-minute boundary (`window_start`).  
   - Any unstructured payloads (LOB snapshots, news, derived blobs) hang off this table.

3. **Hypertable storage (`market_data.btc_ticks`)**  
   - Every tick stores both its exact `ts` (100 ms resolution) and the parent `window_start` so scans over a single 5-minute slice can fetch all raw rows quickly.  
   - Deduplication uses `(exchange, source_seq)`; compression turns on after 7 days.
   - Analysts who need the “raw stream” can simply `SELECT * FROM market_data.btc_ticks WHERE window_start BETWEEN t AND t + interval '5 minutes' ORDER BY ts;` to obtain the contiguous 100 ms feed.

4. **Aggregations (`market_data.btc_ohlcv_100ms`)**  
   - Timescale continuous aggregates roll ticks into OHLCV/vwap summaries per 100 ms bucket.  
   - Queries hitting mid/long horizons read from this materialized view rather than raw ticks.

5. **Feature / embedding generation (`signals.btc_embeddings`)**  
   - A feature pipeline (PyTorch/NumPy job) reads recent candles, computes engineered features or embeddings (dimension configurable), and stores them in a vector column.  
   - `pgvector` indexes (`ivfflat`) enable fast nearest-neighbour searches for similar micro-structures.

6. **Ops metadata**  
   - `ingestion.stream_offsets` tracks per-source offsets so ETL jobs resume exactly.  
   - `ops.retention_jobs` records compression/retention policies applied via Timescale background workers.

## Storage Characteristics
- All hypertables are chunked daily, with retention policies (e.g., drop raw ticks after 90 days, keep aggregates forever).
- WAL archiving to S3 ensures point-in-time recovery; the EC2 instance requires IAM permissions limited to the backup bucket.

## Next Steps
- Add Terraform modules (VPC endpoint, security groups, EC2 bootstrap) to this directory.
- Define SQL migration files (e.g., `migrations/001_init.sql`) that create the schemas/tables described in `SCHEMA.md`.

Use this README as the high-level reference for anyone extending the database layer.
