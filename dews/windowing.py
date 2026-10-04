from __future__ import annotations

import json

import pandas as pd
import redis


class RedisSlidingWindow:
    def __init__(self, window_size: int, redis_url: str = "redis://localhost:6379/0", key_prefix: str = "dews:window"):
        self.window_size = window_size
        self.key = f"{key_prefix}:live"
        self.ref_key = f"{key_prefix}:reference"
        self.redis = redis.from_url(redis_url)

    def initialize_reference(self, reference: pd.DataFrame) -> None:
        self.redis.set(self.ref_key, reference.to_json(orient="records"))

    def get_reference(self) -> pd.DataFrame | None:
        data = self.redis.get(self.ref_key)
        if not data:
            return None
        import io
        return pd.read_json(io.StringIO(data.decode("utf-8") if isinstance(data, bytes) else data), orient="records")

    def add_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        records = batch.to_dict(orient="records")
        pipeline = self.redis.pipeline()
        for r in records:
            pipeline.rpush(self.key, json.dumps(r))
        # Keep only the last `window_size` items
        pipeline.ltrim(self.key, -self.window_size, -1)
        pipeline.execute()
        return self.get_live_window()

    def get_live_window(self) -> pd.DataFrame:
        items = self.redis.lrange(self.key, 0, -1)
        if not items:
            return pd.DataFrame()
        records = [json.loads(item) for item in items]
        return pd.DataFrame.from_records(records)
