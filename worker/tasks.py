import os
import pandas as pd
from worker.celery_app import app
from dews import MonitorConfig
from dews.windowing import RedisSlidingWindow
from dews.factory import load_system

# Initialize a global system instance for the worker to avoid reloading model every task
SYSTEM = None
REFERENCE_FRAME = None
REDIS_WINDOW = None

def get_system():
    global SYSTEM, REFERENCE_FRAME, REDIS_WINDOW
    if SYSTEM is None:
        config = MonitorConfig()
        store_path = os.getenv("DRIFT_DB_PATH", "artifacts/dews.sqlite3")
        SYSTEM, REFERENCE_FRAME = load_system(config, store_path)
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        REDIS_WINDOW = RedisSlidingWindow(config.window_size, redis_url=redis_url)
        # Seed reference
        REDIS_WINDOW.initialize_reference(REFERENCE_FRAME)
    return SYSTEM, REDIS_WINDOW

@app.task
def process_drift_batch(batch_data: list[dict], batch_index: int):
    system, window = get_system()
    
    batch_df = pd.DataFrame(batch_data)
    live_window_df = window.add_batch(batch_df)
    
    reference_df = REFERENCE_FRAME if REFERENCE_FRAME is not None else window.get_reference()
    
    result = system.process_batch(
        batch_index=batch_index,
        reference_frame=reference_df,
        live_window=live_window_df
    )
    return {"batch_index": batch_index, "drift_score": result.drift_score}
