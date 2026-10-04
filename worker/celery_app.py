import os
import sys
from pathlib import Path
from celery import Celery

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("dews_worker", broker=redis_url, backend=redis_url)
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)
