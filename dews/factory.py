import os
import importlib
import logging
from typing import Tuple, Any

from dews.config import MonitorConfig
from dews.monitoring import DriftMonitoringSystem, build_demo_system
import pandas as pd

logger = logging.getLogger(__name__)

def load_system(config: MonitorConfig, store_path: str) -> Tuple[DriftMonitoringSystem, Any]:
    """
    Dynamically loads the monitoring system based on the DEWS_SYSTEM_FACTORY env var.
    If not set, defaults to the internal Digits demo scenario.
    
    The factory function must return: (DriftMonitoringSystem, ReferenceDataFrame)
    or for the demo it returns (DriftMonitoringSystem, DemoScenario)
    """
    factory_path = os.getenv("DEWS_SYSTEM_FACTORY")
    if not factory_path:
        logger.info("No DEWS_SYSTEM_FACTORY set, defaulting to internal digits demo.")
        system, scenario = build_demo_system(config, store_path)
        reference_frame = scenario.reference_frame[scenario.feature_schema.continuous + scenario.feature_schema.categorical]
        return system, reference_frame
    
    logger.info(f"Loading custom DEWS factory from {factory_path}")
    try:
        import sys
        from pathlib import Path
        cwd_str = str(Path.cwd())
        if cwd_str not in sys.path:
            sys.path.insert(0, cwd_str)
        module_name, func_name = factory_path.split(":")
        module = importlib.import_module(module_name)
        factory_func = getattr(module, func_name)
        
        # Factory must return (system, reference_frame)
        system, reference_frame = factory_func(config, store_path)
        return system, reference_frame
    except Exception as exc:
        logger.error(f"Failed to load custom system factory '{factory_path}': {exc}")
        raise
