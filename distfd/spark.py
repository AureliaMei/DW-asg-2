"""Local Spark session with N cores standing in for N nodes."""
import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)


def get_spark(nodes=4):
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in [ROOT, os.environ.get("PYTHONPATH")] if p)
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master(f"local[{nodes}]")
        .appName("DisTFD")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "4g")
        .config("spark.python.worker.reuse", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
