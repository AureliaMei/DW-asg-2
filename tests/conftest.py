import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def sc():
    from distfd.spark import get_spark

    spark = get_spark(4)
    yield spark.sparkContext
    spark.stop()
