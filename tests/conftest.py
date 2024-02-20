import pytest
import sys
from typing import Any


@pytest.fixture
def capture_stdout(monkeypatch):
    buffer: dict[str, Any] = {"stdout": "", "write_calls": 0}

    def fake_write(s) -> None:
        buffer["stdout"] += s
        buffer["write_calls"] += 1

    monkeypatch.setattr(sys.stdout, 'write', fake_write)
    return buffer


@pytest.fixture(scope="session")
def db_conn() -> None:
    pass
