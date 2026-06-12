"""
config/azure.py
HEQP — Azure connection config loader.

Reads credentials from environment variables or a local .env file.
Never commit real credentials to git — .env is in .gitignore.
"""

import os
from pathlib import Path


def load_env(env_file: str = ".env"):
    """
    Parse a simple KEY=VALUE .env file and populate os.environ.
    Skips blank lines and comments (#).
    """
    search_paths = [
        Path(env_file),
        Path(__file__).parent / env_file,
        Path(__file__).parent.parent / env_file
    ]
    
    p = next((path for path in search_paths if path.is_file()), None)
    if not p:
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def get_eventhubs_connection_string() -> str:
    load_env()
    val = os.getenv("EVENTHUBS_CONNECTION_STRING", "")
    if not val:
        raise EnvironmentError(
            "EVENTHUBS_CONNECTION_STRING not set. "
            "Add it to .env or export it before running."
        )
    return val


EVENTHUB_NAME = "heqp-episodes"
EVENTHUB_PARTITIONS = 32