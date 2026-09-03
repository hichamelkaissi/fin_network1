import pandas as pd
import subprocess
import psutil
import logging
from logging.handlers import RotatingFileHandler
import csv
import os
import socket
import uuid
from datetime import datetime, timezone
import time
import atexit
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

def gpu_mem():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            check=True
        )

        mem_info = []

        for line in result.stdout.strip().split('\n'):
            used, total = line.split(',')
            mem_info.append((int(used), int(total)))

        return mem_info

    except Exception:
        return []


def cpu_mem():
    process = psutil.Process()
    return process.memory_info().rss / 1024 ** 2

def setup_logger(
        logfile: str,
        name: str = 'app',
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        max_mb: int = 10,
        backup_count: int = 5
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))
    logger.handlers.clear()
    fmt = '%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    fh = RotatingFileHandler(
        logfile,
        maxBytes=max_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setLevel(file_level)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    return logger

def load_reconstruction_data(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)
    df = df[(df['IA'] > 0) | (df['IL'] > 0)].reset_index(drop=True)


    return df

class MetricsCSV:
    def __init__(
            self,
            path: str,
            fieldnames,
            run_id: str | None = None,
            flush_every: int = 1
    ):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        self.path = path
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.flush_every = flush_every
        self._buffer_count = 0

        self.fieldnames = [
            "ts",
            "run_id",
            "host",
            "pid",
            *fieldnames,
        ]

        file_exists = os.path.exists(self.path)

        self.file = open(self.path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)

        if not file_exists:
            self.writer.writeheader()
            self.file.flush()

        atexit.register(self.close)

    def log(self, **row):
        timestamp = datetime.now(timezone.utc).isoformat()

        full_row = {
            "ts": timestamp,
            "run_id": self.run_id,
            "host": self.hostname,
            "pid": self.pid,
            **row,
        }

        self.writer.writerow(full_row)
        self._buffer_count += 1

        if self._buffer_count >= self.flush_every:
            self.file.flush()
            self._buffer_count = 0

    def close(self):
        if not self.file.closed:
            self.file.flush()
            self.file.close()

# ---- walking two paths ----
QUARTER_RE = re.compile(r"^(\d{4}Q[1-4])\.(npy|csv)$")
def index_by_quarter(root: str, ext: str) -> dict[str, Path]:
    ext = ext.lstrip('.')
    out: dict[str, Path] = {}
    rootp = Path(root)

    for dirpath, _, filenames in os.walk(rootp):
        for name in filenames:
            m = QUARTER_RE.match(name)
            if not m:
                continue
            quarter, found_ext = m.group(1), m.group(2)
            if found_ext != ext:
                continue
            out[quarter] = Path(dirpath) / name
    return out

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        return tomllib.load(f)

# ---- Safe config loading ---
def require(cfg, *keys):
    cur = cfg
    path = []
    for k in keys:
        path.append(k)
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f'Missing required config key: {".".join(path)}')

        cur = cur[k]
    return cur
