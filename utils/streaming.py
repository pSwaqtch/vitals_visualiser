"""WebSocket streaming pipeline: buffer, recorder, and client."""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"


class DataBuffer:
    """Thread-safe ring buffer backed by collections.deque."""

    def __init__(self, maxlen: int = 60_000, columns: list[str] | None = None):
        self._lock = threading.Lock()
        self._maxlen = maxlen
        self._buf: collections.deque[dict] = collections.deque(maxlen=maxlen)
        self._columns = list(columns) if columns else []
        self._total_received = 0

    def append(self, row: dict):
        with self._lock:
            self._buf.append(row)
            self._total_received += 1
            if not self._columns:
                self._columns = list(row.keys())

    def extend(self, rows: list[dict]):
        with self._lock:
            self._buf.extend(rows)
            self._total_received += len(rows)
            if not self._columns and rows:
                self._columns = list(rows[0].keys())

    def snapshot(self) -> pd.DataFrame:
        """Return current buffer contents as a DataFrame (copy)."""
        with self._lock:
            if not self._buf:
                return pd.DataFrame(columns=self._columns)
            return pd.DataFrame(list(self._buf), columns=self._columns)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @property
    def total_received(self) -> int:
        with self._lock:
            return self._total_received

    def clear(self):
        with self._lock:
            self._buf.clear()
            self._total_received = 0


class StreamRecorder:
    """Records raw data to Parquet and exports filtered data as CSV."""

    def __init__(self, output_dir: Path, session_id: str):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = session_id
        self._parquet_path = output_dir / f"raw_{session_id}.parquet"
        self._writer = None
        self._schema = None
        self._recording = False
        self._rows_written = 0
        self._lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def parquet_path(self) -> Path:
        return self._parquet_path

    @property
    def rows_written(self) -> int:
        return self._rows_written

    def start(self):
        self._recording = True

    def stop(self):
        self._recording = False
        self._close_writer()

    def write_rows(self, rows: list[dict]):
        """Append rows to the Parquet file. Called from the WS client thread."""
        if not self._recording or not rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        with self._lock:
            table = pa.Table.from_pylist(rows)
            if self._writer is None:
                self._schema = table.schema
                self._writer = pq.ParquetWriter(str(self._parquet_path), self._schema)
            self._writer.write_table(table)
            self._rows_written += len(rows)

    def export_filtered_csv(self, df: pd.DataFrame, suffix: str = "filtered") -> Path:
        """Save a filtered DataFrame as CSV."""
        path = self._output_dir / f"{suffix}_{self._session_id}.csv"
        df.to_csv(path, index=False)
        return path

    def _close_writer(self):
        with self._lock:
            if self._writer:
                self._writer.close()
                self._writer = None


class WebSocketClient:
    """Connects to a WebSocket URL in a background thread and feeds a DataBuffer."""

    def __init__(
        self,
        url: str,
        buffer: DataBuffer,
        recorder: StreamRecorder | None = None,
        decoder: callable = None,
    ):
        self._url = url
        self._buffer = buffer
        self._recorder = recorder
        self._decoder = decoder or self._default_decoder
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._connected = False
        self._error: str | None = None

    @staticmethod
    def _default_decoder(msg: str | bytes) -> list[dict]:
        """Default decoder: JSON object or array of objects."""
        data = json.loads(msg)
        if isinstance(data, dict):
            return [data]
        return data

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def error(self) -> str | None:
        return self._error

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._connected = False

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._listen())
        except Exception as e:
            if not self._stop_event.is_set():
                self._error = str(e)
                logger.exception("WebSocket client error")
        finally:
            self._connected = False
            loop.close()

    async def _listen(self):
        import websockets

        try:
            async with websockets.connect(self._url) as ws:
                self._connected = True
                self._error = None
                batch: list[dict] = []
                while not self._stop_event.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if batch:
                            self._flush(batch)
                            batch = []
                        continue

                    rows = self._decoder(msg)
                    batch.extend(rows)

                    if len(batch) >= 50:
                        self._flush(batch)
                        batch = []

                if batch:
                    self._flush(batch)
        except Exception as e:
            self._error = str(e)
            self._connected = False

    def _flush(self, batch: list[dict]):
        self._buffer.extend(batch)
        if self._recorder:
            self._recorder.write_rows(batch)


def get_or_create_stream(url: str, buffer_size: int = 60_000):
    """Retrieve or create streaming objects stored in st.session_state."""
    import streamlit as st

    key = "_ws_stream"
    if key not in st.session_state:
        buf = DataBuffer(maxlen=buffer_size)
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        rec = StreamRecorder(output_dir=RECORDINGS_DIR, session_id=session_id)
        client = WebSocketClient(url=url, buffer=buf, recorder=rec)
        st.session_state[key] = {"client": client, "buffer": buf, "recorder": rec}

    s = st.session_state[key]
    return s["client"], s["buffer"], s["recorder"]
