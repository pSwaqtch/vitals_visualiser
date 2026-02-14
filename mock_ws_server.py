#!/usr/bin/env python3
"""Mock WebSocket server that replays ADPD7000 xlsx data as a stream."""

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd


async def stream_handler(websocket, df: pd.DataFrame, rate: float, loop: bool):
    """Stream rows from df to a connected client."""
    delay = 1.0 / rate
    columns = list(df.columns)
    idx = 0
    while True:
        row = df.iloc[idx]
        msg = json.dumps({col: float(row[col]) for col in columns})
        try:
            await websocket.send(msg)
        except Exception:
            break
        idx += 1
        if idx >= len(df):
            if loop:
                idx = 0
            else:
                break
        await asyncio.sleep(delay)


async def main(args):
    import websockets

    if str(args.file).endswith(".csv"):
        df = pd.read_csv(args.file)
    else:
        sheets = pd.read_excel(args.file, sheet_name=None, engine="openpyxl")
        df = sheets.get("Data", pd.DataFrame())
        if df.empty:
            df = pd.read_excel(args.file, engine="openpyxl")

    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    print(f"Streaming at {args.rate} Hz on ws://{args.host}:{args.port}")
    if args.loop:
        print("Looping enabled — will restart from beginning when data is exhausted")

    async def handler(ws):
        addr = ws.remote_address
        print(f"Client connected: {addr}")
        try:
            await stream_handler(ws, df, args.rate, args.loop)
        finally:
            print(f"Client disconnected: {addr}")

    async with websockets.serve(handler, args.host, args.port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    default_file = Path(__file__).parent / "export_data" / "anonymous_7000_ppg_20260214-173257.xlsx"

    parser = argparse.ArgumentParser(description="Mock ADPD7000 WebSocket server")
    parser.add_argument("--file", type=Path, default=default_file,
                        help="Data file to replay (xlsx or csv)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rate", type=float, default=100.0,
                        help="Samples per second (default: 100)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop data indefinitely")

    asyncio.run(main(parser.parse_args()))
