"""OpenPI websocket evaluation against recorded observations."""
from contextlib import ExitStack
import json
from threading import Event, Lock, Thread
import time

import av
import numpy as np
import pyarrow.parquet as pq
from websockets.sync.client import connect
from openpi_client import msgpack_numpy

from paired import CAMERAS


class Cancelled(Exception):
    pass


class Inference:
    def __init__(self, dataset, host="10.132.90.113"):
        self.dataset = dataset
        self.host = host
        self.lock = Lock()
        self.stop = Event()
        self.closed = Event()
        self.running = False
        self.ids = []
        self.states = {}
        self.results = {}
        self.error = ""
        self.ready = False
        Thread(target=self.monitor, daemon=True).start()

    def connection(self, port):
        return connect(f"ws://{self.host}:{port}", open_timeout=3, close_timeout=1,
                       compression=None, max_size=None)

    def monitor(self):
        while not self.closed.is_set():
            with self.lock:
                running = self.running
            if running:
                self.closed.wait(5)
                continue
            ready = True
            for port in (8002, 8001):
                try:
                    with self.connection(port) as ws:
                        metadata = msgpack_numpy.unpackb(ws.recv(timeout=3))
                        if not isinstance(metadata, dict):
                            ready = False
                except Exception:
                    ready = False
            with self.lock:
                self.ready = ready
            self.closed.wait(5)

    def snapshot(self):
        with self.lock:
            return {"ready": self.ready, "running": self.running,
                    "stopping": self.running and self.stop.is_set(),
                    "ids": self.ids[:], "states": dict(self.states), "error": self.error}

    def start(self, ids):
        if (not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids)
                or len(set(ids)) != len(ids) or any(i not in self.dataset.ids for i in ids)):
            raise ValueError("样本名单无效")
        with self.lock:
            if self.running:
                raise ValueError("请等待当前推理停止")
            if not self.ready:
                raise ValueError("等待 Solid 和 Liquid 模型服务就绪")
            if self.ids != ids:
                self.results = {}
                self.states = {}
            self.ids = ids[:]
            self.error = ""
            self.stop.clear()
            self.running = True
        Thread(target=self.run, daemon=True).start()

    def cancel(self):
        self.stop.set()

    def check_cancel(self):
        if self.stop.is_set() or self.closed.is_set():
            raise Cancelled()

    def receive(self, ws):
        started = time.monotonic()
        while True:
            if self.closed.is_set():
                raise Cancelled()
            try:
                response = ws.recv(timeout=.2)
                if isinstance(response, str):
                    raise ValueError(response)
                return msgpack_numpy.unpackb(response)
            except TimeoutError:
                if time.monotonic() - started > 120:
                    raise TimeoutError("模型请求超过 120 秒")

    def segment(self, source, episode, port):
        table = pq.read_table(source.path_for(episode),
                              columns=["observation.state", "task_index"]).to_pylist()
        tasks = {row["task_index"]: row["task"] for row in
                 map(json.loads, (source.root / "meta/tasks.jsonl").read_text().splitlines())}
        predictions = []
        with ExitStack() as stack:
            ws = stack.enter_context(self.connection(port))
            self.receive(ws)
            containers = [stack.enter_context(av.open(str(source.video_path(episode, camera)))) for camera in CAMERAS]
            readers = [iter(container.decode(video=0)) for container in containers]
            decoded = -1
            frames = None
            index = 0
            while index < len(table):
                self.check_cancel()
                while decoded < index:
                    self.check_cancel()
                    frames = [next(reader).to_ndarray(format="rgb24") for reader in readers]
                    decoded += 1
                images = {key: image.transpose(2, 0, 1) for key, image in
                          zip(("hand_left", "top_head", "hand_right"), frames)}
                payload = {"state": np.asarray(table[index]["observation.state"], dtype=np.float32),
                           "images": images, "prompt": tasks[int(table[index]["task_index"])]}
                ws.send(msgpack_numpy.packb(payload))
                response = self.receive(ws)
                actions = np.asarray(response["actions"], dtype=np.float64)
                if actions.ndim != 2 or not len(actions) or actions.shape[1] != 14 or not np.isfinite(actions).all():
                    raise ValueError(f"模型 action 格式无效 {actions.shape}")
                count = min(len(actions), len(table) - index)
                predictions.extend(actions[:count, source.indices].tolist())
                index += count
        return predictions

    def run(self):
        episode = None
        try:
            for episode in self.ids:
                self.check_cancel()
                with self.lock:
                    if episode in self.results:
                        continue
                    self.states[episode] = "running"
                solid = self.segment(self.dataset.solid, episode, 8002)
                liquid = self.segment(self.dataset.liquid, episode, 8001)
                self.check_cancel()
                prediction = np.asarray(solid + liquid).T
                reference = np.asarray(json.loads(self.dataset.trajectory(episode))["reference"])
                if not np.isfinite(reference).all() or prediction.shape != reference.shape:
                    raise ValueError("参考与推理轨迹长度或数值无效")
                rmse = np.sqrt(np.mean((prediction - reference) ** 2, axis=1))
                result = {"prediction": prediction.tolist(), "rmse": rmse.tolist(),
                          "overall": float(np.sqrt(np.mean(rmse[:6] ** 2)))}
                with self.lock:
                    self.results[episode] = result
                    self.states[episode] = "completed"
        except Cancelled:
            with self.lock:
                if episode is not None and self.states.get(episode) == "running":
                    self.states[episode] = "pending"
        except Exception as error:
            with self.lock:
                self.error = f"第 {episode} 条推理失败：{error}"
                if episode is not None:
                    self.states[episode] = "pending"
        finally:
            with self.lock:
                self.running = False
