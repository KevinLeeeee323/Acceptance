"""Serve the trajectory review UI and the local Parquet API."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
from threading import Thread
from urllib.parse import parse_qs, urlsplit

import av

from paired import CACHE, CAMERAS, DEFAULT_SEED, FPS, PairedDataset
from trajectories import DATASET, default_liquid
from inference import Inference
from atomic_actions import load_actions
from export_results import RunRecords

WEB = Path(__file__).resolve().parent / "web"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, address, dataset):
        self.dataset = dataset
        self.actions = load_actions(dataset.cache)
        self.inference = Inference(dataset)
        self.records = RunRecords(self.inference, dataset, self.actions)
        self.exit_token = secrets.token_urlsafe(32)
        super().__init__(address, ReviewHandler)


class ReviewHandler(BaseHTTPRequestHandler):
    def send_bytes(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_bytes(status, body, "application/json; charset=utf-8")

    def do_GET(self):
        url = urlsplit(self.path)
        path = url.path
        dataset = self.server.dataset
        if path in ASSETS:
            filename, content_type = ASSETS[path]
            body = (WEB / filename).read_bytes()
            if path == "/":
                body = body.replace(b"__EXIT_TOKEN__", self.server.exit_token.encode("ascii"))
            self.send_bytes(HTTPStatus.OK, body, content_type)
            return
        if path == "/api/reviews":
            with self.server.records.lock:
                self.send_json(HTTPStatus.OK, {"reviews": self.server.records.reviews, **self.server.records.status()})
            return
        if path == "/api/inference":
            self.send_json(HTTPStatus.OK, {**self.server.inference.snapshot(), **self.server.records.status()})
            return
        result_match = re.fullmatch(r"/api/inference/([0-9]{1,2})", path)
        if result_match:
            with self.server.inference.lock:
                result = self.server.inference.results.get(int(result_match[1]))
            self.send_json(HTTPStatus.OK, result)
            return
        if path == "/api/samples":
            self.send_json(HTTPStatus.OK, {
                "dataset": [dataset.solid.root.name, dataset.liquid.root.name],
                "fps": FPS,
                "joints": dataset.solid.joints,
                "samples": dataset.samples,
                "cameras": CAMERAS,
                "defaultSeed": DEFAULT_SEED,
                "atomicActions": self.server.actions,
            })
            return
        if path == "/api/batch":
            params = parse_qs(url.query)
            try:
                count = int(params["count"][0])
                seed = int(params["seed"][0])
                previous = [int(value) for value in params["previous"][0].split(",")] if "previous" in params else []
                ids = dataset.sample(count, seed, previous)
            except (KeyError, ValueError):
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "抽样条数或随机种子无效。"})
            else:
                self.send_json(HTTPStatus.OK, {"ids": ids})
            return
        match = re.fullmatch(r"/api/samples/([0-9]{1,2})(?:/video/(cam_[a-z_]+)\.mp4)?", path)
        if match:
            episode = int(match[1])
            camera = match[2]
            if episode in dataset.ids and (camera is None or camera in CAMERAS):
                try:
                    if camera:
                        video = dataset.video(episode, camera)
                    else:
                        body = dataset.trajectory(episode)
                except FileNotFoundError:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": f"第 {episode} 条配对数据文件缺失。"})
                except (OSError, ValueError, KeyError, RuntimeError, av.FFmpegError) as error:
                    self.log_error("Episode %d (%s): %s", episode, camera or "action", error)
                    self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"第 {episode} 条数据拼接失败。"})
                else:
                    if camera:
                        self.send_video(video)
                    else:
                        self.send_bytes(HTTPStatus.OK, body, "application/json; charset=utf-8")
                return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "找不到请求的资源。"})

    def send_video(self, video):
        with video.open("rb") as stream:
            size = video.stat().st_size
            start, end = 0, size - 1
            status = HTTPStatus.OK
            if requested := self.headers.get("Range"):
                interval = re.fullmatch(r"bytes=([0-9]{0,20})-([0-9]{0,20})", requested)
                if interval and any(interval.groups()):
                    first, last = interval.groups()
                    if first:
                        start = int(first)
                        end = min(int(last), end) if last else end
                    else:
                        start = max(0, size - int(last))
                if not interval or not any(interval.groups()) or start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT

            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-cache")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            stream.seek(start)
            remaining = end - start + 1
            try:
                while remaining:
                    chunk = stream.read(min(remaining, 256 * 1024))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_POST(self):
        path = urlsplit(self.path).path
        if path in ("/api/reviews", "/api/export"):
            try:
                if path == "/api/reviews":
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 2_000_000:
                        raise ValueError("保存内容过大或为空")
                    payload = self.server.records.update_reviews(json.loads(self.rfile.read(size)))
                else:
                    payload = self.server.records.export()
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            else:
                self.send_json(HTTPStatus.OK, payload)
            return
        if path in ("/api/inference/start", "/api/inference/stop"):
            try:
                if path.endswith("/start"):
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 4096:
                        raise ValueError("请求大小无效")
                    data = json.loads(self.rfile.read(size))
                    self.server.records.begin(data["ids"], data["seed"])
                else:
                    self.server.inference.cancel()
            except (ValueError, KeyError, TypeError) as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            else:
                self.send_json(HTTPStatus.OK, {**self.server.inference.snapshot(), **self.server.records.status()})
            return
        if path != "/api/shutdown":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "找不到请求的资源。"})
            return
        provided_token = self.headers.get("X-Exit-Token", "").encode("utf-8")
        if not secrets.compare_digest(provided_token, self.server.exit_token.encode("ascii")):
            self.send_json(HTTPStatus.FORBIDDEN, {"error": "退出请求失效，请刷新页面后重试。"})
            return
        with self.server.records.lock:
            if self.server.records.started and not self.server.records.verified():
                self.send_json(HTTPStatus.CONFLICT, {"error": "没有核验通过的最新导出结果，请先完成全部推理并导出后再退出。"})
                return
        self.server.inference.cancel()
        self.server.inference.closed.set()
        self.send_json(HTTPStatus.OK, {"status": "stopping"})
        self.wfile.flush()
        Thread(target=self.server.shutdown, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--liquid-dataset", type=Path, default=default_liquid())
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--restore-run", type=Path, help="恢复已完成推理的运行快照")
    args = parser.parse_args()
    try:
        dataset = PairedDataset(args.dataset, args.liquid_dataset, args.cache_dir)
        with ReviewServer((args.host, args.port), dataset) as server:
            if args.restore_run:
                server.records.restore(json.loads(args.restore_run.read_text(encoding="utf-8")))
            print(f"验收 UI  http://{args.host}:{server.server_port}/", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"启动失败：{error}\n")


if __name__ == "__main__":
    main()
