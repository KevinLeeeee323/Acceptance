"""Pair episodes and cache a seven-dimensional trajectory and three lossless video joins."""

from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from tempfile import mkstemp
from threading import Lock

import av

from trajectories import Dataset

CAMERAS = ("cam_left_wrist", "cam_high", "cam_right_wrist")
SOLID_FRAMES = 2050
LIQUID_FRAMES = 1100
FPS = 30
CACHE = Path(__file__).resolve().parent / ".acceptance-cache"


class PairedDataset:
    def __init__(self, solid, liquid, cache):
        self.solid = Dataset(solid)
        left_joints = [f"left_{joint.removeprefix('right_')}" for joint in self.solid.joints]
        self.liquid = Dataset(liquid, arm="left", joint_names=left_joints)
        if any(source.info["fps"] != FPS for source in (self.solid, self.liquid)):
            raise ValueError("固体与液体数据都应为 30 Hz")
        self.cache = Path(cache)
        self.episodes = range(min(len(self.solid.episodes), len(self.liquid.episodes)))
        self.locks = {episode: Lock() for episode in self.episodes}
        self.samples = []
        for episode in self.episodes:
            if (self.solid.samples[episode]["frames"] != SOLID_FRAMES
                    or self.liquid.samples[episode]["frames"] != LIQUID_FRAMES):
                continue
            if all(source.video_path(episode, camera).is_file()
                   for source in (self.solid, self.liquid) for camera in CAMERAS):
                self.samples.append({"id": episode, "frames": SOLID_FRAMES + LIQUID_FRAMES})
        self.ids = [sample["id"] for sample in self.samples]
        if not self.ids:
            raise ValueError("两套数据集中没有文件齐全、帧数匹配的同编号配对样本")

    def load_selection(self):
        path = Path(__file__).resolve().parent / "OverAllInpsect/selected_samples.json"
        if not path.is_file():
            raise ValueError("尚未完成合格样本筛选，缺少 OverAllInpsect/selected_samples.json")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        ids = manifest["ordered_ids"]
        if len(ids) != 50 or len(set(ids)) != 50 or any(type(i) is not int or i not in self.ids for i in ids):
            raise ValueError("固定清单必须包含 50 条有效且不重复的配对数据")
        lookup = {row["episode"]: row for row in manifest["records"]}
        if any(i not in lookup or not 0 <= lookup[i]["overall"] < .1 for i in ids):
            raise ValueError("固定清单中的当次 RMSE 不符合 <0.1 rad")
        available = {sample["id"]: sample for sample in self.samples}
        self.ids = ids
        self.samples = [available[i] for i in ids]
        self.selection = manifest

    def cache_path(self, episode, name, sources, configuration=()):
        identity = ["paired-v1", *configuration]
        for source in sources:
            stat = source.stat()
            identity.append([str(source.resolve()), stat.st_size, stat.st_mtime_ns])
        digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
        return self.cache / f"episode_{episode:06d}" / digest / name

    def trajectory(self, episode):
        path = self.cache_path(episode, "trajectory.json",
                               [self.solid.path_for(episode), self.liquid.path_for(episode)],
                               [self.solid.indices, self.liquid.indices])
        with self.locks[episode]:
            if not path.is_file():
                solid = self.solid.trajectory(episode)
                liquid = self.liquid.trajectory(episode)
                if solid["frames"] != list(range(SOLID_FRAMES)) or liquid["frames"] != list(range(LIQUID_FRAMES)):
                    raise ValueError("源轨迹帧索引应从 0 开始且连续")
                frames = list(range(SOLID_FRAMES + LIQUID_FRAMES))
                payload = {
                    "id": episode,
                    "frames": frames,
                    "timestamps": [frame / FPS for frame in frames],
                    "splitFrame": SOLID_FRAMES,
                    "reference": [solid_series + liquid_series for solid_series, liquid_series in zip(solid["reference"], liquid["reference"])],
                }
                with temporary_output(path) as output:
                    output.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        return path.read_bytes()

    def video(self, episode, camera):
        sources = [source.video_path(episode, camera) for source in (self.solid, self.liquid)]
        path = self.cache_path(episode, f"{camera}.mp4", sources)
        with self.locks[episode]:
            if not path.is_file():
                with temporary_output(path) as output:
                    join_video(*sources, output)
        return path


@contextmanager
def temporary_output(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = mkstemp(prefix="building-", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    output = Path(filename)
    try:
        yield output
        output.replace(path)
    finally:
        output.unlink(missing_ok=True)


def video_format(stream):
    codec = stream.codec_context
    return codec.name, codec.extradata, stream.width, stream.height, stream.time_base


def join_video(solid_path, liquid_path, output_path):
    """Remux each source packet onto the shared 30 Hz timeline without changing its bytes."""
    with av.open(str(solid_path)) as solid, av.open(str(liquid_path)) as liquid:
        solid_stream = solid.streams.video[0]
        liquid_stream = liquid.streams.video[0]
        if video_format(solid_stream) != video_format(liquid_stream):
            raise ValueError("两段视频的编码参数不同，无法直接拼接")
        with av.open(str(output_path), "w", options={"movflags": "+faststart"}) as output:
            track = output.add_stream_from_template(solid_stream, opaque=True)
            offset = 0
            for container, stream, frames in (
                (solid, solid_stream, SOLID_FRAMES), (liquid, liquid_stream, LIQUID_FRAMES)
            ):
                ticks = Fraction(1, FPS) / stream.time_base
                if stream.average_rate != FPS or ticks.denominator != 1 or stream.start_time is None:
                    raise ValueError("视频没有有效的 30 Hz 时间轴")
                step = int(ticks)
                count = 0
                for packet in container.demux(stream):
                    if packet.size == 0:
                        continue
                    expected = stream.start_time + count * step
                    if (packet.pts != expected or packet.dts != expected or packet.duration != step
                            or (count == 0 and not packet.is_keyframe)):
                        raise ValueError("源视频帧时间戳不连续，或片段未从关键帧开始")
                    packet.pts = packet.dts = offset + count * step
                    packet.stream = track
                    packet.time_base = stream.time_base
                    output.mux(packet)
                    count += 1
                if count != frames:
                    raise ValueError(f"视频解复用得到 {count} 帧，预期 {frames} 帧")
                offset += frames * step
