"""Read the expert right-arm action trajectory for episodes 0–49."""

import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as parquet

DATASET = Path("/data2/organ_data_le/solid/measure_solid_0630")
LIQUID = Path("/data/organ_data_le/liquid/measure_liquid_full_0622")
EPISODES = range(50)


def default_liquid():
    alternate = Path("/data2") / LIQUID.relative_to("/data")
    return LIQUID if LIQUID.exists() or not alternate.exists() else alternate


class Dataset:
    def __init__(self, root, arm="right", joint_names=None):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text(encoding="utf-8"))
        names = self.info["features"]["action"]["names"][0]
        self.joints = list(joint_names) if joint_names is not None else [
            name for name in names if name.startswith(arm + "_")
        ]
        if len(self.joints) != 7 or len(set(self.joints)) != 7:
            raise ValueError(f"{self.root.name} 应包含 7 个 {arm} 臂 action 维度")
        self.indices = tuple(names.index(name) for name in self.joints)
        self.action_width = len(names)
        self.samples = []
        for episode in EPISODES:
            path = self.path_for(episode)
            self.samples.append({
                "id": episode,
                "frames": parquet.read_metadata(path).num_rows if path.is_file() else None,
            })

    def path_for(self, episode):
        return self.format_path("data_path", episode)

    def video_path(self, episode, camera):
        return self.format_path("video_path", episode, video_key=f"observation.images.{camera}")

    def format_path(self, template, episode, **fields):
        if episode not in EPISODES:
            raise ValueError("条目编号应在 0–49 之间")
        return self.root / self.info[template].format(
            episode_chunk=episode // self.info["chunks_size"],
            episode_index=episode,
            **fields,
        )

    def trajectory(self, episode):
        path = self.path_for(episode)
        table = parquet.ParquetFile(path).read(columns=["frame_index", "timestamp", "action"])
        rows = table.to_pylist()
        if not rows or any(row["frame_index"] is None for row in rows):
            raise ValueError("轨迹缺少帧索引")
        rows.sort(key=lambda row: row["frame_index"])
        frames = []
        timestamps = []
        reference = [[] for _ in self.indices]
        for row in rows:
            frame = row["frame_index"]
            timestamp = row["timestamp"]
            action = row["action"]
            if (timestamp is None or not math.isfinite(timestamp)
                    or (frames and (frame <= frames[-1] or timestamp < timestamps[-1]))
                    or action is None or len(action) != self.action_width):
                raise ValueError("轨迹的帧索引、时间戳或 action 格式无效")
            frames.append(frame)
            timestamps.append(timestamp)
            for series, index in zip(reference, self.indices):
                value = action[index]
                series.append(value if value is not None and math.isfinite(value) else None)
        return {
            "id": episode,
            "frames": frames,
            "timestamps": timestamps,
            "reference": reference,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=int, choices=EPISODES, metavar="0–49")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()
    dataset = Dataset(args.dataset)
    print(json.dumps({"joints": dataset.joints, **dataset.trajectory(args.episode)}))
