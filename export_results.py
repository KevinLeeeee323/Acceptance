"""Write and verify a complete acceptance export from immutable snapshots."""
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from threading import RLock
import uuid
import zipfile
from zoneinfo import ZoneInfo

import numpy as np

from paired import temporary_output

ROOT = Path(__file__).resolve().parent
TIMEZONE = ZoneInfo("Asia/Shanghai")
INCIDENT_TYPES = ("机械臂故障", "断电", "推理模型服务端故障", "验收 UI 脚本异常", "其他")


def now():
    return datetime.now(TIMEZONE).isoformat(timespec="seconds")


def write_json(path, value):
    with temporary_output(path) as output:
        output.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
        return checksum.hexdigest()


class RunRecords:
    def __init__(self, inference, dataset, actions, logs=None):
        self.inference, self.dataset, self.actions = inference, dataset, actions
        self.logs = Path(logs) if logs else ROOT / "logs"
        self.created_at = now()
        self.run_id = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        self.lock = RLock()
        self.reviews = {}
        self.active_key = None
        self.started = False
        self.revision = 0
        self.export_revision = -1
        self.last_export = None
        self.directory = self.logs / self.run_id

    def status(self):
        return {"runId": self.run_id, "started": self.started,
                "exported": self.export_revision == self.revision and self.last_export is not None,
                "export": self.last_export}

    def begin(self, ids, seed):
        if type(seed) is not int or seed < 42:
            raise ValueError("批次种子无效")
        key = json.dumps([seed, ids], separators=(",", ":"))
        with self.lock:
            if self.started and key != self.active_key and not self.verified():
                raise ValueError("请先完成当前运行并导出，再开始另一批推理")
            self.inference.completed_callback = self.save_result
            self.inference.start(ids)
            self.active_key = key
            self.started = True
            self.revision += 1
            self.save()

    def restore(self, snapshot):
        ids = snapshot["state"]["ids"]
        seed = snapshot.get("seed", 42)
        if (not ids or len(ids) != len(set(ids)) or any(type(i) is not int or i not in self.dataset.ids for i in ids)
                or type(seed) is not int or seed < 42):
            raise ValueError("恢复文件的样本名单或种子无效")
        results = {}
        for episode in ids:
            result = snapshot["results"][str(episode)]
            prediction = np.asarray(result["prediction"], dtype=float)
            reference = np.asarray(json.loads(self.dataset.trajectory(episode))["reference"])
            if prediction.shape != (7, 3150) or not np.isfinite(prediction).all():
                raise ValueError("恢复文件的推理轨迹无效")
            rmse = np.sqrt(np.mean((prediction - reference) ** 2, axis=1))
            if not np.allclose(rmse, result["rmse"], rtol=1e-9, atol=1e-10):
                raise ValueError("恢复文件与当前参考轨迹不匹配")
            results[episode] = {"prediction": prediction.tolist(), "rmse": rmse.tolist(),
                                "overall": float(np.sqrt(np.mean(rmse[:6] ** 2)))}
        self.active_key = json.dumps([seed, ids], separators=(",", ":"))
        self.started = True
        self.revision += 1
        self.reviews = snapshot.get("reviews", {})
        with self.inference.lock:
            self.inference.ids = ids
            self.inference.results = results
            self.inference.states = {i: "completed" for i in ids}
        self.inference.completed_callback = self.save_result
        self.save()
        for episode, result in results.items():
            self.save_result(episode, result)

    def save(self):
        if self.started:
            write_json(self.dataset.cache / "run-state" / self.run_id / "评定缓存.json", {
                "run_id": self.run_id, "created_at": self.created_at,
                "active_batch": self.active_key, "reviews": self.reviews,
                "revision": self.revision,
            })

    def save_result(self, episode, result):
        write_json(self.directory / "results" / f"episode_{episode:06d}.json", result)

    def update_reviews(self, records):
        if not isinstance(records, dict):
            raise ValueError("评定记录无效")
        # Only accept the bounded, documented fields exported below.
        for key, record in records.items():
            seed, ids = json.loads(key)
            if type(seed) is not int or not isinstance(ids, list) or not ids or any(i not in self.dataset.ids for i in ids):
                raise ValueError("评定批次无效")
            if not isinstance(record.get("notes", ""), str) or not isinstance(record.get("items", {}), dict):
                raise ValueError("备注或评定格式无效")
            for episode, item in record.get("items", {}).items():
                if int(episode) not in ids or item.get("choice") not in ("success", "failure") or type(item.get("confirmed")) is not bool:
                    raise ValueError("评定条目无效")
            for episode, incident in record.get("incidents", {}).items():
                if int(episode) not in ids or incident.get("type") not in (*INCIDENT_TYPES, ""):
                    raise ValueError("异常现象无效")
                for field in ("reason", "occurred_at", "handler", "follow_up", "other", "updated_at"):
                    if not isinstance(incident.get(field, ""), str):
                        raise ValueError("异常记录格式无效")
                if incident.get("occurred_at"):
                    datetime.strptime(incident["occurred_at"], "%Y-%m-%dT%H:%M")
        with self.lock:
            if records != self.reviews:
                active_changed = self.reviews.get(self.active_key) != records.get(self.active_key)
                self.reviews = json.loads(json.dumps(records, ensure_ascii=False))
                if self.started and active_changed:
                    self.revision += 1
                self.save()
        return self.status()

    def verified(self):
        if not self.status()["exported"]:
            return False
        try:
            folder = Path(self.last_export["folder"])
            if digest(folder / "batch_summary.json") != self.last_export["summary_sha256"]:
                return False
            summary = json.loads((folder / "batch_summary.json").read_text())
            for name, expected in summary["files"].items():
                if digest(folder / name) != expected:
                    return False
            archive = Path(self.last_export["archive"])
            if digest(archive) != self.last_export["sha256"]:
                return False
            with zipfile.ZipFile(archive) as handle:
                if handle.testzip() is not None:
                    return False
            return True
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            return False

    def export(self):
        with self.lock:
            with self.inference.lock:
                ids = self.inference.ids[:]
                if not self.started or self.inference.running or not ids or any(i not in self.inference.results for i in ids):
                    raise ValueError("全部抽样数据推理完成后才能导出")
                results = dict(self.inference.results)
            seed, expected_ids = json.loads(self.active_key)
            if expected_ids != ids:
                raise ValueError("当前批次与推理结果不匹配")
            record = self.reviews.get(self.active_key, {})
            items, incidents = record.get("items", {}), record.get("incidents", {})
            stamp = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S_%f")
            name = stamp + "_验收结果"
            self.directory.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix="export-", dir=self.directory) as temp:
                folder = Path(temp) / name
                folder.mkdir()
                (folder / "plots").mkdir()
                (folder / "results").mkdir()
                shutil.copyfile(ROOT / "导出文件说明.md", folder / "导出文件说明.md")
                rows = []
                exported_incidents = {}
                for index, episode in enumerate(ids, 1):
                    result = results[episode]
                    write_json(folder / "results" / f"episode_{episode:06d}.json", result)
                    reference = json.loads(self.dataset.trajectory(episode))["reference"]
                    plot_trajectory(folder / "plots" / f"sample_{index:03d}.png", episode, reference, result["prediction"], self.actions)
                    item = items.get(str(episode), {})
                    incident = incidents.get(str(episode), {})
                    if any(incident.get(field) for field in ("type", "other", "reason", "occurred_at", "handler", "follow_up")):
                        exported_incidents[str(index - 1)] = {
                            **{field: incident.get(field, "") for field in
                               ("type", "other", "reason", "occurred_at", "handler", "follow_up", "updated_at")},
                            "sample_id": index, "episode": episode,
                        }
                    rows.append({
                        "sample_id": f"SAMPLE {index:02d}", "source_item": f"第 {episode} 条",
                        "original_episode": episode, "solid_model_version": "solid_awbc_99999",
                        "liquid_model_version": "liquid_awbc_119999", "inference_status": "completed",
                        "review_result": item.get("choice", "") if item.get("confirmed") else "",
                        "review_confirmed": bool(item.get("confirmed")), "has_incident": str(index - 1) in exported_incidents,
                        "paired_points_solid": 2050, "paired_points_liquid": 1100, "paired_points_total": 3150,
                        "six_joint_rmse_rad": result["overall"],
                        **{f"J{i + 1:02d}_rmse_{'m' if i == 6 else 'rad'}": value for i, value in enumerate(result["rmse"])},
                    })
                with (folder / "samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                write_json(folder / "异常记录.json", {"schema_version": 1, "timezone": "Asia/Shanghai", "records": exported_incidents})
                write_json(folder / "评定结果.json", record)
                (folder / "备注.txt").write_text(record.get("notes", ""), encoding="utf-8")
                write_json(folder / "原子动作.json", self.actions)
                shutil.copyfile(ROOT / "inference_configuration.json", folder / "推理配置.json")
                (folder / "轨迹对齐方法.txt").write_text(
                    "采样频率为30 Hz。同编号固体数据的2050帧右臂 action 与液体数据的1100帧左臂 action，按同名关节及爪夹拼接为3150个采样点，时长105秒。"
                    "模型每次接收动作段起点的参考观测，返回一段14维 actions，采用该段动作后推进至下一段起点，任务边界截断。"
                    "两段分别取右臂、左臂，按相同帧索引与参考 action 对齐，七维逐一计算全部3150点RMSE，前六个关节RMSE的均方根作为总体rad指标，爪夹以m单列。", encoding="utf-8")
                with (folder / "有效动作判据.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["编号", "任务", "动作名称", "起始帧", "终止帧_不含", "描述", "人工核验判据"])
                    for action in self.actions:
                        writer.writerow([action["number"], action["phase"], action["name"], action["start"], action["end"],
                                         action["description"], "核对参考轨迹与视频动作对应关系，结合模型action拟合情况及人工评定确认动作有效性。"])
                success = sum(bool(v.get("confirmed")) and v.get("choice") == "success" for k, v in items.items() if int(k) in ids)
                confirmed = sum(bool(v.get("confirmed")) for k, v in items.items() if int(k) in ids)
                files = {str(path.relative_to(folder)): digest(path) for path in folder.rglob("*") if path.is_file()}
                write_json(folder / "batch_summary.json", {
                    "schema_version": 1, "run_id": self.run_id, "seed": seed, "sample_ids": ids,
                    "created_at": self.created_at, "exported_at": now(), "sample_count": len(ids),
                    "completed_count": len(ids), "confirmed_count": confirmed, "success_count": success,
                    "success_rate": success / len(ids), "files": files,
                })
                archive = Path(shutil.make_archive(str(Path(temp) / name), "zip", root_dir=temp, base_dir=name))
                final_folder, final_archive = self.directory / name, self.directory / (name + ".zip")
                shutil.move(folder, final_folder)
                archive.replace(final_archive)
            self.last_export = {"folder": str(final_folder), "archive": str(final_archive), "sha256": digest(final_archive),
                                "summary_sha256": digest(final_folder / "batch_summary.json")}
            self.export_revision = self.revision
            if not self.verified():
                self.export_revision = -1
                raise ValueError("导出文件核验失败，请重新导出")
            self.save()
            return self.last_export


def plot_trajectory(path, episode, reference, prediction, actions):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/acceptance-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    figure = Figure(figsize=(16, 16), layout="constrained")
    FigureCanvasAgg(figure)
    all_axes = figure.subplots(8, 1, gridspec_kw={"height_ratios": [1] * 7 + [1.5]})
    axes = all_axes[:7]
    legend = all_axes[-1]
    legend.set_axis_off()
    for index, action in enumerate(actions):
        column, row = index % 4, index // 4
        legend.text(column * .25, .9 - row * .24,
                    f"{action['number']:02d} {action['name']}",
                    color=action["color"], fontproperties=font, fontsize=10,
                    transform=legend.transAxes)
    for i, ax in enumerate(axes):
        for action in actions:
            ax.axvspan(action["start"], action["end"], color=action["color"], alpha=.16)
        for start, end in ((0, 2050), (2050, 3150)):
            frames = np.arange(start, end)
            ax.plot(frames, reference[i][start:end], color="#006b56", linewidth=1,
                    label="参考 action" if start == 0 else None)
            ax.plot(frames, prediction[i][start:end], color="#d34242", linewidth=1,
                    label="推理 action" if start == 0 else None)
        ax.axvline(2050, color="#344842", linestyle="--", linewidth=1)
        ax.set_ylabel(f"J{i + 1:02d} ({'m' if i == 6 else 'rad'})")
        ax.grid(alpha=.2)
    axes[0].legend(prop=font)
    axes[0].set_title(f"第 {episode} 条 · 七维轨迹对比", fontproperties=font)
    axes[-1].set_xlabel("帧索引", fontproperties=font)
    figure.savefig(path, dpi=150)
    from PIL import Image
    with Image.open(path) as image:
        image.verify()
    figure.clear()
