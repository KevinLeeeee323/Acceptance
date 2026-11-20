"""Inspect once per episode, in windows 0–69, 70–79, ...; select 50 passing episodes."""
import csv
import json
import time
import traceback
from pathlib import Path
from random import Random

import numpy as np

from export_results import now, write_json
from inference import Inference
from paired import CACHE, CAMERAS, PairedDataset
from trajectories import DATASET, default_liquid

ROOT = Path(__file__).resolve().parent / "OverAllInpsect"
SELECTION_SEED = 42
ORDER_SEED = 43
THRESHOLD = .1


def finalize_selection(state, upper):
    accepted = [row for row in state["records"] if row["accepted"] and row["episode"] <= upper]
    if len(accepted) < 50:
        raise ValueError(f"编号 0–{upper} 内只有 {len(accepted)} 条合格数据")
    eligible_ids = sorted(row["episode"] for row in accepted)
    selected = sorted(Random(SELECTION_SEED).sample(eligible_ids, 50))
    ordered = selected[:]
    Random(ORDER_SEED).shuffle(ordered)
    lookup = {row["episode"]: row for row in accepted}
    manifest = {
        "schema_version": 1, "created_at": now(), "threshold_rad": THRESHOLD,
        "selection_seed": SELECTION_SEED, "order_seed": ORDER_SEED,
        "eligible_ids": eligible_ids, "selected_ids": selected, "ordered_ids": ordered,
        "solid_dataset": str(DATASET), "liquid_dataset": str(default_liquid()),
        "selection": "uniform random sample of 50 qualified episodes, then independently shuffled display order",
        "selection_algorithm": "random.Random(selection_seed).sample(sorted(eligible_ids), 50)",
        "ordering_algorithm": "random.Random(order_seed).shuffle(sorted(selected_ids))",
        "inspected_count": len(state["records"]),
        "inspected_through": max(row["episode"] for row in state["records"]),
        "selection_scope": {"start": 0, "end": upper},
        "records": [lookup[episode] for episode in selected],
    }
    write_json(ROOT / "selected_samples.json", manifest)
    with (ROOT / "合格清单.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream, lineterminator="
")
        writer.writerow(["展示顺序", "原始条目编号", "当次六关节总体RMSE_rad"])
        for index, episode in enumerate(ordered, 1):
            writer.writerow([index, episode, lookup[episode]["overall"]])
    return manifest


def add_reference_diagnostics(state, dataset):
    for exception in state["exceptions"]:
        if exception["type"] != "extreme_values":
            continue
        reference = np.asarray(json.loads(dataset.trajectory(exception["episode"]))["reference"])
        exception["reference_max_abs_joint"] = float(abs(reference[:6]).max())
        exception["reference_outlier_points"] = [
            {"joint_id": int(j + 1), "frame": int(t), "value_rad": float(reference[j, t]),
             "phase": "Solid" if t < 2050 else "Liquid",
             "local_frame": int(t if t < 2050 else t - 2050)}
            for j, t in np.argwhere(abs(reference[:6]) > 100)
        ]
    write_json(ROOT / "异常情况.json", state["exceptions"])


def main():
    ROOT.mkdir(exist_ok=True)
    path = ROOT / "inspection.json"
    state = json.loads(path.read_text()) if path.exists() else {
        "created_at": now(), "threshold_rad": THRESHOLD,
        "selection_seed": SELECTION_SEED, "order_seed": ORDER_SEED,
        "records": [], "exceptions": [], "status": "running",
        "solid_dataset": str(DATASET), "liquid_dataset": str(default_liquid()),
        "outlier_rule": "max_abs_joint_action > 100 rad, max_abs_gripper_action > 1 m, or overall RMSE >= 1 rad",
    }
    if state["status"] in ("complete", "exhausted"):
        print("Inspection already finished:", state["status"], flush=True)
        return
    ds = PairedDataset(DATASET, default_liquid(), CACHE)
    engine = Inference(ds)
    print(f"Shared episodes 0–{len(ds.episodes)-1}; valid pairs {len(ds.ids)}", flush=True)
    processed = {row["episode"] for row in state["records"]}
    upper = min(69, len(ds.episodes) - 1)
    lower = 0
    try:
        while lower < len(ds.episodes):
            for episode in range(lower, upper + 1):
                if episode in processed:
                    continue
                started = time.monotonic()
                row = {"episode": episode, "started_at": now()}
                try:
                    if episode not in ds.ids:
                        raise ValueError("文件缺失或长度不符合 2050+1100 帧")
                    reference = np.asarray(json.loads(ds.trajectory(episode))["reference"])
                    for camera in CAMERAS:
                        ds.video(episode, camera)
                    solid = engine.segment(ds.solid, episode, 8002)
                    liquid = engine.segment(ds.liquid, episode, 8001)
                    prediction = np.asarray(solid + liquid).T
                    if prediction.shape != (7, 3150) or not np.isfinite(prediction).all() or not np.isfinite(reference).all():
                        raise ValueError("轨迹形状或数值无效")
                    rmse = np.sqrt(np.mean((prediction - reference) ** 2, axis=1))
                    overall = float(np.sqrt(np.mean(rmse[:6] ** 2)))
                    result = {"prediction": prediction.tolist(), "rmse": rmse.tolist(), "overall": overall}
                    write_json(ROOT / "results" / f"episode_{episode:06d}.json", result)
                    row.update(overall=overall, rmse=rmse.tolist(), accepted=overall < THRESHOLD,
                               max_abs_joint=float(np.abs(prediction[:6]).max()),
                               max_abs_gripper=float(np.abs(prediction[6]).max()),
                               max_abs_reference_joint=float(np.abs(reference[:6]).max()),
                               max_abs_reference_gripper=float(np.abs(reference[6]).max()))
                    if row["max_abs_joint"] > 100 or row["max_abs_gripper"] > 1 or overall >= 1:
                        state["exceptions"].append({**row, "type": "extreme_values"})
                except Exception as error:
                    row.update(accepted=False, error=str(error))
                    state["exceptions"].append({**row, "type": "inference_failure", "traceback": traceback.format_exc()})
                row.update(finished_at=now(), elapsed_seconds=round(time.monotonic()-started, 3))
                state["records"].append(row)
                state["updated_at"] = now()
                write_json(path, state)
                write_json(ROOT / "异常情况.json", state["exceptions"])
                accepted = sum(record["accepted"] for record in state["records"])
                print(f"episode={episode} rmse={row.get('overall')} accepted={accepted} inspected={len(state['records'])} error={row.get('error','')}", flush=True)
            accepted = [row for row in state["records"] if row["accepted"] and row["episode"] <= upper]
            if len(accepted) >= 50:
                finalize_selection(state, upper)
                state["status"] = "complete"
                break
            lower = upper + 1
            upper = min(upper + 10, len(ds.episodes) - 1)
        else:
            state["status"] = "exhausted"
        state["finished_at"] = now()
        add_reference_diagnostics(state, ds)
        write_json(path, state)
        print("FINISHED", state["status"], flush=True)
    finally:
        engine.closed.set()


if __name__ == "__main__":
    main()
