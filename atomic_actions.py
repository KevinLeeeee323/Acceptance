"""Load the shared, manually annotated episode-40 action template."""
import hashlib
import json
from pathlib import Path

from paired import CACHE, SOLID_FRAMES, LIQUID_FRAMES, temporary_output

TEMPLATE = Path(__file__).resolve().parent / "AtomActionsInfo.json"

# Variations of the reference figure's cyan, green, ochre and purple families.
COLORS = (
    "#24869d", "#5c914c", "#b58026", "#8961a8",
    "#23758b", "#4d844e", "#a76a24", "#785495",
    "#397f9b", "#668c3d", "#b28232", "#956ab0",
    "#307f91", "#4c905d", "#b17235", "#765e9f",
)


def load_actions(cache=CACHE):
    contents = TEMPLATE.read_bytes()
    digest = hashlib.sha256(b"atomic-actions-v1:" + contents).hexdigest()[:24]
    path = Path(cache) / "atomic-actions" / digest / "annotations.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    template = json.loads(contents)
    actions = []
    for phase, offset, length in (
        ("Solid", 0, SOLID_FRAMES), ("Liquid", SOLID_FRAMES, LIQUID_FRAMES)
    ):
        previous_end = 0
        for key, item in sorted(template[phase].items(), key=lambda pair: int(pair[0])):
            start, end = int(item["StartFrameID"]), int(item["EndFrameID"])
            if not previous_end <= start < end <= length:
                raise ValueError(f"{phase} 原子动作 {key} 的区间重叠或超出数据长度")
            previous_end = end
            actions.append({
                "number": len(actions) + 1,
                "phase": phase,
                "name": item["AtomActionName"],
                "description": item["Description"],
                "localStart": start,
                "localEnd": end,
                "start": offset + start,
                "end": offset + end,
                "color": COLORS[(len(actions)) % len(COLORS)],
            })
    with temporary_output(path) as output:
        output.write_text(json.dumps(actions, ensure_ascii=False), encoding="utf-8")
    return actions
