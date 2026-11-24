"""Why did a frame measure the way it did? Measures frames exactly as Prepare does (same decoding, same input,
same regional vote) and writes, for each frame, every region's vote and brightness, which rule decided the
frame's change, and for the added-light (flare) rule which of its requirements held.

    python benchmarks/diagnose_votes.py "C:/Users/James/Pictures/sunset-keys" 118 130 > votes.txt

Frame numbers are 1-based, as in the app. Send votes.txt back.
"""

import sys
from pathlib import Path

import numpy as np

from harbium_timelapse import measure as M
from harbium_timelapse.sequencer import load_sequence


def explain(previous: np.ndarray, current: np.ndarray) -> list[str]:
    out = []
    ok = (previous > M.BLACK) & (current > M.BLACK) & (previous < M.CLIPPED) & (current < M.CLIPPED)
    out.append(f"  usable pixels (not black, not clipped, in both frames): {ok.mean():.0%}")
    if ok.sum() < M.MIN_STEADY_PIXELS:
        return out + ["  too few usable pixels: falls back to the tone measure"]
    d = np.zeros(previous.shape)
    d[ok] = np.log2(current[ok]) - np.log2(previous[ok])
    votes, levels = M._regional_changes(previous, current, ok, d)
    out.append(f"  voting regions: {len(votes)} of {M.REGIONS[0] * M.REGIONS[1]}")
    if len(votes) < M.MIN_VOTERS:
        return out + ["  too few voting regions: whole-frame ratio"]
    v, lv = np.array([x for x, _ in votes]), np.array(levels)
    w = np.array([n for _, n in votes], dtype=np.float64)
    spread = np.percentile(v, 90) - np.percentile(v, 10)
    corr = abs(np.corrcoef(lv, v)[0, 1]) if np.ptp(lv) > 0 and np.ptp(v) > 0 else 0.0
    slope = np.polyfit(lv, v, 1, w=np.sqrt(w))[0] if np.ptp(lv) > 0 else 0.0
    sign = 1.0 if np.median(v) >= 0 else -1.0
    checks = [("votes spread (10th-90th)", f"{spread:.2f} stops", spread > M.ADDED_SPREAD, f"> {M.ADDED_SPREAD}"),
              ("brightness range", f"{np.ptp(lv):.2f} stops", np.ptp(lv) >= M.ADDED_RANGE, f">= {M.ADDED_RANGE}"),
              ("changes follow brightness", f"{corr:.2f}", corr >= M.ADDED_FOLLOW, f">= {M.ADDED_FOLLOW}"),
              ("shrinking with brightness", f"{sign * slope:+.2f}", sign * slope < M.ADDED_SLOPE,
               f"< {M.ADDED_SLOPE}")]
    out.append("  added-light (flare) rule: " + ("APPLIES" if all(c[2] for c in checks) else "does not apply"))
    for name, value, held, need in checks:
        out.append(f"     {'ok ' if held else 'NO '} {name}: {value} (needs {need})")
    out.append(f"  frame's change as measured: {M._vote(votes, levels):+.3f} stops "
               f"(plain middle vote would be {M._vote(votes):+.3f})")
    out.append("  regions, darkest to brightest: brightness (stops) -> change (stops), pixels")
    for k in np.argsort(lv):
        out.append(f"     {lv[k]:7.2f} -> {v[k]:+6.2f}   {int(w[k])}")
    return out


def main(folder: Path, first: int, last: int) -> None:
    seq = load_sequence(folder)
    print(f"{folder}: {len(seq)} frames; frames {first}-{last}, measured as Prepare does\n")
    previous = None
    for i in range(max(0, first - 2), min(len(seq), last)):
        image, _ = M._load_for_measuring((seq[i].path, "careful", None))
        current = M.steady_input(image)
        if previous is not None and i + 1 >= first:
            print(f"frame {i + 1} ({seq[i].name}) against frame {i}:")
            print("\n".join(explain(previous, current)), "\n", flush=True)
        previous = current


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
