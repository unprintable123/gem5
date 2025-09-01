#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch launcher and plotter for gem5 Garnet sweeps (general X-axis) with enforced figure aspect.

Key ideas:
- X axis is any CLI flag you choose (e.g., --injectionrate, --vcs-per-vnet, --synthetic).
- X values can be a numeric range (start:end:step) or a discrete comma list (numbers or strings).
- Multiple curves defined via --grid (cartesian product) or --curves-json (explicit list).
- Parallel across curves; sequential within a curve (enables early-stop/clip for numeric X).
- Each run has its own outdir; stdout/stderr are persisted under each run folder.
- Stats are parsed from stats.txt and exported as CSV/JSON; plotting supports numeric or categorical X.
- Legend is placed outside the axes area to avoid occluding curves.
- Figure size/aspect is strictly enforced to a reasonable proportion (default 16:9, width=10 in).

Notes:
- Early-stop/clip is applied ONLY when X is numeric and increasing (meaningful "larger X" semantics).
- All comments are written in English as requested.
"""

from __future__ import annotations
import argparse, json, os, shlex, csv, itertools, importlib, subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------- Data models -----------------------------


@dataclass(frozen=True)
class CurveVariant:
    """A curve = base_args overridden by these flags; label_keys control legend label."""

    name: str
    overrides: Dict[str, str] = field(default_factory=dict)
    label_keys: Optional[
        List[str]
    ] = None  # if provided, only highlight these key diffs


@dataclass
class RunResult:
    """One (curve, x) outcome."""

    curve_name: str
    x_value: str  # textual X value (e.g., "0.10" or "uniform_random")
    outdir: str
    ok: bool
    reason: str
    stats: Dict[str, float] = field(default_factory=dict)
    x_is_num: bool = False  # whether x_value is numeric
    x_num: float | None = None  # numeric value if x_is_num is True


# ----------------------------- Utilities ------------------------------


def float_range(start: float, end: float, step: float) -> List[float]:
    """Inclusive float range with decimal stability."""
    if step <= 0:
        raise ValueError("step must be positive")
    n = int(round((end - start) / step))
    vals = [round(start + i * step, 10) for i in range(n + 1)]
    if vals and vals[-1] < end - 1e-9:
        vals.append(round(end, 10))
    return vals


def sanitize_val(v: Any) -> str:
    """Make a CLI/dir-safe token."""
    s = str(v).strip().replace("/", "_")
    return s


def build_outdir(
    root: Path, curve: CurveVariant, x_key: str, x_val: str
) -> Path:
    """Per-(curve, x) unique outdir path."""
    cname = sanitize_val(curve.name)
    xkey = x_key.lstrip("-")
    xval = sanitize_val(x_val)
    return root / cname / f"{xkey}_{xval}"


def dict_to_cli_flags(dd: Dict[str, str]) -> List[str]:
    """
    Convert {"--k": "v", "--flag-without-value": ""} into ["--k=v", "--flag-without-value"].
    Only keys starting with '-' are considered CLI flags; others ignored.
    """
    parts = []
    for k, v in dd.items():
        if not k.startswith("-"):
            continue
        if v == "" or v is None:
            parts.append(k)
        else:
            parts.append(f"{k}={v}")
    return parts


def read_stats(stats_path: Path) -> Dict[str, float]:
    """
    Parse gem5 stats.txt into a dict. Extract both fully-qualified and short keys under ruby.network.
    Returns only numeric values.
    """
    wanted_prefix = "system.ruby.network."
    out: Dict[str, float] = {}
    if not stats_path.exists():
        return out
    with stats_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name, val = parts[0], parts[1]
            try:
                v = float(val)
            except ValueError:
                continue
            out[name] = v
            if name.startswith(wanted_prefix):
                short = name[len(wanted_prefix) :]
                out[short] = v
    return out


def salient_label(
    base_overrides: Dict[str, str],
    curve: CurveVariant,
    prefer_keys: Optional[List[str]],
) -> str:
    """
    Build a compact legend label: highlight differences from base_overrides.
    If 'prefer_keys' provided (curve.label_keys or CLI --label-keys), only show those keys.
    """
    diffs = []
    keys = set(curve.overrides.keys())
    if prefer_keys:
        keys = [k for k in prefer_keys if k in keys]
    for k in keys:
        b = base_overrides.get(k, None)
        c = curve.overrides.get(k, None)
        if c != b:
            kdisp = k.lstrip("-")
            diffs.append(f"{kdisp}={c}")
    if not diffs:
        return curve.name
    return f"{curve.name} ({', '.join(diffs)})"


# --------------------------- Clipping policies -------------------------


class Clipper:
    """Strategy object. Return True to STOP further larger X for the SAME curve (numeric X only)."""

    def should_stop(self, records: List[RunResult]) -> bool:
        return False


class AbsThresholdClipper(Clipper):
    def __init__(self, y_key: str, threshold: float):
        self.y_key = y_key
        self.threshold = threshold

    def should_stop(self, records: List[RunResult]) -> bool:
        if not records or not records[-1].ok:
            return False
        y = records[-1].stats.get(self.y_key)
        return (y is not None) and (y >= self.threshold)


class RelToBaselineClipper(Clipper):
    def __init__(self, y_key: str, factor: float):
        self.y_key = y_key
        self.factor = factor

    def should_stop(self, records: List[RunResult]) -> bool:
        okrecs = [r for r in records if r.ok and self.y_key in r.stats]
        if len(okrecs) < 1:
            return False
        base = okrecs[0].stats[self.y_key]
        cur = okrecs[-1].stats[self.y_key]
        if base is None or cur is None:
            return False
        return cur >= base * self.factor


class MonotonicGrowthClipper(Clipper):
    """Stop if multiplicative jump between adjacent numeric X points is large."""

    def __init__(self, y_key: str, jump_factor: float = 2.0):
        self.y_key = y_key
        self.jump_factor = jump_factor

    def should_stop(self, records: List[RunResult]) -> bool:
        okrecs = [r for r in records if r.ok and self.y_key in r.stats]
        if len(okrecs) < 2:
            return False
        prev = okrecs[-2].stats[self.y_key]
        cur = okrecs[-1].stats[self.y_key]
        if prev is None or cur is None:
            return False
        return cur >= prev * self.jump_factor


class UserModuleClipper(Clipper):
    """
    Load user function `module_path:func_name`.
    The function signature: def fn(records: List[RunResult]) -> bool
    """

    def __init__(self, spec: str):
        mod, fn = spec.split(":")
        m = importlib.import_module(mod)
        self.fn = getattr(m, fn)

    def should_stop(self, records: List[RunResult]) -> bool:
        return bool(self.fn(records))


def make_clipper(args, y_key: str) -> Clipper:
    """Factory for clipper based on CLI."""
    if args.clip_module:
        return UserModuleClipper(args.clip_module)
    if args.clip == "none":
        return Clipper()
    if args.clip == "abs":
        return AbsThresholdClipper(y_key, args.clip_threshold)
    if args.clip == "rel":
        return RelToBaselineClipper(y_key, args.clip_factor)
    if args.clip == "mono":
        return MonotonicGrowthClipper(y_key, args.clip_jump_factor)
    return Clipper()


# --------------------------- Command assembly --------------------------


def build_gem5_cmd(
    gem5_bin: Path,
    config_py: Path,
    base_cfg: Dict[str, str],
    curve: CurveVariant,
    x_key: str,
    x_val: str,
    outdir: Path,
) -> List[str]:
    """
    Build the full gem5 command with --outdir and config flags.
    We set the sweep flag (x_key) to x_val for this run.
    """
    merged = dict(base_cfg)
    merged.update(curve.overrides)
    if not x_key.startswith("-"):
        raise ValueError(f"--x-flag must start with '-', got: {x_key}")
    merged[x_key] = x_val

    g5 = [str(gem5_bin), "--outdir", str(outdir)]
    cfg_flags = dict_to_cli_flags(merged)
    return g5 + [str(config_py)] + cfg_flags


# ------------------------------ Runners --------------------------------


def run_one(
    cmd: List[str],
    env: Optional[Dict[str, str]],
    timeout_sec: Optional[int],
    log_prefix: Optional[str] = None,
) -> Tuple[bool, str]:
    """Execute one gem5 run and persist stdout/stderr to files if log_prefix is given."""
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec if timeout_sec and timeout_sec > 0 else None,
            check=False,
            encoding="utf-8",
            errors="ignore",
        )
        if log_prefix:
            try:
                with open(
                    f"{log_prefix}.stdout.txt",
                    "w",
                    encoding="utf-8",
                    errors="ignore",
                ) as fo:
                    fo.write(proc.stdout or "")
                with open(
                    f"{log_prefix}.stderr.txt",
                    "w",
                    encoding="utf-8",
                    errors="ignore",
                ) as fe:
                    fe.write(proc.stderr or "")
            except Exception:
                pass
        if proc.returncode != 0:
            return False, f"nonzero_exit({proc.returncode})"
        return True, "ok"
    except subprocess.TimeoutExpired:
        if log_prefix:
            with open(
                f"{log_prefix}.stderr.txt",
                "a",
                encoding="utf-8",
                errors="ignore",
            ) as fe:
                fe.write("\n[TIMEOUT]\n")
        return False, "timeout"
    except Exception as e:
        if log_prefix:
            with open(
                f"{log_prefix}.stderr.txt",
                "a",
                encoding="utf-8",
                errors="ignore",
            ) as fe:
                fe.write(f"\n[EXCEPTION] {e}\n")
        return False, f"exception:{e}"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run_curve_sequential(
    args,
    base_cfg: Dict[str, str],
    curve: CurveVariant,
    x_values: List[str],
    x_is_num: bool,
    x_key: str,
    gem5_bin: Path,
    config_py: Path,
    out_root: Path,
    y_key: str,
    clipper: Clipper,
    env: Optional[Dict[str, str]],
) -> List[RunResult]:
    """
    Run one curve variant: sweep x_values IN ORDER.
    For numeric X, early-stop clipping can be triggered; for categorical X, clipping is disabled.
    """
    results: List[RunResult] = []

    def x_to_num(s: str) -> Optional[float]:
        try:
            return float(s)
        except Exception:
            return None

    for xv in x_values:
        outdir = build_outdir(out_root, curve, x_key, xv)
        stats_path = outdir / "stats.txt"

        if args.skip_existing and stats_path.exists():
            stats = read_stats(stats_path)
            ok = y_key in stats
            results.append(
                RunResult(
                    curve.name,
                    xv,
                    str(outdir),
                    ok,
                    "cached",
                    stats,
                    x_is_num=x_is_num,
                    x_num=(x_to_num(xv) if x_is_num else None),
                )
            )
            if x_is_num and clipper.should_stop(results):
                break
            continue

        ensure_dir(outdir)
        cmd = build_gem5_cmd(
            gem5_bin, config_py, base_cfg, curve, x_key, xv, outdir
        )

        if args.dry_run:
            print("[DRY] " + " ".join(shlex.quote(c) for c in cmd))
            results.append(
                RunResult(
                    curve.name,
                    xv,
                    str(outdir),
                    False,
                    "dry_run",
                    {},
                    x_is_num=x_is_num,
                    x_num=(x_to_num(xv) if x_is_num else None),
                )
            )
            continue

        ok, reason = run_one(
            cmd, env, args.timeout_sec, log_prefix=str(outdir / "gem5")
        )
        stats = read_stats(stats_path) if ok else {}
        if ok and (y_key not in stats):
            ok = False
            reason = f"no_metric:{y_key}"

        results.append(
            RunResult(
                curve.name,
                xv,
                str(outdir),
                ok,
                reason,
                stats,
                x_is_num=x_is_num,
                x_num=(x_to_num(xv) if x_is_num else None),
            )
        )

        if x_is_num and clipper.should_stop(results):
            break

        if reason == "timeout" and x_is_num:
            break

    return results


def parallel_run_all_curves(
    args,
    base_cfg: Dict[str, str],
    curves: List[CurveVariant],
    x_values: List[str],
    x_is_num: bool,
    x_key: str,
    gem5_bin: Path,
    config_py: Path,
    out_root: Path,
    y_key: str,
    env: Optional[Dict[str, str]],
) -> List[RunResult]:
    """
    Parallelize across CURVES (each curve handled sequentially inside).
    """
    clip_maker = lambda: make_clipper(args, y_key)
    all_results: List[RunResult] = []

    def work(curve: CurveVariant) -> List[RunResult]:
        clipper = clip_maker()
        return run_curve_sequential(
            args,
            base_cfg,
            curve,
            x_values,
            x_is_num,
            x_key,
            gem5_bin,
            config_py,
            out_root,
            y_key,
            clipper,
            env,
        )

    max_workers = max(1, int(args.max_parallel))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2curve = {ex.submit(work, cv): cv for cv in curves}
        for fut in as_completed(fut2curve):
            res = fut.result()
            all_results.extend(res)
            ok_cnt = sum(1 for r in res if r.ok)
            cname = res[0].curve_name if res else "(empty)"
            print(f"[done curve] {cname}: {ok_cnt}/{len(res)} ok")

    def sort_key(r: RunResult):
        return (
            r.curve_name,
            r.x_num if (r.x_is_num and r.x_num is not None) else float("inf"),
            r.x_value,
        )

    all_results.sort(key=sort_key)
    return all_results


# ---------------------------- Curves config ----------------------------


def parse_grid_spec(spec: str) -> List[CurveVariant]:
    """
    Example: "--synthetic=uniform_random,bit_complement; --vcs-per-vnet=1,16"
    Returns cartesian product variants, names auto-generated.
    """
    if not spec:
        return []
    dims: List[Tuple[str, List[str]]] = []
    for group in spec.split(";"):
        group = group.strip()
        if not group:
            continue
        if "=" not in group:
            raise ValueError(f"Invalid grid group: {group}")
        k, vs = group.split("=", 1)
        k = k.strip()
        values = [v.strip() for v in vs.split(",") if v.strip()]
        dims.append((k, values))
    if not dims:
        return []
    variants = []
    for combo in itertools.product(*[vals for _, vals in dims]):
        overrides = {dims[i][0]: combo[i] for i in range(len(dims))}
        name = ",".join(
            f"{dims[i][0].lstrip('-')}={combo[i]}" for i in range(len(dims))
        )
        variants.append(CurveVariant(name=name, overrides=overrides))
    return variants


def load_curves_json(path: Optional[str]) -> List[CurveVariant]:
    """
    JSON format: [{"name":"UR","overrides":{"--synthetic":"uniform_random","--vcs-per-vnet":"1"}, "label_keys":["--synthetic"]}, ...]
    """
    if not path:
        return []
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    out: List[CurveVariant] = []
    for d in data:
        out.append(
            CurveVariant(
                name=d["name"],
                overrides=d.get("overrides", {}),
                label_keys=d.get("label_keys"),
            )
        )
    return out


# ------------------------------ Storage --------------------------------


def write_csv_json(
    results: List[RunResult], y_key: str, outdir: Path, tag: str
):
    ensure_dir(outdir)
    csv_p = outdir / f"results_{tag}.csv"
    json_p = outdir / f"results_{tag}.json"

    cols = ["curve_name", "x_value", "ok", "reason", "outdir", y_key]
    extra_candidates = [
        "average_packet_queueing_latency",
        "average_packet_network_latency",
        "average_packet_latency",
        "average_hops",
        "packets_injected::total",
        "packets_received::total",
        "reception_rate",
    ]
    present_extras = [
        k for k in extra_candidates if any(k in r.stats for r in results)
    ]
    cols += present_extras

    with csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            row = [
                r.curve_name,
                r.x_value,
                int(r.ok),
                r.reason,
                r.outdir,
                r.stats.get(y_key, ""),
            ]
            for k in present_extras:
                row.append(r.stats.get(k, ""))
            w.writerow(row)

    json.dump(
        [asdict(r) for r in results],
        json_p.open("w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    print(f"[saved] {csv_p}")
    print(f"[saved] {json_p}")


# ------------------------------- Plotting ------------------------------


def _compute_figsize(
    fig_width: float, fig_height: float, fig_aspect: float
) -> Tuple[float, float]:
    """Compute (width, height) in inches with precedence: (width,height) > (width,aspect) > defaults."""
    if fig_width > 0 and fig_height > 0:
        return fig_width, fig_height
    if fig_width > 0 and fig_aspect > 0:
        return fig_width, max(1e-3, fig_width / fig_aspect)
    # final safe default (10in width, 16:9)
    w = 10.0
    h = w / (fig_aspect if fig_aspect > 0 else (16 / 9))
    return w, h


def plot_results(
    results: List[RunResult],
    y_key: str,
    base_overrides: Dict[str, str],
    curves: List[CurveVariant],
    label_keys_cli: Optional[List[str]],
    title: str,
    save_path: Path,
    y_max: Optional[float],
    x_title: str,
    fig_width: float = 10.0,
    fig_height: float = 0.0,
    fig_aspect: float = 16 / 9,
    legend_right_frac: float = 0.20,
    dpi: int = 160,
    legend_fontsize: float = 10.0,
    legend_ncol: int = 1,
    legend_compact: bool = False,
):
    """
    Make a single chart with multiple curves; legend sits outside to the right.
    - The final exported figure strictly follows the requested width/height/aspect.
    - We reserve a fixed fraction on the right for the legend using tight_layout(rect=[...]).
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Compute enforced figure size
    W, H = _compute_figsize(fig_width, fig_height, fig_aspect)

    # Prepare figure and axes
    fig, ax = plt.subplots(figsize=(W, H))
    # Force size again just in case backends try to adjust
    fig.set_size_inches(W, H, forward=True)

    # Group by curve
    by_curve: Dict[str, List[RunResult]] = {}
    for r in results:
        by_curve.setdefault(r.curve_name, []).append(r)

    # Determine whether all ok points have numeric X
    any_ok = [r for r in results if r.ok and (y_key in r.stats)]
    x_is_num_global = (
        all(r.x_is_num and (r.x_num is not None) for r in any_ok)
        and len(any_ok) > 0
    )

    cmap = mpl.colormaps.get("tab20")
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", ">", "<"]
    color_idx = 0
    marker_idx = 0

    # For categorical X, create a stable label->index mapping across all ok points
    x_label_order: List[str] = []
    x_label_to_idx: Dict[str, int] = {}
    if not x_is_num_global:
        for cv in curves:
            pts = [
                r
                for r in by_curve.get(cv.name, [])
                if r.ok and (y_key in r.stats)
            ]
            for r in pts:
                if r.x_value not in x_label_to_idx:
                    x_label_to_idx[r.x_value] = len(x_label_order)
                    x_label_order.append(r.x_value)

    for cv in curves:
        pts = [
            r for r in by_curve.get(cv.name, []) if r.ok and (y_key in r.stats)
        ]
        if not pts:
            continue

        # sort by X for plotting
        if x_is_num_global:
            pts.sort(key=lambda r: r.x_num)
            xs = [r.x_num for r in pts]
            ax.set_xlabel(x_title)
        else:
            pts.sort(key=lambda r: x_label_to_idx[r.x_value])
            xs = [x_label_to_idx[r.x_value] for r in pts]
            ax.set_xlabel(x_title + " (categorical)")

        ys = [r.stats[y_key] for r in pts]

        color = cmap(color_idx % cmap.N)
        marker = markers[marker_idx % len(markers)]
        color_idx += 1
        marker_idx += 1

        label = salient_label(
            base_overrides, cv, cv.label_keys or label_keys_cli
        )
        ax.plot(
            xs,
            ys,
            marker=marker,
            linestyle="-",
            linewidth=1.8,
            label=label,
            color=color,
        )

    if not x_is_num_global:
        ax.set_xticks(list(range(len(x_label_order))))
        ax.set_xticklabels(x_label_order, rotation=20, ha="right")

    ax.set_ylabel(y_key)
    ax.grid(True, linestyle="--", alpha=0.4)
    if title:
        ax.set_title(title)
    if y_max is not None:
        ax.set_ylim(top=y_max)

    # Legend outside on the right; reserve fixed fraction to keep the main axes area consistent
    legend_right_frac = max(
        0.05, min(0.40, legend_right_frac)
    )  # clamp to reasonable range

    # Compact legend knobs
    handlelength = 1.2 if legend_compact else 1.6
    markerscale = 0.8 if legend_compact else 1.0
    labelspacing = 0.3 if legend_compact else 0.5
    borderpad = 0.3 if legend_compact else 0.5
    columnspacing = 0.6 if legend_compact else 1.0

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1 + 0.02 / (1 - legend_right_frac), 0.5),
        frameon=True,
        fontsize=legend_fontsize,
        ncol=max(1, int(legend_ncol)),
        handlelength=handlelength,
        markerscale=markerscale,
        labelspacing=labelspacing,
        borderpad=borderpad,
        columnspacing=columnspacing,
    )

    # Reserve right margin for legend; rect=[left,bottom,right,top]
    fig.tight_layout(rect=[0.0, 0.0, 1.0 - legend_right_frac, 1.0])

    # Enforce final size again and save with requested DPI
    fig.set_size_inches(W, H, forward=True)
    from pathlib import Path as _P

    out_png = _P(save_path.with_suffix(".png"))
    out_pdf = _P(save_path.with_suffix(".pdf"))
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_pdf)
    print(f"[saved plot] {out_png}")
    print(f"[saved plot] {out_pdf}")


# -------------------------------- Main ---------------------------------


def parse_base_flags(flags: List[str]) -> Dict[str, str]:
    """
    Convert a list like ["--network=garnet","--num-cpus=64","--synthetic=bit_complement","--inj-vnet=0"]
    into a dict; tolerate "--flag value" style as well.
    """
    out: Dict[str, str] = {}
    i = 0
    while i < len(flags):
        tok = flags[i]
        if tok.startswith("-"):
            if "=" in tok:
                k, v = tok.split("=", 1)
                out[k] = v
                i += 1
            else:
                if i + 1 < len(flags) and not flags[i + 1].startswith("-"):
                    out[tok] = flags[i + 1]
                    i += 2
                else:
                    out[tok] = ""
                    i += 1
        else:
            i += 1
    return out


def parse_x_values_by_range(spec: str) -> Tuple[List[str], bool]:
    """Parse numeric range 'start:end:step' into (values_as_str, is_numeric=True)."""
    try:
        start_s, end_s, step_s = spec.split(":")
        start, end, step = float(start_s), float(end_s), float(step_s)
    except Exception:
        raise ValueError(
            f"Invalid --x-range format: {spec} (expect start:end:step)"
        )
    vals = float_range(start, end, step)
    return [f"{v:.6f}".rstrip("0").rstrip(".") for v in vals], True


def parse_x_values_by_list(spec: str) -> Tuple[List[str], bool]:
    """Parse comma-separated list into (values_as_str, is_numeric_if_all_float)."""
    raw = [s.strip() for s in spec.split(",") if s.strip()]
    is_num = True
    for s in raw:
        try:
            float(s)
        except Exception:
            is_num = False
            break
    return raw, is_num


def main():
    ap = argparse.ArgumentParser(
        description="Batch launcher & plotter for gem5 Garnet sweeps (general X-axis) with enforced figure aspect"
    )

    # Paths
    ap.add_argument("--gem5", type=Path, default=Path("./build/NULL/gem5.opt"))
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/example/garnet_synth_traffic.py"),
    )

    # IO / labeling
    ap.add_argument("--out-root", type=Path, default=Path("runs_out"))
    ap.add_argument("--title", type=str, default="Garnet Sweep")

    # Base flags applied to ALL runs (do NOT include the sweep flag)
    ap.add_argument(
        "--base-flags",
        type=str,
        default=(
            "--network=garnet --num-cpus=64 --num-dirs=64 "
            "--topology=HyperX --mesh-rows=4 "
            "--routing-algorithm=3 --inj-vnet=0 "
            "--synthetic=bit_complement --vcs-per-vnet=16 "
            "--sim-cycles=20000 --dimwar-weight=cong"
        ),
    )

    # Curves: either a grid spec or a JSON list
    ap.add_argument(
        "--grid",
        type=str,
        default="",
        help="E.g. '--synthetic=uniform_random,bit_complement;--vcs-per-vnet=1,16'",
    )
    ap.add_argument(
        "--curves-json",
        type=str,
        default="",
        help="Path to JSON describing explicit curve variants",
    )
    ap.add_argument(
        "--label-keys",
        type=str,
        default="",
        help="Comma-separated flag keys to show in legend (override)",
    )

    # X axis definition
    ap.add_argument(
        "--x-flag",
        type=str,
        default="--injectionrate",
        help="Which CLI flag to sweep as the X axis (e.g., --injectionrate, --vcs-per-vnet, --synthetic)",
    )
    mx = ap.add_argument_group("X sweep values (choose one)")
    mx.add_argument(
        "--x-range",
        type=str,
        default="",
        help="Numeric range 'start:end:step', e.g., '0.02:1.00:0.02'",
    )
    mx.add_argument(
        "--x-values",
        type=str,
        default="",
        help="Comma-separated values, e.g., '1,4,8,16' or 'uniform_random,bit_complement'",
    )
    ap.add_argument(
        "--x-title",
        type=str,
        default="X",
        help="X axis label; default is the flag name without leading dashes",
    )

    # Metric / clip / timeout / y limit
    ap.add_argument("--y-metric", type=str, default="average_packet_latency")
    ap.add_argument(
        "--clip", choices=["none", "abs", "rel", "mono"], default="none"
    )
    ap.add_argument("--clip-threshold", type=float, default=1000.0)
    ap.add_argument("--clip-factor", type=float, default=5.0)
    ap.add_argument("--clip-jump-factor", type=float, default=2.0)
    ap.add_argument(
        "--clip-module",
        type=str,
        default="",
        help="Custom module:func taking List[RunResult] -> bool",
    )
    ap.add_argument(
        "--timeout-sec",
        type=int,
        default=0,
        help="Per-run timeout in seconds; 0 disables",
    )
    ap.add_argument(
        "--y-max",
        type=float,
        default=0.0,
        help="If >0, cap y-axis upper bound",
    )

    # Execution
    ap.add_argument(
        "--max-parallel",
        type=int,
        default=2,
        help="How many curves run in parallel",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse finished points by parsing stats only",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print commands; no execution",
    )
    ap.add_argument(
        "--env", type=str, default="", help="Extra env as JSON dict string"
    )

    # Plot & export
    ap.add_argument("--save-tag", type=str, default="default")
    ap.add_argument(
        "--plot", action="store_true", help="Generate plot after running"
    )
    ap.add_argument("--plot-out", type=Path, default=Path("plots/sweep"))

    # Figure size/aspect enforcement
    ap.add_argument(
        "--fig-width", type=float, default=10.0, help="Figure width in inches"
    )
    ap.add_argument(
        "--fig-height",
        type=float,
        default=0.0,
        help="Figure height in inches (if 0, computed from aspect)",
    )
    ap.add_argument(
        "--fig-aspect",
        type=float,
        default=16 / 9,
        help="Figure aspect ratio = width/height (used if height==0)",
    )
    ap.add_argument(
        "--legend-right-frac",
        type=float,
        default=0.20,
        help="Right fraction reserved for legend (0.05~0.40)",
    )
    ap.add_argument("--dpi", type=int, default=160, help="Output image DPI")

    # Fine-grained legend controls
    ap.add_argument(
        "--legend-fontsize", type=float, default=10.0, help="Legend font size"
    )
    ap.add_argument(
        "--legend-ncol",
        type=int,
        default=1,
        help="Legend columns to pack entries horizontally",
    )
    ap.add_argument(
        "--legend-compact",
        action="store_true",
        help="Use tighter paddings/handles for a more compact legend",
    )

    args = ap.parse_args()

    # Parse base flags into dict
    base_cfg = parse_base_flags(shlex.split(args.base_flags))

    # Determine X values
    if args.x_range and args.x_values:
        raise SystemExit("Use either --x-range or --x-values (not both).")
    if args.x_range:
        x_values, x_is_num = parse_x_values_by_range(args.x_range)
    elif args.x_values:
        x_values, x_is_num = parse_x_values_by_list(args.x_values)
    else:
        # Default to injection rate example if not provided
        x_values, x_is_num = parse_x_values_by_range("0.02:1.00:0.02")
    x_key = args.x_flag
    x_title = args.x_title if args.x_title else x_key.lstrip("-")

    # Construct curve variants
    curves: List[CurveVariant] = []
    curves.extend(parse_grid_spec(args.grid))
    curves.extend(load_curves_json(args.curves_json))
    if not curves:
        curves = [CurveVariant(name="base", overrides={})]

    # Legend highlight keys (global override)
    label_keys_cli = (
        [k.strip() for k in args.label_keys.split(",") if k.strip()]
        if args.label_keys
        else None
    )

    # Prepare environment
    env = os.environ.copy()
    if args.env:
        env.update(json.loads(args.env))

    # Execute all runs
    results = parallel_run_all_curves(
        args=args,
        base_cfg=base_cfg,
        curves=curves,
        x_values=x_values,
        x_is_num=x_is_num,
        x_key=x_key,
        gem5_bin=args.gem5,
        config_py=args.config,
        out_root=args.out_root,
        y_key=args.y_metric,
        env=env,
    )

    # Save outputs
    write_csv_json(results, args.y_metric, args.out_root, args.save_tag)

    # Plot
    if args.plot:
        plot_results(
            results=results,
            y_key=args.y_metric,
            base_overrides=base_cfg,
            curves=curves,
            label_keys_cli=label_keys_cli,
            title=args.title,
            save_path=args.plot_out,
            y_max=(args.y_max if args.y_max > 0 else None),
            x_title=x_title,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            fig_aspect=args.fig_aspect,
            legend_right_frac=args.legend_right_frac,
            dpi=args.dpi,
            legend_fontsize=args.legend_fontsize,
            legend_ncol=args.legend_ncol,
            legend_compact=args.legend_compact,
        )


if __name__ == "__main__":
    main()
