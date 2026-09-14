#!/usr/bin/env python3
"""Scan a directory of .npy/.npz files and render its structure as a mind map.

Produces three artifacts from one scan:
  - <name>.json  machine-readable schema (recursive: path / shape / dtype / stats)
  - <name>.html  self-contained interactive mind map + list view, no CDN, offline
  - <name>.png   static mind map for embedding into documents

Recursion walks into .npz keys, object arrays, nested dicts/lists/tuples, so a
`data.npz` holding an object array of per-session dicts is fully expanded.

Usage:
  python inspect_npy.py <path> [--out DIR] [--no-pickle] [--max-depth N]

Safe by design: read-only, no network, never opens a GUI window.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

STAT_SAMPLE = 100_000          # max elements used for min/max/unique stats
UNIQUE_CAP = 200_000           # skip np.unique above this many sampled elements
LABEL_MAX_CLASSES = 50         # unique-count ceiling for the "label-ish" hint


# --------------------------------------------------------------------------
# scalar / stat helpers
# --------------------------------------------------------------------------

def _num(x):
    """numpy scalar -> compact JSON-safe number."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(f):
        return str(f)
    if f == 0 or (1e-4 <= abs(f) < 1e6):
        return round(f, 4)
    return float(f"{f:.4g}")


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def _preview_scalar(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, (bytes, bytearray)):
        return repr(bytes(v[:60]).decode("utf-8", "replace"))
    if isinstance(v, str):
        return repr(v[:80])
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (np.floating, float)):
        return f"{float(v):.4g}"
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    return f"<{type(v).__name__}>"


def _label_hint(arr, uniq) -> str | None:
    k = arr.dtype.kind
    if k == "b":
        return "布尔掩码"
    if k in "iu" and 2 <= uniq.size <= LABEL_MAX_CLASSES and arr.ndim <= 2:
        return f"疑似类别标签（{uniq.size} 类）"
    if k == "f" and uniq.size == 2:
        vals = set(np.asarray(uniq, dtype=float).tolist())
        if vals <= {0.0, 1.0}:
            return "疑似 0/1 掩码"
    return None


def array_stats(a: np.ndarray) -> dict:
    st: dict = {
        "shape": [int(x) for x in a.shape],
        "dtype": str(a.dtype),
        "size": int(a.size),
        "nbytes": int(a.nbytes),
        "ndim": int(a.ndim),
    }
    if a.size == 0:
        st["empty"] = True
        return st

    if a.dtype.kind == "O":
        st["kind"] = "object"
        flat = a.reshape(-1)
        n = min(flat.size, 200)
        counts: dict[str, int] = {}
        for i in range(n):
            t = type(flat[i]).__name__
            counts[t] = counts.get(t, 0) + 1
        st["element_types"] = counts
        return st

    flat = a.reshape(-1)
    sampled = False
    if flat.size > STAT_SAMPLE:
        step = max(1, flat.size // STAT_SAMPLE)
        s = flat[::step]
        sampled = True
        st["sampled"] = f"每 {step} 个取 1，共 {s.size} 个进入统计"
    else:
        s = flat

    if a.dtype.kind in "biufc":
        try:
            if a.dtype.kind == "c":
                mag = np.abs(s)
                st["min"] = _num(mag.min())
                st["max"] = _num(mag.max())
                st["note"] = "复数数组，min/max 为模"
            else:
                st["min"] = _num(np.min(s))
                st["max"] = _num(np.max(s))
                st["mean"] = _num(np.mean(s))
                st["std"] = _num(np.std(s))
            if a.dtype.kind == "f":
                nan = int(np.isnan(s).sum())
                if nan:
                    st["nan_count"] = nan
        except Exception as e:  # pragma: no cover - defensive
            st["stats_error"] = str(e)

    if a.dtype.kind in "biuf" and s.size <= UNIQUE_CAP:
        try:
            uniq = np.unique(s)
            st["n_unique"] = int(uniq.size)
            hint = _label_hint(a, uniq)
            if hint:
                st["label_hint"] = hint
            if uniq.size <= 12:
                st["unique_values"] = [_num(x) for x in uniq]
        except Exception:
            pass

    if a.dtype.kind == "U":
        st["sample"] = [str(x) for x in flat[:5]]
    elif a.dtype.kind == "S":
        st["sample"] = [bytes(x).decode("utf-8", "replace") for x in flat[:5]]
    elif a.dtype.kind in "biuf" and a.ndim <= 2 and a.size <= 64:
        st["values"] = np.array2string(a, precision=3, threshold=64,
                                       max_line_width=200).splitlines()

    if sampled:
        pass
    return st


# --------------------------------------------------------------------------
# recursive scan
# --------------------------------------------------------------------------

def _trunc_node(n_hidden: int) -> dict:
    return {"name": f"… 另有 {n_hidden} 项", "kind": "truncated",
            "detail": "超出 --max-items，已省略", "children": []}


def walk_array(a: np.ndarray, name: str, args, depth: int, acc: str = "") -> dict:
    """`acc` is the Python subscript that reaches this node from its parent's
    variable -- "" means the node *is* the parent's value (a .npy file holds its
    array directly, no key to subscript). The HTML turns the chain of these into
    a copy-pasteable Jupyter cell; see `indexLines` there."""
    st = array_stats(a)
    detail = f"{tuple(int(x) for x in a.shape)}  {a.dtype}"
    node = {"name": name, "kind": "array", "detail": detail, "acc": acc,
            "stats": st, "children": []}
    if "label_hint" in st:
        # Own field, not appended to `detail`: the most actionable finding in the
        # whole scan should not read as one more lump of metadata.
        node["hint"] = st["label_hint"]

    if a.dtype.kind == "O" and depth < args.max_depth:
        node["kind"] = "object_array"
        flat = a.reshape(-1)
        n = min(flat.size, args.max_items)
        for i in range(n):
            if a.ndim > 1:
                idx = tuple(int(x) for x in np.unravel_index(i, a.shape))
                label = str(idx)
                sub = "[" + ", ".join(str(x) for x in idx) + "]"
            else:
                label = f"[{i}]"
                sub = f"[{i}]"
            try:
                node["children"].append(
                    walk_value(flat[i], label, args, depth + 1, sub))
            except Exception as e:
                node["children"].append({
                    "name": label, "kind": "error", "acc": sub,
                    "detail": f"展开失败: {type(e).__name__}: {e}", "children": []})
        if flat.size > n:
            node["children"].append(_trunc_node(flat.size - n))
    return node


def walk_value(v, name: str, args, depth: int, acc: str = "") -> dict:
    if depth > args.max_depth:
        return {"name": name, "kind": "truncated", "acc": None,
                "detail": "达到 --max-depth", "children": []}
    if isinstance(v, np.ndarray):
        return walk_array(v, name, args, depth, acc)
    if isinstance(v, dict):
        node = {"name": name, "kind": "dict", "detail": f"dict · {len(v)} 键",
                "acc": acc, "children": []}
        for i, (k, vv) in enumerate(v.items()):
            if i >= args.max_items:
                node["children"].append(_trunc_node(len(v) - i))
                break
            node["children"].append(
                walk_value(vv, str(k), args, depth + 1, f"[{str(k)!r}]"))
        return node
    if isinstance(v, (list, tuple)):
        t = type(v).__name__
        node = {"name": name, "kind": t, "detail": f"{t} · {len(v)} 项",
                "acc": acc, "children": []}
        for i, vv in enumerate(v[: args.max_items]):
            node["children"].append(
                walk_value(vv, f"[{i}]", args, depth + 1, f"[{i}]"))
        if len(v) > args.max_items:
            node["children"].append(_trunc_node(len(v) - args.max_items))
        return node
    if isinstance(v, np.generic):
        v = v.item()
    return {"name": name, "kind": "scalar", "detail": _preview_scalar(v),
            "acc": acc, "children": []}


def scan_file(path: Path, args) -> dict:
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    node = {"name": path.name, "kind": "file", "detail": _human(size),
            "path": str(path), "npz": path.suffix == ".npz", "children": []}

    if path.suffix == ".npz":
        try:
            try:
                z = np.load(path, allow_pickle=args.allow_pickle, mmap_mode="r")
            except Exception:
                z = np.load(path, allow_pickle=args.allow_pickle)
        except Exception as e:
            node["detail"] += f"   打开失败: {type(e).__name__}"
            node["error"] = f"{type(e).__name__}: {e}"
            node["kind"] = "error"
            return node
        node["detail"] += f"   · npz · {len(z.files)} 个键"
        for k in z.files:
            try:
                node["children"].append(walk_value(z[k], k, args, 1, f"[{k!r}]"))
            except Exception as e:
                node["children"].append({
                    "name": k, "kind": "error", "acc": f"[{k!r}]",
                    "detail": f"读取失败: {type(e).__name__}: {e}", "children": []})
        try:
            z.close()
        except Exception:
            pass
    else:
        try:
            try:
                a = np.load(path, allow_pickle=args.allow_pickle, mmap_mode="r")
            except Exception:
                a = np.load(path, allow_pickle=args.allow_pickle)
        except Exception as e:
            node["detail"] += f"   打开失败: {type(e).__name__}"
            node["error"] = f"{type(e).__name__}: {e}"
            node["kind"] = "error"
            return node
        try:
            node["children"].append(walk_value(a, path.stem, args, 1))
        except Exception as e:
            node["children"].append({
                "name": path.stem, "kind": "error",
                "detail": f"展开失败: {type(e).__name__}: {e}", "children": []})
    return node


def build_tree(root: Path, args) -> dict:
    files: list[Path] = []
    if root.is_file():
        files = [root]
    else:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(".") and d != "__pycache__"]
            for fn in sorted(filenames):
                if Path(fn).suffix in (".npy", ".npz"):
                    files.append(Path(dirpath) / fn)
            if len(files) >= args.max_files:
                break
    files = sorted(files)[: args.max_files]

    # Full paths blow up the root box and carry no information -- the user
    # already knows where their dataset lives. Last two components is enough
    # to confirm the script scanned the right place.
    if root.is_file():
        where = root.name
    else:
        where = "…/" + "/".join(root.parts[-2:])
    tree = {"name": root.name or str(root), "kind": "root", "path": str(root),
            "detail": "", "children": []}

    if not files:
        tree["detail"] = f"没找到 .npy/.npz @ {where}"
        return tree

    if root.is_file():
        tree["children"] = [scan_file(f, args) for f in files]
    else:
        groups: dict[str, dict] = {}
        for f in files:
            rel = f.parent.relative_to(root)
            key = "." if str(rel) == "." else str(rel).replace("\\", "/")
            if key not in groups:
                groups[key] = {"name": key, "kind": "folder",
                               "detail": "子目录", "path": str(f.parent),
                               "children": []}
            groups[key]["children"].append(scan_file(f, args))
        tree["children"] = [groups[k] for k in sorted(groups)]

    # Count from the aggregates, not from len(files): a file that fails to open
    # is tagged `error` and is deliberately not counted as a file, so the root
    # box and the header would otherwise disagree (4 files vs 3).
    st = _tag_stats(tree)
    detail = f"{st['files']} 个数组文件"
    if st["errors"]:
        detail += f" · {len(st['errors'])} 个读取失败"
    tree["detail"] = f"{detail} @ {where}"
    return tree


def _tag_stats(node: dict) -> dict:
    """Bottom-up: attach aggregate counters used by the HTML header."""
    agg = {"files": 0, "arrays": 0, "scalars": 0, "bytes": 0,
           "label_hints": [], "errors": []}
    if node.get("kind") == "file":
        agg["files"] = 1
        try:
            agg["bytes"] = Path(node["path"]).stat().st_size
        except (KeyError, OSError):
            pass
    if node.get("kind") == "error":
        agg["errors"].append(node["name"])
    st = node.get("stats")
    if st:
        agg["arrays"] = 1
        hint = st.get("label_hint")
        if hint:
            path_name = node.get("name")
            agg["label_hints"].append(f"{path_name}  —  {hint}")
    for c in node.get("children", []):
        sub = _tag_stats(c)
        for k in ("files", "arrays", "scalars", "bytes"):
            agg[k] += sub[k]
        agg["label_hints"] += sub["label_hints"]
        agg["errors"] += sub["errors"]
    # collapse repeats of the same array name (a per-session object array yields
    # one hint per element, all saying the same thing)
    seen: set = set()
    dedup: list = []
    for h in agg["label_hints"]:
        nm = h.split("  —  ")[0]
        if nm in seen:
            continue
        seen.add(nm)
        dedup.append(h)
    agg["label_hints"] = dedup[:40]
    node["_agg"] = agg
    return agg


# --------------------------------------------------------------------------
# rendering: shared layout
# --------------------------------------------------------------------------

# kind -> (ink, fill). Kept in sync with the HTML's LIGHT/DARK tables by hand;
# both are short enough that a shared source would cost more than it saves.
#
# Deliberately restrained: containers (folders, files, scalars) are all one
# neutral family, so colour carries information instead of decorating. Only
# three hues survive -- teal = real data, violet = nested structure, amber =
# object array (opaque until you index into it) -- plus red for errors.
KIND_STYLE = {
    "root": ("#0f172a", "#eef2f7"),
    "folder": ("#64748b", "#f8fafc"),
    "file": ("#64748b", "#f8fafc"),
    "array": ("#0d9488", "#f0fdfa"),
    "object_array": ("#b45309", "#fffbeb"),
    "dict": ("#7c3aed", "#f5f3ff"),
    "list": ("#7c3aed", "#f5f3ff"),
    "tuple": ("#7c3aed", "#f5f3ff"),
    "scalar": ("#94a3b8", "#f8fafc"),
    "error": ("#dc2626", "#fef2f2"),
    "truncated": ("#94a3b8", "#f8fafc"),
}
LINK_COLOR = "#cbd5e1"
INK = "#0f172a"
MUTED = "#64748b"
SHAPE_INK = "#334155"     # shapes read a shade stronger than the dtype beside them
HINT_INK = "#b45309"      # the amber used for "this looks like a label"
SEG_COLOR = {"shape": SHAPE_INK, "dtype": MUTED, "plain": MUTED, "hint": HINT_INK}


def _fmt_shape(sh) -> str:
    """[1200, 40] -> '(1200, 40)'; numpy's own repr, including the 1-tuple comma."""
    body = ", ".join(str(int(x)) for x in sh)
    if len(sh) == 1:
        return f"({body},)"
    return f"({body})"


def _segments(n: dict):
    """Split a node's second line into styled pieces.

    Returns [(text, style)] where style is shape | dtype | hint | plain. Array
    nodes get shape and dtype as separate pieces so the shape -- the thing the
    eye is hunting for -- can carry more weight than the dtype beside it.
    """
    st = n.get("stats") or {}
    segs = []
    if st.get("shape") is not None:
        segs.append((_fmt_shape(st["shape"]), "shape"))
    if st.get("dtype"):
        segs.append((str(st["dtype"]), "dtype"))
    if not segs and n.get("detail"):
        segs.append((n["detail"], "plain"))
    if n.get("hint"):
        segs.append((n["hint"], "hint"))
    return segs


def _ascii(s: str) -> str:
    return "".join(ch if ord(ch) < 128 else "" for ch in s)


def _flatten(root: dict):
    nodes = []

    def rec(n, depth):
        n["_d"] = depth
        nodes.append(n)
        for c in n.get("children", []):
            rec(c, depth + 1)

    rec(root, 0)
    return nodes



# --------------------------------------------------------------------------
# rendering: PNG
# --------------------------------------------------------------------------

def _slim(node: dict, threshold: int) -> dict:
    """Collapse runs of identically-shaped leaf siblings into one summary row.

    A file with `key_000 … key_059`, all `(4,) float64`, costs 60 rows of diagram
    height and tells you nothing extra. Grouping them keeps the interesting
    structure (the one big tensor, the nested object arrays) legible.
    Render-only: the JSON and HTML keep every node.
    """
    kids = node.get("children", [])
    if not kids:
        return node
    if not threshold or threshold <= 1:
        return {**node, "children": [_slim(k, threshold) for k in kids]}

    leaf_groups: dict = {}
    for k in kids:
        if not k.get("children"):
            leaf_groups.setdefault((k.get("kind"), k.get("detail")), []).append(k)

    out_kids, emitted = [], set()
    for k in kids:
        if k.get("children"):
            out_kids.append(_slim(k, threshold))
            continue
        key = (k.get("kind"), k.get("detail"))
        grp = leaf_groups[key]
        if len(grp) < threshold:
            out_kids.append(k)
            continue
        if key in emitted:
            continue
        emitted.add(key)
        names = [g["name"] for g in grp]
        pfx = os.path.commonprefix(names)
        label = f"{pfx}*（{names[0]} … {names[-1]}）" if len(pfx) >= 3 \
            else f"{names[0]} … {names[-1]}"
        out_kids.append({
            "name": f"×{len(grp)} 个同型数组", "kind": k.get("kind", "array"),
            "detail": f"{label}   {k.get('detail', '')}", "children": [],
        })
    return {**node, "children": out_kids}


def _resolve_fonts(fm, sizes):
    """Pick a font family that covers Latin *and* CJK.

    Returns (props_by_key, cjk_available). Deliberately resolves a concrete
    family rather than passing a family *list*: matplotlib's per-glyph fallback
    silently produced tofu boxes on this machine, so a bold face would quietly
    render as blank rectangles.
    """
    cjk = None
    try:
        available = {f.name for f in fm.fontManager.ttflist}
        for cand in ("Microsoft YaHei", "Noto Sans SC", "SimHei", "Noto Sans CJK SC",
                     "Source Han Sans SC", "PingFang SC", "Microsoft JhengHei"):
            if cand in available:
                cjk = cand
                break
    except Exception:
        pass

    props = {}
    for key, (size, weight) in sizes.items():
        if cjk:
            props[key] = fm.FontProperties(family=cjk, size=size, weight=weight)
        else:
            props[key] = fm.FontProperties(family="monospace", size=size,
                                           weight=weight)
    return props, cjk is not None


def render_png(root: dict, out: Path, max_label: int = 56,
               collapse: int = 8) -> None:
    """Static mind map.

    Node boxes are sized from *measured* text extents, not guessed character
    counts, so the rounded rectangles always fit their contents exactly.
    """
    import matplotlib
    matplotlib.use("Agg")               # never open a window
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.patches import FancyBboxPatch, PathPatch, Circle
    from matplotlib.path import Path as MPath
    import warnings
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")

    root = _slim(root, collapse)
    plt.rcParams["axes.unicode_minus"] = False

    props, have_cjk = _resolve_fonts(fm, {
        "title": (12.0, "bold"),
        "name": (8.0, "bold"),
        "detail": (7.2, "normal"),
        "toggle": (9.0, "bold"),
    })
    name_font, det_font = props["name"], props["detail"]
    title_font, tog_font = props["title"], props["toggle"]

    if not have_cjk:
        for n in _flatten(root):
            n["detail"] = _ascii(n.get("detail", ""))
            n["name"] = _ascii(n["name"])
        max_label = min(max_label, 46)
    dash = "−" if have_cjk else "-"

    for n in _flatten(root):
        d = n.get("detail", "")
        if len(n["name"]) > max_label:
            n["name"] = n["name"][: max_label - 1] + ("…" if have_cjk else "...")
        if len(d) > max_label:
            n["detail"] = d[: max_label - 1] + ("…" if have_cjk else "...")

    # --- measure every label once, in inches ------------------------------
    DPI = 170
    probe = plt.figure(figsize=(4, 4), dpi=DPI)
    renderer = probe.canvas.get_renderer()

    def w_in(s, prop):
        if not s:
            return 0.0
        return renderer.get_text_width_height_descent(s, prop, False)[0] / DPI

    PAD_X, GAP_X, BOX_H, GAP_Y, TOG_R = 0.085, 0.46, 0.27, 0.062, 0.058
    SEG_GAP = 0.085

    nodes = _flatten(root)
    for n in nodes:
        n["_nw"] = w_in(n["name"], name_font)
        n["_segs"] = _segments(n)
        n["_sw"] = [w_in(t, det_font) for t, _ in n["_segs"]]
        # reserve room for the toggle circle that straddles the right edge
        extra = TOG_R if n.get("children") else 0.0
        n["_bw"] = (PAD_X * 2 + extra + n["_nw"]
                    + sum(SEG_GAP + w for w in n["_sw"]))

    # --- tidy tree: leaves stack, parents centre on their children --------
    max_depth = max(n["_d"] for n in nodes)
    colw = [0.0] * (max_depth + 1)
    for n in nodes:
        colw[n["_d"]] = max(colw[n["_d"]], n["_bw"])
    x0 = [0.0] * (max_depth + 1)
    for d in range(1, max_depth + 1):
        x0[d] = x0[d - 1] + colw[d - 1] + GAP_X

    cursor = 0.0

    def assign(n):
        nonlocal cursor
        kids = n.get("children", [])
        if not kids:
            n["_y"] = cursor
            cursor += BOX_H + GAP_Y
        else:
            for c in kids:
                assign(c)
            n["_y"] = (kids[0]["_y"] + kids[-1]["_y"]) / 2.0

    assign(root)
    plt.close(probe)

    ml, mt, mb = 0.34, 0.66, 0.46
    fig_w = x0[max_depth] + colw[max_depth] + ml + 0.18
    fig_h = max(cursor, BOX_H) + mt + mb

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.axis("off")

    def X(n):
        return x0[n["_d"]] + ml

    def Y(n):
        return n["_y"] + mt

    # links: cubic bezier from the parent's toggle out to each child's left edge
    for n in nodes:
        kids = n.get("children", [])
        if not kids:
            continue
        pxf, py = X(n) + n["_bw"] + TOG_R, Y(n)
        for c in kids:
            cx, cy = X(c), Y(c)
            dx = max(0.12, (cx - pxf) * 0.5)
            ax.add_patch(PathPatch(
                MPath([(pxf, py), (pxf + dx, py), (cx - dx, cy), (cx, cy)],
                      [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]),
                fill=False, edgecolor=LINK_COLOR, lw=1.15, zorder=1,
                capstyle="round"))

    for n in nodes:
        fg, bg = KIND_STYLE.get(n["kind"], ("#374151", "#f8fafc"))
        x, y, bw = X(n), Y(n), n["_bw"]
        # object arrays get a dashed edge: they look like leaves but hide a tree
        dashed = n["kind"] == "object_array"
        ax.add_patch(FancyBboxPatch(
            (x, y - BOX_H / 2), bw, BOX_H,
            boxstyle="round,pad=0,rounding_size=0.055",
            facecolor=bg, edgecolor=fg, linewidth=1.05, zorder=3,
            linestyle=(0, (3.2, 2.2)) if dashed else "solid"))
        ax.text(x + PAD_X, y, n["name"], fontproperties=name_font, color=fg,
                va="center_baseline", ha="left", zorder=4)
        cx = x + PAD_X + n["_nw"]
        for (txt, style), w in zip(n["_segs"], n["_sw"]):
            cx += SEG_GAP
            ax.text(cx, y, txt, fontproperties=det_font,
                    color=SEG_COLOR[style], va="center_baseline", ha="left",
                    zorder=4)
            cx += w
        if n.get("children"):
            ax.add_patch(Circle((x + bw, y), TOG_R, facecolor="white",
                                edgecolor=fg, linewidth=1.1, zorder=5))
            ax.text(x + bw, y, dash, fontproperties=tog_font, color=fg,
                    ha="center", va="center_baseline", zorder=6)

    # --- title + legend ---------------------------------------------------
    agg = root.get("_agg", {})
    ax.text(ml, 0.30, root["name"], fontproperties=title_font, color=INK,
            va="center_baseline", ha="left", zorder=4)
    sub = (f"{agg.get('files', 0)} 个文件 · {agg.get('arrays', 0)} 个数组 · "
           f"{_human(agg.get('bytes', 0))}")
    if not have_cjk:
        sub = (f"{agg.get('files', 0)} files / {agg.get('arrays', 0)} arrays / "
               f"{_human(agg.get('bytes', 0))}")
    ax.text(ml + w_in(root["name"], title_font) + 0.22, 0.30, sub,
            fontproperties=det_font, color=MUTED, va="center_baseline",
            ha="left", zorder=4)

    present = {n["kind"] for n in nodes}
    groups = [
        ("文件 / 目录", "file", ("file", "folder")),
        ("数组", "array", ("array",)),
        ("object 数组", "object_array", ("object_array",)),
        ("dict / list", "dict", ("dict", "list", "tuple")),
        ("标量", "scalar", ("scalar",)),
        ("读取失败", "error", ("error",)),
    ]
    if not have_cjk:
        en = ["file / folder", "array", "object array", "dict / list",
              "scalar", "error"]
        groups = [(e, k, ks) for (_, k, ks), e in zip(groups, en)]
    # only legend the kinds this dataset actually contains
    legend = [(label, kind) for label, kind, ks in groups if present & set(ks)]
    lx, ly = ml, fig_h - mb + 0.20
    for label, kind in legend:
        fg, bg = KIND_STYLE[kind]
        ax.add_patch(FancyBboxPatch(
            (lx, ly - 0.075), 0.15, 0.15,
            boxstyle="round,pad=0,rounding_size=0.04",
            facecolor=bg, edgecolor=fg, linewidth=1.0, zorder=4))
        ax.text(lx + 0.21, ly, label, fontproperties=det_font, color=MUTED,
                va="center_baseline", ha="left", zorder=4)
        lx += 0.21 + w_in(label, det_font) + 0.26

    fig.savefig(out, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# rendering: HTML
# --------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{
    --bg:#f6f7f9; --fg:#0f172a; --muted:#64748b; --line:#e2e8f0;
    --card:#ffffff; --accent:#2563eb; --halo:#fbbf24; --hint:#b45309;
  }
  @media (prefers-color-scheme: dark){
    :root{ --bg:#0b0f19; --fg:#e2e8f0; --muted:#94a3b8; --line:#1f2937;
           --card:#111827; --accent:#60a5fa; --halo:#f59e0b; --hint:#fbbf24; }
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--fg);display:flex;flex-direction:column;
       font:14px/1.55 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
  header{background:var(--card);border-bottom:1px solid var(--line);padding:12px 18px;
         flex:none;z-index:10}
  h1{margin:0 0 5px;font-size:15.5px;font-weight:650}
  .stats{color:var(--muted);font-size:12.5px;display:flex;gap:16px;flex-wrap:wrap}
  .stats b{color:var(--fg);font-weight:600}
  .toolbar{display:flex;gap:8px;margin-top:9px;flex-wrap:wrap;align-items:center}
  input[type=search]{flex:1;min-width:200px;padding:7px 11px;border-radius:7px;
       border:1px solid var(--line);background:var(--bg);color:var(--fg);font-size:13px}
  input[type=search]:focus{outline:none;border-color:var(--accent)}
  button{padding:7px 12px;border-radius:7px;border:1px solid var(--line);
       background:var(--bg);color:var(--fg);cursor:pointer;font-size:12.5px;
       white-space:nowrap}
  button:hover{border-color:var(--accent);color:var(--accent)}
  button.on{background:var(--accent);border-color:var(--accent);color:#fff}
  button.on:hover{color:#fff}
  .seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;
       overflow:hidden;background:var(--bg)}
  .seg button{border:0;border-radius:0;padding:7px 15px;background:transparent}
  .seg button.on{background:var(--accent);color:#fff}
  .seg button.on:hover{color:#fff}
  .depth{display:inline-flex;align-items:center;gap:7px;font-size:12.5px;
       color:var(--muted);border:1px solid var(--line);border-radius:7px;
       padding:4px 10px;background:var(--bg);white-space:nowrap}
  .depth input{width:44px;border:0;background:transparent;color:var(--fg);
       font-size:13px;font-family:ui-monospace,Consolas,monospace;
       text-align:center;padding:2px 0}
  .depth input:focus{outline:none;color:var(--accent)}
  main{flex:1;min-height:0;display:flex;flex-direction:column;padding:11px 16px 14px;gap:9px}
  body.focus .hints,body.focus .idx{display:none}
  body.focus main{padding:8px 10px 10px}
  /* One row of chips, not a bulleted list: the label hints are the first thing
     worth acting on, and a list of them cost ~95px of the diagram's height. */
  .hints{background:var(--card);border:1px solid var(--line);border-radius:9px;
       padding:6px 13px;font-size:12.5px;flex:none;display:flex;align-items:center;
       gap:8px;overflow-x:auto;scrollbar-width:thin}
  .hints .hlabel{font-size:11px;font-weight:650;color:var(--muted);flex:none;
       text-transform:uppercase;letter-spacing:.05em}
  .chip{padding:1px 9px;border-radius:20px;border:1px solid var(--hint);
       background:transparent;color:var(--hint);font-size:11.5px;
       font-weight:600;cursor:pointer;white-space:nowrap;flex:none}
  .chip:hover{background:var(--hint);border-color:var(--hint);color:#fff}
  #diagram{flex:1;min-height:340px;position:relative;border:1px solid var(--line);
       border-radius:10px;background:var(--card);overflow:hidden;cursor:grab}
  #diagram.grabbing{cursor:grabbing}
  #svg{position:absolute;inset:0;width:100%;height:100%;display:block;
       user-select:none;-webkit-user-select:none}
  .zoomhint{position:absolute;right:10px;bottom:8px;font-size:11px;color:var(--muted);
       pointer-events:none}
  /* ---- diagram node text ---- */
  .nd text{font-family:ui-monospace,Consolas,"Courier New",monospace;
       dominant-baseline:central;pointer-events:none}
  .nd .nm{font-size:12.5px;font-weight:650}
  .nd .dt{font-size:11px}
  .nd .tg{font-size:12px;font-weight:700;text-anchor:middle;
       dominant-baseline:central;cursor:pointer;pointer-events:all}
  .nd .tgcircle{cursor:pointer;pointer-events:all;transition:r .12s}
  .nd{cursor:pointer}                       /* every box selects; the circle folds */
  .link{fill:none;stroke-width:1.2;stroke-linecap:round}
  /* ---- list view ---- */
  #treewrap{flex:1;overflow:auto;background:var(--card);border:1px solid var(--line);
       border-radius:10px;padding:12px 16px}
  ul.tree{list-style:none;margin:0;padding-left:0}
  ul.tree ul{list-style:none;margin:0;padding-left:20px;border-left:1px solid var(--line)}
  li.node{position:relative;padding:1.5px 0}
  .row{display:inline-flex;align-items:baseline;gap:7px;padding:2px 8px;
       border-radius:6px;cursor:default;max-width:100%}
  .row.has-kids{cursor:pointer}
  .row:hover{outline:1px solid var(--line)}
  .tw{display:inline-block;width:14px;padding:2px 3px;margin-right:2px;
      color:var(--muted);font-size:10px;text-align:center;
      user-select:none;flex:none;cursor:pointer;border-radius:4px}
  .tw:hover{background:var(--line);color:var(--fg)}
  .nm2{font-weight:600;font-family:ui-monospace,Consolas,monospace;font-size:12.5px}
  .dt2{color:var(--muted);font-family:ui-monospace,Consolas,monospace;font-size:11.5px}
  .sh2{color:var(--fg);font-weight:500}
  .hint{color:var(--hint);font-weight:600}
  .meta{color:var(--muted);font-size:11px;font-family:ui-monospace,Consolas,monospace}
  .badge{display:inline-block;padding:1px 6px;border-radius:5px;font-size:10.5px;
         border:1px solid currentColor;opacity:.85;flex:none}
  .k-file>.row{background:#eff6ff} .k-array>.row{background:#f0fdfa}
  .k-object_array>.row{background:#fffbeb} .k-dict>.row,.k-list>.row,.k-tuple>.row{background:#f5f3ff}
  .k-error>.row{background:#fef2f2;color:#dc2626} .k-folder>.row{background:#f8fafc}
  .k-root>.row{background:#eef2f7}
  li.collapsed>ul{display:none}
  li.hit>.row{outline:2.5px solid var(--halo);outline-offset:1px}
  li.sel>.row{outline:2px solid var(--accent);outline-offset:1px}
  /* ---- 参考索引 bar ---- */
  .idx{background:var(--card);border:1px solid var(--line);border-radius:9px;
       flex:none;padding:7px 13px 8px}
  .idx:has(pre.hidden){padding:7px 13px}     /* idle: no code, so no bottom pad */
  .idxhead{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
  .idx h2{margin:0;font-size:11.5px;font-weight:650;color:var(--muted);
       text-transform:uppercase;letter-spacing:.05em;flex:none}
  .crumb{font-family:ui-monospace,Consolas,monospace;font-size:12px;
       color:var(--fg);flex:1;min-width:0;overflow:hidden;
       text-overflow:ellipsis;white-space:nowrap}
  .crumb.iscrumb{font-family:inherit;font-size:12.5px;color:var(--muted)}
  .idxhead button{padding:3px 10px;font-size:11.5px}
  .idx pre{margin:6px 0 0;font-family:ui-monospace,Consolas,"Courier New",monospace;
       font-size:12px;line-height:1.7;max-height:112px;overflow:auto;
       white-space:pre;tab-size:2}
  .idx .ln{display:block;padding:0 6px;border-radius:4px}
  .idx .ln .cm{color:var(--muted)}
  .idx .ln.hot{background:color-mix(in srgb, var(--accent) 11%, transparent);
       box-shadow:inset 2px 0 0 var(--accent)}
  .idx .ln.dim{color:var(--muted)}
  .idx .tgt{color:var(--accent);font-weight:650}
  .hidden{display:none}
  footer{flex:none;color:var(--muted);font-size:11.5px;padding:0 2px 2px}
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="stats">__STATS__</div>
  <div class="toolbar">
    <div class="seg">
      <button id="v-map" class="on">导图</button>
      <button id="v-list">列表</button>
    </div>
    <input type="search" id="q" placeholder="搜索键名 / dtype / 形状…（例如 label、float32、(1200,）">
    <label class="depth">展开层数
      <input type="number" id="depth" min="0" max="12" step="1" value="2">
    </label>
    <button id="expand">全部展开</button>
    <button id="collapse">全部折叠</button>
    <button id="fit">适应窗口</button>
    <button id="grp" class="on" title="形状与 dtype 完全相同的成组兄弟合并成一行，可再展开">同型合并</button>
    <button id="focus" title="隐藏标签栏与参考索引，把整个窗口给导图">专注</button>
  </div>
</header>
<main>
  <div class="hints" id="hints"></div>
  <div class="idx" id="idx">
    <div class="idxhead">
      <h2>参考索引</h2>
      <div class="crumb iscrumb" id="crumb"></div>
      <button id="idxcopy" class="hidden">复制代码</button>
    </div>
    <pre id="idxcode" class="hidden"></pre>
  </div>
  <div id="diagram">
    <svg id="svg"><g id="stage"></g></svg>
    <div class="zoomhint">滚轮缩放 · 拖拽平移 · 点方块看索引代码 · 点圆圈 +/- 展开下一级</div>
  </div>
  <div id="treewrap" class="hidden"><ul class="tree" id="tree"></ul></div>
</main>
<footer>由 npy-inspector 生成 · 只读扫描，未修改任何数据</footer>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const RAW = JSON.parse(document.getElementById('payload').textContent);
let ROOT = RAW;          // rebuilt from RAW whenever 同型合并 is toggled

/* ---------------------------------------------------------------- palette */
// Containers stay one neutral family so colour carries information rather than
// decoration: teal = real data, violet = nested structure, amber = object array
// (opaque until indexed), red = error.
const LIGHT = {
  root:['#0f172a','#eef2f7'], folder:['#64748b','#f8fafc'], file:['#64748b','#f8fafc'],
  array:['#0d9488','#f0fdfa'], object_array:['#b45309','#fffbeb'],
  dict:['#7c3aed','#f5f3ff'], list:['#7c3aed','#f5f3ff'], tuple:['#7c3aed','#f5f3ff'],
  scalar:['#94a3b8','#f8fafc'], error:['#dc2626','#fef2f2'], truncated:['#94a3b8','#f8fafc'],
  // a 同型 group is a container, so it stays in the neutral family -- the ×N in
  // the name is what says it stands for many siblings, not the colour
  group:['#64748b','#f1f5f9']
};
const DARK = {
  root:['#e2e8f0','#1e293b'], folder:['#94a3b8','#1e293b'], file:['#94a3b8','#1e293b'],
  array:['#5eead4','#0f2e2b'], object_array:['#fcd34d','#3a2405'],
  dict:['#c4b5fd','#241453'], list:['#c4b5fd','#241453'], tuple:['#c4b5fd','#241453'],
  scalar:['#cbd5e1','#1e293b'], error:['#fca5a5','#3b0d0d'], truncated:['#94a3b8','#1e293b'],
  group:['#94a3b8','#1e293b']
};
const darkMQ = window.matchMedia('(prefers-color-scheme: dark)');
let COLORS = darkMQ.matches ? DARK : LIGHT;
let LINK   = darkMQ.matches ? '#334155' : '#cbd5e1';
let MUTED  = darkMQ.matches ? '#94a3b8' : '#64748b';
// SVG presentation attributes do not resolve CSS var(), so these are literals.
let HALO   = darkMQ.matches ? '#f59e0b' : '#fbbf24';
let CARD   = darkMQ.matches ? '#111827' : '#ffffff';
let SHAPE_C = darkMQ.matches ? '#cbd5e1' : '#334155';   // shapes read a shade stronger
let HINT_C  = darkMQ.matches ? '#fbbf24' : '#b45309';   // "this looks like a label"
let ACCENT  = darkMQ.matches ? '#60a5fa' : '#2563eb';   // selection ring
const KIND_LABEL = {root:'根',folder:'目录',file:'文件',array:'数组',
  object_array:'object 数组',dict:'dict',list:'list',tuple:'tuple',
  scalar:'标量',error:'错误',truncated:'省略',group:'同型组'};

/* -------------------------------------------------------- 同型兄弟合并 */
/* model_000 … model_059 sharing one shape and dtype is one fact, not sixty:
   as a flat column it buries the structure you came here to read. Runs of
   >= GROUP_MIN identical siblings fold into a single row that still opens,
   so the detail stays one click away instead of gone. Off = show every row. */
const GROUP_MIN = 8;
let grouping = true;

/* Only leaf arrays with a real shape+dtype are groupable. Containers are not:
   two dicts both reading "dict · 3 键" can hold completely different things,
   so folding them would claim a sameness that isn't there. */
function sigOf(n){
  if(n.kind === 'truncated' || n.kind === 'error' || n.kind === 'root') return null;
  if((n.children || []).length) return null;
  const st = n.stats || {};
  if(!st.shape) return null;
  return n.kind + '|' + st.shape.join(',') + '|' + (st.dtype || '');
}
function commonPrefix(xs){
  let p = xs[0];
  for(const s of xs){
    let i = 0;
    while(i < p.length && i < s.length && p[i] === s[i]) i++;
    p = p.slice(0, i);
  }
  return p;
}
function groupChildren(kids){
  const out = [];
  let i = 0;
  while(i < kids.length){
    const sig = sigOf(kids[i]);
    let j = i + 1;
    if(sig) while(j < kids.length && sigOf(kids[j]) === sig) j++;
    // j is >= i+1 by construction, so it is the fixed upper bound. Using
    // Math.max(j, i+1) here reads like the same thing but is not: once i
    // reaches j the bound becomes i+1, and the loop never terminates.
    if(!sig || j - i < GROUP_MIN){
      for(; i < j; i++) out.push(kids[i]);
      continue;
    }
    const run = kids.slice(i, j);
    const names = run.map(x => x.name);
    const pfx = commonPrefix(names);
    const first = names[0], last = names[names.length - 1];
    const g = {
      name: (pfx.length >= 3 ? pfx + '*' : first + ' … ' + last) + '  ×' + run.length,
      kind: 'group', detail: run[0].detail, _grp: run.length,
      _range: first + ' … ' + last, children: run
    };
    if(run[0].stats) g.stats = run[0].stats;
    if(run[0].hint) g.hint = run[0].hint;
    out.push(g);
    i = j;
  }
  return out;
}
function applyGrouping(n){
  if(!n.children || !n.children.length) return n;
  n.children = groupChildren(n.children);
  // Never descend into a group: its members are by definition an identical run,
  // so re-grouping them would fold the same 60 keys into a group of one, again
  // and again, until the stack runs out.
  n.children.forEach(c => { if(!c._grp) applyGrouping(c); });
  return n;
}

/* ------------------------------------------------------------ node index */
const NODES = [];
let collapsed, MAX_DEPTH;
function prep(n, d, parent){
  n._d = d; n._id = NODES.length; n._parent = parent;
  NODES.push(n);
  (n.children || []).forEach(c => prep(c, d + 1, n));
}
function rebuildTree(){
  ROOT = grouping ? applyGrouping(structuredClone(RAW)) : structuredClone(RAW);
  NODES.length = 0;
  prep(ROOT, 0, null);
  MAX_DEPTH = NODES.reduce((m, n) => Math.max(m, n._d), 0);
  collapsed = new Set(NODES.filter(n =>
    (n.children || []).length && n._d >= 2).map(n => n._id));
  if(selected >= 0 && !NODES[selected]) selected = -1;
}

const textCtx = document.createElement('canvas').getContext('2d');
const F_NAME = '650 12.5px ui-monospace, Consolas, monospace';
const F_DET  = '11px ui-monospace, Consolas, monospace';
function measure(s, font){ textCtx.font = font; return textCtx.measureText(s).width; }

const BOX_H = 24, PAD_X = 9, GAP_X = 46, GAP_Y = 7, R = 8, RAD = 7;
const SEG_GAP = 7;
let highlighted = new Set();
let selected = -1;        // node id whose index recipe is showing, -1 = none

/* ------------------------------------------------------------- text utils */
function esc(s){
  return String(s).replace(/[&<>"]/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}
function kidsOf(n){ return collapsed.has(n._id) ? [] : (n.children || []); }
function isVisible(n){
  for(let p = n._parent; p; p = p._parent) if(collapsed.has(p._id)) return false;
  return true;
}
function matched(n){
  return highlighted.has(n._id);
}
function fmtShape(sh){
  const body = sh.map(x => String(x)).join(', ');
  return sh.length === 1 ? '(' + body + ',)' : '(' + body + ')';
}
/* Split the second line into styled pieces: the shape is what the eye hunts
   for, so it must not sit at the same weight as the dtype next to it. */
function segsOf(n){
  const st = n.stats || {};
  const out = [];
  if(st.shape) out.push([fmtShape(st.shape), 'shape']);
  if(st.dtype) out.push([String(st.dtype), 'dtype']);
  if(!out.length && n.detail) out.push([n.detail, 'plain']);
  if(n.hint) out.push([n.hint, 'hint']);
  return out;
}
function segColor(k){
  if(k === 'shape') return SHAPE_C;
  if(k === 'hint')  return HINT_C;
  return MUTED;
}

/* ---------------------------------------------------------------- layout */
let LAST = null;
function layout(){
  for(const n of NODES){
    n._nw = measure(n.name, F_NAME);
    n._segs = segsOf(n);
    n._sw = n._segs.map(s => measure(s[0], F_DET));
    // SEG_GAP is a drawn offset, not spaces: SVG <text> collapses whitespace
    // runs, so a space-padded string measures wider than it renders and the
    // name would collide with its own detail.
    // The toggle circle straddles the right edge, so reserve R for it or it
    // sits on top of the tail of the detail text.
    const extra = (n.children || []).length ? R : 0;
    n._bw = PAD_X * 2 + extra + n._nw
          + n._sw.reduce((a, w) => a + w + SEG_GAP, 0);
  }
  const vis = NODES.filter(isVisible);
  const maxD = vis.reduce((m, n) => Math.max(m, n._d), 0);
  const colw = new Array(maxD + 1).fill(0);
  for(const n of vis) colw[n._d] = Math.max(colw[n._d], n._bw);

  const x0 = new Array(maxD + 1).fill(0);
  let acc = 0;
  for(let d = 0; d <= maxD; d++){ x0[d] = acc; acc += colw[d] + GAP_X; }

  let cur = 0;
  (function assign(n){
    const kids = kidsOf(n);
    if(!kids.length){ n._y = cur; cur += BOX_H + GAP_Y; }
    else {
      kids.forEach(assign);
      n._y = (kids[0]._y + kids[kids.length - 1]._y) / 2;
    }
  })(ROOT);

  LAST = {vis, x0, width: acc - GAP_X, height: cur - GAP_Y, maxD};
  return LAST;
}

/* ---------------------------------------------------------------- render */
const svg = document.getElementById('svg');
const stage = document.getElementById('stage');
let view = {x: 30, y: 30, k: 1};

function applyView(){
  stage.setAttribute('transform',
    'translate(' + view.x + ' ' + view.y + ') scale(' + view.k + ')');
}

function render(){
  const L = layout();
  const M = 12;
  let out = '';

  for(const n of L.vis){
    n._x = L.x0[n._d] + M;
    n._yy = n._y + M;
  }
  for(const n of L.vis){
    const x1 = n._x + n._bw + R, y1 = n._yy;
    for(const c of kidsOf(n)){
      const x2 = c._x, y2 = c._yy;
      const dx = Math.max(12, (x2 - x1) * 0.5);
      out += '<path class="link" stroke="' + LINK + '" d="M' + x1 + ' ' + y1 +
             ' C' + (x1 + dx) + ' ' + y1 + ' ' + (x2 - dx) + ' ' + y2 +
             ' ' + x2 + ' ' + y2 + '"/>';
    }
  }

  for(const n of L.vis){
    const c = COLORS[n.kind] || COLORS.scalar;
    const hasKids = (n.children || []).length > 0;
    const isCol = collapsed.has(n._id);
    out += '<g class="nd' + (hasKids ? ' has-kids' : '') + '" data-id="' + n._id + '">';
    if(matched(n)){
      out += '<rect x="' + (n._x - 3.5) + '" y="' + (n._yy - BOX_H/2 - 3.5) +
             '" width="' + (n._bw + 7) + '" height="' + (BOX_H + 7) +
             '" rx="9" fill="none" stroke="' + HALO + '" stroke-width="2.5" opacity="0.95"/>';
    }
    // Selection ring sits a shade further out than the search halo so the two
    // can be read at once when you search and then click a hit.
    if(n._id === selected){
      out += '<rect x="' + (n._x - 6) + '" y="' + (n._yy - BOX_H/2 - 6) +
             '" width="' + (n._bw + 12) + '" height="' + (BOX_H + 12) +
             '" rx="11" fill="none" stroke="' + ACCENT + '" stroke-width="2"/>';
    }
    // object arrays get a dashed edge: they look like leaves but hide a tree
    const dash = n.kind === 'object_array' ? ' stroke-dasharray="4 2.6"' : '';
    out += '<rect x="' + n._x + '" y="' + (n._yy - BOX_H/2) + '" width="' + n._bw +
           '" height="' + BOX_H + '" rx="' + RAD + '" fill="' + c[1] +
           '" stroke="' + c[0] + '" stroke-width="1.1"' + dash + '/>';
    out += '<text class="nm" x="' + (n._x + PAD_X) + '" y="' + n._yy +
           '" fill="' + c[0] + '">' + esc(n.name) + '</text>';
    let cx = n._x + PAD_X + n._nw;
    n._segs.forEach((s, i) => {
      cx += SEG_GAP;
      out += '<text class="dt" x="' + cx + '" y="' + n._yy + '" fill="' +
             segColor(s[1]) + '">' + esc(s[0]) + '</text>';
      cx += n._sw[i];
    });
    if(hasKids){
      out += '<circle class="tgcircle" data-id="' + n._id + '" cx="' +
             (n._x + n._bw) + '" cy="' + n._yy + '" r="' + R + '" fill="' + CARD + '" stroke="' +
             c[0] + '" stroke-width="1.2"/>';
      out += '<text class="tg" data-id="' + n._id + '" x="' + (n._x + n._bw) +
             '" y="' + n._yy + '" fill="' + c[0] + '">' + (isCol ? '+' : '−') + '</text>';
    }
    out += '</g>';
  }
  stage.innerHTML = out;
}

function fit(){
  const box = document.getElementById('diagram').getBoundingClientRect();
  const L = LAST || layout();
  const k = Math.min((box.width - 44) / Math.max(L.width, 1),
                     (box.height - 44) / Math.max(L.height, 1), 2.2);
  view.k = Math.max(k, 0.12);
  view.x = (box.width - L.width * view.k) / 2;
  view.y = (box.height - L.height * view.k) / 2;
  applyView();
}

function redraw(refit){
  render();
  if(refit) fit(); else applyView();
}

/* ------------------------------------------------------------ interaction */
let drag = null, moved = 0;
svg.addEventListener('pointerdown', e => {
  drag = {sx: e.clientX, sy: e.clientY, vx: view.x, vy: view.y};
  moved = 0;
  document.getElementById('diagram').classList.add('grabbing');
});
svg.addEventListener('pointermove', e => {
  if(!drag) return;
  const dx = e.clientX - drag.sx, dy = e.clientY - drag.sy;
  moved = Math.max(moved, Math.abs(dx) + Math.abs(dy));
  view.x = drag.vx + dx; view.y = drag.vy + dy;
  applyView();
});
function endDrag(){
  drag = null;
  document.getElementById('diagram').classList.remove('grabbing');
}
svg.addEventListener('pointerup', endDrag);
svg.addEventListener('pointercancel', endDrag);
svg.addEventListener('pointerleave', endDrag);

svg.addEventListener('click', e => {
  if(moved > 4) return;                       // a pan, not a click
  const hit = e.target.closest('[data-id]');
  if(!hit) return;
  const n = NODES[+hit.dataset.id];
  if(!n) return;
  // The circle is the only thing that folds a subtree; the box body selects,
  // so "look at this level" and "open this level" are separate gestures.
  if(e.target.closest('.tg,.tgcircle') && (n.children || []).length){
    if(collapsed.has(n._id)) collapsed.delete(n._id); else collapsed.add(n._id);
    render();
    applyView();
    return;
  }
  select(n._id);
});

svg.addEventListener('wheel', e => {
  e.preventDefault();
  const box = svg.getBoundingClientRect();
  const mx = e.clientX - box.left, my = e.clientY - box.top;
  const k2 = Math.min(4, Math.max(0.1, view.k * Math.exp(-e.deltaY * 0.0015)));
  const s = k2 / view.k;
  view.x = mx - (mx - view.x) * s;
  view.y = my - (my - view.y) * s;
  view.k = k2;
  applyView();
}, {passive: false});

/* --------------------------------------------------------- expand/collapse */
const DEPTH_DEFAULT = 2, DEPTH_MAX = 12;

/* "展开到第 N 层": levels 0..N visible, anything at depth >= N folds away. */
function expandToDepth(d){
  collapsed.clear();
  NODES.forEach(n => {
    if((n.children || []).length && n._d >= d) collapsed.add(n._id);
  });
  render(); applyView(); syncListDepth(d);
  // Show the depth actually in effect, so "全部展开" cannot leave the box
  // claiming 2 while the whole tree is open.
  const shown = Math.min(d, MAX_DEPTH);
  const inp = document.getElementById('depth');
  if(inp.value !== String(shown)) inp.value = shown;
}
function syncListDepth(d){
  treeEl.querySelectorAll('li.node').forEach(li => {
    if(!li.querySelector(':scope > ul')) return;
    const n = NODES[+li.dataset.nid];
    const col = n._d >= d;
    li.classList.toggle('collapsed', col);
    const tw = li.querySelector(':scope > .row > .tw');
    if(tw) tw.textContent = col ? '▸' : '▾';
  });
}

document.getElementById('expand').onclick = () => {
  highlighted.clear();
  document.getElementById('q').value = '';
  expandToDepth(999);
  fit();
};
document.getElementById('collapse').onclick = () => { expandToDepth(1); fit(); };
document.getElementById('depth').addEventListener('change', e => {
  const raw = parseInt(e.target.value, 10);
  const d = Math.max(0, Math.min(DEPTH_MAX, isNaN(raw) ? DEPTH_DEFAULT : raw));
  e.target.value = d;
  expandToDepth(d);
  fit();
});
document.getElementById('fit').onclick = fit;
const focusBtn = document.getElementById('focus');
focusBtn.onclick = () => {
  const on = document.body.classList.toggle('focus');
  focusBtn.classList.toggle('on', on);
  fit();
};

/* ----------------------------------------------------------------- search */
function applySearch(qRaw){
  const q = qRaw.trim().toLowerCase();
  highlighted = new Set();
  if(!q){ redraw(false); syncList(); return; }

  for(const n of NODES){
    const hay = (n.name + ' ' + (n.detail || '') + ' ' + (n.kind || '')).toLowerCase();
    if(!hay.includes(q)) continue;
    highlighted.add(n._id);
    for(let p = n._parent; p; p = p._parent) collapsed.delete(p._id);  // reveal it
  }
  redraw(false);
  syncList();

  const first = NODES.find(n => highlighted.has(n._id));
  if(first && isVisible(first)){
    const box = document.getElementById('diagram').getBoundingClientRect();
    view.x = box.width / 2 - (first._x + first._bw / 2) * view.k;
    view.y = box.height / 2 - first._yy * view.k;
    applyView();
  }
}
document.getElementById('q').addEventListener('input', e => applySearch(e.target.value));

/* --------------------------------------------------------- 参考索引 (index) */
/* Turns the root->node chain into a Jupyter cell. Each in-memory hop needs the
   subscript the scanner recorded in `acc`:
     "[0]"      object-array element / list item
     "['key']"  dict or npz key
     ""         the node *is* its parent's value (.npy holds its array directly)
     null       the hop was never expanded, so no subscript can be given
   The filesystem hops are special: they are where the data enters memory, and
   .npz and .npy differ in what np.load hands back. */
function pyIdent(s, fallback){
  const t = String(s).replace(/[^0-9A-Za-z_一-鿿]+/g, '_')
                     .replace(/^_+|_+$/g, '');
  return (t && !/^[0-9]/.test(t)) ? t : fallback;
}
/* A name that sanitises away (an object-array element is literally "[0]")
   reads far better as parent + index than as a generic "v". */
function childVar(parent, name){
  const clean = pyIdent(name, null);
  if(clean) return clean;
  const idx = String(name).replace(/\D+/g, '_').replace(/_+/g, '_')
                          .replace(/^_+|_+$/g, '');
  return parent + '_' + (idx || 'item');
}
/* r'...' keeps Windows backslashes readable. A raw string can't end in a
   backslash or contain its own quote, so fall back when it would. */
function pyPath(p){
  p = String(p || '');
  if(!/\\$/.test(p)){
    if(p.indexOf("'") < 0) return "r'" + p + "'";
    if(p.indexOf('"') < 0) return 'r"' + p + '"';
  }
  return JSON.stringify(p);
}
/* Synthetic 同型 groups carry no subscript of their own, so they are skipped:
   a member indexes straight off the npz/dict that really holds it. */
function chainOf(n){
  const c = [];
  for(let p = n; p; p = p._parent) if(!p._grp) c.unshift(p);
  return c;
}
/* The "." sibling is the dataset root itself; "data › . › x.npz" reads worse
   than "data › x.npz" and says nothing extra. */
function crumbOf(n){
  return chainOf(n).map(x => x.name).filter(s => s !== '.').join(' › ');
}
function noteOf(x){
  const st = x.stats || {}, b = [];
  if(st.shape) b.push(fmtShape(st.shape));
  if(st.dtype) b.push(String(st.dtype));
  const s = b.join(' ');
  if(x.hint) return (s ? s + '  ·  ' : '') + x.hint;
  return s || (x.detail || '');
}

function indexCell(target){
  const chain = chainOf(target);
  const used = new Set();
  const uniq = base => { let v = base, k = 2;
    while(used.has(v)) v = base + '_' + (k++); used.add(v); return v; };

  const lines = [];          // {code, note, cls, hot}
  let cur = null, defLine = -1;

  for(const x of chain){
    if(x.kind === 'root' || x.kind === 'folder'){
      // The "." group is the dataset root itself; re-assigning the same path
      // to a second variable is pure noise in the cell.
      if(cur && x.path === chain[0].path) continue;
      cur = uniq(x.kind === 'root' ? 'root' : pyIdent(x.name, 'sub'));
      lines.push({code: cur + ' = ' + pyPath(x.path),
                  note: x.kind === 'root' ? '数据集根目录' : '子目录'});
      defLine = lines.length - 1;
      continue;
    }
    if(x.kind === 'file'){
      const f = uniq('f');
      lines.push({code: f + ' = ' + pyPath(x.path), note: ''});
      if(x.npz){
        cur = uniq('z');
        lines.push({code: cur + ' = np.load(' + f + ', allow_pickle=True)',
                    note: 'NpzFile · ' + (x.children || []).length + ' 个键'});
      }else{
        // a .npy file *is* one array, so name the variable after the file
        cur = uniq(pyIdent(String(x.name).replace(/\.npy$/i, ''), 'a'));
        lines.push({code: cur + ' = np.load(' + f + ', allow_pickle=True)',
                    note: '整份 .npy 就是这个数组'});
      }
      defLine = lines.length - 1;
      continue;
    }
    if(x.acc == null){
      lines.push({code: '# ' + x.name + '  ——  ' + (x.detail || '未展开'),
                  note: '这一层没展开，给不出下标', cls: 'dim'});
      cur = null; defLine = -1;
      continue;
    }
    if(x.acc === ''){        // .npy 的数组子节点就是上面那个 np.load 的结果
      // defLine is -1 only if no np.load line was ever emitted -- reachable
      // when the node has no parent chain to walk (a detached subtree).
      if(x === target && defLine >= 0) lines[defLine].note = noteOf(x);
      continue;
    }
    const v = uniq(childVar(cur, x.name));
    lines.push({code: v + ' = ' + cur + x.acc, note: noteOf(x)});
    cur = v; defLine = lines.length - 1;
  }

  if(target._grp){
    // a 同型 group is a view, not a thing you can name in Python
    const m = (target.children || [])[0];
    lines.push({code: '# ' + target._range + '  ——  共 ' + target._grp + ' 个同型键',
                note: '各自独立索引', cls: 'dim'});
    if(cur && m && m.acc != null)
      lines.push({code: cur + m.acc, note: '例：组的第一个成员', cls: 'dim'});
  } else if(cur == null){
    lines.push({code: '# 这一段没有可用的下标', note: '', cls: 'dim'});
  }

  const head = ['import numpy as np'];
  if(target.kind === 'root' || target.kind === 'folder'){
    head.push('import os');
    lines.push({code: 'sorted(os.listdir(' + cur + '))',
                note: '这一层有什么', cls: 'dim'});
  }
  if(defLine >= 0) lines[defLine].hot = true;
  return {head, lines};
}

const idxCode = document.getElementById('idxcode');
const crumbEl = document.getElementById('crumb');
const copyBtn = document.getElementById('idxcopy');
let cellText = '';

function renderIndex(){
  const n = selected >= 0 ? NODES[selected] : null;
  idxCode.textContent = '';
  if(!n){
    // The empty state rides in the header row rather than taking a block of
    // its own: the diagram is what the page is for, and every idle row above
    // it is height taken from the map.
    crumbEl.textContent = '点任意方块（导图或列表），这里给出在 Jupyter 里索引到它的代码';
    crumbEl.classList.add('iscrumb');
    crumbEl.title = '';
    cellText = '';
    idxCode.classList.add('hidden');
    copyBtn.classList.add('hidden');
    return;
  }
  const cell = indexCell(n);
  crumbEl.textContent = crumbOf(n);
  crumbEl.title = crumbOf(n);
  crumbEl.classList.remove('iscrumb');
  idxCode.classList.remove('hidden');
  copyBtn.classList.remove('hidden');

  const rows = cell.head.map(h => ({code: h}))
    .concat([{code: ''}], cell.lines);
  cellText = rows.map(l => l.code + (l.note ? '  # ' + l.note : '')).join('\n');

  for(const l of rows){
    const s = el('span', 'ln' + (l.cls ? ' ' + l.cls : '') + (l.hot ? ' hot' : ''));
    s.appendChild(el('span', null, l.code));
    if(l.note) s.appendChild(el('span', 'cm', '   # ' + l.note));
    if(l.hot) s.appendChild(el('span', 'tgt', '   ← 目标'));
    idxCode.appendChild(s);
  }
}

function select(id){
  selected = id;
  renderIndex();
  if(document.getElementById('treewrap').classList.contains('hidden')) render();
  else syncList();
  applyView();
  const li = treeEl.querySelector('li.node[data-nid="' + id + '"]');
  if(li && !li.classList.contains('collapsed')) li.scrollIntoView({block: 'nearest'});
}

copyBtn.onclick = () => {
  const done = ok => {
    copyBtn.textContent = ok ? '已复制 ✓' : '复制失败，请手动选中';
    setTimeout(() => { copyBtn.textContent = '复制代码'; }, 1600);
  };
  if(navigator.clipboard && window.isSecureContext){
    navigator.clipboard.writeText(cellText).then(() => done(true), () => done(false));
    return;
  }
  // A file:// page is not a secure context, so the async clipboard API is
  // unavailable exactly where this tool is most often opened.
  try{
    const ta = document.createElement('textarea');
    ta.value = cellText;
    ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    done(ok);
  }catch(e){ done(false); }
};

/* -------------------------------------------------------------- list view */
function el(tag, cls, txt){
  const e = document.createElement(tag);
  if(cls) e.className = cls;
  if(txt != null) e.textContent = txt;
  return e;
}
function fmtNum(v){
  if(typeof v !== 'number') return String(v);
  if(v !== 0 && Math.abs(v) < 1e-3) return v.toExponential(2);
  return String(v);
}
/* Statistics only. shape/dtype live in the styled segments just before this --
   repeating them here was pure noise. min/max/mean are suppressed for booleans
   and for anything already flagged as a label, where they say nothing. */
function statsText(n){
  const s = n.stats; if(!s) return '';
  if(s.empty) return '空数组';
  const bits = [];
  if(s.dtype !== 'bool' && !n.hint && s.min !== undefined)
    bits.push('min=' + fmtNum(s.min), 'max=' + fmtNum(s.max),
              'mean=' + fmtNum(s.mean));
  // a unique count near the element count just means "all distinct"
  if(s.n_unique !== undefined && !s.unique_values && s.size &&
     s.n_unique < s.size * 0.5)
    bits.push('uniq=' + s.n_unique);
  if(s.nan_count) bits.push('nan=' + s.nan_count);
  if(s.sampled) bits.push('抽样统计');
  return bits.join('  ');
}
function build(node, depth){
  const li = el('li', 'node k-' + node.kind);
  const row = el('div', 'row');
  const kids = node.children || [];
  if(kids.length) row.classList.add('has-kids');
  const tw = el('span', 'tw', kids.length ? (depth < DEPTH_DEFAULT ? '▾' : '▸') : '');
  row.appendChild(tw);
  if(node.kind && node.kind !== 'root' && node.kind !== 'folder')
    row.appendChild(el('span', 'badge', KIND_LABEL[node.kind] || node.kind));
  row.appendChild(el('span', 'nm2', node.name));
  segsOf(node).forEach(s => {
    const cls = s[1] === 'hint' ? 'dt2 hint' : (s[1] === 'shape' ? 'dt2 sh2' : 'dt2');
    row.appendChild(el('span', cls, s[0]));
  });
  const st = statsText(node);
  if(st) row.appendChild(el('span', 'meta', st));
  if(node.stats && node.stats.unique_values)
    row.appendChild(el('span', 'meta', '→ ' + JSON.stringify(node.stats.unique_values)));
  if(node.stats && node.stats.sample)
    row.appendChild(el('span', 'meta', '→ ' + node.stats.sample.join(' | ')));
  li.appendChild(row);
  if(kids.length){
    const ul = el('ul');
    kids.forEach(c => ul.appendChild(build(c, depth + 1)));
    li.appendChild(ul);
    // Same split as the map: the twisty folds, the row body selects.
    tw.addEventListener('click', ev => {
      ev.stopPropagation();
      li.classList.toggle('collapsed');
      tw.textContent = li.classList.contains('collapsed') ? '▸' : '▾';
    });
  }
  row.addEventListener('click', () => select(node._id));
  li.dataset.nid = node._id;
  return li;
}

const treeEl = document.getElementById('tree');
function rebuildList(){
  treeEl.textContent = '';
  treeEl.appendChild(build(ROOT, 0));
  syncList();
}

function syncList(){
  treeEl.querySelectorAll('li.node').forEach(li => {
    const n = NODES[+li.dataset.nid];
    const hit = n && highlighted.has(n._id);
    li.classList.toggle('hit', hit);
    li.classList.toggle('sel', n && n._id === selected);
    if(!hit) return;
    // open every ancestor row so the hit is actually on screen
    let p = li.parentElement.closest('li');
    while(p){
      p.classList.remove('collapsed');
      const tw = p.querySelector(':scope > .row > .tw');
      if(tw) tw.textContent = '▾';
      p = p.parentElement.closest('li');
    }
  });
}

/* ---------------------------------------------------------------- hints */
/* Chips double as navigation: clicking one jumps the map to that array, so
   the hint list is a way into the structure rather than a label to read. */
const hints = document.getElementById('hints');
const agg = ROOT._agg || {};
if((agg.label_hints || []).length){
  hints.appendChild(el('span', 'hlabel', '疑似标签'));
  agg.label_hints.slice(0, 25).forEach(h => {
    const nm = h.split('  —  ')[0];
    const c = el('button', 'chip', nm);
    c.title = h + '　（点击定位）';
    c.addEventListener('click', () => {
      document.getElementById('q').value = nm;
      applySearch(nm);              // opens the ancestors so it is on screen
      const n = NODES.find(x => x.name === nm && x.hint);
      if(n) select(n._id);
    });
    hints.appendChild(c);
  });
} else hints.classList.add('hidden');

/* ------------------------------------------------------------ view switch */
const diagramEl = document.getElementById('diagram');
const treeWrap = document.getElementById('treewrap');
function setView(which){
  const map = which === 'map';
  diagramEl.classList.toggle('hidden', !map);
  treeWrap.classList.toggle('hidden', map);
  document.getElementById('v-map').classList.toggle('on', map);
  document.getElementById('v-list').classList.toggle('on', !map);
  if(map) fit();
}
document.getElementById('v-map').onclick = () => setView('map');
document.getElementById('v-list').onclick = () => setView('list');

/* ------------------------------------------------------------------ boot */
const grpBtn = document.getElementById('grp');
grpBtn.onclick = () => {
  grouping = !grouping;
  grpBtn.classList.toggle('on', grouping);
  rebuildTree();
  rebuildList();
  document.getElementById('depth').max = Math.max(MAX_DEPTH, 1);
  expandToDepth(Math.min(+document.getElementById('depth').value || DEPTH_DEFAULT,
                         MAX_DEPTH));
  renderIndex();
  fit();
};

rebuildTree();
rebuildList();
renderIndex();            // fills in the idle hint before anything is selected
document.getElementById('depth').max = Math.max(MAX_DEPTH, 1);
expandToDepth(DEPTH_DEFAULT);
fit();
if(location.hash === '#list') setView('list');   // deep-link / headless check
darkMQ.addEventListener('change', e => {
  COLORS  = e.matches ? DARK : LIGHT;
  LINK    = e.matches ? '#334155' : '#cbd5e1';
  MUTED   = e.matches ? '#94a3b8' : '#64748b';
  HALO    = e.matches ? '#f59e0b' : '#fbbf24';
  CARD    = e.matches ? '#111827' : '#ffffff';
  SHAPE_C = e.matches ? '#cbd5e1' : '#334155';
  HINT_C  = e.matches ? '#fbbf24' : '#b45309';
  ACCENT  = e.matches ? '#60a5fa' : '#2563eb';
  render(); applyView();
});
window.addEventListener('resize', () => fit());
</script>
</body>
</html>
"""


def render_html(root: dict, out: Path) -> None:
    agg = root.get("_agg", {})
    payload = json.dumps(root, ensure_ascii=False, default=str)
    payload = payload.replace("</", "<\\/")

    stats = [
        f"<span><b>{agg.get('files', 0)}</b> 个文件</span>",
        f"<span><b>{agg.get('arrays', 0)}</b> 个数组</span>",
        f"<span><b>{_human(agg.get('bytes', 0))}</b> 总大小</span>",
    ]
    if agg.get("errors"):
        stats.append(f"<span style='color:#b91c1c'><b>{len(agg['errors'])}</b> 个文件读取失败</span>")

    title = html.escape(f"{root['name']} 数据结构图")
    doc = (HTML_TEMPLATE
           .replace("__TITLE__", title)
           .replace("__STATS__", "".join(stats))
           .replace("__PAYLOAD__", payload))
    out.write_text(doc, encoding="utf-8")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    # Windows consoles default to cp936; without this the Chinese summary prints
    # as mojibake and the user cannot read which arrays were flagged.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="Scan .npy/.npz datasets and render a structure mind map.")
    ap.add_argument("path", help="dataset directory or single .npy/.npz file")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <path>_structure_map)")
    ap.add_argument("--name", default=None, help="output basename")
    ap.add_argument("--max-depth", type=int, default=6,
                    help="max recursion depth into object arrays / dicts (default 6)")
    ap.add_argument("--max-items", type=int, default=25,
                    help="max children expanded per node (default 25)")
    ap.add_argument("--max-files", type=int, default=400,
                    help="max files scanned (default 400)")
    ap.add_argument("--no-pickle", dest="allow_pickle", action="store_false",
                    help="refuse object arrays (safer, but skips nested structures)")
    ap.add_argument("--formats", default="html,png,json",
                    help="comma list of html,png,json (default all)")
    ap.add_argument("--collapse", type=int, default=8, metavar="N",
                    help="PNG only: merge runs of >=N identically-shaped leaf "
                         "siblings into one row (default 8, 0 disables)")
    ap.add_argument("--open", dest="open_what", default="auto",
                    choices=["auto", "html", "png", "all", "none"],
                    help="open the result when done. 'auto' (the default) opens "
                         "the HTML when running in a terminal and does nothing "
                         "when piped/redirected, so scripts stay quiet")
    ap.set_defaults(allow_pickle=True)
    args = ap.parse_args(argv)

    root_path = Path(args.path).expanduser().resolve()
    if not root_path.exists():
        print(f"路径不存在: {root_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out).expanduser() if args.out else \
        root_path.parent / f"{root_path.stem or root_path.name}_structure_map"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.name or (root_path.stem or root_path.name) + "_structure_map"

    tree = build_tree(root_path, args)
    formats = {f.strip().lower() for f in args.formats.split(",") if f.strip()}
    made = []

    json_path = out_dir / f"{base}.json"
    json_path.write_text(json.dumps(tree, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
    made.append(json_path)

    if "html" in formats:
        p = out_dir / f"{base}.html"
        render_html(tree, p)
        made.append(p)
    if "png" in formats:
        p = out_dir / f"{base}.png"
        try:
            render_png(tree, p, collapse=args.collapse)
            made.append(p)
        except Exception as e:
            print(f"PNG 渲染失败（HTML/JSON 已生成）: {type(e).__name__}: {e}",
                  file=sys.stderr)

    agg = tree.get("_agg", {})
    print(f"扫描完成: {agg.get('files', 0)} 文件 / "
          f"{agg.get('arrays', 0)} 数组 / {_human(agg.get('bytes', 0))}")
    if agg.get("label_hints"):
        print("疑似标签数组:")
        for h in agg["label_hints"][:15]:
            print(f"  - {h}")
    if agg.get("errors"):
        print(f"读取失败 {len(agg['errors'])} 个: {', '.join(agg['errors'][:5])}")
    for p in made:
        print(f"  -> {p}")

    _maybe_open(args.open_what, made)
    return 0


def _maybe_open(what: str, made: list) -> None:
    """Pop the result open. `auto` means: do it when a human is watching."""
    if what == "none":
        return
    if what == "auto":
        # Only when a human is watching. Piped or redirected output means this
        # is running inside a script, and popping a browser there is a surprise.
        if not sys.stdout.isatty():
            return
        what = "html"
    want = []
    if what in ("html", "all"):
        want.append(".html")
    if what in ("png", "all"):
        want.append(".png")
    for suffix in want:
        for p in made:
            if p.suffix != suffix:
                continue
            try:
                if sys.platform == "win32":
                    os.startfile(str(p))          # noqa: S606 -- user's own file
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(p)])
                else:
                    subprocess.Popen(["xdg-open", str(p)])
            except Exception as e:
                print(f"  （自动打开 {p.name} 失败: {type(e).__name__}，"
                      f"手动打开即可）", file=sys.stderr)
            break


if __name__ == "__main__":
    raise SystemExit(main())
