---
name: npy-inspector
description: Scan an unfamiliar .npy/.npz dataset directory and render its structure as an interactive mind map (collapsible HTML + static PNG + JSON schema). Recurses into npz keys, object arrays, and nested dicts/lists so you can see what each array holds, its shape/dtype/range, and which arrays look like labels. Use when the user points at a folder of npy/npz files and asks what's in it, how it's organized, what a given array contains, or asks for a data-structure diagram.
---

# npy-inspector

Turns an opaque folder of `.npy` / `.npz` files into a structure map you can
read at a glance. Built for the case where you did **not** create the dataset
and have no README: the script has to discover the organization itself.

## Cost

Scanning cost does not scale with dataset size, which is worth saying out loud
when a user hesitates to point it at a 400 GB folder. The scan reads headers and
a strided sample, never the bulk data, so what comes back describes structure:

| Dataset | `stdout` | JSON |
|---|---|---|
| 6.5 GB, 1 array | 4 lines, 261 chars | 1.5 KB |
| 1.0 MB, 73 arrays, nested | 9 lines, 377 chars | 91.3 KB |

The stdout summary is capped at ~20 lines regardless (`label_hints[:15]`,
`errors[:5]`, three artifact paths). What costs tokens is the **node count** —
files × arrays × depth — bounded by `--max-files` and `--max-depth`. The one
shape that can still grow the JSON is a single npz with tens of thousands of
keys, since npz keys are always listed in full.

## When to use

- "I have a dataset on drive D full of npy/npz, I don't know its structure."
- "What's inside `data.npz`?" / "Which array is the labels?"
- "Draw me a mind map / 思维导图 / structure diagram of this dataset."
- Before writing any analysis code against an unfamiliar dataset.

## Run it

```bash
python ~/.claude/skills/npy-inspector/scripts/inspect_npy.py <path-to-dataset>
```

`<path>` is a directory (scanned recursively) or a single `.npy`/`.npz` file.

Outputs land in `<path>_structure_map/` alongside the dataset:

| File | Use |
|---|---|
| `<name>_structure_map.html` | **Primary artifact.** Self-contained, no CDN. Two views: an interactive mind map (circles with `+`/`−` expand one level, wheel-zoom, drag-pan, click any box to select) and a flat list view. Both share the search box, the depth control, and the 参考索引 bar. |
| `<name>_structure_map.png` | Static mind map — rounded node boxes, bezier links, kind legend. For embedding into a `.docx`/slides. |
| `<name>_structure_map.json` | Machine-readable schema — read this yourself to answer follow-up questions without re-scanning. |

Opening `<name>_structure_map.html#list` starts in the list view instead of the
map (handy for deep-link sharing and for headless screenshots).

Run from a terminal it opens the HTML by itself; run with the output piped or
redirected it stays quiet. `说明书.docx` in this folder is the user-facing
manual (Chinese) — hand it over as-is rather than re-describing the tool.

Useful flags:

| Flag | Meaning |
|---|---|
| `--out DIR` | Write elsewhere (e.g. the user's Desktop). |
| `--max-depth N` | Recursion ceiling into object arrays / dicts (default 6). Raise for deeply nested session structs. |
| `--max-items N` | Children expanded per node (default 25). Raise if a dict has 60 keys and you need them all. |
| `--max-files N` | Cap on files scanned (default 400). Never silent — see below. |
| `--max-array-mb MB` | Arrays larger than this are mapped from their `.npy` header but **never loaded** (default 256). Shape/dtype/size stay exact; only the value statistics are skipped. Applies to npz members and object-dtype `.npy`; numeric `.npy` is memory-mapped and unaffected. |
| `--no-pickle` | Refuse object arrays. **Safer, but nested structures go unexpanded** — only use on untrusted data. |
| `--formats html,json` | Skip the PNG when you don't need a document-ready image. |
| `--collapse N` | PNG only: merge runs of ≥N identically-shaped leaf siblings into one row (default 8, `0` disables). A file of 60 same-shaped keys otherwise eats 60 rows of diagram height and buries the real structure. The HTML/JSON always keep every node. |
| `--open html\|png\|all\|none` | Pop the result open when done. Default `auto`: opens the HTML when stdout is a tty, does nothing when piped or redirected — so a script that captures the output stays quiet. Pass `all` to open both artifacts, `none` to suppress. Auto-open is best-effort: a failure prints a note and does not affect the exit status. |

## Environment notes (this machine, verified)

- No Graphviz, and none needed — the PNG uses a self-written tidy tree on
  matplotlib, so there is nothing extra to install.
- Windows consoles default to cp936; the script reconfigures stdout to UTF-8 so
  the Chinese summary is readable. Without that the flagged-array list prints as
  mojibake.
- CJK in the PNG: matplotlib's per-glyph fallback (passing a font *family list*)
  silently produced tofu here. The script resolves a single CJK font file
  (Microsoft YaHei on this box) and binds labels to it with `fontproperties=`.
  If no CJK font exists it strips non-ASCII rather than drawing boxes.

## Reading the output

- **Colour carries meaning, never decoration.** There are only four accent
  families on purpose — teal = data array, amber = `object_array`, violet =
  nested `dict`/`list`, red = read failure. Every container (`root`, `folder`,
  `file`, `scalar`) shares one neutral slate so the structure reads as one
  skeleton instead of a rainbow. Don't add a colour without a kind to attach it
  to; if you do, extend the legend in the same commit or the map starts lying.
- **`object_array` is the interesting one.** It's what makes naive inspection
  fail: `arr.shape` tells you nothing, you must index into it. Its box is
  **dashed** so it reads as "looks like a leaf, hides a tree". The script
  expands the first `--max-items` elements and reports the element type
  composition in `stats.element_types`.
- **`疑似类别标签` hints** come from a heuristic: integer/bool dtype, `ndim <= 2`,
  and 2–50 unique values. Treat as a *lead, not a fact* — verify with the
  unique-value list before trusting it. A hint is its own field (`node.hint`),
  rendered amber and set apart from the rest of the metadata, precisely because
  it is the one line a reader is scanning for; it must not read as one more lump
  of detail.
- **Statistics are trimmed to what the shape/dtype line doesn't already say.**
  min/max/mean are suppressed for booleans and for anything already flagged as a
  label; a unique count within half the element count (i.e. "all distinct") is
  dropped; tiny values print in exponent form. If you add a stat back, check it
  isn't a re-statement of the shape.
- **`（抽样统计）`** means the array was too big to fully reduce; min/max/mean
  came from a strided subsample. Shape and dtype are always exact.
- **Root summary agrees with the header.** Both count only files that actually
  opened — a file that fails to load is tagged `error` and reported separately
  as `· N 个读取失败`, so the root box never claims 4 files when the header
  says 3.
- **同型合并 (on by default).** A run of ≥ 8 *siblings* with an identical shape
  and dtype folds into one `key_0* ×60` row that still opens to all 60. Without
  it, a file of 60 same-shaped keys is 60 rows of nothing to read. Only leaf
  arrays qualify — two dicts both labelled `dict · 3 键` can hold completely
  different things, so folding them would claim a sameness that isn't there.
  Toggle `同型合并` in the toolbar to see every row; the PNG has a static
  equivalent in `--collapse N`.

### Huge datasets: memory follows the largest npz member, not the dataset

Measured on this machine (7.8 GB RAM), and the shape of the curve is the thing
to remember:

| Input | Peak RSS |
|---|---|
| `.npy` 100 MB | 132 MB |
| `.npy` 800 MB | 422 MB |
| `.npy` 1.12 GB | **422 MB** — the same as 800 MB |
| `.npy` 6.5 GB | **422 MB** — still the same |
| `.npz` 100 MB | 132 MB |
| `.npz` 800 MB | **832 MB** — tracks the member |

A numeric `.npy` is opened with `mmap_mode='r'` and only sampled pages are ever
faulted in, so its cost is **bounded and independent of file size** — a 400 GB
folder of `.npy` costs about the same as a 1 GB one. Two things break that:

- **`.npz` members.** `mmap_mode` does nothing for a compressed member, so
  `z[key]` decompresses the whole array into RAM. One 40 GB member would try to
  allocate 40 GB.
- **Object-dtype `.npy`.** `mmap_mode='r'` refuses them outright —
  `ValueError: Array can't be memory-mapped: Python objects in dtype` — so the
  scanner falls back to a plain `np.load`, which has no ceiling. Measured at
  roughly **3.7× the file size resident** (3.2 MB → 42 MB, 30 MB → 140 MB), and
  worse for fat objects: a 300k-element array of small dicts was 8.5 MB on disk
  and 117 MB in RAM.

`--max-array-mb` closes both. Every array's shape and dtype are readable from its
own `.npy` header — a few hundred bytes, whether through the zip stream or off
disk — so the load-or-skip decision is made before any array data is touched.
Object arrays are costed at a flat 512 bytes per element rather than by
`itemsize`, since a pointer array's `nbytes` says almost nothing about what
loading it costs.

A 1.12 GB npz that previously peaked at ~1.15 GB now peaks at **31.9 MB**, with
`dff (1200, 250000) float32` still mapped exactly and its small sibling
`labels (1200,)` still fully summarised. A 7 GB `.npy` peaks at 422 MB and takes
under 7 seconds.

Object arrays are what this tool exists to look inside, so the cap has to be
loose enough not to gut the map: the default 256 MB still expands an object
array of ~500k elements. Raise `--max-array-mb` when the nesting is the point.

Skipped arrays are reported on every surface — `未加载` in the map box and the
PNG, the reason string in the list view, and a header chip. `stats.min/max/
unique` are simply absent, and `stats.unsampled` says why. This matters: the map
shows no value statistics anywhere, so their absence is not a signal a reader
can decode on their own.

### `--max-files` never truncates silently

The scan counts every file first, then caps. A cap you cannot see is worse than
no cap — on a large dataset you would get a map of the first N files
alphabetically and no way to know it was partial. So truncation prints a stdout
warning with the real total, adds a header chip, and records
`_agg.truncated` / `_agg.files_found` in the JSON:

```
注意: 只扫描了 10 / 共 37 个文件，还有 27 个没扫 —— 这张图是残缺的，
要全扫请加 --max-files 37
```

### Layout: the diagram is the page

Everything above the map is a row that could have been taken from the map's
height, so each one is kept to a single line and can be dismissed:

- **疑似标签 hints are one row of chips, not a list.** They double as
  navigation — clicking a chip runs that search and jumps the map to the array,
  so the hint row is a way into the structure rather than a label to read.
- **The 参考索引 empty state rides in its own header row** instead of taking a
  block of its own. Idle, the bar is one ~35px line.
- **`专注` in the toolbar** hides the hint row and the index bar entirely and
  refits the map. Measured on the 1500×1000 sample: the map gets **73%** of the
  viewport idle, **60%** with a node selected (the index code is on screen and
  is the point of having selected something), and **83%** in 专注. Before this
  pass it was 56% idle — a list of hints cost ~95px of a ~900px window.

If you add a row above the diagram, check the share again rather than assuming.

### 同型合并 keeps its members in the tree

The group node is a *view*: `prep()` still walks into it, so the members keep
their `_id`s and stay addressable. Toggling grouping off therefore removes
exactly one node per group and nothing else — don't assert that the ungrouped
tree is larger, because it is one node *smaller* per group. What changes is the
row count at that level (61 rows → 2).

### 参考索引 bar (HTML)

Selecting any box (map or list) fills the bar above the diagram with the Python
you would type in a Jupyter cell to reach that exact level, plus a 复制代码
button. The last line is marked `← 目标`.

The scanner records an `acc` field on every node — the subscript that reaches it
from its parent's variable — and the page turns the root→node chain into code:

| `acc` | Meaning | Emitted as |
|---|---|---|
| `"[0]"` / `"[0, 1]"` | object-array element, list item | `meta_0 = meta[0]` |
| `"['key']"` | dict / npz key | `dff = z['dff']` |
| `""` | the node *is* its parent's value (a `.npy` holds its array directly) | no new line; the `np.load` line is the target |
| `null` | the hop was never expanded | a `#` comment saying no subscript can be given |

The filesystem hops are handled specially because that is where data enters
memory: `.npz` becomes `z = np.load(f, allow_pickle=True)` (an `NpzFile`), while
a `.npy` becomes the array itself, named after the file. Paths print as
`r'...'` so Windows backslashes stay readable, switching quotes or falling back
to a plain string if the path contains one.

**Interaction split:** the `+`/`−` circle folds a subtree; clicking the box body
selects. Same in the list — the twisty folds, the row selects. Keeping these
separate is what lets "look at this level" and "open this level" be different
gestures.

### Depth control (HTML)

The toolbar has a `展开层数` number input that stays in sync with whatever is on
screen, plus `全部折叠` / `全部展开` buttons:

- `全部折叠` → depth 1 (only the root's direct children).
- `全部展开` → the real tree depth; the input snaps to `max` rather than staying
  at its old value, so it never lies about how much is open. On a large dataset
  this can produce thousands of nodes — that is a rendering cost, not a hang.
- Typing a number expands to exactly that level, and the list view follows.
- The map opens at depth 2: you see the shape of the whole dataset first, then
  drill in. Raise `--max-depth` at scan time if the input's `max` is too low for
  the nesting you need.

## Afterwards

1. Open the HTML yourself is not needed — read the JSON to answer follow-ups.
2. When you report back, summarize: the top-level grouping, the array that is
   most likely the main data tensor, the likely label/condition arrays, and any
   arrays whose shape is ambiguous (state the ambiguity, don't guess silently).
3. If the user wants the diagram in a document, embed the PNG into a `.docx`.

## Safety

Read-only; never writes to the dataset directory. `allow_pickle=True` is the
default because object arrays are otherwise invisible — only run it on data the
user trusts, and use `--no-pickle` if they flag the source as untrusted.
