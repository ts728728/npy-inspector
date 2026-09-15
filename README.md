# npy-inspector

Turn an opaque folder of `.npy` / `.npz` files into a structure map you can read
at a glance — an interactive HTML mind map, a static PNG, and a JSON schema.

Built for the case where you did **not** create the dataset and there is no
README: the tool has to discover the organization itself. It recurses into npz
keys, object arrays, and nested dicts/lists, so you can see what each array
holds, its shape/dtype/range, and which arrays look like labels.

<!-- Absolute jsDelivr URL, not a relative path, on purpose: GitHub rewrites a
     relative image path to raw.githubusercontent.com, which is unreachable from
     mainland China, so the screenshot renders as a broken image there. -->
![npy-inspector showing a structure map, with a node selected and its Jupyter index code shown above](https://gcore.jsdelivr.net/gh/ts728728/npy-inspector@main/examples/screenshot.png)

<sub>Interactive map. Click any box and the 参考索引 bar above it gives the
copy-pasteable Jupyter code that reaches that level.</sub>

---

## Why

`data.npz` tells you nothing. `np.load(f).files` gives you a list of 12 key
names and no idea which one is the main data tensor, which is the labels, or
which one is a `(3,)` object array that secretly holds three dicts of further
arrays. Answering that by hand means a loop of `print(k, v.shape, v.dtype)`,
and it breaks the moment something is `dtype=object`.

This does that loop properly, recursively, and draws the result.

## Cost

**Token cost does not scale with dataset size.** The scan reads array headers
and a strided sample — never the bulk data — so what comes back describes
*structure*, and structure is not what makes a dataset big. Measured:

| Dataset | `stdout` | JSON |
|---|---|---|
| 6.5 GB, 1 array in 1 file | 4 lines, 261 chars | **1.5 KB** |
| 1.0 MB, 73 arrays, nested | 9 lines, 377 chars | **91.3 KB** |

A 6,500× larger dataset produced a 60× *smaller* artifact. The stdout summary is
capped at ~20 lines no matter what (`label_hints[:15]`, `errors[:5]`, and three
artifact paths).

What does cost tokens is the **number of structural nodes** — files × arrays ×
nesting depth — and that is bounded by `--max-files` (400) and `--max-depth` (6).
One honest caveat: npz keys are always listed in full, so a single npz with tens
of thousands of keys is the one shape that can still grow the JSON. Nothing else
about a 400 GB folder makes it more expensive to look at than a 1 GB one.

## Install

**As a Claude Code skill** — clone it into your skills directory:

```bash
git clone https://github.com/ts728728/npy-inspector ~/.claude/skills/npy-inspector
```

Then just ask: *"I have a folder of npy/npz on drive D, show me its structure."*
Claude picks up the skill from `SKILL.md` and runs it.

**As a plain script** — no install, no package:

```bash
python scripts/inspect_npy.py /path/to/dataset
```

Requires Python 3 and `numpy`. [matplotlib](https://matplotlib.org/) is optional
— it is only needed for the PNG; the HTML and JSON are produced without it.
No Graphviz.

## Usage

```bash
python scripts/inspect_npy.py <path> [flags]
```

`<path>` is a directory (scanned recursively) or a single `.npy`/`.npz` file.
Outputs land in `<name>_structure_map/` next to the dataset.

Run it from a terminal and it opens the HTML for you; run it with the output
piped or redirected and it stays quiet, so it drops into a script cleanly.

| Flag | Meaning |
|---|---|
| `--out DIR` | Write elsewhere (e.g. your Desktop). |
| `--max-depth N` | Recursion ceiling into object arrays / dicts (default 6). Raise for deeply nested session structs. |
| `--max-items N` | Children expanded per node (default 25). Raise if a dict has 60 keys and you need them all — it also decides how complete the field relations are. Applies to `dict` / `list` / object-array elements; **npz member lists are never cut**. |
| `--max-files N` | Cap on files scanned (default 400). Never silent — see *Large datasets* below. |
| `--expect PATH` | Reconcile a reference list against the dataset: a `.py` is parsed statically (never executed) for `os.path.join` / `Path(...) / ...`, anything else is one path per line. See *`--expect`* below. |
| `--max-array-mb MB` | Arrays larger than this are mapped from their `.npy` header but never loaded (default 256). |
| `--no-pickle` | Refuse object arrays. **Safer, but nested structures go unexpanded** — only use on untrusted data. |
| `--formats html,png,json` | Which artifacts to produce. Drop the PNG when you don't need a document-ready image. |
| `--collapse N` | PNG only: merge runs of ≥N identically-shaped leaf siblings into one row (default 8, `0` disables). |
| `--open html\|png\|all\|none` | Pop the result open when done. Default `auto`: opens the HTML only when stdout is a terminal. |

## What you get

| File | Use |
|---|---|
| `<name>_structure_map.html` | **Primary artifact.** Self-contained, no CDN, works offline. An interactive mind map (circles fold/unfold one level, wheel-zoom, drag-pan, click to select) plus a flat list view. Both share the search box, the depth control, and the index bar. |
| `<name>_structure_map.png` | Static map — rounded node boxes, bezier links, kind legend, and the same relation badges the map shows. For embedding into a `.docx` or slides. **This one draws the whole tree** — the HTML's depth control does not apply — so a deep dataset makes an enormous image: a 27-file behavioural dataset with ~7100 leaves came out 4471×400914 px / 94 MB, which nothing opens. Past 20000 px on either side the script says so on stderr. For datasets that deep, ship the HTML and skip the PNG (`--formats html,json`). |
| `<name>_structure_map.json` | Machine-readable schema. Feed it to an assistant to answer follow-ups without re-scanning. |

Opening `<name>_structure_map.html#list` starts in the list view instead of the
map.

## Reading the output

**Colour carries meaning, never decoration.** There are only four accent
families on purpose:

| Colour | Kind |
|---|---|
| teal | data array |
| amber | `object_array` — a leaf that hides a tree |
| violet | nested `dict` / `list` |
| red | read failure |

Every container (`root`, `folder`, `file`, `scalar`) shares one neutral slate,
so the structure reads as a single skeleton instead of a rainbow.

- **`object_array` is the interesting one.** It is what makes naive inspection
  fail: `arr.shape` tells you nothing, you have to index into it. Its box is
  **dashed** to read as "looks like a leaf, hides a tree", and the element type
  composition is reported in `stats.element_types`.
- **`疑似类别标签` hints** come from a heuristic: integer/bool dtype, `ndim <= 2`,
  and 2–50 unique values. Treat it as a *lead, not a fact* — check the unique
  values before trusting it.
- **Statistics are trimmed** to what the shape/dtype line doesn't already say.
  `（抽样统计）` means the array was too big to fully reduce, so min/max/mean came
  from a strided subsample. Shape and dtype are always exact.
- **同型合并 (on by default).** A run of ≥8 *siblings* with an identical shape and
  dtype folds into one `key_0* ×60` row that still opens to all 60. Without it,
  a file of 60 same-shaped keys is 60 rows of nothing to read. Only leaf arrays
  qualify — two dicts both labelled `dict · 3 键` can hold completely different
  things, so folding them would claim a sameness that isn't there.

### 字段关系: how the fields relate

A per-node view describes one array at a time, so the one thing it structurally
cannot tell you is how arrays relate — which of them share an axis, and which
one is a pointer into another. Read one box at a time and `LickTrind` is just "a
float array of 1287 values"; read the container and it is the join that makes
`LickTime` per-trial in a 348-trial session. That gap is not cosmetic, and the
collapsible **字段关系** row above the map exists to close it:

| Row | Means |
|---|---|
| `348 ×24  ntrials  Trial_start_time  SoundTime …` | 24 sibling arrays share a leading dimension of 348, and a sibling scalar `ntrials` says what that axis is. The anchor is a button too — clicking it selects the scalar. |
| `= arange(348)  trInd` | `trInd` runs 0…347, each value once: it is the row id *of* the 348-axis, not a pointer into some other family. |
| `→ 348  ntrials  LickTrind` | Every value of `LickTrind` lands in `[0, 348)` and spans it: a per-lick → per-trial join. A `nan` note on the row means part of the array is padding. |
| `各容器: 348 / 453 / 485` | The same layout recurs at different sizes. The row prints one instance; the other values are listed rather than dropped, because this entry describes 195 containers, not one. |

Indexes and row ids also get a badge **on the box**, so the link is visible in
the map without opening the panel. Families deliberately do **not** get a badge
— 24 members carrying the same label would tile the map with it. Membership
lives in the panel.

Four things bound the claim, and all four are visible rather than silent:

- **A shared length below 8 is only a family if a scalar sibling names it.** Two
  unrelated length-2 arrays agreeing is a coincidence, not an axis.
- **A 2–50-value integer array is a label, not a pointer**, so it gets no
  relation badge. Amber `疑似类别标签` and a blue relation badge on one box would
  be two claims arguing with each other.
- **A `*` on a badge means the sibling set was cut by `--max-items`.** A family is
  exactly what an invisible missing sibling corrupts: the default `--max-items 25`
  on a 59-key session dict yields a partial family, so the panel entry, the
  container note, and a stdout line all say so — the stdout line names the
  `--max-items` that would have been enough. **If you are reading relations rather
  than shapes, raise it and re-scan.** The same applies in miniature to indexes
  over 200k elements, which get no relation at all, because a values-only claim
  cannot honestly be made from a strided subsample.
- **`--max-items` reaches `dict`, `list`/`tuple` and object-array elements — not
  `.npz` members.** An npz's key list comes from the zip directory and is always
  shown whole, so relations inside an npz container are never partial and an npz
  never gets a `… 另有 N 项` row. Deliberate: capping npz members would hide the
  arrays you opened the file to find, at no saving. The honest reading is
  one-directional — no `partial` mark means genuinely complete — so raising
  `--max-items` cannot change an npz's relations, and `--max-array-mb` remains
  the only cap that applies to npz members.
- **Nothing is cached by node id.** The panel stores names and re-resolves on
  click, so toggling 同型合并 — which renumbers every node — cannot make a panel
  link jump to the wrong array.

### `--expect`: reconcile the code against the disk

```bash
python scripts/inspect_npy.py D:/data --expect ../paper/fig1.py
```

Point it at the script that consumes the dataset and it reports, per referenced
path, whether that file is actually there:

```
参照清单对帐 (fig1.py → Zhong-et-2025):
  ✓ 找到 4
      retinotopy/areas.npz
      beh/example_bef_and_aft_learning_behavior.npy
      beh/Beh_sup_train1_before_learning.npy
      beh/Beh_sup_train1_after_learning.npy
  ✗ 不在磁盘上 10
      process_data/sup_train1_before_learning_leaf1_circle1_dprime_distribution.npy
      …
      … 另有 2 条
  （清单与磁盘的差集就是这些；要看磁盘上真有什么，读 JSON 或导图）
```

A `.py` is parsed with `ast.parse` and **never executed** — this is usually code
from a paper repo nobody has audited. It resolves `os.path.join(...)` and
`Path(...) / ...`, expands `for fn in [literals]` into one path per element, and
drops the leading `root` argument so the result is relative to the dataset you
are scanning. Anything it cannot resolve statically is listed as such rather
than omitted: a short list is indistinguishable from a complete one. Any other
extension is read as one path per line.

| State | Means | Action |
|---|---|---|
| `✓ 找到` | on disk, scanned | — |
| `▣ 是目录，不是文件` | the path names a folder | it is right there — the reference points one level up |
| `△ 在磁盘上但未扫描` | on disk, cut by `--max-files` | raise `--max-files` |
| `✗ 不在磁盘上` | not in the release | nothing to find — the code expects something unpublished |
| `? 无法静态确定` | f-string, variable concatenation | read that line yourself |

The `✗` state is why the feature exists. "The map doesn't show my file" has two
completely different causes — the tool capped something, or the file was never
published — and only the first is the tool's fault. Distinguishing them by hand
is the loop where you conclude the map is wrong when it is not.

### The 参考索引 bar

Selecting any box fills the bar above the diagram with the Python you would
type in a Jupyter cell to reach that exact level, plus a 复制代码 button. The
last line is marked `← 目标`. It exists to remove the step where you see an
array name in the diagram and then have to go write `np.load(...)['...']` by
hand.

## Large datasets

A 400 GB folder costs about the same as a 1 GB one, **provided it is numeric
`.npy`**. Measured here on a 7.8 GB machine:

| Input | Peak RSS |
|---|---|
| `.npy` 100 MB | 132 MB |
| `.npy` 800 MB | 422 MB |
| `.npy` 1.12 GB | **422 MB** — identical to 800 MB |
| `.npy` 6.5 GB | **422 MB** — still identical |
| `.npz` 100 MB | 132 MB |
| `.npz` 800 MB | **832 MB** — tracks the member |

A numeric `.npy` is memory-mapped, so only the sampled pages are ever read: the
cost is bounded and does not grow with file size. Two things break that bound:

- **`.npz` members.** `mmap_mode` does nothing for a compressed member, so
  `z[key]` decompresses the whole array at once — one 40 GB member would ask for
  40 GB.
- **Object-dtype `.npy`.** `mmap_mode='r'` refuses them (`ValueError: Array can't
  be memory-mapped: Python objects in dtype`), so they fall back to a plain
  `np.load` with no ceiling. Measured at ~3.7× the file size resident, and worse
  for fat objects: a 300k-element array of small dicts was 8.5 MB on disk and
  117 MB in RAM.

`--max-array-mb` (default 256) is the fix for both. Every array's shape and dtype
live in its own `.npy` header — a few hundred bytes, through the zip stream or
off disk — so the load-or-skip decision happens before any array data is touched.
Object arrays are costed at a flat 512 bytes per element, since a pointer array's
`nbytes` says almost nothing about what loading it costs, and the default still
leaves room to expand an object array of ~500k elements — they are what the tool
exists to look inside.

A 1.12 GB npz that used to peak at ~1.15 GB now peaks at **31.9 MB**, with
`dff (1200, 250000) float32` still mapped exactly and its small sibling
`labels (1200,)` still fully summarised. A 7 GB `.npy` scans in under 7 seconds.

Skipped arrays are labelled `未加载` / *not loaded* in the map, the list, and the
PNG — the map shows no value statistics anywhere, so an unmarked array would be
indistinguishable from a scanned one.

`--max-files` likewise never truncates silently: it counts every file first, and
prints the real total along with what was skipped.

## Safety

- **Read-only.** It never writes into the dataset directory; every output goes
  to a separate `<name>_structure_map/` folder.
- **`allow_pickle=True` is the default**, because object arrays are otherwise
  invisible — and that means scanning will execute whatever pickled objects the
  npz constructs. Only run it on data you trust. Use `--no-pickle` on anything
  of unknown origin.

## Examples

`examples/` contains a small synthetic dataset (3 files, 73 arrays) and the
outputs generated from it, so you can see what the tool produces before
pointing it at your own data:

```bash
python scripts/inspect_npy.py examples/dataset --out /tmp/demo
```

It deliberately includes an `object` array of dicts, a 61-key nested npz, and
one corrupt file.

### Worked example: where does `befRew` come from?

Real data, Zhong et al. 2025. The analysis code reads:

```python
beh0 = np.load(os.path.join(root, 'beh/Beh_sup_train1_before_learning.npy'), allow_pickle=1).item()
dat['mean_beh_bef'] = utils.get_mean_lick_response(beh0, lick_typ='befRew')
```

`befRew` appears nowhere in the data — grep all 27 files in `beh/` and there is
no such string. It is computed, in `utils.py`:

```python
cue_del[n] = SoundTime[n] + Reward_Delay_ms / (1000*3600*24)   # datenum conversion
befRew[n]  = any(LickTime in [Trial_start_time[n], cue_del[n]])
```

No list of node names gets you there. The relations the map reports for one
session do:

| On the map | What it gives you |
|---|---|
| `348 ×15  ntrials  …` | `ntrials = 348`; the trial axis is real and named |
| `LickTime` in the `1287` family, `SoundTime` / `Trial_start_time` in `348` | licks and trials are different axes — a lick-level quantity cannot be indexed by trial |
| `→ 348  ntrials  LickTrind` | `LickTrind` joins them: `LickTrind[i]` is the trial that lick `i` belongs to |
| `Reward_Mode = 'Passive'`, `Reward_Delay_ms = 0` | in this session `cue_del[n] == SoundTime[n]`, so the window is `[Trial_start_time[n], SoundTime[n]]` |

That is the whole definition minus the arithmetic. Verified against the source:
43.97% of VR2's 348 trials have at least one lick in that window, and averaging
the four mice in the file reproduces the paper's Fig 1b (0.719 before learning
for `circle1`, 0.724 for `leaf1`).

Same dataset, second question — the code loads
`process_data/…_dprime_distribution.npy`, so why is it not on the map?

```bash
python scripts/inspect_npy.py E:/Zhong-et-2025 --expect fig1.py
```

4 paths found, 10 not on disk, and `process_data/` does not exist anywhere in
the release — the local `process_data/` folder is the published `SVD_dec/`
(90 files, matching one for one). The map was right. The 10 dprime files are
intermediates the authors computed and never published.

When a scan finds relations it also warns if they were cut short, because a
truncated family is a claim with a silent hole in it:

```
注意: 195 处字段关系基于被 --max-items 截断的兄弟集合（带 * 的角标不完整）——
要完整的关系请加 --max-items 59
```

## License

MIT — see [LICENSE](LICENSE).

---

# 中文说明

一个文件夹里全是 `.npy` / `.npz`，完全分不清谁是谁？这个工具扫一遍，给你一张
一眼能看懂的结构图：可交互的 HTML 思维导图、一张能插进 Word 的 PNG、一份给机器读的 JSON。

它就是为这种情况写的：**数据不是你整理的，也没人给你留 README**，结构只能由工具自己发现。
npz 里的键、object 数组、嵌套的 dict/list，它都会逐层递归进去，告诉你每个数组装了什么、
多大、什么类型、数值范围多少，以及哪几个看着像标签。

![结构图示例](https://gcore.jsdelivr.net/gh/ts728728/npy-inspector@main/examples/screenshot.png)

## 为什么要用它

对着一个 `data.npz`，`np.load(f).files` 只给出一串键名。哪个是主数据、哪个是标签、
哪个 `(3,)` 的 object 数组背后还藏着三个 dict——这些只能靠猜。手动排查就是循环
`print(k, v.shape, v.dtype)`，一碰上 `dtype=object` 就无从下手。

这个工具把这个循环正确地、递归地做了一遍，并把结果画出来。

## 花多少 token

**跟数据集多大没关系。** 扫描只读数组头部和抽样数据，不读数据本体，所以返回的是「结构」，
而结构不会因为数据变大而变大。实测：

| 数据集 | `stdout` | JSON |
|---|---|---|
| 6.5GB，1 个文件 1 个数组 | 4 行 261 字符 | **1.5 KB** |
| 1.0MB，73 个数组，多层嵌套 | 9 行 377 字符 | **91.3 KB** |

数据集大了 6500 倍，产物反而小了 60 倍。命令行摘要无论数据集多大都限制在 20 行以内
（`label_hints[:15]`、`errors[:5]`，再加 3 条产物路径）。

真正消耗 token 的是**结构节点数**——文件数 × 数组数 × 嵌套层数——它受 `--max-files`（默认 400）
和 `--max-depth`（默认 6）约束。有一个例外需要说明：npz 的键是全量列举的，所以
「单个 npz 里有几万个键」是唯一还能把 JSON 撑大的情况。除此之外，400GB 的文件夹不会比
1GB 的更费 token。

## 安装

**当 Claude Code 技能用**：

```bash
git clone https://github.com/ts728728/npy-inspector ~/.claude/skills/npy-inspector
```

装完直接跟 Claude 说：「我 D 盘某个文件夹里有一批 npy 和 npz，帮我看看数据结构」。
它会从 `SKILL.md` 认出这个技能并自动调用。

**当普通脚本用**——不用安装，没有打包：

```bash
python scripts/inspect_npy.py /你的/数据集/路径
```

只依赖 Python 3 和 `numpy`。[matplotlib](https://matplotlib.org/) 是可选的，只有画 PNG 才需要，
HTML 和 JSON 没它也能出。不需要 Graphviz。

## 用法

```bash
python scripts/inspect_npy.py <路径> [参数]
```

`<路径>` 可以是目录（递归扫描），也可以是单个 `.npy`/`.npz` 文件。产物落在数据集旁边的
`<名字>_structure_map/` 里。

在终端里跑会自动打开 HTML；输出被管道或重定向接走时不弹窗，可以安静地嵌进脚本。

| 参数 | 说明 |
|---|---|
| `--out DIR` | 输出到别处（比如桌面）。 |
| `--max-depth N` | 向 object 数组 / dict 递归的深度上限，默认 6。嵌套很深的 session 结构可以调大。 |
| `--max-items N` | 每个节点展开的子项数，默认 25。某个 dict 有 60 个键、想全看到时就调大；字段关系完整与否也由它决定。管 `dict` / `list` / object 数组的元素，**不砍 npz 的成员表**。 |
| `--max-files N` | 最多扫描多少个文件，默认 400。会明确报告漏掉了什么，不会静默截断。 |
| `--expect PATH` | 拿一份「参照清单」跟磁盘对帐：`.py` 走静态解析（**不执行**）提取 `os.path.join` / `Path(...) / ...` 路径，其他后缀按每行一条路径读。详见下文「参照清单对帐」。 |
| `--max-array-mb MB` | 超过这个大小的数组只读 `.npy` 头部、不加载数据，默认 256。 |
| `--no-pickle` | 拒绝 object 数组。**更安全，但嵌套结构也就展不开了**——只对来源不明的数据用。 |
| `--formats html,png,json` | 生成哪些产物。不需要图片时去掉 png。 |
| `--collapse N` | 仅对 PNG 生效：把 ≥N 个形状相同的兄弟合并成一行，默认 8，填 `0` 关闭。 |
| `--open html\|png\|all\|none` | 跑完自动打开。默认 `auto`，即只在 stdout 是终端时打开 HTML。 |

## 生成什么

| 文件 | 用途 |
|---|---|
| `<名字>_structure_map.html` | **主要产物。** 单文件、无 CDN、离线可用。交互式导图（圆圈逐级折叠/展开、滚轮缩放、拖拽平移、点击选中）加一个平铺列表视图。两个视图共用搜索框、展开层数和参考索引栏。 |
| `<名字>_structure_map.png` | 静态结构图。圆角节点框、贝塞尔连线、图例，以及和导图一致的字段关系角标。用来插进 Word / PPT。**它画的是全树**，不受 HTML 那个「展开层数」控制，所以层数深的数据集会出一张巨大的图：一个 27 文件的行为数据集（约 7100 个叶子）量出来是 4471×400914 像素 / 94 MB，任何看图程序都打不开。超过 20000 像素时脚本会在 stderr 上明说。这种数据集就发 HTML、跳过 PNG（`--formats html,json`）。 |
| `<名字>_structure_map.json` | 机器可读的完整 schema。喂给 AI 就能接着回答后续问题，不用重扫。 |

打开 `<名字>_structure_map.html#list` 会直接进列表视图。

## 这张图怎么读

**颜色只表示类型，不做装饰。** 强调色刻意只有四种：

| 颜色 | 类型 |
|---|---|
| 青绿 | 数据数组 |
| 琥珀 | `object_array`——看着是叶子节点，展开后是一棵子树 |
| 紫 | 嵌套的 `dict` / `list` |
| 红 | 读取失败 |

根节点、目录、文件、标量共用同一种中性灰蓝，整个结构因此读起来是一个骨架，
而不是一道彩虹。

- **`object_array` 是最需要留意的一类。** 它正是「直接看 shape」会失效的原因：
  `arr.shape` 提供不了任何有效信息，必须索引进去才知道里面装了什么。它的方框画成
  **虚线**，提示「看着是叶子节点，展开后是一棵子树」；元素类型构成记录在
  `stats.element_types` 里。
- **`疑似类别标签` 是线索，不是结论。** 判据有三条：dtype 为整数或布尔、`ndim <= 2`、
  唯一值个数在 2~50 之间。采信之前请先核对唯一值列表。
- **统计量只补充 shape/dtype 没有给出的信息。** 标注 `（抽样统计）` 表示数组过大、
  无法完整归约，min/max/mean 取自等距抽样；shape 和 dtype 则始终精确。
- **同型合并（默认开启）。** 连续 ≥8 个 shape 与 dtype 完全相同的**兄弟节点**会折叠为
  一行 `key_0* ×60`，展开后依然是全部 60 个。没有这个机制，一个含 60 个同型键的文件
  就是 60 行读不出差异的内容。**只有叶数组参与合并**——两个 dict 都显示 `dict · 3 键`，
  内容却可能毫无关系，合并它们等于宣称二者相同，而事实上并非如此。

### 字段关系

逐节点看结构，天然看不到的东西只有一样：**字段之间的关系**——哪几个数组共用一条轴，
哪个数组是指向另一族的索引。一个一个方框看过去，`LickTrind` 只是「一个 1287 长的浮点
数组」；把容器当成一个整体看，它才是让 `LickTime` 变成「逐试次」的那条连接。上面那行
可折叠的**「字段关系」**面板就是补这个缺口：

| 行 | 含义 |
|---|---|
| `348 ×24  ntrials  Trial_start_time  SoundTime …` | 24 个兄弟数组共享首维 348，同级的标量 `ntrials` 给这条轴命名。锚点也是按钮，点它会选中那个标量。 |
| `= arange(348)  trInd` | `trInd` 取值 0…347 各一次：它是 348 轴的**行号**，不是在索引别的族。 |
| `→ 348  ntrials  LickTrind` | `LickTrind` 的取值全落在 `[0, 348)` 且铺满——一条「逐舔舐 → 逐试次」的连接。行上标 `nan` 表示数组尾部是补齐位。 |
| `各容器: 348 / 453 / 485` | 同一套字段布局在不同容器里长度不同。行里印的是其中一个实例的数字，**其余的列出来而不是丢掉**——这条目描述的是 195 个容器，不是一个。 |

索引和行号还会在**方框上**带一个角标，不开面板也能在图上看到这条联系。家族归属刻意
**不**带角标——24 个成员顶着同一个标签会把图刷花，成员名单只进面板。

有四条闸门限制这些断言的强度，且四条都是**看得见的**，没有静默降级：

- **共享长度小于 8 时，必须有同值标量为它命名才算家族。** 两个长度为 2 的无关数组凑巧
  相同，那是巧合不是轴。
- **2~50 个唯一值的整数数组是标签，不是指针**，因此不给关系角标。让一个方框同时顶着琥珀色
  「疑似类别标签」和蓝色关系角标，等于让两条互相矛盾的断言打架。
- **角标上的 `*` 表示兄弟集合被 `--max-items` 截断过。** 家族恰恰是最怕「看不见的缺失兄弟」
  的东西：默认 `--max-items 25` 遇到 59 个键的 session dict，家族就是不完整的——所以面板条目、
  容器标注、stdout 三处都会说明，stdout 那行还会给出**够用的那个 `--max-items` 数值**。
  **要读关系而不只是读形状，就把它调大重扫。** 同理，超过 20 万元素的索引数组拿不到任何关系，
  因为抽样统计撑不起一句关于全部取值的断言。
- **`--max-items` 的管辖范围是 `dict`、`list`/`tuple` 和 object 数组的元素，不包括 `.npz`
  的成员。** npz 的键名表是从 zip 目录里读出来的，永远完整显示，所以 npz 容器里的关系不会
  出现不完整，npz 也永远不会出现「… 另有 N 项」这一行。这是故意的：给 npz 成员设上限，等于
  把读者打开这个文件想找的那些数组藏起来，而且省不下任何开销。可读的方向只有一个——没有
  `partial` 标记就说明真的完整——所以调大 `--max-items` 不会改变 npz 的关系，能管到 npz
  成员的只有 `--max-array-mb`。
- **面板不缓存任何节点 id。** 它只存名字，点击时惰性解析——所以切换「同型合并」（会重排
  全部节点编号）不会让面板的链接跳到错误的数组上。

### 参照清单对帐（`--expect`）

```bash
python scripts/inspect_npy.py E:/Zhong-et-2025 --expect fig1.py
```

把消费这个数据集的脚本丢给它，它逐条告诉你这个路径在不在磁盘上：

```
参照清单对帐 (fig1.py → Zhong-et-2025):
  ✓ 找到 4
      retinotopy/areas.npz
      beh/Beh_sup_train1_before_learning.npy
      …
  ✗ 不在磁盘上 10
      process_data/sup_train1_before_learning_leaf1_circle1_dprime_distribution.npy
      …
      … 另有 2 条
  （清单与磁盘的差集就是这些；要看磁盘上真有什么，读 JSON 或导图）
```

`.py` 走 `ast.parse` **静态解析、绝不执行**——这种代码通常是论文仓库里没人审过的脚本。
它会解析 `os.path.join(...)` 和 `Path(...) / ...`，把 `for fn in [字面量列表]` 展开成多条路径，
并把开头的 `root` 参数丢掉，所以结果总是相对于你正在扫的数据集根目录。**静态定不下来的
单独列成「无法静态确定」，不会静默丢弃**——短一截的清单和完整的清单长得一模一样。
其他后缀按「每行一条路径」读。

| 状态 | 含义 | 该做什么 |
|---|---|---|
| `✓ 找到` | 在磁盘上，且已扫描 | — |
| `▣ 是目录，不是文件` | 这条路径指的是个文件夹 | 它就在那儿——引用往上指了一层 |
| `△ 在磁盘上但未扫描` | 在磁盘上，被 `--max-files` 截掉了 | 调大 `--max-files` |
| `✗ 不在磁盘上` | 这个 release 里就没有 | 找不到的——代码期待的是没发布的东西 |
| `? 无法静态确定` | f-string、变量拼接 | 那一行自己去读 |

`✗` 这一态正是这个功能存在的理由。「图上看不到我说的那个文件」有**两种完全不同的成因**
——工具截掉了，或者文件根本没发布过——而只有第一种是工具的锅。靠肉眼分辨这两者，就是
「明明图是对的却以为图错了」的那个循环。

### 参考索引栏

选中任意节点，上方就会显示在 Jupyter 中索引到这一层的完整代码，最后一行标注 `← 目标`，
旁边有「复制代码」。它省掉的是「在导图里看到一个数组名，再回 notebook 从头手写
`np.load(...)['...']`」这一步。

## 大文件

400GB 的文件夹和 1GB 的代价差不多，**前提是数值型的 `.npy`**。本机（7.8GB 内存）实测：

| 输入 | 峰值内存 |
|---|---|
| `.npy` 100MB | 132MB |
| `.npy` 800MB | 422MB |
| `.npy` 1.12GB | **422MB**——和 800MB 一模一样 |
| `.npy` 6.5GB | **422MB**——还是一模一样 |
| `.npz` 100MB | 132MB |
| `.npz` 800MB | **832MB**——跟着成员大小走 |

数值型 `.npy` 是 mmap 打开的，只有被抽样的那几页会被读进来，所以内存有上界、
**不随文件变大**。有两种情况会突破这个上界：

- **`.npz` 里的成员。** `mmap_mode` 对压缩成员无效，`z[key]` 会一次性把整个数组解压
  出来，一个 40GB 的成员就会申请 40GB。
- **object 类型的 `.npy`。** `mmap_mode='r'` 直接拒绝
  （`ValueError: Array can't be memory-mapped: Python objects in dtype`），只能退回不带
  mmap 的 `np.load`，这条路没有任何上限。实测约占文件大小的 **3.7 倍**，元素本身越大，
  膨胀得越厉害：一个装小 dict 的 30 万元素数组，磁盘上 8.5MB，进内存变 117MB。

`--max-array-mb`（默认 256）把这两种情况一并限制住。每个数组的形状和 dtype 都写在自己的
`.npy` 头部里——从 zip 流里读还是从磁盘上读，都只要几百个字节——所以「加载还是跳过」
在任何数组数据被读取之前就已确定。object 数组按**每个元素 512 字节**估算，因为指针数组的
`nbytes` 基本说明不了加载它要多少内存；这个默认值仍然够展开约 50 万元素的 object 数组，
而那正是这个工具存在的意义。

效果：一个 1.12GB 的 npz，峰值从约 1.15GB 掉到 **31.9MB**，`dff (1200, 250000) float32`
照样被精确画出，旁边的 `labels (1200,)` 统计量一个不少；7GB 的 `.npy` 扫完不到 7 秒。

跳过的数组在导图、列表、PNG 里都标着「未加载」——这张图本来就不显示取值统计，不标注的话，
它与已扫描的数组无从区分。`--max-files` 同理：**先数清总数再截断**，警告里给出真实总数
和漏掉了多少。

## 安全

- **只读。** 从不往数据集目录里写东西，所有产物都落在单独的 `<名字>_structure_map/` 里。
- **`allow_pickle=True` 是默认值**，因为否则 object 数组完全不可见——而这也就意味着
  扫描会执行 npz 里 pickle 进来的对象构造代码。**只对你信得过的数据用**；来源不明的
  数据请加 `--no-pickle`。

## 示例

`examples/` 里有一份很小的合成数据集（3 个文件 / 73 个数组）和它生成的产物，
可以先看看它产出什么，再拿去扫自己的数据：

```bash
python scripts/inspect_npy.py examples/dataset --out /tmp/demo
```

里面故意放了一个装着 dict 的 object 数组、一个 61 键的嵌套 npz，和一个损坏的文件。

### 实战范例：`befRew` 到底是怎么算出来的？

真实数据，Zhong et al. 2025。分析代码里写的是：

```python
beh0 = np.load(os.path.join(root, 'beh/Beh_sup_train1_before_learning.npy'), allow_pickle=1).item()
dat['mean_beh_bef'] = utils.get_mean_lick_response(beh0, lick_typ='befRew')
```

而 `befRew` 在数据里**根本不存在**——把 `beh/` 下 27 个文件全 grep 一遍也没有这个字符串。
它是算出来的，在 `utils.py` 里：

```python
cue_del[n] = SoundTime[n] + Reward_Delay_ms / (1000*3600*24)   # 换算成 datenum
befRew[n]  = any(LickTime in [Trial_start_time[n], cue_del[n]])
```

光看节点名单到不了这里，图上报出来的字段关系到得了：

| 图上的东西 | 它给出的信息 |
|---|---|
| `348 ×15  ntrials  …` | `ntrials = 348`，试次这条轴是真实存在的，而且有名字 |
| `LickTime` 在 `1287` 家族里，`SoundTime` / `Trial_start_time` 在 `348` 家族里 | 舔舐和试次是两条不同的轴——逐舔舐的量不可能按试次索引 |
| `→ 348  ntrials  LickTrind` | `LickTrind` 把两条轴连起来：`LickTrind[i]` 是第 `i` 次舔舐所属的试次 |
| `Reward_Mode = 'Passive'`、`Reward_Delay_ms = 0` | 这个 session 里 `cue_del[n] == SoundTime[n]`，所以窗口就是 `[Trial_start_time[n], SoundTime[n]]` |

减去算术，这就是完整的定义。已按源码实测核对：VR2 的 348 个试次里有 **43.97%**
在该窗口内至少舔过一次；对文件里四只鼠取平均，复现出论文 Fig 1b 的数值
（学习前 `circle1` 0.719、`leaf1` 0.724）。

同一个数据集，第二个问题——代码要读 `process_data/…_dprime_distribution.npy`，
图上为什么没有？

```bash
python scripts/inspect_npy.py E:/Zhong-et-2025 --expect fig1.py
```

答案：4 条路径命中，10 条不在磁盘上，而且 `process_data/` 这个目录**在整个 release 里
就不存在**——本地的 `process_data/` 装的是公开发布的 `SVD_dec/`（90 个文件，逐个对得上）。
**图是对的。** 那 10 个 dprime 文件是作者自己算的中间产物，没随数据集发布。

关系被截断时也会出声，因为残缺的家族等于一句带着隐形窟窿的断言：

```
注意: 195 处字段关系基于被 --max-items 截断的兄弟集合（带 * 的角标不完整）——
要完整的关系请加 --max-items 59
```


## 许可证

MIT，见 [LICENSE](LICENSE)。
