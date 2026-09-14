# npy-inspector

Turn an opaque folder of `.npy` / `.npz` files into a structure map you can read
at a glance — an interactive HTML mind map, a static PNG, and a JSON schema.

Built for the case where you did **not** create the dataset and there is no
README: the tool has to discover the organization itself. It recurses into npz
keys, object arrays, and nested dicts/lists, so you can see what each array
holds, its shape/dtype/range, and which arrays look like labels.

![npy-inspector showing a structure map, with a node selected and its Jupyter index code shown above](examples/screenshot.png)

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
| `--max-items N` | Children expanded per node (default 25). Raise if a dict has 60 keys and you need them all. |
| `--max-files N` | Cap on files scanned (default 400). Never silent — see *Large datasets* below. |
| `--max-array-mb MB` | Arrays larger than this are mapped from their `.npy` header but never loaded (default 256). |
| `--no-pickle` | Refuse object arrays. **Safer, but nested structures go unexpanded** — only use on untrusted data. |
| `--formats html,png,json` | Which artifacts to produce. Drop the PNG when you don't need a document-ready image. |
| `--collapse N` | PNG only: merge runs of ≥N identically-shaped leaf siblings into one row (default 8, `0` disables). |
| `--open html\|png\|all\|none` | Pop the result open when done. Default `auto`: opens the HTML only when stdout is a terminal. |

## What you get

| File | Use |
|---|---|
| `<name>_structure_map.html` | **Primary artifact.** Self-contained, no CDN, works offline. An interactive mind map (circles fold/unfold one level, wheel-zoom, drag-pan, click to select) plus a flat list view. Both share the search box, the depth control, and the index bar. |
| `<name>_structure_map.png` | Static map — rounded node boxes, bezier links, kind legend. For embedding into a `.docx` or slides. |
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

## License

MIT — see [LICENSE](LICENSE).

---

# 中文说明

一个文件夹里全是 `.npy` / `.npz`，谁是谁完全看不出来？这个工具扫一遍，给你一张
一眼能看懂的结构图：能点的 HTML 思维导图、一张能插进 Word 的 PNG、一份给机器读的 JSON。

它专门对付这种情况：**数据不是你整理的，也没人给你留 README**，只能让工具自己去摸结构。
npz 里的键、object 数组、嵌套的 dict/list，它都一层层钻进去，告诉你每个数组装了什么、
多大、什么类型、数值范围多少，以及哪几个看着像标签。

![结构图示例](examples/screenshot.png)

## 为什么要用它

对着一个 `data.npz`，`np.load(f).files` 只会甩给你一串键名。哪个是主数据、哪个是标签、
哪个 `(3,)` 的 object 数组背后还藏着三个 dict——全靠猜。手动排查就是循环
`print(k, v.shape, v.dtype)`，一碰上 `dtype=object` 立刻没辙。

这个工具把这个循环做对了、做全了，还顺手把结果画出来。

## 花多少 token

**跟数据集多大没关系。** 扫描只看数组头部加抽样，不读数据本体，所以返回的是「结构」，
而结构不会因为数据大就变大。实测：

| 数据集 | `stdout` | JSON |
|---|---|---|
| 6.5GB，1 个文件 1 个数组 | 4 行 261 字符 | **1.5 KB** |
| 1.0MB，73 个数组，多层嵌套 | 9 行 377 字符 | **91.3 KB** |

数据集大了 6500 倍，产物反而小了 60 倍。命令行摘要无论数据集多大都压在 20 行以内
（`label_hints[:15]`、`errors[:5]`，再加 3 条产物路径）。

真正吃 token 的是**结构节点数**——文件数 × 数组数 × 嵌套层数——它被 `--max-files`（默认 400）
和 `--max-depth`（默认 6）卡着。老实说一个例外：npz 的键是全量列出来的，所以
「单个 npz 里几万把键」是唯一还能把 JSON 撑大的情况。除此之外，400GB 的文件夹不会比
1GB 的更费 token。

## 安装

**当 Claude Code 技能用**：

```bash
git clone https://github.com/ts728728/npy-inspector ~/.claude/skills/npy-inspector
```

装完直接跟 Claude 说：「我 D 盘某个文件夹里有一批 npy 和 npz，帮我看看数据结构」。
它会照 `SKILL.md` 自己调用。

**当普通脚本用**——不用安装，没有打包：

```bash
python scripts/inspect_npy.py /你的/数据集/路径
```

只要 Python 3 和 `numpy`。[matplotlib](https://matplotlib.org/) 是可选的，只有画 PNG 才需要，
HTML 和 JSON 没它也能出。不需要 Graphviz。

## 用法

```bash
python scripts/inspect_npy.py <路径> [参数]
```

`<路径>` 可以是目录（递归扫描），也可以是单个 `.npy`/`.npz` 文件。产物落在数据集旁边的
`<名字>_structure_map/` 里。

在终端里跑会自动打开 HTML；输出被管道接走或重定向了就不弹窗，可以安静地塞进脚本。

| 参数 | 说明 |
|---|---|
| `--out DIR` | 输出到别处（比如桌面）。 |
| `--max-depth N` | 往 object 数组 / dict 里钻几层，默认 6。session 结构套得深的可以调大。 |
| `--max-items N` | 每个节点显示几个子项，默认 25。某个 dict 有 60 个键想全看到就调大。 |
| `--max-files N` | 最多扫几个文件，默认 400。会明确告诉你漏了什么，不会闷声截断。 |
| `--max-array-mb MB` | 超过这个大小的数组只读头部、不加载数据，默认 256。 |
| `--no-pickle` | 拒绝 object 数组。**更安全，但嵌套结构也就展不开了**——只对来源不明的数据用。 |
| `--formats html,png,json` | 生成哪些文件。不需要图就砍掉 png。 |
| `--collapse N` | 只管 PNG：把 ≥N 个形状相同的兄弟合并成一行，默认 8，填 `0` 关掉。 |
| `--open html\|png\|all\|none` | 跑完自动打开。默认 `auto`，也就是只在终端里跑才打开 HTML。 |

## 生成什么

| 文件 | 用途 |
|---|---|
| `<名字>_structure_map.html` | **主要产物。** 单文件、无 CDN、离线可用。交互式导图（圆圈逐级折叠/展开、滚轮缩放、拖拽平移、点击选中）加一个平铺列表视图。两个视图共用搜索框、展开层数和参考索引栏。 |
| `<名字>_structure_map.png` | 静态结构图。圆角节点框、贝塞尔连线、图例。用来插进 Word / PPT。 |
| `<名字>_structure_map.json` | 机器可读的完整 schema。喂给 AI 就能接着回答后续问题，不用重扫。 |

打开 `<名字>_structure_map.html#list` 会直接进列表视图。

## 这张图怎么读

**颜色只用来说事，不做装饰。** 一共就四类：

| 颜色 | 类型 |
|---|---|
| 青绿 | 数据数组 |
| 琥珀 | `object_array`——看着像叶子，其实藏着一棵树 |
| 紫 | 嵌套 `dict` / `list` |
| 红 | 读取失败 |

根节点、目录、文件、标量统一用同一种中性灰蓝，让骨架读起来是一体的，而不是一道彩虹。

- **`object_array` 是重点。** 它就是让「直接看 shape」失效的东西：`arr.shape` 什么也
  说明不了，必须索引进去才知道。它的框画成**虚线**，意思是「看着像叶子，其实藏着一棵树」，
  元素类型构成在 `stats.element_types` 里。
- **`疑似类别标签` 是线索，不是结论。** 判据是整数或布尔 dtype、`ndim <= 2`、
  唯一值在 2~50 个之间。请对着唯一值列表确认过再信它。
- **统计量只保留 shape/dtype 没说过的信息。** 出现 `（抽样统计）` 说明数组太大没法完整
  统计，min/max/mean 来自等距抽样；形状和 dtype 永远是准的。
- **同型合并（默认开启）**：≥8 个形状和 dtype 完全相同的兄弟并成一行 `key_0* ×60`，
  点开还是全部 60 个。没有它，一个 60 个同型键的文件就是 60 行什么也看不出来的东西。
  **只有叶数组会被合并**——两个 dict 都写着 `dict · 3 键`，内容可能天差地别，
  合并等于谎报它们一样。

### 参考索引栏

选中任意方块，上方就会给出在 Jupyter 里索引到这一层的完整代码，最后一行标着 `← 目标`，
旁边有「复制代码」。它省掉的是「在导图里看见一个数组名，然后回 notebook 从头手写
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
**不随文件变大**。有两种东西会顶破这个上界：

- **`.npz` 里的成员。** `mmap_mode` 对压缩成员无效，`z[key]` 会一口气把整个数组解压
  出来，一个 40GB 的成员就向你要 40GB。
- **object 类型的 `.npy`。** `mmap_mode='r'` 直接不收
  （`ValueError: Array can't be memory-mapped: Python objects in dtype`），只能退回不带
  mmap 的 `np.load`，这条路没有任何上限。实测约占文件大小的 **3.7 倍**，对象越胖越夸张：
  一个装小 dict 的 30 万元素数组，磁盘上 8.5MB，进内存变 117MB。

`--max-array-mb`（默认 256）把这两条路一起堵上。每个数组的形状和 dtype 都写在自己的
`.npy` 头部里——从 zip 流里读还是从磁盘上读，都只要几百个字节——所以「加载还是跳过」
在任何数据被碰之前就定下来了。object 数组按**每个元素 512 字节**估算，因为指针数组的
`nbytes` 基本说明不了加载它要多少内存；这个默认值仍然够展开约 50 万元素的 object 数组，
而那恰恰是这工具最该看的东西。

效果：一个 1.12GB 的 npz，峰值从约 1.15GB 掉到 **31.9MB**，`dff (1200, 250000) float32`
照样被精确画出，旁边的 `labels (1200,)` 统计量一个不少；7GB 的 `.npy` 扫完不到 7 秒。

跳过的数组在导图、列表、PNG 里都标着「未加载」——这图本来就不显示取值统计，不标的话
它跟扫过的数组完全分不出来。`--max-files` 同理：**先数清总数再截**，警告里给出真实总数
和漏掉了多少。

## 安全

- **只读。** 从不往数据集目录里写东西，所有产物都落在单独的 `<名字>_structure_map/` 里。
- **`allow_pickle=True` 是默认值**，因为不这样的话 object 数组完全不可见——而这也就意味着
  扫描会执行 npz 里 pickle 进来的对象构造代码。**只对你信得过的数据用**；来源不明的
  数据请加 `--no-pickle`。

## 示例

`examples/` 里有一份很小的合成数据集（3 个文件 / 73 个数组）和它生成的产物，
可以先看看这工具产出什么样，再拿去扫自己的数据：

```bash
python scripts/inspect_npy.py examples/dataset --out /tmp/demo
```

里面故意放了一个装着 dict 的 object 数组、一个 61 键的嵌套 npz，和一个损坏的文件。

## 许可证

MIT，见 [LICENSE](LICENSE)。
