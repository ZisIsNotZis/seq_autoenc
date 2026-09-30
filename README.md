# seq_autoenc

Multi-scale **sequence autoencoders** over Chinese news text and Python source:
reconstruct a token sequence *simultaneously at every down-sampling resolution*,
plus the n-gram / suffix-array baselines that measure how predictable the data
actually is.

## What this is / why it exists

This is a 2026 personal research scratchpad, not a library. It attacks one
question: **how well can a convolutional encoder–decoder reconstruct a sequence,
and how does that depend on the resolution you ask it to reconstruct at?**

The organising idea is the `EncDec` module: instead of a single bottleneck, the
encoder walks down one `MaxPool1d(2)` step at a time and, at *each* level, a
small head predicts the full input sequence; the decoder then walks back up. All
levels' logits are concatenated, and `cross_entropy` is applied at **every scale
at once**. This turns the autoencoder into a coarse-to-fine reconstruction
pyramid: the deepest level has the most context but the fewest positions, the
shallowest level is nearly a local n-gram model. The accuracy-vs-scale curves
(saved as images, see *What is here vs. not*) are the result.

Around that core sit three supporting strands: (a) how to tokenise the data
(character-level 7622-way vs. byte-level 256-way), (b) a second, explicitly
autoregressive decoder (`DecR`, a GRU unrolled over a time embedding) to compare
against the parallel multi-scale one, and (c) measurements of the data's own
predictability — order-k conditional entropy, most-frequent-continuation n-gram
baselines, and a suffix-array next-character predictor — which is what tells you
whether the autoencoder is doing anything beyond lookup.

**No notebook contains prose explaining any of this.** There is no written
hypothesis, method, or discussion anywhere in the directory; the framing above is
reconstructed entirely from the code. The whole directory contains just six
markdown cells: four are captions for the saved result plots (`64 ![image.png] …`)
and two are the `inews` loader pasted in as markdown instead of code.

## Data

Two corpora, both loaded from the author's private parent archive — **neither is
included here, and the generated versions are not either.**

**1. `inews` — Chinese news text.** Loaded by:

```python
def inews(a="../nlpdat/inews/dev.txt"):
    a = np.loadtxt(a, "U", None, "_!_", usecols=(2, 3), encoding="utf-8").view("I").view("2H")[..., 0]
    a[:, -1] = ord("。")
    return a[a.astype("?")]
```

`../nlpdat/inews/{dev,test,train}.txt` are `_!_`-delimited records; columns 2 and
3 are read as text, concatenated across all three splits, and held as UTF-16 code
units. `Untitled-Copy1.ipynb` computes the vocabularies and writes the derived
binaries:

| Artifact | How it is built | Vocabulary |
| --- | --- | --- |
| `inews.U` | raw UTF-16 units, `tofile` | all chars |
| `inews.7622` | `toH()`: character → id map | 7,622 characters |
| `inews.B` | `toB()`: keep only CJK `U+4E00..U+9FA5`, take the top-256 characters, map each to a 2-byte code and drop the high byte | 256 bytes |

**2. Python source.** `../nlpdat/py/all.B` (byte-level) and `../nlpdat/py/all.90`
(90-symbol), assembled in `test.ipynb` from `../nlpdat/py/{py,pyp,ipynb,ipynbp}.txt`
via an intermediate `../nlpdat/py/all.sort.I`. The `p`-suffixed files are a second
variant of the same sources (the notebooks do not say how they differ).

**None of `inews.B`, `inews.7622`, `inews.U`, `all.B`, `all.90`, or the raw text
files are present in this repo.** The training notebooks read the derived
binaries; the analysis notebooks read `../nlpdat/py/all.B` directly. So the
project is not runnable end-to-end without the private corpus — only the
tokenisation notebook (`Untitled-Copy1.ipynb`, cells 0) could regenerate the
`inews.*` files, and it too needs `../nlpdat/inews/*.txt`.

## Architecture

The containers are re-defined inline in each notebook (no shared module). The core
set, as it appears in `Untitled-Copy1.ipynb`:

| Name | What it does |
| --- | --- |
| `F` | Wraps a function as a `Module` (`s.f(x)` or `s.f(x, s.a)`). |
| `Enc` | Runs a `Sequential` stack until the length axis is 1 and returns `x[..., 0]` — a fixed-size encoder. |
| `EncDec(*a, n=0)` | **The core.** See below. |
| `Add` | `Sequential` that sums branch outputs: `sum((s(x) for s in s[1:]), s[0](x))`. |
| `Res` | `Sequential` plus a residual add. |
| `Rep(n, *a)` | Applies a stack `n` times. |
| `Mean(Add)` | `Add` divided by the number of branches. |
| `DecR(gru, head)` | Autoregressive decoder: starts from the encoder's final state and repeatedly steps a `GRU(1, k)` fed a `linspace(0, 1, i)` time embedding, yielding a reconstruction of length `i` at each step. |
| `EmbeddingAuto(EmbeddingBag)` | A static n-gram embedding: `EmbeddingBag(n, k, mode='sum')` with a learned exponentially-decaying per-position weight `l = 0.5 ** arange(l-1, -1, -1)` passed as `per_sample_weights`. Learned n-gram mixture, trained by SGD — the neural counterpart of the n-gram baselines below. |
| `Affine`, `Log` | Per-channel scale+bias; and a live `matplotlib` logger that plots the running metric with `clear_output` / `canvas.draw`. |

`EncDec` in detail:

```python
class EncDec(Sequential):
    def forward(s, x):
        Y = []
        for i in range(int(np.log2(x.size(2)))):
            x = s[0](x)                          # downsample one more level
            y = s[1](x)                          # predict the sequence at this level
            for i in range(i + 1):
                y = s[2](y)                      # upsample the prediction back to full length
            Y.append(y)
        return torch.cat(Y)                      # one logit block per scale
```

with factories `s[0]` = encoder stage (conv + `MaxPool1d(2)`), `s[1]` = per-scale
head (1×1 convs, often down-projecting to a small `k`), `s[2]` = upsampling
(`ConvTranspose1d(k, k, 3, 2, 1, 1)`; some notebooks drop it entirely in favour of
`F(lambda x: x.repeat_interleave(2, 2))`). The `n=` argument stops the recursion
`n` levels early.

Assembled model (the canonical run):

```python
Embedding(256, k) → transpose(1,2) → EncDec(...) → Conv1d(k,k,1) → BatchNorm1d → SiLU → Conv1d(k,256,1)
```

## The training loop

```python
a = torch.tensor(np.memmap("inews.B", "B")).unfold(0, 256, 1)   # sliding windows
...
for t in range(O.total_steps // 100):
    for t in range(100):
        y = a[torch.randint(len(a), (4,))].long()
        x = m(y).unflatten(0, (int(np.log2(a.size(1))), -1))     # split logits by scale
        functional.cross_entropy(x.transpose(1, 2),
                                 y.expand(len(x), -1, -1)).backward()  # supervise EVERY scale
        o.step(); o.zero_grad(); O.step()
    log((x.argmax(2) == y).float().mean((1, 2)).cpu())           # one accuracy per scale
```

Adam + `OneCycleLR`. Peak LR varies from `0.001` to `0.01` and `total_steps` from
10,000 to 1,000,000 across notebooks. The reported metric is top-1 accuracy **per
scale**, produced by the `unflatten` that splits the concatenated logits back into
one block per resolution. `Log` renders those per-scale curves live.

Window lengths used: 8, 16, 32, 64, 256, 512, 1024, 2048; batch 4–1024;
embedding width `k`/`K` from 16 to 512.

## The baselines and predictability analysis

These are separate from the autoencoders and are the most self-contained code here.

* **`Untitled1.ipynb`** — order-k conditional entropy of Chinese text in bits/char,
  `k = 1..8`, computed as `(log2(s) - m @ log2(m) / s) / k` over the count
  distribution of k-grams. Also a joint-bigram variant. This is the measurement
  that says how much context the autoencoder would need to be worth anything.
* **`test.ipynb`** (61 cells) — n-gram context statistics on `inews.7622` and
  `../nlpdat/py/all.B`: for context length `n = 4, 8, 12`, the fraction of
  positions whose context is unique, computed with `np.unique` over strided
  `V{n+1}` views, plus `numba`-`njit` helpers (`cat`, a hand-written `qsort` over
  context bytes) for the same counting faster.
* **`test2.ipynb`** — builds **most-frequent-continuation tables** for `n = 2..5`
  at character and byte level, sorts them, writes `inews.-{n}H.H` /
  `inews.{n}B.B`, and scores the fraction of positions where the majority
  continuation is correct. A pure static n-gram predictor baseline.
* **`Untitled2.ipynb`** — a **suffix-array predictor**: `argsort` over strided
  16-byte views of `inews.-U`, then `np.searchsorted(sorter=...)` to find the
  longest matching context and emit the following character. `f('你是谁?我', 64)`
  greedily generates text from it. This is the classical compression-style model
  the autoencoder is implicitly competing with.

## How to run

Notebook-only. There is **no single entrypoint** and no package to install.

```bash
pip install torch numpy matplotlib tqdm numba jupyterlab
jupyter lab
```

Then open a notebook and run all cells.

### Read this before running

* **CUDA-only.** Most notebooks set
  `torch.set_default_tensor_type(torch.cuda.FloatTensor)` with no `.cpu()`
  fallback. Remove that line for a CPU host.
* **The data is missing** (see *Data*). Place `inews.B`, `inews.7622`, `inews.U`
  and/or `../nlpdat/py/all.B` where the notebook expects them, or edit the paths.
  The `inews.*` files can be regenerated from `../nlpdat/inews/*.txt` by
  `Untitled-Copy1.ipynb` cell 0; `all.B` / `all.90` cannot, the sources are gone.
* **Checkpoints referenced but absent:** `60.pk`, `214.pk`, and the output of
  `torch.save(..., 'Untitled-Copy7.pk')`. `Untitled-Copy7.ipynb` cell 4 writes
  that file; nothing reads it.
* `%matplotlib notebook` requires a live Jupyter kernel for the `Log` plots.
* The analysis notebooks (`Untitled1`, `test`, `test2`, `Untitled2`) depend only
  on numpy/numba and their data files — they are the easiest to run.

### Which notebook should I read?

**Canonical: [`Untitled-Copy1.ipynb`](Untitled-Copy1.ipynb)** — 17 cells,
~1,197 effective code lines, the largest and the most complete. It contains the
whole pipeline in order: the `inews` loader and the `toH`/`toB` tokenisers that
build `inews.U`/`.7622`/`.B`; the full container set (`F`, `Enc`, `EncDec`, `Add`,
`Res`, `Rep`, `Mean`, `Log`); the canonical 256-byte multi-scale autoencoder run;
the `DecR` GRU autoregressive decoder comparison; a 7622-character-vocab variant;
and the four saved accuracy-vs-scale result images. Start here.

`Untitled.ipynb` is a near-duplicate of it (308 code lines, the same four result
images, a trimmed version of the same runs) — read `Untitled-Copy1.ipynb` instead.

Then, for the supporting strands: `test.ipynb` + `test2.ipynb` + `Untitled1.ipynb`
for the predictability baselines, and `Untitled2.ipynb` for the suffix-array
predictor.

Quick map of the rest: `Untitled-Copy2.ipynb` (3 cells) is the same diagram
repeated for several lengths; `Untitled-Copy3.ipynb` (8 cells) sweeps window
lengths 8/16/32 with the full toolkit; `Untitled-Copy4.ipynb` uses the `Enc`
(fixed-vector) variant instead of `EncDec` and a `Mean` head; `Untitled-Copy5`/
`-Copy6`/`-Copy7` are the `EmbeddingAuto` (learned n-gram embedding) strand —
`-Copy6` also tests `Softmax`/`LogSoftmax` and a custom `Affine` on the bag
output, and prints the learned decay weights (`m[0].l/m[0].l.sum()`) to see the
effective n-gram order.

## What is here vs. not

**Here**

* 12 `Untitled*.ipynb` / `test*.ipynb` notebooks, ~3,275 effective code lines,
  `nbformat` 4.5, `python3` kernel, no `.py` files.
* The complete architecture set and both training loops, re-defined per notebook.
* **Four saved result images**, embedded as cell attachments in
  `Untitled-Copy1.ipynb` (and duplicated in `Untitled.ipynb`) — these are the only
  recorded results in the directory. Three are labelled 64 / 256 / 512 (cell 3) and
  are per-scale top-1 accuracy curves; one is the autoregressive `DecR` curve
  (cell 5). They are real plots of real runs, but the axes are unlabelled, the run
  configuration is not recorded next to them, and the numbers behind them are gone.
  What they show: within a run, accuracy falls monotonically as you move to
  coarser scales (one curve saturates just under 1.0 while the others plateau at
  roughly 0.8 / 0.5 / 0.2 / 0.1 / 0.05 after ~1,000 logged points), and for a
  given curve position the whole family sits higher as the training window grows
  64 → 256 → 512. The `DecR` autoregressive plot shows the same shape as a
  function of decode position: near-1.0 for the first few characters, collapsing
  into the 0.25–0.4 band by position ~90.

**Not here**

* **No written-up results.** No paper, report, notes, tables, or metrics files.
  The entire directory has six markdown cells: four image captions and two copies
  of the `inews` loader mistakenly typed as markdown.
* **No numeric logs.** Unlike a sibling project, nothing here appends metrics to
  `.txt`. Every training metric exists only inside the two saved plots.
* **No checkpoints.** `60.pk` and `214.pk` are loaded and absent.
* **No data.** Both raw corpora and all derived binaries (`inews.B`, `inews.U`,
  `inews.7622`, `all.B`, `all.90`, `inews.*-n*`) are outside the archive.
* **No cell outputs** — all `stream`/`execute_result`/`error` outputs were cleared
  before archiving; only the four markdown image attachments survive.
* **No packaging, tests, CLI, or importable module.**
* **No git history** — extracted from the private archive with no `.git`.
* **No explanation** of the hypothesis or of the `pyp.txt`/`ipynbp.txt` variants.

## Relationship to `nn.py` / `cznn`

Checked and **not applicable**: this directory contains **no `.py` files at all**,
no notebook imports `nn` or `from nn import *`, and `cznn` is never referenced.
Nothing was deleted. The containers (`F`, `Add`, `Res`, `Rep`, `Mean`, `EncDec`)
are hand-written inline and self-contained; they use the same
fold-into-`Sequential` idiom as the author's published
[`cznn`](https://github.com/ZisIsNotZis/cznn) (where `Add`/`Mul`/`Res` are
similarly `Sequential` subclasses), but there is no code dependency in either
direction. `cznn` would be a natural home if this were ever tidied into a library.

## Security / redaction

* Absolute home-directory paths: **none were found** in this directory, so
  none needed genericising. All data references are already relative
  (`../nlpdat/...`, `inews.B`).
* The standard credential gate (private-key PEM headers, key/token/password
  assignments, the employer mail domain, internal-network hostnames, bare IPv4
  literals) returns **nothing** here, and a broader JWT/PII sweep (JWT-shaped
  strings, e-mail addresses, phone-shaped digit runs) also returns nothing. No
  redaction was necessary in this directory.
* The source notebooks write derived data **into the working directory** their
  parents expected (`inews.B`, `inews.7622`, `inews.U` in the notebook's own cwd,
  and `../nlpdat/py/all.sort.I` outside it). Run them in a scratch directory — do
  not let them drop generated binaries into this repo or write into `../`.

## Status

`Status: research/archive` — a 2026 personal research scratchpad, published as-is
for provenance and readability. Not maintained, not reproducible without the
missing corpora.

**License: none.** No `LICENSE` file was present in the source and none is added
here. Treat as all-rights-reserved / personal research; add an explicit license
before relying on it or reusing the code.
