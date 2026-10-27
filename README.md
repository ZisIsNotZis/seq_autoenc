<p align="right"><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a></p>

<p align="center">
  <img src="assets/logo.svg" width="112" alt="seq_autoenc logo: an hourglass of token bars">
</p>

<h1 align="center">seq_autoenc</h1>

<p align="center"><strong>Multi-scale sequence autoencoders over Chinese news text and Python source.</strong></p>

<p align="center">
  Most autoencoders squeeze a sequence into one vector.<br>
  This one reconstructs the whole sequence at every down-sampling scale, at once.
</p>

<p align="center">
  Around that core sit the n-gram and suffix-array baselines that measure how predictable the data really is.
</p>

---

<p align="center">
  <img src="assets/results/accuracy-512.png" width="680" alt="Per-scale top-1 accuracy curves from a 512-token training run">
</p>

<p align="center">
  <sub>Per-scale top-1 accuracy during a 512-token training run. Saved from <a href="01_multiscale_autoencoder.ipynb"><code>01_multiscale_autoencoder.ipynb</code></a>; the original plot has no axis labels and the run configuration was not recorded.</sub>
</p>

## What it does

The core module is `EncDec`. Instead of a single bottleneck, the encoder walks down one `MaxPool1d(2)` step at a time. At each level a small head predicts the full input sequence, and the decoder upsamples that prediction back to full length. All levels' logits are concatenated, and `cross_entropy` is applied at **every scale at once**.

That turns an autoencoder into a coarse-to-fine reconstruction pyramid: the deepest level has the most context but the fewest positions; the shallowest level is almost a local n-gram model. The saved accuracy-vs-scale curves are the result.

Three related strands run alongside it:

- **Tokenisation** — character-level (7,622-way) versus byte-level (256-way).
- **Decoder shape** — a parallel multi-scale decoder versus `DecR`, an explicitly autoregressive GRU decoder.
- **Data predictability** — order-k conditional entropy, most-frequent-continuation n-grams, context-uniqueness counts, and a suffix-array next-character predictor.

## Quick start

This is a notebook archive, not a package — there is nothing to build or install from PyPI.

```bash
git clone https://github.com/ZisIsNotZis/seq_autoenc.git
cd seq_autoenc
pip install torch numpy matplotlib tqdm numba jupyterlab
jupyter lab
```

Then open **[`01_multiscale_autoencoder.ipynb`](01_multiscale_autoencoder.ipynb)** — the canonical notebook, with the full pipeline and the four saved result plots.

> **Before you run:** neither corpus is in this repository. The training notebooks stop at their first `np.memmap` unless you supply `inews.B` / `inews.7622` or `../nlpdat/py/all.B`. See [Data](#data). The analysis notebooks need the same derived binaries.

## What it can do now

- **Reconstruct at every scale.** One convolutional encoder–decoder predicts the full sequence at each down-sampling level, supervised by a single `cross_entropy` over all levels.
- **Compare two vocabularies.** Byte-level 256-symbol Chinese text, or a 7,622-character vocabulary; Python source is also covered with a 90-symbol vocabulary.
- **Compare decoder shapes.** The parallel multi-scale `EncDec` against `DecR`, a GRU stepped over a time embedding, and against a fixed-vector `Enc` next-token baseline.
- **Measure the data, not only the model.** Order-k conditional entropy, unique-context fractions, majority-continuation n-gram tables, and a suffix-array predictor that can greedily generate text.
- **Watch per-scale metrics live.** A small `Log` helper redraws the running accuracy of every scale inside a Jupyter notebook.
- **Probe learned n-gram order.** `EmbeddingAuto` is an `EmbeddingBag` with a learned exponential position decay; the notebooks print the normalised weights to show the effective context length.

## Notebook map

| Notebook | What it does |
| --- | --- |
| [`01_multiscale_autoencoder.ipynb`](01_multiscale_autoencoder.ipynb) | **Canonical.** Loader, `toH`/`toB` tokenisers, full container set, 256-byte multi-scale run, `DecR` comparison, 7,622-character variant, four result plots. |
| [`02_multiscale_short_windows.ipynb`](02_multiscale_short_windows.ipynb) | Short-window (8 / 16 token) variants of the canonical multi-scale run. |
| [`03_scale_decomposition.ipynb`](03_scale_decomposition.ipynb) | The 32-token model with different numbers of retained scales. |
| [`04_window_length_sweep.ipynb`](04_window_length_sweep.ipynb) | Window-length / depth sweep (256, 16, 9) plus a fixed-vector `Enc` comparison. |
| [`05_fixed_vector_encoder.ipynb`](05_fixed_vector_encoder.ipynb) | Fixed-vector `Enc` next-token model on `inews.7622`, with an n-gram entropy check. |
| [`06_learned_ngram_embedding.ipynb`](06_learned_ngram_embedding.ipynb) | `EmbeddingAuto` learned n-gram embedding with a linear head, at several widths. |
| [`07_embedding_head_variants.ipynb`](07_embedding_head_variants.ipynb) | `EmbeddingAuto` head variants: raw, Softmax, LogSoftmax, ReLU, Affine; prints learned decay weights. |
| [`08_python_source_encoder.ipynb`](08_python_source_encoder.ipynb) | Fixed-vector `Enc` next-token model on Python source (90 symbols), k = 64 / 128 / 512. |
| [`09_conditional_entropy.ipynb`](09_conditional_entropy.ipynb) | Order-k conditional entropy of Chinese text, k = 1..8, in bits per character. |
| [`10_context_uniqueness.ipynb`](10_context_uniqueness.ipynb) | Unique-context fractions for n = 4, 8, 12, with `numba` helpers. |
| [`11_ngram_continuation.ipynb`](11_ngram_continuation.ipynb) | Most-frequent-continuation tables for n = 2..5, at character and byte level. |
| [`12_suffix_array_predictor.ipynb`](12_suffix_array_predictor.ipynb) | Suffix-array next-character predictor and greedy text generation. |

## Saved results

Four training curves survive from the original runs. They are embedded in the canonical notebook and extracted here as images. The axes are unlabelled and the numeric logs are gone, so read them as shapes, not as reproducible benchmarks.

| Window 64 | Window 256 | Window 512 |
| :---: | :---: | :---: |
| <img src="assets/results/accuracy-64.png" width="260" alt="Accuracy curves, 64-token window"> | <img src="assets/results/accuracy-256.png" width="260" alt="Accuracy curves, 256-token window"> | <img src="assets/results/accuracy-512.png" width="260" alt="Accuracy curves, 512-token window"> |

<p align="center">
  <img src="assets/results/autoregressive-decoder.png" width="420" alt="Autoregressive DecR accuracy by decode position">
</p>

<p align="center"><sub><code>DecR</code> accuracy as a function of decode position: strong for the first few characters, then flattening out.</sub></p>

## Architecture

The containers are redefined inline in each notebook; there is no shared module. The core set, as it appears in the canonical notebook:

| Name | What it does |
| --- | --- |
| `F` | Wraps a function as a `Module` (`s.f(x)` or `s.f(x, s.a)`). |
| `Enc` | Runs a `Sequential` stack until the length axis is 1 and returns `x[..., 0]` — a fixed-size encoder. |
| `EncDec(*a, n=0)` | **The core.** Downsamples one level, predicts at that level, upsamples back to full length; repeats and concatenates. `n` stops the recursion early. |
| `Add` | `Sequential` that sums branch outputs: `sum((s(x) for s in s[1:]), s[0](x))`. |
| `Res` | `Sequential` plus a residual add. |
| `Rep(n, *a)` | Applies a stack `n` times. |
| `Mean(Add)` | `Add` divided by the number of branches. |
| `DecR(gru, head)` | Autoregressive decoder: steps a `GRU(1, k)` over a `linspace(0, 1, i)` time embedding, yielding a reconstruction of length `i` at each step. |
| `EmbeddingAuto(EmbeddingBag)` | Static learned n-gram embedding: `mode='sum'` with a learned exponential decay `l = 0.5 ** arange(l-1, -1, -1)` passed as `per_sample_weights`. |
| `Affine`, `Log` | Per-channel scale+bias; and a live `matplotlib` logger that redraws the running metric. |

`EncDec` in full:

```python
class EncDec(Sequential):
    def forward(s, x):
        Y = []
        for i in range(int(np.log2(x.size(2)))):
            x = s[0](x)                          # downsample one more level
            y = s[1](x)                          # predict the sequence at this level
            for i in range(i + 1):
                y = s[2](y)                      # upsample the prediction to full length
            Y.append(y)
        return torch.cat(Y)                      # one logit block per scale
```

The canonical model:

```python
Embedding(256, k) → transpose(1, 2) → EncDec(...) → Conv1d(k, k, 1)
→ BatchNorm1d → SiLU → Conv1d(k, 256, 1)
```

Training splits the concatenated logits by scale and supervises every block against the same target:

```python
x = m(y).unflatten(0, (int(np.log2(a.size(1))), -1))
functional.cross_entropy(x.transpose(1, 2),
                         y.expand(len(x), -1, -1)).backward()
```

Adam + `OneCycleLR`; the reported metric is top-1 accuracy per scale. Window lengths across the notebooks range from 8 to 2,048 tokens.

The fold-into-`Sequential` idiom is shared with the author's [cznn](https://github.com/ZisIsNotZis/cznn), but there is no code dependency in either direction.

## Data

Both corpora come from the author's private parent archive. **Neither the raw text nor the derived binaries are included here.**

| Corpus | Source | Derived artifacts |
| --- | --- | --- |
| Chinese news (`inews`) | `../nlpdat/inews/{train,dev,test}.txt`, `_!_`-delimited | `inews.U` (raw UTF-16 units), `inews.7622` (character vocabulary), `inews.B` (CJK, top-256 bytes) |
| Python source | `../nlpdat/py/{py,pyp,ipynb,ipynbp}.txt` | `all.B` (byte-level), `all.90` (90-symbol vocabulary); `all.sort.I` is an intermediate |

`inews` is loaded as UTF-16 code units:

```python
def inews(a="../nlpdat/inews/dev.txt"):
    a = np.loadtxt(a, "U", None, "_!_", usecols=(2, 3), encoding="utf-8")
    a = a.view("I").view("2H")[..., 0]
    a[:, -1] = ord("。")
    return a[a.astype("?")]
```

`01_multiscale_autoencoder.ipynb` can regenerate `inews.U`, `inews.7622`, and `inews.B` from the raw text. `all.B` / `all.90` cannot be rebuilt from anything in this repository: their source text is gone.

## How to develop

There is no build, no test suite, and no CI. The notebooks are the source of truth, and each one intentionally redefines its containers so it can be read on its own.

- **Environment:** Python 3, PyTorch, NumPy, matplotlib, tqdm, numba, JupyterLab. `pip install torch numpy matplotlib tqdm numba jupyterlab`.
- **CUDA-only:** most training notebooks call `torch.set_default_tensor_type(torch.cuda.FloatTensor)` with no `.cpu()` fallback. Remove that line for a CPU host.
- **Data paths:** place the derived binaries where the notebook expects them, or edit the `np.memmap` paths. Do not commit corpora, checkpoints, or generated binaries — `.gitignore` covers `*.pk`, `*.pt`, `*.npy`, and the `inews.*` / `all.*` families.
- **Run in a scratch directory:** the notebooks write generated binaries into their own working directory and to `../nlpdat/py/all.sort.I` outside the repo.
- **Before committing:** clear notebook outputs. The four saved plots are already extracted under [`assets/results/`](assets/results/); there is no need to keep large embedded attachments in new work.

## How to contribute

This is a research archive, not a maintained library, but focused fixes are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the short version: open an issue with a clear reproduction, keep pull requests small, and never add the private corpora or model checkpoints.

Useful contributions: reproducibility notes, clearer data-prep steps, a CPU path, and extracting the repeated containers into an importable module.

## Roadmap

**Implemented**

- Multi-scale `EncDec` with reconstruction supervised at every scale.
- Byte-level (256) and character-level (7,622) tokenisation for Chinese news text.
- `DecR` autoregressive decoder and fixed-vector `Enc` next-token baselines.
- `EmbeddingAuto` learned n-gram embedding and head variants.
- Order-k conditional entropy, context-uniqueness counts, majority-continuation n-grams, and a suffix-array predictor.
- Four saved per-scale accuracy curves, extracted under `assets/results/`.

**Planned**

- Extract `F`, `Enc`, `EncDec`, `Add`, `Res`, `Rep`, `Mean`, `DecR`, and `EmbeddingAuto` into an importable `seq_autoenc` package.
- Document a corpus-prep recipe that reproduces `inews.B`, `inews.7622`, and `inews.U` from the private raw text.
- Add CPU/MPS fallbacks and a small synthetic smoke test that runs without the corpora.
- Record run configurations and numeric logs next to every saved plot.
- Write a short technical note that turns the four curves into a result, not just an artifact.

### Where this could go

The idea worth keeping is the per-scale supervision: a single convolutional stack that is forced to be useful at several temporal resolutions at once. With a documented corpus recipe and a small importable module, the same object could serve as a compact testbed for hierarchical sequence models — and the n-gram / suffix-array baselines already in the repository would keep that testbed honest.

## Status and limitations

This is a 2026 personal research archive, published for provenance and readability. It is not maintained, and it is not reproducible as shipped.

- The corpora are missing, so the training notebooks cannot run end-to-end.
- Most notebooks are CUDA-only.
- Results survive as four plots, not as numeric logs or checkpoints.
- The plot axes are unlabelled and the run configurations were not recorded.
- There is no packaging, CLI, test suite, or CI.

The detailed evidence and boundaries live in [`docs/project-status.md`](docs/project-status.md).

## License

[GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0-only).
