# seq_autoenc project status

## Classification

Personal research archive: multi-scale sequence autoencoders (a convolutional
encoder–decoder that reconstructs the input sequence at every down-sampling
resolution at once) over Chinese news text and Python source, together with the
n-gram and suffix-array baselines that measure the data's own predictability.

## Status

Archived / inactive. 2026 personal research, published for provenance. The
notebooks were run interactively on CUDA machines; four result plots survive as
cell attachments and are extracted under `assets/results/`, but no run is
reproducible as shipped because both the raw and the derived corpora are
missing. Not maintained.

License: AGPL-3.0-only (see `../LICENSE`).

## Evidence

- 12 notebooks, `nbformat` 4.5, `python3` kernel, no `.py` files.
- `01_multiscale_autoencoder.ipynb` — canonical. Full pipeline in order: the
  `inews` loader plus `toH`/`toB` tokenisers that write `inews.U` (raw UTF-16),
  `inews.7622` (7,622-character vocab) and `inews.B` (256-byte vocab, CJK-only),
  the container set (`F`, `Enc`, `EncDec`, `Add`, `Res`, `Rep`, `Mean`, `Log`),
  the canonical 256-byte multi-scale autoencoder, the `DecR` GRU autoregressive
  decoder, a 7,622-character-vocab variant, and the four saved result images.
- Core module, `EncDec.forward`: at each of `log2(seq_len)` levels, downsample one
  more step, run a per-scale head, upsample the head's output back to full length,
  append; `torch.cat` the blocks. Training splits the concatenated logits with
  `unflatten(0, (log2(len), -1))` and applies `cross_entropy` against `y.expand(...)`
  so **every scale is supervised**; Adam + `OneCycleLR`; metric is per-scale top-1 accuracy.
- `DecR` — autoregressive alternative: a `GRU(1, k)` stepped over a
  `linspace(0, 1, i)` time embedding, yielding reconstructions of length `i`.
- `EmbeddingAuto(EmbeddingBag)` — static learned n-gram embedding: `mode='sum'`
  with a learned exponential decay `l = 0.5 ** arange(l-1, -1, -1)` passed as
  `per_sample_weights`. Used in `06_learned_ngram_embedding.ipynb` and
  `07_embedding_head_variants.ipynb`; the latter also prints
  `m[0].l/m[0].l.sum()` to expose the effective n-gram order, and tests
  `Softmax`/`LogSoftmax`/custom `Affine` on the bag output.
- Fixed-vector baselines: `05_fixed_vector_encoder.ipynb` (`inews.7622`) and
  `08_python_source_encoder.ipynb` (Python source, 90 symbols).
- Baselines: `09_conditional_entropy.ipynb` measures order-k conditional entropy
  of Chinese text for `k = 1..8` (`(log2(s) - m @ log2(m) / s) / k`);
  `10_context_uniqueness.ipynb` counts unique-context fractions for `n = 4, 8, 12`
  over `inews.7622` and `../nlpdat/py/all.B` with `numba`-`njit` helpers (`cat`,
  hand-written `qsort`); `11_ngram_continuation.ipynb` builds
  most-frequent-continuation tables for `n = 2..5` at both char and byte level and
  scores the majority-continuation accuracy; `12_suffix_array_predictor.ipynb` is a
  suffix-array predictor (`argsort` over strided 16-byte views + `np.searchsorted`)
  that greedily generates text.
- Data paths: `../nlpdat/inews/{train,dev,test}.txt` (`_!_`-delimited),
  `../nlpdat/py/{py,pyp,ipynb,ipynbp}.txt`, derived `inews.B` / `inews.7622` /
  `inews.U` / `all.B` / `all.90`, intermediate `../nlpdat/py/all.sort.I`.

## Boundaries

- **No written-up results.** No paper, report, notes, or tables. The directory
  holds exactly six markdown cells: four are image captions and two are the
  `inews` loader mistakenly pasted as markdown rather than code.
- **Results survive only as four embedded PNGs** in
  `01_multiscale_autoencoder.ipynb` (three per-scale accuracy plots labelled
  64 / 256 / 512, one `DecR` autoregressive plot), extracted to
  `assets/results/`. Axes are unlabelled and the run configuration is not
  recorded. No numeric logs, no `.csv`, no metrics files.
- **No cell outputs.** All `stream` / `execute_result` / `error` outputs were
  cleared before archiving; the four attachments are the only outputs left.
- **No data, raw or derived.** Neither `../nlpdat/...` nor the generated
  `inews.B` / `inews.U` / `inews.7622` / `all.B` / `all.90` binaries are included,
  and `all.B` / `all.90` cannot be regenerated from anything here (their source
  `.txt` files are gone). The training loops therefore cannot be re-run end-to-end.
- **No checkpoints.** `60.pk` and `214.pk` are loaded and absent;
  `08_python_source_encoder.ipynb` writes a checkpoint that nothing reads.
- **CUDA-only.** Most notebooks set
  `torch.set_default_tensor_type(torch.cuda.FloatTensor)` with no CPU fallback.
- **No packaging, tests, CLI, or importable module.** Notebooks only.
- **No prose** explaining the hypothesis, and no record of how the
  `pyp.txt` / `ipynbp.txt` variants differ from `py.txt` / `ipynb.txt`.
- **No CI** by design.
- **Side effects on run:** the source notebooks write generated binaries into the
  notebook's own cwd (`inews.B`, `inews.U`, `inews.7622`) and to
  `../nlpdat/py/all.sort.I` outside the repo. Run them in a scratch directory.
