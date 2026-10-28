# Large files on Zenodo

DOI: `10.5281/zenodo.XXXXXXX`

The Zenodo record holds files too large for git, gzip-compressed, with paths mirroring
the run layout (`runs/<run>/stages/<stage>/<file>.gz`):

| File | Content |
|---|---|
| `runs/<run>/stages/s2/calls.jsonl.gz` | every retrieval call: task, source, turn, query text, retrieved passage ids, scores, cost |
| `runs/<run>/stages/s4/pairs.jsonl.gz` | candidate query pairs from blocking, with cosine, Jaccard, RBO, same-text flag |
| `runs/<run>/stages/s5/l1_pairs.jsonl.gz` | the stratified sample sent to the judge (with stratum weights) |
| `runs/<run>/stages/s6/l1.jsonl.gz` | per-pair judge verdicts: both directions, required facts, eligibility, weights |
| `runs/phase3/e5/*/partial.jsonl.gz` | raw per-pair outputs of every E5 re-judge, including parse failures |
| `runs/phase3/e5/*/selfcov.jsonl.gz` | cross-family judge self-coverage verdicts per query |
| `runs/phase3/e4/collect_partial.jsonl.gz` | raw Tongyi agent outputs on 2WikiMultihopQA tasks |
| `runs/phase3/audit/fingerprint.jsonl.gz` | prompt-audit re-judgements |
| `phase3_logs.tar.gz` | step logs of the E4/E5 driver |
| `SHA256SUMS`, `MANIFEST.json` | checksums, line counts, and which files are symbolic links to `main_v10` |

Runs derived from `main_v10` link its stages s1–s5 instead of copying them; those files
appear once, under `main_v10`.

Unpack:

```bash
cd work && mkdir -p runs
for f in $(cd zenodo && find runs -name '*.gz'); do
  mkdir -p "$(dirname "${f%.gz}")"; gunzip -c "zenodo/$f" > "${f%.gz}"; done
cp -r ../results/runs/* runs/      # small JSON outputs from the repository
```

## How the bundle is built

On the experiment machine, from the working directory that holds `runs/`:

```bash
python tools/export_artifacts.py --projb . --repo <repo> --out ../release \
    --models-dir $R/models --labels labels_data.tsv
```

Then upload every file in `../release/zenodo/` to a new Zenodo upload (type: Dataset),
publish it, and replace the DOI placeholder in `README.md` and in this file.
