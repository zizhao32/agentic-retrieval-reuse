#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_artifacts.py — collect the reproducibility package from the experiment server.

Run it on the machine that holds the `runs/` directory (the "projB" working directory).
It does three things and never modifies anything under `runs/`:

  1. code     Copies the scripts that were actually run (projB/*.py, *.sh) into
              <repo>/code/, and reports any file whose content differs from the copy
              already in the repository.
  2. repo     Copies small result files (JSON summaries, calibrations, configs, human
              labels) into <repo>/results/ and <repo>/human_labels/.
  3. zenodo   Writes large files (retrieval traces, per-pair judge verdicts, raw judge
              outputs, candidate pairs) gzip-compressed into <out>/zenodo/, with
              SHA256SUMS and MANIFEST.json, ready to upload to Zenodo.

It also records model revisions from the Hugging Face download metadata.

Usage (from projB):
    R=/mnt/data/user_workspace/zhangzizhao/matview
    $R/envs/main/bin/python <repo>/tools/export_artifacts.py \
        --projb . --repo <repo> --out ../release \
        --models-dir $R/models --labels labels_data.tsv
    # add --with-corpus to also export the passage corpus (check dataset licenses first)
    # add --dry-run to only print what would be copied
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Runs that back numbers in the paper. Missing runs are reported and skipped.
RUNS = [
    "main_v10",                        # primary run (Qwen3-32B-AWQ judge, base prompt)
    "main_v12_think",                  # same judge, reasoning enabled ("think1")
    "main_v11_k3",                     # same judge, order-invariant prompt, 3 shuffled orders
    "agent_b",                         # second agent model on MuSiQue tasks
    "distractor",                      # corpus expanded with distractor passages
    "e4_agent2wiki",                   # Tongyi agent on 2WikiMultihopQA (E4)
    "e5a_xfam_llama33_70b_awq",        # cross-family judge (E5a)
    "e5d_main_botheligible",           # primary judge on the both-eligible subset
    "e5d_xfam_llama33_botheligible",   # cross-family judge on the both-eligible subset
    "e5l_main_lenient",                # PARTIAL counted as covered, primary judge
    "e5l_xfam_llama33_lenient",        # PARTIAL counted as covered, cross-family judge
]
# Large per-run files that go to Zenodo (relative to runs/<run>/stages/).
BIG_STAGE_FILES = ["s2/calls.jsonl", "s6/l1.jsonl", "s5/l1_pairs.jsonl",
                   "s4/pairs.jsonl", "s7/l2.jsonl"]
CORPUS_FILES = ["s1/chunks.jsonl"]
# Raw judge / agent outputs from the phase-3 driver (go to Zenodo).
PHASE3_BIG = ["e5/*/partial.jsonl", "e5/*/selfcov.jsonl", "e4/collect_partial.jsonl",
              "e4/agent_2wiki_calls.jsonl", "e4/validate_pairs.jsonl",
              "audit/fingerprint.jsonl"]
SMALL_MAX = 20 * 2**20      # files above this never go into the git repository
# BrowseComp-Plus is distributed encrypted to prevent benchmark contamination; its
# plaintext must not be redistributed.
EXCLUDE_TOKENS = ("bcp", "browsecomp", "phase2_B")
CODE_SKIP = {"__pycache__"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def excluded(p: Path) -> bool:
    s = str(p).lower()
    return any(t.lower() in s for t in EXCLUDE_TOKENS)


class Exporter:
    def __init__(self, a):
        self.a = a
        self.projb = Path(a.projb).resolve()
        self.runs = self.projb / "runs"
        self.repo = Path(a.repo).resolve()
        self.out = Path(a.out).resolve()
        self.zen = self.out / "zenodo"
        self.manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "repo_files": [],
                         "zenodo_files": [], "missing": [], "skipped_symlinks": [],
                         "code_diffs": [], "models": {}}
        if not self.runs.is_dir():
            sys.exit(f"{self.runs} not found: run from projB or pass --projb")

    def log(self, *x):
        print(*x, flush=True)

    # ------------------------------------------------------------------ helpers
    def to_repo(self, src: Path, rel: str):
        if excluded(src):
            return
        dst = self.repo / rel
        size = src.stat().st_size
        if size > SMALL_MAX:
            self.log(f"  [too big for git -> zenodo] {rel} ({size/2**20:.1f} MB)")
            return self.to_zenodo(src, rel)
        if self.a.dry_run:
            self.log(f"  repo   {rel}  ({size/1024:.0f} KB)")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self.manifest["repo_files"].append({"path": rel, "bytes": size, "sha256": sha256(dst)})

    def to_zenodo(self, src: Path, rel: str):
        if excluded(src):
            return
        dst = self.zen / (rel + ".gz")
        size = src.stat().st_size
        if self.a.dry_run:
            self.log(f"  zenodo {rel}.gz  ({size/2**20:.1f} MB raw)")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
            for line in fi:
                fo.write(line)
                n += 1
        self.manifest["zenodo_files"].append({
            "path": rel + ".gz", "raw_bytes": size, "gz_bytes": dst.stat().st_size,
            "lines": n, "sha256_gz": sha256(dst), "sha256_raw": sha256(src)})
        self.log(f"  zenodo {rel}.gz  {n} lines, {size/2**20:.1f} MB -> {dst.stat().st_size/2**20:.1f} MB")

    # ------------------------------------------------------------------ steps
    def code(self):
        self.log("\n[1] code: scripts that were actually run")
        dst = self.repo / "code"
        for src in sorted(list(self.projb.glob("*.py")) + list(self.projb.glob("*.sh"))):
            if src.name in CODE_SKIP or src.name.startswith("."):
                continue
            mine = dst / src.name
            if mine.exists() and sha256(mine) != sha256(src):
                self.manifest["code_diffs"].append(src.name)
                self.log(f"  differs from repo copy, taking the server version: {src.name}")
            elif not mine.exists():
                self.log(f"  new (not in repo yet): {src.name}  -- check whether it belongs")
            if not self.a.dry_run:
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, mine)

    def runs_small_and_big(self):
        self.log("\n[2] per-run results")
        for run in RUNS:
            rd = self.runs / run
            if not rd.is_dir():
                self.manifest["missing"].append(run)
                self.log(f"  missing run: {run}")
                continue
            self.log(f"  {run}")
            for f in ("config_effective.json", "state.json", "autotune.json"):
                if (rd / f).exists():
                    self.to_repo(rd / f, f"results/runs/{run}/{f}")
            st = rd / "stages"
            for p in sorted(st.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(st).as_posix()
                if p.is_symlink():
                    # e5*/e4 runs link s1-s5 from main_v10; record the link, export once.
                    self.manifest["skipped_symlinks"].append(
                        {"run": run, "file": rel, "target": os.path.realpath(p)})
                    continue
                if p.suffix == ".json":
                    self.to_repo(p, f"results/runs/{run}/stages/{rel}")
                elif rel in BIG_STAGE_FILES:
                    self.to_zenodo(p, f"runs/{run}/stages/{rel}")
                elif rel in CORPUS_FILES and self.a.with_corpus:
                    self.to_zenodo(p, f"runs/{run}/stages/{rel}")
            # human dev labels (the gold set) live in main_v10's s5
            if run == "main_v10" and (st / "s5" / "gold_pairs.jsonl").exists():
                self.to_repo(st / "s5" / "gold_pairs.jsonl", "human_labels/dev/gold_pairs.jsonl")

    def phase2(self):
        self.log("\n[3] phase-2 experiment outputs (BrowseComp-Plus excluded)")
        d = self.runs / "phase2"
        if not d.is_dir():
            self.manifest["missing"].append("phase2")
            return
        for p in sorted(d.rglob("*")):
            if p.is_file() and not p.is_symlink() and not excluded(p):
                rel = p.relative_to(self.runs).as_posix()
                if p.suffix in (".json", ".md", ".txt", ".tsv", ".csv"):
                    self.to_repo(p, f"results/{rel}")
                elif p.suffix == ".jsonl":
                    self.to_zenodo(p, f"runs/{rel}")

    def phase3(self):
        self.log("\n[4] phase-3 (E4/E5) outputs")
        d = self.runs / "phase3"
        if not d.is_dir():
            self.manifest["missing"].append("phase3")
            return
        big = set()
        for pat in PHASE3_BIG:
            big |= set(d.glob(pat))
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            rel = p.relative_to(d).as_posix()
            if rel.startswith("logs/") or rel.endswith(".pid") or rel.endswith(".tmp"):
                continue
            if p in big:
                self.to_zenodo(p, f"runs/phase3/{rel}")
            elif rel.startswith("e5/human/"):
                self.to_repo(p, f"human_labels/heldout/{rel[len('e5/human/'):]}")
            else:
                self.to_repo(p, f"results/phase3/{rel}")
        # logs are useful for auditing but noisy: archive them on Zenodo as one file
        logs = d / "logs"
        if logs.is_dir() and not self.a.dry_run:
            self.zen.mkdir(parents=True, exist_ok=True)
            shutil.make_archive(str(self.zen / "phase3_logs"), "gztar", root_dir=d, base_dir="logs")
            self.log(f"  zenodo phase3_logs.tar.gz")

    def labels(self):
        if self.a.labels and Path(self.a.labels).exists():
            self.log("\n[5] held-out human labels")
            self.to_repo(Path(self.a.labels), "human_labels/heldout/labels.tsv")
        for extra in self.a.extra_labels or []:
            if Path(extra).exists():
                self.to_repo(Path(extra), f"human_labels/heldout/{Path(extra).name}")

    def models(self):
        self.log("\n[6] model revisions (from Hugging Face download metadata)")
        md = Path(self.a.models_dir) if self.a.models_dir else None
        if not md or not md.is_dir():
            self.log("  --models-dir not given; skipped")
            return
        for d in sorted(md.iterdir()):
            if not (d / "config.json").exists():
                continue
            rev = None
            meta = d / ".cache" / "huggingface" / "download" / "config.json.metadata"
            if meta.exists():
                rev = meta.read_text(errors="ignore").splitlines()[0].strip() or None
            self.manifest["models"][d.name] = {"revision": rev,
                                               "config_sha256": sha256(d / "config.json")}
            self.log(f"  {d.name:<40} revision={rev or 'unknown (not downloaded with hf download?)'}")

    def finish(self):
        if self.a.dry_run:
            self.log("\n(dry run: nothing written)")
            return
        self.out.mkdir(parents=True, exist_ok=True)
        if self.manifest["zenodo_files"]:
            with open(self.zen / "SHA256SUMS", "w") as f:
                for e in self.manifest["zenodo_files"]:
                    f.write(f"{e['sha256_gz']}  {e['path']}\n")
            (self.zen / "MANIFEST.json").write_text(
                json.dumps(self.manifest, indent=2, ensure_ascii=False))
        (self.repo / "results" / "EXPORT_MANIFEST.json").parent.mkdir(parents=True, exist_ok=True)
        (self.repo / "results" / "EXPORT_MANIFEST.json").write_text(
            json.dumps({k: v for k, v in self.manifest.items() if k != "zenodo_files"} |
                       {"zenodo_files": [{k: e[k] for k in ("path", "lines", "sha256_gz")}
                                         for e in self.manifest["zenodo_files"]]},
                       indent=2, ensure_ascii=False))
        rb = sum(e["bytes"] for e in self.manifest["repo_files"])
        zb = sum(e["gz_bytes"] for e in self.manifest["zenodo_files"])
        self.log("\nsummary")
        self.log(f"  repo:   {len(self.manifest['repo_files'])} files, {rb/2**20:.1f} MB")
        self.log(f"  zenodo: {len(self.manifest['zenodo_files'])} files, {zb/2**30:.2f} GB "
                 f"compressed -> {self.zen}")
        if self.manifest["missing"]:
            self.log(f"  missing (check): {self.manifest['missing']}")
        if self.manifest["code_diffs"]:
            self.log(f"  code files that differ from the repo copy: {self.manifest['code_diffs']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projb", default=".")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out", default="../release")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--labels", default=None, help="held-out human labels TSV (e.g. labels_data.tsv)")
    ap.add_argument("--extra-labels", nargs="*", help="further label files, e.g. overlap rounds")
    ap.add_argument("--with-corpus", action="store_true",
                    help="also export s1/chunks.jsonl (passages from MuSiQue/HotpotQA/2Wiki)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    e = Exporter(a)
    e.code()
    e.runs_small_and_big()
    e.phase2()
    e.phase3()
    e.labels()
    e.models()
    e.finish()


if __name__ == "__main__":
    main()
