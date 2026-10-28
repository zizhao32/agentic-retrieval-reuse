# Human labels

Two sets of human labels are used in the paper. Both label the same question the LLM
judge answers: *given a query and a set of retrieved passages, do the passages contain
every fact needed to answer the query?*

## `dev/` — development labels (the "gold" set)

`dev/gold_pairs.jsonl` (exported from `runs/main_v10/stages/s5/gold_pairs.jsonl`).
One record per query pair; `human_label` is the pair-level label (1 = the two queries
are mutually substitutable). These labels were **used to select the judge prompt**
(stage s22 bake-off), so judge accuracy measured on them is optimistic for the selected
configuration. The paper reports them as development labels for that reason.

## `heldout/` — held-out labels (experiment E5c)

Built with `code/exp45.py e5c-export`; never used for any selection or tuning.

* **Sampling.** 200 query pairs drawn from the primary run's eligible pairs, stratified
  into five cosine-similarity bands — `[0.80,0.90)`, `[0.90,0.95)`, `[0.95,0.98)`,
  `[0.98,0.999)`, `[0.999,1]` — 40 pairs per band, excluding every development-set pair
  and every pair whose label is logically forced (byte-identical queries with identical
  result sets). Seed: 20261029.
* **Items.** Each pair yields two items, one per direction (query *i* against *j*'s
  passages, and vice versa); the 400 items were shuffled globally, so the two
  directions of a pair are generally not adjacent and may go to different annotators.
  Annotators saw exactly the judge's context: the query and up to 10 passages truncated
  to 1,100 characters. No judge output, similarity score, or pair identity was shown.
* **Annotators and protocol.** Four annotators each labelled a disjoint quarter of the
  items (100 items each) following `heldout/GUIDE_zh.txt` (English translation in
  `heldout/GUIDE_en.txt`). Each item therefore has a single label. Labelling time was
  not recorded reliably (the `ts` column is not a per-item timestamp).
* **Labels.** `1` = sufficient, `0` = not sufficient, `?` = unsure (excluded from
  evaluation). The `note` column holds the annotator's justification, in Chinese.
* **Pair labels.** A pair is labelled 1 iff both of its directions are labelled 1;
  pairs with an unresolved direction are listed in `to_adjudicate.tsv`.
* **Mapping items to pairs.** `key.json` maps every `item_id` to `pair_id`, direction,
  band and cosine similarity. `annotate.html` is the exact page the annotators used.

### Known limitations

* Single annotation per item; inter-annotator agreement is not available for this round.
* Negatives are rare and clustered: of 23 negative item labels, 20 come from two
  underlying questions. Specificity estimates computed on this set therefore rest on
  very few independent cases and should not be read as general specificity.

Evaluate any judge run against these labels with
`python code/exp45.py e5c-eval --labels human_labels/heldout/labels.tsv`.
