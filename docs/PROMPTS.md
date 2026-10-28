# Prompts used in the paper

All judge and agent prompts are reproduced here verbatim from `code/phase1_v13.py`.
The prompts are written in Chinese; English translations are given for reference.

## Substitutability judge (primary prompt, `L1_PROMPT`, variant `base`)

SHA-256 of the template string: `c2a47c11a8ba8225c9ffceb2932e8d4f953d7d090305dc2c81c3673f11dfa9dc`

This is the prompt used by the primary run (`main_v10`), by the reasoning-enabled
configuration (`main_v12_think`), and by the cross-family judge (`e5a_*`).
Which prompt `main_v10` actually used was verified empirically, not only from
configuration files: re-judging 160 stored pairs at temperature 0 reproduces the stored
`required_facts` verbatim for 98.1% of pairs under this prompt and for at most 40.6%
under any other candidate (`results/phase3/audit/prompt_audit.json`).

`{q}` is the query; `{docs}` is the other query's top-10 retrieved passages, each
truncated to 1,100 characters and numbered in retrieval-rank order.

```text
你要判断一组检索结果是否足以回答一个问题。

**问题**
{q}

**检索结果**
{docs}

请按以下步骤：
1. 列出回答这个问题**必须**具备的事实点（1-3 条，具体到实体和属性）。
2. 逐条核对检索结果中是否明确包含该事实。
3. 给出结论。

注意：只看检索结果里有没有，不要用你自己的知识补全。结果里没有明确写出的，一律 false。

只输出 JSON，不要任何其他文字：
{{"required_facts":["..."],"found":[true,false],"verdict":"SUFFICIENT"|"PARTIAL"|"INSUFFICIENT","confidence":0.0}}
```

A call counts as *covered* only when `verdict == "SUFFICIENT"`. The paper's
"lenient" sensitivity analysis (`e5l_*` runs) also counts `PARTIAL` as covered.

### English translation (used only in the language-sensitivity check)

SHA-256 prefix `83cc2809ff1f`. Used by the `xfam_llama33_en` and `qwen_base_en` gate
checks (`results/phase3/e5/*_en/`), never for a full run.

```text
You need to decide whether a set of retrieval results is sufficient to answer a question.

**Question**
{q}

**Retrieval results**
{docs}

Follow these steps:
1. List the facts that are **required** to answer this question (1-3 items, specific to entities and attributes).
2. Check one by one whether the retrieval results explicitly contain each fact.
3. Give the conclusion.

Note: only consider what is in the retrieval results; do not fill in with your own knowledge. Anything not explicitly stated in the results counts as false.

Output only JSON, with no other text:
{{"required_facts":["..."],"found":[true,false],"verdict":"SUFFICIENT"|"PARTIAL"|"INSUFFICIENT","confidence":0.0}}
```

### Other prompt variants

`code/phase1_v13.py` defines six further variants (`_prompt_variants()`) that were
compared in a prompt bake-off (stage s22, selected by accuracy on the development
labels). `main_v11_k3` used the order-invariant variant (`L1_PROMPT_ORDER_INVARIANT`),
which adds an explicit instruction that passage order is irrelevant.

## Agent system prompt (`AGENT_SYS`)

```text
你是研究助手，可以多次调用 search 工具检索资料。每次只提出一个具体、自包含的检索查询。收集到足够信息后用 FINAL: <答案> 给出最终答案。
```

English: "You are a research assistant and may call the `search` tool multiple times.
Issue one specific, self-contained search query at a time. When you have enough
information, give the final answer as FINAL: <answer>."

The agent has a single tool, `search(query: string)`, returning the top-10 passages.
