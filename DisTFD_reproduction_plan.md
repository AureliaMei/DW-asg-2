# Reproduction plan: DisTFD

**Paper:** Wanqing Wu and Wenyu Mao, "An Efficient and Scalable Algorithm to Mine Functional Dependencies from Distributed Big Data", *Sensors* 22(10):3856, 2022. DOI [10.3390/s22103856](https://doi.org/10.3390/s22103856). Full text: [PMC9142976](https://pmc.ncbi.nlm.nih.gov/articles/PMC9142976/).

**Goal:** re-implement DisTFD in PySpark and reproduce the paper's **accuracy experiment** (Table 3) on the paper's own dataset, UCI [Abalone](https://archive.ics.uci.edu/dataset/1/abalone) (4,177 × 9). Only DisTFD is compared with the TANE ground truth: the paper's Cet and HFDD baselines and its other experiments (node scalability, data scale, cache, load balancing) are out of scope. We have no code from the authors, so this is a re-implementation from the paper's text and pseudocode.

---

## 1. What the paper claims

DisTFD finds all **minimal, non-trivial functional dependencies** (X → A) in a table that is horizontally split across a cluster. An FD can hold inside every partition yet fail globally, so checking must be global. Four ideas make it fast:

| Technique | What it does | Paper reference |
|---|---|---|
| **Sampling** | Systematic sample of the data; any candidate that fails on the sample is discarded without touching the full data | Sampler, SamplingValidate |
| **Two-way lattice traversal** | Validate levels alternately from the bottom (small LHS) and top (large LHS) of the attribute lattice. A valid FD prunes its supersets (Lemma 3); an invalid one prunes its subsets (Lemma 2). The paper applies it in sample validation | §3.4, Lemmas 2–3 |
| **Redistribution + parallel validation** | Hash-partition rows on one attribute so equal values sit on the same node; compute equivalence classes locally, then merge (Eq. 2). X → A holds iff Π_X = Π_XA (Theorem 1) | Algorithm 5 GlobalValidate |
| **Prefix-tree partition cache** | Cache stripped partitions in a prefix tree keyed by attribute sets; build new partitions by intersecting the cached subsets with the fewest rows | Algorithm 3 RetrievePartition |
| **Greedy load balancing** | For skewed attributes (Inc_A = max value frequency / n above a threshold t), split heavy keys across nodes and fill the remaining nodes up to the average load | Algorithm 4 Assignment |

**Result to reproduce (Table 3, Abalone 4,177 × 9, TANE as truth):**

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Cet | 0.9852 | 0.9708 | 0.9780 |
| HFDD | 1 | 0.9781 | 0.9890 |
| DisTFD | 1 | 0.9854 | 0.9926 |

We reproduce the DisTFD row only. The paper does not say what counts as one TP / FP / FN, the LHS limit, the sample ratio, the skew threshold, or the node count for this experiment.

---

## 2. Gaps in the paper (decide these up front, and document each choice)

| Gap | Our default | Why |
|---|---|---|
| Sample size / ratio not given (Algorithm 1 calls n a "sampling ratio", §3.3 a "sample size") | Ratio 10% → n = 418 rows, min 50 rows; sweep 5%, 10%, 25% | §3.3 needs a row count; the sweep shows the effect of the choice |
| Skew threshold t not given | t = 0.1 (most frequent value > 10% of rows) | Matches the lowest skew level in Fig. 10 |
| Cache eviction period / memory budget not given | LRU, at most 4,096 cached partitions per task | Simple and deterministic |
| Lattice order is shown only for 4 attributes | Alternate inside the LHS cap L: levels 1, L, 2, L−1, …, for both sample and global validation; FDs found top-down are kept only if minimal after the lower levels finish | Follows the paper's example; global results then prune between levels |
| No LHS limit given | No cap (\|LHS\| ≤ 8, the full lattice) for DisTFD **and** for the ground truth | With 9 columns the full lattice is small, so every minimal FD is tested |
| TP / FP / FN not defined | One minimal FD (X, A) is one instance: TP in both, FP only ours, FN only TANE's | The natural reading |
| Null handling not given | Empty cells are one value (null = null), as in TANE's default | Same rule for DisTFD and the truth |
| Algorithm 4, step 3 is ambiguous | Split a key larger than Avg into ⌈freq/Avg⌉ chunks spread over the least-loaded nodes | Standard reading of greedy splitting |

---

## 3. Tech stack

- **PySpark 4.x** (Java 21) in **local mode with 4 cores**, standing in for 4 nodes. The paper used 8 Xeon servers; only accuracy is measured here, so a single machine is enough.
- **Ground truth:** Desbordante's exact TANE (`desbordante.afd.algorithms.Tane`, `error=0`, `max_lhs=8`, i.e. no cap). This is an independent C++ implementation.
- **Data:** UCI Abalone, downloaded from https://archive.ics.uci.edu/static/public/1/abalone.zip by `data/download_abalone.py`.

## 4. Work plan

1. **Data.** `data/download_abalone.py` downloads Abalone, adds the column names from `abalone.names`, keeps every value as its original string, and records the row and column counts and the SHA-256 hash in `data/abalone/manifest.json`.
2. **Core.** Stripped partitions, the prefix-tree cache (Algorithm 3), the lattice with Lemma 2/3 pruning, and two-way traversal (`distfd/`).
3. **Distributed validation.** Redistribute on each attribute, validate locally, and merge classes of split keys across nodes (Algorithm 5, Eq. 2). Greedy Assignment (Algorithm 4) handles skewed attributes.
4. **Tests.** Paper Example 1; brute-force agreement on toy tables; 1/2/4/8 partitions; all 16 on/off combinations of sampling, cache, balancing and two-way; Desbordante vs brute force.
5. **Experiment.** `experiments/accuracy.py`: DisTFD on Abalone with seeds 0, 1, 2, plus a sampling-ratio sweep, each compared with TANE (→ `results/abalone/`). `experiments/report_abalone.ipynb` compares the results with Table 3.

## 5. Success criteria

1. DisTFD's FD set matches TANE exactly (F1 = 1) on every seed. DisTFD is exact, so anything below 1 is a bug.
2. Report our numbers next to the paper's Table 3.
3. Reproducible: one command per step, fixed seeds, raw results saved as CSV and JSON.

## 6. Results

Run on 2026-10-16 (PySpark 4.2.0, Desbordante 2.5.0, local[4]). Full tables and discussion are in `experiments/report_abalone.ipynb`; raw numbers are in `results/abalone/`.

| Method | Precision (paper / ours) | Recall (paper / ours) | F1 (paper / ours) |
|---|---|---|---|
| DisTFD | 1 / **1.0** | 0.9854 / **1.0** | 0.9926 / **1.0** |

All 3 runs (seeds 0–2) return exactly TANE's 137 minimal FDs (|LHS| 3–6), with 0 disagreements. Sampling rejects 27–55% of candidates before the distributed check, depending on the ratio, and F1 stays 1.0.

## References
- Wu, W.; Mao, W. *Sensors* 2022, 22, 3856. https://www.mdpi.com/1424-8220/22/10/3856
- Huhtala, Y. et al. TANE: An efficient algorithm for discovering functional and approximate dependencies. *The Computer Journal*, 1999.
- Desbordante data profiler (TANE implementation used as ground truth): https://github.com/Desbordante/desbordante-core
- Abalone dataset (UCI Machine Learning Repository, Nash et al. 1994): https://archive.ics.uci.edu/dataset/1/abalone
