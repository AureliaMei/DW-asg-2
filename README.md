# DW Assignment 2: DisTFD

This assignment re-implements DisTFD in PySpark and reproduces its accuracy experiment on the Abalone dataset, with TANE as the ground truth. DisTFD is from Wu & Mao, "An Efficient and Scalable Algorithm to Mine Functional Dependencies from Distributed Big Data", *Sensors* 22(10):3856, 2022 ([doi:10.3390/s22103856](https://doi.org/10.3390/s22103856)).

## 1. The paper

**Problem.** A functional dependency (FD) X → A says that rows with equal values on the attributes X also have equal values on A. FDs are used to detect and repair dirty data, and are usually mined automatically. Classic miners such as TANE, FUN, FastFDs and HyFD assume one centralized table. On horizontally distributed data, checking each fragment separately is not enough. In the paper's Example 1, A → B holds on both fragments R1 and R2 but not on the full table R.

**Contribution.** DisTFD is a Spark algorithm that finds all minimal, non-trivial FDs in distributed data. It targets three weaknesses of earlier distributed miners:
- validating many candidates that are clearly invalid;
- load imbalance when data is redistributed on a skewed attribute;
- recomputing the same partitions for every candidate.

**Evaluation in the paper.** On an 8-server cluster, DisTFD is compared with a centralized baseline (Cet) and the distributed HFDD. Scalability, the partition cache and load balancing are tested on ONTS, Airline and Stud. Accuracy is tested on Abalone (4,177 × 9) against TANE (Table 3). For DisTFD it reports precision 1, recall 0.9854 and F1 0.9926.

## 2. The algorithm

DisTFD searches a lattice of candidate FDs X → A and returns the minimal ones that hold on the full data. The steps below follow the paper's Algorithm 1.

| Step | Paper | Code |
|---|---|---|
| Draw a systematic sample D′ (every k-th row from a random start). Candidates that fail on D′ fail on the full data and are discarded without distributed work. | §3.3 | [sampler.py](distfd/sampler.py) |
| Compute each attribute's skew (share of its most frequent value) and cardinality. Non-skewed attributes come first, sorted by cardinality, and skewed ones last. | Algorithm 2 | [distfd.py](distfd/distfd.py) |
| Generate candidates level by level from both ends of the lattice (two-way traversal). A valid FD prunes its supersets (Lemma 3), and an invalid FD prunes its subsets (Lemma 2). The paper applies this on the sample; the code also uses it for global validation, keeping the data redistributed on every attribute. | §3.4 | [lattice.py](distfd/lattice.py) |
| Redistribute the rows on an attribute of the candidate's LHS, so that rows with equal values sit on one node. Each node checks \|Π_X\| = \|Π_XA\| (Theorem 1), and classes of keys split across nodes are merged (Eq. 2). | Algorithm 5 | [spark_validate.py](distfd/spark_validate.py) |
| For a skewed attribute, split keys larger than the average node load into chunks and fill the remaining nodes up to that average (greedy assignment). | Algorithm 4 | [balance.py](distfd/balance.py) |
| Cache computed partitions in a prefix tree, and build new ones from the cached subsets with the fewest intersections. Least recently used partitions are evicted. | Algorithm 3 | [cache.py](distfd/cache.py) |

Partitions and Theorem 1 are in [partitions.py](distfd/partitions.py), and the ground truth is in [truth.py](distfd/truth.py). Where each part of the paper is implemented, and why, is described in [DisTFD_implementation_report.md](DisTFD_implementation_report.md).

## 3. The experiment

**Question.** Does DisTFD find the same FDs as TANE on the paper's accuracy dataset?

**Setup.**
- Data: UCI Abalone, 4,177 × 9.
- Ground truth: Desbordante's exact TANE.
- Search: the full lattice (|LHS| ≤ 8).
- Spark local mode with 4 cores as 4 nodes, seeds 0–2.
- Sample ratio 10%, skew threshold 0.1.

**Result.**

| | Precision | Recall | F1 |
|---|---|---|---|
| Paper (Table 3) | 1 | 0.9854 | 0.9926 |
| This reproduction | 1.0 | 1.0 | 1.0 |

DisTFD returns the same 137 minimal FDs as TANE on every seed, with 0 false positives and 0 false negatives. The agreement holds at every LHS size (3 to 6 attributes) and at sample ratios of 5%, 10% and 25%. The report with all sub-results is [experiments/report_abalone.ipynb](experiments/report_abalone.ipynb). The choices made where the paper gives no value are listed in [DisTFD_reproduction_plan.md](DisTFD_reproduction_plan.md).

## 4. Files

| Path | Content |
|---|---|
| [distfd/](distfd/) | DisTFD in PySpark, plus the TANE ground truth |
| [experiments/accuracy.py](experiments/accuracy.py) | Runs the experiment and writes `results/abalone/` |
| [experiments/report_abalone.ipynb](experiments/report_abalone.ipynb) | Experiment report |
| [data/download_abalone.py](data/download_abalone.py) | Downloads Abalone from UCI into `data/abalone/` |
| [tests/](tests/) | pytest suite: paper Example 1, brute-force agreement, node counts, optimization on/off combinations |
| [results/abalone/](results/abalone/) | Raw results: precision, recall and F1 per seed, sampling sweep, FD lists |
| [DisTFD_reproduction_plan.md](DisTFD_reproduction_plan.md) | Reproduction plan and the settings chosen where the paper gives none |
| [DisTFD_implementation_report.md](DisTFD_implementation_report.md) | Paper-to-code walkthrough |

## 5. How to run

Requires Python 3.12 and Java 21 (for PySpark).

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python data/download_abalone.py
.venv/bin/python -m pytest tests
.venv/bin/python experiments/accuracy.py
```
