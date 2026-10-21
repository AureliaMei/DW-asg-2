# DisTFD: from the paper to the code

A close reading of Wu & Mao, *An Efficient and Scalable Algorithm to Mine Functional Dependencies from Distributed Big Data* (*Sensors* 22(10):3856, 2022), from a student's point of view. For each part of the paper, this note shows where it lives in [distfd/](distfd/), how it is coded, and why it was coded that way. It also notes where the paper is vague or inconsistent, and what the code chose in those places.

Each section has the same four parts:
- **Paper:** what the paper says.
- **Code:** where it is implemented.
- **Why:** the reasons for the design.
- **Student notes:** ambiguities, slips in the paper, deviations, and limits.

Results are discussed in [DisTFD_reproduction_plan.md](DisTFD_reproduction_plan.md), and [experiments/report_abalone.ipynb](experiments/report_abalone.ipynb). The experiment compares DisTFD with TANE on the paper's Abalone data.

---

## 0. Map

| Paper | Module | Main functions |
|---|---|---|
| Definitions 1–6, Theorem 1 (§2.1, §3.5.1) | [partitions.py](distfd/partitions.py) | `Partition`, `product`, `holds` |
| Data input (not described) | [encode.py](distfd/encode.py) | `read_table`, `encode` |
| Sampler, SamplingValidate (§3.3) | [sampler.py](distfd/sampler.py) | `sample_size`, `systematic_indices`, `sampling_validate` |
| Pre_processing, Algorithm 2, Eq. 1 (§3.2) | [distfd.py](distfd/distfd.py#L56-L65), [balance.py](distfd/balance.py#L5) | `skewness`, attribute ordering in `discover` |
| Lattice, Lemmas 1–3, two-way traversal (§3.4) | [lattice.py](distfd/lattice.py) | `level_order`, `Lattice` |
| Prefix-tree cache, Algorithm 3 (§3.5.1) | [cache.py](distfd/cache.py) | `PartitionCache.get`, `_retrieve`, `_store` |
| Redistribution, Algorithm 5, Eq. 2 (§3.5.2) | [spark_validate.py](distfd/spark_validate.py) | `build_blocks`, `_check_block`, `global_validate` |
| Greedy Assignment, Algorithm 4 (§3.5.2) | [balance.py](distfd/balance.py) | `assignment`, `Router` |
| Algorithm 1 (driver) | [distfd.py](distfd/distfd.py#L35) | `discover`, `Config` |
| Ground truth TANE (§4.3) | [truth.py](distfd/truth.py) | `tane` |
| Cluster (§4.1) | [spark.py](distfd/spark.py) | `get_spark` |

**How data flows through `discover()`** ([distfd.py:35](distfd/distfd.py#L35)):

```
codes (int matrix)
  ├─ parallelize over cfg.nodes partitions            D = D1..Dn
  ├─ reduceByKey → exact value frequencies            constants (∅→A), input to Algorithm 4
  ├─ systematic sample → collect to driver            D' = Sampler(D, n)
  ├─ skew + cardinality on D' → attribute order       Algorithm 2
  ├─ Router per attribute (+ Assignment if skewed)    Algorithm 4
  ├─ build_blocks: redistribute on every attribute    "ReDistributedDataSet(D, Ai)"
  └─ for level in 1, L, 2, L-1, ...:                  two-way traversal
        lattice.candidates(level)                     Lemma 2 / Lemma 3 pruning
        sampling_validate on D'                       reject what fails on the sample
        route each survivor to its home attribute     Z = first LHS attribute in the order
        global_validate on the blocks                 Algorithm 5 + Eq. 2
        lattice.record(valid, invalid)
  → lattice.minimal()                                 keep only minimal FDs
```

The driver differs from the paper in one structural way. In the paper, two-way traversal runs only on the sample (§3.4), and global validation is then one pass over the attributes (Algorithm 1, lines 5–14). In the code, global validation also follows the two-way level order. That needs the data redistributed on every attribute at every level, so all redistributions are kept in memory. Section 7 explains the trade-off.

---

## 1. Partitions and Theorem 1

**Paper.** Definition 4 defines an equivalence class [t]_X, Definition 5 a partition Π_X, and Definition 6 the stripped partition Π̂_X, which drops classes of size 1. Theorem 1 says X → Y holds iff |Π_X| = |Π_XY|. §3.5.1 builds Π_XY as Π_X ∩ Π_Y, using an "attribute vector" in which singleton values are coded 0.

**Code.** [partitions.py](distfd/partitions.py)
- [`Partition`](distfd/partitions.py#L10) stores a partition as a **label array** (row → class id) plus `n_classes`. It also stores `stripped_size`, which is ‖Π̂‖: the number of rows in classes larger than 1.
- [`product`](distfd/partitions.py#L31) computes Π_X ∩ Π_Y. It combines the two labels into one key, `p.labels * q.n_classes + q.labels`, and relabels with `np.unique`.
- [`holds`](distfd/partitions.py#L38) is Theorem 1: `px.n_classes == pxa.n_classes`.

**Why.**
- A label array makes the intersection a single vectorised numpy call. The paper's attribute-vector method does the same job with Python-level grouping.
- Theorem 1 needs |Π| with **all** classes counted, singletons included. Comparing stripped partitions would need TANE's error measure e(X) instead. So the code compares full class counts and keeps ‖Π̂‖ only for the "smallest partition" choice in Algorithm 3 (section 6). That is the only place the paper itself uses partition size.

**Student notes.**
- The proof of Theorem 1 is informal: it argues from equal class counts to equal class contents. A cleaner argument is that Π_XY always refines Π_X. Equal counts therefore mean no X-class was split, so A is constant inside every X-class.
- The symbols in Definition 3 (minimal FD) are garbled in the PDF ("for any X̄ ∈ X, X̄ ∈ X ∈ A does not hold"). The intended meaning is the usual one: no proper subset X̄ ⊂ X has X̄ → A. That is what [`Lattice.minimal`](distfd/lattice.py#L74) implements.

---

## 2. Reading and encoding the data

**Paper.** Nothing is said about value types, nulls, or how rows are represented.

**Code.** [encode.py](distfd/encode.py)
- [`read_table`](distfd/encode.py#L8) reads every column as a **string**. Empty cells stay `""`, so null = null. Columns that are entirely empty are dropped.
- [`encode`](distfd/encode.py#L15) dictionary-encodes each column to int32 with `pd.factorize`.

**Why.**
- **Strings:** two cells are equal exactly when their text is equal. The Desbordante truth sees the same strings.
- **null = null:** this is TANE's default. Using the same rule for DisTFD and for the truth means null handling cannot cause disagreements.
- **Integer codes:** they are small, fast to hash, and pack into numpy blocks. They also hash the same way in every process. `Router` sends rows with `hash(key) % n_nodes`, and Python's string hash changes with each process's random seed (`hash('M') % 4` gave 2 and 0 under two seeds), while `hash(7)` is always 7. With string keys, Spark workers could disagree on where a value belongs.

---

## 3. Sampler and SamplingValidate (§3.3)

**Paper.** Systematic sampling: k is the integer closest to N/n, the start r is random in [1, k], and then every k-th row is taken until n rows are drawn. Candidates are checked on the sample D′ first. Those that fail go to the ResultSet as invalid; those that hold go on to global validation. Two properties are claimed: *Completeness* (an FD that holds on D holds on D′) and *Minimality*. Algorithm 1 calls the sampler's input a "sampling ratio n" (`D′ = Sampler(D, n)`), while §3.3 calls n the "preset sample size", a number of rows. Neither value is given.

**Code.** [sampler.py](distfd/sampler.py)
- [`systematic_indices`](distfd/sampler.py#L18) follows the paper exactly: `k = round(N/n)`, `r ∈ [1, k]` drawn from a seeded RNG, `arange(r-1, N, k)[:n]`.
- [`sample_size`](distfd/sampler.py#L14) follows §3.3: it turns a ratio into a row count n = max(`min_rows`, ⌈ratio·N⌉), capped at N. The defaults are 10% and at least 50 rows ([`Config`](distfd/distfd.py#L16)).
- [`sampling_validate`](distfd/sampler.py#L25) applies Theorem 1 on the sample, using a `PartitionCache` built on the sample. The driver builds the sample by filtering the distributed rows by index and collecting them ([distfd.py:49-54](distfd/distfd.py#L49-L54)).

**Why.**
- The paper gives no sample size or ratio. 10% gives 418 of Abalone's 4,177 rows, and the floor of 50 keeps a usable sample on small tables such as the 200-row test tables. The experiment also sweeps 5%, 10% and 25%.
- The sample is collected to the driver because the paper keeps D′ "only on the master node", and it is small.

**Student notes.**
- **Why sampling is safe.** Completeness is the important property, read in its contrapositive form: if X → A fails on D′, it fails on D, because the two violating rows are also in D. So sampling can only **reject**. It never accepts an FD without the global check, and it cannot remove a true FD. This is the core of the argument that DisTFD should score F1 = 1.
- The paper's *Minimality* property is true: if every proper subset fails on D′, it also fails on D. The code does not need it, because `Lattice.minimal()` checks minimality at the end anyway.
- Because k is rounded, the sample can have fewer than n rows. On Abalone, n = 418 gives k = 10 and 417 rows. The sampling sweep shows that the sample size changes cost only.
- On Abalone (seed 0), the sample has 417 rows. Sampling rejects all 72 level-1 candidates and 228 of 252 at level 2 ([stats](results/abalone/fds/abalone__DisTFD__seed0__stats.json)).

---

## 4. Pre_processing: skew and attribute order (Algorithm 2, Eq. 1)

**Paper.** For each attribute, Inc(A) = (frequency of its most common value) / n, computed **on the sample** to avoid repeated passes over the full data. Attributes with Inc > t are *skewed*. The non-skewed attributes are sorted by cardinality and the skewed ones are placed after them. The paper gives no value for t.

**Code.**
- [`skewness`](distfd/balance.py#L5) is Eq. 1.
- [distfd.py:56-65](distfd/distfd.py#L56-L65) computes skew and cardinality on the sample, applies `skew_t = 0.1`, and sorts by `(skewed, -cardinality, index)`: non-skewed first, then highest cardinality first.
- [distfd.py:42-47](distfd/distfd.py#L42-L47) also computes **exact** value frequencies on the full data with one `reduceByKey`.

**Why.**
- **t = 0.1:** this is the lowest skew level used in the paper's load-balancing experiment (Fig. 10).
- **Descending cardinality.** The paper says "SortByCardinality" but not in which direction. Descending matters because of how candidates are routed (section 7): each candidate is validated on the block of its *first* LHS attribute in this order. High-cardinality, non-skewed attributes make the best redistribution keys. They have many small groups that hash evenly across nodes. With this order, a skewed attribute is used as the key only when the whole LHS is skewed.
- **Exact frequencies on top of sample statistics** (a deviation, made on purpose):
  - *Constants:* the code finds ∅ → A from the full data. A column can be constant on the sample but not on D, and a constant found only on the sample would be a false FD.
  - *Algorithm 4:* the paper's line 1 reads key frequencies "in preprocessing", which means from the sample. Routing whole rows needs real key sizes, though, and one `reduceByKey` over the data is cheap compared with validation.

**Student note.** On Abalone the sample gives Sex (Inc 0.386) and Rings (0.170) as skewed. Every other attribute is at 0.067 or below. The resulting order is Whole_weight, Shucked_weight, Viscera_weight, Shell_weight, Length, Diameter, Height, then Rings and Sex.

---

## 5. Lattice, Lemmas 1–3 and two-way traversal (§3.4)

**Paper.** The candidates form the lattice of attribute sets (Fig. 3). Lemma 2: if Y ⊂ X and X ↛ Z, then Y ↛ Z (an invalid FD prunes its subsets). Lemma 3: if Y ⊂ X and Y → Z, then X → Z (a valid FD prunes its supersets). Following [39], DisTFD visits levels from both ends: level i, then level n + 1 − i, then i + 1, and so on.

**Code.** [lattice.py](distfd/lattice.py)
- [`level_order`](distfd/lattice.py#L5) returns 1, L, 2, L−1, … when two-way is on (L = `max_lhs`), and 1..L when it is off. Levels count |LHS|.
- [`Lattice.candidates`](distfd/lattice.py#L55) yields the candidates X → A with |X| = level that are not yet resolved. It skips any candidate that has a valid subset (Lemma 3, [`_valid_subset`](distfd/lattice.py#L34)). It marks a candidate invalid without testing if it has an invalid superset (Lemma 2, [`_invalid_superset`](distfd/lattice.py#L42)).
- [`record`](distfd/lattice.py#L69) stores the results of a level. [`minimal`](distfd/lattice.py#L74) returns the valid FDs that have no valid proper subset.
- Constant attributes give ∅ → A when the lattice is built and are removed from every LHS ([lattice.py:26-32](distfd/lattice.py#L26-L32)).

**Why.**
- **Constants up front.** The ∅ level is decided by the exact frequencies (section 4) at no cost. Any X → A with a constant B in X is equivalent to X∖{B} → A, so it is never minimal and need not be generated.
- **Checking the nearest processed level above is enough for Lemma 2.** `_invalid_superset` takes j, the lowest level above |Y| that has already been processed, and checks *every* superset of Y at level j (`combinations(extra, j - |Y|)`). This is enough because the top frontier descends one level at a time (L, L−1, …), so the processed levels above any level form one unbroken range j..L. Suppose X → A is invalid at a level above j, and pick Z with Y ⊂ Z ⊂ X and |Z| = j. Level j was processed after X → A was recorded as invalid. At that point Z → A had no valid subset, because it is invalid (Lemma 2), so it was not skipped. It was therefore either marked invalid by the same check or tested and recorded as invalid. The step from j down to |Y| across levels not yet visited is covered by checking all supersets at level j.
  On Abalone, level 4 is processed when the nearest processed level above it is 6 (level 5 is not yet visited). The check marks 237 level-4 candidates invalid from level 6 without testing them, and 316 are left to test. At level 5, 117 are marked invalid and 119 are tested. A variant that checks every processed level above gives the same candidates, the same prune counts and the same FDs at every level.
- **Why `minimal()` is needed.** Lemma 3 prunes upwards only. When levels are processed top-down, the top levels are reached before their subsets are known, so they produce valid but **non-minimal** FDs. On Abalone (seed 0), levels 8, 7 and 6 return 9, 72 and 120 valid FDs. Abalone's minimal FDs have |LHS| ≤ 6 and only 8 have |LHS| = 6, so 193 of these are non-minimal. Without the final filter they would be reported as false positives.

**Student notes.**
- **The paper swaps the two lemmas in its two-way description.** It says that if level i "is true, Lemma 2 is used to prune the functional dependencies greater than Level-i", and that if level j is false, "Lemma 3 is used to prune [those] smaller". It is the other way round: pruning supersets of a valid FD is Lemma 3, and pruning subsets of an invalid FD is Lemma 2. The paragraph just before uses them correctly. The code follows the lemmas as stated, not this paragraph.
- **Level numbering.** Fig. 3 counts ∅ as Level 1, so A → B joins levels 2 and 3. The worked example "Level-2: ∅ → A; Level-4: ABC → D" mixes labels: it names the edge by its lower node in one case and its upper node in the other. The code avoids this by using |LHS| as the level.
- **Lemma 1 counts candidates, not minimal FDs.** n·2ⁿ⁻¹ − n is the number of non-trivial candidates X → A with a non-empty LHS (Σ C(n,k)·(n−k) for k ≥ 1). The proof itself says "number of candidate functional dependencies". For Abalone (n = 9) that is 2,295 candidates, against 137 minimal FDs.
- **Where two-way is used.** The paper uses the alternating order "in the sampling validation process". The code uses it for global validation too, which shapes how the data is redistributed (section 7). Accuracy does not depend on the order: `test_flag_combinations` checks both orders.
- **The LHS cap.** `max_lhs` limits the lattice. It does not appear in the paper. The Abalone experiment runs uncapped (L = 8), because 119 of its 137 FDs have |LHS| ≥ 4.

---

## 6. Prefix-tree partition cache (Algorithm 3, §3.5.1)

**Paper.** Computed partitions are stored in a prefix tree keyed by attribute lists (Fig. 4). To build Π_X, Algorithm 3:
1. finds every cached subset of X;
2. starts from the smallest one;
3. repeatedly adds the cached subset that contributes the most new attributes, until X is covered (Rule 1: few intersections);
4. intersects them in ascending size (Rule 2: small operands first).

Entries that "have not been accessed for a long time" are cleared periodically.

**Code.** [cache.py](distfd/cache.py)
- **The tree:** [`_Node`](distfd/cache.py#L7) and [`_store`](distfd/cache.py#L101) store each partition at the path of its sorted attributes.
- **Lookup:** [`get`](distfd/cache.py#L39) handles ∅, single columns and the cache-off mode. Otherwise it tries an exact hit and falls back to `_retrieve`.
- **Finding subsets:** [`_subsets`](distfd/cache.py#L68) walks the tree, following only edges whose attribute is in X. That collects every cached proper subset without listing all 2^|X| subsets.
- **Building Π_X:** [`_retrieve`](distfd/cache.py#L84) is Algorithm 3. It starts from the subset with the smallest ‖Π̂‖, then greedily picks the subset adding the most uncovered attributes (ties go to the smaller one), then sorts the chosen subsets by size and intersects them.
- **Eviction:** an LRU (`OrderedDict`) with `capacity = 4096`. Single-attribute partitions are kept outside the LRU and never evicted.

**Why.**
- Sorted attribute tuples give each set a single path in the tree. That plays the role of the paper's "attribute order in the relational schema".
- "Smallest" means ‖Π̂‖ (rows in non-singleton classes). Fig. 4 labels nodes with a partition size and TANE measures it this way. A smaller ‖Π̂‖ means fewer rows can still collide.
- The paper gives no eviction period or memory budget, so the code uses LRU. It is deterministic, matches "not accessed for a long time", and needs no timer. Single columns are the base of every intersection, so they are pinned.

**Student notes.**
- **Text and figure disagree in the example.** The worked example lists Π_AD in Φ, but Fig. 4 stores CD. The result Π_ABCDE = Π_CE ∩ Π_AB ∩ Π_D follows the same rules either way.
- **Where the cache lives: a deviation.** In the paper, the PartitionMgr keeps one shared cache, and Algorithm 5 (lines 15–16) stores each merged global Π_X in it. The code has two kinds of cache:
  1. The **sample cache** on the driver ([distfd.py:74](distfd/distfd.py#L74)). It lasts for the whole run, so partitions computed at one level are reused at later levels.
  2. A **per-task cache** inside each Spark task ([spark_validate.py:45,52](distfd/spark_validate.py#L45)). It is created fresh on every `global_validate` call, so it is shared by all candidates of one level on one block but does **not** carry over between levels. Merged global partitions are never cached.

  The reason is that Spark tasks are stateless. Keeping per-block caches alive between jobs would need a long-lived store per executor, which goes beyond a re-implementation of the paper's text. This costs speed, not accuracy: every partition is still computed exactly. The paper's cache experiment (Fig. 9) is out of scope here.

---

## 7. Redistribution and GlobalValidate (Algorithm 5, Eq. 2, Fig. 6)

**Paper.** Rows are hash-partitioned on a "public attribute", so that rows with the same value end up on the same node. Each node computes its local equivalence classes. Local classes with the same value are merged (Fig. 6), and |Π_X| = Σ_j |Π_Xj| (Eq. 2). Algorithm 5 checks the FD on each node first (lines 4–6), then merges and checks again (lines 12–14), and finally caches Π_X.

**Code.** [spark_validate.py](distfd/spark_validate.py), with routing in [distfd.py:67-91](distfd/distfd.py#L67-L91).
- **Redistribution, done once.** [`build_blocks`](distfd/spark_validate.py#L19) shuffles the rows once per attribute: `rows.map(router).partitionBy(n_nodes)`. Each resulting partition is packed into one numpy block `(attribute, rows, split-key mask)`. All blocks are unioned into one persisted RDD and reused at every level.
- **Routing candidates.** Each candidate X → A that survives sampling gets a *home* attribute Z, the first attribute of X in the order from section 4 ([distfd.py:85](distfd/distfd.py#L85)). The candidate lists are broadcast, and each block checks only the candidates whose home is its attribute ([`global_validate`](distfd/spark_validate.py#L68)).
- **The local check.** [`_check_block`](distfd/spark_validate.py#L39) builds a `PartitionCache` on the block's rows and tests `holds(Π_X, Π_XA)`.
- **Split keys and Eq. 2.** Rows whose Z value was split across nodes by Algorithm 4 are checked separately. If the local check fails, the FD fails. Otherwise the block emits one record per local class, `(("merge", cid, X-value), A-value)`. `reduceByKey` takes the union of the A-values of classes with the same X value across nodes. The FD fails if any merged class has more than one A value.

**Why.**
- **Why routing on Z ∈ X is correct.** This is the key idea of the method, and the paper states it only briefly. If Z ∈ X, every class of Π_X has a single Z value. After redistribution on Z, each class therefore sits entirely on one node. A local |Π_X| = |Π_XA| check is then a global check, and Eq. 2 is just a sum of local counts. This is how DisTFD handles Example 1, where A → B holds on R1 and on R2 but not on R. The rows with A = 1 are brought together before checking ([`test_paper_example_1`](tests/test_distfd.py#L57)).
- **Two-way traversal and redistribution: one design decision.** Each candidate needs the data redistributed on one attribute of its LHS. If global validation runs in two-way level order, every level has candidates for every attribute, so each level needs all m redistributions. There are three ways to arrange this (m = 9 attributes, L = 8 levels on Abalone):

  | Arrangement | Shuffles | Copies of the data in memory | Global results prune between levels |
  |---|---|---|---|
  | Paper: two-way on the sample only, then one pass over the attributes (Algorithm 1) | m = 9 | 1 at a time | no |
  | This code: redistribute on every attribute once, keep all blocks, validate level by level | m = 9 | m = 9 | yes |
  | Two-way for global validation, reshuffling at every level | m·L = 72 | 1 at a time | yes |

  The code takes the second row. Global validity results then feed Lemma 2 and Lemma 3 pruning between levels, such as the 237 and 117 candidates pruned at levels 4 and 5, at the cost of keeping 9 redistributed copies of the table. In all three arrangements each candidate is validated once, on its home attribute.
- **Why the merge uses X values instead of summing class counts.** Two local classes belong to the same global class exactly when they share an X value. Grouping by `(candidate, X-value)` and collecting the A-values merges the classes (Fig. 6) and checks the FD (Theorem 1) in one shuffle. Counting |Π_X| and |Π_XA| separately would need two. The local check runs first because a local failure already proves a global one. This is Algorithm 5's line 5.

**Student notes.**
- The paper's Algorithm 5 line 15 builds Π_X = Π_X/Z ∩ Π_Z from a cached Π_Z. The code builds Π_X locally from single columns through the task cache (section 6), so this split between cached and remaining attributes is not needed.

---

## 8. Greedy load balancing (Algorithm 4, Fig. 5)

**Paper.** When the redistribution key is a skewed attribute:
1. Avg = total rows / m nodes.
2. Keys larger than Avg are cut into Avg-sized pieces, each sent to a node with load 0.
3. The remaining keys fill nodes up to Avg.

The splitting loop ends with `else break`.

**Code.** [balance.py](distfd/balance.py)
- [`assignment`](distfd/balance.py#L10) sets Avg = ⌈Σ/m⌉. It walks keys from the largest down and cuts each key larger than Avg into Avg-sized chunks, each sent to the **least-loaded** node. Then it fills each node in turn with the remaining keys (largest first) while the load stays at or below Avg. Anything still unplaced goes to the least-loaded node.
- [`Router`](distfd/balance.py#L47) decides where each row goes:
  - keys with no plan go to `hash(key) % n`;
  - keys with a single chunk go to that chunk's node;
  - rows of a split key are spread over its chunks in proportion to chunk size, using a multiplicative hash of the row index (Knuth's 2654435761).
- [distfd.py:67-70](distfd/distfd.py#L67-L70) applies `assignment` only to skewed attributes, using the exact full-data frequencies.

**Why, step by step against the paper.**
- **Largest first.** Step 1 says to sort keys "from small to large", but the loop's `else break` stops at the first key not larger than Avg. That only works if keys are sorted from large to small, so the code sorts descending.
- **Least-loaded instead of "load 0".** With more heavy chunks than empty nodes, there would be no node with load 0. Least-loaded is the natural generalisation and gives the same result when empty nodes exist.
- **Leftover keys.** The paper does not say what happens to a key that fits on no node. The code sends it to the least-loaded node.
- **Spreading a split key's rows.** The rows of one split key need a deterministic, roughly even spread that does not depend on Spark's partition order. A hash of the row id gives that.
- **Check against the paper.** Running `assignment` on Fig. 5's numbers ({K1: 865, K2: 325, K3: 175, K4: 135}, 3 nodes) gives exactly the figure. Avg is 500, K1 is split 500 + 365, one node gets K1:365 + K4:135, another gets K2 + K3, and every node ends at 500.

**Student notes.**
- **Splitting a key breaks the "one class on one node" guarantee.** That guarantee is what made the local check enough in section 7. This is why the split-key mask and the Eq. 2 merge exist. The paper's Fig. 6 shows the merge but never says it is needed *because* of the split.
- **Abalone.** Sex is skewed, and all 3 of its keys (M 1,528, I 1,342, F 1,307 rows) are larger than Avg = 1,045, so all are split. Final loads are 1,045 / 1,045 / 1,045 / 1,042. Rings is skewed but no key exceeds Avg, so nothing is split. Because skewed attributes come last in the order, Sex is the home attribute only for candidates whose LHS is exactly {Sex}. The merge path therefore runs for those few candidates only. [`test_balancing_splits_keys_and_stays_correct`](tests/test_distfd.py#L94) covers the path on a table built for it.

---

## 9. Ground truth (§4.3)

**Paper.** Cet (centralized) and HFDD are compared with DisTFD, using TANE as the truth. The paper gives no implementation details.

**Code.** [`tane`](distfd/truth.py#L5) runs Desbordante's exact TANE (error = 0) with the same LHS cap as DisTFD. It also filters minimal FDs as a safeguard. The experiment compares DisTFD with this truth only; the Cet and HFDD baselines are out of scope and are not implemented.

**Why.** The truth comes from an **independent** C++ implementation, so a bug in our code would show up as a disagreement instead of being shared by the truth. The tests also check Desbordante itself against a brute force (section 11).

**Student note.** DisTFD is exact, so it should agree with TANE. On Abalone it does: F1 = 1.0 in all 3 runs. The paper reports F1 0.9926 ([report_abalone.ipynb](experiments/report_abalone.ipynb)).

---

## 10. Spark setup (§4.1)

**Paper.** 8 Xeon servers with Spark and HDFS. The node count for the accuracy experiment is not given.

**Code.** [`get_spark`](distfd/spark.py#L9) starts `local[4]`: 4 cores in one JVM stand in for 4 nodes. The UI is disabled, workers are reused, and the driver has 4 GB of memory.

**Why.** Accuracy does not depend on the hardware. What matters is that redistribution and the merge across nodes really run through Spark shuffles, and they do in local mode. `test_partition_count_does_not_change_result` also checks 1, 2, 4 and 8 partitions. Timings from one laptop are not comparable with the paper's cluster and are not used as a scalability result.

---

## 11. How correctness is checked

| Test ([tests/test_distfd.py](tests/test_distfd.py)) | What it guards |
|---|---|
| `test_level_order` | Two-way order 1, L, 2, L−1, … |
| `test_assignment_splits_heavy_keys` | Algorithm 4 splits a key larger than Avg, and no rows are lost |
| `test_paper_example_1` | Example 1: locally valid but globally invalid A → B is rejected |
| `test_matches_brute_force` | DisTFD equals a brute-force check of the FD definition on 3 random tables |
| `test_partition_count_does_not_change_result` | Same FDs with 1, 2, 4 and 8 nodes |
| `test_flag_combinations` | All 16 on/off combinations of sampling, cache, balancing and two-way give the same FDs |
| `test_balancing_splits_keys_and_stays_correct` | The split-key and Eq. 2 merge path is exact |
| `test_desbordante_matches_brute_force` | The ground truth itself is right |

The brute force works directly from Definition 1 (a dictionary from X values to A values) and does not use partitions. It is therefore independent of the code it checks.

---

## 12. Summary of choices where the paper is silent or inconsistent

| Topic | Paper | Code | Reason |
|---|---|---|---|
| Value equality, nulls | not stated | strings, null = null | exact, same rule as TANE |
| Sample size | "sampling ratio n" (Algorithm 1), "sample size n" (§3.3); no value | ratio 10% → n = 418 rows → k = 10 | §3.3 needs a row count; ratio sweep reported |
| Skew threshold t | not stated | 0.1 | lowest level in Fig. 10 |
| Sort direction | "SortByCardinality" | descending, skewed last | best redistribution keys (section 7) |
| Constants, Algorithm 4 frequencies | from the sample | exact, one `reduceByKey` | sample-only constants could be false; routing needs real sizes |
| Level numbering | ∅ = Level 1, mixed in the example | level = \|LHS\|; ∅ through constants | unambiguous |
| Two-way lemma roles | Lemma 2 and 3 swapped in §3.4 | lemmas as stated | the swapped version would be wrong |
| Minimality after top-down | not stated | final `minimal()` filter | top levels give non-minimal FDs |
| LHS cap | not stated | none | 9 columns keep the full lattice small; Abalone's FDs go to \|LHS\| = 6 |
| Cache eviction | "periodically" | LRU, 4,096 entries, single columns pinned | deterministic |
| Cache scope | one shared PartitionMgr | driver cache (whole run) plus per-task cache (one level) | Spark tasks are stateless; costs speed, not accuracy |
| Two-way scope | sample validation only; global validation is one pass over attributes | sample and global validation; blocks for all attributes kept | global results prune between levels; costs m copies of the table |
| Eq. 2 merge | sum class counts | group by X value, collect A values | one shuffle |
| Algorithm 4 order, target node | sort small→large; node with load 0 | large→small; least-loaded | needed for `else break`; works with more chunks than nodes |
| Cet, HFDD baselines | compared in Table 3 | out of scope | the experiment covers DisTFD against TANE only |

## What a student should take away

1. **DisTFD is exact by construction.**
   - Theorem 1 is an exact test.
   - Sampling can only reject candidates, and a failure on a subset of rows is a failure on all rows.
   - Redistributing on an attribute of the LHS puts every equivalence class in one place, and the Eq. 2 merge repairs the classes that load balancing split.
   - The lattice lemmas prune only what is logically implied.

   So a correct implementation must match TANE (F1 = 1), and ours does on Abalone.
2. **The four optimizations change cost, never the answer.** Sampling, the cache, two-way traversal and balancing are each a speed-up with a correctness argument. The flag-combination test checks exactly this.
3. **Read pseudocode critically.** Several lines in the paper (the lemma roles, the sort direction in Algorithm 4, the level labels, the claim in Lemma 1, the two meanings of n for the sampler) contradict the surrounding text or the definitions. Re-implementing from the definitions and lemmas, and then testing against a brute force and an independent TANE, is what made these slips visible.
