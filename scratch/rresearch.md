# Research Briefing for the Lume White Paper
## A Hybrid Lexical–Neural Document-Memory Architecture — Prior Art, Positioning, and Editorial Corrections

---

## TL;DR
- **The Lume thesis is well-grounded in 2024–2026 literature**: Takeshita et al. (EMNLP 2025, arXiv:2508.17744) show that ~50% of embedding dimensions can be deleted with <10% performance loss and that 430 of 1024 E5-large dimensions are *actively degrading*; Morris et al. (FAIR/DeepMind/Cornell/NVIDIA, arXiv:2505.24832, May 2025) put model capacity at ~3.6 bits per parameter; Templeton et al. (Anthropic, *Scaling Monosemanticity*, May 2024) extract on the order of 34 million monosemantic features from Claude 3 Sonnet's residual stream. Together these support Lume's claim that dense embeddings are *information-light* relative to their dimensionality and should serve as a re-ranker rather than a primary filter.
- **One factual correction is required, and one user phrasing turns out to be correct**: (1) the `kordless/lume` repository was *not publicly accessible* as of 25 May 2026 and the white paper must either make the repo public or be written as a self-contained primary artefact; (2) the user's reference to **"Gemma 4 E2B" is correct** — per Google's official `ai.google.dev/gemma/docs/core` ("Gemma 4 model overview"), Google released Gemma 4 on 2 April 2026 under Apache 2.0, with four sizes E2B, E4B, 31B, and 26B A4B. (Earlier drafts of this briefing erroneously substituted Gemma 3n; that substitution has been retracted and Gemma 4 E2B is correct.)
- **The strongest scholarly anchors are**: Takeshita et al. 2025 (embedding redundancy), Templeton et al. 2024 (Scaling Monosemanticity), Turner et al. 2023 (Activation Addition / ActAdd), Zou et al. 2023 (Representation Engineering), Balsam, Deng, Nguyen, Gorton, Shihipar, Ho, and McGrath, *Goodfire Ember: Scaling Interpretability for Frontier Model Alignment* (Goodfire Research, 23 Dec 2024), Khattab & Zaharia 2020 / Santhanam et al. 2022 (ColBERT/ColBERTv2), Cormack, Clarke & Büttcher 2009 (RRF), Huh, Cheung, Wang & Isola 2024 (Platonic Representation), and Hatcher / Gospodnetić / McCandless (*Lucene in Action*, 2nd ed., Manning 2010). The Lume project's "ratio is backwards" argument has genuine — not retrofitted — support in the recent literature.

---

## Key Findings

1. **Embeddings are mostly empty.** Direct experimental evidence from Mannheim (Takeshita et al., arXiv:2508.17744, EMNLP 2025 Main, pp. 27705–27726) confirms that the *bulk* of dimensions in modern text embedders are either redundant or actively harmful. This is the closest published support for the user's recollected "few hundred concepts" claim — though no paper uses that exact phrasing.

2. **Steering vectors are a real, well-developed field, but Lume's flavour is unusual.** The dominant academic paradigm (Turner et al. 2023; Zou et al. 2023; Rimsky et al. 2024; Anthropic's "Golden Gate Claude" via Templeton et al. 2024; Goodfire Ember, Balsam et al. 23 Dec 2024) intervenes in the **mid-layer residual stream**. Lume's **FST-driven logit-bias** approach is closer to Dathathri et al.'s 2020 "Plug-and-Play Language Model" and to constrained-decoding work (Outlines, Guidance, XGrammar, llama.cpp grammar) than to mainstream activation steering. The white paper should position Lume as a *symbolic* steering layer that complements rather than competes with feature-level steering.

3. **Hybrid search consensus has moved in Lume's direction.** Weaviate, Elasticsearch, OpenSearch, and Vespa now all ship BM25+vector hybrid retrieval as a first-class primitive, almost universally fused with Reciprocal Rank Fusion (RRF). The literature confirms BM25 and cosine live on incompatible scales; Lume's MinMax-normalised multiplicative kernel is a defensible alternative to RRF, particularly when one signal should *modulate* rather than *vote alongside* the other.

4. **Late interaction (ColBERT) is the missing comparand.** The white paper currently positions itself only against "vector first" and "BM25 first" — but ColBERT / ColBERTv2 / PLAID (Khattab, Santhanam, Zaharia et al.) constitute a third paradigm that lives between the two and now dominates BEIR benchmarks. Lume should acknowledge it.

5. **Gemma 4 E2B is the correct edge target.** Per Google's official model overview at `ai.google.dev/gemma/docs/core`, "*Gemma 4 models are available in 4 parameter sizes: E2B, E4B, 31B and 26B A4B.*" The Wikipedia entry for the Gemma language model confirms: "*On April 2, 2026, Google released Gemma 4 under the free and open-source Apache 2.0 license.*" The E-prefix scheme (Effective parameters) carries forward from Gemma 3n — practitioners deploying Lume today should target Gemma 4 E2B; older Lume documentation referring to Gemma 3n E2B remains valid for backward deployments (~1.91 B effective / ~2 GB VRAM per Google's `ai.google.dev/gemma/docs/gemma-3n`, last updated 1 November 2025).

6. **Deliberate overfit-as-feature has weak academic precedent but strong industrial precedent.** Character-level RNN work (Karpathy 2015), small-corpus stylistic LMs, and the broader "memorisation is the point" framing in narrow-domain retrieval do not have a canonical paper, but the Morris et al. (2505.24832) capacity result — 3.6 bits/parameter — gives the white paper a quantitative way to justify *why* a 26 M-parameter model trained narrowly on Faraday is the right size: it has roughly 11–12 MB of memorisable capacity, well-matched to a single-volume corpus.

7. **The "Coherence Laundering Trap" is a real failure mode without a canonical name.** The phenomenon Lume describes — a fluent decoder smoothing a broken upstream model's BPE-noise hallucinations into authoritative prose — is well-documented under other names: "generation drift" in RAG-evaluation literature (RAGAS, DeepEval, TruLens, Arize Phoenix), and the related "glitch token" / under-trained-token research (Rumbelow & Watkins 2023; Land & Bartolo, "BPE Gets Picky," EMNLP 2024). The white paper is justified in coining a new term, but should cite this prior work explicitly.

8. **The Ferricula / Lume programme is coherent.** Ferricula (DeepBlue Dynamics' published memory engine, ferricula.com) already commits to: Roaring Bitmap-based filtering, *exact* cosine over a bitmap-pre-filtered candidate set (no ANN), per-agent rotational encryption, and vec2text inversion as a fidelity check. Lume sits naturally next to it as the document-grounded sibling of the agent-memory engine.

---

## Details — Organised by the Eight Research Areas

### 1. Embedding-capacity literature (highest priority)

The state of the field as of late 2025 / early 2026 strongly supports the claim that **modern dense text embeddings are heavily over-parameterised relative to the semantic information they carry**.

**Strongest reference — Takeshita, Takeshita, Ruffinelli, Ponzetto (Mannheim), "Randomly Removing 50% of Dimensions in Text Embeddings has Minimal Impact on Retrieval and Classification Tasks," arXiv:2508.17744, EMNLP 2025 Main, pp. 27705–27726.** Verbatim: "*We consistently observe across 6 state-of-the-art text encoders and 26 downstream tasks, that randomly removing up to 50% of embedding dimensions results in only a minor drop in performance, less than 10%, in retrieval and classification tasks.*" The same paper identifies **430 actively degrading dimensions out of 1024 in E5-large** — i.e. ~42% of dimensions hurt performance when included. This is the most direct empirical support for the Lume thesis that the bulk of embedding dimensions are noise.

**Supporting reference — Tsukagoshi & Sasano, "Redundancy, Isotropy, and Intrinsic Dimensionality of Prompt-based Text Embeddings," arXiv:2506.01435 (2025).** Shows that even a naive truncation to the first 25% of dimensions yields minimal degradation; for classification and clustering "*even when embeddings are reduced to less than 0.5% of the original dimensionality the performance degradation is very small.*"

**Supporting reference — Kataiwa, Onoda & Yokoi, "Measuring Intrinsic Dimension of Token Embeddings," arXiv:2503.02142 (4 March 2025).** Estimates intrinsic dimension of Word2Vec/GloVe/Pythia token embeddings at ~10–30, against extrinsic dimension of 300 (3–10% utilisation).

**Mechanistic interpretability angle — Templeton et al., "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet," Anthropic, transformer-circuits.pub, May 2024.** Trained sparse autoencoders extracting features at 1M, 4M, and 34M scales from a residual stream of ~12 K dimensions — feature ratios of 83×, 333×, 2833×. This is the canonical "concepts per dimension" reference. Templeton et al. (and earlier Bricken et al., "Towards Monosemanticity," 2023) found ~15,000 interpretable features from a 512-neuron MLP layer of GPT-2 Small (~29× feature/neuron ratio). The *count* of meaningful concepts in a representation is much larger than the dimensionality, but **the useful concepts are sparse** — which is precisely the architectural justification for late, optional re-ranking.

**Capacity bound — Morris, Sitawarin, Guo, Kokhlikyan, Suh, Rush, Chaudhuri, Mahloujifar, "How Much Do Language Models Memorize?" arXiv:2505.24832 (FAIR at Meta, Google DeepMind, Cornell, NVIDIA; 30 May 2025).** Verbatim from the abstract: "*our measurements estimate that GPT-style models have a capacity of approximately 3.6 bits per parameter.*" For a 26 M-parameter Lume oracle, this implies roughly 11–12 MB of total memorisable content — a quantitative justification for the model size being matched to a single-volume corpus.

**Convergence framing — Huh, Cheung, Wang, Isola, "Position: The Platonic Representation Hypothesis," ICML 2024, arXiv:2405.07987.** Verbatim: "*We hypothesize that this convergence is driving toward a shared statistical model of reality, akin to Plato's concept of an ideal reality.*" Useful for the Victorian-scientific register: representations across vision and language increasingly agree on a small shared substrate of concepts.

**Where Lume aligns / extends / contradicts**: Lume aligns with Takeshita et al. and Tsukagoshi & Sasano; *extends* by using the redundancy claim as a structural argument for re-ranking rather than as a compression engineering argument; does *not* contradict the Platonic Representation Hypothesis, but qualifies it — convergence to a shared substrate explains *why* a small late-stage re-rank suffices.

### 2. Activation / logit steering and concept vectors

**Canonical lineage**:
- Subramani, Suresh, Peters, "Extracting Latent Steering Vectors from Pretrained Language Models" (ACL Findings 2022) — earliest extraction of latent steering vectors.
- Turner, Thiergart, Leech, MacDiarmid, Udell, Mini, Vazquez, "Activation Addition: Steering Language Models Without Optimization" (arXiv:2308.10248, 2023). The ActAdd method — add a contrastive activation vector at a layer.
- Li et al., "Inference-Time Intervention" (ITI, NeurIPS 2023).
- Zou, Phan, Chen, Campbell, Guo, Ren, Pan et al., "Representation Engineering: A Top-Down Approach to AI Transparency" (arXiv:2310.01405, 2023). Umbrella framework.
- Rimsky et al., "Steering Llama 2 via Contrastive Activation Addition" (CAA, 2024).
- Hendel, Geva, Globerson, "In-Context Learning Creates Task Vectors" (EMNLP Findings 2023).
- Templeton et al. ("Golden Gate Claude" demo, May 2024) — SAE-feature-based steering at production scale.
- **Goodfire Ember (Balsam, Deng, Nguyen, Gorton, Shihipar, Ho, McGrath), "Goodfire Ember: Scaling Interpretability for Frontier Model Alignment," Goodfire Research, 23 December 2024.** First hosted mechanistic-interpretability API; open-sourced SAEs for Llama-3.1-8B and Llama-3.3-70B on HuggingFace. Demonstrates conditional steering (e.g. refusal-feature activation on jailbreak detection), auto-steering from natural-language descriptions, dynamic prompt injection on feature activation, and a 75 %-accurate financial-sentiment classifier built from just three features.

**Subtractive / negative steering**: Arditi et al. ("Refusal in LLMs is mediated by a single direction," 2024) and Wang & Shu ("Trojan Activation Attack," 2024) show that *subtracting* a learned direction reliably removes a concept; this is what the white paper needs to cite for the "steer away from undesirable directions" use case.

**Fine-grained lexical substitution (the "damn for fuck" use case)**: There is **no** canonical paper that does single-word lexical substitution via mid-layer activation steering — the closest prior art is (a) **constrained decoding with banned-token regexes** in Outlines / Guidance / XGrammar / llama.cpp grammars, (b) **logit-bias** parameters in the OpenAI Chat Completions API (a direct logit mask), and (c) the Plug-and-Play Language Model (Dathathri et al., ICLR 2020), which uses a classifier-gradient to nudge activations away from undesired attributes. Lume's FST-driven logit-bias approach is in this family — it is *not* activation steering, and the white paper should be precise about this distinction.

**Logit-bias vs mid-layer activation steering — tradeoff table for the white paper**:
- *Logit-bias (Lume)*: deterministic, interpretable, single-token granularity, zero additional compute per token, but cannot affect concepts that span multiple tokens or that are encoded sub-symbolically.
- *Activation steering (Turner / Zou / Anthropic / Goodfire)*: concept-level granularity, generalises across surface forms, but requires labelled contrast pairs, is brittle out-of-distribution (see Tan et al., "Analysing the Generalisation and Reliability of Steering Vectors," arXiv:2407.12404), and has side-effects (a "happier" model refuses fewer harmful requests — Zou et al.).

The Lume position should be: **logit-bias and activation steering are complementary, not competing. Lume chooses logit-bias because it is auditable, deterministic, and FST-symbolic — properties that matter for grounded, customer-facing knowledge systems.**

### 3. Hybrid search / lexical-primary architectures

**Vendor landscape (verified)**:
- **Weaviate** ships hybrid BM25+vector with two fusion modes: `rankedFusion` (rank-based, default ≤ v1.23) and `relativeScoreFusion` (min-max-normalised). Per the Weaviate blog "Unlocking the Power of Hybrid Search": "*the relativeScoreFusion algorithm was added in Weaviate version 1.20. Since v1.24 the default method is Relative Score Fusion.*" Pre-filter selectors are implemented via roaring-bitmap "allow lists" (see Weaviate Filtering docs and `indexFilterable` / `indexSearchable` / `indexRangeFilters`). The user's claim that "Weaviate has ANN vector search with prefilter selectors" is **correct**; however, GitHub issue weaviate/weaviate#7681 (2025) documents that hybrid filters are not always *strictly* pre-applied — non-matching documents can leak through if they score high enough on one leg. Lume's claim of *strict* lexical pre-filtering is therefore a defensible differentiator.
- **Elasticsearch**: per Elastic's 8.14 release notes — "*A retriever is an abstraction that was added to the Search API in 8.14.0 and was made generally available in 8.16.0.*" The `rrf` retriever fuses sub-retrievers using Reciprocal Rank Fusion.
- **OpenSearch**: per the official OpenSearch blog "Introducing reciprocal rank fusion for hybrid search" — "*OpenSearch 2.19 introduces reciprocal rank fusion (RRF), a new feature in the Neural Search plugin that enhances hybrid search.*" The Score ranker processor was introduced in 2.19 (April 2025).
- **Vespa** offers BM25 and ANN as composable rank phases.

**Reciprocal Rank Fusion — Cormack, Clarke, Büttcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods," SIGIR 2009.** Formula: `RRF(d) = Σ 1 / (k + rank_i(d))`, k typically 60. Used as default fusion in Weaviate, Elasticsearch, OpenSearch.

**ColBERT — Khattab & Zaharia, SIGIR 2020 (arXiv:2004.12832); Santhanam et al., ColBERTv2, NAACL 2022 (arXiv:2112.01488); PLAID, CIKM 2022.** Late-interaction multi-vector retrieval — independently encodes query and document tokens, then scores via per-token MaxSim. ColBERTv2 with residual compression is now the dominant single-stage neural IR baseline; PLAID is the production engine. ECIR 2026 hosted the first dedicated Late Interaction Workshop. **This paradigm should be acknowledged as the third axis** the Lume white paper is implicitly arguing against — and Lume's response is that token-level neural retrieval still requires neural inference at query time, whereas BM25-first hybrid pushes neural cost to an optional re-rank.

**Erik Hatcher**: verified. Co-author of *Lucene in Action* (Manning, 1st ed. 2004; 2nd ed. 2010 with McCandless and Gospodnetić). Apache Software Foundation member; committer on Ant, Lucene, Tapestry. Co-founded Lucid Imagination (now Lucidworks); Senior Solutions Architect there at the time of writing the second edition. The "Hatcherik" naming is a legitimate homage. Note that he is *Hatcher*, not "Hatcherik" — the white paper should spell out on first use that "Hatcherik" is a coined portmanteau / honorific.

**Tantivy & ParadeDB**: Tantivy (Paul Masurel, Quickwit) is the Rust analog of Lucene; uses FSTs for the term dictionary, BM25 scoring, roaring-bitmap-style compression, and is the foundation for Quickwit (distributed log search) and ParadeDB (Postgres-embedded BM25). Lume's architectural choices (FSTs, roaring bitmaps, BM25) directly mirror Tantivy/Lucene primitives and should cite both.

**Practitioner argument for BM25-first**: Industry blog literature (e.g. Singh, "Hybrid Search Done Right," Medium 2026; the OpenSearch RRF blog 2025; Rana, "12 Hybrid Weaviate Patterns That Outrank BM25") consistently argues that vector-only retrieval misses exact-token signals like IPs, error codes, SKUs, proper nouns — i.e. *precisely the cases that matter in technical document collections*. This is the customer-relevant version of the "ratio is backwards" argument.

### 4. Small-model hallucination reduction and grounding

**Edge-model state of the world (corrected after enrichment)**: Per Wikipedia's *Gemma language model* entry: "*On April 2, 2026, Google released Gemma 4 under the free and open-source Apache 2.0 license.*" Per Google's official `ai.google.dev/gemma/docs/core` ("Gemma 4 model overview"): "*Gemma 4 models are available in 4 parameter sizes: E2B, E4B, 31B and 26B A4B.*" The user's reference to **"Gemma 4 E2B" is therefore correct** — the white paper should use this designation. For background continuity, the previous family (Gemma 3n) is documented at `ai.google.dev/gemma/docs/gemma-3n` (last updated 1 November 2025): the 3n E2B variant is a MatFormer with raw parameter count of 5 B (per Google's developer blog "Introducing Gemma 3n: The Developer Guide," June 2025) or 6 B (per the HuggingFace card `google/gemma-3n-E2B`, which states: "*While the raw parameter count of this model is 6B*"), running at "*just under 2 billion (1.91B) parameters*" effective via Per-Layer Embeddings, 2 GB VRAM, 32 K context. The E-prefix scheme (Effective parameters) and MatFormer architecture carry forward into Gemma 4 E2B.

**Grounded-RAG state of the art (late 2025 / early 2026)**:
- Faithfulness / groundedness metrics standardised across RAGAS, DeepEval, TruLens, Arize Phoenix, LangSmith, Braintrust, Maxim, Confident AI. Decomposition into atomic claims followed by NLI-style entailment checking is now the dominant pattern.
- Five standard RAG metrics: answer relevancy, faithfulness, contextual relevancy, contextual recall, contextual precision (DeepEval / Confident AI canonical list).
- "Generation drift" is the named failure mode where retrieval is correct but the model rephrases in a way that changes meaning — this is *precisely* the failure Lume names "Coherence Laundering."

**Retrieval-augmented decoding / copy mechanisms**: kNN-LM (Khandelwal et al., ICLR 2020); RETRO (Borgeaud et al., DeepMind 2021); In-Context RALM (Ram et al., 2023). The Lume architecture's "lexical at both gates" is a stronger version of these — the same lexical index that supplies retrieval candidates also gates the output as a provenance check.

**"Small model + good retrieval beats big model alone for domain-specific tasks"**: This is supported by Lewis et al. (RAG, NeurIPS 2020), Mallen et al. ("When Not to Trust Language Models," ACL 2023), and the industrial Phi-3, Mistral-Small, Gemma 3n / Gemma 4 E2B positioning. The argument generalises directly to Lume.

### 5. The overfit-as-feature paradigm

This is the *weakest* area in terms of canonical academic citations. The closest references:
- Karpathy's "The Unreasonable Effectiveness of Recurrent Neural Networks" (2015) — char-RNN trained on single-author corpora (Shakespeare, Linux kernel) demonstrated stylistic memorisation as a desirable property.
- Carlini et al., "Quantifying Memorization Across Neural Language Models" (ICLR 2023) — quantifies that memorisation grows with model capacity, training repetitions, and prompt-context length. *Most of this literature treats memorisation as a privacy risk; Lume re-frames it as a fidelity feature.*
- Morris et al. (2505.24832, 2025) capacity result, above — directly relevant to sizing.
- Allen-Zhu & Li, "Physics of Language Models, Part 3.3: Knowledge Capacity Scaling Laws" (2024) — estimated ~2 bits/parameter via quantisation.

**Recommended white-paper framing**: rather than claim novelty, cite Karpathy and Carlini and *invert the sign* — what is a privacy bug at frontier scale is a fidelity feature at narrow scale. This is the most honest position.

### 6. Symbolic-neural hybrid and FST / automata

**FST / automata for retrieval**:
- Lucene's FST-based term dictionary (since ~2010) — McCandless's key contribution; the basis Tantivy adopted.
- Levenshtein automata for fuzzy matching — Schulz & Mihov 2002 algorithm, used in Lucene since 2011, exposed in Tantivy as `LevenshteinAutomatonBuilder`.
- Aho–Corasick (1975) for multi-pattern dictionary matching; spaCy's `PhraseMatcher`; Intel Hyperscan.
- FlashText (Singh, 2017) — keyword-dictionary library widely used in NER preprocessing.

**Constrained decoding via FSTs / grammars (the symbolic-at-generation analog)**:
- **Outlines** (Willard & Louf, 2023) — compiles regex / JSON-schema to FSMs, indexes into the vocabulary for O(1) valid-token lookup.
- **Guidance** (Microsoft, Lundberg) — CFG / regex / JSON-schema; introduced token-healing to handle BPE-boundary mismatch.
- **LMQL** (Beurer-Kellner, Fischer, Vechev, ETH Zürich, ICML 2023).
- **XGrammar** (MLC-AI, 2024) — CFG with system-level optimisations; near-zero overhead.
- **llama.cpp grammar** — GBNF.
- **DOMINO** (Beurer-Kellner et al., ICML 2024) — non-invasive constrained generation aligned to BPE subwords.

**The "ideas can also be profane" analogy**: this is rhetorically strong but should be explicitly grounded by citing Goodfire Ember's *conditional steering* demonstration (Balsam et al., 23 Dec 2024): a detector activates on jailbreak patterns, then turns up the model's refusal feature, drastically increasing robustness to StrongREJECT-style attacks without affecting latency. The Lume FST version is the symbolic dual: a detector matches a profane *or conceptually undesirable* token-sequence, then a logit-bias suppresses the continuation. The white paper should make the symbolic↔neural duality explicit.

### 7. The document-as-ground-truth paradigm

**Closed-world / faithful retrieval**:
- Nakano et al., WebGPT (OpenAI 2021) — early citation-grounded generation.
- Menick et al., GopherCite (DeepMind 2022) — quote-grounded generation with attribution.
- Bohnet et al., "Attributed Question Answering" (Google 2022).
- Gao et al., "Enabling Large Language Models to Generate Text with Citations" (EMNLP 2023).
- RAGAS (Es et al., 2023) — faithfulness as a measurable metric.

**Faithfulness metrics** (already covered in §4): claim decomposition + NLI entailment is now standard.

**The Lume contribution** is the *three-way provenance classification at the output gate*: verbatim recall vs. concept-grounded synthesis vs. pure-hallucination-from-BPE-noise. This trichotomy does not appear in the published literature in this exact form, but the components do: substring-attribution work (Lee et al., "Deduplicating Training Data Makes Language Models Better," ACL 2022), the "is this in the training data" line of work (Carlini et al.), and the "Accidental Archaeology" concept the user describes is closest to what Carlini et al. call *near-verbatim* memorisation with token-boundary slip. **This is genuinely novel framing and should be highlighted.**

### 8. Specific contextual items

**Lume repository (github.com/kordless/lume)**: As of 25 May 2026, the repository is not publicly accessible. Neither the direct GitHub URL nor `raw.githubusercontent.com` paths resolved, and the repo does not appear in `kordless`'s or `DeepBlueDynamics`'s listings. The white paper must therefore be sourced from internal / authorial documentation, *not* presented as if peer-reviewable from a public link. This is a publication-time risk that should be resolved by making the repo public before release, or by including the README and architectural diagrams as appendices.

**Ferricula / DeepBlue Dynamics programme**: ferricula.com hosts a coherent technical manifesto for a "thermodynamic memory engine" with:
- Roaring Bitmap bit-sliced index (BSI) for fidelity ranges (microsecond resolution).
- Exact cosine over a bitmap-pre-filtered set ("filter first, scan last") — explicit rejection of HNSW / IVF on the grounds that approximate algorithms degrade under constant decay / consolidation, and that "Recall@k lies."
- Per-agent X25519-derived orthogonal rotation encryption (similarity preserved under ciphertext via Q · v / Qᵀ · v').
- Vec2text inversion pipeline (gtr-t5-base + projection + T5 encoder + decoder, four ONNX models) as a fidelity-gate quality check.
- SDR (RTL-SDR) entropy source for non-deterministic decay selection.
- Authored by Kord Campbell; ©2026 DeepBlue Dynamics, LLC.

**The DeepBlue Dynamics ecosystem** (per `deepbluedynamics.com` and `github.com/DeepBlueDynamics`) includes: **nemesis8** (agent runtime, "the kernel" — 69+ MCP tools, multi-provider Claude / Gemini / Codex / OpenClaw, "Pokeball" environments); **hyperia** ("ghost in your shell" terminal/MCP sidecar, Rust fork of vercel/hyper); **grubcrawler** ("world's fastest agentic crawler", Python); **ferricula / ferricula-arena** (thermodynamic memory); **shivvr** ("Chunk, embed & search documents with exact semantic matching — exact k-NN, not approximate. No ANN shortcuts. No external dependency.", Rust — this is the closest public analog to Lume's described retrieval layer); **autoresearch** ("Physics-Inspired Optimizer from Weber's Electrodynamics + RTL-SDR Hardware Entropy for Autonomous LLM Training"); and several radio-entropy projects (`sdr-rand`, `sdrrand-site`). The N.U.T.S. ("Neural Unified Telegraph Service") naming and the Faraday / Weber / telegraph motifs are programmatic, not incidental — they signal a deliberate "natural-philosophy" research stance with which the Victorian-scientific register of the white paper is consistent.

**Beguine framing**: "Beguine" denotes the medieval lay religious communities (12th–13th century Low Countries, especially the Spanish Netherlands and Rhineland) — women who lived in semi-monastic community without taking permanent vows, devoted to manual labour, charity, contemplative reading, and vernacular translation of Latin scripture. Marguerite Porete's *The Mirror of Simple Souls* (c. 1290, in Old French) is the canonical text; Porete was burned for heresy in 1310 at the Place de Grève in Paris following the Council of Vienne. For the white paper's purposes: the Beguine analogy positions sovereign agentic infrastructure as a *lay, non-cloistered, community-scale* alternative to the monastic frontier-lab paradigm — small, devotional, locally-grounded knowledge communities rather than centralised oracular authority. This is consistent with DeepBlue Dynamics' "Sovereign Agentic Infrastructure" tagline.

**Abhidharma as the forthcoming corpus**: The *Abhidharma* (Sanskrit: "higher dharma" / "about the dharma"; Pāli: *Abhidhamma*) is the third "basket" (*piṭaka*) of canonical Buddhist scholastic literature, alongside the Sutta and Vinaya. The Theravāda Abhidhamma Piṭaka has seven books (notably the *Dhammasaṅgaṇī*, *Vibhaṅga*, and *Paṭṭhāna*), and the Sarvāstivāda school's Sanskrit Abhidharma is preserved largely in Chinese (Xuanzang's translations of Vasubandhu's *Abhidharmakośa*). It is *exquisitely well-suited* to the Lume architecture for three reasons: (i) it is a *closed* canon with a fixed, enumerable list of dhammas/factors — a finite ontology a 26 M-parameter oracle can plausibly memorise; (ii) the scholastic genre is intensely *formulaic and combinatorial* (the *Paṭṭhāna* enumerates 24 conditional relations exhaustively across all possible dhamma pairs), exactly the structure where overfit memorisation excels; (iii) the tradition itself values *verbatim recitation with provenance attribution* — the three-way provenance classifier maps directly onto the Abhidhamma's own commentarial distinction between *pāḷi* (canonical text), *aṭṭhakathā* (commentary), and *ṭīkā* (sub-commentary). This is a remarkable second-corpus choice and deserves a dedicated section.

---

## Recommendations

**Stage 1 — Editorial corrections (before any draft circulates)**:
1. Keep the user's "**Gemma 4 E2B**" designation as written — it is correct (Wikipedia: *Gemma language model*; Google's `ai.google.dev/gemma/docs/core`, *Gemma 4 model overview*: "*Gemma 4 models are available in 4 parameter sizes: E2B, E4B, 31B and 26B A4B.*"). Add a footnote describing the architectural lineage (MatFormer → Per-Layer Embeddings → ~2 GB VRAM target → Apache 2.0 release on 2 April 2026).
2. Make `github.com/kordless/lume` **public** before publication, or restructure the white paper to be self-contained (i.e. publish architecture as a primary artefact, not as commentary on a repo). At time of research the repository is not visible on the public web.
3. Clarify "Hatcherik" on first use as a portmanteau / honorific for Erik Hatcher (Apache Lucene/Solr committer; co-author of *Lucene in Action* with McCandless and Gospodnetić, Manning 2nd ed. 2010).

**Stage 2 — Strengthen the scholarly anchors**:
4. Cite **Takeshita et al. (EMNLP 2025, arXiv:2508.17744)** as the central empirical anchor for the "embeddings are mostly noise" argument. Quote the 50% / <10% / 430-of-1024 numbers explicitly.
5. Cite **Templeton et al. (Anthropic, *Scaling Monosemanticity*, May 2024)** and **Balsam, Deng, Nguyen, Gorton, Shihipar, Ho & McGrath, "Goodfire Ember: Scaling Interpretability for Frontier Model Alignment" (Goodfire Research, 23 Dec 2024)** as the activation-steering comparand. Frame Lume's logit-bias as *complementary, not competing*.
6. Cite **Khattab & Zaharia (ColBERT, SIGIR 2020)** and **Cormack, Clarke & Büttcher (RRF, SIGIR 2009)** in the hybrid-retrieval section. Acknowledge ColBERT as the third paradigm. Use the corrected Elasticsearch version (`retrievers` introduced in 8.14, GA in 8.16) and OpenSearch version (RRF and Score ranker processor introduced in 2.19) when discussing vendor support.
7. Cite **Morris et al. (arXiv:2505.24832, May 2025)** to justify the 26 M-parameter sizing decision quantitatively. The 3.6-bits-per-parameter figure should appear in the section motivating model size.
8. Cite **Karpathy 2015 (char-RNN) and Carlini et al. 2023 (memorisation quantification)** to ground the overfit-as-feature inversion.

**Stage 3 — Frame the novel contributions honestly**:
9. The genuinely new contributions are: (a) the *three-way provenance trichotomy* at the output gate (verbatim / concept-grounded / BPE-noise hallucination); (b) the "Coherence Laundering Trap" naming of a real but unnamed failure mode; (c) the multiplicative Hatcherik kernel `Score_hybrid = Score_BM25 × (1.0 + Score_semantic × Weight)` with MinMax normalisation as an alternative to RRF when one signal should *modulate* rather than *vote alongside* the other; (d) the use of the same lexical index at *both* ends of the cognitive loop. These deserve dedicated sections.
10. Acknowledge that *FST-driven logit-bias steering* is not novel as a technique (Outlines / Guidance / llama.cpp grammar / Dathathri et al. PPLM are prior art) — but its *purpose* (steering a deliberately overfit narrow-domain oracle for fidelity rather than syntactic conformance) is novel.

**Stage 4 — Position for the customer audience**:
11. For *knowledge collection construction*: lead with the Mannheim redundancy result and the BM25-first hybrid argument. Show, concretely, that BM25 + roaring-bitmap pre-filter + optional re-rank scales to millions of documents on a laptop.
12. For *edge deployment with small models*: lead with Gemma 4 E2B's ~2 GB VRAM footprint plus a 26 M-parameter oracle as a complete, offline, deterministic stack.
13. For *profanity / concept filtering*: present the FST↔SAE duality table (symbolic dual of Goodfire Ember's conditional steering).
14. For *hallucination reduction*: position the three-way provenance classifier as a *measurable, auditable* alternative to RAGAS-style post-hoc faithfulness scoring.
15. For *grounding to fixed corpora*: lead with the Abhidharma example as a flagship use case — it generalises immediately to legal codes, regulatory canon, engineering standards (ISO, IEEE), pharmacopoeias, and corporate policy.

**Benchmarks / thresholds that would change these recommendations**:
- If a future paper shows that activation-steered SAE features achieve sub-token granularity comparable to logit-bias, recommendation #5 should be revisited.
- If `kordless/lume` is published with a permissive license and a benchmark suite, recommendations #2–#3 are obviated.
- If Google releases a Gemma 4.x successor before the white paper goes to press, update the edge-model designation accordingly.

---

## Caveats

1. **The exact phrase "few hundred concepts" has no published source** as of May 2026. The closest published claims are Takeshita et al.'s ~50% dimension-removal robustness and the Anthropic SAE feature-count results. The white paper should *paraphrase* rather than attribute that phrasing.
2. **"Coherence Laundering" is a coinage** — no public source uses this term. The white paper should explicitly mark it as a coinage and cite the underlying failure mode under its existing names ("generation drift," "glitch tokens," "near-verbatim memorisation with boundary slip").
3. **The Lume repository was not publicly verifiable** at research time (25 May 2026). All technical claims about the implementation must be sourced from authorial materials.
4. **Weaviate's strict-pre-filter behaviour is a known limitation** (issue #7681) — the white paper can use this to differentiate Lume, but should not misrepresent Weaviate as broken; this is an ongoing engineering issue, not a design failure.
5. **The Platonic Representation Hypothesis** (Huh et al. 2024) is *positioned* — its title explicitly marks it as a "Position" paper. It is contested. Use it for framing, not as established fact.
6. **Steering vectors generalise unreliably out of distribution** (Tan et al., arXiv:2407.12404). This *helps* Lume's case for symbolic logit-bias in narrow-domain settings, but the white paper should not overclaim that activation steering "doesn't work" — only that it makes different tradeoffs.
7. **The Morris et al. capacity result (3.6 bits/param)** is for GPT-style transformers trained from scratch on synthetic data; extrapolation to a 26 M-parameter model overfit on Faraday is suggestive but not strictly derived. Frame as "consistent with" rather than "implied by."
8. **The Abhidharma corpus framing is the author's editorial extension** of the user's brief — the user noted Abhidharma is "the next document/corpus"; the analytical reasoning for *why* it fits is the author's, and should be presented as such, not as the original architects' design intent unless the authors confirm.
9. **Gemma raw-parameter counts vary across sources for Gemma 3n E2B**: Google's developer blog (June 2025) cites 5 B raw; the HuggingFace card `google/gemma-3n-E2B` states "*While the raw parameter count of this model is 6B.*" Both sources agree on the ~1.91 B effective figure. The discrepancy reflects different accounting (whether Per-Layer Embeddings are counted as raw parameters). For Gemma 4 E2B, official numbers should be confirmed from `ai.google.dev/gemma/docs/core` at the time of publication.