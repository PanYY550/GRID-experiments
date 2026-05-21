You are a research literature reviewer specializing in recommender systems and representation learning. I need you to conduct a thorough novelty assessment of a proposed contribution. Please search for prior work and evaluate whether each component has been done before.

## Research Context

**Domain**: Generative Retrieval with Semantic IDs (SID) for recommendation. Items are mapped to discrete semantic tokens via Residual Quantization (RQ-VAE). A T5-based encoder-decoder (TIGER) generates item SIDs autoregressively for next-item recommendation.

**Key problem**: RQ-VAE training uses HaMR (Hard Mixture of Repulsions) loss to prevent codebook collapse. The repulsion is symmetric — all items experience equal gradient — but items have vastly different exposure (popularity). This causes long-tail item representations to drift away from their semantic manifold.

**Our proposed mechanism**: Exposure-Aware Asymmetric Repulsion (Path B) — partial gradient detachment so popular items absorb more repulsion, protecting tail items. Controlled by a parameter α ∈ [0, 1] that dictates what fraction of gradient tail items receive.

## Our Causal Chain (5 Links, All Closed)

**Link 1** — α → Asymmetric Gradient Allocation (★★★)
- Code-level evidence: partial detach in RQ loss means α directly controls gradient allocation ratio between head/tail items
- Validated by parameter sweep: 8 α values

**Link 2** — Asymmetric Allocation → Directional Drift (★★☆)
- We measure ECRD (Exposure-Conditional Representation Drift): 1 - cos_sim(normalized(z_group), normalized(z_G0)) per item
- Higher ECRD in head items, lower in tail items under asymmetric repulsion
- Evidence: head-tail ECRD gradient aligns with α

**Link 3** — Directional Drift → High DSF (★★☆)
- We propose DSF (Drift Semantic Fidelity): |kNN(z_group) ∩ kNN(z_G0)| / K
- Measures preservation of semantic neighbors after drift
- Validated on 2 independent experiments (Path B and alpha_grid, 6 groups total)

**Link 4** — DSF → NDCG (cross-group) (★★☆)
- Cross-group: higher mean DSF predicts higher NDCG
- 2 independent experiments confirm DSF ranking matches NDCG ranking, while collision rate does NOT

**Link 5** — DSF → Per-Item Hit Rate (within-group, per-item) (★★★)
- Per-item Pearson r = +0.064, p < 1e-43, n = 47,868 items across 6 groups
- DSF-healthy items have higher hit rates at the individual item level
- Strongest in high-exposure bins (7+ exposures, r ≈ +0.07-0.09)

## Specific Claims to Assess for Novelty

Please search for prior work and assess each claim:

1. **Exposure-Aware Asymmetric Repulsion in RQ-VAE / VQ**: Has anyone proposed using exposure/popularity to modulate repulsion loss gradients in vector quantization for recommendation? Search terms: "asymmetric repulsion VQ", "exposure-aware RQ-VAE", "popularity-weighted commitment loss", "gradient detachment VQ".

2. **ECRD (Exposure-Conditional Representation Drift)**: Has anyone measured item representation drift conditioned on exposure bins to analyze how training objectives affect tail vs head items differently? Search: "exposure-conditional representation drift", "popularity-stratified embedding analysis", "tail item representation shift".

3. **DSF (Drift Semantic Fidelity)**: Has anyone proposed measuring k-NN neighbor preservation as a quality metric for learned item representations? Search: "semantic fidelity VQ", "kNN preservation recommendation", "neighbor consistency quantization", "semantic neighbor overlap".

4. **Collision-NDCG Inversion**: Has anyone reported that lower SID collision rate does NOT predict better recommendation performance? Search: "semantic ID collision", "RQ-VAE codebook collision recommendation", "collision rate NDCG", "codebook diversity recommendation quality".

5. **Full causal chain from gradient allocation to recommendation quality**: Has anyone established a complete causal pathway from training-time gradient modulation → representation drift metrics → semantic fidelity → downstream recommendation quality in generative retrieval? Search: "generative retrieval causal analysis", "SID training downstream causal", "representation learning to recommendation causal chain".

## Instructions for Each Claim

For each of the 5 claims above:
- Find the closest prior work (paper title, authors, year, venue if available)
- Summarize how similar or different it is from our claim
- Rate novelty: HIGH / MEDIUM / LOW
- If LOW or MEDIUM, explain what would need to change for the claim to be novel

## Overall Assessment

After assessing individual claims, provide:
1. Whether the COMBINATION of all 5 links as a complete causal chain is novel
2. Whether this is publishable at a top-tier venue (KDD, WWW, RecSys, SIGIR)
3. What are the weakest novelty claims we should strengthen
4. Suggested framing to maximize novelty perception
