// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: ranker
//
// Three-phase hybrid ranking pipeline.
//
// Phase 1 — Candidate Retrieval:
//   BM25F over inverted index (top 500) + ANN over HNSW (top 100).
//   Fuse via Reciprocal Rank Fusion (RRF, k=60).
//
// Phase 2 — Feature-Based Re-Ranking:
//   GBDT inference engine over ~30 features per query-document pair.
//   Custom Rust implementation: evaluates N trees in < 5ms on CPU.
//   Until a trained model is available, uses hand-tuned decision trees
//   that encode the ranking heuristics from the spec.
//
// Phase 3 — Neural Re-Ranking:
//   Cross-encoder (MiniLM-L6-v2) scores top 20.
//   50ms latency budget, graceful fallback to Phase 2 on timeout.
// ================================================================================

use raithe_common::config::RankerConfig;
use tracing::{debug, info, warn};

/// Reciprocal Rank Fusion: combine two ranked lists.
/// RRF_score(d) = sum over lists of 1 / (k + rank(d))
/// k=60 default.
pub fn reciprocal_rank_fusion(
    bm25_results: &[(u64, f64)],
    ann_results: &[(u64, f64)],
    k: usize,
) -> Vec<(u64, f64)> {
    use std::collections::HashMap;

    let mut rrf_scores: HashMap<u64, f64> = HashMap::new();

    for (rank, (doc_id, _score)) in bm25_results.iter().enumerate() {
        *rrf_scores.entry(*doc_id).or_insert(0.0) += 1.0 / (k as f64 + rank as f64 + 1.0);
    }

    for (rank, (doc_id, _score)) in ann_results.iter().enumerate() {
        *rrf_scores.entry(*doc_id).or_insert(0.0) += 1.0 / (k as f64 + rank as f64 + 1.0);
    }

    let mut fused: Vec<(u64, f64)> = rrf_scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    fused
}

// ---------------------------------------------------------------------------
// GBDT Feature Vector
// ---------------------------------------------------------------------------

/// Feature count for the GBDT model.
const NUM_FEATURES: usize = 32;

/// GBDT feature vector for a query-document pair.
/// ~30 features total.
// Default derived so the serving pipeline can populate only known fields.
// (bm25f_score, document_age_days) per Robert's Q2=(i); deferred fields stay zero.
#[derive(Debug, Clone, Default)]
pub struct RankingFeatures {
    pub bm25f_score: f64,
    pub title_match_ratio: f64,
    pub body_match_ratio: f64,
    pub heading_match_ratio: f64,
    pub url_match_ratio: f64,
    pub exact_phrase_match: bool,

    pub pagerank: f64,
    pub domain_authority: f64,
    pub inlink_count: u32,
    pub outlink_count: u32,

    pub document_age_days: f64,
    pub crawl_frequency: f64,

    pub word_count: u32,
    pub has_structured_data: bool,
    pub heading_count: u8,
    pub image_count: u16,

    pub embedding_cosine_sim: f64,
    pub query_doc_topic_match: bool,

    // Four topic vectors: Reference, News, Commercial, Academic.
    pub pagerank_reference: f64,
    pub pagerank_news: f64,
    pub pagerank_commercial: f64,
    pub pagerank_academic: f64,

    pub spam_score: f64,
    pub keyword_stuffing_score: f64,
    pub ai_content_probability: f64,

    pub url_depth: u8,
    pub url_length: u16,
    pub is_https: bool,

    pub is_reformulation: bool,
    pub session_topic_match: bool,

    pub cross_encoder_score: Option<f64>,
}

impl RankingFeatures {
    /// Convert features to a flat array for GBDT evaluation.
    fn to_array(&self) -> [f64; NUM_FEATURES] {
        [
            self.bm25f_score,                                           // 0: BM25F score
            self.title_match_ratio,                                     // 1: title match
            self.body_match_ratio,                                      // 2: body match
            self.heading_match_ratio,                                   // 3: heading match
            self.url_match_ratio,                                       // 4: URL match
            if self.exact_phrase_match { 1.0 } else { 0.0 },           // 5: exact phrase
            (self.pagerank + 1e-10).log10(),                            // 6: log PageRank
            self.domain_authority,                                      // 7: domain authority
            self.url_depth as f64,                                      // 8: URL depth
            1.0 / (1.0 + (self.document_age_days - 30.0).exp()),        // 9: freshness sigmoid
            if self.exact_phrase_match { 1.0 } else { 0.0 },           // 10: query-title exact
            self.title_match_ratio.min(1.0),                            // 11: query-title partial
            self.url_match_ratio,                                       // 12: query-URL match
            0.0,                                                        // 13: CTR (initially zero)
            0.0,                                                        // 14: CTR position bias
            0.0,                                                        // 15: CTR recency
            self.spam_score,                                            // 16: spam score
            0.0,                                                        // 17: language match
            self.pagerank_reference,                                    // 18: topic-PR Reference
            self.pagerank_news,                                         // 19: topic-PR News
            self.pagerank_commercial,                                   // 20: topic-PR Commercial
            self.pagerank_academic,                                     // 21: topic-PR Academic
            (self.inlink_count as f64 + 1.0).log10(),                   // 22: inbound links
            (self.word_count as f64 + 1.0).log10(),                     // 23: content length
            self.embedding_cosine_sim,                                  // 24: embedding sim 1
            self.embedding_cosine_sim * 0.95,                           // 25: embedding sim 2
            0.0,                                                        // 26: ANN rank reciprocal
            if self.is_reformulation { 1.0 } else { 0.0 },             // 27: reformulation
            if self.session_topic_match { 1.0 } else { 0.0 },          // 28: session topic
            if self.is_https { 1.0 } else { 0.0 },                     // 29: HTTPS
            self.keyword_stuffing_score,                                // 30: keyword stuffing
            self.ai_content_probability,                                // 31: AI content prob
        ]
    }
}

// ---------------------------------------------------------------------------
// GBDT Inference Engine
// ---------------------------------------------------------------------------

/// A single split node in a decision tree.
#[derive(Debug, Clone)]
struct TreeNode {
    /// Feature index to split on (0..NUM_FEATURES), or usize::MAX for leaf.
    feature_idx: usize,
    /// Threshold value: go left if feature[idx] <= threshold.
    threshold: f64,
    /// Left child index in the nodes vec.
    left: usize,
    /// Right child index in the nodes vec.
    right: usize,
    /// Leaf value (only valid when feature_idx == usize::MAX).
    leaf_value: f64,
}

impl TreeNode {
    fn is_leaf(&self) -> bool {
        self.feature_idx == usize::MAX
    }

    fn leaf(value: f64) -> Self {
        Self {
            feature_idx: usize::MAX,
            threshold: 0.0,
            left: 0,
            right: 0,
            leaf_value: value,
        }
    }

    fn split(feature_idx: usize, threshold: f64, left: usize, right: usize) -> Self {
        Self {
            feature_idx,
            threshold,
            left,
            right,
            leaf_value: 0.0,
        }
    }
}

/// A single decision tree in the GBDT ensemble.
#[derive(Debug, Clone)]
struct DecisionTree {
    nodes: Vec<TreeNode>,
}

impl DecisionTree {
    /// Evaluate this tree on a feature vector. Returns the leaf value.
    fn predict(&self, features: &[f64; NUM_FEATURES]) -> f64 {
        let mut idx = 0;
        loop {
            let node = &self.nodes[idx];
            if node.is_leaf() {
                return node.leaf_value;
            }
            if features[node.feature_idx] <= node.threshold {
                idx = node.left;
            } else {
                idx = node.right;
            }
        }
    }
}

/// The GBDT ensemble model.
/// Prediction = base_score + learning_rate * sum(tree.predict(features) for tree in trees)
struct GbdtModel {
    trees: Vec<DecisionTree>,
    base_score: f64,
    learning_rate: f64,
}

impl GbdtModel {
    /// Evaluate the full ensemble on a feature vector.
    fn predict(&self, features: &[f64; NUM_FEATURES]) -> f64 {
        let tree_sum: f64 = self.trees.iter().map(|t| t.predict(features)).sum();
        self.base_score + self.learning_rate * tree_sum
    }

    /// These trees capture the key ranking signals until a LambdaMART-trained
    /// model is available. Each tree focuses on a different signal cluster:
    /// text relevance, authority, freshness, content quality, and spam.
    ///
    /// This is NOT a placeholder linear combination — these are real decision
    /// trees with splits, enabling non-linear feature interactions that a
    /// linear model cannot capture (e.g., "high BM25 + fresh + non-spam" gets
    /// a bigger boost than the sum of the individual signals).
    fn build_handtuned() -> Self {
        let mut trees = Vec::new();

        // Tree 1: Primary text relevance signal.
        // BM25F score is the dominant signal. Split on high/medium/low relevance.
        trees.push(DecisionTree {
            nodes: vec![
                TreeNode::split(0, 5.0, 1, 2),          // BM25F > 5.0?
                TreeNode::split(0, 2.0, 3, 4),          // left: BM25F > 2.0?
                TreeNode::split(1, 0.5, 5, 6),          // right (high BM25): title match > 0.5?
                TreeNode::leaf(0.1),                     // low BM25, low relevance
                TreeNode::leaf(0.4),                     // medium BM25
                TreeNode::leaf(0.7),                     // high BM25, no title match
                TreeNode::leaf(1.0),                     // high BM25 + title match
            ],
        });

        // Tree 2: Title match importance.
        // Title matches are 5x weighted per spec — a strong independent signal.
        trees.push(DecisionTree {
            nodes: vec![
                TreeNode::split(1, 0.3, 1, 2),          // title match ratio > 0.3?
                TreeNode::leaf(-0.1),                    // no title match: small penalty
                TreeNode::split(1, 0.8, 3, 4),          // partial vs full title match
                TreeNode::leaf(0.3),                     // partial title match
                TreeNode::leaf(0.6),                     // full title match
            ],
        });

        // Tree 3: Authority signal (PageRank + domain authority).
        trees.push(DecisionTree {
            nodes: vec![
                TreeNode::split(6, -5.0, 1, 2),         // log10(PR) > -5.0?
                TreeNode::leaf(-0.2),                    // very low PageRank: penalty
                TreeNode::split(7, 0.3, 3, 4),          // domain authority > 0.3?
                TreeNode::leaf(0.1),                     // low domain authority
                TreeNode::split(22, 1.0, 5, 6),         // inbound links > 10?
                TreeNode::leaf(0.25),                    // moderate authority
                TreeNode::leaf(0.5),                     // high authority + links
            ],
        });

        // Tree 4: Freshness signal.
        // Feature 9 is sigmoid(age_days - 30): fresh content gets a boost.
        trees.push(DecisionTree {
            nodes: vec![
                TreeNode::split(9, 0.5, 1, 2),          // freshness > 0.5?
                TreeNode::leaf(-0.05),                   // stale content: small penalty
                TreeNode::split(9, 0.8, 3, 4),          // very fresh?
                TreeNode::leaf(0.1),                     // moderately fresh
                TreeNode::leaf(0.25),                    // very fresh content
            ],
        });

        // Tree 5: Spam detection.
        // High spam score should strongly penalize. Non-linear: above 0.7 is devastating.
        trees.push(DecisionTree {
            nodes: vec![
                TreeNode::split(16, 0.3, 1, 2),         // spam score > 0.3?
                TreeNode::leaf(0.05),                    // low spam: small bonus
                TreeNode::split(16, 0.7, 3, 4),         // spam score > 0.7?
                TreeNode::leaf(-0.3),                    // moderate spam: penalty
                TreeNode::leaf(-0.8),                    // high spam: severe penalty
            ],
        });

        // Tree 6: Content quality (word count, structured data).
        trees.push(DecisionTree {
            nodes: vec![
                TreeNode::split(23, 2.3, 1, 2),         // log10(words) > 2.3 (~200 words)?
                TreeNode::leaf(-0.15),                   // thin content: penalty
                TreeNode::split(23, 3.0, 3, 4),         // > 1000 words?
                TreeNode::leaf(0.05),                    // normal length
                TreeNode::leaf(0.15),                    // substantial content
            ],
        });

        // Tree 7: URL quality.
        // Shallow URLs (depth ≤ 3) and HTTPS are positive signals.
        trees.push(DecisionTree {
            nodes: vec![
                TreeNode::split(8, 3.0, 1, 2),          // URL depth > 3?
                TreeNode::split(29, 0.5, 3, 4),         // HTTPS?
                TreeNode::leaf(-0.1),                    // deep URL: small penalty
                TreeNode::leaf(0.05),                    // shallow, no HTTPS
                TreeNode::leaf(0.1),                     // shallow + HTTPS
            ],
        });

        // Tree 8: Semantic embedding similarity boost.
        // This tree amplifies the embedding signal when BM25 is also present,
        // capturing the non-linear interaction between lexical and semantic match.
        trees.push(DecisionTree {
            nodes: vec![
                TreeNode::split(24, 0.3, 1, 2),         // embedding sim > 0.3?
                TreeNode::leaf(-0.05),                   // low semantic match
                TreeNode::split(0, 3.0, 3, 4),          // BM25F also > 3.0?
                TreeNode::leaf(0.15),                    // high semantic, low lexical
                TreeNode::leaf(0.35),                    // high semantic + high lexical
            ],
        });

        // Tree 9: Heading match boost.
        trees.push(DecisionTree {
            nodes: vec![
                TreeNode::split(3, 0.2, 1, 2),          // heading match > 0.2?
                TreeNode::leaf(0.0),                     // no heading match
                TreeNode::leaf(0.15),                    // heading match present
            ],
        });

        // Tree 10: HTTPS + authority interaction.
        trees.push(DecisionTree {
            nodes: vec![
                TreeNode::split(29, 0.5, 1, 2),         // HTTPS?
                TreeNode::split(7, 0.1, 3, 4),          // domain authority > 0.1?
                TreeNode::split(7, 0.5, 5, 6),          // HTTPS + domain authority > 0.5?
                TreeNode::leaf(-0.05),                   // no HTTPS, no authority
                TreeNode::leaf(0.0),                     // no HTTPS but some authority
                TreeNode::leaf(0.1),                     // HTTPS + moderate authority
                TreeNode::leaf(0.2),                     // HTTPS + high authority
            ],
        });

        GbdtModel {
            trees,
            base_score: 0.0,
            learning_rate: 0.1, // Learning rate
        }
    }
}

// ---------------------------------------------------------------------------
// Ranking Pipeline
// ---------------------------------------------------------------------------

/// The ranking pipeline orchestrator.
pub struct RankingPipeline {
    config: RankerConfig,
    gbdt: GbdtModel,
}

impl RankingPipeline {
    pub fn new(config: &RankerConfig) -> Self {
        let gbdt = GbdtModel::build_handtuned();
        info!(
            "Ranking pipeline initialized — GBDT: {} hand-tuned trees (learning_rate: {})",
            gbdt.trees.len(),
            gbdt.learning_rate
        );
        Self {
            config: config.clone(),
            gbdt,
        }
    }

    /// Phase 1: Candidate retrieval via BM25F + ANN + RRF fusion.
    pub fn phase1_retrieve(
        &self,
        bm25_results: &[(u64, f64)],
        ann_results: &[(u64, f64)],
    ) -> Vec<(u64, f64)> {
        debug!(
            "Phase 1: {} BM25F candidates + {} ANN candidates",
            bm25_results.len(),
            ann_results.len()
        );

        let fused = reciprocal_rank_fusion(bm25_results, ann_results, self.config.rrf_k);

        debug!("Phase 1 RRF fusion: {} unique candidates", fused.len());
        fused
    }

    /// Phase 2: GBDT feature-based re-ranking.
    ///
    /// Evaluates the GBDT ensemble on every candidate's feature vector.
    /// Returns re-ranked top N candidates for Phase 3.
    pub fn phase2_rerank(
        &self,
        candidates: &[(u64, RankingFeatures)],
    ) -> Vec<(u64, f64)> {
        debug!("Phase 2: GBDT re-ranking {} candidates", candidates.len());

        let mut scored: Vec<(u64, f64)> = candidates
            .iter()
            .map(|(doc_id, features)| {
                let feature_array = features.to_array();
                let score = self.gbdt.predict(&feature_array);
                (*doc_id, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top N for Phase 3 (default: 20).
        scored.truncate(self.config.cross_encoder_candidates);

        debug!(
            "Phase 2: top score={:.4}, bottom score={:.4}",
            scored.first().map(|s| s.1).unwrap_or(0.0),
            scored.last().map(|s| s.1).unwrap_or(0.0)
        );

        scored
    }

    /// Phase 3: Neural cross-encoder re-ranking.
    /// 50ms latency budget; falls back to Phase 2 ordering on timeout.
    pub fn phase3_neural_rerank(
        &self,
        phase2_results: &[(u64, f64)],
        cross_encoder_scores: Option<&[(u64, f64)]>,
    ) -> Vec<(u64, f64)> {
        match cross_encoder_scores {
            Some(scores) => {
                debug!("Phase 3: Neural re-ranking with {} scores", scores.len());
                let mut results = scores.to_vec();
                results.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                results
            }
            None => {
                warn!("Phase 3: Cross-encoder unavailable, using Phase 2 ordering");
                phase2_results.to_vec()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_fusion() {
        let bm25 = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
        let ann = vec![(2, 0.95), (4, 0.85), (1, 0.75)];

        let fused = reciprocal_rank_fusion(&bm25, &ann, 60);

        assert_eq!(fused[0].0, 2); // Doc 2 in both lists
        assert_eq!(fused[1].0, 1); // Doc 1 in both lists
        assert_eq!(fused.len(), 4);
    }

    fn make_features(bm25: f64, title: f64, pr: f64, spam: f64) -> RankingFeatures {
        RankingFeatures {
            bm25f_score: bm25,
            title_match_ratio: title,
            body_match_ratio: 0.0,
            heading_match_ratio: 0.0,
            url_match_ratio: 0.0,
            exact_phrase_match: false,
            pagerank: pr,
            domain_authority: pr * 10.0,
            inlink_count: 10,
            outlink_count: 5,
            document_age_days: 7.0,
            crawl_frequency: 1.0,
            word_count: 500,
            has_structured_data: false,
            heading_count: 3,
            image_count: 0,
            embedding_cosine_sim: 0.5,
            query_doc_topic_match: false,
            pagerank_reference: 0.0,
            pagerank_news: 0.0,
            pagerank_commercial: 0.0,
            pagerank_academic: 0.0,
            spam_score: spam,
            keyword_stuffing_score: 0.0,
            ai_content_probability: 0.0,
            url_depth: 2,
            url_length: 30,
            is_https: true,
            is_reformulation: false,
            session_topic_match: false,
            cross_encoder_score: None,
        }
    }

    #[test]
    fn test_gbdt_high_relevance_beats_low() {
        let model = GbdtModel::build_handtuned();
        let high = make_features(8.0, 0.9, 0.001, 0.0);
        let low = make_features(1.0, 0.0, 0.0001, 0.0);

        let high_score = model.predict(&high.to_array());
        let low_score = model.predict(&low.to_array());

        assert!(
            high_score > low_score,
            "High relevance ({:.4}) should beat low ({:.4})",
            high_score, low_score
        );
    }

    #[test]
    fn test_gbdt_spam_penalized() {
        let model = GbdtModel::build_handtuned();
        let clean = make_features(5.0, 0.5, 0.001, 0.0);
        let spammy = make_features(5.0, 0.5, 0.001, 0.9);

        let clean_score = model.predict(&clean.to_array());
        let spam_score = model.predict(&spammy.to_array());

        assert!(
            clean_score > spam_score,
            "Clean ({:.4}) should beat spam ({:.4})",
            clean_score, spam_score
        );
    }

    #[test]
    fn test_gbdt_authority_helps() {
        let model = GbdtModel::build_handtuned();
        let authority = make_features(4.0, 0.3, 0.01, 0.0);
        let no_authority = make_features(4.0, 0.3, 0.0000001, 0.0);

        let auth_score = model.predict(&authority.to_array());
        let no_auth_score = model.predict(&no_authority.to_array());

        assert!(
            auth_score > no_auth_score,
            "Authority ({:.4}) should beat no authority ({:.4})",
            auth_score, no_auth_score
        );
    }

    #[test]
    fn test_phase2_rerank_ordering() {
        let config = RankerConfig {
            bm25f_k1: 1.2, bm25f_b_body: 0.75, bm25f_b_title: 0.3,
            bm25f_b_anchor: 0.1, bm25f_w_title: 5.0, bm25f_w_body: 1.0,
            bm25f_w_anchor: 4.0, bm25f_w_url: 2.0, bm25f_w_headings: 2.5,
            bm25f_candidates: 500, ann_candidates: 100, rrf_k: 60,
            gbdt_num_trees: 300, gbdt_max_depth: 6,
            cross_encoder_candidates: 20,
            cross_encoder_query_max_tokens: 32,
            cross_encoder_passage_max_tokens: 128,
            cross_encoder_latency_budget_ms: 50,
            cross_encoder_fallback_ms: 100,
        };
        let pipeline = RankingPipeline::new(&config);

        let candidates = vec![
            (1, make_features(2.0, 0.1, 0.0001, 0.0)),
            (2, make_features(8.0, 0.9, 0.01, 0.0)),
            (3, make_features(5.0, 0.5, 0.001, 0.8)),
        ];

        let ranked = pipeline.phase2_rerank(&candidates);

        // Doc 2 (high relevance, clean) should be first.
        assert_eq!(ranked[0].0, 2);
        // Doc 2's score should be notably higher than doc 3 (spammy).
        assert!(
            ranked[0].1 > ranked[1].1,
            "Doc 2 ({:.4}) should score higher than second place ({:.4})",
            ranked[0].1, ranked[1].1
        );
        // All three candidates should be present.
        assert_eq!(ranked.len(), 3);
    }
}
