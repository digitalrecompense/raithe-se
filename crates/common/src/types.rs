// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: common :: types
//
// Core shared data types used across all raithe-se subsystems.
// These types define the data contracts between crates.
// ================================================================================

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A document as it comes out of the parser, ready for indexing.
/// Fields correspond to structural feature extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedDocument {
    /// Normalized canonical URL.
    pub url: String,
    /// Original URL before normalization.
    pub original_url: String,
    /// Page title, cleaned of site name suffix (weight: 5.0x).
    pub title: String,
    /// All heading text, by level (weight: 3.0x h1 down to 1.2x h6).
    pub headings: Vec<Heading>,
    /// Main content after boilerplate removal (weight: 1.0x baseline).
    pub body: String,
    /// Tokenized URL path segments (weight: 2.0x).
    pub url_tokens: Vec<String>,
    /// Meta description (weight: 1.5x).
    pub meta_description: String,
    /// Image alt text within main content (weight: 0.8x).
    pub alt_texts: Vec<String>,
    /// JSON-LD / Microdata structured data (weight: 1.5x).
    pub structured_data: Vec<String>,
    /// Outgoing links extracted from the page.
    pub outgoing_links: Vec<ExtractedLink>,
    /// Detected language code (e.g., "en", "fr").
    pub language: String,
    /// Open Graph metadata.
    pub og_tags: OgTags,
    /// Canonical URL from <link rel="canonical">.
    pub canonical_url: Option<String>,
    /// SimHash fingerprint of the body text.
    pub simhash: u64,
    /// Word count of the body after boilerplate removal.
    pub word_count: usize,
    /// Timestamp when this document was crawled.
    pub crawled_at: DateTime<Utc>,
    /// HTTP status code received.
    pub http_status: u16,
    /// Content-Type header value.
    pub content_type: String,
}

/// A heading extracted from the page with its level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heading {
    /// Heading level: 1 (h1) through 6 (h6).
    pub level: u8,
    /// Text content of the heading.
    pub text: String,
}

/// An outgoing link extracted from a page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedLink {
    /// Target URL (normalized).
    pub url: String,
    /// Anchor text.
    pub anchor_text: String,
    /// Whether the link is followed (true) or nofollowed (false).
    pub is_followed: bool,
}

/// Open Graph metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OgTags {
    pub title: Option<String>,
    pub description: Option<String>,
    pub image: Option<String>,
    pub og_type: Option<String>,
    pub site_name: Option<String>,
}

/// A search query as understood by the query pipeline.
#[derive(Debug, Clone)]
pub struct ParsedQuery {
    /// Original query string as typed by the user.
    pub original: String,
    /// Tokenized and normalized query terms.
    pub terms: Vec<String>,
    /// Spell-corrected query (if different from original).
    pub corrected: Option<String>,
    /// Intent classification result.
    pub intent: QueryIntent,
    /// Intent classifier confidence score.
    pub intent_confidence: f32,
    /// Detected phrase segments (wrapped in implicit quotes).
    pub phrases: Vec<String>,
    /// Synonym expansions: (original_term, synonym, weight).
    pub synonyms: Vec<(String, String, f32)>,
    /// Whether the enhanced LLM path should be triggered.
    pub needs_enhanced_path: bool,
    /// LLM reformulation result (if enhanced path ran).
    pub llm_reformulation: Option<LlmReformulation>,
    /// Detected numeric ranges, dates, versions.
    pub special_tokens: Vec<SpecialToken>,
}

/// Query intent classification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryIntent {
    Navigational,
    Informational,
    Transactional,
    Local,
}

/// LLM reformulation output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmReformulation {
    /// Reformulated query optimized for lexical retrieval.
    pub reformulated_query: String,
    /// Alternative interpretations for ambiguous queries.
    pub alternatives: Vec<String>,
    /// Extracted entities and constraints.
    pub entities: Vec<(String, String)>, // (entity_type, entity_value)
}

/// Special token types detected in queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpecialToken {
    NumericRange { min: f64, max: f64 },
    Date(DateTime<Utc>),
    VersionNumber(String),
}

/// A single search result ready for rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Document ID in the index.
    pub doc_id: u64,
    /// Page URL.
    pub url: String,
    /// Page title.
    pub title: String,
    /// Generated snippet with highlighted query terms.
    pub snippet: String,
    /// Final blended relevance score.
    pub score: f64,
    /// Meta description (fallback if snippet is empty).
    pub meta_description: String,
    /// Domain name for display.
    pub domain: String,
    /// When the page was last crawled.
    pub last_crawled: DateTime<Utc>,
    /// Favicon URL (if known).
    pub favicon_url: Option<String>,
}

/// A complete search response including results and instant answers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// The (possibly corrected) query string.
    pub query: String,
    /// Spell correction suggestion (if any).
    pub did_you_mean: Option<String>,
    /// Instant answer card (if triggered).
    pub instant_answer: Option<InstantAnswer>,
    /// Ranked organic results.
    pub results: Vec<SearchResult>,
    /// Total number of matching documents.
    pub total_hits: u64,
    /// Query latency in milliseconds.
    pub latency_ms: u64,
    /// Which ranking phases were used.
    pub ranking_phases: Vec<String>,
}

/// Instant answer card.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstantAnswer {
    /// Answer type (entity, calculation, conversion, definition, etc.).
    pub answer_type: InstantAnswerType,
    /// The direct answer text or value.
    pub answer: String,
    /// Additional structured data for rich rendering.
    pub details: serde_json::Value,
    /// Source attribution.
    pub source: String,
    /// Confidence score (threshold: 0.8.
    pub confidence: f32,
}

/// Instant answer types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstantAnswerType {
    EntityKnowledgePanel,
    Calculation,
    UnitConversion,
    CurrencyConversion,
    Dictionary,
    DateTime,
    Weather,
    StockQuote,
}

/// URL frontier entry for the crawler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierEntry {
    /// Normalized URL to crawl.
    pub url: String,
    /// Domain of the URL.
    pub domain: String,
    /// Priority score (composite of authority, staleness, change rate).
    pub priority: f64,
    /// Last time this URL was crawled (None if never crawled).
    pub last_crawled: Option<DateTime<Utc>>,
    /// Estimated change probability.
    pub change_probability: f64,
    /// Number of times this URL has been crawled.
    pub crawl_count: u32,
}

/// Per-document link data stored in the .linkdata sidecar.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LinkData {
    /// PageRank score.
    pub pagerank: f32,
    /// Domain authority score (aggregate PageRank of the domain).
    pub domain_authority: f32,
}

/// Session context for a user's search session.
#[derive(Debug, Clone)]
pub struct SessionContext {
    /// Session identifier.
    pub session_id: String,
    /// Previous queries in this session (most recent first).
    pub previous_queries: Vec<String>,
    /// Embedding similarities to previous queries.
    pub query_similarities: Vec<f32>,
    /// Whether this appears to be a query reformulation.
    pub is_reformulation: bool,
    /// Topic cluster ID for this session.
    pub topic_cluster_id: Option<u32>,
}
