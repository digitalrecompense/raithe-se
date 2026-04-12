// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: common :: traits
//
// Scale-ready trait abstractions.
//
// Every inter-component boundary is defined by a trait. The single-node
// implementation calls functions directly; the distributed implementation
// swaps in gRPC clients. Same interface, same tests, different transport.
// ================================================================================

use crate::error::RaitheResult;
use crate::types::*;
use std::future::Future;
use std::pin::Pin;

/// Async result type alias for trait methods.
pub type AsyncResult<'a, T> = Pin<Box<dyn Future<Output = RaitheResult<T>> + Send + 'a>>;

// ---------------------------------------------------------------------------
// IndexReader trait
// Currently calls Tantivy directly. At Tier 1, wraps gRPC calls to remote
// shard nodes.
// ---------------------------------------------------------------------------

pub trait IndexReader: Send + Sync {
    /// Retrieve BM25F candidate documents for the given query terms.
    fn retrieve_bm25f(
        &self,
        terms: &[String],
        max_candidates: usize,
    ) -> AsyncResult<'_, Vec<(u64, f64)>>; // (doc_id, score)

    /// Fetch stored fields for a document by ID.
    fn get_document(&self, doc_id: u64) -> AsyncResult<'_, Option<StoredDocument>>;

    /// Get the total number of indexed documents.
    fn doc_count(&self) -> AsyncResult<'_, u64>;
}

/// Stored fields retrieved from the index for result enrichment.
#[derive(Debug, Clone)]
pub struct StoredDocument {
    pub doc_id: u64,
    pub url: String,
    pub title: String,
    pub body_snippet_text: String,
    pub meta_description: String,
    pub language: String,
    pub last_crawled: chrono::DateTime<chrono::Utc>,
}

// ---------------------------------------------------------------------------
// CrawlEmitter trait
// Currently writes to a local file log. At Tier 2, produces to Kafka.
// ---------------------------------------------------------------------------

pub trait CrawlEmitter: Send + Sync {
    /// Emit a parsed document to the crawl log for the indexer to consume.
    fn emit(&self, document: ParsedDocument) -> AsyncResult<'_, ()>;

    /// Get the number of documents pending indexing.
    fn pending_count(&self) -> AsyncResult<'_, u64>;
}

// ---------------------------------------------------------------------------
// ConfigSource trait
// Currently reads a TOML file. At Tier 2, watches etcd.
// ---------------------------------------------------------------------------

pub trait ConfigSource: Send + Sync {
    /// Get the current configuration value for a key.
    fn get(&self, key: &str) -> Option<String>;

    /// Check if a feature flag is enabled.
    fn is_feature_enabled(&self, flag: &str) -> bool;
}

// ---------------------------------------------------------------------------
// SessionStore trait
// Currently uses in-memory moka cache. At Tier 1, swaps in Redis.
// ---------------------------------------------------------------------------

pub trait SessionStore: Send + Sync {
    /// Get session context for a session ID.
    fn get_session(&self, session_id: &str) -> AsyncResult<'_, Option<SessionContext>>;

    /// Update session with a new query.
    fn update_session(
        &self,
        session_id: &str,
        query: &str,
    ) -> AsyncResult<'_, SessionContext>;
}

// ---------------------------------------------------------------------------
// NeuralInference trait
// Currently runs on local GTX 1080. At Tier 2, calls a remote GPU
// inference fleet.
// ---------------------------------------------------------------------------

pub trait NeuralInference: Send + Sync {
    /// Compute dense embedding for a query string.
    fn embed_query(&self, query: &str) -> AsyncResult<'_, Vec<f32>>;

    /// Compute dense embeddings for a batch of documents.
    fn embed_documents(&self, texts: &[String]) -> AsyncResult<'_, Vec<Vec<f32>>>;

    /// Score query-document pairs with the cross-encoder.
    fn cross_encode(
        &self,
        query: &str,
        passages: &[String],
    ) -> AsyncResult<'_, Vec<f64>>;

    /// Run LLM query reformulation.
    fn reformulate_query(&self, query: &str) -> AsyncResult<'_, Option<LlmReformulation>>;

    /// Check if the GPU is available and healthy.
    fn is_gpu_available(&self) -> bool;
}

// ---------------------------------------------------------------------------
// InstantAnswerProvider trait
// ---------------------------------------------------------------------------

pub trait InstantAnswerProvider: Send + Sync {
    /// Attempt to produce an instant answer for the given query.
    fn answer(&self, query: &ParsedQuery) -> AsyncResult<'_, Option<InstantAnswer>>;
}

// ---------------------------------------------------------------------------
// SpamClassifier trait
// ---------------------------------------------------------------------------

pub trait SpamClassifier: Send + Sync {
    /// Compute a spam score for a parsed document. Returns 0.0 (not spam) to
    /// 1.0 (definitely spam).
    fn score(&self, document: &ParsedDocument) -> f64;

    /// Check if a domain is blacklisted.
    fn is_blacklisted(&self, domain: &str) -> bool;
}
