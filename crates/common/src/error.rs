// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: common :: error
//
// Centralized error types for raithe-se.
// Uses thiserror for ergonomic error definitions.
// ================================================================================

use thiserror::Error;

/// Top-level error type for raithe-se.
#[derive(Error, Debug)]
pub enum RaitheError {
    // -- Crawler errors --
    #[error("Crawl error for URL {url}: {source}")]
    Crawl {
        url: String,
        #[source]
        source: anyhow::Error,
    },

    #[error("Robots.txt denied access to {url}")]
    RobotsDenied { url: String },

    #[error("HTTP fetch timeout after {timeout_secs}s for {url}")]
    FetchTimeout { url: String, timeout_secs: u64 },

    // -- Parser errors --
    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Content too short: {word_count} words (minimum: 200)")]
    ContentTooShort { word_count: usize },

    // -- Index errors --
    #[error("Index error: {0}")]
    Index(#[from] IndexError),

    // -- Storage errors --
    #[error("Storage I/O error: {0}")]
    Storage(#[from] std::io::Error),

    #[error("SQLite error: {0}")]
    Sqlite(String),

    // -- Ranking errors --
    #[error("Ranking error: {0}")]
    Ranking(String),

    // -- Neural inference errors --
    #[error("Neural inference error: {0}")]
    NeuralInference(String),

    #[error("GPU not available, falling back to CPU-only ranking")]
    GpuUnavailable,

    // -- Query errors --
    #[error("Query parse error: {0}")]
    QueryParse(String),

    // -- Config errors --
    #[error("Configuration error: {0}")]
    Config(String),

    // -- General --
    #[error("Internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

/// Index-specific errors.
#[derive(Error, Debug)]
pub enum IndexError {
    #[error("Tantivy error: {0}")]
    Tantivy(String),

    #[error("Segment merge failed: {0}")]
    MergeFailed(String),

    #[error("Document not found: {doc_id}")]
    DocumentNotFound { doc_id: u64 },
}

/// Result type alias for raithe-se operations.
pub type RaitheResult<T> = std::result::Result<T, RaitheError>;
