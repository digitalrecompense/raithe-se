// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: freshness
//
// Incremental re-crawl pipeline.
//
// File-based pipeline for single-node:
// 1. Indexer polls crawl log every 60 seconds.
// 2. New documents are added; deletions are tombstoned.
// 3. Merges consolidate tombstones periodically.
// ================================================================================

use raithe_common::config::IndexerConfig;
use tracing::{debug, info};

/// Tombstone record marking a document for deletion at next merge.
#[derive(Debug, Clone)]
pub struct Tombstone {
    pub doc_id: u64,
    pub url: String,
    pub deleted_at: chrono::DateTime<chrono::Utc>,
    pub reason: DeletionReason,
}

/// Reasons a document may be tombstoned.
#[derive(Debug, Clone)]
pub enum DeletionReason {
    /// URL returned 404/410 on re-crawl.
    Gone,
    /// Content changed — old version replaced by new.
    Updated,
    /// Detected as spam after re-evaluation.
    Spam,
    /// Near-duplicate of another document.
    Duplicate { canonical_url: String },
}

/// Manage the freshness pipeline: scheduling re-crawls and applying updates.
pub struct FreshnessPipeline {
    _config: IndexerConfig,
    tombstones: Vec<Tombstone>,
}

impl FreshnessPipeline {
    pub fn new(config: &IndexerConfig) -> Self {
        Self {
            _config: config.clone(),
            tombstones: Vec::new(),
        }
    }

    /// Add a tombstone for a document.
    pub fn tombstone(&mut self, doc_id: u64, url: &str, reason: DeletionReason) {
        let ts = Tombstone {
            doc_id,
            url: url.to_string(),
            deleted_at: chrono::Utc::now(),
            reason,
        };
        debug!(doc_id = doc_id, url = url, "Tombstoned document");
        self.tombstones.push(ts);
    }

    /// Get pending tombstones for the next merge cycle.
    pub fn pending_tombstones(&self) -> &[Tombstone] {
        &self.tombstones
    }

    /// Clear tombstones after a successful merge.
    pub fn clear_tombstones(&mut self) {
        let count = self.tombstones.len();
        self.tombstones.clear();
        info!("Cleared {} tombstones after merge", count);
    }
}
