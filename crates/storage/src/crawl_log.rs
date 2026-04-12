// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: storage :: crawl_log
//
// On-disk append-only crawl log.
// Replaces Kafka for single-node: parsed documents are written as length-
// prefixed JSON records. The indexer polls this log every 60 seconds.
//
// This is the abstraction boundary: in distributed mode, this becomes
// a Kafka producer. The CrawlEmitter trait interface is identical.
// ================================================================================

use raithe_common::error::RaitheResult;
use raithe_common::traits::{AsyncResult, CrawlEmitter};
use raithe_common::types::ParsedDocument;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use tracing::{debug, error};

/// File-based crawl log writer.
/// Documents are serialized as length-prefixed JSON records in an append-only file.
pub struct FileCrawlLog {
    log_dir: PathBuf,
    current_file: Arc<Mutex<Option<tokio::fs::File>>>,
    current_file_size: AtomicU64,
    pending_count: AtomicU64,
    /// Max file size before rotating (default: 256 MB).
    max_file_size: u64,
    /// Sequence number for file rotation.
    file_sequence: AtomicU64,
}

impl FileCrawlLog {
    /// Create a new crawl log writer in the given directory.
    pub async fn new(log_dir: PathBuf) -> RaitheResult<Self> {
        std::fs::create_dir_all(&log_dir)?;

        // Count existing log files to set the sequence number.
        let mut max_seq: u64 = 0;
        if let Ok(entries) = std::fs::read_dir(&log_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Some(seq_str) = name.strip_prefix("crawl-").and_then(|s| s.strip_suffix(".log")) {
                        if let Ok(seq) = seq_str.parse::<u64>() {
                            max_seq = max_seq.max(seq);
                        }
                    }
                }
            }
        }

        Ok(Self {
            log_dir,
            current_file: Arc::new(Mutex::new(None)),
            current_file_size: AtomicU64::new(0),
            pending_count: AtomicU64::new(0),
            max_file_size: 256 * 1024 * 1024, // 256 MB
            file_sequence: AtomicU64::new(max_seq + 1),
        })
    }

    /// Get the path to a log file by sequence number.
    fn log_file_path(&self, sequence: u64) -> PathBuf {
        self.log_dir.join(format!("crawl-{:08}.log", sequence))
    }

    /// Open or rotate the current log file.
    async fn ensure_file(&self) -> RaitheResult<()> {
        let mut file_guard = self.current_file.lock().await;

        if file_guard.is_some() && self.current_file_size.load(Ordering::Relaxed) < self.max_file_size {
            return Ok(());
        }

        let seq = self.file_sequence.fetch_add(1, Ordering::SeqCst);
        let path = self.log_file_path(seq);
        debug!("Opening new crawl log file: {:?}", path);

        let file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;

        *file_guard = Some(file);
        self.current_file_size.store(0, Ordering::Relaxed);

        Ok(())
    }

    /// List all unprocessed log files in order.
    pub fn list_log_files(&self) -> Vec<PathBuf> {
        let mut files = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.log_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "log") {
                    files.push(path);
                }
            }
        }
        files.sort();
        files
    }

    /// Read all documents from a log file.
    pub fn read_log_file(path: &Path) -> RaitheResult<Vec<ParsedDocument>> {
        let data = std::fs::read(path)?;
        let mut documents = Vec::new();
        let mut offset = 0;

        while offset + 4 <= data.len() {
            // Read 4-byte length prefix (big-endian).
            let len = u32::from_be_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + len > data.len() {
                error!("Truncated record in crawl log at offset {}", offset - 4);
                break;
            }

            match serde_json::from_slice::<ParsedDocument>(&data[offset..offset + len]) {
                Ok(doc) => documents.push(doc),
                Err(e) => {
                    error!("Failed to deserialize crawl log record: {}", e);
                }
            }
            offset += len;
        }

        Ok(documents)
    }
}

impl CrawlEmitter for FileCrawlLog {
    fn emit(&self, document: ParsedDocument) -> AsyncResult<'_, ()> {
        Box::pin(async move {
            self.ensure_file().await?;

            let json = serde_json::to_vec(&document)
                .map_err(|e| raithe_common::error::RaitheError::Internal(e.into()))?;

            let len_bytes = (json.len() as u32).to_be_bytes();

            let mut file_guard = self.current_file.lock().await;
            if let Some(ref mut file) = *file_guard {
                file.write_all(&len_bytes).await?;
                file.write_all(&json).await?;
                file.flush().await?;

                self.current_file_size.fetch_add(
                    (4 + json.len()) as u64,
                    Ordering::Relaxed,
                );
                self.pending_count.fetch_add(1, Ordering::Relaxed);

                debug!(
                    url = %document.url,
                    "Emitted document to crawl log ({} bytes)",
                    json.len()
                );
            }

            Ok(())
        })
    }

    fn pending_count(&self) -> AsyncResult<'_, u64> {
        Box::pin(async move { Ok(self.pending_count.load(Ordering::Relaxed)) })
    }
}
