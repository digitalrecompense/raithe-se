// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: storage
//
// Storage engine: local SSD, memory-mapped file access, segment lifecycle
// management, and crawl log.
//
// The on-disk crawl log replaces Kafka for single-node deployment.
// The CrawlEmitter trait implementation writes to an append-only file log;
// in distributed mode, this becomes a Kafka producer.
// ================================================================================

pub mod crawl_log;
pub mod mmap_file;
pub mod backup;

use raithe_common::config::StorageConfig;
use std::path::Path;
use tracing::info;

/// Initialize all storage directories.
pub fn init_storage_dirs(config: &StorageConfig) -> std::io::Result<()> {
    let dirs = [
        &config.index_dir,
        &config.vectors_dir,
        &config.linkgraph_dir,
        &config.crawl_log_dir,
        &config.knowledge_dir,
        &config.models_dir,
        &config.backups_dir,
    ];

    // Also ensure the frontier DB parent directory exists.
    if let Some(parent) = config.frontier_db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    for dir in &dirs {
        std::fs::create_dir_all(dir)?;
        info!("Storage directory ready: {:?}", dir);
    }

    Ok(())
}

/// Check disk usage and return (used_bytes, total_bytes) for the given path.
pub fn disk_usage(path: &Path) -> std::io::Result<(u64, u64)> {
    // Use statvfs on Linux to get filesystem stats.
    // Fallback: return zeroes if not available.
    #[cfg(target_os = "linux")]
    {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let c_path = CString::new(path.as_os_str().as_bytes())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

        unsafe {
            let mut stat: libc::statvfs = std::mem::zeroed();
            if libc::statvfs(c_path.as_ptr(), &mut stat) == 0 {
                let total = stat.f_blocks as u64 * stat.f_frsize as u64;
                let available = stat.f_bavail as u64 * stat.f_frsize as u64;
                let used = total - available;
                Ok((used, total))
            } else {
                Err(std::io::Error::last_os_error())
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        Ok((0, 0))
    }
}
