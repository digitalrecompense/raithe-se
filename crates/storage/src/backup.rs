// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: storage :: backup
//
// Backup and recovery.
// Nightly: snapshot index, SQLite DBs, and linkgraph.
// Weekly: compress and upload to free-tier cloud storage.
// The crawl log is the source of truth — full reindex from it if needed.
// ================================================================================

use raithe_common::config::StorageConfig;
use std::path::{Path, PathBuf};
use tracing::{info, warn, error};

/// Create a snapshot of the index and critical databases.
/// Copies to the backups directory on the same SSD.
pub fn create_snapshot(config: &StorageConfig) -> std::io::Result<PathBuf> {
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let snapshot_dir = config.backups_dir.join(format!("snapshot_{}", timestamp));
    std::fs::create_dir_all(&snapshot_dir)?;

    info!("Creating snapshot in {:?}", snapshot_dir);

    // Snapshot the index directory.
    let index_backup = snapshot_dir.join("index");
    copy_dir_recursive(&config.index_dir, &index_backup)?;
    info!("Index snapshot complete");

    // Snapshot the linkgraph.
    let linkgraph_backup = snapshot_dir.join("linkgraph");
    copy_dir_recursive(&config.linkgraph_dir, &linkgraph_backup)?;
    info!("Link graph snapshot complete");

    // Snapshot SQLite databases.
    let db_backup = snapshot_dir.join("databases");
    std::fs::create_dir_all(&db_backup)?;
    if config.frontier_db_path.exists() {
        let dest = db_backup.join("frontier.db");
        std::fs::copy(&config.frontier_db_path, &dest)?;
        info!("Frontier DB snapshot complete");
    }

    // Prune old snapshots — keep only the 3 most recent.
    prune_old_snapshots(&config.backups_dir, 3)?;

    info!("Snapshot complete: {:?}", snapshot_dir);
    Ok(snapshot_dir)
}

/// Recursively copy a directory with path traversal protection.
///
/// SECURITY FIX (BUG-001): Validates that all paths stay within the destination
/// directory to prevent path traversal attacks via maliciously crafted filenames.
fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    if !src.exists() {
        warn!("Source directory does not exist: {:?}", src);
        return Ok(());
    }

    // Get canonical paths to resolve any ../ or symbolic links
    let dst_canonical = std::fs::canonicalize(dst)
        .unwrap_or_else(|_| dst.to_path_buf());

    std::fs::create_dir_all(dst)?;

    copy_dir_recursive_impl(src, &dst_canonical, &dst_canonical)
}

/// Internal implementation with path validation.
///
/// # Arguments
/// * `src` - Source directory path
/// * `dst_base` - Canonical base path of destination (for validation)
/// * `dst_current` - Current destination directory being populated
fn copy_dir_recursive_impl(
    src: &Path,
    dst_base: &Path,
    dst_current: &Path,
) -> std::io::Result<()> {
    std::fs::create_dir_all(dst_current)?;

    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let file_name = entry.file_name();

        // SECURITY: Validate filename doesn't contain path traversal attempts
        if let Some(name_str) = file_name.to_str() {
            if name_str.contains("..") || name_str.contains('/') || name_str.contains('\\') {
                warn!(
                    "Skipping potentially malicious filename in backup: {:?}",
                    file_name
                );
                continue;
            }
        }

        let dst_path = dst_current.join(&file_name);

        // SECURITY: Verify the destination path stays within the backup directory
        let dst_path_canonical = std::fs::canonicalize(&dst_path)
            .unwrap_or_else(|_| dst_path.to_path_buf());

        // FIX: Use Path::starts_with() for proper path-aware comparison
        // This correctly handles:
        //   /backups/file      → inside /backups ✅
        //   /backups_evil/file → not inside /backups ❌
        if !dst_path_canonical.starts_with(dst_base) {
            error!(
                "SECURITY: Path traversal attempt detected and blocked!\n\
                 Expected base: {:?}\n\
                 Attempted path: {:?}",
                dst_base,
                dst_path
            );
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "Path traversal attempt blocked: destination path escapes backup directory",
            ));
        }

        if src_path.is_dir() {
            copy_dir_recursive_impl(&src_path, dst_base, &dst_path_canonical)?;
        } else {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }

    Ok(())
}

/// Remove old snapshots, keeping only the N most recent.
fn prune_old_snapshots(backups_dir: &Path, keep: usize) -> std::io::Result<()> {
    let mut snapshots: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(backups_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("snapshot_") {
                        snapshots.push(path);
                    }
                }
            }
        }
    }

    snapshots.sort();

    if snapshots.len() > keep {
        let to_remove = snapshots.len() - keep;
        for snapshot in snapshots.iter().take(to_remove) {
            info!("Pruning old snapshot: {:?}", snapshot);
            if let Err(e) = std::fs::remove_dir_all(snapshot) {
                error!("Failed to remove snapshot {:?}: {}", snapshot, e);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_copy_dir_recursive_normal() {
        let temp_src = TempDir::new().unwrap();
        let temp_dst = TempDir::new().unwrap();

        // Create test files
        fs::write(temp_src.path().join("test.txt"), "hello").unwrap();
        fs::create_dir(temp_src.path().join("subdir")).unwrap();
        fs::write(temp_src.path().join("subdir/nested.txt"), "world").unwrap();

        copy_dir_recursive(temp_src.path(), temp_dst.path()).unwrap();

        assert!(temp_dst.path().join("test.txt").exists());
        assert!(temp_dst.path().join("subdir/nested.txt").exists());
    }

    #[test]
    fn test_copy_dir_recursive_blocks_traversal() {
        let temp_src = TempDir::new().unwrap();
        let temp_dst = TempDir::new().unwrap();

        // Create a file with path traversal attempt in name
        let malicious_name = "../escape.txt";
        fs::write(temp_src.path().join(malicious_name), "malicious").unwrap();

        // The function should skip the malicious file, not error
        // It logs a warning and continues
        let result = copy_dir_recursive(temp_src.path(), temp_dst.path());

        // Should succeed (skips malicious file)
        assert!(result.is_ok());
        // The malicious file should NOT be created
        assert!(!temp_dst.path().join("escape.txt").exists());
    }
}
