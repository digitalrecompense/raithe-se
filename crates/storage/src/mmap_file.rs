// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: storage :: mmap_file
//
// Memory-mapped file access (zero-copy everywhere).
// The OS page cache becomes our caching layer.
// ================================================================================

use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::path::{Path, PathBuf};
use tracing::debug;

/// A read-only memory-mapped file.
/// Used for index segments, link graph files, and embedding vectors.
pub struct MmapFile {
    path: PathBuf,
    mmap: Mmap,
    len: usize,
}

impl MmapFile {
    /// Open a file and memory-map it read-only.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let len = metadata.len() as usize;

        // SAFETY: The file is opened read-only and we hold no mutable references.
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        debug!("Memory-mapped {:?} ({} bytes)", path, len);

        Ok(Self {
            path: path.to_path_buf(),
            mmap,
            len,
        })
    }

    /// Get a zero-copy slice of the mapped data.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }

    /// Get a sub-slice at the given offset and length.
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> Option<&[u8]> {
        if offset + len <= self.len {
            Some(&self.mmap[offset..offset + len])
        } else {
            None
        }
    }

    /// Total length of the mapped file.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the mapped file is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The path to the underlying file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Advise the OS to prefetch the entire file into the page cache.
    /// Useful for files that will be accessed sequentially (e.g., PageRank iteration).
    pub fn advise_sequential(&self) {
        #[cfg(target_os = "linux")]
        {
            unsafe {
                libc::posix_madvise(
                    self.mmap.as_ptr() as *mut libc::c_void,
                    self.len,
                    libc::POSIX_MADV_SEQUENTIAL,
                );
            }
        }
    }

    /// Advise the OS that access will be random.
    /// Useful for index lookups and HNSW traversals.
    pub fn advise_random(&self) {
        #[cfg(target_os = "linux")]
        {
            unsafe {
                libc::posix_madvise(
                    self.mmap.as_ptr() as *mut libc::c_void,
                    self.len,
                    libc::POSIX_MADV_RANDOM,
                );
            }
        }
    }
}

/// Read an array of f32 values from a memory-mapped file at a given byte offset.
/// Used for reading PageRank scores and domain authority from .linkdata files.
#[inline]
pub fn read_f32_at(mmap: &MmapFile, byte_offset: usize) -> Option<f32> {
    let bytes = mmap.slice(byte_offset, 4)?;
    Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Read a slice of f32 values from a memory-mapped file.
#[inline]
pub fn read_f32_slice(mmap: &MmapFile, byte_offset: usize, count: usize) -> Option<&[u8]> {
    mmap.slice(byte_offset, count * 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_mmap_roundtrip() {
        let mut tmp = NamedTempFile::new().unwrap();
        let data = b"RAiTHE INDUSTRIES test data for mmap";
        tmp.write_all(data).unwrap();
        tmp.flush().unwrap();

        let mmap = MmapFile::open(tmp.path()).unwrap();
        assert_eq!(mmap.len(), data.len());
        assert_eq!(mmap.as_slice(), data);
    }

    #[test]
    fn test_mmap_slice() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"0123456789").unwrap();
        tmp.flush().unwrap();

        let mmap = MmapFile::open(tmp.path()).unwrap();
        assert_eq!(mmap.slice(2, 4), Some(b"2345".as_slice()));
        assert_eq!(mmap.slice(8, 5), None); // out of bounds
    }
}
