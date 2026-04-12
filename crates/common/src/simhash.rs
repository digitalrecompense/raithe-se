// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: common :: simhash
//
// 64-bit SimHash for near-duplicate page detection.
// Custom implementation — no external crate dependency.
// ================================================================================

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Compute a 64-bit SimHash fingerprint from an iterator of tokens.
///
/// SimHash produces locality-sensitive hashes: similar documents will have
/// similar hashes (small Hamming distance). Used for deduplication in the
/// crawl pipeline.
pub fn compute_simhash<'a>(tokens: impl Iterator<Item = &'a str>) -> u64 {
    let mut v = [0i32; 64];

    for token in tokens {
        let hash = hash_token(token);
        for i in 0..64 {
            if (hash >> i) & 1 == 1 {
                v[i] += 1;
            } else {
                v[i] -= 1;
            }
        }
    }

    let mut fingerprint: u64 = 0;
    for i in 0..64 {
        if v[i] > 0 {
            fingerprint |= 1 << i;
        }
    }

    fingerprint
}

/// Compute the Hamming distance between two SimHash fingerprints.
/// Two documents are considered near-duplicates if distance <= threshold
/// (default threshold: 3).
#[inline]
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Check if two fingerprints are near-duplicates (distance <= threshold).
#[inline]
pub fn is_near_duplicate(a: u64, b: u64, threshold: u32) -> bool {
    hamming_distance(a, b) <= threshold
}

/// Hash a single token to a 64-bit value.
fn hash_token(token: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    token.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_documents_have_zero_distance() {
        let tokens = vec!["the", "quick", "brown", "fox"];
        let h1 = compute_simhash(tokens.iter().copied());
        let h2 = compute_simhash(tokens.iter().copied());
        assert_eq!(hamming_distance(h1, h2), 0);
    }

    #[test]
    fn test_similar_documents_have_small_distance() {
        let doc1 = vec!["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"];
        let doc2 = vec!["the", "fast", "brown", "fox", "jumps", "over", "the", "lazy", "cat"];
        let h1 = compute_simhash(doc1.iter().copied());
        let h2 = compute_simhash(doc2.iter().copied());
        let dist = hamming_distance(h1, h2);
        // Similar documents should have relatively small distance
        assert!(dist < 20, "Expected small distance, got {}", dist);
    }

    #[test]
    fn test_very_different_documents_have_large_distance() {
        let doc1 = vec!["rust", "programming", "language", "systems"];
        let doc2 = vec!["cooking", "recipe", "chocolate", "cake"];
        let h1 = compute_simhash(doc1.iter().copied());
        let h2 = compute_simhash(doc2.iter().copied());
        let dist = hamming_distance(h1, h2);
        // Very different documents should have larger distance
        assert!(dist > 5, "Expected large distance, got {}", dist);
    }

    #[test]
    fn test_near_duplicate_check() {
        let h1: u64 = 0b1111_0000;
        let h2: u64 = 0b1111_0001; // 1 bit different
        assert!(is_near_duplicate(h1, h2, 3));
        assert!(is_near_duplicate(h1, h2, 1));
        assert!(!is_near_duplicate(h1, h2, 0));
    }
}
