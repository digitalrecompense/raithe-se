// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: query
//
// Query understanding pipeline.
//
// Dual-path architecture:
//   Fast Path — 10ms budget:
//     Tokenize → spell correction → synonym expansion → phrase detection →
//     intent classification (rule-based).
//
//   Enhanced Path — 200ms budget:
//     Triggered when intent confidence < 0.7 OR query tokens ≥ 8.
//     LLM reformulation → entity extraction → alternative interpretations.
//     LRU cache (100K entries) to avoid redundant LLM calls.
//
// Spell Correction:
//   SymSpell-inspired symmetric delete algorithm. Dictionary built from
//   index term frequencies. Max edit distance 1 for short queries, 2 for
//   longer. Only suggests corrections when corrected term frequency is
//   >10x the original.
// ================================================================================

use raithe_common::config::QueryConfig;
use raithe_common::types::{LlmReformulation, ParsedQuery, QueryIntent, SpecialToken};
use moka::sync::Cache;
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Duration;
use tracing::debug;
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

// ---------------------------------------------------------------------------
// SymSpell-inspired Spell Correction Dictionary
// ---------------------------------------------------------------------------

/// A spell correction dictionary using the symmetric delete approach.
///
/// For each word in the dictionary, we pre-compute all delete variants
/// within max_edit_distance. At query time, we generate deletes of the
/// misspelled word and look them up. If a delete of the query word matches
/// a delete of a dictionary word, the edit distance between them is at most
/// the sum of the delete distances.
///
/// This is an O(1) lookup after initialization — no expensive dynamic
/// programming at query time. Engineering specifies this exact algorithm.
struct SpellDictionary {
    /// term → document frequency count.
    term_freqs: HashMap<String, u64>,
    /// delete variant → list of (original_word, delete_distance) pairs.
    deletes: HashMap<String, Vec<(String, u8)>>,
    /// Maximum edit distance for corrections.
    max_edit_distance: u8,
}

impl SpellDictionary {
    fn new(max_edit_distance: u8) -> Self {
        Self {
            term_freqs: HashMap::new(),
            deletes: HashMap::new(),
            max_edit_distance,
        }
    }

    /// Add a word to the dictionary with its document frequency.
    fn add_word(&mut self, word: &str, freq: u64) {
        let lower = word.to_lowercase();
        if lower.len() < 2 || lower.len() > 30 {
            return;
        }

        self.term_freqs.insert(lower.clone(), freq);

        // Generate all delete variants within max_edit_distance.
        let mut deletes_set = Vec::new();
        generate_deletes(&lower, self.max_edit_distance, &mut deletes_set);

        for (variant, dist) in deletes_set {
            self.deletes
                .entry(variant)
                .or_default()
                .push((lower.clone(), dist));
        }

        // The word itself is a "0-delete" variant.
        self.deletes
            .entry(lower.clone())
            .or_default()
            .push((lower, 0));
    }

    /// Look up spelling corrections for a word.
    /// Returns the best correction if one exists with edit distance ≤ max
    /// and frequency >10x the original.
    fn correct(&self, word: &str, max_dist: u8) -> Option<String> {
        let lower = word.to_lowercase();

        // If the word is already in the dictionary, no correction needed.
        if self.term_freqs.contains_key(&lower) {
            return None;
        }

        let effective_max = max_dist.min(self.max_edit_distance);
        let mut best: Option<(String, u8, u64)> = None; // (word, distance, freq)

        // Generate deletes of the query word and look up candidates.
        let mut query_deletes = Vec::new();
        generate_deletes(&lower, effective_max, &mut query_deletes);
        // Also check the word itself (handles candidates that are deletes of dict words).
        query_deletes.push((lower.clone(), 0));

        for (variant, q_dist) in &query_deletes {
            if let Some(candidates) = self.deletes.get(variant) {
                for (dict_word, d_dist) in candidates {
                    let total_dist = q_dist + d_dist;
                    if total_dist > effective_max {
                        continue;
                    }

                    // Verify with real edit distance (the delete approach is a filter,
                    // not exact — verify with Levenshtein).
                    let actual_dist = edit_distance(&lower, dict_word);
                    if actual_dist > effective_max as usize {
                        continue;
                    }

                    let freq = *self.term_freqs.get(dict_word).unwrap_or(&0);

                    // Only suggest when corrected term frequency >10x original.
                    let original_freq = *self.term_freqs.get(&lower).unwrap_or(&0);
                    if freq <= original_freq * 10 && original_freq > 0 {
                        continue;
                    }

                    let is_better = match &best {
                        None => true,
                        Some((_, best_dist, best_freq)) => {
                            (actual_dist as u8) < *best_dist
                                || ((actual_dist as u8) == *best_dist && freq > *best_freq)
                        }
                    };

                    if is_better {
                        best = Some((dict_word.clone(), actual_dist as u8, freq));
                    }
                }
            }
        }

        best.map(|(word, _, _)| word)
    }

    /// Number of words in the dictionary.
    fn len(&self) -> usize {
        self.term_freqs.len()
    }
}

/// Generate all delete variants of a word within max_distance.
/// Each variant is (deleted_string, number_of_deletes).
fn generate_deletes(word: &str, max_distance: u8, results: &mut Vec<(String, u8)>) {
    if max_distance == 0 || word.is_empty() {
        return;
    }

    let chars: Vec<char> = word.chars().collect();
    for i in 0..chars.len() {
        let mut deleted: String = chars[..i].iter().collect();
        deleted.extend(&chars[i + 1..]);

        // Avoid duplicate entries.
        let already_present = results.iter().any(|(s, _)| s == &deleted);
        if !already_present {
            results.push((deleted.clone(), 1));
            // Recurse for multi-character deletes.
            if max_distance > 1 {
                let mut sub_deletes = Vec::new();
                generate_deletes(&deleted, max_distance - 1, &mut sub_deletes);
                for (sub, sub_dist) in sub_deletes {
                    let total = 1 + sub_dist;
                    if !results.iter().any(|(s, _)| s == &sub) {
                        results.push((sub, total));
                    }
                }
            }
        }
    }
}

/// Compute the Levenshtein edit distance between two strings.
fn edit_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 { return n; }
    if n == 0 { return m; }

    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for j in 0..=n {
        prev[j] = j;
    }

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

// ---------------------------------------------------------------------------
// Query Understanding Engine
// ---------------------------------------------------------------------------

/// The query understanding engine.
pub struct QueryEngine {
    config: QueryConfig,
    /// LRU cache for LLM reformulations.
    llm_cache: Cache<String, LlmReformulation>,
    /// Spell correction dictionary, populated from index term frequencies.
    /// Behind RwLock so it can be updated without replacing the engine.
    spell_dict: RwLock<SpellDictionary>,
}

impl QueryEngine {
    pub fn new(config: &QueryConfig) -> Self {
        let llm_cache = Cache::builder()
            .max_capacity(config.llm_cache_size)
            .time_to_live(Duration::from_secs(3600))
            .build();

        // Initialize with an empty dictionary. Populated later from index
        // term frequencies via populate_spell_dictionary().
        let max_edit = config.spell_max_edit_long as u8;
        let spell_dict = RwLock::new(SpellDictionary::new(max_edit));

        Self {
            config: config.clone(),
            llm_cache,
            spell_dict,
        }
    }

    /// Check the LLM reformulation cache for a previously computed result.
    /// Called before invoking the (expensive) neural LLM reformulation.
    pub fn get_cached_reformulation(&self, query: &str) -> Option<LlmReformulation> {
        self.llm_cache.get(&query.to_lowercase())
    }

    /// Store an LLM reformulation result in the cache.
    /// The cache is an LRU with 100K entries and 1-hour TTL.
    pub fn cache_reformulation(&self, query: &str, reformulation: LlmReformulation) {
        self.llm_cache.insert(query.to_lowercase(), reformulation);
    }

    /// Populate the spell dictionary from index term frequencies.
    /// Call this after the index is built or periodically as new docs are indexed.
    pub fn populate_spell_dictionary(&self, term_freqs: &[(String, u64)]) {
        let mut dict = self.spell_dict.write().unwrap();
        for (term, freq) in term_freqs {
            dict.add_word(term, *freq);
        }
        debug!("Spell dictionary populated with {} terms", dict.len());
    }

    /// Run the full query understanding pipeline.
    pub fn parse_query(&self, raw_query: &str) -> ParsedQuery {
        let start = std::time::Instant::now();

        // Step 1: Unicode normalization and tokenization.
        let normalized: String = raw_query.nfkc().collect();
        let terms: Vec<String> = normalized
            .unicode_words()
            .map(|w| w.to_lowercase())
            .collect();

        // Step 2: Spell correction.
        let corrected = self.spell_correct(&terms);

        // Step 3: Phrase detection — detect quoted phrases.
        let phrases = self.detect_phrases(&normalized);

        // Step 4: Special token detection (dates, numbers, versions).
        let special_tokens = self.detect_special_tokens(&terms);

        // Step 5: Rule-based intent classification.
        let (intent, intent_confidence) = self.classify_intent(&terms);

        // Step 6: Synonym expansion.
        let synonyms = self.expand_synonyms(&terms);

        // Step 7: Determine if enhanced path is needed.
        let needs_enhanced = intent_confidence < self.config.intent_confidence_threshold
            || terms.len() >= self.config.enhanced_path_token_threshold;

        debug!(
            query = %raw_query,
            intent = ?intent,
            confidence = intent_confidence,
            enhanced = needs_enhanced,
            elapsed_ms = start.elapsed().as_millis(),
            "Query parsed"
        );

        ParsedQuery {
            original: raw_query.to_string(),
            terms,
            corrected,
            intent,
            intent_confidence,
            phrases,
            synonyms,
            needs_enhanced_path: needs_enhanced,
            llm_reformulation: None,
            special_tokens,
        }
    }

    /// Spell correction using symmetric delete algorithm.
    /// Max edit distance: 1 for queries ≤ 4 tokens, 2 for longer queries.
    fn spell_correct(&self, terms: &[String]) -> Option<String> {
        let dict = match self.spell_dict.read() {
            Ok(d) => d,
            Err(_) => return None,
        };

        // No dictionary yet — skip correction.
        if dict.len() == 0 {
            return None;
        }

        let max_dist = if terms.len() <= 4 {
            self.config.spell_max_edit_short as u8
        } else {
            self.config.spell_max_edit_long as u8
        };

        let mut any_corrected = false;
        let corrected_terms: Vec<String> = terms
            .iter()
            .map(|term| {
                // Don't correct very short terms or terms that look like special tokens.
                if term.len() <= 2 || term.chars().any(|c| c.is_ascii_digit()) {
                    return term.clone();
                }

                match dict.correct(term, max_dist) {
                    Some(correction) => {
                        debug!(original = %term, corrected = %correction, "Spell correction");
                        any_corrected = true;
                        correction
                    }
                    None => term.clone(),
                }
            })
            .collect();

        if any_corrected {
            Some(corrected_terms.join(" "))
        } else {
            None
        }
    }

    /// Detect quoted phrase segments in the query.
    fn detect_phrases(&self, query: &str) -> Vec<String> {
        let mut phrases = Vec::new();
        let mut in_quote = false;
        let mut current = String::new();

        for ch in query.chars() {
            if ch == '"' {
                if in_quote {
                    let trimmed = current.trim().to_string();
                    if !trimmed.is_empty() {
                        phrases.push(trimmed);
                    }
                    current.clear();
                }
                in_quote = !in_quote;
            } else if in_quote {
                current.push(ch);
            }
        }

        phrases
    }

    /// Detect special tokens: numeric ranges, dates, version numbers.
    fn detect_special_tokens(&self, terms: &[String]) -> Vec<SpecialToken> {
        let mut tokens = Vec::new();

        for term in terms.iter() {
            let cleaned = term.strip_prefix('v').unwrap_or(term);
            if cleaned.contains('.') && cleaned.chars().all(|c| c.is_ascii_digit() || c == '.') {
                tokens.push(SpecialToken::VersionNumber(term.clone()));
            }
        }

        for term in terms {
            if let Some(dash_pos) = term.find('-') {
                let left = term[..dash_pos].trim_start_matches('$');
                let right = term[dash_pos + 1..].trim_start_matches('$');
                if let (Ok(min), Ok(max)) = (left.parse::<f64>(), right.parse::<f64>()) {
                    tokens.push(SpecialToken::NumericRange { min, max });
                }
            }
        }

        tokens
    }

    /// Rule-based intent classification.
    fn classify_intent(&self, terms: &[String]) -> (QueryIntent, f32) {
        let nav_signals = terms.iter().any(|t| {
            t.contains('.') && (t.contains("com") || t.contains("org") || t.contains("net"))
        }) || terms.iter().any(|t| {
            matches!(t.as_str(), "go" | "goto" | "navigate" | "site" | "login" | "homepage")
        });

        if nav_signals {
            return (QueryIntent::Navigational, 0.85);
        }

        let local_signals = terms.iter().any(|t| {
            matches!(t.as_str(), "near" | "nearby" | "directions" | "map" | "local")
        });

        if local_signals {
            return (QueryIntent::Local, 0.80);
        }

        let transactional_signals = terms.iter().any(|t| {
            matches!(
                t.as_str(),
                "buy" | "price" | "cheap" | "deal" | "order" | "purchase"
                    | "download" | "free" | "coupon" | "discount" | "shop"
            )
        });

        if transactional_signals {
            return (QueryIntent::Transactional, 0.75);
        }

        let informational_signals = terms.iter().any(|t| {
            matches!(
                t.as_str(),
                "how" | "what" | "why" | "when" | "where" | "who"
                    | "which" | "define" | "meaning" | "tutorial" | "guide"
                    | "explain" | "difference" | "vs" | "versus" | "compare"
            )
        });

        let confidence = if informational_signals { 0.80 } else { 0.55 };
        (QueryIntent::Informational, confidence)
    }

    /// Synonym expansion from a static dictionary.
    fn expand_synonyms(&self, terms: &[String]) -> Vec<(String, String, f32)> {
        let mut synonyms = Vec::new();
        let weight = self.config.synonym_weight;

        for term in terms {
            let syns = match term.as_str() {
                "fast" => vec!["quick", "rapid", "speedy"],
                "big" => vec!["large", "huge", "enormous"],
                "small" => vec!["tiny", "little", "compact"],
                "cheap" => vec!["affordable", "inexpensive", "budget"],
                "good" => vec!["great", "excellent", "quality"],
                "bad" => vec!["poor", "terrible", "awful"],
                "fix" => vec!["repair", "solve", "resolve"],
                "error" => vec!["bug", "issue", "problem"],
                _ => vec![],
            };

            for syn in syns {
                synonyms.push((term.clone(), syn.to_string(), weight));
            }
        }

        synonyms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> QueryConfig {
        QueryConfig {
            fast_path_budget_ms: 10,
            enhanced_path_budget_ms: 200,
            enhanced_path_fallback_ms: 400,
            intent_confidence_threshold: 0.7,
            enhanced_path_token_threshold: 8,
            llm_cache_size: 100,
            spell_max_edit_short: 1,
            spell_max_edit_long: 2,
            synonym_weight: 0.3,
        }
    }

    #[test]
    fn test_navigational_intent() {
        let engine = QueryEngine::new(&make_config());
        let parsed = engine.parse_query("github.com login");
        assert_eq!(parsed.intent, QueryIntent::Navigational);
    }

    #[test]
    fn test_informational_intent() {
        let engine = QueryEngine::new(&make_config());
        let parsed = engine.parse_query("how does rust borrow checker work");
        assert_eq!(parsed.intent, QueryIntent::Informational);
    }

    #[test]
    fn test_phrase_detection() {
        let engine = QueryEngine::new(&make_config());
        let parsed = engine.parse_query("\"exact phrase\" other words");
        assert_eq!(parsed.phrases, vec!["exact phrase".to_string()]);
    }

    #[test]
    fn test_edit_distance() {
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("rust", "rust"), 0);
        assert_eq!(edit_distance("rust", "ruts"), 2);
        assert_eq!(edit_distance("rust", "rust"), 0);
        assert_eq!(edit_distance("teh", "the"), 2);
    }

    #[test]
    fn test_spell_dictionary_basic() {
        let mut dict = SpellDictionary::new(2);
        dict.add_word("the", 1000000);
        dict.add_word("there", 500000);
        dict.add_word("their", 400000);
        dict.add_word("they", 300000);
        dict.add_word("programming", 100000);

        // "teh" should correct to "the" (edit distance 2, "the" has high freq).
        let result = dict.correct("teh", 2);
        assert_eq!(result, Some("the".to_string()));

        // "programming" is already correct — no correction.
        let result = dict.correct("programming", 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_spell_dictionary_edit_distance_1() {
        let mut dict = SpellDictionary::new(2);
        dict.add_word("python", 500000);
        dict.add_word("rust", 400000);
        dict.add_word("java", 300000);

        // "pythn" → "python" (1 delete).
        let result = dict.correct("pythn", 1);
        assert_eq!(result, Some("python".to_string()));

        // "rudt" is edit distance 2 from "rust", should not correct at max_dist=1.
        let result = dict.correct("rudt", 1);
        assert!(result.is_none());
    }

    #[test]
    fn test_spell_correction_in_pipeline() {
        let engine = QueryEngine::new(&make_config());

        // Populate dictionary with some terms.
        engine.populate_spell_dictionary(&[
            ("programming".to_string(), 100000),
            ("language".to_string(), 80000),
            ("python".to_string(), 90000),
            ("rust".to_string(), 70000),
        ]);

        // "programing" → "programming" (missing 'm').
        let parsed = engine.parse_query("programing language");
        assert!(
            parsed.corrected.is_some(),
            "Should have suggested a correction"
        );
        if let Some(ref c) = parsed.corrected {
            assert!(c.contains("programming"), "Correction: {}", c);
        }
    }

    #[test]
    fn test_no_correction_when_dict_empty() {
        let engine = QueryEngine::new(&make_config());
        let parsed = engine.parse_query("teh quick brown fox");
        assert!(parsed.corrected.is_none());
    }
}
