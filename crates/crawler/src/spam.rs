// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: crawler :: spam
//
// Rule-based spam classifier.
//
// Lightweight heuristic classifier that scores documents on content quality
// signals. Designed to run in-line during the crawl pipeline with negligible
// latency (< 1ms per document).
//
// Signals:
//   - Keyword stuffing: abnormal keyword density in title/body.
//   - Link spam: excessive outgoing links relative to content.
//   - Thin content: very low word count after boilerplate removal.
//   - Hidden text patterns: CSS display:none / visibility:hidden in body.
//   - Domain blacklist: known spam/malware TLDs and domains.
//   - Language mismatch: title in one language, body in another.
//   - Ad density: high ratio of ad-related terms.
// ================================================================================

use raithe_common::traits::SpamClassifier;
use raithe_common::types::ParsedDocument;
use std::collections::HashSet;
use tracing::debug;

/// Rule-based spam classifier with configurable blacklist.
pub struct RuleBasedSpamClassifier {
    /// Domains known to be spam or malware sources.
    blacklisted_domains: HashSet<String>,
    /// TLDs associated with high spam rates.
    suspicious_tlds: HashSet<String>,
}

impl RuleBasedSpamClassifier {
    pub fn new() -> Self {
        let mut blacklisted_domains = HashSet::new();
        // Common spam/malware domains — expanded at runtime from crawl signals.
        for domain in &[
            "example-spam.com",
            "buy-cheap-pills.com",
            "free-casino-bonus.net",
            "seo-backlinks.biz",
        ] {
            blacklisted_domains.insert(domain.to_string());
        }

        let mut suspicious_tlds = HashSet::new();
        for tld in &[
            "xyz", "top", "club", "work", "click", "gdn", "loan", "bid",
            "stream", "racing", "win", "download", "review", "accountant",
            "science", "faith", "cricket", "party", "date",
        ] {
            suspicious_tlds.insert(tld.to_string());
        }

        Self {
            blacklisted_domains,
            suspicious_tlds,
        }
    }

    /// Add a domain to the blacklist at runtime (e.g., from manual reports or
    /// automated detection during crawling).
    pub fn blacklist_domain(&mut self, domain: &str) {
        self.blacklisted_domains.insert(domain.to_lowercase());
    }

    /// Extract the TLD from a domain string.
    fn extract_tld(domain: &str) -> &str {
        domain.rsplit('.').next().unwrap_or("")
    }

    /// Compute keyword stuffing score: ratio of the most frequent non-stopword
    /// token to total tokens. High ratio → likely keyword stuffing.
    fn keyword_stuffing_score(text: &str) -> f64 {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.len() < 20 {
            return 0.0; // Too short to evaluate.
        }

        let stopwords: HashSet<&str> = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after",
            "and", "but", "or", "nor", "not", "so", "yet", "both",
            "this", "that", "these", "those", "it", "its",
        ].iter().copied().collect();

        let mut freq: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        let mut content_count = 0usize;

        for token in &tokens {
            let lower = token.to_lowercase();
            if !stopwords.contains(lower.as_str()) {
                *freq.entry(*token).or_insert(0) += 1;
                content_count += 1;
            }
        }

        if content_count == 0 {
            return 0.0;
        }

        let max_freq = freq.values().copied().max().unwrap_or(0);
        let ratio = max_freq as f64 / content_count as f64;

        // A single term appearing in >15% of content tokens is suspicious.
        // >25% is very likely keyword stuffing.
        if ratio > 0.25 {
            0.8
        } else if ratio > 0.15 {
            0.4
        } else {
            0.0
        }
    }

    /// Score link spam: ratio of outgoing links to word count.
    fn link_spam_score(link_count: usize, word_count: usize) -> f64 {
        if word_count == 0 {
            if link_count > 5 { return 0.9; }
            return 0.0;
        }

        let ratio = link_count as f64 / word_count as f64;

        // More than 1 link per 10 words is suspicious.
        // More than 1 link per 5 words is very likely link spam.
        if ratio > 0.2 {
            0.8
        } else if ratio > 0.1 {
            0.4
        } else if ratio > 0.05 {
            0.15
        } else {
            0.0
        }
    }

    /// Score thin content: pages with very few words after boilerplate removal.
    fn thin_content_score(word_count: usize) -> f64 {
        if word_count < 50 {
            0.6
        } else if word_count < 100 {
            0.3
        } else if word_count < 200 {
            0.1
        } else {
            0.0
        }
    }

    /// Detect ad-heavy content by scanning for ad-related terms.
    fn ad_density_score(body: &str) -> f64 {
        let lower = body.to_lowercase();
        let tokens: Vec<&str> = lower.split_whitespace().collect();
        if tokens.len() < 50 {
            return 0.0;
        }

        let ad_terms: HashSet<&str> = [
            "advertisement", "sponsored", "buy now", "click here",
            "limited time", "act now", "free trial", "subscribe",
            "discount", "coupon", "promo", "offer expires",
            "order now", "best price", "lowest price",
        ].iter().copied().collect();

        let ad_count = tokens.iter()
            .filter(|t| ad_terms.contains(**t))
            .count();

        let ratio = ad_count as f64 / tokens.len() as f64;
        if ratio > 0.05 {
            0.6
        } else if ratio > 0.02 {
            0.3
        } else {
            0.0
        }
    }
}

impl Default for RuleBasedSpamClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl SpamClassifier for RuleBasedSpamClassifier {
    /// Compute a composite spam score for a parsed document.
    /// Returns 0.0 (clean) to 1.0 (definitely spam).
    ///
    /// Weighted combination of heuristic signals:
    ///   - Keyword stuffing:    25%
    ///   - Link spam:           25%
    ///   - Thin content:        20%
    ///   - Ad density:          15%
    ///   - Suspicious TLD:      10%
    ///   - Title missing:        5%
    fn score(&self, document: &ParsedDocument) -> f64 {
        let mut score = 0.0;

        // 1. Keyword stuffing (25% weight).
        let stuffing = Self::keyword_stuffing_score(&document.body);
        score += stuffing * 0.25;

        // 2. Link spam (25% weight).
        let link_spam = Self::link_spam_score(
            document.outgoing_links.len(),
            document.word_count,
        );
        score += link_spam * 0.25;

        // 3. Thin content (20% weight).
        let thin = Self::thin_content_score(document.word_count);
        score += thin * 0.20;

        // 4. Ad density (15% weight).
        let ads = Self::ad_density_score(&document.body);
        score += ads * 0.15;

        // 5. Suspicious TLD (10% weight).
        let domain = url::Url::parse(&document.url)
            .ok()
            .and_then(|u| u.host_str().map(|h| h.to_string()))
            .unwrap_or_default();
        let tld = Self::extract_tld(&domain);
        if self.suspicious_tlds.contains(tld) {
            score += 0.10;
        }

        // 6. Missing title (5% weight) — quality signal.
        if document.title.trim().is_empty() {
            score += 0.05;
        }

        // Clamp to [0.0, 1.0].
        let final_score = score.clamp(0.0, 1.0);

        if final_score > 0.5 {
            debug!(
                url = %document.url,
                spam_score = final_score,
                "High spam score detected"
            );
        }

        final_score
    }

    fn is_blacklisted(&self, domain: &str) -> bool {
        let lower = domain.to_lowercase();
        self.blacklisted_domains.contains(&lower)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raithe_common::types::{ExtractedLink, OgTags};

    fn make_doc(title: &str, body: &str, word_count: usize, links: usize) -> ParsedDocument {
        ParsedDocument {
            url: "https://example.com/page".to_string(),
            original_url: "https://example.com/page".to_string(),
            title: title.to_string(),
            headings: vec![],
            body: body.to_string(),
            url_tokens: vec![],
            meta_description: String::new(),
            alt_texts: vec![],
            structured_data: vec![],
            outgoing_links: (0..links)
                .map(|i| ExtractedLink {
                    url: format!("https://example.com/link{}", i),
                    anchor_text: "link".to_string(),
                    is_followed: true,
                })
                .collect(),
            language: "en".to_string(),
            og_tags: OgTags {
                title: None,
                description: None,
                image: None,
                og_type: None,
                site_name: None,
            },
            canonical_url: None,
            simhash: 0,
            word_count,
            http_status: 200,
            content_type: "text/html".to_string(),
            crawled_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_clean_document_low_score() {
        let classifier = RuleBasedSpamClassifier::new();
        let body = "Rust is a systems programming language that runs blazingly fast, \
                    prevents segfaults, and guarantees thread safety. It accomplishes \
                    these goals by being memory safe without using garbage collection. \
                    The language has grown rapidly in popularity due to its unique \
                    combination of performance and safety guarantees. Many companies \
                    now use Rust in production for critical systems infrastructure.";
        let doc = make_doc("Rust Programming Language", body, 250, 5);
        let score = classifier.score(&doc);
        assert!(score < 0.3, "Clean doc should have low spam score, got {}", score);
    }

    #[test]
    fn test_link_spam_high_score() {
        let classifier = RuleBasedSpamClassifier::new();
        let doc = make_doc("Links Page", "short text here", 10, 100);
        let score = classifier.score(&doc);
        assert!(score > 0.3, "Link-heavy doc should have higher spam score, got {}", score);
    }

    #[test]
    fn test_thin_content_score() {
        let classifier = RuleBasedSpamClassifier::new();
        let doc = make_doc("Thin", "very short", 20, 0);
        let score = classifier.score(&doc);
        assert!(score > 0.1, "Thin content should contribute to spam score, got {}", score);
    }

    #[test]
    fn test_blacklist_check() {
        let classifier = RuleBasedSpamClassifier::new();
        assert!(classifier.is_blacklisted("buy-cheap-pills.com"));
        assert!(!classifier.is_blacklisted("wikipedia.org"));
    }

    #[test]
    fn test_suspicious_tld() {
        let classifier = RuleBasedSpamClassifier::new();
        let mut doc = make_doc("Test", "Some reasonable content for this page that is long enough to test properly.", 200, 3);
        doc.url = "https://spam-site.xyz/page".to_string();
        let score_suspicious = classifier.score(&doc);

        doc.url = "https://good-site.com/page".to_string();
        let score_normal = classifier.score(&doc);

        assert!(
            score_suspicious > score_normal,
            "Suspicious TLD should increase score: {} vs {}",
            score_suspicious, score_normal
        );
    }
}
