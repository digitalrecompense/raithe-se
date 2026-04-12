// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: crawler :: robots
//
// robots.txt parsing and caching.
// LRU cache via moka: 1,000,000 entries, TTL 24h.
// Default crawl delay: 1 req/sec/domain. Honor Crawl-delay directives.
// Exponential backoff on 429/503: base 5s, max 1h, jitter ±20%.
// ================================================================================

use moka::sync::Cache;
use std::time::Duration;
use tracing::debug;

/// Parsed robots.txt rules for a single domain.
#[derive(Debug, Clone)]
pub struct RobotsRules {
    /// Disallowed path prefixes for our user-agent.
    pub disallowed: Vec<String>,
    /// Allowed path prefixes (overrides disallow).
    pub allowed: Vec<String>,
    /// Crawl delay in milliseconds (0 if not specified).
    pub crawl_delay_ms: u64,
    /// Sitemap URLs found in robots.txt.
    pub sitemaps: Vec<String>,
}

impl RobotsRules {
    /// Check if a path is allowed by these rules.
    /// Uses longest-match-wins: the rule with the longest matching prefix
    /// determines the outcome. On tie, Allow wins.
    pub fn is_allowed(&self, path: &str) -> bool {
        let mut best_match_len: usize = 0;
        let mut best_is_allow = true; // Default: allowed.

        for allow in &self.allowed {
            if path.starts_with(allow) && allow.len() >= best_match_len {
                best_match_len = allow.len();
                best_is_allow = true;
            }
        }

        for disallow in &self.disallowed {
            if disallow.is_empty() {
                continue;
            }
            if path.starts_with(disallow) && disallow.len() > best_match_len {
                best_match_len = disallow.len();
                best_is_allow = false;
            }
        }

        best_is_allow
    }

    /// Parse a robots.txt body for the "raithe" and "*" user-agents.
    pub fn parse(body: &str) -> Self {
        let mut disallowed = Vec::new();
        let mut allowed = Vec::new();
        let mut crawl_delay_ms: u64 = 0;
        let mut sitemaps = Vec::new();
        let mut relevant_section = false;
        let mut seen_raithe = false;

        for line in body.lines() {
            let line = line.trim();

            // Skip comments and empty lines.
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let lower = line.to_lowercase();

            if lower.starts_with("user-agent:") {
                let agent = line[11..].trim().to_lowercase();
                if agent == "raithe" || agent == "raithebot" {
                    if !seen_raithe {
                        disallowed.clear();
                        allowed.clear();
                        crawl_delay_ms = 0;
                    }
                    relevant_section = true;
                    seen_raithe = true;
                } else if agent == "*" && !seen_raithe {
                    relevant_section = true;
                } else {
                    relevant_section = false;
                }
            } else if relevant_section {
                if lower.starts_with("disallow:") {
                    let path = line[9..].trim().to_string();
                    if !path.is_empty() {
                        disallowed.push(path);
                    }
                } else if lower.starts_with("allow:") {
                    let path = line[6..].trim().to_string();
                    if !path.is_empty() {
                        allowed.push(path);
                    }
                } else if lower.starts_with("crawl-delay:") {
                    if let Ok(delay) = line[12..].trim().parse::<f64>() {
                        crawl_delay_ms = (delay * 1000.0) as u64;
                    }
                }
            }

            // Sitemaps are global, not section-specific.
            if lower.starts_with("sitemap:") {
                sitemaps.push(line[8..].trim().to_string());
            }
        }

        Self {
            disallowed,
            allowed,
            crawl_delay_ms,
            sitemaps,
        }
    }
}

/// LRU cache for robots.txt rules, keyed by domain.
pub struct RobotsCache {
    cache: Cache<String, RobotsRules>,
}

impl RobotsCache {
    /// Create a new cache with the given capacity and TTL.
    pub fn new(max_capacity: u64, ttl_secs: u64) -> Self {
        let cache = Cache::builder()
            .max_capacity(max_capacity)
            .time_to_live(Duration::from_secs(ttl_secs))
            .build();

        Self { cache }
    }

    /// Get cached robots.txt rules for a domain.
    pub fn get(&self, domain: &str) -> Option<RobotsRules> {
        self.cache.get(domain)
    }

    /// Insert parsed robots.txt rules for a domain.
    pub fn insert(&self, domain: String, rules: RobotsRules) {
        debug!(domain = %domain, "Cached robots.txt rules");
        self.cache.insert(domain, rules);
    }

    /// Check if a URL is allowed by the cached robots.txt rules.
    /// Returns None if the domain is not yet cached.
    pub fn is_allowed(&self, domain: &str, path: &str) -> Option<bool> {
        self.cache.get(domain).map(|rules| rules.is_allowed(path))
    }

    /// Get the crawl delay for a domain (in ms). Returns the default if not cached.
    pub fn crawl_delay_ms(&self, domain: &str, default_ms: u64) -> u64 {
        self.cache
            .get(domain)
            .map(|rules| {
                if rules.crawl_delay_ms > 0 {
                    rules.crawl_delay_ms
                } else {
                    default_ms
                }
            })
            .unwrap_or(default_ms)
    }
}

/// Compute exponential backoff delay for 429/503 responses.
/// Base: 5s, max: 1h, jitter: ±20%.
pub fn backoff_delay(attempt: u32) -> Duration {
    let base_ms: u64 = 5_000;
    let max_ms: u64 = 3_600_000; // 1 hour
    let delay_ms = base_ms.saturating_mul(1u64 << attempt.min(15)).min(max_ms);

    // Add jitter ±20%.
    let jitter_range = delay_ms / 5;
    let jitter = (rand::random::<u64>() % (jitter_range * 2)).saturating_sub(jitter_range);
    let final_ms = delay_ms.saturating_add(jitter);

    Duration::from_millis(final_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_robots_parse_basic() {
        let body = "User-agent: *\nDisallow: /admin/\nDisallow: /private/\nAllow: /admin/public/\nCrawl-delay: 2\n\nSitemap: https://example.com/sitemap.xml";
        let rules = RobotsRules::parse(body);

        assert!(!rules.is_allowed("/admin/settings"));
        assert!(rules.is_allowed("/admin/public/page"));
        assert!(rules.is_allowed("/about"));
        assert_eq!(rules.crawl_delay_ms, 2000);
        assert_eq!(rules.sitemaps.len(), 1);
    }

    #[test]
    fn test_robots_raithe_specific() {
        let body = "User-agent: *\nDisallow: /\n\nUser-agent: raithe\nDisallow: /secret/\nAllow: /";
        let rules = RobotsRules::parse(body);

        // raithe-specific rules should override wildcard.
        assert!(rules.is_allowed("/about"));
        assert!(!rules.is_allowed("/secret/page"));
    }
}
