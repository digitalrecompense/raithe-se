// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: session
//
// Session context tracking.
//
// In-memory LRU via moka (max 1000 sessions, TTL 30 minutes).
// Tracks last 5 queries per session to detect reformulation patterns.
// At Tier 1, this swaps in Redis. Same SessionStore trait interface.
// ================================================================================

use raithe_common::config::SessionConfig;
use raithe_common::traits::{AsyncResult, SessionStore};
use raithe_common::types::SessionContext;
use moka::sync::Cache;
use std::sync::Arc;
use std::time::Duration;
use tracing::debug;
use uuid::Uuid;

/// In-memory session store backed by moka LRU cache.
pub struct InMemorySessionStore {
    cache: Cache<String, Arc<SessionContext>>,
    max_queries: usize,
}

impl InMemorySessionStore {
    pub fn new(config: &SessionConfig) -> Self {
        let cache = Cache::builder()
            .max_capacity(config.max_sessions)
            .time_to_live(Duration::from_secs(config.session_ttl_secs))
            .build();

        Self {
            cache,
            max_queries: config.queries_per_session,
        }
    }

    /// Generate a new session ID.
    pub fn new_session_id() -> String {
        Uuid::new_v4().to_string()
    }
}

impl SessionStore for InMemorySessionStore {
    fn get_session(&self, session_id: &str) -> AsyncResult<'_, Option<SessionContext>> {
        let session_id = session_id.to_string();
        Box::pin(async move {
            Ok(self.cache.get(&session_id).map(|arc| (*arc).clone()))
        })
    }

    fn update_session(
        &self,
        session_id: &str,
        query: &str,
    ) -> AsyncResult<'_, SessionContext> {
        let session_id = session_id.to_string();
        let query = query.to_string();
        let max_queries = self.max_queries;

        Box::pin(async move {
            let existing = self.cache.get(&session_id);

            let mut previous_queries = existing
                .as_ref()
                .map(|s| s.previous_queries.clone())
                .unwrap_or_default();

            // Detect reformulation: is the new query similar to the previous one?
            let is_reformulation = previous_queries
                .first()
                .map(|prev| is_likely_reformulation(prev, &query))
                .unwrap_or(false);

            // Prepend new query, keep only last N.
            previous_queries.insert(0, query.clone());
            previous_queries.truncate(max_queries);

            let context = SessionContext {
                session_id: session_id.clone(),
                previous_queries,
                query_similarities: Vec::new(), // Filled by semantic layer.
                is_reformulation,
                topic_cluster_id: None, // Filled by query understanding.
            };

            self.cache.insert(session_id, Arc::new(context.clone()));

            debug!(
                session = %context.session_id,
                query_count = context.previous_queries.len(),
                is_reformulation = is_reformulation,
                "Session updated"
            );

            Ok(context)
        })
    }
}

/// Simple heuristic to detect query reformulation.
/// True if queries share > 50% of tokens.
fn is_likely_reformulation(previous: &str, current: &str) -> bool {
    let prev_tokens: std::collections::HashSet<&str> =
        previous.split_whitespace().collect();
    let curr_tokens: std::collections::HashSet<&str> =
        current.split_whitespace().collect();

    if prev_tokens.is_empty() || curr_tokens.is_empty() {
        return false;
    }

    let overlap = prev_tokens.intersection(&curr_tokens).count();
    let max_len = prev_tokens.len().max(curr_tokens.len());

    (overlap as f64 / max_len as f64) > 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reformulation_detection() {
        assert!(is_likely_reformulation(
            "rust async programming",
            "rust async await programming"
        ));
        assert!(!is_likely_reformulation(
            "rust programming",
            "cooking recipes"
        ));
    }
}
