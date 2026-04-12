// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: crawler :: fetcher
//
// HTTP fetch pipeline.
// DNS resolution -> TLS handshake -> HTTP fetch -> content extraction ->
// deduplication -> link extraction -> emit to crawl log.
// ================================================================================

use reqwest::Client;
use std::time::Duration;
use url::Url;

/// Build the shared HTTP client.
/// Uses reqwest with rustls, HTTP/2, connection pooling.
pub fn build_http_client(
    max_connections_per_host: usize,
    _total_connections: usize,
    timeout_secs: u64,
) -> reqwest::Result<Client> {
    Client::builder()
        .use_rustls_tls()
        .timeout(Duration::from_secs(timeout_secs))
        .pool_max_idle_per_host(max_connections_per_host)
        .pool_idle_timeout(Duration::from_secs(90))
        .redirect(reqwest::redirect::Policy::limited(5))
        .user_agent("RaitheBot/0.1 (+https://raithe.ca/bot)")
        .gzip(true)
        .brotli(true)
        .build()
}

/// Fetch a single URL and return the response body as bytes.
/// Enforces: max body size, content-type filter, timeout.
pub async fn fetch_url(
    client: &Client,
    url: &str,
    max_body_size: usize,
) -> anyhow::Result<FetchResult> {
    let response = client.get(url).send().await?;

    let status = response.status().as_u16();

    // Check content type — only process text/html, application/xhtml+xml, text/plain.
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_lowercase();

    let is_acceptable = content_type.contains("text/html")
        || content_type.contains("application/xhtml+xml")
        || content_type.contains("text/plain");

    if !is_acceptable && status == 200 {
        return Err(anyhow::anyhow!(
            "Unacceptable content type: {} for {}",
            content_type,
            url
        ));
    }

    // Stream the body with size limit.
    let body_bytes = response.bytes().await?;
    if body_bytes.len() > max_body_size {
        return Err(anyhow::anyhow!(
            "Response body too large: {} bytes (max: {}) for {}",
            body_bytes.len(),
            max_body_size,
            url
        ));
    }

    Ok(FetchResult {
        url: url.to_string(),
        status,
        content_type,
        body: body_bytes.to_vec(),
    })
}

/// Result of an HTTP fetch.
pub struct FetchResult {
    pub url: String,
    pub status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
}

/// Normalize a URL:
/// lowercase scheme and host, remove default ports, sort query parameters,
/// remove fragment identifiers, resolve relative URLs against base.
pub fn normalize_url(url_str: &str, base: Option<&Url>) -> Option<String> {
    let parsed = if let Some(base) = base {
        base.join(url_str).ok()?
    } else {
        Url::parse(url_str).ok()?
    };

    // Only crawl http/https.
    match parsed.scheme() {
        "http" | "https" => {}
        _ => return None,
    }

    let mut normalized = Url::parse(&format!(
        "{}://{}",
        parsed.scheme(),
        parsed.host_str()?,
    ))
    .ok()?;

    // Set port only if non-default.
    if let Some(port) = parsed.port() {
        let is_default = (parsed.scheme() == "http" && port == 80)
            || (parsed.scheme() == "https" && port == 443);
        if !is_default {
            let _ = normalized.set_port(Some(port));
        }
    }

    normalized.set_path(parsed.path());

    // Sort query parameters for canonical form.
    if let Some(_query) = parsed.query() {
        let mut params: Vec<(String, String)> = parsed.query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        params.sort();

        let sorted_query: String = params
            .iter()
            .map(|(k, v)| {
                if v.is_empty() {
                    k.clone()
                } else {
                    format!("{}={}", k, v)
                }
            })
            .collect::<Vec<_>>()
            .join("&");

        if !sorted_query.is_empty() {
            normalized.set_query(Some(&sorted_query));
        }
    }

    // Remove fragment identifiers.
    normalized.set_fragment(None);

    // Remove trailing slash for non-root paths.
    let path = normalized.path().to_string();
    if path.len() > 1 && path.ends_with('/') {
        normalized.set_path(&path[..path.len() - 1]);
    }

    Some(normalized.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_url_basic() {
        assert_eq!(
            normalize_url("HTTPS://Example.COM/Path?b=2&a=1#frag", None),
            Some("https://example.com/Path?a=1&b=2".to_string())
        );
    }

    #[test]
    fn test_normalize_url_default_port() {
        assert_eq!(
            normalize_url("https://example.com:443/page", None),
            Some("https://example.com/page".to_string())
        );
    }

    #[test]
    fn test_normalize_url_relative() {
        let base = Url::parse("https://example.com/dir/page.html").unwrap();
        assert_eq!(
            normalize_url("../other.html", Some(&base)),
            Some("https://example.com/other.html".to_string())
        );
    }
}
