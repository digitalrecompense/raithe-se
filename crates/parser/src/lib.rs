// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: parser
//
// HTML parsing, boilerplate removal, and structural feature extraction
// Quality of content extraction directly impacts ranking quality.
//
// Primary parser: lol_html (Cloudflare's streaming rewriter) — single pass,
// O(1) memory. Fallback: scraper (html5ever) for DOM manipulation.
// ================================================================================

pub mod boilerplate;
pub mod extractor;

use raithe_common::simhash::compute_simhash;
use raithe_common::types::{ExtractedLink, Heading, OgTags, ParsedDocument};
use chrono::Utc;
use url::Url;

/// Parse raw HTML into a structured ParsedDocument.
///
/// This is the main entry point for content extraction.
/// Extracts: title, meta description, OG tags, canonical URL, language,
/// outgoing links, structured data (JSON-LD), headings, body text after
/// boilerplate removal, alt text, and SimHash fingerprint.
pub fn parse_html(
    raw_html: &[u8],
    source_url: &str,
    http_status: u16,
    content_type: &str,
) -> Result<ParsedDocument, String> {
    let html_str = String::from_utf8_lossy(raw_html);
    let base_url = Url::parse(source_url).map_err(|e| format!("Invalid URL: {}", e))?;

    // Use scraper for full DOM access (fallback path).
    // For documents under 5MB this is acceptable.
    let document = scraper::Html::parse_document(&html_str);

    let title = extract_title(&document);

    let meta_description = extract_meta(&document, "description");

    // Extract Open Graph tags.
    let og_tags = extract_og_tags(&document);

    // Extract canonical URL.
    let canonical_url = extract_canonical(&document);

    // Detect language.
    let language = extract_language(&document);

    // Extract all headings (weight 3.0x h1 to 1.2x h6).
    let headings = extract_headings(&document);

    // Extract outgoing links with normalization.
    let outgoing_links = extract_links(&document, &base_url);

    // Extract structured data (JSON-LD, weight 1.5x).
    let structured_data = extract_json_ld(&document);

    // Extract alt texts from images (weight 0.8x).
    let alt_texts = extract_alt_texts(&document);

    // Boilerplate removal and main content extraction.
    let body = boilerplate::extract_main_content(&document, &title);

    // Tokenize URL path segments (weight 2.0x).
    let url_tokens = tokenize_url_path(&base_url);

    // Word count.
    let word_count = body.split_whitespace().count();

    // SimHash fingerprint.
    let simhash = compute_simhash(body.split_whitespace());

    Ok(ParsedDocument {
        url: source_url.to_string(),
        original_url: source_url.to_string(),
        title,
        headings,
        body,
        url_tokens,
        meta_description,
        alt_texts,
        structured_data,
        outgoing_links,
        language,
        og_tags,
        canonical_url,
        simhash,
        word_count,
        crawled_at: Utc::now(),
        http_status,
        content_type: content_type.to_string(),
    })
}

// -- Extraction helpers --

fn extract_title(doc: &scraper::Html) -> String {
    let selector = scraper::Selector::parse("title").unwrap();
    doc.select(&selector)
        .next()
        .map(|el| {
            let raw = el.text().collect::<String>();
            // Clean site name suffix (e.g., " - Example.com", " | Site Name").
            let cleaned = raw
                .split(&['-', '|', '—', '–'][..])
                .next()
                .unwrap_or(&raw)
                .trim()
                .to_string();
            cleaned
        })
        .unwrap_or_default()
}

fn extract_meta(doc: &scraper::Html, name: &str) -> String {
    let selector = scraper::Selector::parse(&format!("meta[name='{}']", name)).unwrap();
    doc.select(&selector)
        .next()
        .and_then(|el| el.value().attr("content"))
        .unwrap_or("")
        .to_string()
}

fn extract_og_tags(doc: &scraper::Html) -> OgTags {
    let get_og = |property: &str| -> Option<String> {
        let sel = scraper::Selector::parse(&format!("meta[property='og:{}']", property)).ok()?;
        doc.select(&sel)
            .next()
            .and_then(|el| el.value().attr("content"))
            .map(String::from)
    };

    OgTags {
        title: get_og("title"),
        description: get_og("description"),
        image: get_og("image"),
        og_type: get_og("type"),
        site_name: get_og("site_name"),
    }
}

fn extract_canonical(doc: &scraper::Html) -> Option<String> {
    let selector = scraper::Selector::parse("link[rel='canonical']").ok()?;
    doc.select(&selector)
        .next()
        .and_then(|el| el.value().attr("href"))
        .map(String::from)
}

fn extract_language(doc: &scraper::Html) -> String {
    let selector = scraper::Selector::parse("html").unwrap();
    doc.select(&selector)
        .next()
        .and_then(|el| el.value().attr("lang"))
        .unwrap_or("en")
        .to_string()
}

fn extract_headings(doc: &scraper::Html) -> Vec<Heading> {
    let mut headings = Vec::new();
    for level in 1..=6u8 {
        let selector = scraper::Selector::parse(&format!("h{}", level)).unwrap();
        for el in doc.select(&selector) {
            let text = el.text().collect::<String>().trim().to_string();
            if !text.is_empty() {
                headings.push(Heading { level, text });
            }
        }
    }
    headings
}

fn extract_links(doc: &scraper::Html, base_url: &Url) -> Vec<ExtractedLink> {
    let selector = scraper::Selector::parse("a[href]").unwrap();
    let mut links = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for el in doc.select(&selector) {
        let href = match el.value().attr("href") {
            Some(h) => h,
            None => continue,
        };

        // Normalize the URL.
        let resolved = match base_url.join(href) {
            Ok(u) => u.to_string(),
            Err(_) => continue,
        };

        if seen.contains(&resolved) {
            continue;
        }
        seen.insert(resolved.clone());

        let anchor_text = el.text().collect::<String>().trim().to_string();
        let is_followed = el.value().attr("rel").map_or(true, |rel| {
            !rel.to_lowercase().contains("nofollow")
        });

        links.push(ExtractedLink {
            url: resolved,
            anchor_text,
            is_followed,
        });
    }

    links
}

fn extract_json_ld(doc: &scraper::Html) -> Vec<String> {
    let selector = scraper::Selector::parse("script[type='application/ld+json']").unwrap();
    doc.select(&selector)
        .map(|el| el.text().collect::<String>().trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn extract_alt_texts(doc: &scraper::Html) -> Vec<String> {
    let selector = scraper::Selector::parse("img[alt]").unwrap();
    doc.select(&selector)
        .filter_map(|el| el.value().attr("alt"))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn tokenize_url_path(url: &Url) -> Vec<String> {
    url.path_segments()
        .map(|segments| {
            segments
                .flat_map(|seg| {
                    seg.split(&['-', '_', '.'][..])
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_lowercase())
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_html() {
        let html = r#"
            <html lang="en">
            <head>
                <title>Test Page - Example Site</title>
                <meta name="description" content="A test page for RAiTHE">
            </head>
            <body>
                <h1>Welcome to the Test</h1>
                <p>This is the main content of the test page with enough words to pass the minimum threshold.</p>
                <a href="/other-page">Link to other page</a>
            </body>
            </html>
        "#;

        let doc = parse_html(
            html.as_bytes(),
            "https://example.com/test-page",
            200,
            "text/html",
        )
        .unwrap();

        assert_eq!(doc.title, "Test Page");
        assert_eq!(doc.meta_description, "A test page for RAiTHE");
        assert_eq!(doc.language, "en");
        assert!(!doc.headings.is_empty());
        assert_eq!(doc.headings[0].level, 1);
        assert!(!doc.outgoing_links.is_empty());
        assert!(doc.url_tokens.contains(&"test".to_string()));
        assert!(doc.url_tokens.contains(&"page".to_string()));
    }
}
