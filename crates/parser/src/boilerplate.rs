// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: parser :: boilerplate
//
// Boilerplate removal.
// Hybrid approach: DOM density analysis, block fusion, content block
// classification, and title-content coherence scoring.
// ================================================================================

use scraper::{ElementRef, Html, Selector};
use std::collections::HashSet;

/// Boilerplate indicator class/id patterns.
const BOILERPLATE_PATTERNS: &[&str] = &[
    "sidebar", "nav", "navigation", "footer", "header", "menu", "comment",
    "ad", "advertisement", "widget", "social", "share", "related", "popup",
    "modal", "cookie", "banner", "promo", "sponsor",
];

/// Content indicator tags.
const CONTENT_TAGS: &[&str] = &["article", "main", "section"];

/// Extract the main content text from an HTML document after boilerplate removal.
///
/// Strategy:
/// 1. Prefer content within <article> or <main> tags.
/// 2. DOM density analysis: compute text-to-tag ratio per subtree.
/// 3. Filter out subtrees with boilerplate class/id patterns.
/// 4. Score candidate blocks by TF overlap with the title.
/// 5. Fuse adjacent text blocks.
pub fn extract_main_content(doc: &Html, title: &str) -> String {
    // Step 1: Try to find content within <article> or <main> tags.
    for tag in CONTENT_TAGS {
        if let Ok(selector) = Selector::parse(tag) {
            for element in doc.select(&selector) {
                let text = extract_text_recursive(&element);
                if text.split_whitespace().count() >= 50 {
                    return text;
                }
            }
        }
    }

    // Step 2: Fall back to body and filter by density/boilerplate heuristics.
    if let Ok(body_sel) = Selector::parse("body") {
        if let Some(body) = doc.select(&body_sel).next() {
            return extract_filtered_text(&body, title);
        }
    }

    // Last resort: all text.
    doc.root_element()
        .text()
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Extract text from an element recursively, joining with spaces.
fn extract_text_recursive(element: &ElementRef) -> String {
    element
        .text()
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Extract text from the body, filtering out boilerplate elements.
fn extract_filtered_text(body: &ElementRef, title: &str) -> String {
    let title_words: HashSet<String> = title
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();

    // Collect text blocks from <p>, <div>, and text-heavy elements,
    // skipping those inside boilerplate containers.
    let mut blocks: Vec<(String, f64)> = Vec::new();

    // Extract paragraphs.
    if let Ok(p_sel) = Selector::parse("p") {
        for p in body.select(&p_sel) {
            if is_inside_boilerplate(&p) {
                continue;
            }

            let text = extract_text_recursive(&p);
            let word_count = text.split_whitespace().count();

            if word_count < 5 {
                continue;
            }

            // Compute text-to-tag ratio.
            let tag_count = count_child_tags(&p).max(1) as f64;
            let density = word_count as f64 / tag_count;

            // Score by title overlap.
            let overlap = text
                .split_whitespace()
                .filter(|w| title_words.contains(&w.to_lowercase()))
                .count() as f64
                / title_words.len().max(1) as f64;

            let score = density + overlap * 2.0;

            blocks.push((text, score));
        }
    }

    // Sort by score descending, take blocks above threshold.
    blocks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Fuse adjacent blocks (block fusion).
    let content: String = blocks
        .iter()
        .filter(|(_, score)| *score > 0.5)
        .map(|(text, _)| text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    if content.split_whitespace().count() >= 20 {
        content
    } else {
        // Fallback: use all paragraph text.
        blocks
            .iter()
            .map(|(text, _)| text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Check if an element is inside a boilerplate container.
fn is_inside_boilerplate(element: &ElementRef) -> bool {
    let mut current = Some(*element);
    while let Some(el) = current {
        let value = el.value();
        let classes = value.attr("class").unwrap_or("");
        let id = value.attr("id").unwrap_or("");

        let combined = format!("{} {}", classes.to_lowercase(), id.to_lowercase());

        for pattern in BOILERPLATE_PATTERNS {
            if combined.contains(pattern) {
                return true;
            }
        }

        // Check tag name.
        let tag = value.name();
        if tag == "nav" || tag == "footer" || tag == "header" || tag == "aside" {
            return true;
        }

        current = el.parent().and_then(ElementRef::wrap);
    }

    false
}

/// Count direct child tags of an element.
fn count_child_tags(element: &ElementRef) -> usize {
    element
        .children()
        .filter(|child| child.value().is_element())
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_article_content() {
        let html = r#"
            <html>
            <body>
                <nav>Navigation menu items here</nav>
                <article>
                    <p>This is the main article content with plenty of words to make it past the threshold. raithe-se processes this text for indexing and ranking purposes. Quality content extraction is essential for good search results.</p>
                </article>
                <footer>Footer content here</footer>
            </body>
            </html>
        "#;
        let doc = Html::parse_document(html);
        let content = extract_main_content(&doc, "Test Article");

        assert!(content.contains("main article content"));
        assert!(!content.contains("Navigation"));
        assert!(!content.contains("Footer"));
    }
}
