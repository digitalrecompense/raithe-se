// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: instant
//
// Structured instant answers.
//
// Types:
//   Calculator — math expression evaluation via evalexpr.
//   Unit conversion — common units, currencies.
//   Dictionary — definitions from Wikidata/Wiktionary SQLite DB.
//   Entity panel — structured knowledge from Wikidata.
//   Date/Time — timezone conversions, day calculations.
//
// Confidence threshold: 0.8.
// Total instant answer budget: ≤ 20ms.
// ================================================================================

use raithe_common::config::InstantConfig;
use raithe_common::traits::{AsyncResult, InstantAnswerProvider};
use raithe_common::types::{InstantAnswer, InstantAnswerType, ParsedQuery};
use rusqlite::Connection;
use std::sync::Mutex;
use tracing::{debug, info, warn};
use chrono::{Datelike};

/// The instant answer engine that tries each provider in priority order.
pub struct InstantAnswerEngine {
    config: InstantConfig,
    /// SQLite connection for Wiktionary definitions.
    definitions_db: Option<Mutex<Connection>>,
    /// SQLite connection for Wikidata knowledge graph.
    knowledge_db: Option<Mutex<Connection>>,
}

impl InstantAnswerEngine {
    pub fn new(config: &InstantConfig) -> Self {
        // Open definitions database if it exists.
        let definitions_db = open_or_create_db(&config.definitions_db_path, |conn| {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS definitions (
                    word        TEXT PRIMARY KEY COLLATE NOCASE,
                    part_of_speech TEXT NOT NULL DEFAULT '',
                    definition  TEXT NOT NULL,
                    example     TEXT NOT NULL DEFAULT '',
                    etymology   TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_definitions_word
                    ON definitions (word COLLATE NOCASE);"
            ).ok();
        });

        // Open knowledge graph database if it exists.
        let knowledge_db = open_or_create_db(&config.knowledge_db_path, |conn| {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS entities (
                    name        TEXT PRIMARY KEY COLLATE NOCASE,
                    entity_type TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',
                    properties  TEXT NOT NULL DEFAULT '{}',
                    aliases     TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_entities_name
                    ON entities (name COLLATE NOCASE);
                CREATE INDEX IF NOT EXISTS idx_entities_aliases
                    ON entities (aliases COLLATE NOCASE);"
            ).ok();
        });

        let defs_status = if definitions_db.is_some() { "ready" } else { "unavailable" };
        let kg_status = if knowledge_db.is_some() { "ready" } else { "unavailable" };

        info!(
            "Instant answer engine initialized (confidence threshold: {}, \
             definitions DB: {}, knowledge DB: {})",
            config.confidence_threshold, defs_status, kg_status
        );

        Self {
            config: config.clone(),
            definitions_db,
            knowledge_db,
        }
    }

    /// Try to compute a calculator result.
    fn try_calculator(&self, query: &str) -> Option<InstantAnswer> {
        // Detect math expressions: contains digits and operators.
        let cleaned = query
            .replace("what is", "")
            .replace("calculate", "")
            .replace("compute", "")
            .replace("=", "")
            .trim()
            .to_string();

        if cleaned.is_empty() {
            return None;
        }

        // Use evalexpr to evaluate the expression.
        match evalexpr::eval(&cleaned) {
            Ok(value) => {
                let answer_str = match value {
                    evalexpr::Value::Int(i) => i.to_string(),
                    evalexpr::Value::Float(f) => {
                        if f.fract() == 0.0 && f.abs() < 1e15 {
                            format!("{}", f as i64)
                        } else {
                            format!("{:.6}", f).trim_end_matches('0').trim_end_matches('.').to_string()
                        }
                    }
                    _ => return None,
                };

                debug!(expression = %cleaned, result = %answer_str, "Calculator instant answer");

                Some(InstantAnswer {
                    answer_type: InstantAnswerType::Calculation,
                    answer: format!("{} = {}", cleaned, answer_str),
                    details: serde_json::json!({
                        "expression": cleaned,
                        "result": answer_str,
                    }),
                    source: "RAiTHE Calculator".to_string(),
                    confidence: 0.95,
                })
            }
            Err(_) => None,
        }
    }

    /// Try to detect and handle unit conversion queries.
    fn try_unit_conversion(&self, query: &str) -> Option<InstantAnswer> {
        let lower = query.to_lowercase();

        // Pattern: "<number> <unit> in/to <unit>"
        let parts: Vec<&str> = lower.split_whitespace().collect();

        // Find "in" or "to" keyword.
        let separator_idx = parts.iter().position(|&w| w == "in" || w == "to")?;

        if separator_idx < 2 || separator_idx >= parts.len() - 1 {
            return None;
        }

        let value: f64 = parts[0].parse().ok()?;
        let from_unit = parts[1..separator_idx].join(" ");
        let to_unit = parts[separator_idx + 1..].join(" ");

        let result = convert_units(value, &from_unit, &to_unit)?;

        Some(InstantAnswer {
            answer_type: InstantAnswerType::UnitConversion,
            answer: format!("{} {} = {:.4} {}", value, from_unit, result, to_unit),
            details: serde_json::json!({
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "result": result,
            }),
            source: "RAiTHE Unit Converter".to_string(),
            confidence: 0.90,
        })
    }

    /// Try to answer a date/time query.
    fn try_datetime(&self, query: &str) -> Option<InstantAnswer> {
        let lower = query.to_lowercase();

        // "what time is it" / "current time" / "what is the time"
        if lower.contains("what time") || lower.contains("current time")
            || lower.contains("time now") || lower == "time"
        {
            let now = chrono::Local::now();
            let answer = now.format("%I:%M %p, %A %B %e, %Y").to_string();
            return Some(InstantAnswer {
                answer_type: InstantAnswerType::DateTime,
                answer: answer.clone(),
                details: serde_json::json!({
                    "iso": now.to_rfc3339(),
                    "formatted": answer,
                }),
                source: "RAiTHE Clock".to_string(),
                confidence: 0.95,
            });
        }

        // "what day is it" / "today's date" / "current date" / "what is the date"
        if lower.contains("what day") || lower.contains("today") || lower.contains("current date")
            || lower.contains("what is the date") || lower == "date"
        {
            let now = chrono::Local::now();
            let answer = now.format("%A, %B %e, %Y").to_string();
            return Some(InstantAnswer {
                answer_type: InstantAnswerType::DateTime,
                answer: answer.clone(),
                details: serde_json::json!({
                    "iso": now.format("%Y-%m-%d").to_string(),
                    "formatted": answer,
                    "day_of_week": now.format("%A").to_string(),
                }),
                source: "RAiTHE Clock".to_string(),
                confidence: 0.95,
            });
        }

        // "days until <date>" / "days between <date> and <date>"
        if lower.contains("days until") || lower.contains("days to") {
            if let Some(target) = parse_natural_date(&lower) {
                let today = chrono::Local::now().date_naive();
                let diff = target.signed_duration_since(today).num_days();
                let answer = if diff > 0 {
                    format!("{} days from now ({})", diff, target.format("%B %e, %Y"))
                } else if diff == 0 {
                    format!("That's today! ({})", target.format("%B %e, %Y"))
                } else {
                    format!("{} days ago ({})", diff.abs(), target.format("%B %e, %Y"))
                };
                return Some(InstantAnswer {
                    answer_type: InstantAnswerType::DateTime,
                    answer,
                    details: serde_json::json!({
                        "days": diff,
                        "target_date": target.format("%Y-%m-%d").to_string(),
                    }),
                    source: "RAiTHE Calendar".to_string(),
                    confidence: 0.90,
                });
            }
        }

        // "days in <month>" / "days in <month> <year>"
        if lower.starts_with("days in ") || lower.starts_with("how many days in ") {
            let tail = lower
                .trim_start_matches("how many ")
                .trim_start_matches("days in ")
                .trim();
            if let Some(days) = days_in_month_query(tail) {
                return Some(InstantAnswer {
                    answer_type: InstantAnswerType::DateTime,
                    answer: format!("{} has {} days", capitalize(tail), days),
                    details: serde_json::json!({ "month": tail, "days": days }),
                    source: "RAiTHE Calendar".to_string(),
                    confidence: 0.92,
                });
            }
        }

        // "unix timestamp" / "epoch time"
        if lower.contains("unix timestamp") || lower.contains("epoch time")
            || lower.contains("unix time")
        {
            let now = chrono::Utc::now().timestamp();
            return Some(InstantAnswer {
                answer_type: InstantAnswerType::DateTime,
                answer: format!("Current Unix timestamp: {}", now),
                details: serde_json::json!({ "timestamp": now }),
                source: "RAiTHE Clock".to_string(),
                confidence: 0.95,
            });
        }

        None
    }

    /// Try to look up a dictionary definition.
    fn try_dictionary(&self, query: &str) -> Option<InstantAnswer> {
        let lower = query.to_lowercase();

        // Pattern: "define <word>", "definition of <word>", "meaning of <word>",
        // "what does <word> mean", "what is <word>"
        let word = if let Some(w) = lower.strip_prefix("define ") {
            w.trim()
        } else if let Some(w) = lower.strip_prefix("definition of ") {
            w.trim()
        } else if let Some(w) = lower.strip_prefix("meaning of ") {
            w.trim()
        } else if lower.starts_with("what does ") && lower.ends_with(" mean") {
            lower.trim_start_matches("what does ")
                .trim_end_matches(" mean")
                .trim()
        } else {
            return None;
        };

        if word.is_empty() || word.contains(' ') && word.split_whitespace().count() > 3 {
            return None;
        }

        // Look up in SQLite definitions database.
        if let Some(ref db) = self.definitions_db {
            if let Ok(conn) = db.lock() {
                let result = conn.query_row(
                    "SELECT word, part_of_speech, definition, example FROM definitions
                     WHERE word = ?1 COLLATE NOCASE LIMIT 1",
                    rusqlite::params![word],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                        ))
                    },
                );

                if let Ok((found_word, pos, definition, example)) = result {
                    let mut answer = format!("{}", definition);
                    if !pos.is_empty() {
                        answer = format!("({}) {}", pos, answer);
                    }

                    let mut details = serde_json::json!({
                        "word": found_word,
                        "definition": definition,
                    });
                    if !pos.is_empty() {
                        details["part_of_speech"] = serde_json::json!(pos);
                    }
                    if !example.is_empty() {
                        details["example"] = serde_json::json!(example);
                    }

                    debug!(word = %found_word, "Dictionary instant answer");

                    return Some(InstantAnswer {
                        answer_type: InstantAnswerType::Dictionary,
                        answer,
                        details,
                        source: "RAiTHE Dictionary".to_string(),
                        confidence: 0.88,
                    });
                }
            }
        }

        // Fallback: provide a built-in mini dictionary for common demo terms.
        lookup_builtin_definition(word)
    }

    /// Try to look up an entity knowledge panel.
    fn try_entity_panel(&self, query: &str) -> Option<InstantAnswer> {
        let lower = query.to_lowercase();

        // Pattern: "who is <entity>", "what is <entity>", "tell me about <entity>",
        // or a bare proper-noun-like query (capitalized words).
        let entity = if let Some(e) = lower.strip_prefix("who is ") {
            e.trim()
        } else if let Some(e) = lower.strip_prefix("who was ") {
            e.trim()
        } else if let Some(e) = lower.strip_prefix("what is ") {
            // Avoid matching "what is the time" etc.
            let e = e.trim();
            if e.starts_with("the ") || e.starts_with("a ") {
                return None;
            }
            e
        } else if let Some(e) = lower.strip_prefix("what is a ") {
            e.trim()
        } else if let Some(e) = lower.strip_prefix("what is an ") {
            e.trim()
        } else if let Some(e) = lower.strip_prefix("tell me about ") {
            e.trim()
        } else {
            return None;
        };

        if entity.is_empty() {
            return None;
        }

        // Look up in SQLite knowledge graph database.
        if let Some(ref db) = self.knowledge_db {
            if let Ok(conn) = db.lock() {
                let result = conn.query_row(
                    "SELECT name, entity_type, description, properties FROM entities
                     WHERE name = ?1 COLLATE NOCASE
                     OR aliases LIKE '%' || ?1 || '%' COLLATE NOCASE
                     LIMIT 1",
                    rusqlite::params![entity],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                        ))
                    },
                );

                if let Ok((name, entity_type, description, properties_json)) = result {
                    let properties: serde_json::Value =
                        serde_json::from_str(&properties_json).unwrap_or_default();

                    let answer = if entity_type.is_empty() {
                        description.clone()
                    } else {
                        format!("{} — {}", entity_type, description)
                    };

                    debug!(entity = %name, "Entity knowledge panel instant answer");

                    return Some(InstantAnswer {
                        answer_type: InstantAnswerType::EntityKnowledgePanel,
                        answer,
                        details: serde_json::json!({
                            "name": name,
                            "type": entity_type,
                            "description": description,
                            "properties": properties,
                        }),
                        source: "RAiTHE Knowledge Graph".to_string(),
                        confidence: 0.85,
                    });
                }
            }
        }

        // Fallback: built-in entities for demo showcase.
        lookup_builtin_entity(entity)
    }
}

impl InstantAnswerProvider for InstantAnswerEngine {
    fn answer(&self, query: &ParsedQuery) -> AsyncResult<'_, Option<InstantAnswer>> {
        // Clone the query string into the async block to avoid lifetime conflict.
        let q = query.original.clone();
        let threshold = self.config.confidence_threshold;
        Box::pin(async move {
            // Try calculator first (fastest).
            if let Some(answer) = self.try_calculator(&q) {
                if answer.confidence >= threshold {
                    return Ok(Some(answer));
                }
            }

            // Try unit conversion.
            if let Some(answer) = self.try_unit_conversion(&q) {
                if answer.confidence >= threshold {
                    return Ok(Some(answer));
                }
            }

            // Try date/time calculations.
            if let Some(answer) = self.try_datetime(&q) {
                if answer.confidence >= threshold {
                    return Ok(Some(answer));
                }
            }

            // Try dictionary lookup.
            if let Some(answer) = self.try_dictionary(&q) {
                if answer.confidence >= threshold {
                    return Ok(Some(answer));
                }
            }

            // Try entity knowledge panel.
            if let Some(answer) = self.try_entity_panel(&q) {
                if answer.confidence >= threshold {
                    return Ok(Some(answer));
                }
            }

            Ok(None)
        })
    }
}

// ---------------------------------------------------------------------------
// Database helpers
// ---------------------------------------------------------------------------

/// Open or create a SQLite database, applying an init function on creation.
fn open_or_create_db<F>(path: &std::path::Path, init_fn: F) -> Option<Mutex<Connection>>
where
    F: FnOnce(&Connection),
{
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match Connection::open(path) {
        Ok(conn) => {
            let _ = conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;");
            init_fn(&conn);
            Some(Mutex::new(conn))
        }
        Err(e) => {
            warn!("Could not open database {:?}: {} — feature unavailable", path, e);
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Date/Time helpers
// ---------------------------------------------------------------------------

/// Try to parse a natural language date reference from a query.
fn parse_natural_date(query: &str) -> Option<chrono::NaiveDate> {
    let today = chrono::Local::now().date_naive();

    // "days until christmas" / "days until new year"
    let year = today.year();
    if query.contains("christmas") {
        let xmas = chrono::NaiveDate::from_ymd_opt(year, 12, 25)?;
        return Some(if xmas >= today { xmas } else {
            chrono::NaiveDate::from_ymd_opt(year + 1, 12, 25)?
        });
    }
    if query.contains("new year") {
        let ny = chrono::NaiveDate::from_ymd_opt(year + 1, 1, 1)?;
        return Some(ny);
    }
    if query.contains("halloween") {
        let hw = chrono::NaiveDate::from_ymd_opt(year, 10, 31)?;
        return Some(if hw >= today { hw } else {
            chrono::NaiveDate::from_ymd_opt(year + 1, 10, 31)?
        });
    }
    if query.contains("valentine") {
        let vd = chrono::NaiveDate::from_ymd_opt(year, 2, 14)?;
        return Some(if vd >= today { vd } else {
            chrono::NaiveDate::from_ymd_opt(year + 1, 2, 14)?
        });
    }

    // Try to parse YYYY-MM-DD or MM/DD/YYYY patterns from the tail.
    let words: Vec<&str> = query.split_whitespace().collect();
    for word in words.iter().rev() {
        if let Ok(d) = chrono::NaiveDate::parse_from_str(word, "%Y-%m-%d") {
            return Some(d);
        }
        if let Ok(d) = chrono::NaiveDate::parse_from_str(word, "%m/%d/%Y") {
            return Some(d);
        }
    }

    None
}

/// Return the number of days in a given month (by name or number).
fn days_in_month_query(input: &str) -> Option<u32> {
    let parts: Vec<&str> = input.split_whitespace().collect();
    let month_str = parts.first()?;
    let year: i32 = if parts.len() > 1 {
        parts[1].parse().ok()?
    } else {
        chrono::Local::now().year()
    };

    let month = match *month_str {
        "january" | "jan" => 1,
        "february" | "feb" => 2,
        "march" | "mar" => 3,
        "april" | "apr" => 4,
        "may" => 5,
        "june" | "jun" => 6,
        "july" | "jul" => 7,
        "august" | "aug" => 8,
        "september" | "sep" | "sept" => 9,
        "october" | "oct" => 10,
        "november" | "nov" => 11,
        "december" | "dec" => 12,
        _ => month_str.parse::<u32>().ok()?,
    };

    if !(1..=12).contains(&month) {
        return None;
    }

    // Days in month: next month day 1, minus 1 day.
    let first_of_next = if month == 12 {
        chrono::NaiveDate::from_ymd_opt(year + 1, 1, 1)?
    } else {
        chrono::NaiveDate::from_ymd_opt(year, month + 1, 1)?
    };
    let first_of_this = chrono::NaiveDate::from_ymd_opt(year, month, 1)?;
    Some((first_of_next - first_of_this).num_days() as u32)
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}

/// Basic unit conversion engine (minimal safe implementation)
fn convert_units(value: f64, from: &str, to: &str) -> Option<f64> {
    let from = from.trim().to_lowercase();
    let to = to.trim().to_lowercase();

    // Normalize to meters first (base unit)
    let meters = match from.as_str() {
        "m" | "meter" | "meters" => value,
        "km" | "kilometer" | "kilometers" => value * 1000.0,
        "cm" | "centimeter" | "centimeters" => value * 0.01,
        "mm" | "millimeter" | "millimeters" => value * 0.001,
        "mi" | "mile" | "miles" => value * 1609.34,
        "ft" | "foot" | "feet" => value * 0.3048,
        "in" | "inch" | "inches" => value * 0.0254,
        _ => return None,
    };

    // Convert from meters to target unit
    let result = match to.as_str() {
        "m" | "meter" | "meters" => meters,
        "km" | "kilometer" | "kilometers" => meters / 1000.0,
        "cm" | "centimeter" | "centimeters" => meters / 0.01,
        "mm" | "millimeter" | "millimeters" => meters / 0.001,
        "mi" | "mile" | "miles" => meters / 1609.34,
        "ft" | "foot" | "feet" => meters / 0.3048,
        "in" | "inch" | "inches" => meters / 0.0254,
        _ => return None,
    };

    Some(result)
}

// ---------------------------------------------------------------------------
// Built-in dictionary for demo showcase
// ---------------------------------------------------------------------------

fn lookup_builtin_definition(word: &str) -> Option<InstantAnswer> {
    let (pos, definition) = match word {
        "algorithm" => ("noun", "A step-by-step procedure for solving a problem or accomplishing a task, especially by a computer."),
        "rust" => ("noun", "A systems programming language focused on safety, speed, and concurrency. Also: iron oxide formed by oxidation."),
        "search" => ("verb", "To look through or examine carefully in order to find something; to query a database or index for matching results."),
        "index" => ("noun", "An organized data structure that enables fast retrieval of information, such as an inverted index mapping terms to documents."),
        "pagerank" => ("noun", "An algorithm used by search engines to rank web pages based on the number and quality of links pointing to them."),
        "crawl" | "crawler" => ("noun", "A program that systematically browses the web to discover and download pages for indexing."),
        "query" => ("noun", "A request for information from a database or search engine; the text a user types into a search box."),
        "latency" => ("noun", "The delay between a request and its response; in search engines, the time from query submission to results display."),
        "semantic" => ("adjective", "Relating to meaning in language; semantic search understands the intent behind queries, not just keywords."),
        "embedding" => ("noun", "A dense vector representation of text that captures semantic meaning, enabling similarity comparisons."),
        "boilerplate" => ("noun", "Standardized, repetitive text or code; in web parsing, the non-content portions of a page (navigation, footers)."),
        "tokenize" | "tokenization" => ("verb", "To break text into individual meaningful units (tokens) for processing by a search engine or language model."),
        _ => return None,
    };

    Some(InstantAnswer {
        answer_type: InstantAnswerType::Dictionary,
        answer: format!("({}) {}", pos, definition),
        details: serde_json::json!({
            "word": word,
            "part_of_speech": pos,
            "definition": definition,
        }),
        source: "RAiTHE Dictionary".to_string(),
        confidence: 0.85,
    })
}

// ---------------------------------------------------------------------------
// Built-in entity knowledge for demo showcase
// ---------------------------------------------------------------------------

fn lookup_builtin_entity(entity: &str) -> Option<InstantAnswer> {
    let (name, etype, desc, props) = match entity {
        "rust" | "rust programming language" => (
            "Rust",
            "Programming Language",
            "A multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency.",
            serde_json::json!({
                "developer": "Rust Foundation",
                "first_appeared": "2010",
                "stable_release": "2024",
                "paradigm": "Multi-paradigm: concurrent, functional, generic, imperative, structured",
                "typing": "Static, strong, inferred",
                "website": "rust-lang.org",
            }),
        ),
        "linux" => (
            "Linux",
            "Operating System Kernel",
            "A free and open-source Unix-like operating system kernel first released by Linus Torvalds in 1991.",
            serde_json::json!({
                "developer": "Linus Torvalds and community",
                "initial_release": "1991",
                "written_in": "C, Assembly",
                "license": "GPL-2.0",
            }),
        ),
        "wikipedia" => (
            "Wikipedia",
            "Online Encyclopedia",
            "A free, multilingual online encyclopedia written and maintained by a community of volunteer contributors.",
            serde_json::json!({
                "launched": "January 15, 2001",
                "founders": "Jimmy Wales, Larry Sanger",
                "articles": "60+ million (all languages)",
                "website": "wikipedia.org",
            }),
        ),
        "canada" => (
            "Canada",
            "Country",
            "A country in North America, the world's second-largest country by total area. Capital: Ottawa.",
            serde_json::json!({
                "capital": "Ottawa",
                "population": "~40 million",
                "area": "9.98 million km²",
                "official_languages": "English, French",
                "currency": "Canadian Dollar (CAD)",
            }),
        ),
        "google" => (
            "Google",
            "Technology Company",
            "An American multinational corporation specializing in Internet-related services and products, including search.",
            serde_json::json!({
                "founded": "September 4, 1998",
                "founders": "Larry Page, Sergey Brin",
                "headquarters": "Mountain View, California",
                "parent": "Alphabet Inc.",
            }),
        ),
        "python" | "python programming language" => (
            "Python",
            "Programming Language",
            "A high-level, general-purpose programming language emphasizing code readability with significant whitespace.",
            serde_json::json!({
                "developer": "Python Software Foundation",
                "first_appeared": "1991",
                "designer": "Guido van Rossum",
                "paradigm": "Multi-paradigm: object-oriented, procedural, functional",
                "typing": "Dynamic, strong",
            }),
        ),
        _ => return None,
    };

    Some(InstantAnswer {
        answer_type: InstantAnswerType::EntityKnowledgePanel,
        answer: format!("{} — {}", etype, desc),
        details: serde_json::json!({
            "name": name,
            "type": etype,
            "description": desc,
            "properties": props,
        }),
        source: "RAiTHE Knowledge Graph".to_string(),
        confidence: 0.85,
    })
}
