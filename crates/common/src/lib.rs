// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: common
//
// Shared types, error handling, configuration loading, and feature flags.
// This crate is the foundation that every other raithe-se crate depends on.
// ================================================================================

pub mod config;
pub mod error;
pub mod types;
pub mod simhash;
pub mod traits;

/// raithe-se version string.
pub const RAITHE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// RAiTHE branding banner for startup logging.
pub const RAITHE_BANNER: &str = r#"

                             ██████╗  █████╗ ██╗████████╗██╗  ██╗███████╗
                             ██╔══██╗██╔══██╗   ╚══██╔══╝██║  ██║██╔════╝
                             ██████╔╝███████║██║   ██║   ███████║█████╗
                             ██╔══██╗██╔══██║██║   ██║   ██╔══██║██╔══╝
                             ██║  ██║██║  ██║██║   ██║   ██║  ██║███████╗
                             ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝
                                            SEARCH ENGINE

                                © RAiTHE INDUSTRIES INCORPORATED 2026
                          Engineered in Rust — Smarter paths, faster results.

"#;
