// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: parser :: extractor
//
// Structural feature extraction utilities.
// Field weights for indexing are defined here and used by the indexer.
// ================================================================================

/// Index field weights.
pub struct FieldWeights;

impl FieldWeights {
    pub const TITLE: f32 = 5.0;
    pub const HEADING_H1: f32 = 3.0;
    pub const HEADING_H2: f32 = 2.5;
    pub const HEADING_H3: f32 = 2.0;
    pub const HEADING_H4: f32 = 1.5;
    pub const HEADING_H5: f32 = 1.3;
    pub const HEADING_H6: f32 = 1.2;
    pub const BODY: f32 = 1.0;
    pub const URL_TOKENS: f32 = 2.0;
    pub const ANCHOR_IN: f32 = 4.0;
    pub const META_DESC: f32 = 1.5;
    pub const ALT_TEXT: f32 = 0.8;
    pub const STRUCTURED: f32 = 1.5;

    /// Get the heading weight for a given level (1-6).
    pub fn heading_weight(level: u8) -> f32 {
        match level {
            1 => Self::HEADING_H1,
            2 => Self::HEADING_H2,
            3 => Self::HEADING_H3,
            4 => Self::HEADING_H4,
            5 => Self::HEADING_H5,
            6 => Self::HEADING_H6,
            _ => Self::BODY,
        }
    }
}
