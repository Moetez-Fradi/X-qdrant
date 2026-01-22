use std::cmp::Ordering;

use ordered_float::OrderedFloat;
use strum::EnumIter;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Type of vector matching score
pub type ScoreType = f32;
/// Type of point index inside a segment
pub type PointOffsetType = u32;

#[derive(Copy, Clone, PartialEq, Debug, Default, FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct ScoredPointOffset {
    pub idx: PointOffsetType,
    pub score: ScoreType,
}

impl Eq for ScoredPointOffset {}

impl Ord for ScoredPointOffset {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.score).cmp(&OrderedFloat(other.score))
    }
}

impl PartialOrd for ScoredPointOffset {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// the contribution of a single dimension to the similarity score
#[derive(Clone, Debug, PartialEq)]
pub struct DimensionContribution {
    pub dimension: usize,
    pub contribution: ScoreType,
}

/// Explanation of how a similarity score was computed
#[derive(Clone, Debug, Default)]
pub struct ScoreExplanation {
    pub top_dimensions: Vec<DimensionContribution>,
}

impl ScoreExplanation {
    pub fn new(mut contributions: Vec<DimensionContribution>, top_n: usize) -> Self {
        contributions.sort_by(|a, b| {
            OrderedFloat(b.contribution.abs())
                .cmp(&OrderedFloat(a.contribution.abs()))
        });
        contributions.truncate(top_n);
        Self {
            top_dimensions: contributions,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TelemetryDetail {
    pub level: DetailsLevel,
    pub histograms: bool,
}

impl TelemetryDetail {
    pub fn new(level: DetailsLevel, histograms: bool) -> Self {
        Self { level, histograms }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumIter)]
pub enum DetailsLevel {
    /// Minimal information level
    /// - app info
    /// - minimal telemetry by endpoint
    /// - cluster status
    Level0,
    /// Detailed common info level
    /// - app info details
    /// - system info
    ///   - hardware flags
    ///   - hardware usage per collection
    ///   - RAM usage
    /// - cluster basic details
    /// - collections basic info
    Level1,
    /// Detailed consensus info - peers info
    /// Collections:
    ///  - detailed config
    ///  - Shards - basic config
    Level2,
    /// Shards:
    ///  - detailed config
    ///  - Optimizers info
    Level3,
    /// Segment level telemetry
    Level4,
}

impl Default for TelemetryDetail {
    fn default() -> Self {
        TelemetryDetail {
            level: DetailsLevel::Level0,
            histograms: false,
        }
    }
}

impl From<usize> for DetailsLevel {
    fn from(value: usize) -> Self {
        match value {
            0 => DetailsLevel::Level0,
            1 => DetailsLevel::Level1,
            2 => DetailsLevel::Level2,
            3 => DetailsLevel::Level3,
            4 => DetailsLevel::Level4,
            _ => DetailsLevel::Level4,
        }
    }
}
