//! allowing users to understand which dimensions contributed most to the similarity score.

use common::types::{DimensionContribution, ScoreExplanation, ScoreType};

use crate::data_types::vectors::VectorElementType;
use crate::types::Distance;

pub const DEFAULT_TOP_DIMENSIONS: usize = 10;


/// For dot product, the contribution of each dimension is simply `v1[i] * v2[i]`.
/// The total score is the sum of all contributions.
pub fn dot_product_contributions(
    v1: &[VectorElementType],
    v2: &[VectorElementType],
) -> Vec<DimensionContribution> {
    v1.iter()
        .zip(v2.iter())
        .enumerate()
        .map(|(dimension, (a, b))| DimensionContribution {
            dimension,
            contribution: a * b,
        })
        .collect()
}

/// For Euclidean distance, the contribution of each dimension is `-(v1[i] - v2[i])^2`.
/// The total score is the negative sum of squared differences.
/// Larger (less negative) contributions indicate dimensions where vectors are more similar.
pub fn euclidean_contributions(
    v1: &[VectorElementType],
    v2: &[VectorElementType],
) -> Vec<DimensionContribution> {
    v1.iter()
        .zip(v2.iter())
        .enumerate()
        .map(|(dimension, (a, b))| {
            let diff = a - b;
            DimensionContribution {
                dimension,
                contribution: -(diff * diff), // Negative because smaller distance = more similar
            }
        })
        .collect()
}


/// For cosine similarity, we compute the contribution of each dimension to the dot product
/// portion of the formula. The contributions are normalized by the product of the norms.
pub fn cosine_contributions(
    v1: &[VectorElementType],
    v2: &[VectorElementType],
) -> Vec<DimensionContribution> {
    let norm1: ScoreType = v1.iter().map(|x| x * x).sum::<ScoreType>().sqrt();
    let norm2: ScoreType = v2.iter().map(|x| x * x).sum::<ScoreType>().sqrt();
    
    let denominator = norm1 * norm2;
    if denominator == 0.0 {
        // If either vector has zero norm, all contributions are zero
        return v1.iter()
            .enumerate()
            .map(|(dimension, _)| DimensionContribution {
                dimension,
                contribution: 0.0,
            })
            .collect();
    }

    v1.iter()
        .zip(v2.iter())
        .enumerate()
        .map(|(dimension, (a, b))| DimensionContribution {
            dimension,
            contribution: (a * b) / denominator,
        })
        .collect()
}

/// For Manhattan distance, the contribution of each dimension is `-|v1[i] - v2[i]|`.
/// Larger (less negative) contributions indicate dimensions where vectors are more similar.
pub fn manhattan_contributions(
    v1: &[VectorElementType],
    v2: &[VectorElementType],
) -> Vec<DimensionContribution> {
    v1.iter()
        .zip(v2.iter())
        .enumerate()
        .map(|(dimension, (a, b))| DimensionContribution {
            dimension,
            contribution: -(a - b).abs(),
        })
        .collect()
}

/// Compute per-dimension contributions based on the distance metric.
pub fn compute_contributions(
    distance: Distance,
    v1: &[VectorElementType],
    v2: &[VectorElementType],
) -> Vec<DimensionContribution> {
    match distance {
        Distance::Dot => dot_product_contributions(v1, v2),
        Distance::Cosine => cosine_contributions(v1, v2),
        Distance::Euclid => euclidean_contributions(v1, v2),
        Distance::Manhattan => manhattan_contributions(v1, v2),
    }
}

/// Compute a score explanation for the similarity between two vectors.
/// 
/// # Arguments
/// * `distance` - The distance metric used for similarity
/// * `v1` - The first vector (typically the query vector)
/// * `v2` - The second vector (typically the stored vector)
/// * `top_n` - Number of top contributing dimensions to include (default: 10)
/// 
/// # Returns
/// A `ScoreExplanation` containing the top N dimensions that contributed most to the score.
pub fn compute_explanation(
    distance: Distance,
    v1: &[VectorElementType],
    v2: &[VectorElementType],
    top_n: Option<usize>,
) -> ScoreExplanation {
    let contributions = compute_contributions(distance, v1, v2);
    ScoreExplanation::new(contributions, top_n.unwrap_or(DEFAULT_TOP_DIMENSIONS))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_contributions() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        
        let contributions = dot_product_contributions(&v1, &v2);
        
        assert_eq!(contributions.len(), 3);
        assert_eq!(contributions[0].dimension, 0);
        assert_eq!(contributions[0].contribution, 4.0); // 1*4
        assert_eq!(contributions[1].dimension, 1);
        assert_eq!(contributions[1].contribution, 10.0); // 2*5
        assert_eq!(contributions[2].dimension, 2);
        assert_eq!(contributions[2].contribution, 18.0); // 3*6
        
        let total: ScoreType = contributions.iter().map(|c| c.contribution).sum();
        assert_eq!(total, 32.0); // 4+10+18
    }

    #[test]
    fn test_euclidean_contributions() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        
        let contributions = euclidean_contributions(&v1, &v2);
        
        assert_eq!(contributions.len(), 3);
        // Each contribution should be negative (since we're looking at distance)
        assert!(contributions[0].contribution < 0.0);
        assert_eq!(contributions[0].contribution, -9.0); // -(1-4)^2
        assert_eq!(contributions[1].contribution, -9.0); // -(2-5)^2
        assert_eq!(contributions[2].contribution, -9.0); // -(3-6)^2
    }

    #[test]
    fn test_cosine_contributions() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![1.0, 0.0];
        
        let contributions = cosine_contributions(&v1, &v2);
        
        assert_eq!(contributions.len(), 2);
        assert_eq!(contributions[0].contribution, 1.0); // perfectly aligned
        assert_eq!(contributions[1].contribution, 0.0); // no contribution from dimension 1
    }

    #[test]
    fn test_explanation_top_n() {
        let v1 = vec![1.0, 5.0, 2.0, 8.0, 3.0];
        let v2 = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        
        let explanation = compute_explanation(Distance::Dot, &v1, &v2, Some(3));
        
        assert_eq!(explanation.top_dimensions.len(), 3);
        // Should be sorted by absolute contribution (descending)
        // Contributions: [1, 5, 2, 8, 3]
        // Top 3 should be dimensions 3 (8), 1 (5), 4 (3)
        assert_eq!(explanation.top_dimensions[0].dimension, 3);
        assert_eq!(explanation.top_dimensions[0].contribution, 8.0);
        assert_eq!(explanation.top_dimensions[1].dimension, 1);
        assert_eq!(explanation.top_dimensions[1].contribution, 5.0);
        assert_eq!(explanation.top_dimensions[2].dimension, 4);
        assert_eq!(explanation.top_dimensions[2].contribution, 3.0);
    }
}
