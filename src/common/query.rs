use std::time::Duration;

use api::rest::SearchGroupsRequestInternal;
use collection::collection::distance_matrix::*;
use collection::common::batching::batch_requests;
use collection::grouping::group_by::GroupRequest;
use collection::operations::consistency_params::ReadConsistency;
use collection::operations::shard_selector_internal::ShardSelectorInternal;
use collection::operations::types::*;
use collection::operations::universal_query::collection_query::*;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::types::ScoreExplanation;
use segment::data_types::vectors::{DenseVector, VectorInternal};
use segment::spaces::explainability::compute_explanation;
use segment::types::{Distance, ScoredPoint, WithVector};
use shard::query::query_enum::QueryEnum;
use shard::retrieve::record_internal::RecordInternal;
use shard::search::CoreSearchRequestBatch;
use storage::content_manager::errors::StorageError;
use storage::content_manager::toc::TableOfContent;
use storage::rbac::Access;

#[allow(clippy::too_many_arguments)]
pub async fn do_core_search_points(
    toc: &TableOfContent,
    collection_name: &str,
    mut request: CoreSearchRequest,
    read_consistency: Option<ReadConsistency>,
    shard_selection: ShardSelectorInternal,
    access: Access,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<Vec<ScoredPoint>, StorageError> {
    let with_explanation = request.with_explanation;
    let original_with_vector = request.with_vector.clone();
    
    // if explanation is requested, we need vectors to compute it
    if with_explanation {
        request.with_vector = Some(WithVector::Bool(true));
    }
    
    let query_vector: Option<DenseVector> = if with_explanation {
        extract_query_vector(&request.query)
    } else {
        None
    };
    
    let batch_res = do_core_search_batch_points(
        toc,
        collection_name,
        CoreSearchRequestBatch {
            searches: vec![request],
        },
        read_consistency,
        shard_selection.clone(),
        access.clone(),
        timeout,
        hw_measurement_acc,
    )
    .await?;
    
    let mut results = batch_res
        .into_iter()
        .next()
        .ok_or_else(|| StorageError::service_error("Empty search result"))?;
    
    // Compute explanations if requested
    if with_explanation {
        if let Some(ref query_vec) = query_vector {
            // Get the distance metric from collection config
            let distance = get_collection_distance(toc, collection_name, &access, &shard_selection).await
                .unwrap_or(Distance::Cosine);
            
            for point in &mut results {
                if let Some(ref vector_struct) = point.vector {
                    // Try to get the default vector or first named vector
                    if let Some(result_vec) = extract_dense_vector_from_struct(vector_struct) {
                        let explanation = compute_explanation_for_distance(
                            query_vec,
                            &result_vec,
                            distance,
                            10, // top 10
                        );
                        point.score_explanation = Some(explanation);
                    }
                }
            }
        }
        
        if original_with_vector.is_none() || matches!(original_with_vector, Some(WithVector::Bool(false))) {
            for point in &mut results {
                point.vector = None;
            }
        }
    }
    
    Ok(results)
}

/// Extract the query vector from a QueryEnum (for Nearest queries with dense vectors)
fn extract_query_vector(query: &QueryEnum) -> Option<DenseVector> {
    match query {
        QueryEnum::Nearest(named_query) => {
            match &named_query.query {
                VectorInternal::Dense(dense) => Some(dense.clone()),
                VectorInternal::Sparse(_) | VectorInternal::MultiDense(_) => None,
            }
        }
        _ => None, // Only Nearest queries have vectors
    }
}

/// Extract a dense vector from a VectorStruct
fn extract_dense_vector_from_struct(vector_struct: &segment::data_types::vectors::VectorStructInternal) -> Option<DenseVector> {
    use segment::data_types::vectors::VectorStructInternal;
    match vector_struct {
        VectorStructInternal::Single(dense) => Some(dense.clone()), // Single is already a DenseVector
        VectorStructInternal::MultiDense(_) => None, // Multi-dense not supported yet
        VectorStructInternal::Named(named_map) => {
            // Get the first dense vector from named vectors
            for vec in named_map.values() {
                if let VectorInternal::Dense(dense) = vec {
                    return Some(dense.clone());
                }
            }
            None
        }
    }
}

/// Get the distance metric for a collection
async fn get_collection_distance(
    toc: &TableOfContent,
    collection_name: &str,
    access: &Access,
    shard_selection: &ShardSelectorInternal,
) -> Option<Distance> {
    // Try to get collection info to determine the distance metric
    // This is a simplified approach - in a real implementation you might cache this
    let _ = (toc, collection_name, access, shard_selection);
    // For now, return None and let the caller use a default
    // A full implementation would query the collection config
    None
}

/// Compute explanation based on the distance metric
fn compute_explanation_for_distance(
    query: &[f32],
    result: &[f32],
    distance: Distance,
    top_n: usize,
) -> ScoreExplanation {
    compute_explanation(distance, query, result, Some(top_n))
}

pub async fn do_search_batch_points(
    toc: &TableOfContent,
    collection_name: &str,
    requests: Vec<(CoreSearchRequest, ShardSelectorInternal)>,
    read_consistency: Option<ReadConsistency>,
    access: Access,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<Vec<Vec<ScoredPoint>>, StorageError> {
    let requests = batch_requests::<
        (CoreSearchRequest, ShardSelectorInternal),
        ShardSelectorInternal,
        Vec<CoreSearchRequest>,
        Vec<_>,
    >(
        requests,
        |(_, shard_selector)| shard_selector,
        |(request, _), core_reqs| {
            core_reqs.push(request);
            Ok(())
        },
        |shard_selector, core_requests, res| {
            if core_requests.is_empty() {
                return Ok(());
            }

            let core_batch = CoreSearchRequestBatch {
                searches: core_requests,
            };

            let req = toc.core_search_batch(
                collection_name,
                core_batch,
                read_consistency,
                shard_selector,
                access.clone(),
                timeout,
                hw_measurement_acc.clone(),
            );
            res.push(req);
            Ok(())
        },
    )?;

    let results = futures::future::try_join_all(requests).await?;
    let flatten_results: Vec<Vec<_>> = results.into_iter().flatten().collect();
    Ok(flatten_results)
}

#[allow(clippy::too_many_arguments)]
pub async fn do_core_search_batch_points(
    toc: &TableOfContent,
    collection_name: &str,
    request: CoreSearchRequestBatch,
    read_consistency: Option<ReadConsistency>,
    shard_selection: ShardSelectorInternal,
    access: Access,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<Vec<Vec<ScoredPoint>>, StorageError> {
    toc.core_search_batch(
        collection_name,
        request,
        read_consistency,
        shard_selection,
        access,
        timeout,
        hw_measurement_acc,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn do_search_point_groups(
    toc: &TableOfContent,
    collection_name: &str,
    request: SearchGroupsRequestInternal,
    read_consistency: Option<ReadConsistency>,
    shard_selection: ShardSelectorInternal,
    access: Access,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<GroupsResult, StorageError> {
    toc.group(
        collection_name,
        GroupRequest::from(request),
        read_consistency,
        shard_selection,
        access,
        timeout,
        hw_measurement_acc,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn do_recommend_point_groups(
    toc: &TableOfContent,
    collection_name: &str,
    request: RecommendGroupsRequestInternal,
    read_consistency: Option<ReadConsistency>,
    shard_selection: ShardSelectorInternal,
    access: Access,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<GroupsResult, StorageError> {
    toc.group(
        collection_name,
        GroupRequest::from(request),
        read_consistency,
        shard_selection,
        access,
        timeout,
        hw_measurement_acc,
    )
    .await
}

pub async fn do_discover_batch_points(
    toc: &TableOfContent,
    collection_name: &str,
    request: DiscoverRequestBatch,
    read_consistency: Option<ReadConsistency>,
    access: Access,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<Vec<Vec<ScoredPoint>>, StorageError> {
    let requests = request
        .searches
        .into_iter()
        .map(|req| {
            let shard_selector = match req.shard_key {
                None => ShardSelectorInternal::All,
                Some(shard_key) => ShardSelectorInternal::from(shard_key),
            };

            (req.discover_request, shard_selector)
        })
        .collect();

    toc.discover_batch(
        collection_name,
        requests,
        read_consistency,
        access,
        timeout,
        hw_measurement_acc,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn do_count_points(
    toc: &TableOfContent,
    collection_name: &str,
    request: CountRequestInternal,
    read_consistency: Option<ReadConsistency>,
    timeout: Option<Duration>,
    shard_selection: ShardSelectorInternal,
    access: Access,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<CountResult, StorageError> {
    toc.count(
        collection_name,
        request,
        read_consistency,
        timeout,
        shard_selection,
        access,
        hw_measurement_acc,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn do_get_points(
    toc: &TableOfContent,
    collection_name: &str,
    request: PointRequestInternal,
    read_consistency: Option<ReadConsistency>,
    timeout: Option<Duration>,
    shard_selection: ShardSelectorInternal,
    access: Access,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<Vec<RecordInternal>, StorageError> {
    toc.retrieve(
        collection_name,
        request,
        read_consistency,
        timeout,
        shard_selection,
        access,
        hw_measurement_acc,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn do_scroll_points(
    toc: &TableOfContent,
    collection_name: &str,
    request: ScrollRequestInternal,
    read_consistency: Option<ReadConsistency>,
    timeout: Option<Duration>,
    shard_selection: ShardSelectorInternal,
    access: Access,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<ScrollResult, StorageError> {
    toc.scroll(
        collection_name,
        request,
        read_consistency,
        timeout,
        shard_selection,
        access,
        hw_measurement_acc,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn do_query_points(
    toc: &TableOfContent,
    collection_name: &str,
    request: CollectionQueryRequest,
    read_consistency: Option<ReadConsistency>,
    shard_selection: ShardSelectorInternal,
    access: Access,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<Vec<ScoredPoint>, StorageError> {
    let requests = vec![(request, shard_selection)];
    let batch_res = toc
        .query_batch(
            collection_name,
            requests,
            read_consistency,
            access,
            timeout,
            hw_measurement_acc,
        )
        .await?;
    batch_res
        .into_iter()
        .next()
        .ok_or_else(|| StorageError::service_error("Empty query result"))
}

#[allow(clippy::too_many_arguments)]
pub async fn do_query_batch_points(
    toc: &TableOfContent,
    collection_name: &str,
    requests: Vec<(CollectionQueryRequest, ShardSelectorInternal)>,
    read_consistency: Option<ReadConsistency>,
    access: Access,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<Vec<Vec<ScoredPoint>>, StorageError> {
    toc.query_batch(
        collection_name,
        requests,
        read_consistency,
        access,
        timeout,
        hw_measurement_acc,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn do_query_point_groups(
    toc: &TableOfContent,
    collection_name: &str,
    request: CollectionQueryGroupsRequest,
    read_consistency: Option<ReadConsistency>,
    shard_selection: ShardSelectorInternal,
    access: Access,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<GroupsResult, StorageError> {
    toc.group(
        collection_name,
        GroupRequest::from(request),
        read_consistency,
        shard_selection,
        access,
        timeout,
        hw_measurement_acc,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn do_search_points_matrix(
    toc: &TableOfContent,
    collection_name: &str,
    request: CollectionSearchMatrixRequest,
    read_consistency: Option<ReadConsistency>,
    shard_selection: ShardSelectorInternal,
    access: Access,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> Result<CollectionSearchMatrixResponse, StorageError> {
    toc.search_points_matrix(
        collection_name,
        request,
        read_consistency,
        shard_selection,
        access,
        timeout,
        hw_measurement_acc,
    )
    .await
}
