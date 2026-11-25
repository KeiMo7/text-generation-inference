/// Batching and inference logic
use crate::client::{
    Batch, CachedBatch, ClientError, Generation, Health, InfoResponse, ShardedClient,
};
use crate::queue::{
    CancelReason, Entry, EntryState, Queue, QueueConfig, QueueError, DEFAULT_PRIORITY,
};
use async_trait::async_trait;
use nohash_hasher::IntMap;
use std::sync::Arc;
use text_generation_router::infer::{Backend, GeneratedText, InferError, InferStreamResponse};
use text_generation_router::validation::ValidGenerateRequest;
use text_generation_router::{FinishReason, PrefillToken, Token};
use tokio::sync::mpsc::error::SendError;
use tokio::sync::{mpsc, Notify};
use tokio::time::{sleep, Duration, Instant};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{info_span, instrument, Instrument, Span};

pub struct BackendV3 {
    /// Request queue
    queue: Queue,
    /// Notify batcher on queue appends
    batching_task_notifier: Arc<Notify>,
    /// Client clone, used for health checks to skip the queue
    client: ShardedClient,
}

impl BackendV3 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        client: ShardedClient,
        waiting_served_ratio: f32,
        max_batch_prefill_tokens: u32,
        max_batch_total_tokens: u32,
        max_waiting_tokens: usize,
        max_batch_size: Option<usize>,
        shard_info: InfoResponse,
    ) -> Self {
        if shard_info.support_chunking {
            tracing::warn!("Model supports prefill chunking. `waiting_served_ratio` and `max_waiting_tokens` will be ignored.");
        }

        let config = QueueConfig {
            requires_padding: shard_info.requires_padding,
            block_size: shard_info.block_size,
            prefix_caching: shard_info.use_prefix_caching,
            window_size: shard_info.window_size,
            speculate: shard_info.speculate,
            max_batch_total_tokens,
            max_batch_prefill_tokens,
            max_batch_size,
            support_chunking: shard_info.support_chunking,
            ..QueueConfig::default()
        };
        let retry_interval = config.retry_interval;
        let queue = Queue::new(config);
        let batching_task_notifier = Arc::new(Notify::new());

        // Spawn batching background task that contains all the inference logic
        tokio::spawn(batching_task(
            client.clone(),
            waiting_served_ratio,
            max_batch_prefill_tokens,
            max_batch_total_tokens,
            max_waiting_tokens,
            max_batch_size,
            shard_info.support_chunking,
            queue.clone(),
            batching_task_notifier.clone(),
            retry_interval,
        ));

        Self {
            queue,
            batching_task_notifier,
            client,
        }
    }

    /// Test-only constructor: same wiring as [`BackendV3::new`] but without
    /// spawning the batching task, so that admission control can be exercised
    /// deterministically. The queue can be drained through [`Queue::cancel`] or
    /// [`Queue::shutdown`].
    #[cfg(test)]
    pub(crate) fn for_schedule_tests(config: QueueConfig, client: ShardedClient) -> Self {
        let queue = Queue::new(config);
        Self {
            queue,
            batching_task_notifier: Arc::new(Notify::new()),
            client,
        }
    }

    #[cfg(test)]
    pub(crate) fn queue(&self) -> Queue {
        self.queue.clone()
    }
}

#[async_trait]
impl Backend for BackendV3 {
    #[instrument(skip_all)]
    fn schedule(
        &self,
        request: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, InferError>>, InferError> {
        // MPSC channel to communicate with the background batching task
        let (response_tx, response_rx) = mpsc::unbounded_channel();

        // Append the request to the queue. Admission (id allocation, size and
        // capacity checks) happens synchronously here.
        let entry = Entry {
            id: 0,
            // Requests do not carry a priority yet; the queue clamps and owns
            // the final value.
            priority: DEFAULT_PRIORITY,
            state: EntryState::Queued,
            skipped: 0,
            forced: false,
            request,
            response_tx,
            span: Span::current(),
            temp_span: None,
            queue_time: Instant::now(),
            batch_time: None,
            block_allocation: None,
        };
        if let Err(err) = self.queue.append(entry) {
            return Err(match err {
                QueueError::Full { capacity } => InferError::QueueFull { capacity },
                QueueError::RequestTooLarge { tokens, limit } => {
                    InferError::RequestTooLarge { tokens, limit }
                }
                QueueError::Shutdown => {
                    InferError::GenerationError("queue is shutting down".to_string())
                }
            });
        }

        // Notify the background task that we have a new entry in the queue that
        // needs to be batched
        self.batching_task_notifier.notify_one();

        // Return stream
        Ok(UnboundedReceiverStream::new(response_rx))
    }

    async fn health(&self, current_health: bool) -> bool {
        if current_health {
            // Generation is healthy, we only check that the shards can allocate on device
            self.client.device_health().await
        } else {
            self.client.model_health().await
        }
        .is_ok()
    }

    fn start_health(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "tgi-v3"
    }
}

/// Batching logic
/// Will be launched in a background Tokio task
///
/// Batches requests and sends them to the inference server
#[allow(clippy::too_many_arguments)]
pub(crate) async fn batching_task(
    mut client: ShardedClient,
    waiting_served_ratio: f32,
    max_batch_prefill_tokens: u32,
    max_batch_total_tokens: u32,
    max_waiting_tokens: usize,
    max_batch_size: Option<usize>,
    support_chunking: bool,
    queue: Queue,
    notifier: Arc<Notify>,
    retry_interval: Duration,
) {
    // Infinite loop
    loop {
        // Wait for a notification from the Infer struct
        notifier.notified().await;

        // Get the next batch from the queue
        // This batch might be smaller than the maximum batch size if there are
        // not enough requests waiting in the queue
        while let Some(next) = next_batch_with_retry(
            &queue,
            None,
            max_batch_size,
            max_batch_prefill_tokens,
            max_batch_total_tokens,
            &notifier,
            retry_interval,
        )
        .await
        {
            let Some((mut entries, batch, span)) = next else {
                // Queue shut down.
                return;
            };
            let mut cached_batch = prefill(&mut client, batch, None, &mut entries)
                .instrument(span)
                .await;
            let mut waiting_tokens = 1;

            // We loop until we do not receive any cached batch from the
            // inference server (== until all requests have met their stopping
            // criteria)
            while let Some(batch) = cached_batch {
                // Get current batch info
                let batch_size = batch.size;
                let batch_max_tokens = batch.max_tokens;
                let current_tokens = batch.current_tokens;
                let mut batches = vec![batch];
                metrics::gauge!("tgi_batch_current_size").set(batch_size as f64);
                metrics::gauge!("tgi_batch_current_max_tokens").set(batch_max_tokens as f64);

                let token_budget = max_batch_total_tokens.saturating_sub(batch_max_tokens);

                let (min_size, max_size, prefill_token_budget) = if support_chunking {
                    // Since the next batch will be concatenated with the
                    // current batch, the current batch tokens must be
                    // subtracted to the prefill budget
                    let prefill_token_budget =
                        max_batch_prefill_tokens.saturating_sub(current_tokens);
                    // We can ignore min_size and max_size
                    // Models than rely on max_size cannot support chunking
                    // Regarding min_size, chunking allow us to consistently
                    // run at the compute bound, making min_size useless.
                    (None, None, prefill_token_budget)
                } else {
                    let min_size = if waiting_tokens >= max_waiting_tokens {
                        // If we didn't onboard any new requests since >=
                        // max_waiting_tokens, we try to add a new batch even
                        // though its size might be small
                        None
                    } else {
                        // Minimum batch size
                        // TODO: temporarily disable to avoid incorrect
                        //       deallocation + reallocation when using prefix
                        //       caching.
                        Some((batch_size as f32 * waiting_served_ratio).floor() as usize)
                    };

                    let max_size =
                        max_batch_size.map(|max_size| max_size.saturating_sub(batch_size as usize));

                    (min_size, max_size, max_batch_prefill_tokens)
                };

                // Try to get a new batch
                if let Some(next) = next_batch_with_retry(
                    &queue,
                    min_size,
                    max_size,
                    prefill_token_budget,
                    token_budget,
                    &notifier,
                    retry_interval,
                )
                .await
                {
                    let Some((mut new_entries, new_batch, span)) = next else {
                        // Queue shut down.
                        return;
                    };
                    // Tracking metrics
                    if min_size.is_some() {
                        metrics::counter!("tgi_batch_concat", "reason" => "backpressure")
                            .increment(1);
                    } else {
                        let counter = if support_chunking {
                            metrics::counter!("tgi_batch_concat", "reason" => "chunking")
                        } else {
                            metrics::counter!("tgi_batch_concat", "reason" => "wait_exceeded")
                        };
                        counter.increment(1);
                    }

                    let new_cached_batch = if support_chunking {
                        // Get cached batch
                        let cached_batch = batches.pop();
                        // Extend entries with the new entries since the batch
                        // will be concatenated during the prefill op server
                        // side
                        entries.extend(new_entries);
                        // Generate one token for both the cached batch and the
                        // new batch
                        let new_cached_batch =
                            prefill(&mut client, new_batch, cached_batch, &mut entries)
                                .instrument(span)
                                .await;
                        if new_cached_batch.is_none() {
                            // New cached batch is empty, no work left
                            break;
                        }
                        new_cached_batch
                    } else {
                        // Request are waiting because we cannot concatenate
                        // the batches if the model/server does not support
                        // chunking
                        entries.iter_mut().for_each(|(_, entry)| {
                            // Create a new span to add the info that this entry
                            // is waiting because a new batch is being computed
                            let entry_waiting_span = info_span!(parent: &entry.span, "waiting");
                            // Add relationships
                            span.follows_from(&entry_waiting_span);
                            entry_waiting_span.follows_from(&span);
                            // Update entry
                            entry.temp_span = Some(entry_waiting_span);
                        });

                        // Generate one token for this new batch to have the
                        // attention past in cache
                        let new_cached_batch =
                            prefill(&mut client, new_batch, None, &mut new_entries)
                                .instrument(span)
                                .await;
                        if new_cached_batch.is_some() {
                            // Extend entries
                            entries.extend(new_entries);
                        }
                        new_cached_batch
                    };

                    // Reset waiting counter
                    waiting_tokens = 1;
                    // Extend current batch with the new batch
                    if let Some(new_cached_batch) = new_cached_batch {
                        batches.push(new_cached_batch);
                    }
                }

                // Create span for this batch to add context to inference calls
                let next_batch_size = entries.len();
                let next_batch_span =
                    info_span!(parent: None, "batch", batch_size = next_batch_size);
                entries.iter_mut().for_each(|(_, entry)| {
                    // Create a new span to link the batch back to this entry
                    let entry_batch_span = info_span!(parent: &entry.span, "infer");
                    // Add relationships
                    next_batch_span.follows_from(&entry_batch_span);
                    entry_batch_span.follows_from(&next_batch_span);
                    // Update entry
                    entry.temp_span = Some(entry_batch_span);
                });

                cached_batch = decode(&mut client, batches, &mut entries)
                    .instrument(next_batch_span)
                    .await;
                waiting_tokens += 1;
            }
            metrics::gauge!("tgi_batch_current_size").set(0.0);
            metrics::gauge!("tgi_batch_current_max_tokens").set(0.0);
        }
    }
}

/// Fetch the next batch, retrying while the queue holds entries but none can
/// fit yet (e.g. block allocation failures). An empty queue parks on the
/// notifier without any polling: the retry timer is only armed when
/// [`Queue::len`] reports pending work.
///
/// Returns `None` when the queue shuts down and `Some(None)` is never
/// produced: an empty batch is retried internally instead.
async fn next_batch_with_retry(
    queue: &Queue,
    min_size: Option<usize>,
    max_size: Option<usize>,
    prefill_token_budget: u32,
    token_budget: u32,
    notifier: &Notify,
    retry_interval: Duration,
) -> Option<Option<(IntMap<u64, Entry>, Batch, Span)>> {
    loop {
        match queue
            .next_batch(min_size, max_size, prefill_token_budget, token_budget)
            .await
        {
            Ok(Some(next_batch)) => return Some(Some(next_batch)),
            Ok(None) => {
                // No work or nothing fit yet.
                if queue.len() == 0 {
                    // Empty queue: wait for an append without polling.
                    notifier.notified().await;
                    continue;
                }
                // Entries are waiting but could not be scheduled: retry after
                // the configured delay, waking immediately on notification.
                tokio::select! {
                    _ = sleep(retry_interval) => {}
                    _ = notifier.notified() => {}
                }
            }
            Err(QueueError::Shutdown) => return None,
            Err(err) => {
                tracing::error!("Unexpected queue error: {err}");
                return None;
            }
        }
    }
}

#[instrument(skip_all)]
async fn prefill(
    client: &mut ShardedClient,
    batch: Batch,
    cached_batch: Option<CachedBatch>,
    entries: &mut IntMap<u64, Entry>,
) -> Option<CachedBatch> {
    let start_time = Instant::now();
    let batch_id = batch.id;
    metrics::counter!("tgi_batch_inference_count", "method" => "prefill").increment(1);

    match client.prefill(batch, cached_batch).await {
        Ok((generations, next_batch, timings)) => {
            let start_filtering_time = Instant::now();
            // Send generated tokens and filter stopped entries
            filter_send_generations(generations, entries);

            // Filter next batch and remove requests that were stopped
            let next_batch = filter_batch(client, next_batch, entries).await;

            if let Some(concat_duration) = timings.concat {
                metrics::histogram!("tgi_batch_concat_duration", "method" => "decode")
                    .record(concat_duration.as_secs_f64());
            }
            metrics::histogram!("tgi_batch_forward_duration", "method" => "prefill")
                .record(timings.forward.as_secs_f64());
            metrics::histogram!("tgi_batch_decode_duration", "method" => "prefill")
                .record(timings.decode.as_secs_f64());
            metrics::histogram!("tgi_batch_filter_duration", "method" => "prefill")
                .record(start_filtering_time.elapsed().as_secs_f64());
            metrics::histogram!("tgi_batch_inference_duration", "method" => "prefill")
                .record(start_time.elapsed().as_secs_f64());
            metrics::counter!("tgi_batch_inference_success", "method" => "prefill").increment(1);
            next_batch
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            handle_batch_failure(client, &[batch_id], entries, err, "prefill").await;
            None
        }
    }
}

#[instrument(skip_all)]
async fn decode(
    client: &mut ShardedClient,
    batches: Vec<CachedBatch>,
    entries: &mut IntMap<u64, Entry>,
) -> Option<CachedBatch> {
    let start_time = Instant::now();
    let batch_ids: Vec<u64> = batches.iter().map(|b| b.id).collect();
    metrics::counter!("tgi_batch_inference_count", "method" => "decode").increment(1);

    match client.decode(batches).await {
        Ok((generations, next_batch, timings)) => {
            let start_filtering_time = Instant::now();
            // Send generated tokens and filter stopped entries
            filter_send_generations(generations, entries);

            // Filter next batch and remove requests that were stopped
            let next_batch = filter_batch(client, next_batch, entries).await;

            if let Some(concat_duration) = timings.concat {
                metrics::histogram!("tgi_batch_concat_duration", "method" => "decode")
                    .record(concat_duration.as_secs_f64());
            }
            metrics::histogram!("tgi_batch_forward_duration", "method" => "decode")
                .record(timings.forward.as_secs_f64());
            metrics::histogram!("tgi_batch_decode_duration", "method" => "decode")
                .record(timings.decode.as_secs_f64());
            metrics::histogram!("tgi_batch_filter_duration", "method" => "decode")
                .record(start_filtering_time.elapsed().as_secs_f64());
            metrics::histogram!("tgi_batch_inference_duration", "method" => "decode")
                .record(start_time.elapsed().as_secs_f64());
            metrics::counter!("tgi_batch_inference_success", "method" => "decode").increment(1);
            next_batch
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            handle_batch_failure(client, &batch_ids, entries, err, "decode").await;
            None
        }
    }
}

/// Shared failure handling for prefill/decode and cache management failures:
/// best-effort clear every affected shard batch, then fail the entries that
/// are still connected. Entries whose client disconnected are only reaped and
/// never counted twice.
async fn handle_batch_failure(
    client: &mut ShardedClient,
    batch_ids: &[u64],
    entries: &mut IntMap<u64, Entry>,
    error: ClientError,
    method: &'static str,
) {
    // Clear each affected batch id one by one, best effort.
    for id in batch_ids {
        let _ = client.clear_cache(Some(*id)).await;
    }
    send_errors(error, entries);
    metrics::counter!("tgi_batch_inference_failure", "method" => method).increment(1);
}

/// What the batching task should do with a cached batch before issuing the
/// next decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FilterPlan {
    /// Every request in the batch is still active: forward the batch as-is.
    Unchanged,
    /// Some requests disappeared: retain only the deduplicated `keep` ids.
    Filter { batch_id: u64, keep: Vec<u64> },
    /// Every request disappeared: drop the shard cache and skip the decode.
    ClearAll { batch_id: u64 },
}

/// Pure decision function describing how a cached batch must be filtered given
/// the entries still tracked by the batching task. An id is kept only when the
/// entry exists, is not in a terminal state and its response channel is still
/// open; duplicates in `request_ids` collapse into a single retained id.
pub(crate) fn plan_filter(cached_batch: &CachedBatch, entries: &IntMap<u64, Entry>) -> FilterPlan {
    let batch_id = cached_batch.id;
    let mut keep: Vec<u64> = Vec::with_capacity(cached_batch.request_ids.len());
    for id in &cached_batch.request_ids {
        let active = entries
            .get(id)
            .map(|entry| !entry.state.is_terminal() && !entry.response_tx.is_closed())
            .unwrap_or(false);
        if active && !keep.contains(id) {
            keep.push(*id);
        }
    }

    if keep.is_empty() {
        FilterPlan::ClearAll { batch_id }
    } else if keep.len() == cached_batch.request_ids.len() {
        FilterPlan::Unchanged
    } else {
        FilterPlan::Filter { batch_id, keep }
    }
}

/// Filter a `batch` and remove all requests not present in `entries`.
///
/// Cache management failures are surfaced to the caller instead of panicking
/// so the whole batch can be discarded gracefully.
#[instrument(skip_all)]
async fn filter_batch(
    client: &mut ShardedClient,
    next_batch: Option<CachedBatch>,
    entries: &IntMap<u64, Entry>,
) -> Option<CachedBatch> {
    let batch = next_batch?;

    match plan_filter(&batch, entries) {
        FilterPlan::Unchanged => Some(batch),
        FilterPlan::ClearAll { batch_id } => {
            // All requests have been filtered out: clear the Python shards
            // cache. A failure here means the whole batch must be discarded.
            if let Err(err) = client.clear_cache(Some(batch_id)).await {
                tracing::error!("clear_cache failed for batch {batch_id}: {err}");
            }
            None
        }
        FilterPlan::Filter { batch_id, keep } => match client.filter_batch(batch_id, keep).await {
            Ok(filtered) => filtered,
            Err(err) => {
                tracing::error!("filter_batch failed for batch {batch_id}: {err}");
                let _ = client.clear_cache(Some(batch_id)).await;
                None
            }
        },
    }
}

/// Send one or multiple `InferStreamResponse` to Infer for all `entries`
/// and filter entries
#[instrument(skip_all)]
fn filter_send_generations(generations: Vec<Generation>, entries: &mut IntMap<u64, Entry>) {
    generations.into_iter().for_each(|generation| {
        let id = generation.request_id;
        // Get entry. A missing id means the request was cancelled or already
        // finished: the shard may still emit tokens for it, but they must not
        // be sent.
        let Some(entry) = entries.get(&id) else {
            tracing::debug!(id, "Dropping generation for unknown/cancelled entry");
            return;
        };

        // Cancellation wins over completion: never send anything (not even an
        // `End`) on a closed or terminal entry. The dropped metric is recorded
        // at most once per request, right here at the send checkpoint.
        if entry.state.is_terminal() || entry.response_tx.is_closed() {
            tracing::debug!(id, "Dropping generation for cancelled entry");
            if let Some(mut entry) = entries.remove(&id) {
                if !entry.state.is_terminal() {
                    entry.state =
                        EntryState::Cancelled(CancelReason::ClientDisconnected);
                    metrics::counter!("tgi_request_failure", "err" => "dropped").increment(1);
                }
            }
            return;
        }

        // Create and enter a span to link this function back to the entry
        let _span = info_span!(parent: entry.temp_span.as_ref().expect("batch_span is None. This is a bug."), "send_generation", generation = ?generation).entered();
        // Send generation responses back to the infer task
        // If the receive an error from the Flume channel, it means that the
        // client dropped the request and we need to stop generating hence why
        // we unwrap_or(true)
        let send_result = send_responses(generation, entry);
        let failed = send_result.is_err();
        let stopped = send_result.unwrap_or(true);
        if failed {
            tracing::error!("Entry response channel error.");
            metrics::counter!("tgi_request_failure", "err" => "dropped").increment(1);
        }
        if stopped {
            if let Some(mut entry) = entries.remove(&id) {
                entry.state = if failed {
                    EntryState::Cancelled(CancelReason::ClientDisconnected)
                } else {
                    EntryState::Finished
                };
            }
        }
    });
}

/// Send responses through the `entry` response channel
fn send_responses(
    generation: Generation,
    entry: &Entry,
) -> Result<bool, Box<SendError<Result<InferStreamResponse, InferError>>>> {
    // Return directly if the channel is disconnected. Dropped accounting is
    // owned by the caller (`filter_send_generations`) so it is recorded once.
    if entry.response_tx.is_closed() {
        return Ok(true);
    }

    let mut stopped = false;

    if let Some(prefill_tokens) = generation.prefill_tokens {
        // Create Token objects
        // We do that here instead of in the Python code as Rust for loops are
        // faster
        let prefill_tokens = prefill_tokens
            .ids
            .into_iter()
            .zip(prefill_tokens.logprobs)
            .zip(prefill_tokens.texts)
            .map(|((id, logprob), text)| PrefillToken { id, text, logprob })
            .collect();

        // Send message
        entry
            .response_tx
            .send(Ok(InferStreamResponse::Prefill(prefill_tokens)))?;
    }

    // Create last Token
    let tokens_ = generation.tokens.expect("Non empty tokens in generation");
    let n = tokens_.ids.len();
    metrics::histogram!("tgi_request_skipped_tokens").record((n - 1) as f64);
    let mut iterator = tokens_
        .ids
        .into_iter()
        .zip(tokens_.logprobs)
        .zip(tokens_.texts)
        .zip(tokens_.is_special)
        .enumerate()
        .peekable();
    while let Some((i, (((id, logprob), text), special))) = iterator.next() {
        let token = Token {
            id,
            text,
            logprob,
            special,
        };
        let top_tokens = if let Some(top_tokens_) = generation.top_tokens.get(i) {
            top_tokens_
                .ids
                .iter()
                .zip(top_tokens_.logprobs.iter())
                .zip(top_tokens_.texts.iter())
                .zip(top_tokens_.is_special.iter())
                .map(|(((&id, &logprob), text), &special)| Token {
                    id,
                    text: text.to_string(),
                    logprob,
                    special,
                })
                .collect()
        } else {
            vec![]
        };
        match (&generation.generated_text, iterator.peek()) {
            (Some(generated_text), None) => {
                // Generation has ended
                stopped = true;
                // Send message
                entry.response_tx.send(Ok(InferStreamResponse::End {
                    token,
                    top_tokens,
                    generated_text: GeneratedText::from(generated_text.clone()),
                    queued: entry.queue_time,
                    start: entry.batch_time.unwrap(),
                }))?;
            }
            _ => {
                // Send message
                entry
                    .response_tx
                    .send(Ok(InferStreamResponse::Intermediate { token, top_tokens }))?;
            }
        }
    }

    Ok(stopped)
}

/// Send errors to Infer for all `entries`
#[instrument(skip_all)]
fn send_errors(error: ClientError, entries: &mut IntMap<u64, Entry>) {
    entries.drain().for_each(|(_, mut entry)| {
        // Cancelled entries only have their resources reaped: they must not
        // receive an additional error and dropped is counted at most once.
        if entry.state.is_terminal() || entry.response_tx.is_closed() {
            tracing::debug!("Skipping error send for cancelled entry {}", entry.id);
            return;
        }
        // Create and enter a span to link this function back to the entry
        let _send_error_span = info_span!(parent: entry.temp_span.as_ref().expect("batch_span is None. This is a bug."), "send_error").entered();
        let err = InferError::GenerationError(error.to_string());
        metrics::counter!("tgi_request_failure", "err" => "generation").increment(1);
        tracing::error!("{err}");
        entry.state = EntryState::Finished;

        // unwrap_or is valid here as we don't care if the receiver is gone.
        entry.response_tx.send(Err(err)).unwrap_or(());
    });
}

impl From<crate::client::GeneratedText> for GeneratedText {
    fn from(value: crate::client::GeneratedText) -> Self {
        let v3_finish_reason = crate::client::FinishReason::try_from(value.finish_reason).unwrap();
        let finish_reason = match v3_finish_reason {
            crate::client::FinishReason::Length => FinishReason::Length,
            crate::client::FinishReason::EosToken => FinishReason::EndOfSequenceToken,
            crate::client::FinishReason::StopSequence => FinishReason::StopSequence,
        };

        Self {
            text: value.text,
            generated_tokens: value.generated_tokens,
            finish_reason,
            seed: value.seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::CachedBatch;

    fn cached_batch(id: u64, request_ids: Vec<u64>) -> CachedBatch {
        CachedBatch {
            id,
            request_ids,
            size: 0,
            max_tokens: 0,
            current_tokens: 0,
        }
    }

    fn open_entry(
        id: u64,
    ) -> (
        Entry,
        mpsc::UnboundedReceiver<Result<InferStreamResponse, InferError>>,
    ) {
        let (response_tx, response_rx) = mpsc::unbounded_channel();
        let entry = Entry {
            id,
            priority: 1,
            state: EntryState::Running,
            skipped: 0,
            forced: false,
            request: tests_support::valid_request(),
            response_tx,
            span: Span::none(),
            temp_span: None,
            queue_time: Instant::now(),
            batch_time: None,
            block_allocation: None,
        };
        (entry, response_rx)
    }

    #[test]
    fn plan_filter_clear_all_when_every_channel_is_closed() {
        let (entry0, _guard0) = open_entry(0);
        let (entry1, _guard1) = open_entry(1);
        let mut entries = IntMap::default();
        entries.insert(0, entry0);
        entries.insert(1, entry1);
        // Dropping both receivers closes the response channels.
        drop(_guard0);
        drop(_guard1);

        let plan = plan_filter(&cached_batch(7, vec![0, 1]), &entries);
        assert_eq!(plan, FilterPlan::ClearAll { batch_id: 7 });
    }

    #[test]
    fn plan_filter_keeps_deduplicated_active_ids() {
        let (entry0, _guard0) = open_entry(0);
        let (entry1, _guard1) = open_entry(1);
        let mut entries = IntMap::default();
        entries.insert(0, entry0);
        entries.insert(1, entry1);
        drop(_guard0);

        let plan = plan_filter(&cached_batch(7, vec![0, 1, 1]), &entries);
        assert_eq!(
            plan,
            FilterPlan::Filter {
                batch_id: 7,
                keep: vec![1]
            }
        );
    }

    #[test]
    fn plan_filter_unchanged_when_all_active() {
        let (entry0, _guard0) = open_entry(0);
        let (entry1, _guard1) = open_entry(1);
        let mut entries = IntMap::default();
        entries.insert(0, entry0);
        entries.insert(1, entry1);

        let plan = plan_filter(&cached_batch(7, vec![0, 1]), &entries);
        assert_eq!(plan, FilterPlan::Unchanged);
        // Guards must outlive the plan check.
        drop(_guard0);
        drop(_guard1);
    }

    /// Minimal request builder for plan_filter tests without pulling the full
    /// router validation machinery.
    mod tests_support {
        use std::sync::Arc;
        use text_generation_router::validation::{
            ValidGenerateRequest, ValidParameters, ValidStoppingParameters,
        };

        pub(super) fn valid_request() -> ValidGenerateRequest {
            ValidGenerateRequest {
                inputs: vec![],
                input_ids: Some(Arc::new(vec![])),
                input_length: 1,
                add_special_tokens: true,
                truncate: 0,
                decoder_input_details: false,
                parameters: ValidParameters {
                    temperature: 0.0,
                    top_k: 0,
                    top_p: 0.0,
                    typical_p: 0.0,
                    do_sample: false,
                    seed: 0,
                    repetition_penalty: 0.0,
                    frequency_penalty: 0.0,
                    watermark: false,
                    grammar: None,
                },
                stopping_parameters: ValidStoppingParameters {
                    ignore_eos_token: false,
                    max_new_tokens: 1,
                    max_total_new_tokens: 1024,
                    stop_sequences: vec![],
                },
                top_n_tokens: 0,
                adapter_id: None,
            }
        }
    }

    #[tokio::test]
    async fn schedule_rejects_with_queue_full_then_accepts_again() {
        let config = QueueConfig {
            max_queue_size: 1,
            ..crate::queue::QueueConfig::default()
        };
        let backend = BackendV3::for_schedule_tests(config, ShardedClient::empty_for_tests());

        let first_stream = backend.schedule(tests_support::valid_request());
        assert!(first_stream.is_ok(), "first request must be admitted");

        // The second request hits the capacity limit.
        let err = backend
            .schedule(tests_support::valid_request())
            .expect_err("second request must be rejected");
        assert!(matches!(err, InferError::QueueFull { capacity: 1 }));
        assert_eq!(err.error_type(), "queue_full");

        // Disconnect the first request's stream: the queue task reaps it at
        // the next scan, freeing the slot.
        drop(first_stream);
        let queue = backend.queue();
        // Park-and-reap: a scan observes the closed channel and drops the
        // entry without producing a batch.
        assert!(queue.next_batch(None, None, 1, 1).await.unwrap().is_none());
        assert_eq!(queue.len(), 0);

        // The queue accepts new work again.
        let third_stream = backend.schedule(tests_support::valid_request());
        assert!(third_stream.is_ok(), "queue must admit after reaping");
        let stats = queue.stats().await.unwrap();
        assert_eq!(stats.queued, 1);
        assert_eq!(stats.dropped_queued, 1);
        drop(third_stream);
    }

    #[tokio::test]
    async fn schedule_rejects_oversized_request() {
        let mut request = tests_support::valid_request();
        request.input_length = 32;
        request.stopping_parameters.max_new_tokens = 32;
        let config = QueueConfig {
            max_batch_total_tokens: 16,
            max_batch_prefill_tokens: 16,
            ..crate::queue::QueueConfig::default()
        };
        let backend = BackendV3::for_schedule_tests(config, ShardedClient::empty_for_tests());

        let err = backend.schedule(request).expect_err("oversized request");
        match err {
            InferError::RequestTooLarge { tokens, limit } => {
                assert_eq!(tokens, 64);
                assert_eq!(limit, 16);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
