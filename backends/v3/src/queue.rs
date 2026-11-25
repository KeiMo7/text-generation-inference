use crate::block_allocator::{BlockAllocation, BlockAllocator};
use crate::client;
use crate::client::{
    Batch, GrammarType, NextTokenChooserParameters, Request, StoppingCriteriaParameters,
};
use nohash_hasher::{BuildNoHashHasher, IntMap};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use text_generation_router::infer::InferError;
use text_generation_router::infer::InferStreamResponse;
use text_generation_router::usage_stats::Env;
use text_generation_router::validation::{
    Chunk, ChunksToString, ValidGenerateRequest, ValidGrammar, ValidParameters,
    ValidStoppingParameters,
};
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use tracing::{info_span, instrument, Span};

/// Default maximum number of entries that can wait in the queue.
pub(crate) const DEFAULT_MAX_QUEUE_SIZE: usize = 4096;
/// Default number of times an entry can be skipped before being forced.
pub(crate) const DEFAULT_MAX_SKIPS: u8 = 8;
/// Default aging interval used to bump the effective priority of old entries.
pub(crate) const DEFAULT_AGING_INTERVAL: Duration = Duration::from_millis(50);
/// Default delay after which the oldest entry is exempted from `min_size`.
pub(crate) const DEFAULT_MAX_QUEUE_DELAY: Duration = Duration::from_millis(250);
/// Default delay before the batching task retries a non-empty queue.
pub(crate) const DEFAULT_RETRY_INTERVAL: Duration = Duration::from_millis(5);

/// Lowest/highest supported priority. Priority 0 is the most urgent.
pub(crate) const MIN_PRIORITY: u8 = 0;
pub(crate) const MAX_PRIORITY: u8 = 3;
/// Priority assigned to requests that do not carry one explicitly.
pub(crate) const DEFAULT_PRIORITY: u8 = 1;

/// Scheduler configuration.
#[derive(Debug, Clone)]
pub(crate) struct QueueConfig {
    pub requires_padding: bool,
    pub block_size: u32,
    pub prefix_caching: bool,
    pub window_size: Option<u32>,
    pub speculate: u32,
    pub max_batch_total_tokens: u32,
    pub max_batch_prefill_tokens: u32,
    /// Kept as part of the documented scheduler configuration; the batching
    /// task passes the effective value through `next_batch(max_size, ...)`.
    #[allow(dead_code)]
    pub max_batch_size: Option<usize>,
    pub support_chunking: bool,
    pub max_queue_size: usize,
    pub max_queue_delay: Duration,
    pub aging_interval: Duration,
    pub max_skips: u8,
    pub retry_interval: Duration,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            requires_padding: false,
            block_size: 1,
            prefix_caching: false,
            window_size: None,
            speculate: 0,
            max_batch_total_tokens: 16,
            max_batch_prefill_tokens: 16,
            max_batch_size: None,
            support_chunking: false,
            max_queue_size: DEFAULT_MAX_QUEUE_SIZE,
            max_queue_delay: DEFAULT_MAX_QUEUE_DELAY,
            aging_interval: DEFAULT_AGING_INTERVAL,
            max_skips: DEFAULT_MAX_SKIPS,
            retry_interval: DEFAULT_RETRY_INTERVAL,
        }
    }
}

/// Errors returned by the [`Queue`] scheduler.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum QueueError {
    /// The queue already holds `capacity` reserved entries.
    #[error("Queue is full (capacity: {capacity})")]
    Full { capacity: usize },
    /// The request needs `tokens` tokens but the batch limit is `limit`.
    #[error("Request is too large: {tokens} tokens > maximum of {limit}")]
    RequestTooLarge { tokens: u32, limit: u32 },
    /// The queue background task has shut down.
    #[error("Queue is shutting down")]
    Shutdown,
}

/// Why a request entered the `Cancelled` terminal state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CancelReason {
    ClientDisconnected,
    Explicit,
    QueueShutdown,
}

/// Lifecycle state of a queue entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EntryState {
    Queued,
    Running,
    Finished,
    Cancelled(CancelReason),
}

impl EntryState {
    pub(crate) fn is_terminal(self) -> bool {
        matches!(self, EntryState::Finished | EntryState::Cancelled(_))
    }
}

/// Outcome of a [`Queue::cancel`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CancelOutcome {
    /// The entry was waiting in the queue and has been cancelled.
    CancelledQueued { id: u64 },
    /// The id is not currently in the `Queued` state. This does **not** mean
    /// that the id never existed: it may be running, finished or already
    /// cancelled. Repeated cancels always resolve to this variant.
    NotFound { id: u64 },
}

/// Scheduler counters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct QueueStats {
    /// Entries currently waiting in the queue.
    pub queued: usize,
    /// Entries admitted to the queue since startup.
    pub admitted: u64,
    /// Entries moved to the running state since startup.
    pub scheduled: u64,
    /// Entries rejected because the queue was full.
    pub rejected_full: u64,
    /// Entries rejected for being too large.
    pub rejected_too_large: u64,
    /// Entries explicitly cancelled while queued.
    pub cancelled_queued: u64,
    /// Entries dropped while queued because their client disconnected.
    pub dropped_queued: u64,
    /// Number of batches produced.
    pub batches: u64,
    /// Largest batch produced so far.
    pub max_batch_size: usize,
}

/// Report returned by [`Queue::shutdown`].
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct ShutdownReport {
    pub cancelled: u64,
    pub stats: QueueStats,
}

/// Queue entry
#[derive(Debug)]
pub(crate) struct Entry {
    /// Monotonic, never reused scheduler id. It drives scheduling and logging
    /// and, because `ValidGenerateRequest` carries no identifier of its own,
    /// is also the correlation key placed on `Request.id` so the shard's
    /// `request_id` responses can be routed back to this entry. It is never
    /// derived from request parameters.
    pub id: u64,
    /// Priority in the `0..=3` range. Lower is more urgent.
    pub priority: u8,
    /// Current lifecycle state.
    pub state: EntryState,
    /// Number of batch scans this entry survived because it could not fit.
    pub skipped: u8,
    /// When set, the entry must be scheduled on its own at the next scan.
    pub forced: bool,
    /// Request
    pub request: ValidGenerateRequest,
    /// Response sender to communicate between the Infer struct and the batching_task
    pub response_tx: mpsc::UnboundedSender<Result<InferStreamResponse, InferError>>,
    /// Span that will live as long as entry
    pub span: Span,
    /// Temporary span used as a guard when logging inference, wait times...
    pub temp_span: Option<Span>,
    /// Instant when this entry was queued
    pub queue_time: Instant,
    /// Instant when this entry was added to a batch
    pub batch_time: Option<Instant>,
    /// Block Allocation
    pub block_allocation: Option<BlockAllocation>,
}

impl Entry {
    /// Total token footprint used for admission control.
    fn admission_tokens(&self, speculate: u32) -> u32 {
        self.request
            .input_length
            .saturating_add(self.request.stopping_parameters.max_new_tokens)
            .saturating_add(speculate)
    }
}

/// Counters shared between the [`Queue`] handle and its background task.
#[derive(Debug, Default)]
struct SharedCounters {
    /// Monotonic id generator. Ids are never reused.
    next_id: AtomicU64,
    /// Number of reserved slots (admitted or in-flight append commands).
    slots_used: AtomicUsize,
    admitted: AtomicU64,
    scheduled: AtomicU64,
    rejected_full: AtomicU64,
    rejected_too_large: AtomicU64,
    cancelled_queued: AtomicU64,
    dropped_queued: AtomicU64,
    batches: AtomicU64,
    max_batch_size: AtomicUsize,
}

/// Request Queue
#[derive(Debug, Clone)]
pub(crate) struct Queue {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommand>,
    /// Scheduler configuration
    config: Arc<QueueConfig>,
    /// Shared counters
    counters: Arc<SharedCounters>,
}

impl Queue {
    pub(crate) fn new(config: QueueConfig) -> Self {
        // Create channel
        let (queue_sender, queue_receiver) = mpsc::unbounded_channel();

        let config = Arc::new(config);
        let counters = Arc::new(SharedCounters::default());

        // Launch background queue task
        tokio::spawn(queue_task(config.clone(), counters.clone(), queue_receiver));

        Self {
            queue_sender,
            config,
            counters,
        }
    }

    /// Append an entry to the queue.
    ///
    /// Admission is performed on the handle: a monotonic id is allocated first
    /// (and never recycled on rejection), then the request size and queue
    /// capacity are checked. A slot is reserved before the command is handed to
    /// the background task and rolled back if the command cannot be delivered.
    #[instrument(skip_all)]
    pub(crate) fn append(&self, mut entry: Entry) -> Result<u64, QueueError> {
        // Clamp priority to the supported range.
        entry.priority = entry.priority.clamp(MIN_PRIORITY, MAX_PRIORITY);

        // Allocate a monotonic, never reused id before any admission check so
        // that rejected requests do not recycle ids.
        let id = self.counters.next_id.fetch_add(1, Ordering::Relaxed);
        entry.id = id;
        entry.state = EntryState::Queued;
        entry.skipped = 0;
        entry.forced = false;

        // Admission: total token footprint must fit inside a single batch.
        let tokens = entry.admission_tokens(self.config.speculate);
        if tokens > self.config.max_batch_total_tokens {
            self.counters
                .rejected_too_large
                .fetch_add(1, Ordering::Relaxed);
            metrics::counter!("tgi_request_too_large").increment(1);
            return Err(QueueError::RequestTooLarge {
                tokens,
                limit: self.config.max_batch_total_tokens,
            });
        }

        // Without chunking the whole prefill must fit in the prefill budget.
        if !self.config.support_chunking
            && entry.request.input_length > self.config.max_batch_prefill_tokens
        {
            self.counters
                .rejected_too_large
                .fetch_add(1, Ordering::Relaxed);
            metrics::counter!("tgi_request_too_large").increment(1);
            return Err(QueueError::RequestTooLarge {
                tokens: entry.request.input_length,
                limit: self.config.max_batch_prefill_tokens,
            });
        }

        // Reserve a slot.
        let reserved = self.counters.slots_used.fetch_add(1, Ordering::AcqRel);
        if reserved >= self.config.max_queue_size {
            // Roll back the reservation.
            self.counters.slots_used.fetch_sub(1, Ordering::AcqRel);
            self.counters.rejected_full.fetch_add(1, Ordering::Relaxed);
            metrics::counter!("tgi_queue_full").increment(1);
            return Err(QueueError::Full {
                capacity: self.config.max_queue_size,
            });
        }

        // Hand the entry to the background task managing the queue state.
        if let Err(send_err) = self.queue_sender.send(QueueCommand::Append {
            entry: Box::new(entry),
            span: Span::current(),
        }) {
            // The task is gone: roll back the slot reservation.
            self.counters.slots_used.fetch_sub(1, Ordering::AcqRel);
            // Reclaim the entry so its block allocation (if any) is dropped.
            let QueueCommand::Append { entry, .. } = send_err.0 else {
                unreachable!("send error returns the exact command that failed");
            };
            drop(entry);
            return Err(QueueError::Shutdown);
        }

        Ok(id)
    }

    /// Number of entries currently reserving a queue slot.
    pub(crate) fn len(&self) -> usize {
        self.counters.slots_used.load(Ordering::Acquire)
    }

    // Get the next batch
    #[instrument(skip(self))]
    pub(crate) async fn next_batch(
        &self,
        min_size: Option<usize>,
        max_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
    ) -> Result<Option<NextBatch>, QueueError> {
        if prefill_token_budget == 0 || token_budget == 0 {
            return Ok(None);
        };

        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send next batch command to the background task managing the state
        if self
            .queue_sender
            .send(QueueCommand::NextBatch {
                min_size,
                max_size,
                prefill_token_budget,
                token_budget,
                now: Instant::now(),
                response_sender,
                span: Span::current(),
            })
            .is_err()
        {
            return Err(QueueError::Shutdown);
        }
        // Await on response channel
        response_receiver.await.unwrap_or(Err(QueueError::Shutdown))
    }

    /// Test-only variant of [`Queue::next_batch`] with an injected clock.
    #[cfg(test)]
    pub(crate) async fn next_batch_at(
        &self,
        now: Instant,
        min_size: Option<usize>,
        max_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
    ) -> Result<Option<NextBatch>, QueueError> {
        if prefill_token_budget == 0 || token_budget == 0 {
            return Ok(None);
        };

        let (response_sender, response_receiver) = oneshot::channel();
        if self
            .queue_sender
            .send(QueueCommand::NextBatch {
                min_size,
                max_size,
                prefill_token_budget,
                token_budget,
                now,
                response_sender,
                span: Span::current(),
            })
            .is_err()
        {
            return Err(QueueError::Shutdown);
        }
        response_receiver.await.unwrap_or(Err(QueueError::Shutdown))
    }

    /// Cancel a queued entry. Only entries currently in the `Queued` state can
    /// be cancelled through this path; running, finished, already cancelled or
    /// unknown ids resolve to [`CancelOutcome::NotFound`]. The call is
    /// idempotent.
    #[instrument(skip(self))]
    pub(crate) async fn cancel(&self, id: u64) -> Result<CancelOutcome, QueueError> {
        let (response_sender, response_receiver) = oneshot::channel();
        if self
            .queue_sender
            .send(QueueCommand::Cancel {
                id,
                response_sender,
            })
            .is_err()
        {
            return Err(QueueError::Shutdown);
        }
        response_receiver.await.unwrap_or(Err(QueueError::Shutdown))
    }

    /// Snapshot of the scheduler counters.
    #[instrument(skip(self))]
    pub(crate) async fn stats(&self) -> Result<QueueStats, QueueError> {
        let (response_sender, response_receiver) = oneshot::channel();
        if self
            .queue_sender
            .send(QueueCommand::Stats { response_sender })
            .is_err()
        {
            return Err(QueueError::Shutdown);
        }
        response_receiver.await.unwrap_or(Err(QueueError::Shutdown))
    }

    /// Drain the queue, cancel every queued entry and shut the task down.
    /// Pending `next_batch` and `cancel` calls resolve to
    /// [`QueueError::Shutdown`].
    #[instrument(skip(self))]
    pub(crate) async fn shutdown(&self) -> Result<ShutdownReport, QueueError> {
        let (response_sender, response_receiver) = oneshot::channel();
        if self
            .queue_sender
            .send(QueueCommand::Shutdown { response_sender })
            .is_err()
        {
            return Err(QueueError::Shutdown);
        }
        response_receiver.await.unwrap_or(Err(QueueError::Shutdown))
    }
}

// Background task responsible of the queue state
async fn queue_task(
    config: Arc<QueueConfig>,
    counters: Arc<SharedCounters>,
    mut receiver: mpsc::UnboundedReceiver<QueueCommand>,
) {
    let mut state = State::new(config, counters);

    while let Some(cmd) = receiver.recv().await {
        match cmd {
            QueueCommand::Append { entry, span } => {
                span.in_scope(|| state.append(*entry));
            }
            QueueCommand::NextBatch {
                min_size,
                max_size,
                prefill_token_budget,
                token_budget,
                now,
                response_sender,
                span,
            } => {
                // An empty queue always answers immediately with an empty
                // batch: waiting and retrying are the batching task's job,
                // driven by `Notify` and the retry interval.
                state
                    .answer_next_batch(
                        response_sender,
                        min_size,
                        max_size,
                        prefill_token_budget,
                        token_budget,
                        now,
                    )
                    .await;
                drop(span);
            }
            QueueCommand::Cancel {
                id,
                response_sender,
            } => {
                let outcome = state.cancel(id);
                if response_sender.send(Ok(outcome)).is_err() {
                    tracing::debug!("cancel response channel closed before reply");
                }
            }
            QueueCommand::Stats { response_sender } => {
                if response_sender.send(Ok(state.stats())).is_err() {
                    tracing::debug!("stats response channel closed before reply");
                }
            }
            QueueCommand::Shutdown { response_sender } => {
                let report = state.shutdown();
                let _ = response_sender.send(Ok(report));
                // Dropping the receiver ends the task; remaining Entries (all
                // terminal) drop their block allocations through `Drop`.
                return;
            }
        }
    }

    // All queue handles were dropped without an explicit shutdown: drain the
    // queue the same way.
    state.shutdown();
}

/// Queue State
#[derive(Debug)]
struct State {
    /// Queue entries. Ids are strictly increasing while the queue is at rest.
    entries: Vec<Entry>,

    /// Id of the next batch
    next_batch_id: u64,

    /// Scheduler configuration
    config: Arc<QueueConfig>,

    /// Paged Attention block size
    block_size: u32,

    /// Speculation amount
    speculate: u32,

    /// Whether the model allow the prefill chunking
    /// If it does, the last request in the batch will be split to exactly
    /// match the prefill token budget
    support_chunking: bool,

    /// Paged Attention Block Allocation
    block_allocator: Option<BlockAllocator>,

    /// indicate if it's hpu device, the hpu device needs padding to generate
    /// first token.
    is_hpu_device: bool,

    /// Shared scheduler counters
    counters: Arc<SharedCounters>,
}

impl State {
    fn new(config: Arc<QueueConfig>, counters: Arc<SharedCounters>) -> Self {
        let block_allocator = (!config.requires_padding).then(|| {
            BlockAllocator::new(
                config.max_batch_total_tokens,
                config.block_size,
                config.prefix_caching,
                config.window_size,
            )
        });

        Self {
            entries: Vec::with_capacity(128),
            next_batch_id: 0,
            block_size: config.block_size,
            speculate: config.speculate,
            support_chunking: config.support_chunking,
            block_allocator,
            is_hpu_device: Env::new().is_hpu_device(),
            config,
            counters,
        }
    }

    /// Test-only constructor allowing an injected block allocator.
    #[cfg(test)]
    fn for_tests(
        config: QueueConfig,
        counters: Arc<SharedCounters>,
        block_allocator: Option<BlockAllocator>,
    ) -> Self {
        let block_size = config.block_size;
        let speculate = config.speculate;
        let support_chunking = config.support_chunking;
        Self {
            entries: Vec::with_capacity(128),
            next_batch_id: 0,
            block_size,
            speculate,
            support_chunking,
            block_allocator,
            is_hpu_device: false,
            config: Arc::new(config),
            counters,
        }
    }

    /// Test-only append that also reserves a slot and assigns a monotonic id
    /// from the shared counter, mirroring what [`Queue::append`] does before
    /// handing the command to the task.
    #[cfg(test)]
    fn reserve_and_append(&mut self, mut entry: Entry) {
        self.counters.slots_used.fetch_add(1, Ordering::AcqRel);
        entry.id = self.counters.next_id.fetch_add(1, Ordering::Relaxed);
        entry.state = EntryState::Queued;
        entry.skipped = 0;
        entry.forced = false;
        self.append(entry);
    }

    /// Append an entry to the queue
    fn append(&mut self, mut entry: Entry) {
        debug_assert_eq!(entry.state, EntryState::Queued);
        // Create a span that will live as long as the entry is in the queue
        // waiting to be batched
        let queue_span = info_span!(parent: &entry.span, "queued");
        entry.temp_span = Some(queue_span);

        self.counters.admitted.fetch_add(1, Ordering::Relaxed);
        // Push entry in the queue. The canonical (strictly increasing id)
        // ordering is restored by `rebuild_queue` after every scan; an append
        // interleaved between scans only extends the vector.
        self.entries.push(entry);
        self.set_queue_gauge();
    }

    /// Cancel an entry currently in the `Queued` state.
    fn cancel(&mut self, id: u64) -> CancelOutcome {
        let Some(position) = self.entries.iter().position(|entry| entry.id == id) else {
            return CancelOutcome::NotFound { id };
        };

        let mut entry = self.entries.remove(position);
        entry.state = EntryState::Cancelled(CancelReason::Explicit);
        self.release_slot();
        self.counters
            .cancelled_queued
            .fetch_add(1, Ordering::Relaxed);
        metrics::counter!("tgi_request_cancelled").increment(1);
        // Dropping the entry releases any block allocation through `Drop`.
        drop(entry);
        self.debug_assert_id_ordering();
        CancelOutcome::CancelledQueued { id }
    }

    /// Snapshot the scheduler counters.
    fn stats(&self) -> QueueStats {
        QueueStats {
            queued: self.entries.len(),
            admitted: self.counters.admitted.load(Ordering::Relaxed),
            scheduled: self.counters.scheduled.load(Ordering::Relaxed),
            rejected_full: self.counters.rejected_full.load(Ordering::Relaxed),
            rejected_too_large: self.counters.rejected_too_large.load(Ordering::Relaxed),
            cancelled_queued: self.counters.cancelled_queued.load(Ordering::Relaxed),
            dropped_queued: self.counters.dropped_queued.load(Ordering::Relaxed),
            batches: self.counters.batches.load(Ordering::Relaxed),
            max_batch_size: self.counters.max_batch_size.load(Ordering::Relaxed),
        }
    }

    /// Cancel every queued entry and prepare the shutdown report.
    fn shutdown(&mut self) -> ShutdownReport {
        let mut cancelled = 0u64;
        for mut entry in self.entries.drain(..) {
            if !entry.state.is_terminal() {
                entry.state = EntryState::Cancelled(CancelReason::QueueShutdown);
                cancelled += 1;
                self.counters
                    .cancelled_queued
                    .fetch_add(1, Ordering::Relaxed);
                metrics::counter!("tgi_request_cancelled").increment(1);
            }
            // Entries drop here, releasing their block allocations once.
        }
        // Every reserved slot is now gone.
        self.counters.slots_used.store(0, Ordering::Release);
        self.set_queue_gauge();
        ShutdownReport {
            cancelled,
            stats: self.stats(),
        }
    }

    fn release_slot(&mut self) {
        let previous = self.counters.slots_used.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "slot released without a reservation");
        self.set_queue_gauge();
    }

    fn set_queue_gauge(&self) {
        metrics::gauge!("tgi_queue_size").set(self.entries.len() as f64);
    }

    #[cfg(debug_assertions)]
    fn debug_assert_id_ordering(&self) {
        let mut previous = None;
        for entry in &self.entries {
            if let Some(previous) = previous {
                debug_assert!(previous < entry.id, "queue ids must be strictly increasing");
            }
            previous = Some(entry.id);
        }
    }

    #[cfg(not(debug_assertions))]
    fn debug_assert_id_ordering(&self) {}

    /// Run a batch scan and answer a parked `next_batch` reply.
    async fn answer_next_batch(
        &mut self,
        response_sender: oneshot::Sender<Result<Option<NextBatch>, QueueError>>,
        min_size: Option<usize>,
        max_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
        now: Instant,
    ) {
        let result = self
            .next_batch_at(now, min_size, max_size, prefill_token_budget, token_budget)
            .await;
        // A closed reply channel is harmless: the caller stopped waiting.
        if response_sender.send(result).is_err() {
            tracing::debug!("next_batch response channel closed before reply");
        }
    }

    // Get the next batch
    async fn next_batch_at(
        &mut self,
        now: Instant,
        min_size: Option<usize>,
        max_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
    ) -> Result<Option<NextBatch>, QueueError> {
        if self.entries.is_empty() {
            tracing::debug!("No queue");
            return Ok(None);
        }

        if let Some(max_size) = max_size {
            if max_size == 0 {
                tracing::debug!("No capacity");
                return Ok(None);
            }
        }

        // Deadline exemption: the oldest entry waited long enough, ignore
        // `min_size`. A zero delay means the exemption always applies.
        let oldest_wait = self
            .entries
            .iter()
            .filter_map(|entry| now.checked_duration_since(entry.queue_time))
            .max()
            .unwrap_or(Duration::ZERO);
        let deadline_exempt =
            self.config.max_queue_delay.is_zero() || oldest_wait >= self.config.max_queue_delay;

        if !deadline_exempt {
            if let Some(min_size) = min_size {
                if self.entries.len() < min_size {
                    tracing::debug!("Not enough entries");
                    return Ok(None);
                }
            }
        }

        // Pad prefill_token_budget to be a multiple of block size
        let prefill_token_budget = prefill_token_budget.div_ceil(self.block_size) * self.block_size;

        // Sort entries by `(effective_priority, queue_time, id)` using an index
        // vector: the entries themselves are never cloned.
        let mut order: Vec<usize> = (0..self.entries.len()).collect();
        order.sort_by(|&left, &right| {
            let left_entry = &self.entries[left];
            let right_entry = &self.entries[right];
            let left_key = (
                effective_priority(
                    left_entry.priority,
                    left_entry.queue_time,
                    now,
                    self.config.aging_interval,
                ),
                left_entry.queue_time,
                left_entry.id,
            );
            let right_key = (
                effective_priority(
                    right_entry.priority,
                    right_entry.queue_time,
                    now,
                    self.config.aging_interval,
                ),
                right_entry.queue_time,
                right_entry.id,
            );
            left_key.cmp(&right_key)
        });

        // Create span for this batch to add context to inference calls
        let next_batch_span = info_span!(parent: None, "batch", batch_size = tracing::field::Empty);
        next_batch_span.follows_from(Span::current());

        // Entries selected for the batch, in selection order.
        let mut chosen: Vec<ChosenEntry> = Vec::new();
        // Entries that stay queued. The queue is rebuilt from this vector at
        // the end of the scan, so this never touches per-element removal and
        // the whole selection stays O(n log n).
        let mut kept: Vec<Entry> = Vec::with_capacity(self.entries.len());
        let mut max_input_length = 0u32;
        let mut prefill_tokens: u32 = 0;
        let mut decode_tokens: u32 = 0;
        let mut max_blocks = 0u32;

        // A forced entry must be scheduled alone.
        let forced_id = self
            .entries
            .iter()
            .find(|entry| entry.forced)
            .map(|entry| entry.id);

        // Take ownership of every entry behind `Option`s so entries can be
        // moved out without invalidating the scan indices.
        let mut pending: Vec<Option<Entry>> = std::mem::take(&mut self.entries)
            .into_iter()
            .map(Some)
            .collect();

        'scan: for index in order {
            let entry_id = pending[index]
                .as_ref()
                .expect("scan index out of bounds")
                .id;
            if let Some(forced_id) = forced_id {
                if entry_id != forced_id {
                    continue;
                }
            }
            if let Some(max_size) = max_size {
                if chosen.len() >= max_size {
                    break 'scan;
                }
            }

            let mut entry = pending[index].take().expect("entry visited twice");

            // Checkpoint: filter entries whose response receiver was dropped
            // (== entries where the request was dropped by the client).
            if entry.response_tx.is_closed() {
                self.drop_disconnected(entry);
                continue;
            }

            let fit = self
                .fit_entry(
                    &entry,
                    chosen.len(),
                    &mut max_input_length,
                    &mut prefill_tokens,
                    &mut decode_tokens,
                    &mut max_blocks,
                    prefill_token_budget,
                    token_budget,
                )
                .await;

            match fit {
                FitOutcome::Fit {
                    block_allocation,
                    chunk_len,
                } => {
                    entry.block_allocation = block_allocation;
                    chosen.push(ChosenEntry { entry, chunk_len });
                    if forced_id.is_some() {
                        // Forced entries form a singleton batch.
                        break 'scan;
                    }
                }
                FitOutcome::Chunk {
                    block_allocation,
                    chunk_len,
                } => {
                    entry.block_allocation = block_allocation;
                    chosen.push(ChosenEntry {
                        entry,
                        chunk_len: Some(chunk_len),
                    });
                    // The prefill budget is now fully consumed.
                    break 'scan;
                }
                FitOutcome::DoesNotFit => {
                    // Adaptive scan: record the skip and keep trying the
                    // smaller requests behind this one.
                    self.record_skip(&mut entry);
                    kept.push(entry);
                }
            }
        }

        // Entries never visited (or skipped by the forced singleton rule) go
        // back into the queue.
        kept.extend(pending.into_iter().flatten());

        // Empty batch: keep the queue in its canonical order.
        if chosen.is_empty() {
            tracing::debug!("No entry could fit in this batch");
            self.entries = kept;
            self.rebuild_queue();
            return Ok(None);
        }

        // Min size still applies unless the deadline exemption fired.
        if !deadline_exempt {
            if let Some(min_size) = min_size {
                if chosen.len() < min_size {
                    tracing::debug!("Batch too small");
                    for chosen_entry in chosen {
                        let mut entry = chosen_entry.entry;
                        // Nothing left the queue, so the entry goes back to
                        // `Queued` and its tentative allocation is dropped.
                        entry.state = EntryState::Queued;
                        entry.block_allocation = None;
                        entry.batch_time = None;
                        kept.push(entry);
                    }
                    self.entries = kept;
                    self.rebuild_queue();
                    return Ok(None);
                }
            }
        }

        let mut batch_requests = Vec::with_capacity(chosen.len());
        let mut batch_entries =
            IntMap::with_capacity_and_hasher(chosen.len(), BuildNoHashHasher::default());

        let mut scheduled = 0u64;
        for chosen_entry in chosen {
            let ChosenEntry {
                mut entry,
                chunk_len,
            } = chosen_entry;

            // Checkpoint right before the batch is committed: the client may
            // have disconnected while blocks were being allocated.
            if entry.response_tx.is_closed() {
                self.drop_disconnected(entry);
                continue;
            }

            // Create a new span to link the batch back to this entry
            let entry_batch_span = info_span!(parent: &entry.span, "infer");
            // Add relationships
            next_batch_span.follows_from(&entry_batch_span);
            entry_batch_span.follows_from(&next_batch_span);
            // Update entry
            entry.temp_span = Some(entry_batch_span);

            let (blocks, slots, prefix_len) = match &entry.block_allocation {
                None => (Vec::new(), Vec::new(), 0),
                Some(block_allocation) => (
                    block_allocation.blocks.clone(),
                    block_allocation.slots.clone(),
                    block_allocation.prefix_len,
                ),
            };

            // Terminal state transition: leaving `Queued` releases exactly one
            // slot.
            entry.state = EntryState::Running;
            entry.batch_time = Some(Instant::now());
            self.release_slot();
            scheduled += 1;

            let id = entry.id;
            batch_requests.push(Request {
                id,
                prefill_logprobs: entry.request.decoder_input_details,
                input_chunks: Some(client::Input {
                    chunks: entry
                        .request
                        .inputs
                        .clone()
                        .into_iter()
                        .map(|c| client::InputChunk {
                            chunk: Some(match c {
                                Chunk::Text(text) => client::Chunk::Text(text),
                                Chunk::Image(image) => client::Chunk::Image(client::Image {
                                    data: image.data,
                                    mimetype: image.mimetype,
                                }),
                            }),
                        })
                        .collect(),
                }),
                inputs: entry.request.inputs.chunks_to_string(),
                truncate: entry.request.truncate,
                add_special_tokens: entry.request.add_special_tokens,
                parameters: Some(NextTokenChooserParameters::from(
                    entry.request.parameters.clone(),
                )),
                stopping_parameters: Some(StoppingCriteriaParameters::from(
                    entry.request.stopping_parameters.clone(),
                )),
                top_n_tokens: entry.request.top_n_tokens,
                blocks,
                slots,
                cache_len: prefix_len,
                adapter_id: entry.request.adapter_id.clone(),
                chunk_len,
            });
            // Insert in batch_entries IntMap
            batch_entries.insert(id, entry);
        }

        // Every selected entry may have been cancelled at the last checkpoint.
        if batch_requests.is_empty() {
            tracing::debug!("All selected entries were disconnected");
            self.entries = kept;
            self.rebuild_queue();
            return Ok(None);
        }

        self.counters
            .scheduled
            .fetch_add(scheduled, Ordering::Relaxed);
        self.entries = kept;
        self.rebuild_queue();

        // Final batch size
        let size = batch_requests.len() as u32;
        next_batch_span.record("batch_size", size);

        let batch = Batch {
            id: self.next_batch_id,
            requests: batch_requests,
            size,
            max_tokens: (prefill_tokens + decode_tokens),
            max_blocks,
        };
        // Increment batch id
        self.next_batch_id += 1;
        self.counters.batches.fetch_add(1, Ordering::Relaxed);
        let previous_max = self.counters.max_batch_size.load(Ordering::Relaxed);
        if size as usize > previous_max {
            self.counters
                .max_batch_size
                .store(size as usize, Ordering::Relaxed);
        }

        metrics::histogram!("tgi_batch_next_size").record(batch.size as f64);

        Ok(Some((batch_entries, batch, next_batch_span)))
    }

    /// Restore the canonical queue ordering: strictly increasing ids. Sorting
    /// is O(n log n), no per-element removal is performed.
    fn rebuild_queue(&mut self) {
        self.entries.sort_by_key(|entry| entry.id);
        self.debug_assert_id_ordering();
        self.set_queue_gauge();
    }

    /// Mark an entry that could not fit as skipped. Reaching `max_skips` forces
    /// the entry to be scheduled alone at the next scan.
    fn record_skip(&mut self, entry: &mut Entry) {
        entry.skipped = entry.skipped.saturating_add(1);
        metrics::counter!("tgi_batch_skipped_requests").increment(1);
        if entry.skipped >= self.config.max_skips {
            entry.forced = true;
            tracing::debug!(id = entry.id, "Entry forced into a singleton batch");
        }
    }

    /// Transition a disconnected queued entry to its terminal state and
    /// release its slot. The dropped metric is incremented at most once per
    /// request.
    fn drop_disconnected(&mut self, mut entry: Entry) {
        if !entry.state.is_terminal() {
            entry.state = EntryState::Cancelled(CancelReason::ClientDisconnected);
            self.release_slot();
            self.counters.dropped_queued.fetch_add(1, Ordering::Relaxed);
            metrics::counter!("tgi_request_failure", "err" => "dropped").increment(1);
            tracing::debug!(id = entry.id, "Dropping entry");
        }
        // Dropping here releases any block allocation through `Drop` exactly
        // once.
        drop(entry);
    }

    /// Check whether an entry fits the current batch and, when it does,
    /// allocate its blocks. All budget bookkeeping lives in this method so the
    /// scan loop stays allocation-free on the skip path.
    #[allow(clippy::too_many_arguments)]
    async fn fit_entry(
        &mut self,
        entry: &Entry,
        batch_len: usize,
        max_input_length: &mut u32,
        prefill_tokens: &mut u32,
        decode_tokens: &mut u32,
        max_blocks: &mut u32,
        prefill_token_budget: u32,
        token_budget: u32,
    ) -> FitOutcome {
        match &self.block_allocator {
            None => {
                // We pad to max input length in the Python shards
                // We need to take these padding tokens into the equation
                let tentative_max_input = (*max_input_length).max(entry.request.input_length);
                let tentative_prefill = (batch_len + 1) as u32 * tentative_max_input;

                let tentative_decode =
                    *decode_tokens + entry.request.stopping_parameters.max_new_tokens;
                let total_tokens = tentative_prefill + tentative_decode + self.speculate;

                if tentative_prefill > prefill_token_budget || total_tokens > token_budget {
                    tracing::debug!(
                        "Over budget: prefill_tokens={tentative_prefill} > {prefill_token_budget} \
                         || {tentative_prefill} + {tentative_decode} + {} > {token_budget}",
                        self.speculate
                    );
                    return FitOutcome::DoesNotFit;
                }

                *max_input_length = tentative_max_input;
                *prefill_tokens = tentative_prefill;
                *decode_tokens = tentative_decode;
                FitOutcome::Fit {
                    block_allocation: None,
                    chunk_len: None,
                }
            }
            Some(block_allocator) => {
                // If users wants the prefill logprobs, we cannot reuse the
                // cache. So no input_ids for the radix tree.
                let input_ids = if entry.request.decoder_input_details {
                    None
                } else {
                    entry.request.input_ids.clone()
                };

                let tokens = entry.request.input_length
                    + entry.request.stopping_parameters.max_new_tokens
                    + self.speculate
                    - 1;

                let mut block_allocation = match block_allocator.allocate(tokens, input_ids).await {
                    None => {
                        tracing::debug!("Over budget: not enough free blocks");
                        return FitOutcome::DoesNotFit;
                    }
                    Some(block_allocation) => block_allocation,
                };

                *max_blocks = (*max_blocks).max(block_allocation.blocks.len() as u32);

                if block_allocation.prefix_len == entry.request.input_length {
                    // The whole request was found in the radix trie
                    // However, for the transformer forward to work, we need to
                    // have at least one token of postfix.
                    block_allocation.prefix_len -= 1;
                }
                debug_assert!(block_allocation.prefix_len <= entry.request.input_length);

                let postfix_len = entry.request.input_length - block_allocation.prefix_len;

                if *prefill_tokens + postfix_len > prefill_token_budget {
                    if self.support_chunking {
                        // We support chunking: truncate the last entry to
                        // exactly match the remaining prefill budget.
                        let chunk_len = prefill_token_budget.saturating_sub(*prefill_tokens);
                        if chunk_len == 0 {
                            // We cannot prefill even one token for this entry.
                            tracing::debug!("Over budget: no room left for a chunk");
                            return FitOutcome::DoesNotFit;
                        }
                        let chunk_len = chunk_len.min(postfix_len);
                        tracing::debug!(
                            "Matched budget: prefill_tokens={} == {prefill_token_budget}",
                            *prefill_tokens + chunk_len
                        );
                        *prefill_tokens += chunk_len;
                        return FitOutcome::Chunk {
                            block_allocation: Some(block_allocation),
                            chunk_len,
                        };
                    } else {
                        tracing::debug!(
                            "Over budget: prefill_tokens={} > {prefill_token_budget}",
                            *prefill_tokens + postfix_len
                        );
                        return FitOutcome::DoesNotFit;
                    }
                }

                if self.is_hpu_device {
                    // HPU needs to pad for the prefill
                    let tentative_max_input = (*max_input_length).max(entry.request.input_length);
                    let actual_prefill_tokens_for_hpu =
                        (batch_len + 1) as u32 * tentative_max_input;

                    if actual_prefill_tokens_for_hpu > prefill_token_budget {
                        tracing::debug!(
                            "Over budget: prefill_tokens={actual_prefill_tokens_for_hpu} > \
                             {prefill_token_budget}"
                        );
                        return FitOutcome::DoesNotFit;
                    }
                    *max_input_length = tentative_max_input;
                }

                *prefill_tokens += postfix_len;
                // On the paged-attention path the block pool bounds the total
                // token footprint (blocks were requested for
                // `input_length + max_new_tokens + speculate - 1`); matching
                // the original scheduler, `Batch.max_tokens` only carries the
                // prefill amount here.
                FitOutcome::Fit {
                    block_allocation: Some(block_allocation),
                    chunk_len: None,
                }
            }
        }
    }
}

/// Entry selected for a batch together with its optional chunk length.
struct ChosenEntry {
    entry: Entry,
    chunk_len: Option<u32>,
}

/// Result of fitting a single entry against the remaining budget.
enum FitOutcome {
    /// Entry fully fits.
    Fit {
        block_allocation: Option<BlockAllocation>,
        chunk_len: Option<u32>,
    },
    /// Entry only fits by truncating its prefill.
    Chunk {
        block_allocation: Option<BlockAllocation>,
        chunk_len: u32,
    },
    /// Entry must stay in the queue.
    DoesNotFit,
}

/// Compute the effective priority of an entry. Aging bumps the priority of
/// entries that wait for a long time so that starvation is bounded. A zero
/// aging interval disables aging.
fn effective_priority(
    priority: u8,
    queue_time: Instant,
    now: Instant,
    aging_interval: Duration,
) -> u32 {
    let base = priority as u32;
    if aging_interval.is_zero() {
        return base;
    }
    let wait_millis = now
        .saturating_duration_since(queue_time)
        .as_millis()
        .min(u128::from(u32::MAX)) as u32;
    let aging_millis = aging_interval.as_millis() as u32;
    let aged = base + wait_millis / aging_millis.max(1);
    aged.min(MAX_PRIORITY as u32)
}

type NextBatch = (IntMap<u64, Entry>, Batch, Span);

#[derive(Debug)]
#[allow(dead_code)]
enum QueueCommand {
    Append {
        entry: Box<Entry>,
        span: Span,
    },
    NextBatch {
        min_size: Option<usize>,
        max_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
        now: Instant,
        response_sender: oneshot::Sender<Result<Option<NextBatch>, QueueError>>,
        span: Span,
    },
    Cancel {
        id: u64,
        response_sender: oneshot::Sender<Result<CancelOutcome, QueueError>>,
    },
    Stats {
        response_sender: oneshot::Sender<Result<QueueStats, QueueError>>,
    },
    Shutdown {
        response_sender: oneshot::Sender<Result<ShutdownReport, QueueError>>,
    },
}

impl From<ValidParameters> for NextTokenChooserParameters {
    fn from(value: ValidParameters) -> Self {
        let (grammar, grammar_type) = match value.grammar {
            None => (String::new(), GrammarType::None),

            Some(grammar) => match grammar {
                ValidGrammar::Json(grammar_string) => (grammar_string, GrammarType::Json),
                ValidGrammar::Regex(grammar_string) => (grammar_string, GrammarType::Regex),
            },
        };

        Self {
            temperature: value.temperature,
            top_k: value.top_k,
            top_p: value.top_p,
            typical_p: value.typical_p,
            do_sample: value.do_sample,
            seed: value.seed,
            repetition_penalty: value.repetition_penalty,
            frequency_penalty: value.frequency_penalty,
            watermark: value.watermark,
            grammar,
            grammar_type: grammar_type.into(),
        }
    }
}

impl From<ValidStoppingParameters> for StoppingCriteriaParameters {
    fn from(value: ValidStoppingParameters) -> Self {
        Self {
            max_new_tokens: value.max_new_tokens,
            stop_sequences: value.stop_sequences,
            ignore_eos_token: value.ignore_eos_token,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use tracing::info_span;

    fn default_entry() -> (
        Entry,
        mpsc::UnboundedReceiver<Result<InferStreamResponse, InferError>>,
    ) {
        let (response_tx, receiver_tx) = mpsc::unbounded_channel();

        let entry = Entry {
            id: 0,
            priority: DEFAULT_PRIORITY,
            state: EntryState::Queued,
            skipped: 0,
            forced: false,
            request: ValidGenerateRequest {
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
            },
            response_tx,
            span: info_span!("entry"),
            temp_span: None,
            queue_time: Instant::now(),
            batch_time: None,
            block_allocation: None,
        };
        (entry, receiver_tx)
    }

    fn test_config() -> QueueConfig {
        QueueConfig {
            max_batch_total_tokens: 16,
            max_batch_prefill_tokens: 16,
            ..QueueConfig::default()
        }
    }

    fn test_state() -> State {
        let config = test_config();
        let counters = Arc::new(SharedCounters::default());
        // Mirror `State::new` for the default test config.
        let block_allocator = (!config.requires_padding).then(|| {
            BlockAllocator::new(
                config.max_batch_total_tokens,
                config.block_size,
                config.prefix_caching,
                config.window_size,
            )
        });
        State::for_tests(config, counters, block_allocator)
    }

    #[tokio::test]
    async fn test_append() {
        let mut state = test_state();
        let (entry, _guard) = default_entry();

        assert_eq!(state.entries.len(), 0);

        state.reserve_and_append(entry);

        assert_eq!(state.entries.len(), 1);
        assert_eq!(state.entries[0].id, 0);
        assert_eq!(state.entries[0].state, EntryState::Queued);
    }

    #[tokio::test]
    async fn test_next_batch_empty() {
        let mut state = test_state();
        let now = Instant::now();

        assert!(state
            .next_batch_at(now, None, None, 1, 1)
            .await
            .unwrap()
            .is_none());
        assert!(state
            .next_batch_at(now, Some(1), None, 1, 1)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn test_next_batch_min_size() {
        let mut state = test_state();
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        state.reserve_and_append(entry1);
        state.reserve_and_append(entry2);
        let now = Instant::now();

        let (entries, batch, _) = state
            .next_batch_at(now, None, None, 2, 2)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&0));
        assert!(entries.contains_key(&1));
        assert!(entries.get(&0).unwrap().batch_time.is_some());
        assert!(entries.get(&1).unwrap().batch_time.is_some());
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 2);

        assert_eq!(state.entries.len(), 0);
        assert_eq!(state.next_batch_id, 1);

        let (entry3, _guard3) = default_entry();
        state.reserve_and_append(entry3);

        assert!(state
            .next_batch_at(now, Some(2), None, 2, 2)
            .await
            .unwrap()
            .is_none());

        assert_eq!(state.entries.len(), 1);
        assert_eq!(state.entries[0].id, 2);
    }

    #[tokio::test]
    async fn test_next_batch_max_size() {
        let mut state = test_state();
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        state.reserve_and_append(entry1);
        state.reserve_and_append(entry2);

        let (entries, batch, _) = state
            .next_batch_at(Instant::now(), None, Some(1), 2, 2)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert!(entries.get(&0).unwrap().batch_time.is_some());
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);

        assert_eq!(state.entries.len(), 1);
        assert_eq!(state.next_batch_id, 1);
    }

    #[tokio::test]
    async fn test_next_batch_token_budget() {
        let mut state = test_state();
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        state.reserve_and_append(entry1);
        state.reserve_and_append(entry2);

        let (entries, batch, _) = state
            .next_batch_at(Instant::now(), None, None, 1, 1)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);

        assert_eq!(state.entries.len(), 1);
        assert_eq!(state.next_batch_id, 1);

        let (entry3, _guard3) = default_entry();
        state.reserve_and_append(entry3);

        let (entries, batch, _) = state
            .next_batch_at(Instant::now(), None, None, 3, 3)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&1));
        assert!(entries.contains_key(&2));
        assert_eq!(batch.id, 1);
        assert_eq!(batch.size, 2);

        assert_eq!(state.entries.len(), 0);
        assert_eq!(state.next_batch_id, 2);
    }

    #[tokio::test]
    async fn test_queue_append() {
        let queue = Queue::new(test_config());
        let (entry, _guard) = default_entry();
        let id = queue.append(entry).unwrap();
        assert_eq!(id, 0);
    }

    #[tokio::test]
    async fn test_queue_next_batch_empty() {
        let queue = Queue::new(test_config());

        assert!(queue.next_batch(None, None, 1, 1).await.unwrap().is_none());
        assert!(queue
            .next_batch(Some(1), None, 1, 1)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn test_queue_next_batch_min_size() {
        let queue = Queue::new(test_config());
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        queue.append(entry1).unwrap();
        queue.append(entry2).unwrap();

        let (entries, batch, _) = queue.next_batch(None, None, 2, 2).await.unwrap().unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&0));
        assert!(entries.contains_key(&1));
        assert!(entries.get(&0).unwrap().batch_time.is_some());
        assert!(entries.get(&1).unwrap().batch_time.is_some());
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 2);

        let (entry3, _guard3) = default_entry();
        queue.append(entry3).unwrap();

        // Not enough requests pending
        assert!(queue
            .next_batch(Some(2), None, 2, 2)
            .await
            .unwrap()
            .is_none());
        // Not enough token budget
        assert!(queue
            .next_batch(Some(1), None, 0, 0)
            .await
            .unwrap()
            .is_none());
        // Ok
        let (entries2, batch2, _) = queue
            .next_batch(Some(1), None, 2, 2)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(entries2.len(), 1);
        assert!(entries2.contains_key(&2));
        assert!(entries2.get(&2).unwrap().batch_time.is_some());
        assert_eq!(batch2.id, 1);
        assert_eq!(batch2.size, 1);
    }

    #[tokio::test]
    async fn test_queue_next_batch_max_size() {
        let queue = Queue::new(test_config());
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        queue.append(entry1).unwrap();
        queue.append(entry2).unwrap();

        let (entries, batch, _) = queue
            .next_batch(None, Some(1), 2, 2)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert!(entries.get(&0).unwrap().batch_time.is_some());
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);
    }

    #[tokio::test]
    async fn test_queue_next_batch_token_budget() {
        let queue = Queue::new(test_config());
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        queue.append(entry1).unwrap();
        queue.append(entry2).unwrap();

        let (entries, batch, _) = queue.next_batch(None, None, 1, 1).await.unwrap().unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);

        let (entry3, _guard3) = default_entry();
        queue.append(entry3).unwrap();

        let (entries, batch, _) = queue.next_batch(None, None, 3, 3).await.unwrap().unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&1));
        assert!(entries.contains_key(&2));
        assert_eq!(batch.id, 1);
        assert_eq!(batch.size, 2);
    }

    #[tokio::test]
    async fn test_queue_next_batch_token_speculate() {
        // Mirror the original test: padding mode means no block allocator and
        // the total budget explicitly includes `speculate`.
        let config = QueueConfig {
            requires_padding: true,
            speculate: 2,
            ..test_config()
        };
        let queue = Queue::new(config);
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        queue.append(entry1).unwrap();
        queue.append(entry2).unwrap();

        // Budget of 1 is not enough (prefill 1 + decode 1 + speculate 2 = 4)
        assert!(queue.next_batch(None, None, 1, 1).await.unwrap().is_none());

        let (entries, batch, _) = queue.next_batch(None, None, 6, 6).await.unwrap().unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&0));
        assert!(entries.contains_key(&1));
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 2);
    }

    #[tokio::test]
    async fn test_queue_next_batch_dropped_receiver() {
        let queue = Queue::new(test_config());
        let (entry, _) = default_entry();
        queue.append(entry).unwrap();

        assert!(queue.next_batch(None, None, 1, 1).await.unwrap().is_none());
        let stats = queue.stats().await.unwrap();
        assert_eq!(stats.queued, 0);
        assert_eq!(stats.dropped_queued, 1);
        assert_eq!(queue.len(), 0);
    }

    // ----- New scheduler tests -------------------------------------------------

    fn entry_with_size(
        input_length: u32,
        max_new_tokens: u32,
        priority: u8,
        queue_time: Option<Instant>,
    ) -> (
        Entry,
        mpsc::UnboundedReceiver<Result<InferStreamResponse, InferError>>,
    ) {
        let (mut entry, receiver) = default_entry();
        entry.request.input_length = input_length;
        entry.request.stopping_parameters.max_new_tokens = max_new_tokens;
        entry.priority = priority;
        if let Some(queue_time) = queue_time {
            entry.queue_time = queue_time;
        }
        (entry, receiver)
    }

    fn config_with(
        max_queue_size: usize,
        max_batch_total_tokens: u32,
        max_batch_prefill_tokens: u32,
    ) -> QueueConfig {
        QueueConfig {
            max_queue_size,
            max_batch_total_tokens,
            max_batch_prefill_tokens,
            ..QueueConfig::default()
        }
    }

    #[tokio::test]
    async fn test_admission_full_and_id_monotonicity() {
        let config = config_with(2, 16, 16);
        let queue = Queue::new(config);

        let (entry0, _guard0) = default_entry();
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        assert_eq!(queue.append(entry0).unwrap(), 0);
        assert_eq!(queue.append(entry1).unwrap(), 1);

        let err = queue.append(entry2).unwrap_err();
        assert_eq!(err, QueueError::Full { capacity: 2 });
        let stats = queue.stats().await.unwrap();
        assert_eq!(stats.queued, 2);
        assert_eq!(stats.rejected_full, 1);
        // The rejected request does not occupy a slot.
        assert_eq!(queue.len(), 2);

        // Cancelling one entry frees a slot; the new id keeps increasing and
        // is never recycled.
        let outcome = queue.cancel(0).await.unwrap();
        assert_eq!(outcome, CancelOutcome::CancelledQueued { id: 0 });
        let (entry3, _guard3) = default_entry();
        assert_eq!(queue.append(entry3).unwrap(), 3);
        assert_eq!(queue.len(), 2);
        drop(_guard0);
        drop(_guard1);
        drop(_guard2);
        drop(_guard3);
    }

    #[tokio::test]
    async fn test_admission_request_too_large_boundary() {
        // Default entry footprint = input_length(1) + max_new_tokens(1) = 2.
        let config = config_with(16, 2, 2);
        let queue = Queue::new(config);

        let (entry_at_limit, _guard1) = default_entry();
        assert_eq!(queue.append(entry_at_limit).unwrap(), 0);

        let (mut too_large, _guard2) = default_entry();
        too_large.request.stopping_parameters.max_new_tokens = 2;
        let err = queue.append(too_large).unwrap_err();
        assert_eq!(
            err,
            QueueError::RequestTooLarge {
                tokens: 3,
                limit: 2,
            }
        );
        // Rejected requests do not recycle ids.
        let (entry_next, _guard3) = default_entry();
        assert_eq!(queue.append(entry_next).unwrap(), 2);
        let stats = queue.stats().await.unwrap();
        assert_eq!(stats.rejected_too_large, 1);
        drop(_guard1);
        drop(_guard2);
        drop(_guard3);
    }

    #[tokio::test]
    async fn test_cancel_middle_preserves_order_and_is_idempotent() {
        let queue = Queue::new(test_config());
        let (entry0, _g0) = default_entry();
        let (entry1, _g1) = default_entry();
        let (entry2, _g2) = default_entry();
        queue.append(entry0).unwrap();
        queue.append(entry1).unwrap();
        queue.append(entry2).unwrap();

        assert_eq!(
            queue.cancel(1).await.unwrap(),
            CancelOutcome::CancelledQueued { id: 1 }
        );
        // Repeated cancel is idempotent and does not double count.
        assert_eq!(
            queue.cancel(1).await.unwrap(),
            CancelOutcome::NotFound { id: 1 }
        );
        let stats = queue.stats().await.unwrap();
        assert_eq!(stats.queued, 2);
        assert_eq!(stats.cancelled_queued, 1);
        assert_eq!(queue.len(), stats.queued);

        let (entries, _batch, _) = queue.next_batch(None, None, 16, 16).await.unwrap().unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&0));
        assert!(entries.contains_key(&2));
        drop(_g0);
        drop(_g1);
        drop(_g2);
    }

    #[tokio::test]
    async fn test_cancel_unknown_and_running_return_not_found() {
        let queue = Queue::new(test_config());
        assert_eq!(
            queue.cancel(42).await.unwrap(),
            CancelOutcome::NotFound { id: 42 }
        );

        let (entry, _guard) = default_entry();
        let id = queue.append(entry).unwrap();
        let _ = queue.next_batch(None, None, 16, 16).await.unwrap().unwrap();
        // The entry is now running; cancel must report NotFound.
        assert_eq!(
            queue.cancel(id).await.unwrap(),
            CancelOutcome::NotFound { id }
        );
        drop(_guard);
    }

    #[tokio::test]
    async fn test_deadline_exemption_ignores_min_size() {
        let config = QueueConfig {
            max_queue_delay: Duration::from_millis(250),
            ..test_config()
        };
        let queue = Queue::new(config);

        let t0 = Instant::now();
        let (entry0, _g0) = entry_with_size(1, 1, 1, Some(t0));
        let (entry1, _g1) = entry_with_size(1, 1, 1, Some(t0));
        queue.append(entry0).unwrap();
        queue.append(entry1).unwrap();

        // Before the deadline, min_size=3 cannot be satisfied.
        assert!(queue
            .next_batch_at(t0 + Duration::from_millis(100), Some(3), None, 16, 16)
            .await
            .unwrap()
            .is_none());
        let stats = queue.stats().await.unwrap();
        assert_eq!(stats.queued, 2);

        // After the deadline, min_size is ignored and both entries are taken.
        let (entries, batch, _) = queue
            .next_batch_at(t0 + Duration::from_millis(250), Some(3), None, 16, 16)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(batch.size, 2);
        drop(_g0);
        drop(_g1);
    }

    #[tokio::test]
    async fn test_zero_queue_delay_always_exempts() {
        let config = QueueConfig {
            max_queue_delay: Duration::ZERO,
            ..test_config()
        };
        let queue = Queue::new(config);

        let (entry, _guard) = default_entry();
        queue.append(entry).unwrap();
        // Even an unsatisfiable min_size is bypassed when delay is zero.
        let (entries, _, _) = queue
            .next_batch_at(Instant::now(), Some(99), None, 16, 16)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(entries.len(), 1);
        drop(_guard);
    }

    #[tokio::test]
    async fn test_aging_makes_old_low_priority_tie_with_fresh_high_priority() {
        let config = QueueConfig {
            aging_interval: Duration::from_millis(50),
            ..test_config()
        };
        let queue = Queue::new(config);

        let now = Instant::now();
        // Priority 0 request that has waited 200ms.
        let (old_entry, _g_old) = entry_with_size(1, 1, 0, Some(now - Duration::from_millis(200)));
        // Priority 3 request that just arrived.
        let (fresh_entry, _g_fresh) = entry_with_size(1, 1, 3, Some(now));
        let old_id = queue.append(old_entry).unwrap();
        let fresh_id = queue.append(fresh_entry).unwrap();
        assert_eq!(old_id, 0);
        assert_eq!(fresh_id, 1);

        // Both have effective priority 3, so the older queue_time wins.
        let (entries, batch, _) = queue
            .next_batch_at(now, None, Some(1), 16, 16)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(batch.requests[0].id, 0);
        assert!(entries.contains_key(&old_id));

        // The fresh entry is still queued.
        let stats = queue.stats().await.unwrap();
        assert_eq!(stats.queued, 1);
        drop(_g_old);
        drop(_g_fresh);
    }

    #[tokio::test]
    async fn test_adaptive_scan_skips_head_and_forces_after_max_skips() {
        // No block allocator: budgets are purely token based.
        let config = QueueConfig {
            requires_padding: true,
            block_size: 1,
            max_queue_size: 8,
            max_batch_total_tokens: 1024,
            max_batch_prefill_tokens: 1024,
            max_skips: 3,
            ..QueueConfig::default()
        };
        let queue = Queue::new(config);

        // Head needs 8 prefill tokens, second entry needs 2.
        let (big, _g0) = entry_with_size(8, 1, 1, None);
        let (small, _g1) = entry_with_size(2, 1, 1, None);
        let big_id = queue.append(big).unwrap();
        let small_id = queue.append(small).unwrap();

        // Prefill budget of 4 only fits the small entry.
        let (entries, batch, _) = queue
            .next_batch(None, None, 4, 1024)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(batch.size, 1);
        assert!(entries.contains_key(&small_id));

        // The big entry is still at the head and accumulated one skip.
        let stats = queue.stats().await.unwrap();
        assert_eq!(stats.queued, 1);

        // Two more failing scans bring the skip count to max_skips: the entry
        // becomes forced. It still cannot fit a budget of 4, so an empty batch
        // is returned and the entry remains queued.
        assert!(queue
            .next_batch(None, None, 4, 1024)
            .await
            .unwrap()
            .is_none());
        assert!(queue
            .next_batch(None, None, 4, 1024)
            .await
            .unwrap()
            .is_none());

        // With enough budget, the forced entry is scheduled on its own.
        let (entries, batch, _) = queue
            .next_batch(None, None, 8, 1024)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(batch.size, 1);
        assert!(entries.contains_key(&big_id));
        assert_eq!(batch.requests[0].id, big_id);
        drop(_g0);
        drop(_g1);
    }

    #[tokio::test]
    async fn test_shutdown_cancels_queued_and_fails_pending_calls() {
        let queue = Queue::new(test_config());
        let (entry0, _g0) = default_entry();
        let (entry1, _g1) = default_entry();
        queue.append(entry0).unwrap();
        queue.append(entry1).unwrap();

        // Shutdown drains and cancels every queued entry.
        let report = queue.shutdown().await.unwrap();
        assert_eq!(report.cancelled, 2);
        assert_eq!(report.stats.queued, 0);
        assert_eq!(queue.len(), 0);

        // Every subsequent command reports Shutdown, including a parked
        // `next_batch`/`stats` (the task is gone, so the oneshot never
        // completes with a value and the handle maps it to `Shutdown`).
        assert_eq!(
            queue.next_batch(None, None, 1, 1).await.unwrap_err(),
            QueueError::Shutdown
        );
        assert_eq!(queue.stats().await.unwrap_err(), QueueError::Shutdown);
        assert_eq!(queue.cancel(0).await.unwrap_err(), QueueError::Shutdown);
        let (entry, _g2) = default_entry();
        assert_eq!(queue.append(entry).unwrap_err(), QueueError::Shutdown);
        drop(_g0);
        drop(_g1);
        drop(_g2);
    }

    /// Deterministic linear-congruential generator for the budget property
    /// test.
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self, low: u32, high: u32) -> u32 {
            // Numerical Recipes constants.
            self.state = self
                .state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let span = (high - low).max(1);
            low + (self.state as u32 % span)
        }
    }

    #[tokio::test]
    async fn test_budget_invariants_on_lcg_workload() {
        let config = QueueConfig {
            requires_padding: true,
            block_size: 1,
            max_queue_size: 4096,
            max_batch_total_tokens: 512,
            max_batch_prefill_tokens: 256,
            support_chunking: false,
            ..QueueConfig::default()
        };
        let queue = Queue::new(config);

        let mut rng = Lcg::new(0x1234_5678);
        // Ids currently in the queue or already scheduled/terminal. Keeping the
        // receiver guards alive ensures the scheduler never observes a client
        // disconnect during this test.
        let mut guards: Vec<mpsc::UnboundedReceiver<Result<InferStreamResponse, InferError>>> =
            Vec::new();
        let mut queued_ids = std::collections::HashSet::new();
        let mut terminal_ids = std::collections::HashSet::new();
        let mut rejected = 0u64;

        for _ in 0..1000 {
            let input_length = rng.next_u32(1, 40);
            let max_new_tokens = rng.next_u32(1, 40);
            let (entry, guard) = entry_with_size(input_length, max_new_tokens, 1, None);
            let id = match queue.append(entry) {
                Ok(id) => id,
                // Rejected requests never enter the queue and carry no
                // scheduler id visible to the caller.
                Err(QueueError::RequestTooLarge { .. }) | Err(QueueError::Full { .. }) => {
                    rejected += 1;
                    continue;
                }
                Err(err) => panic!("unexpected queue error: {err:?}"),
            };
            guards.push(guard);
            assert!(queued_ids.insert(id), "id {id} queued twice");

            let prefill_budget = rng.next_u32(8, 128);
            let token_budget = rng.next_u32(16, 256);
            if let Some((entries, batch, _)) = queue
                .next_batch(None, None, prefill_budget, token_budget)
                .await
                .unwrap()
            {
                // No duplicate id inside the batch.
                let mut seen = std::collections::HashSet::new();
                for request in &batch.requests {
                    assert!(seen.insert(request.id), "duplicate id in batch");
                    assert!(queued_ids.remove(&request.id));
                    assert!(entries.contains_key(&request.id));
                    // Each id reaches the terminal bookkeeping set once.
                    assert!(terminal_ids.insert(request.id));
                }
                // Padding path: prefill never exceeds the block-aligned
                // budget and total tokens never exceed the token budget.
                let aligned_prefill = prefill_budget;
                let mut max_input = 0u32;
                let mut decode = 0u32;
                for request in &batch.requests {
                    let entry = entries.get(&request.id).unwrap();
                    max_input = max_input.max(entry.request.input_length);
                    decode += entry.request.stopping_parameters.max_new_tokens;
                }
                let prefill = batch.size * max_input;
                assert!(
                    prefill <= aligned_prefill,
                    "prefill {prefill} exceeds budget {aligned_prefill}"
                );
                assert!(
                    batch.max_tokens <= token_budget,
                    "max_tokens {} exceeds token budget {token_budget}",
                    batch.max_tokens
                );
                assert_eq!(batch.max_tokens, prefill + decode);
            }
        }

        // Every id lived in exactly one of the queue/terminal sets.
        for id in &queued_ids {
            assert!(!terminal_ids.contains(id));
        }
        let remaining = queue.stats().await.unwrap().queued;
        assert_eq!(remaining, queued_ids.len());
        assert!(rejected + remaining as u64 + terminal_ids.len() as u64 <= 1000);
        let _ = queue.shutdown().await;
    }

    /// Allocator that blocks inside `allocate` until a test releases the gate.
    /// It lets a test disconnect the client while the scheduler is parked
    /// waiting for blocks, exercising the post-allocation cancellation
    /// checkpoint deterministically.
    struct BlockingAllocator {
        gate: Arc<BlockGate>,
    }

    impl BlockingAllocator {
        fn new(gate: Arc<BlockGate>, _total_blocks: u32) -> Self {
            Self { gate }
        }
    }

    struct BlockGate {
        released: std::sync::Mutex<bool>,
        reached: std::sync::Mutex<bool>,
        cvar: std::sync::Condvar,
    }

    impl crate::block_allocator::Allocator for BlockingAllocator {
        fn allocate(
            &mut self,
            tokens: u32,
            _prefill_tokens: Option<Arc<Vec<u32>>>,
        ) -> Option<BlockAllocation> {
            {
                let mut reached = self.gate.reached.lock().unwrap();
                *reached = true;
            }
            let mut released = self.gate.released.lock().unwrap();
            while !*released {
                released = self.gate.cvar.wait(released).unwrap();
            }
            drop(released);

            let required = tokens.max(1);
            // Blocks 1..required (block 0 stays reserved for health checks).
            let blocks: Vec<u32> = (1..=required).collect();
            let slots = blocks.clone();
            Some(BlockAllocation {
                allocation_id: 1,
                blocks,
                slots,
                prefix_len: 0,
                block_allocator: None,
            })
        }

        fn free(&mut self, _blocks: Vec<u32>, _allocation_id: u64) {}

        #[cfg(test)]
        fn free_block_count(&self) -> usize {
            // The blocking test allocator never permanently holds blocks: a
            // parked allocation is dropped when the entry is reaped.
            15
        }
    }

    fn checkpoint_state(block_allocator: BlockAllocator) -> (State, Arc<SharedCounters>) {
        let config = QueueConfig {
            block_size: 1,
            max_batch_total_tokens: 64,
            max_batch_prefill_tokens: 64,
            ..QueueConfig::default()
        };
        let counters = Arc::new(SharedCounters::default());
        let state = State::for_tests(config, counters.clone(), Some(block_allocator));
        (state, counters)
    }

    #[tokio::test]
    async fn test_checkpoint_disconnect_before_scan_uses_no_blocks() {
        let allocator = crate::block_allocator::tests::ControllableAllocator::new(8);
        let block_allocator = BlockAllocator::with_allocator(Box::new(allocator));
        let free_before = block_allocator.free_block_count().await;

        let (mut state, counters) = checkpoint_state(block_allocator);
        let (entry, receiver) = default_entry();
        state.reserve_and_append(entry);

        // Disconnect before any scan starts.
        drop(receiver);
        let result = state
            .next_batch_at(Instant::now(), None, None, 16, 16)
            .await
            .unwrap();
        assert!(result.is_none());
        assert_eq!(counters.dropped_queued.load(Ordering::Relaxed), 1);
        assert_eq!(state.entries.len(), 0);
        // No block was ever allocated.
        assert_eq!(
            state
                .block_allocator
                .as_ref()
                .unwrap()
                .free_block_count()
                .await,
            free_before
        );
    }

    // Multi-thread runtime: the injected allocator parks on a std condvar
    // inside a background task, which would deadlock a current-thread runtime.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_checkpoint_disconnect_while_allocation_parked() {
        let gate = Arc::new(BlockGate {
            released: std::sync::Mutex::new(false),
            reached: std::sync::Mutex::new(false),
            cvar: std::sync::Condvar::new(),
        });
        let allocator = BlockingAllocator::new(gate.clone(), 16);
        let block_allocator = BlockAllocator::with_allocator(Box::new(allocator));
        let free_before = block_allocator.free_block_count().await;

        let (mut state, counters) = checkpoint_state(block_allocator);
        let (entry, receiver) = default_entry();
        state.reserve_and_append(entry);

        let scan = tokio::spawn(async move {
            let result = state
                .next_batch_at(Instant::now(), None, None, 16, 16)
                .await;
            (state, result)
        });

        // Wait until the allocator task is parked inside `allocate`.
        for _ in 0..1000 {
            if *gate.reached.lock().unwrap() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        assert!(*gate.reached.lock().unwrap(), "allocator never parked");

        // The client disconnects while blocks are being allocated.
        drop(receiver);

        // Release the allocator: allocation succeeds, then the post-allocation
        // checkpoint must observe the closed channel and reap the entry.
        {
            let mut released = gate.released.lock().unwrap();
            *released = true;
            gate.cvar.notify_all();
        }

        let (state, result) = scan.await.unwrap();
        assert!(result.unwrap().is_none());
        assert_eq!(counters.dropped_queued.load(Ordering::Relaxed), 1);
        assert_eq!(state.entries.len(), 0);
        // The tentative allocation is dropped, so every block is back in the
        // pool.
        assert_eq!(
            state
                .block_allocator
                .as_ref()
                .unwrap()
                .free_block_count()
                .await,
            free_before
        );
    }

    #[tokio::test]
    async fn test_checkpoint_allocation_failure_keeps_entry_and_order() {
        let allocator = crate::block_allocator::tests::ControllableAllocator::new(8);
        // Adaptive scanning keeps trying the entries behind a failure, so fail
        // every allocation for this scan to get an empty batch back.
        allocator.fail_next_allocations(u32::MAX);
        let block_allocator = BlockAllocator::with_allocator(Box::new(allocator));
        let free_before = block_allocator.free_block_count().await;

        let (mut state, counters) = checkpoint_state(block_allocator);
        let (entry1, _g1) = default_entry();
        let (entry2, _g2) = default_entry();
        state.reserve_and_append(entry1);
        state.reserve_and_append(entry2);

        let result = state
            .next_batch_at(Instant::now(), None, None, 16, 16)
            .await
            .unwrap();
        // Block allocation failure must produce an empty batch and retain
        // every entry in its original (id) order.
        assert!(result.is_none());
        assert_eq!(state.entries.len(), 2);
        assert_eq!(state.entries[0].id, 0);
        assert_eq!(state.entries[1].id, 1);
        assert_eq!(counters.dropped_queued.load(Ordering::Relaxed), 0);
        // Both entries accumulated a skip.
        assert_eq!(state.entries[0].skipped, 1);
        assert_eq!(state.entries[1].skipped, 1);
        assert_eq!(
            state
                .block_allocator
                .as_ref()
                .unwrap()
                .free_block_count()
                .await,
            free_before
        );
    }

    #[tokio::test]
    async fn test_chunking_truncates_last_entry_to_remaining_budget() {
        // Chunking needs a block allocator (no padding).
        let config = QueueConfig {
            requires_padding: false,
            block_size: 1,
            support_chunking: true,
            max_batch_total_tokens: 1024,
            max_batch_prefill_tokens: 16,
            ..QueueConfig::default()
        };
        let queue = Queue::new(config);

        // A large request whose postfix exceeds the prefill budget.
        let (large, _g0) = entry_with_size(10, 10, 1, None);
        let id = queue.append(large).unwrap();

        let (entries, batch, _) = queue
            .next_batch(None, None, 4, 1024)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(batch.size, 1);
        let request = &batch.requests[0];
        assert_eq!(request.id, id);
        // The chunk must be non-zero, bounded by the postfix and exactly fill
        // the remaining prefill budget.
        let chunk_len = request.chunk_len.expect("chunking must set chunk_len");
        assert!(chunk_len >= 1);
        assert!(chunk_len <= 10);
        assert_eq!(chunk_len, 4);
        assert!(entries.contains_key(&id));
        drop(_g0);
    }

    #[tokio::test]
    async fn test_effective_priority_aging_and_disabled_aging() {
        let base = Instant::now();
        // Aging disabled: effective priority never changes.
        assert_eq!(effective_priority(0, base, base, Duration::ZERO), 0);
        assert_eq!(
            effective_priority(
                0,
                base,
                base + Duration::from_millis(10_000),
                Duration::ZERO
            ),
            0
        );

        // 200ms wait with a 50ms interval adds 4, so priority 0 reaches the
        // MAX_PRIORITY cap; the spec's tie scenario uses an uncapped base of 3.
        let now = base + Duration::from_millis(200);
        assert_eq!(
            effective_priority(0, base, now, Duration::from_millis(50)),
            MAX_PRIORITY as u32
        );
        // 100ms adds 2: priority 1 lands on 3.
        let now100 = base + Duration::from_millis(100);
        assert_eq!(
            effective_priority(1, base, now100, Duration::from_millis(50)),
            MAX_PRIORITY as u32
        );
        // Fresh priority 3 stays at the cap.
        assert_eq!(
            effective_priority(3, base, base, Duration::from_millis(50)),
            MAX_PRIORITY as u32
        );
    }
}
