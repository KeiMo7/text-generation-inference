use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

use crate::radix::RadixAllocator;
use text_generation_router::usage_stats::Env;
#[derive(Debug, Clone)]
pub struct BlockAllocation {
    pub allocation_id: u64,
    pub blocks: Vec<u32>,
    pub slots: Vec<u32>,

    /// Prefix that was cached and for which the KV does not have to
    /// be recomputed.
    pub prefix_len: u32,

    pub(crate) block_allocator: Option<BlockAllocator>,
}

impl Drop for BlockAllocation {
    fn drop(&mut self) {
        if let Some(block_allocator) = self.block_allocator.as_mut() {
            block_allocator.free(self.blocks.clone(), self.allocation_id)
        }
    }
}

#[derive(Debug, Clone)]
pub struct BlockAllocator {
    /// Channel to communicate with the background task
    block_allocator: mpsc::UnboundedSender<BlockAllocatorCommand>,
}

impl BlockAllocator {
    pub(crate) fn new(
        max_batch_total_tokens: u32,
        block_size: u32,
        prefix_caching: bool,
        window_size: Option<u32>,
    ) -> Self {
        // Create channel
        let (sender, receiver) = mpsc::unbounded_channel();

        // Launch background queue task
        tokio::spawn(block_allocator_task(
            max_batch_total_tokens / block_size,
            block_size,
            prefix_caching,
            window_size,
            receiver,
        ));

        Self {
            block_allocator: sender,
        }
    }

    /// Test-only constructor: run the allocator task with an injected
    /// `Allocator` implementation (e.g. to simulate allocation failures).
    #[cfg(test)]
    pub(crate) fn with_allocator(allocator: Box<dyn Allocator + Send>) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        tokio::spawn(block_allocator_task_with(allocator, receiver));
        Self {
            block_allocator: sender,
        }
    }

    pub(crate) async fn allocate(
        &self,
        tokens: u32,
        prefill_tokens: Option<Arc<Vec<u32>>>,
    ) -> Option<BlockAllocation> {
        let (response_sender, response_receiver) = oneshot::channel();
        self.block_allocator
            .send(BlockAllocatorCommand::Allocate {
                tokens,
                prefill_tokens,
                response_sender,
            })
            .ok()?;

        response_receiver
            .await
            .ok()
            .flatten()
            .map(|mut allocation| {
                allocation.block_allocator = Some(self.clone());
                allocation
            })
    }

    pub(crate) fn free(&self, blocks: Vec<u32>, allocation_id: u64) {
        let _ = self.block_allocator.send(BlockAllocatorCommand::Free {
            allocation_id,
            blocks,
        });
    }

    /// Test-only introspection returning the number of free blocks.
    #[cfg(test)]
    pub(crate) async fn free_block_count(&self) -> usize {
        let (response_sender, response_receiver) = oneshot::channel();
        if self
            .block_allocator
            .send(BlockAllocatorCommand::FreeBlockCount { response_sender })
            .is_err()
        {
            return 0;
        }
        response_receiver.await.unwrap_or(0)
    }
}

async fn block_allocator_task(
    blocks: u32,
    block_size: u32,
    prefix_caching: bool,
    window_size: Option<u32>,
    receiver: mpsc::UnboundedReceiver<BlockAllocatorCommand>,
) {
    let allocator: Box<dyn Allocator + Send> = if prefix_caching {
        Box::new(RadixAllocator::new(block_size, blocks, window_size))
    } else {
        Box::new(SimpleAllocator::new(blocks, block_size, window_size))
    };
    block_allocator_task_with(allocator, receiver).await;
}

async fn block_allocator_task_with(
    mut allocator: Box<dyn Allocator + Send>,
    mut receiver: mpsc::UnboundedReceiver<BlockAllocatorCommand>,
) {
    while let Some(cmd) = receiver.recv().await {
        match cmd {
            BlockAllocatorCommand::Free {
                blocks,
                allocation_id,
            } => allocator.free(blocks, allocation_id),
            BlockAllocatorCommand::Allocate {
                tokens,
                prefill_tokens,
                response_sender,
            } => {
                // The receiver may have been dropped while the allocation was
                // in flight; dropping the reply is fine.
                let _ = response_sender.send(allocator.allocate(tokens, prefill_tokens));
            }
            #[cfg(test)]
            BlockAllocatorCommand::FreeBlockCount { response_sender } => {
                let _ = response_sender.send(allocator.free_block_count());
            }
        }
    }
}

#[derive(Debug)]
enum BlockAllocatorCommand {
    Free {
        blocks: Vec<u32>,
        allocation_id: u64,
    },
    Allocate {
        tokens: u32,
        prefill_tokens: Option<Arc<Vec<u32>>>,
        response_sender: oneshot::Sender<Option<BlockAllocation>>,
    },
    /// Test-only introspection: number of blocks currently free.
    #[cfg(test)]
    FreeBlockCount {
        response_sender: oneshot::Sender<usize>,
    },
}

pub trait Allocator {
    fn allocate(
        &mut self,
        tokens: u32,
        prefill_tokens: Option<Arc<Vec<u32>>>,
    ) -> Option<BlockAllocation>;

    fn free(&mut self, blocks: Vec<u32>, allocation_id: u64);

    /// Number of blocks currently available for allocation. Test-only
    /// introspection used by cancellation checkpoint tests.
    #[cfg(test)]
    fn free_block_count(&self) -> usize;
}
pub struct SimpleAllocator {
    free_blocks: Vec<u32>,
    block_size: u32,
    window_size: Option<u32>,
    is_hpu_device: bool,
}

impl SimpleAllocator {
    fn new(blocks: u32, block_size: u32, window_size: Option<u32>) -> Self {
        SimpleAllocator {
            block_size,
            // Block 0 is reserved for health checks
            free_blocks: (1..blocks).collect(),
            window_size,
            is_hpu_device: Env::new().is_hpu_device(),
        }
    }
}

impl Allocator for SimpleAllocator {
    fn allocate(
        &mut self,
        tokens: u32,
        _prefill_tokens: Option<Arc<Vec<u32>>>,
    ) -> Option<BlockAllocation> {
        let mut tokens = tokens;
        if self.is_hpu_device {
            // need 1 slot for ping-pong optimization
            tokens += 1;
        }
        // Apply window size
        let (required_blocks, repeats) = {
            let (tokens, repeats) = match self.window_size {
                None => (tokens, 1),
                Some(window_size) => {
                    let repeats = tokens.div_ceil(window_size);
                    let tokens = core::cmp::min(tokens, window_size);
                    (tokens, repeats as usize)
                }
            };
            // Pad to a multiple of block size
            let required_blocks = tokens.div_ceil(self.block_size);
            (required_blocks, repeats)
        };
        let tokens = tokens as usize;
        if required_blocks > self.free_blocks.len() as u32 {
            None
        } else {
            if self.is_hpu_device {
                self.free_blocks.sort_by(|a, b| b.cmp(a));
            }
            let mut blocks = self
                .free_blocks
                .split_off(self.free_blocks.len() - required_blocks as usize);
            if self.is_hpu_device {
                blocks.sort();
            }
            let mut slots =
                Vec::with_capacity((required_blocks * self.block_size * repeats as u32) as usize);

            'slots: for block_id in blocks.repeat(repeats).iter() {
                for s in (block_id * self.block_size)..((block_id + 1) * self.block_size) {
                    slots.push(s);
                    if slots.len() == tokens {
                        break 'slots;
                    }
                }
            }
            Some(BlockAllocation {
                allocation_id: 0,
                blocks,
                slots,
                prefix_len: 0,
                block_allocator: None,
            })
        }
    }

    fn free(&mut self, blocks: Vec<u32>, _allocation_id: u64) {
        self.free_blocks.extend(blocks)
    }

    #[cfg(test)]
    fn free_block_count(&self) -> usize {
        self.free_blocks.len()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Controllable allocator used to exercise the scheduler cancellation
    /// checkpoints. It hands out blocks from a fixed pool and counts how many
    /// blocks are currently allocated so tests can assert that `Drop` returns
    /// every block exactly once.
    #[derive(Debug)]
    pub(crate) struct ControllableAllocator {
        free_blocks: Vec<u32>,
        allocations: std::collections::HashSet<u64>,
        next_allocation_id: AtomicU64,
        /// When positive, the next N allocations return `None` to simulate
        /// block exhaustion.
        fail_next: std::sync::Mutex<u32>,
    }

    impl ControllableAllocator {
        pub(crate) fn new(total_blocks: u32) -> Self {
            // Block 0 stays reserved for health checks.
            Self {
                free_blocks: (1..total_blocks).collect(),
                allocations: std::collections::HashSet::new(),
                next_allocation_id: AtomicU64::new(1),
                fail_next: std::sync::Mutex::new(0),
            }
        }

        pub(crate) fn fail_next_allocations(&self, count: u32) {
            *self.fail_next.lock().unwrap() = count;
        }
    }

    impl Allocator for ControllableAllocator {
        fn allocate(
            &mut self,
            tokens: u32,
            _prefill_tokens: Option<Arc<Vec<u32>>>,
        ) -> Option<BlockAllocation> {
            let mut fail_next = self.fail_next.lock().unwrap();
            if *fail_next > 0 {
                *fail_next -= 1;
                return None;
            }
            drop(fail_next);

            let required_blocks = tokens.div_ceil(1).max(1);
            if required_blocks > self.free_blocks.len() as u32 {
                return None;
            }
            let blocks: Vec<u32> = self
                .free_blocks
                .split_off(self.free_blocks.len() - required_blocks as usize);
            let slots = blocks
                .iter()
                .flat_map(|block| *block..(block + 1))
                .take(tokens as usize)
                .collect();
            let allocation_id = self.next_allocation_id.fetch_add(1, Ordering::Relaxed);
            self.allocations.insert(allocation_id);
            Some(BlockAllocation {
                allocation_id,
                blocks,
                slots,
                prefix_len: 0,
                block_allocator: None,
            })
        }

        fn free(&mut self, blocks: Vec<u32>, allocation_id: u64) {
            assert!(
                self.allocations.remove(&allocation_id),
                "allocation {allocation_id} freed twice"
            );
            self.free_blocks.extend(blocks);
        }

        fn free_block_count(&self) -> usize {
            self.free_blocks.len()
        }
    }

    #[tokio::test]
    async fn allocation_is_reclaimed_on_drop_once() {
        let allocator = ControllableAllocator::new(4);
        let initial_free = allocator.free_block_count();
        assert_eq!(initial_free, 3);
        let block_allocator = BlockAllocator::with_allocator(Box::new(allocator));

        let allocation = block_allocator.allocate(2, None).await.unwrap();
        assert_eq!(block_allocator.free_block_count().await, 1);

        // Dropping the allocation frees its blocks through `Drop` exactly
        // once; production code never calls `free` manually for the same
        // allocation, so the pool must return to its initial size.
        drop(allocation);
        // The free command is processed asynchronously by the background task.
        let mut observed = 0;
        for _ in 0..100 {
            observed = block_allocator.free_block_count().await;
            if observed == initial_free {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        assert_eq!(observed, initial_free);
    }

    #[tokio::test]
    async fn failed_allocation_returns_none_and_keeps_pool_intact() {
        let allocator = ControllableAllocator::new(4);
        let initial_free = allocator.free_block_count();
        allocator.fail_next_allocations(1);
        let block_allocator = BlockAllocator::with_allocator(Box::new(allocator));

        assert!(block_allocator.allocate(1, None).await.is_none());
        assert_eq!(block_allocator.free_block_count().await, initial_free);

        // The pool is still usable after the simulated exhaustion.
        assert!(block_allocator.allocate(1, None).await.is_some());
    }
}
