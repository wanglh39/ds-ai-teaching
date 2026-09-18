#include "paged_cache.h"

#include <stdlib.h>
#include <string.h>

static int pool_alloc_block(BlockPool *pool) {
    if (pool->free_list_head == KV_INVALID) {
        return KV_INVALID;
    }
    int id = pool->free_list_head;
    pool->free_list_head = pool->blocks[id].next_free;
    pool->blocks[id].next_free = KV_INVALID;
    pool->blocks[id].in_use = true;
    pool->blocks[id].ref_count = 1;
    pool->blocks[id].used_slots = 0;
    pool->n_free--;
    pool->n_used++;
    return id;
}

static void pool_release_block(BlockPool *pool, int id) {
    pool->blocks[id].in_use = false;
    pool->blocks[id].ref_count = 0;
    pool->blocks[id].used_slots = 0;
    pool->blocks[id].next_free = pool->free_list_head;
    pool->free_list_head = id;
    pool->n_free++;
    pool->n_used--;
}

static int pt_ensure_capacity(PageTable *pt, int needed) {
    if (needed <= pt->capacity) {
        return 0;
    }
    int new_cap = pt->capacity == 0 ? 4 : pt->capacity;
    while (new_cap < needed) {
        new_cap *= 2;
    }
    int *new_pages = (int *)realloc(pt->pages, (size_t)new_cap * sizeof(int));
    if (!new_pages) {
        return -1;
    }
    for (int i = pt->capacity; i < new_cap; i++) {
        new_pages[i] = KV_INVALID;
    }
    pt->pages = new_pages;
    pt->capacity = new_cap;
    return 0;
}

static int pt_append(PageTable *pt, int block_id) {
    if (pt_ensure_capacity(pt, pt->n_pages + 1) != 0) {
        return -1;
    }
    pt->pages[pt->n_pages] = block_id;
    pt->n_pages++;
    return 0;
}

static void pt_clear(PageTable *pt) {
    pt->n_pages = 0;
}

int kv_init(KVCacheManager *mgr, int n_blocks, int block_size, int max_seqs) {
    if (n_blocks <= 0 || block_size <= 0 || max_seqs <= 0) {
        return -1;
    }
    mgr->block_size = block_size;
    mgr->pool.n_blocks = n_blocks;
    mgr->pool.block_size = block_size;
    mgr->pool.blocks = (Block *)malloc((size_t)n_blocks * sizeof(Block));
    if (!mgr->pool.blocks) {
        return -1;
    }
    for (int i = 0; i < n_blocks; i++) {
        mgr->pool.blocks[i].block_id = i;
        mgr->pool.blocks[i].ref_count = 0;
        mgr->pool.blocks[i].used_slots = 0;
        mgr->pool.blocks[i].next_free = (i + 1 < n_blocks) ? (i + 1) : KV_INVALID;
        mgr->pool.blocks[i].in_use = false;
    }
    mgr->pool.free_list_head = 0;
    mgr->pool.n_free = n_blocks;
    mgr->pool.n_used = 0;

    mgr->seq_capacity = max_seqs;
    mgr->n_seqs = max_seqs;
    mgr->seqs = (Sequence *)malloc((size_t)max_seqs * sizeof(Sequence));
    if (!mgr->seqs) {
        free(mgr->pool.blocks);
        return -1;
    }
    for (int i = 0; i < max_seqs; i++) {
        mgr->seqs[i].seq_id = KV_INVALID;
        mgr->seqs[i].length = 0;
        mgr->seqs[i].page_table.pages = NULL;
        mgr->seqs[i].page_table.n_pages = 0;
        mgr->seqs[i].page_table.capacity = 0;
        mgr->seqs[i].active = false;
    }
    return 0;
}

void kv_free(KVCacheManager *mgr) {
    for (int i = 0; i < mgr->seq_capacity; i++) {
        free(mgr->seqs[i].page_table.pages);
        mgr->seqs[i].page_table.pages = NULL;
    }
    free(mgr->seqs);
    free(mgr->pool.blocks);
    mgr->seqs = NULL;
    mgr->pool.blocks = NULL;
    mgr->n_seqs = 0;
    mgr->seq_capacity = 0;
    mgr->pool.n_blocks = 0;
}

int kv_find_seq(const KVCacheManager *mgr, int seq_id) {
    for (int i = 0; i < mgr->seq_capacity; i++) {
        if (mgr->seqs[i].active && mgr->seqs[i].seq_id == seq_id) {
            return i;
        }
    }
    return KV_INVALID;
}

static int kv_find_free_seq_slot(const KVCacheManager *mgr) {
    for (int i = 0; i < mgr->seq_capacity; i++) {
        if (!mgr->seqs[i].active) {
            return i;
        }
    }
    return KV_INVALID;
}

static int kv_alloc_blocks_for_seq(KVCacheManager *mgr, Sequence *seq, int n_tokens) {
    if (n_tokens <= 0) {
        return 0;
    }
    int block_size = mgr->block_size;
    int pages_needed = (n_tokens + block_size - 1) / block_size;
    for (int p = 0; p < pages_needed; p++) {
        int block_id = pool_alloc_block(&mgr->pool);
        if (block_id == KV_INVALID) {
            return -1;
        }
        if (pt_append(&seq->page_table, block_id) != 0) {
            pool_release_block(&mgr->pool, block_id);
            return -1;
        }
    }
    int full_blocks = n_tokens / block_size;
    int rem = n_tokens % block_size;
    for (int p = 0; p < full_blocks; p++) {
        int block_id = seq->page_table.pages[p];
        mgr->pool.blocks[block_id].used_slots = block_size;
    }
    if (rem > 0 && full_blocks < pages_needed) {
        int block_id = seq->page_table.pages[full_blocks];
        mgr->pool.blocks[block_id].used_slots = rem;
    }
    seq->length = n_tokens;
    return 0;
}

int kv_allocate_seq(KVCacheManager *mgr, int seq_id, int n_tokens) {
    if (kv_find_seq(mgr, seq_id) != KV_INVALID) {
        return -1;
    }
    int slot = kv_find_free_seq_slot(mgr);
    if (slot == KV_INVALID) {
        return -2;
    }
    Sequence *seq = &mgr->seqs[slot];
    pt_clear(&seq->page_table);
    seq->seq_id = seq_id;
    seq->length = 0;
    seq->active = true;
    if (kv_alloc_blocks_for_seq(mgr, seq, n_tokens) != 0) {
        seq->active = false;
        seq->seq_id = KV_INVALID;
        seq->length = 0;
        pt_clear(&seq->page_table);
        return -3;
    }
    return 0;
}

int kv_append_token(KVCacheManager *mgr, int seq_id, int n_tokens) {
    int slot = kv_find_seq(mgr, seq_id);
    if (slot == KV_INVALID) {
        return -1;
    }
    if (n_tokens <= 0) {
        return 0;
    }
    Sequence *seq = &mgr->seqs[slot];
    int block_size = mgr->block_size;
    int old_length = seq->length;
    int new_length = old_length + n_tokens;

    int old_full = old_length / block_size;
    int old_rem = old_length % block_size;
    if (old_rem > 0 && old_full < seq->page_table.n_pages) {
        int block_id = seq->page_table.pages[old_full];
        int avail = block_size - old_rem;
        int fill = n_tokens < avail ? n_tokens : avail;
        mgr->pool.blocks[block_id].used_slots = old_rem + fill;
        n_tokens -= fill;
        old_length += fill;
    }

    while (n_tokens > 0) {
        int block_id = pool_alloc_block(&mgr->pool);
        if (block_id == KV_INVALID) {
            seq->length = old_length;
            return -2;
        }
        if (pt_append(&seq->page_table, block_id) != 0) {
            pool_release_block(&mgr->pool, block_id);
            seq->length = old_length;
            return -2;
        }
        int fill = n_tokens < block_size ? n_tokens : block_size;
        mgr->pool.blocks[block_id].used_slots = fill;
        n_tokens -= fill;
        old_length += fill;
    }
    seq->length = new_length;
    return 0;
}

int kv_free_seq(KVCacheManager *mgr, int seq_id) {
    int slot = kv_find_seq(mgr, seq_id);
    if (slot == KV_INVALID) {
        return -1;
    }
    Sequence *seq = &mgr->seqs[slot];
    for (int p = 0; p < seq->page_table.n_pages; p++) {
        int block_id = seq->page_table.pages[p];
        if (block_id == KV_INVALID) {
            continue;
        }
        mgr->pool.blocks[block_id].ref_count--;
        if (mgr->pool.blocks[block_id].ref_count <= 0) {
            pool_release_block(&mgr->pool, block_id);
        }
    }
    pt_clear(&seq->page_table);
    seq->active = false;
    seq->seq_id = KV_INVALID;
    seq->length = 0;
    return 0;
}

int kv_share_prefix(KVCacheManager *mgr, int src_seq_id, int dst_seq_id, int n_tokens) {
    int src_slot = kv_find_seq(mgr, src_seq_id);
    if (src_slot == KV_INVALID) {
        return -1;
    }
    if (kv_find_seq(mgr, dst_seq_id) != KV_INVALID) {
        return -2;
    }
    Sequence *src = &mgr->seqs[src_slot];
    if (n_tokens > src->length) {
        return -3;
    }
    int dst_slot = kv_find_free_seq_slot(mgr);
    if (dst_slot == KV_INVALID) {
        return -4;
    }
    Sequence *dst = &mgr->seqs[dst_slot];
    pt_clear(&dst->page_table);
    dst->seq_id = dst_seq_id;
    dst->active = true;

    int block_size = mgr->block_size;
    int full_blocks = n_tokens / block_size;
    int rem = n_tokens % block_size;
    int blocks_to_share = full_blocks + (rem > 0 ? 1 : 0);

    for (int p = 0; p < blocks_to_share; p++) {
        int block_id = src->page_table.pages[p];
        mgr->pool.blocks[block_id].ref_count++;
        if (pt_append(&dst->page_table, block_id) != 0) {
            for (int q = 0; q < p; q++) {
                int bid = dst->page_table.pages[q];
                mgr->pool.blocks[bid].ref_count--;
            }
            dst->active = false;
            dst->seq_id = KV_INVALID;
            pt_clear(&dst->page_table);
            return -5;
        }
    }
    for (int p = 0; p < full_blocks; p++) {
        int block_id = dst->page_table.pages[p];
        if (block_size > mgr->pool.blocks[block_id].used_slots) {
            mgr->pool.blocks[block_id].used_slots = block_size;
        }
    }
    if (rem > 0 && full_blocks < blocks_to_share) {
        int block_id = dst->page_table.pages[full_blocks];
        if (rem > mgr->pool.blocks[block_id].used_slots) {
            mgr->pool.blocks[block_id].used_slots = rem;
        }
    }
    dst->length = n_tokens;
    return 0;
}

int kv_block_of_token(const KVCacheManager *mgr, int seq_id, int token_idx) {
    int slot = kv_find_seq(mgr, seq_id);
    if (slot == KV_INVALID) {
        return KV_INVALID;
    }
    const Sequence *seq = &mgr->seqs[slot];
    if (token_idx < 0 || token_idx >= seq->length) {
        return KV_INVALID;
    }
    int page = token_idx / mgr->block_size;
    if (page >= seq->page_table.n_pages) {
        return KV_INVALID;
    }
    return seq->page_table.pages[page];
}

int kv_seq_length(const KVCacheManager *mgr, int seq_id) {
    int slot = kv_find_seq(mgr, seq_id);
    if (slot == KV_INVALID) {
        return -1;
    }
    return mgr->seqs[slot].length;
}

CacheStats kv_usage_stats(const KVCacheManager *mgr) {
    CacheStats stats;
    stats.total_blocks = mgr->pool.n_blocks;
    stats.allocated_blocks = mgr->pool.n_used;
    stats.free_blocks = mgr->pool.n_free;
    stats.total_slots = mgr->pool.n_blocks * mgr->pool.block_size;

    int used_slots = 0;
    int allocated_slots = 0;
    int n_active_seqs = 0;
    int total_tokens = 0;
    for (int i = 0; i < mgr->pool.n_blocks; i++) {
        if (mgr->pool.blocks[i].in_use) {
            used_slots += mgr->pool.blocks[i].used_slots;
            allocated_slots += mgr->pool.block_size;
        }
    }
    for (int i = 0; i < mgr->seq_capacity; i++) {
        if (mgr->seqs[i].active) {
            n_active_seqs++;
            total_tokens += mgr->seqs[i].length;
        }
    }
    stats.used_slots = used_slots;
    stats.wasted_slots = allocated_slots - used_slots;
    stats.n_active_seqs = n_active_seqs;
    stats.total_tokens = total_tokens;

    if (stats.total_slots > 0) {
        stats.usage_ratio = (double)used_slots / (double)stats.total_slots;
    } else {
        stats.usage_ratio = 0.0;
    }
    if (allocated_slots > 0) {
        stats.fragmentation_ratio = 1.0 - (double)used_slots / (double)allocated_slots;
    } else {
        stats.fragmentation_ratio = 0.0;
    }
    return stats;
}