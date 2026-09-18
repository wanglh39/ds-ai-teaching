#ifndef PAGED_CACHE_H
#define PAGED_CACHE_H

#include <stddef.h>
#include <stdbool.h>

#define KV_INVALID -1

typedef struct {
    int block_id;
    int ref_count;
    int used_slots;
    int next_free;
    bool in_use;
} Block;

typedef struct {
    Block *blocks;
    int n_blocks;
    int block_size;
    int free_list_head;
    int n_free;
    int n_used;
} BlockPool;

typedef struct {
    int *pages;
    int n_pages;
    int capacity;
} PageTable;

typedef struct {
    int seq_id;
    int length;
    PageTable page_table;
    bool active;
} Sequence;

typedef struct {
    BlockPool pool;
    Sequence *seqs;
    int n_seqs;
    int seq_capacity;
    int block_size;
} KVCacheManager;

typedef struct {
    int total_blocks;
    int allocated_blocks;
    int free_blocks;
    int total_slots;
    int used_slots;
    int wasted_slots;
    double usage_ratio;
    double fragmentation_ratio;
    int n_active_seqs;
    int total_tokens;
} CacheStats;

int kv_init(KVCacheManager *mgr, int n_blocks, int block_size, int max_seqs);
void kv_free(KVCacheManager *mgr);
int kv_allocate_seq(KVCacheManager *mgr, int seq_id, int n_tokens);
int kv_append_token(KVCacheManager *mgr, int seq_id, int n_tokens);
int kv_free_seq(KVCacheManager *mgr, int seq_id);
int kv_share_prefix(KVCacheManager *mgr, int src_seq_id, int dst_seq_id, int n_tokens);
CacheStats kv_usage_stats(const KVCacheManager *mgr);
int kv_block_of_token(const KVCacheManager *mgr, int seq_id, int token_idx);
int kv_seq_length(const KVCacheManager *mgr, int seq_id);
int kv_find_seq(const KVCacheManager *mgr, int seq_id);

#endif