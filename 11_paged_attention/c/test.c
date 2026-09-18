#include "paged_cache.h"
#include "../../common/c_utils/ds_test.h"

#include <stdio.h>
#include <stdlib.h>

DS_TEST(test_init_free) {
    KVCacheManager mgr;
    int rc = kv_init(&mgr, 16, 4, 8);
    DS_ASSERT(rc == 0, "init returns 0");
    DS_ASSERT(mgr.pool.n_blocks == 16, "n_blocks == 16");
    DS_ASSERT(mgr.pool.block_size == 4, "block_size == 4");
    DS_ASSERT(mgr.pool.n_free == 16, "n_free == 16 initially");
    DS_ASSERT(mgr.pool.n_used == 0, "n_used == 0 initially");
    DS_ASSERT(mgr.pool.free_list_head == 0, "free list head == 0");
    DS_ASSERT(mgr.seq_capacity == 8, "seq_capacity == 8");
    kv_free(&mgr);
    DS_ASSERT(mgr.pool.blocks == NULL, "blocks freed");
    DS_ASSERT(mgr.seqs == NULL, "seqs freed");
}

DS_TEST(test_init_invalid_args) {
    KVCacheManager mgr;
    DS_ASSERT(kv_init(&mgr, 0, 4, 8) == -1, "n_blocks=0 rejected");
    DS_ASSERT(kv_init(&mgr, 16, 0, 8) == -1, "block_size=0 rejected");
    DS_ASSERT(kv_init(&mgr, 16, 4, 0) == -1, "max_seqs=0 rejected");
}

DS_TEST(test_allocate_single_seq) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    int rc = kv_allocate_seq(&mgr, 1, 10);
    DS_ASSERT(rc == 0, "allocate seq 1 with 10 tokens");
    DS_ASSERT(kv_seq_length(&mgr, 1) == 10, "seq 1 length == 10");
    DS_ASSERT(mgr.pool.n_used == 3, "3 blocks used (10/4 ceil)");
    DS_ASSERT(mgr.pool.n_free == 13, "13 blocks free");
    CacheStats s = kv_usage_stats(&mgr);
    DS_ASSERT(s.allocated_blocks == 3, "stats allocated_blocks == 3");
    DS_ASSERT(s.used_slots == 10, "stats used_slots == 10");
    DS_ASSERT(s.n_active_seqs == 1, "1 active seq");
    DS_ASSERT(s.total_tokens == 10, "total_tokens == 10");
    kv_free(&mgr);
}

DS_TEST(test_allocate_exact_block_multiple) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    DS_ASSERT(kv_allocate_seq(&mgr, 1, 8) == 0, "allocate 8 tokens = 2 blocks exactly");
    DS_ASSERT(mgr.pool.n_used == 2, "2 blocks used");
    CacheStats s = kv_usage_stats(&mgr);
    DS_ASSERT(s.used_slots == 8, "8 used slots");
    DS_ASSERT(s.wasted_slots == 0, "0 wasted slots (exact multiple)");
    DS_ASSERT(s.fragmentation_ratio == 0.0, "fragmentation == 0 for exact multiple");
    kv_free(&mgr);
}

DS_TEST(test_allocate_zero_tokens) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    DS_ASSERT(kv_allocate_seq(&mgr, 1, 0) == 0, "allocate 0 tokens ok");
    DS_ASSERT(kv_seq_length(&mgr, 1) == 0, "length == 0");
    DS_ASSERT(mgr.pool.n_used == 0, "0 blocks used");
    kv_free(&mgr);
}

DS_TEST(test_free_seq_returns_blocks) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    kv_allocate_seq(&mgr, 1, 10);
    DS_ASSERT(mgr.pool.n_used == 3, "3 blocks used after alloc");
    int rc = kv_free_seq(&mgr, 1);
    DS_ASSERT(rc == 0, "free seq 1 returns 0");
    DS_ASSERT(mgr.pool.n_used == 0, "0 blocks used after free");
    DS_ASSERT(mgr.pool.n_free == 16, "16 blocks free after free");
    DS_ASSERT(kv_seq_length(&mgr, 1) == -1, "seq 1 no longer exists");
    CacheStats s = kv_usage_stats(&mgr);
    DS_ASSERT(s.n_active_seqs == 0, "0 active seqs");
    kv_free(&mgr);
}

DS_TEST(test_free_nonexistent_seq) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    DS_ASSERT(kv_free_seq(&mgr, 99) == -1, "free nonexistent seq returns -1");
    kv_free(&mgr);
}

DS_TEST(test_allocate_duplicate_seq_id) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    DS_ASSERT(kv_allocate_seq(&mgr, 1, 4) == 0, "first alloc ok");
    DS_ASSERT(kv_allocate_seq(&mgr, 1, 4) == -1, "duplicate seq_id rejected");
    kv_free(&mgr);
}

DS_TEST(test_oom_when_blocks_exhausted) {
    KVCacheManager mgr;
    kv_init(&mgr, 4, 4, 8);
    DS_ASSERT(kv_allocate_seq(&mgr, 1, 16) == 0, "alloc 16 tokens = 4 blocks, fits");
    DS_ASSERT(mgr.pool.n_used == 4, "all 4 blocks used");
    DS_ASSERT(kv_allocate_seq(&mgr, 2, 1) == -3, "alloc 1 more token fails (no blocks)");
    DS_ASSERT(mgr.pool.n_used == 4, "still 4 blocks used after failed alloc");
    kv_free(&mgr);
}

DS_TEST(test_oom_when_seq_slots_full) {
    KVCacheManager mgr;
    kv_init(&mgr, 64, 4, 2);
    DS_ASSERT(kv_allocate_seq(&mgr, 1, 4) == 0, "seq 1 ok");
    DS_ASSERT(kv_allocate_seq(&mgr, 2, 4) == 0, "seq 2 ok");
    DS_ASSERT(kv_allocate_seq(&mgr, 3, 4) == -2, "seq 3 fails (no seq slots)");
    kv_free(&mgr);
}

DS_TEST(test_block_reuse_after_free) {
    KVCacheManager mgr;
    kv_init(&mgr, 8, 4, 8);
    kv_allocate_seq(&mgr, 1, 16);
    DS_ASSERT(mgr.pool.n_used == 4, "4 blocks used by seq 1");
    int first_blocks[4];
    for (int t = 0; t < 16; t += 4) {
        first_blocks[t / 4] = kv_block_of_token(&mgr, 1, t);
    }
    kv_free_seq(&mgr, 1);
    kv_allocate_seq(&mgr, 2, 16);
    DS_ASSERT(mgr.pool.n_used == 4, "4 blocks reused by seq 2");
    bool reused[4] = {false, false, false, false};
    for (int t = 0; t < 16; t += 4) {
        int b = kv_block_of_token(&mgr, 2, t);
        DS_ASSERT(b != KV_INVALID, "seq 2 has valid block");
        for (int k = 0; k < 4; k++) {
            if (first_blocks[k] == b) {
                reused[k] = true;
            }
        }
    }
    DS_ASSERT(reused[0] && reused[1] && reused[2] && reused[3],
              "seq 2 reuses all 4 blocks freed by seq 1 (LIFO free list)");
    kv_free(&mgr);
}

DS_TEST(test_append_token_grows_seq) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    kv_allocate_seq(&mgr, 1, 6);
    DS_ASSERT(mgr.pool.n_used == 2, "6 tokens = 2 blocks");
    int rc = kv_append_token(&mgr, 1, 4);
    DS_ASSERT(rc == 0, "append 4 tokens ok");
    DS_ASSERT(kv_seq_length(&mgr, 1) == 10, "length now 10");
    DS_ASSERT(mgr.pool.n_used == 3, "10 tokens = 3 blocks");
    CacheStats s = kv_usage_stats(&mgr);
    DS_ASSERT(s.used_slots == 10, "10 used slots");
    kv_free(&mgr);
}

DS_TEST(test_append_fills_partial_block) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    kv_allocate_seq(&mgr, 1, 6);
    DS_ASSERT(mgr.pool.n_used == 2, "6 tokens = 2 blocks (block 1 has 4, block 2 has 2)");
    int block_of_5 = kv_block_of_token(&mgr, 1, 5);
    DS_ASSERT(mgr.pool.blocks[block_of_5].used_slots == 2, "block 2 has 2 used slots");
    kv_append_token(&mgr, 1, 1);
    DS_ASSERT(mgr.pool.n_used == 2, "still 2 blocks (filled partial)");
    DS_ASSERT(mgr.pool.blocks[block_of_5].used_slots == 3, "block 2 now has 3 used slots");
    DS_ASSERT(kv_seq_length(&mgr, 1) == 7, "length 7");
    kv_free(&mgr);
}

DS_TEST(test_append_oom) {
    KVCacheManager mgr;
    kv_init(&mgr, 3, 4, 8);
    kv_allocate_seq(&mgr, 1, 12);
    DS_ASSERT(mgr.pool.n_used == 3, "12 tokens = 3 blocks, all used");
    int rc = kv_append_token(&mgr, 1, 1);
    DS_ASSERT(rc == -2, "append fails with -2 (no blocks)");
    DS_ASSERT(kv_seq_length(&mgr, 1) == 12, "length unchanged after failed append");
    kv_free(&mgr);
}

DS_TEST(test_block_of_token_mapping) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    kv_allocate_seq(&mgr, 1, 10);
    int b0 = kv_block_of_token(&mgr, 1, 0);
    int b3 = kv_block_of_token(&mgr, 1, 3);
    int b4 = kv_block_of_token(&mgr, 1, 4);
    int b9 = kv_block_of_token(&mgr, 1, 9);
    DS_ASSERT(b0 == b3, "tokens 0,3 in same block (page 0)");
    DS_ASSERT(b4 != b0, "token 4 in next block (page 1)");
    DS_ASSERT(b9 != KV_INVALID, "token 9 valid");
    DS_ASSERT(kv_block_of_token(&mgr, 1, 10) == KV_INVALID, "token 10 out of range");
    DS_ASSERT(kv_block_of_token(&mgr, 1, -1) == KV_INVALID, "token -1 invalid");
    DS_ASSERT(kv_block_of_token(&mgr, 99, 0) == KV_INVALID, "nonexistent seq");
    kv_free(&mgr);
}

DS_TEST(test_share_prefix_basic) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    kv_allocate_seq(&mgr, 1, 12);
    DS_ASSERT(mgr.pool.n_used == 3, "seq 1 uses 3 blocks");
    int rc = kv_share_prefix(&mgr, 1, 2, 8);
    DS_ASSERT(rc == 0, "share 8-token prefix to seq 2");
    DS_ASSERT(mgr.pool.n_used == 3, "still 3 blocks (shared, no new blocks)");
    DS_ASSERT(kv_seq_length(&mgr, 2) == 8, "seq 2 length == 8");
    int b0_seq1 = kv_block_of_token(&mgr, 1, 0);
    int b0_seq2 = kv_block_of_token(&mgr, 2, 0);
    DS_ASSERT(b0_seq1 == b0_seq2, "seq 1 and 2 share block 0");
    int b1_seq1 = kv_block_of_token(&mgr, 1, 4);
    int b1_seq2 = kv_block_of_token(&mgr, 2, 4);
    DS_ASSERT(b1_seq1 == b1_seq2, "seq 1 and 2 share block 1");
    kv_free(&mgr);
}

DS_TEST(test_share_prefix_refcount) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    kv_allocate_seq(&mgr, 1, 8);
    kv_share_prefix(&mgr, 1, 2, 8);
    int b0 = kv_block_of_token(&mgr, 1, 0);
    DS_ASSERT(mgr.pool.blocks[b0].ref_count == 2, "block ref_count == 2 after sharing");
    kv_free_seq(&mgr, 2);
    DS_ASSERT(mgr.pool.blocks[b0].ref_count == 1, "ref_count back to 1 after freeing seq 2");
    DS_ASSERT(mgr.pool.blocks[b0].in_use == true, "block still in use (seq 1 alive)");
    kv_free(&mgr);
}

DS_TEST(test_share_prefix_then_free_both) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    kv_allocate_seq(&mgr, 1, 8);
    kv_share_prefix(&mgr, 1, 2, 8);
    DS_ASSERT(mgr.pool.n_used == 2, "2 blocks shared by 2 seqs");
    kv_free_seq(&mgr, 1);
    DS_ASSERT(mgr.pool.n_used == 2, "blocks still alive (seq 2 holds ref)");
    kv_free_seq(&mgr, 2);
    DS_ASSERT(mgr.pool.n_used == 0, "all blocks freed after both seqs released");
    kv_free(&mgr);
}

DS_TEST(test_share_partial_prefix) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    kv_allocate_seq(&mgr, 1, 10);
    kv_share_prefix(&mgr, 1, 2, 6);
    DS_ASSERT(kv_seq_length(&mgr, 2) == 6, "seq 2 has 6 tokens");
    DS_ASSERT(mgr.pool.n_used == 3, "still 3 blocks (6 tokens fit in 2 blocks, both shared)");
    int b0 = kv_block_of_token(&mgr, 1, 0);
    DS_ASSERT(mgr.pool.blocks[b0].ref_count == 2, "block 0 ref_count == 2");
    int b1 = kv_block_of_token(&mgr, 1, 4);
    DS_ASSERT(mgr.pool.blocks[b1].ref_count == 2, "block 1 ref_count == 2 (partial share)");
    DS_ASSERT(mgr.pool.blocks[b1].used_slots == 4, "block 1 used_slots == 4 (seq 1 fills it)");
    kv_free(&mgr);
}

DS_TEST(test_share_prefix_errors) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    DS_ASSERT(kv_share_prefix(&mgr, 99, 2, 4) == -1, "share from nonexistent src");
    kv_allocate_seq(&mgr, 1, 8);
    kv_share_prefix(&mgr, 1, 2, 4);
    DS_ASSERT(kv_share_prefix(&mgr, 1, 2, 4) == -2, "share to existing dst rejected");
    DS_ASSERT(kv_share_prefix(&mgr, 1, 3, 100) == -3, "share more than src length rejected");
    kv_free(&mgr);
}

DS_TEST(test_fragmentation_ratio) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    kv_allocate_seq(&mgr, 1, 5);
    CacheStats s = kv_usage_stats(&mgr);
    DS_ASSERT(s.allocated_blocks == 2, "5 tokens = 2 blocks allocated");
    DS_ASSERT(s.used_slots == 5, "5 slots used");
    DS_ASSERT(s.wasted_slots == 3, "3 slots wasted (8 - 5)");
    double expected_frag = 1.0 - 5.0 / 8.0;
    DS_ASSERT(s.fragmentation_ratio > expected_frag - 1e-9 && s.fragmentation_ratio < expected_frag + 1e-9,
              "fragmentation_ratio == 3/8");
    kv_free(&mgr);
}

DS_TEST(test_usage_ratio_multiple_seqs) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    kv_allocate_seq(&mgr, 1, 8);
    kv_allocate_seq(&mgr, 2, 8);
    kv_allocate_seq(&mgr, 3, 8);
    CacheStats s = kv_usage_stats(&mgr);
    DS_ASSERT(s.allocated_blocks == 6, "3 seqs * 2 blocks = 6 blocks");
    DS_ASSERT(s.used_slots == 24, "24 slots used");
    DS_ASSERT(s.total_slots == 64, "64 total slots");
    double expected_usage = 24.0 / 64.0;
    DS_ASSERT(s.usage_ratio > expected_usage - 1e-9 && s.usage_ratio < expected_usage + 1e-9,
              "usage_ratio == 24/64");
    DS_ASSERT(s.n_active_seqs == 3, "3 active seqs");
    DS_ASSERT(s.total_tokens == 24, "total_tokens == 24");
    kv_free(&mgr);
}

DS_TEST(test_interleaved_alloc_free) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    kv_allocate_seq(&mgr, 1, 8);
    kv_allocate_seq(&mgr, 2, 8);
    kv_free_seq(&mgr, 1);
    kv_allocate_seq(&mgr, 3, 8);
    kv_allocate_seq(&mgr, 4, 8);
    kv_free_seq(&mgr, 2);
    kv_allocate_seq(&mgr, 5, 8);
    CacheStats s = kv_usage_stats(&mgr);
    DS_ASSERT(s.n_active_seqs == 3, "3 active seqs (3, 4, 5)");
    DS_ASSERT(s.allocated_blocks == 6, "6 blocks used");
    DS_ASSERT(s.used_slots == 24, "24 slots used");
    DS_ASSERT(s.fragmentation_ratio == 0.0, "no fragmentation (all exact multiples)");
    kv_free(&mgr);
}

DS_TEST(test_random_workload_consistency) {
    KVCacheManager mgr;
    kv_init(&mgr, 64, 4, 32);
    srand(2024);
    int active_seqs[32] = {0};
    int seq_lengths[32] = {0};
    for (int trial = 0; trial < 500; trial++) {
        int op = rand() % 3;
        if (op == 0) {
            for (int sid = 0; sid < 32; sid++) {
                if (!active_seqs[sid]) {
                    int len = (rand() % 20) + 1;
                    if (kv_allocate_seq(&mgr, sid, len) == 0) {
                        active_seqs[sid] = 1;
                        seq_lengths[sid] = len;
                    }
                    break;
                }
            }
        } else if (op == 1) {
            for (int sid = 0; sid < 32; sid++) {
                if (active_seqs[sid]) {
                    kv_free_seq(&mgr, sid);
                    active_seqs[sid] = 0;
                    seq_lengths[sid] = 0;
                    break;
                }
            }
        } else {
            for (int sid = 0; sid < 32; sid++) {
                if (active_seqs[sid]) {
                    int extra = (rand() % 8) + 1;
                    if (kv_append_token(&mgr, sid, extra) == 0) {
                        seq_lengths[sid] += extra;
                    }
                    break;
                }
            }
        }
    }
    int total_tokens = 0;
    for (int sid = 0; sid < 32; sid++) {
        if (active_seqs[sid]) {
            DS_ASSERT(kv_seq_length(&mgr, sid) == seq_lengths[sid],
                      "tracked length matches manager");
            total_tokens += seq_lengths[sid];
        }
    }
    CacheStats s = kv_usage_stats(&mgr);
    DS_ASSERT(s.total_tokens == total_tokens, "stats total_tokens matches tracked");
    DS_ASSERT(s.used_slots == total_tokens, "used_slots == total_tokens (no sharing)");
    DS_ASSERT(s.used_slots <= s.allocated_blocks * 4, "used_slots <= allocated capacity");
    kv_free(&mgr);
}

DS_TEST(test_page_table_grows_dynamically) {
    KVCacheManager mgr;
    kv_init(&mgr, 256, 4, 8);
    kv_allocate_seq(&mgr, 1, 4);
    DS_ASSERT(mgr.seqs[0].page_table.capacity >= 1, "page table has capacity >= 1");
    for (int i = 0; i < 50; i++) {
        kv_append_token(&mgr, 1, 4);
    }
    DS_ASSERT(kv_seq_length(&mgr, 1) == 4 + 50 * 4, "length == 204");
    DS_ASSERT(mgr.pool.n_used == 51, "51 blocks used");
    DS_ASSERT(mgr.seqs[0].page_table.n_pages == 51, "page table has 51 pages");
    DS_ASSERT(mgr.seqs[0].page_table.capacity >= 51, "page table capacity grew");
    kv_free(&mgr);
}

DS_TEST(test_empty_manager_stats) {
    KVCacheManager mgr;
    kv_init(&mgr, 16, 4, 8);
    CacheStats s = kv_usage_stats(&mgr);
    DS_ASSERT(s.total_blocks == 16, "total_blocks == 16");
    DS_ASSERT(s.allocated_blocks == 0, "allocated_blocks == 0");
    DS_ASSERT(s.free_blocks == 16, "free_blocks == 16");
    DS_ASSERT(s.used_slots == 0, "used_slots == 0");
    DS_ASSERT(s.usage_ratio == 0.0, "usage_ratio == 0");
    DS_ASSERT(s.fragmentation_ratio == 0.0, "fragmentation_ratio == 0 when nothing allocated");
    DS_ASSERT(s.n_active_seqs == 0, "no active seqs");
    kv_free(&mgr);
}

int main(void) {
    DS_RUN(test_init_free);
    DS_RUN(test_init_invalid_args);
    DS_RUN(test_allocate_single_seq);
    DS_RUN(test_allocate_exact_block_multiple);
    DS_RUN(test_allocate_zero_tokens);
    DS_RUN(test_free_seq_returns_blocks);
    DS_RUN(test_free_nonexistent_seq);
    DS_RUN(test_allocate_duplicate_seq_id);
    DS_RUN(test_oom_when_blocks_exhausted);
    DS_RUN(test_oom_when_seq_slots_full);
    DS_RUN(test_block_reuse_after_free);
    DS_RUN(test_append_token_grows_seq);
    DS_RUN(test_append_fills_partial_block);
    DS_RUN(test_append_oom);
    DS_RUN(test_block_of_token_mapping);
    DS_RUN(test_share_prefix_basic);
    DS_RUN(test_share_prefix_refcount);
    DS_RUN(test_share_prefix_then_free_both);
    DS_RUN(test_share_partial_prefix);
    DS_RUN(test_share_prefix_errors);
    DS_RUN(test_fragmentation_ratio);
    DS_RUN(test_usage_ratio_multiple_seqs);
    DS_RUN(test_interleaved_alloc_free);
    DS_RUN(test_random_workload_consistency);
    DS_RUN(test_page_table_grows_dynamically);
    DS_RUN(test_empty_manager_stats);

    printf("--- 性能微基准 ---\n");
    KVCacheManager mgr;
    kv_init(&mgr, 100000, 16, 1000);
    srand(777);
    double t0 = (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    for (int i = 0; i < 500; i++) {
        kv_allocate_seq(&mgr, i, 128);
    }
    for (int i = 0; i < 500; i++) {
        kv_append_token(&mgr, i, 64);
    }
    double t1 = (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    CacheStats s = kv_usage_stats(&mgr);
    printf("[BENCH] 500 seqs * (128 + 64) tokens: %.3f ms, used_blocks=%d, used_slots=%d\n",
           t1 - t0, s.allocated_blocks, s.used_slots);

    double t2 = (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    for (int i = 0; i < 500; i++) {
        kv_free_seq(&mgr, i);
    }
    double t3 = (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    printf("[BENCH] free 500 seqs: %.3f ms, free_blocks=%d\n",
           t3 - t2, mgr.pool.n_free);

    kv_free(&mgr);

    DS_TEST_SUMMARY();
}