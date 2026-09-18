#include "skiplist.h"
#include "hnsw.h"
#include "../../common/c_utils/ds_test.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

DS_TEST(test_skiplist_init_free) {
    SkipList sl;
    skiplist_init(&sl);
    DS_ASSERT(skiplist_size(&sl) == 0, "empty skiplist size 0");
    DS_ASSERT(sl.level == 1, "initial level 1");
    skiplist_free(&sl);
    DS_ASSERT(sl.head == NULL, "head freed");
}

DS_TEST(test_skiplist_insert_search_basic) {
    SkipList sl;
    skiplist_init(&sl);
    skiplist_insert(&sl, 10, 100);
    skiplist_insert(&sl, 20, 200);
    skiplist_insert(&sl, 30, 300);
    DS_ASSERT(skiplist_size(&sl) == 3, "size 3");
    int v;
    DS_ASSERT(skiplist_search(&sl, 10, &v) && v == 100, "search key 10 -> 100");
    DS_ASSERT(skiplist_search(&sl, 20, &v) && v == 200, "search key 20 -> 200");
    DS_ASSERT(skiplist_search(&sl, 30, &v) && v == 300, "search key 30 -> 300");
    DS_ASSERT(!skiplist_search(&sl, 15, NULL), "search missing key 15");
    DS_ASSERT(!skiplist_search(&sl, 0, NULL), "search missing key 0");
    skiplist_free(&sl);
}

DS_TEST(test_skiplist_insert_duplicate_key) {
    SkipList sl;
    skiplist_init(&sl);
    skiplist_insert(&sl, 5, 50);
    skiplist_insert(&sl, 5, 55);
    DS_ASSERT(skiplist_size(&sl) == 2, "duplicate keys both inserted");
    int v;
    DS_ASSERT(skiplist_search(&sl, 5, &v), "search finds key 5");
    skiplist_free(&sl);
}

DS_TEST(test_skiplist_delete_basic) {
    SkipList sl;
    skiplist_init(&sl);
    skiplist_insert(&sl, 1, 10);
    skiplist_insert(&sl, 2, 20);
    skiplist_insert(&sl, 3, 30);
    DS_ASSERT(skiplist_delete(&sl, 2), "delete key 2");
    DS_ASSERT(skiplist_size(&sl) == 2, "size 2 after delete");
    DS_ASSERT(!skiplist_search(&sl, 2, NULL), "key 2 gone");
    DS_ASSERT(skiplist_search(&sl, 1, NULL), "key 1 still there");
    DS_ASSERT(skiplist_search(&sl, 3, NULL), "key 3 still there");
    skiplist_free(&sl);
}

DS_TEST(test_skiplist_delete_missing) {
    SkipList sl;
    skiplist_init(&sl);
    skiplist_insert(&sl, 1, 10);
    DS_ASSERT(!skiplist_delete(&sl, 99), "delete missing returns false");
    DS_ASSERT(skiplist_size(&sl) == 1, "size unchanged");
    skiplist_free(&sl);
}

DS_TEST(test_skiplist_delete_all) {
    SkipList sl;
    skiplist_init(&sl);
    for (int i = 0; i < 20; i++) {
        skiplist_insert(&sl, i, i * 10);
    }
    DS_ASSERT(skiplist_size(&sl) == 20, "20 inserted");
    for (int i = 0; i < 20; i++) {
        DS_ASSERT(skiplist_delete(&sl, i), "delete each");
    }
    DS_ASSERT(skiplist_size(&sl) == 0, "all deleted");
    DS_ASSERT(sl.level == 1, "level back to 1");
    skiplist_free(&sl);
}

DS_TEST(test_skiplist_find_le) {
    SkipList sl;
    skiplist_init(&sl);
    int keys[] = {10, 20, 30, 40, 50};
    for (int i = 0; i < 5; i++) {
        skiplist_insert(&sl, keys[i], keys[i] * 2);
    }
    int k, v;
    DS_ASSERT(skiplist_find_le(&sl, 25, &k, &v) && k == 20 && v == 40, "find_le 25 -> 20");
    DS_ASSERT(skiplist_find_le(&sl, 30, &k, &v) && k == 30 && v == 60, "find_le 30 -> 30 exact");
    DS_ASSERT(skiplist_find_le(&sl, 100, &k, &v) && k == 50, "find_le 100 -> 50");
    DS_ASSERT(!skiplist_find_le(&sl, 5, &k, &v), "find_le 5 -> none");
    skiplist_free(&sl);
}

DS_TEST(test_skiplist_find_ge) {
    SkipList sl;
    skiplist_init(&sl);
    int keys[] = {10, 20, 30, 40, 50};
    for (int i = 0; i < 5; i++) {
        skiplist_insert(&sl, keys[i], keys[i] * 2);
    }
    int k, v;
    DS_ASSERT(skiplist_find_ge(&sl, 25, &k, &v) && k == 30, "find_ge 25 -> 30");
    DS_ASSERT(skiplist_find_ge(&sl, 30, &k, &v) && k == 30, "find_ge 30 -> 30 exact");
    DS_ASSERT(skiplist_find_ge(&sl, 5, &k, &v) && k == 10, "find_ge 5 -> 10");
    DS_ASSERT(!skiplist_find_ge(&sl, 100, &k, &v), "find_ge 100 -> none");
    skiplist_free(&sl);
}

DS_TEST(test_skiplist_find_closest) {
    SkipList sl;
    skiplist_init(&sl);
    int keys[] = {10, 20, 30, 40, 50};
    for (int i = 0; i < 5; i++) {
        skiplist_insert(&sl, keys[i], keys[i] * 2);
    }
    int k, v;
    DS_ASSERT(skiplist_find_closest(&sl, 24, &k, &v) && k == 20, "closest to 24 -> 20");
    DS_ASSERT(skiplist_find_closest(&sl, 26, &k, &v) && k == 30, "closest to 26 -> 30");
    DS_ASSERT(skiplist_find_closest(&sl, 30, &k, &v) && k == 30, "closest to 30 -> 30");
    DS_ASSERT(skiplist_find_closest(&sl, 5, &k, &v) && k == 10, "closest to 5 -> 10");
    DS_ASSERT(skiplist_find_closest(&sl, 100, &k, &v) && k == 50, "closest to 100 -> 50");
    skiplist_free(&sl);
}

DS_TEST(test_skiplist_large_insert_search) {
    SkipList sl;
    skiplist_init(&sl);
    srand(12345);
    int n = 2000;
    int *keys = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        keys[i] = rand() % 100000;
        skiplist_insert(&sl, keys[i], i);
    }
    DS_ASSERT(skiplist_size(&sl) == n, "all 2000 inserted");
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (skiplist_search(&sl, keys[i], NULL)) found++;
    }
    DS_ASSERT(found == n, "all inserted keys searchable");
    int level = skiplist_max_level(&sl);
    DS_ASSERT(level >= 1 && level <= SKIPLIST_MAX_LEVEL, "level in valid range");
    DS_ASSERT(level <= 20, "level roughly O(log n)");
    free(keys);
    skiplist_free(&sl);
}

DS_TEST(test_skiplist_ordered_traversal) {
    SkipList sl;
    skiplist_init(&sl);
    int order[] = {50, 10, 40, 20, 30};
    for (int i = 0; i < 5; i++) {
        skiplist_insert(&sl, order[i], i);
    }
    SkipNode *node = sl.head->next[0];
    int prev = -1;
    int count = 0;
    bool sorted = true;
    while (node) {
        if (node->key < prev) sorted = false;
        prev = node->key;
        count++;
        node = node->next[0];
    }
    DS_ASSERT(count == 5, "traversed 5 nodes");
    DS_ASSERT(sorted, "level 0 is sorted");
    skiplist_free(&sl);
}

DS_TEST(test_hnsw_init_free) {
    HNSW h;
    hnsw_init(&h, 3, 8, 100);
    DS_ASSERT(hnsw_size(&h) == 0, "empty hnsw");
    DS_ASSERT(h.dim == 3, "dim 3");
    DS_ASSERT(h.M == 8, "M 8");
    DS_ASSERT(h.entry_point == -1, "no entry point");
    hnsw_free(&h);
    DS_ASSERT(h.vectors == NULL, "vectors freed");
}

DS_TEST(test_hnsw_insert_single) {
    HNSW h;
    hnsw_init(&h, 2, 4, 100);
    float v1[] = {1.0f, 2.0f};
    int idx = hnsw_insert(&h, v1);
    DS_ASSERT(idx == 0, "first insert returns 0");
    DS_ASSERT(hnsw_size(&h) == 1, "size 1");
    DS_ASSERT(h.entry_point == 0, "entry point is 0");
    hnsw_free(&h);
}

DS_TEST(test_hnsw_insert_multiple) {
    HNSW h;
    hnsw_init(&h, 2, 4, 100);
    float v1[] = {0.0f, 0.0f};
    float v2[] = {1.0f, 0.0f};
    float v3[] = {0.0f, 1.0f};
    float v4[] = {1.0f, 1.0f};
    DS_ASSERT(hnsw_insert(&h, v1) == 0, "insert v1");
    DS_ASSERT(hnsw_insert(&h, v2) == 1, "insert v2");
    DS_ASSERT(hnsw_insert(&h, v3) == 2, "insert v3");
    DS_ASSERT(hnsw_insert(&h, v4) == 3, "insert v4");
    DS_ASSERT(hnsw_size(&h) == 4, "size 4");
    DS_ASSERT(h.neighbor_count[0] > 0, "node 0 has neighbors");
    DS_ASSERT(h.neighbor_count[1] > 0, "node 1 has neighbors");
    hnsw_free(&h);
}

DS_TEST(test_hnsw_search_exact_match) {
    HNSW h;
    hnsw_init(&h, 2, 8, 100);
    float vecs[][2] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
    for (int i = 0; i < 4; i++) {
        hnsw_insert(&h, vecs[i]);
    }
    int result[4];
    int n = hnsw_search(&h, vecs[0], 1, result);
    DS_ASSERT(n >= 1, "search returns at least 1");
    DS_ASSERT(result[0] == 0, "exact match returns self");
    n = hnsw_search(&h, vecs[2], 1, result);
    DS_ASSERT(result[0] == 2, "exact match vecs[2]");
    hnsw_free(&h);
}

DS_TEST(test_hnsw_search_near_neighbor) {
    HNSW h;
    hnsw_init(&h, 2, 8, 100);
    float vecs[][2] = {{0.0f, 0.0f}, {10.0f, 0.0f}, {0.0f, 10.0f}, {10.0f, 10.0f}};
    for (int i = 0; i < 4; i++) {
        hnsw_insert(&h, vecs[i]);
    }
    float query[] = {0.1f, 0.1f};
    int result[4];
    int n = hnsw_search(&h, query, 1, result);
    DS_ASSERT(n >= 1, "search returns result");
    DS_ASSERT(result[0] == 0, "nearest to (0.1,0.1) is node 0 (0,0)");
    float query2[] = {9.9f, 9.9f};
    n = hnsw_search(&h, query2, 1, result);
    DS_ASSERT(result[0] == 3, "nearest to (9.9,9.9) is node 3 (10,10)");
    hnsw_free(&h);
}

DS_TEST(test_hnsw_search_k_results) {
    HNSW h;
    hnsw_init(&h, 2, 8, 100);
    float vecs[][2] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f},
                       {10.0f, 0.0f}, {20.0f, 0.0f}};
    for (int i = 0; i < 6; i++) {
        hnsw_insert(&h, vecs[i]);
    }
    float query[] = {0.0f, 0.0f};
    int result[3];
    int n = hnsw_search(&h, query, 3, result);
    DS_ASSERT(n >= 1, "search returns results");
    DS_ASSERT(n <= 3, "at most 3 results");
    DS_ASSERT(result[0] == 0, "closest is node 0");
    hnsw_free(&h);
}

DS_TEST(test_hnsw_recall_vs_brute) {
    HNSW h;
    int n = 200;
    int dim = 5;
    hnsw_init(&h, dim, 16, n);
    srand(42);
    float *vecs = (float *)malloc(sizeof(float) * n * dim);
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < dim; d++) {
            vecs[i * dim + d] = (float)(rand() % 1000) / 100.0f;
        }
        hnsw_insert(&h, &vecs[i * dim]);
    }
    DS_ASSERT(hnsw_size(&h) == n, "all 200 inserted");

    int n_queries = 50;
    int hits = 0;
    for (int q = 0; q < n_queries; q++) {
        float query[HNSW_MAX_DIM];
        for (int d = 0; d < dim; d++) {
            query[d] = (float)(rand() % 1000) / 100.0f;
        }

        int brute_best = -1;
        float brute_dist = 1e30f;
        for (int i = 0; i < n; i++) {
            float d = 0.0f;
            for (int j = 0; j < dim; j++) {
                float diff = vecs[i * dim + j] - query[j];
                d += diff * diff;
            }
            if (d < brute_dist) {
                brute_dist = d;
                brute_best = i;
            }
        }

        int result[1];
        int rn = hnsw_search(&h, query, 1, result);
        if (rn > 0 && result[0] == brute_best) {
            hits++;
        }
    }
    float recall = (float)hits / n_queries;
    DS_ASSERT(recall >= 0.3f, "recall >= 30% (greedy may miss some)");
    printf("  [INFO] recall@1 = %.0f%% (%d/%d)\n", recall * 100, hits, n_queries);
    free(vecs);
    hnsw_free(&h);
}

DS_TEST(test_hnsw_skiplist_entry_selection) {
    HNSW h;
    hnsw_init(&h, 2, 8, 100);
    float vecs[][2] = {{0.0f, 0.0f}, {3.0f, 4.0f}, {6.0f, 8.0f}, {9.0f, 12.0f}};
    for (int i = 0; i < 4; i++) {
        hnsw_insert(&h, vecs[i]);
    }
    DS_ASSERT(skiplist_size(&h.dist_index) == 4, "dist index has 4 entries");
    int k, v;
    float q_norm = 3.0f * 3.0f + 4.0f * 4.0f;
    int q_key = (int)(q_norm * HNSW_DIST_SCALE);
    DS_ASSERT(skiplist_find_closest(&h.dist_index, q_key, &k, &v), "find closest entry");
    DS_ASSERT(v >= 0 && v < 4, "entry point valid node index");
    hnsw_free(&h);
}

DS_TEST(test_hnsw_empty_search) {
    HNSW h;
    hnsw_init(&h, 2, 8, 100);
    float query[] = {1.0f, 1.0f};
    int result[4];
    int n = hnsw_search(&h, query, 1, result);
    DS_ASSERT(n == 0, "empty hnsw returns 0 results");
    hnsw_free(&h);
}

DS_TEST(test_hnsw_capacity_limit) {
    HNSW h;
    hnsw_init(&h, 2, 4, 3);
    float v[] = {1.0f, 1.0f};
    DS_ASSERT(hnsw_insert(&h, v) == 0, "insert 1 ok");
    DS_ASSERT(hnsw_insert(&h, v) == 1, "insert 2 ok");
    DS_ASSERT(hnsw_insert(&h, v) == 2, "insert 3 ok");
    DS_ASSERT(hnsw_insert(&h, v) == -1, "insert 4 fails (capacity)");
    hnsw_free(&h);
}

int main(void) {
    srand(2024);

    printf("=== SkipList Tests ===\n");
    DS_RUN(test_skiplist_init_free);
    DS_RUN(test_skiplist_insert_search_basic);
    DS_RUN(test_skiplist_insert_duplicate_key);
    DS_RUN(test_skiplist_delete_basic);
    DS_RUN(test_skiplist_delete_missing);
    DS_RUN(test_skiplist_delete_all);
    DS_RUN(test_skiplist_find_le);
    DS_RUN(test_skiplist_find_ge);
    DS_RUN(test_skiplist_find_closest);
    DS_RUN(test_skiplist_large_insert_search);
    DS_RUN(test_skiplist_ordered_traversal);

    printf("=== HNSW Tests ===\n");
    DS_RUN(test_hnsw_init_free);
    DS_RUN(test_hnsw_insert_single);
    DS_RUN(test_hnsw_insert_multiple);
    DS_RUN(test_hnsw_search_exact_match);
    DS_RUN(test_hnsw_search_near_neighbor);
    DS_RUN(test_hnsw_search_k_results);
    DS_RUN(test_hnsw_recall_vs_brute);
    DS_RUN(test_hnsw_skiplist_entry_selection);
    DS_RUN(test_hnsw_empty_search);
    DS_RUN(test_hnsw_capacity_limit);

    DS_TEST_SUMMARY();
}