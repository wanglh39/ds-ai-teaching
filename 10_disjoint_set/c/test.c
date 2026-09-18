#include "dset.h"
#include "../../common/c_utils/ds_test.h"

#include <stdio.h>
#include <stdlib.h>

DS_TEST(test_init_free) {
    DSet ds;
    dset_init(&ds, 5);
    DS_ASSERT(ds.n == 5, "n == 5");
    DS_ASSERT(ds.n_sets == 5, "initial n_sets == 5");
    for (size_t i = 0; i < 5; i++) {
        DS_ASSERT(ds.parent[i] == (int)i, "parent[i] == i");
        DS_ASSERT(ds.rank[i] == 0, "rank[i] == 0");
    }
    dset_free(&ds);
    DS_ASSERT(ds.parent == NULL, "parent freed");
    DS_ASSERT(ds.rank == NULL, "rank freed");
    DS_ASSERT(ds.n == 0, "n zeroed");
}

DS_TEST(test_find_initial) {
    DSet ds;
    dset_init(&ds, 6);
    for (int i = 0; i < 6; i++) {
        DS_ASSERT(dset_find(&ds, i) == i, "find returns self initially");
    }
    dset_free(&ds);
}

DS_TEST(test_union_basic) {
    DSet ds;
    dset_init(&ds, 5);
    DS_ASSERT(dset_union_sets(&ds, 0, 1) == 1, "union 0-1 returns 1 (merged)");
    DS_ASSERT(dset_connected(&ds, 0, 1), "0 and 1 connected");
    DS_ASSERT(ds.n_sets == 4, "n_sets == 4 after one union");
    DS_ASSERT(dset_union_sets(&ds, 0, 1) == 0, "union 0-1 again returns 0 (already same)");
    DS_ASSERT(ds.n_sets == 4, "n_sets unchanged on duplicate union");
    dset_free(&ds);
}

DS_TEST(test_connected_reflexive) {
    DSet ds;
    dset_init(&ds, 4);
    for (int i = 0; i < 4; i++) {
        DS_ASSERT(dset_connected(&ds, i, i), "reflexive: connected to self");
    }
    dset_free(&ds);
}

DS_TEST(test_connected_symmetric) {
    DSet ds;
    dset_init(&ds, 4);
    dset_union_sets(&ds, 0, 2);
    DS_ASSERT(dset_connected(&ds, 0, 2), "0-2 connected");
    DS_ASSERT(dset_connected(&ds, 2, 0), "2-0 connected (symmetric)");
    dset_free(&ds);
}

DS_TEST(test_connected_transitive) {
    DSet ds;
    dset_init(&ds, 6);
    dset_union_sets(&ds, 0, 1);
    dset_union_sets(&ds, 1, 2);
    dset_union_sets(&ds, 2, 3);
    DS_ASSERT(dset_connected(&ds, 0, 3), "0-3 connected via chain");
    DS_ASSERT(dset_connected(&ds, 0, 2), "0-2 connected");
    DS_ASSERT(dset_connected(&ds, 1, 3), "1-3 connected");
    DS_ASSERT(!dset_connected(&ds, 0, 4), "0-4 not connected");
    DS_ASSERT(!dset_connected(&ds, 3, 5), "3-5 not connected");
    DS_ASSERT(ds.n_sets == 3, "n_sets == 3 (one comp of 4 + two singletons)");
    dset_free(&ds);
}

DS_TEST(test_path_compression) {
    DSet ds;
    dset_init(&ds, 8);
    for (int i = 0; i < 7; i++) {
        dset_union_sets(&ds, i, i + 1);
    }
    DS_ASSERT(dset_connected(&ds, 0, 7), "all connected in chain");
    int root = dset_find(&ds, 7);
    DS_ASSERT(ds.parent[7] == root, "find(7) compresses 7 -> root");
    DS_ASSERT(ds.parent[6] == root, "find(7) compresses 6 -> root");
    DS_ASSERT(ds.parent[5] == root, "find(7) compresses 5 -> root");
    DS_ASSERT(ds.parent[4] == root, "find(7) compresses 4 -> root");
    DS_ASSERT(ds.parent[3] == root, "find(7) compresses 3 -> root");
    DS_ASSERT(ds.parent[2] == root, "find(7) compresses 2 -> root");
    DS_ASSERT(ds.parent[1] == root, "find(7) compresses 1 -> root");
    DS_ASSERT(ds.parent[0] == root, "root points to self");
    dset_free(&ds);
}

DS_TEST(test_union_by_rank) {
    DSet ds;
    dset_init(&ds, 8);
    dset_union_sets(&ds, 0, 1);
    dset_union_sets(&ds, 2, 3);
    int r01 = dset_find(&ds, 0);
    DS_ASSERT(ds.rank[r01] == 1, "rank 1 after merging two singletons");
    dset_union_sets(&ds, 0, 2);
    int root = dset_find(&ds, 0);
    DS_ASSERT(ds.rank[root] == 2, "rank 2 after merging two rank-1 trees");
    DS_ASSERT(ds.n_sets == 5, "n_sets == 5 after 3 unions from 8");
    dset_free(&ds);
}

DS_TEST(test_rank_grows_logarithmically) {
    DSet ds;
    dset_init(&ds, 1024);
    for (int i = 1; i < 1024; i++) {
        dset_union_sets(&ds, 0, i);
    }
    int root = dset_find(&ds, 0);
    DS_ASSERT(ds.rank[root] <= 10, "rank <= log2(1024) = 10");
    DS_ASSERT(ds.rank[root] >= 1, "rank at least 1");
    DS_ASSERT(ds.n_sets == 1, "all in one set");
    dset_free(&ds);
}

DS_TEST(test_count_sets) {
    DSet ds;
    dset_init(&ds, 10);
    DS_ASSERT(dset_count_sets(&ds) == 10, "10 singletons");
    dset_union_sets(&ds, 0, 1);
    dset_union_sets(&ds, 2, 3);
    dset_union_sets(&ds, 4, 5);
    DS_ASSERT(dset_count_sets(&ds) == 7, "7 after 3 unions");
    dset_union_sets(&ds, 0, 2);
    dset_union_sets(&ds, 0, 4);
    DS_ASSERT(dset_count_sets(&ds) == 5, "5 after merging 3 pairs into 1 comp");
    dset_free(&ds);
}

DS_TEST(test_random_unions_consistent) {
    DSet ds;
    dset_init(&ds, 200);
    srand(2024);
    int *parent_check = (int *)malloc(200 * sizeof(int));
    for (int i = 0; i < 200; i++) parent_check[i] = i;
    for (int trial = 0; trial < 1000; trial++) {
        int u = rand() % 200;
        int v = rand() % 200;
        dset_union_sets(&ds, u, v);
        int ru = parent_check[u];
        while (parent_check[ru] != ru) ru = parent_check[ru];
        int rv = parent_check[v];
        while (parent_check[rv] != rv) rv = parent_check[rv];
        if (ru != rv) {
            parent_check[rv] = ru;
        }
    }
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < 200; j++) {
            int ri = parent_check[i];
            while (parent_check[ri] != ri) ri = parent_check[ri];
            int rj = parent_check[j];
            while (parent_check[rj] != rj) rj = parent_check[rj];
            int expected = (ri == rj) ? 1 : 0;
            DS_ASSERT(dset_connected(&ds, i, j) == expected, "dset matches naive union-find");
        }
    }
    free(parent_check);
    dset_free(&ds);
}

DS_TEST(test_edge_single_element) {
    DSet ds;
    dset_init(&ds, 1);
    DS_ASSERT(dset_find(&ds, 0) == 0, "find on single element");
    DS_ASSERT(dset_connected(&ds, 0, 0), "connected to self");
    DS_ASSERT(dset_union_sets(&ds, 0, 0) == 0, "union self returns 0");
    DS_ASSERT(dset_count_sets(&ds) == 1, "still 1 set");
    dset_free(&ds);
}

DS_TEST(test_all_in_one_set) {
    DSet ds;
    dset_init(&ds, 100);
    for (int i = 1; i < 100; i++) {
        dset_union_sets(&ds, 0, i);
    }
    int root = dset_find(&ds, 50);
    for (int i = 0; i < 100; i++) {
        DS_ASSERT(dset_find(&ds, i) == root, "all share same root");
    }
    DS_ASSERT(dset_count_sets(&ds) == 1, "1 set total");
    dset_free(&ds);
}

DS_TEST(test_disjoint_groups) {
    DSet ds;
    dset_init(&ds, 12);
    for (int i = 0; i < 4; i++) dset_union_sets(&ds, 0, i);
    for (int i = 4; i < 8; i++) dset_union_sets(&ds, 4, i);
    for (int i = 8; i < 12; i++) dset_union_sets(&ds, 8, i);
    DS_ASSERT(dset_connected(&ds, 0, 3), "group 1 internal");
    DS_ASSERT(dset_connected(&ds, 4, 7), "group 2 internal");
    DS_ASSERT(dset_connected(&ds, 8, 11), "group 3 internal");
    DS_ASSERT(!dset_connected(&ds, 0, 5), "group 1 vs 2 disjoint");
    DS_ASSERT(!dset_connected(&ds, 3, 9), "group 1 vs 3 disjoint");
    DS_ASSERT(!dset_connected(&ds, 6, 10), "group 2 vs 3 disjoint");
    DS_ASSERT(dset_count_sets(&ds) == 3, "3 disjoint groups");
    dset_free(&ds);
}

int main(void) {
    DS_RUN(test_init_free);
    DS_RUN(test_find_initial);
    DS_RUN(test_union_basic);
    DS_RUN(test_connected_reflexive);
    DS_RUN(test_connected_symmetric);
    DS_RUN(test_connected_transitive);
    DS_RUN(test_path_compression);
    DS_RUN(test_union_by_rank);
    DS_RUN(test_rank_grows_logarithmically);
    DS_RUN(test_count_sets);
    DS_RUN(test_random_unions_consistent);
    DS_RUN(test_edge_single_element);
    DS_RUN(test_all_in_one_set);
    DS_RUN(test_disjoint_groups);

    printf("--- 性能微基准 ---\n");
    srand(777);
    DSet ds;
    dset_init(&ds, 100000);
    double t0 = (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    for (int i = 0; i < 200000; i++) {
        int u = rand() % 100000;
        int v = rand() % 100000;
        dset_union_sets(&ds, u, v);
    }
    double t1 = (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    printf("[BENCH] 200000 unions on 100000 elements: %.3f ms, n_sets = %zu\n",
           t1 - t0, dset_count_sets(&ds));

    double t2 = (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    int connected_count = 0;
    for (int i = 0; i < 200000; i++) {
        int u = rand() % 100000;
        int v = rand() % 100000;
        if (dset_connected(&ds, u, v)) connected_count++;
    }
    double t3 = (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    printf("[BENCH] 200000 connected queries: %.3f ms, hits = %d\n",
           t3 - t2, connected_count);

    dset_free(&ds);

    DS_TEST_SUMMARY();
}