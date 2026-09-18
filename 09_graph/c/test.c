#include "graph.h"
#include "../../common/c_utils/ds_test.h"

#include <stdio.h>
#include <stdlib.h>

DS_TEST(test_init_free) {
    Graph g;
    graph_init(&g, 5, 0);
    DS_ASSERT(g.n_vertices == 5, "n_vertices == 5");
    DS_ASSERT(g.directed == 0, "undirected");
    for (size_t i = 0; i < 5; i++) {
        DS_ASSERT(graph_degree(&g, (int)i) == 0, "initial degree 0");
    }
    graph_free(&g);
}

DS_TEST(test_add_edge_undirected) {
    Graph g;
    graph_init(&g, 4, 0);
    graph_add_edge(&g, 0, 1);
    DS_ASSERT(graph_has_edge(&g, 0, 1), "edge 0-1");
    DS_ASSERT(graph_has_edge(&g, 1, 0), "edge 1-0 (undirected)");
    DS_ASSERT(graph_degree(&g, 0) == 1, "degree 0 == 1");
    DS_ASSERT(graph_degree(&g, 1) == 1, "degree 1 == 1");
    graph_add_edge(&g, 0, 2);
    graph_add_edge(&g, 0, 3);
    DS_ASSERT(graph_degree(&g, 0) == 3, "degree 0 == 3");
    DS_ASSERT(graph_degree(&g, 2) == 1, "degree 2 == 1");
    DS_ASSERT(graph_degree(&g, 3) == 1, "degree 3 == 1");
    graph_free(&g);
}

DS_TEST(test_add_edge_directed) {
    Graph g;
    graph_init(&g, 4, 1);
    graph_add_edge(&g, 0, 1);
    DS_ASSERT(graph_has_edge(&g, 0, 1), "directed edge 0->1");
    DS_ASSERT(!graph_has_edge(&g, 1, 0), "no edge 1->0");
    DS_ASSERT(graph_degree(&g, 0) == 1, "out-degree 0 == 1");
    DS_ASSERT(graph_degree(&g, 1) == 0, "out-degree 1 == 0");
    graph_add_edge(&g, 2, 1);
    graph_add_edge(&g, 3, 1);
    DS_ASSERT(graph_degree(&g, 1) == 0, "1 has no out-edge");
    DS_ASSERT(graph_degree(&g, 2) == 1, "2 out-degree 1");
    graph_free(&g);
}

DS_TEST(test_has_edge_nonexistent) {
    Graph g;
    graph_init(&g, 5, 0);
    graph_add_edge(&g, 0, 1);
    graph_add_edge(&g, 2, 3);
    DS_ASSERT(!graph_has_edge(&g, 0, 3), "no edge 0-3");
    DS_ASSERT(!graph_has_edge(&g, 1, 2), "no edge 1-2");
    DS_ASSERT(!graph_has_edge(&g, 0, 4), "no edge 0-4");
    DS_ASSERT(!graph_has_edge(&g, 4, 4), "no self loop at 4");
    graph_free(&g);
}

DS_TEST(test_neighbors_traversal) {
    Graph g;
    graph_init(&g, 6, 0);
    graph_add_edge(&g, 0, 1);
    graph_add_edge(&g, 0, 2);
    graph_add_edge(&g, 0, 3);
    graph_add_edge(&g, 1, 4);
    graph_add_edge(&g, 2, 5);

    size_t count;
    const int *nb = graph_neighbors(&g, 0, &count);
    DS_ASSERT(count == 3, "vertex 0 has 3 neighbors");

    int has1 = 0, has2 = 0, has3 = 0;
    for (size_t i = 0; i < count; i++) {
        if (nb[i] == 1) has1 = 1;
        if (nb[i] == 2) has2 = 1;
        if (nb[i] == 3) has3 = 1;
    }
    DS_ASSERT(has1 && has2 && has3, "neighbors of 0 are {1,2,3}");

    const int *nb1 = graph_neighbors(&g, 1, &count);
    DS_ASSERT(count == 2, "vertex 1 has 2 neighbors");
    int has0 = 0, has4 = 0;
    for (size_t i = 0; i < count; i++) {
        if (nb1[i] == 0) has0 = 1;
        if (nb1[i] == 4) has4 = 1;
    }
    DS_ASSERT(has0 && has4, "neighbors of 1 are {0,4}");
    graph_free(&g);
}

DS_TEST(test_self_loop) {
    Graph g;
    graph_init(&g, 3, 1);
    graph_add_edge(&g, 1, 1);
    DS_ASSERT(graph_has_edge(&g, 1, 1), "self loop");
    DS_ASSERT(graph_degree(&g, 1) == 1, "self loop degree 1");
    graph_free(&g);
}

DS_TEST(test_self_loop_undirected) {
    Graph g;
    graph_init(&g, 3, 0);
    graph_add_edge(&g, 1, 1);
    DS_ASSERT(graph_has_edge(&g, 1, 1), "self loop undirected");
    DS_ASSERT(graph_degree(&g, 1) == 2, "self loop adds two entries in undirected");
    graph_free(&g);
}

DS_TEST(test_complete_graph) {
    Graph g;
    graph_init(&g, 5, 0);
    for (int i = 0; i < 5; i++) {
        for (int j = i + 1; j < 5; j++) {
            graph_add_edge(&g, i, j);
        }
    }
    for (size_t i = 0; i < 5; i++) {
        DS_ASSERT(graph_degree(&g, (int)i) == 4, "complete graph degree n-1");
    }
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (i != j) {
                DS_ASSERT(graph_has_edge(&g, i, j), "complete graph all pairs");
            }
        }
    }
    DS_ASSERT(!graph_has_edge(&g, 0, 0), "no self loop in complete graph");
    graph_free(&g);
}

DS_TEST(test_handshake_lemma) {
    Graph g;
    graph_init(&g, 100, 0);
    srand(42);
    int expected_edges = 0;
    for (int trial = 0; trial < 500; trial++) {
        int u = rand() % 100;
        int v = rand() % 100;
        if (u != v && !graph_has_edge(&g, u, v)) {
            graph_add_edge(&g, u, v);
            expected_edges++;
        }
    }
    size_t total_degree = 0;
    for (size_t i = 0; i < 100; i++) {
        total_degree += graph_degree(&g, (int)i);
    }
    DS_ASSERT(total_degree == (size_t)expected_edges * 2,
              "handshake lemma: sum degree = 2E");
    graph_free(&g);
}

DS_TEST(test_directed_handshake) {
    Graph g;
    graph_init(&g, 50, 1);
    srand(7);
    int expected_edges = 0;
    for (int trial = 0; trial < 300; trial++) {
        int u = rand() % 50;
        int v = rand() % 50;
        if (u != v) {
            graph_add_edge(&g, u, v);
            expected_edges++;
        }
    }
    size_t total_out_degree = 0;
    for (size_t i = 0; i < 50; i++) {
        total_out_degree += graph_degree(&g, (int)i);
    }
    DS_ASSERT(total_out_degree == (size_t)expected_edges,
              "directed: sum out-degree = E");
    graph_free(&g);
}

DS_TEST(test_path_traversal) {
    Graph g;
    graph_init(&g, 6, 1);
    graph_add_edge(&g, 0, 1);
    graph_add_edge(&g, 1, 2);
    graph_add_edge(&g, 2, 3);
    graph_add_edge(&g, 3, 4);
    graph_add_edge(&g, 4, 5);

    int visited[6] = {0};
    int cur = 0;
    int path_len = 1;
    visited[0] = 1;
    for (int step = 0; step < 5; step++) {
        size_t count;
        const int *nb = graph_neighbors(&g, cur, &count);
        if (count == 0) break;
        cur = nb[0];
        visited[cur] = 1;
        path_len++;
    }
    DS_ASSERT(path_len == 6, "path 0->1->2->3->4->5 length 6");
    for (int i = 0; i < 6; i++) {
        DS_ASSERT(visited[i] == 1, "all vertices on path visited");
    }
    graph_free(&g);
}

DS_TEST(test_star_graph) {
    Graph g;
    graph_init(&g, 6, 0);
    for (int i = 1; i < 6; i++) {
        graph_add_edge(&g, 0, i);
    }
    DS_ASSERT(graph_degree(&g, 0) == 5, "center degree 5");
    for (int i = 1; i < 6; i++) {
        DS_ASSERT(graph_degree(&g, i) == 1, "leaf degree 1");
    }
    for (int i = 1; i < 6; i++) {
        for (int j = 1; j < 6; j++) {
            if (i != j) {
                DS_ASSERT(!graph_has_edge(&g, i, j), "leaves not connected");
            }
        }
    }
    graph_free(&g);
}

int main(void) {
    DS_RUN(test_init_free);
    DS_RUN(test_add_edge_undirected);
    DS_RUN(test_add_edge_directed);
    DS_RUN(test_has_edge_nonexistent);
    DS_RUN(test_neighbors_traversal);
    DS_RUN(test_self_loop);
    DS_RUN(test_self_loop_undirected);
    DS_RUN(test_complete_graph);
    DS_RUN(test_handshake_lemma);
    DS_RUN(test_directed_handshake);
    DS_RUN(test_path_traversal);
    DS_RUN(test_star_graph);

    printf("--- 性能微基准 ---\n");
    srand(123);
    Graph g;
    graph_init(&g, 10000, 0);
    double t0 = (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    for (int i = 0; i < 50000; i++) {
        int u = rand() % 10000;
        int v = rand() % 10000;
        if (u != v) graph_add_edge(&g, u, v);
    }
    double t1 = (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    printf("[BENCH] add 50000 edges to 10000-vertex graph: %.3f ms\n", t1 - t0);

    size_t total_deg = 0;
    double t2 = (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    for (size_t i = 0; i < 10000; i++) {
        total_deg += graph_degree(&g, (int)i);
    }
    double t3 = (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    printf("[BENCH] sum degrees (traverse all vertices): %.3f ms, total degree = %zu\n",
           t3 - t2, total_deg);

    int has_count = 0;
    double t4 = (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    for (int trial = 0; trial < 100000; trial++) {
        int u = rand() % 10000;
        int v = rand() % 10000;
        if (graph_has_edge(&g, u, v)) has_count++;
    }
    double t5 = (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    printf("[BENCH] 100000 has_edge queries: %.3f ms, hits = %d\n",
           t5 - t4, has_count);

    graph_free(&g);

    DS_TEST_SUMMARY();
}