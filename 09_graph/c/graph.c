#include "graph.h"

#include <stdlib.h>

static void adj_push(AdjList *a, int v) {
    if (a->size == a->capacity) {
        size_t new_cap = a->capacity == 0 ? 4 : a->capacity * 2;
        int *new_data = (int *)realloc(a->data, new_cap * sizeof(int));
        a->data = new_data;
        a->capacity = new_cap;
    }
    a->data[a->size++] = v;
}

static int adj_contains(const AdjList *a, int v) {
    for (size_t i = 0; i < a->size; i++) {
        if (a->data[i] == v) return 1;
    }
    return 0;
}

void graph_init(Graph *g, size_t n_vertices, int directed) {
    g->n_vertices = n_vertices;
    g->directed = directed;
    g->adj = (AdjList *)calloc(n_vertices, sizeof(AdjList));
}

void graph_free(Graph *g) {
    for (size_t i = 0; i < g->n_vertices; i++) {
        free(g->adj[i].data);
    }
    free(g->adj);
    g->adj = NULL;
    g->n_vertices = 0;
}

void graph_add_edge(Graph *g, int u, int v) {
    adj_push(&g->adj[u], v);
    if (!g->directed) {
        adj_push(&g->adj[v], u);
    }
}

int graph_has_edge(const Graph *g, int u, int v) {
    return adj_contains(&g->adj[u], v);
}

const int *graph_neighbors(const Graph *g, int u, size_t *out_count) {
    *out_count = g->adj[u].size;
    return g->adj[u].data;
}

size_t graph_degree(const Graph *g, int u) {
    return g->adj[u].size;
}