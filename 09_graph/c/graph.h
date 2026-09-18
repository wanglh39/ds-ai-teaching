#ifndef DS_GRAPH_H
#define DS_GRAPH_H

#include <stddef.h>

typedef struct {
    int *data;
    size_t size;
    size_t capacity;
} AdjList;

typedef struct {
    size_t n_vertices;
    AdjList *adj;
    int directed;
} Graph;

void graph_init(Graph *g, size_t n_vertices, int directed);
void graph_free(Graph *g);
void graph_add_edge(Graph *g, int u, int v);
int graph_has_edge(const Graph *g, int u, int v);
const int *graph_neighbors(const Graph *g, int u, size_t *out_count);
size_t graph_degree(const Graph *g, int u);

#endif