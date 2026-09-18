#ifndef HNSW_H
#define HNSW_H

#include "skiplist.h"
#include <stdbool.h>

#define HNSW_MAX_DIM 64
#define HNSW_MAX_M 32
#define HNSW_DIST_SCALE 1000

typedef struct {
    float coords[HNSW_MAX_DIM];
} HnswVector;

typedef struct {
    HnswVector *vectors;
    int **neighbors;
    int *neighbor_count;
    int n_nodes;
    int capacity;
    int dim;
    int M;
    int entry_point;
    SkipList dist_index;
} HNSW;

typedef struct {
    int index;
    float dist;
} HnswHit;

void hnsw_init(HNSW *h, int dim, int M, int capacity);
void hnsw_free(HNSW *h);
int hnsw_insert(HNSW *h, const float *coords);
int hnsw_search(const HNSW *h, const float *query, int k, int *result);
float hnsw_distance_sq(const HNSW *h, int i, const float *query);
float hnsw_distance_sq_vec(const HNSW *h, int i, int j);
int hnsw_size(const HNSW *h);

#endif