#include "hnsw.h"

#include <stdlib.h>
#include <string.h>

static float dist_sq_raw(const float *a, const float *b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

static int quantize_dist(float d) {
    int q = (int)(d * HNSW_DIST_SCALE);
    if (q < 0) q = 0;
    return q;
}

void hnsw_init(HNSW *h, int dim, int M, int capacity) {
    h->dim = dim;
    h->M = M;
    h->capacity = capacity;
    h->n_nodes = 0;
    h->entry_point = -1;
    h->vectors = (HnswVector *)malloc(sizeof(HnswVector) * capacity);
    h->neighbors = (int **)malloc(sizeof(int *) * capacity);
    h->neighbor_count = (int *)malloc(sizeof(int) * capacity);
    for (int i = 0; i < capacity; i++) {
        h->neighbors[i] = (int *)malloc(sizeof(int) * HNSW_MAX_M);
        h->neighbor_count[i] = 0;
    }
    skiplist_init(&h->dist_index);
}

void hnsw_free(HNSW *h) {
    if (h->vectors) {
        free(h->vectors);
        h->vectors = NULL;
    }
    if (h->neighbors) {
        for (int i = 0; i < h->capacity; i++) {
            free(h->neighbors[i]);
        }
        free(h->neighbors);
        h->neighbors = NULL;
    }
    if (h->neighbor_count) {
        free(h->neighbor_count);
        h->neighbor_count = NULL;
    }
    skiplist_free(&h->dist_index);
    h->n_nodes = 0;
    h->capacity = 0;
    h->entry_point = -1;
}

float hnsw_distance_sq(const HNSW *h, int i, const float *query) {
    return dist_sq_raw(h->vectors[i].coords, query, h->dim);
}

float hnsw_distance_sq_vec(const HNSW *h, int i, int j) {
    return dist_sq_raw(h->vectors[i].coords, h->vectors[j].coords, h->dim);
}

static void add_edge(HNSW *h, int u, int v) {
    if (h->neighbor_count[u] >= HNSW_MAX_M) {
        float max_d = 0.0f;
        int max_idx = 0;
        for (int i = 0; i < h->neighbor_count[u]; i++) {
            float d = hnsw_distance_sq_vec(h, u, h->neighbors[u][i]);
            if (d > max_d) {
                max_d = d;
                max_idx = i;
            }
        }
        float d_new = hnsw_distance_sq_vec(h, u, v);
        if (d_new < max_d) {
            h->neighbors[u][max_idx] = v;
        }
    } else {
        h->neighbors[u][h->neighbor_count[u]] = v;
        h->neighbor_count[u]++;
    }
}

int hnsw_insert(HNSW *h, const float *coords) {
    if (h->n_nodes >= h->capacity) {
        return -1;
    }
    int idx = h->n_nodes;
    for (int i = 0; i < h->dim; i++) {
        h->vectors[idx].coords[i] = coords[i];
    }
    h->neighbor_count[idx] = 0;
    h->n_nodes++;

    float norm_sq = dist_sq_raw(coords, coords, h->dim);
    int key = quantize_dist(norm_sq);
    skiplist_insert(&h->dist_index, key, idx);

    if (h->entry_point == -1) {
        h->entry_point = idx;
        return idx;
    }

    int *cand = (int *)malloc(sizeof(int) * h->n_nodes);
    int n_cand = 0;
    for (int i = 0; i < idx; i++) {
        cand[n_cand++] = i;
    }

    int M = h->M;
    if (n_cand < M) {
        M = n_cand;
    }

    int *selected = (int *)malloc(sizeof(int) * M);
    bool *used = (bool *)calloc(idx, sizeof(bool));
    for (int s = 0; s < M; s++) {
        float best_d = 1e30f;
        int best_i = -1;
        for (int i = 0; i < n_cand; i++) {
            if (used[cand[i]]) continue;
            float d = hnsw_distance_sq_vec(h, idx, cand[i]);
            if (d < best_d) {
                best_d = d;
                best_i = cand[i];
            }
        }
        if (best_i == -1) break;
        selected[s] = best_i;
        used[best_i] = true;
    }

    for (int s = 0; s < M; s++) {
        add_edge(h, idx, selected[s]);
        add_edge(h, selected[s], idx);
    }

    free(cand);
    free(selected);
    free(used);
    return idx;
}

static int cmp_hit(const void *a, const void *b) {
    const HnswHit *ha = (const HnswHit *)a;
    const HnswHit *hb = (const HnswHit *)b;
    if (ha->dist < hb->dist) return -1;
    if (ha->dist > hb->dist) return 1;
    return 0;
}

int hnsw_search(const HNSW *h, const float *query, int k, int *result) {
    if (h->n_nodes == 0 || h->entry_point == -1) {
        return 0;
    }

    float q_norm_sq = dist_sq_raw(query, query, h->dim);
    int q_key = quantize_dist(q_norm_sq);

    int entry = h->entry_point;
    int found_key, found_value;
    if (skiplist_find_closest(&h->dist_index, q_key, &found_key, &found_value)) {
        entry = found_value;
    }

    bool *visited = (bool *)calloc(h->n_nodes, sizeof(bool));
    int current = entry;
    float current_dist = hnsw_distance_sq(h, current, query);

    for (int step = 0; step < h->n_nodes; step++) {
        visited[current] = true;
        int best_neighbor = -1;
        float best_dist = current_dist;

        for (int i = 0; i < h->neighbor_count[current]; i++) {
            int nb = h->neighbors[current][i];
            if (visited[nb]) continue;
            float d = hnsw_distance_sq(h, nb, query);
            if (d < best_dist) {
                best_dist = d;
                best_neighbor = nb;
            }
        }

        if (best_neighbor == -1) break;
        current = best_neighbor;
        current_dist = best_dist;
    }

    HnswHit *hits = (HnswHit *)malloc(sizeof(HnswHit) * h->n_nodes);
    int n_hits = 0;

    if (!visited[current]) {
        hits[n_hits].index = current;
        hits[n_hits].dist = hnsw_distance_sq(h, current, query);
        n_hits++;
    } else {
        hits[n_hits].index = current;
        hits[n_hits].dist = current_dist;
        n_hits++;
    }

    for (int i = 0; i < h->neighbor_count[current]; i++) {
        int nb = h->neighbors[current][i];
        bool dup = false;
        for (int j = 0; j < n_hits; j++) {
            if (hits[j].index == nb) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            hits[n_hits].index = nb;
            hits[n_hits].dist = hnsw_distance_sq(h, nb, query);
            n_hits++;
        }
    }

    for (int i = 0; i < h->neighbor_count[current]; i++) {
        int nb = h->neighbors[current][i];
        for (int j = 0; j < h->neighbor_count[nb]; j++) {
            int nb2 = h->neighbors[nb][j];
            bool dup = false;
            for (int t = 0; t < n_hits; t++) {
                if (hits[t].index == nb2) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                hits[n_hits].index = nb2;
                hits[n_hits].dist = hnsw_distance_sq(h, nb2, query);
                n_hits++;
            }
        }
    }

    qsort(hits, n_hits, sizeof(HnswHit), cmp_hit);

    int n_result = k < n_hits ? k : n_hits;
    for (int i = 0; i < n_result; i++) {
        result[i] = hits[i].index;
    }

    free(visited);
    free(hits);
    return n_result;
}

int hnsw_size(const HNSW *h) {
    return h->n_nodes;
}