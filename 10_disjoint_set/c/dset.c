#include "dset.h"

#include <stdlib.h>

void dset_init(DSet *ds, size_t n) {
    ds->n = n;
    ds->n_sets = n;
    ds->parent = (int *)malloc(n * sizeof(int));
    ds->rank = (int *)calloc(n, sizeof(int));
    for (size_t i = 0; i < n; i++) {
        ds->parent[i] = (int)i;
    }
}

void dset_free(DSet *ds) {
    free(ds->parent);
    free(ds->rank);
    ds->parent = NULL;
    ds->rank = NULL;
    ds->n = 0;
    ds->n_sets = 0;
}

void dset_make_set(DSet *ds, int x) {
    if (ds->parent[x] == x) return;
    ds->parent[x] = x;
    ds->rank[x] = 0;
    ds->n_sets++;
}

int dset_find(DSet *ds, int x) {
    if (ds->parent[x] != x) {
        ds->parent[x] = dset_find(ds, ds->parent[x]);
    }
    return ds->parent[x];
}

int dset_union_sets(DSet *ds, int x, int y) {
    int rx = dset_find(ds, x);
    int ry = dset_find(ds, y);
    if (rx == ry) return 0;
    if (ds->rank[rx] < ds->rank[ry]) {
        int tmp = rx;
        rx = ry;
        ry = tmp;
    }
    ds->parent[ry] = rx;
    if (ds->rank[rx] == ds->rank[ry]) {
        ds->rank[rx]++;
    }
    ds->n_sets--;
    return 1;
}

int dset_connected(DSet *ds, int x, int y) {
    return dset_find(ds, x) == dset_find(ds, y);
}

size_t dset_count_sets(DSet *ds) {
    return ds->n_sets;
}