#ifndef DS_DSET_H
#define DS_DSET_H

#include <stddef.h>

typedef struct {
    int *parent;
    int *rank;
    size_t n;
    size_t n_sets;
} DSet;

void dset_init(DSet *ds, size_t n);
void dset_free(DSet *ds);
void dset_make_set(DSet *ds, int x);
int dset_find(DSet *ds, int x);
int dset_union_sets(DSet *ds, int x, int y);
int dset_connected(DSet *ds, int x, int y);
size_t dset_count_sets(DSet *ds);

#endif