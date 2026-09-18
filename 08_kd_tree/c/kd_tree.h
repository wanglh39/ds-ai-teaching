#ifndef DS_KD_TREE_H
#define DS_KD_TREE_H

#include <stddef.h>

#define KD_DIM 2

typedef struct KDNode {
    double point[KD_DIM];
    int split_axis;
    struct KDNode *left;
    struct KDNode *right;
} KDNode;

typedef struct {
    KDNode *root;
    size_t size;
} KDTree;

void kd_init(KDTree *t);
void kd_free(KDTree *t);
KDNode *kd_build(KDTree *t, double points[][KD_DIM], size_t n);
const KDNode *kd_nearest(const KDTree *t, const double query[KD_DIM]);
double kd_dist_sq(const double a[KD_DIM], const double b[KD_DIM]);

#endif