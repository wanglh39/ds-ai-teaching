#include "kd_tree.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void kd_init(KDTree *t) {
    t->root = NULL;
    t->size = 0;
}

static void kd_free_node(KDNode *node) {
    if (node == NULL) return;
    kd_free_node(node->left);
    kd_free_node(node->right);
    free(node);
}

void kd_free(KDTree *t) {
    kd_free_node(t->root);
    t->root = NULL;
    t->size = 0;
}

double kd_dist_sq(const double a[KD_DIM], const double b[KD_DIM]) {
    double s = 0.0;
    for (int i = 0; i < KD_DIM; i++) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

typedef struct {
    double p[KD_DIM];
} KDPoint;

static int cmp_axis(const void *pa, const void *pb, int axis) {
    const KDPoint *a = (const KDPoint *)pa;
    const KDPoint *b = (const KDPoint *)pb;
    if (a->p[axis] < b->p[axis]) return -1;
    if (a->p[axis] > b->p[axis]) return 1;
    return 0;
}

static int cmp_axis0(const void *pa, const void *pb) { return cmp_axis(pa, pb, 0); }
static int cmp_axis1(const void *pa, const void *pb) { return cmp_axis(pa, pb, 1); }

static KDNode *build_rec(KDPoint *pts, size_t n, int depth) {
    if (n == 0) return NULL;
    int axis = depth % KD_DIM;
    qsort(pts, n, sizeof(KDPoint), axis == 0 ? cmp_axis0 : cmp_axis1);
    size_t mid = n / 2;
    KDNode *node = (KDNode *)malloc(sizeof(KDNode));
    for (int i = 0; i < KD_DIM; i++) {
        node->point[i] = pts[mid].p[i];
    }
    node->split_axis = axis;
    node->left = build_rec(pts, mid, depth + 1);
    node->right = build_rec(pts + mid + 1, n - mid - 1, depth + 1);
    return node;
}

KDNode *kd_build(KDTree *t, double points[][KD_DIM], size_t n) {
    kd_free(t);
    if (n == 0) return NULL;
    KDPoint *pts = (KDPoint *)malloc(n * sizeof(KDPoint));
    for (size_t i = 0; i < n; i++) {
        for (int j = 0; j < KD_DIM; j++) {
            pts[i].p[j] = points[i][j];
        }
    }
    t->root = build_rec(pts, n, 0);
    t->size = n;
    free(pts);
    return t->root;
}

static void nearest_rec(const KDNode *node, const double query[KD_DIM],
                        const KDNode **best, double *best_dist_sq) {
    if (node == NULL) return;
    double d = kd_dist_sq(node->point, query);
    if (d < *best_dist_sq) {
        *best_dist_sq = d;
        *best = node;
    }
    int axis = node->split_axis;
    double diff = query[axis] - node->point[axis];
    const KDNode *near_child = (diff < 0.0) ? node->left : node->right;
    const KDNode *far_child = (diff < 0.0) ? node->right : node->left;
    nearest_rec(near_child, query, best, best_dist_sq);
    if (diff * diff < *best_dist_sq) {
        nearest_rec(far_child, query, best, best_dist_sq);
    }
}

const KDNode *kd_nearest(const KDTree *t, const double query[KD_DIM]) {
    if (t->root == NULL) return NULL;
    const KDNode *best = NULL;
    double best_dist_sq = INFINITY;
    nearest_rec(t->root, query, &best, &best_dist_sq);
    return best;
}