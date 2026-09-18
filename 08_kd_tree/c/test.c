#include "kd_tree.h"
#include "../../common/c_utils/ds_test.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static const KDNode *brute_nearest(const KDNode *nodes, size_t n,
                                    const double query[KD_DIM]) {
    if (n == 0) return NULL;
    size_t best_i = 0;
    double best_d = kd_dist_sq(nodes[0].point, query);
    for (size_t i = 1; i < n; i++) {
        double d = kd_dist_sq(nodes[i].point, query);
        if (d < best_d) {
            best_d = d;
            best_i = i;
        }
    }
    return &nodes[best_i];
}

static double rand_d(void) {
    return (double)rand() / RAND_MAX * 20.0 - 10.0;
}

DS_TEST(test_empty) {
    KDTree t;
    kd_init(&t);
    double q[2] = {0.0, 0.0};
    DS_ASSERT(kd_nearest(&t, q) == NULL, "empty tree nearest NULL");
    DS_ASSERT(t.root == NULL, "empty root NULL");
    DS_ASSERT(t.size == 0, "empty size 0");
    kd_free(&t);
}

DS_TEST(test_single) {
    KDTree t;
    kd_init(&t);
    double pts[1][2] = {{3.0, 4.0}};
    kd_build(&t, pts, 1);
    DS_ASSERT(t.size == 1, "size 1");
    double q[2] = {0.0, 0.0};
    const KDNode *n = kd_nearest(&t, q);
    DS_ASSERT(n != NULL, "nearest not NULL");
    DS_ASSERT(n->point[0] == 3.0 && n->point[1] == 4.0, "nearest is the only point");
    double q2[2] = {100.0, 100.0};
    const KDNode *n2 = kd_nearest(&t, q2);
    DS_ASSERT(n2->point[0] == 3.0 && n2->point[1] == 4.0, "far query still only point");
    kd_free(&t);
}

DS_TEST(test_known_2d) {
    KDTree t;
    kd_init(&t);
    double pts[5][2] = {
        {2.0, 3.0},
        {5.0, 4.0},
        {9.0, 6.0},
        {4.0, 7.0},
        {8.0, 1.0},
    };
    kd_build(&t, pts, 5);
    DS_ASSERT(t.size == 5, "size 5");

    double q1[2] = {9.0, 6.0};
    const KDNode *n1 = kd_nearest(&t, q1);
    DS_ASSERT(n1->point[0] == 9.0 && n1->point[1] == 6.0, "exact match (9,6)");

    double q2[2] = {9.0, 5.9};
    const KDNode *n2 = kd_nearest(&t, q2);
    DS_ASSERT(n2->point[0] == 9.0 && n2->point[1] == 6.0, "near (9,6)");

    double q3[2] = {2.1, 3.1};
    const KDNode *n3 = kd_nearest(&t, q3);
    DS_ASSERT(n3->point[0] == 2.0 && n3->point[1] == 3.0, "near (2,3)");

    double q4[2] = {7.0, 1.5};
    const KDNode *n4 = kd_nearest(&t, q4);
    DS_ASSERT(n4->point[0] == 8.0 && n4->point[1] == 1.0, "near (8,1)");

    double q5[2] = {4.5, 6.5};
    const KDNode *n5 = kd_nearest(&t, q5);
    DS_ASSERT(n5->point[0] == 4.0 && n5->point[1] == 7.0, "near (4,7)");

    kd_free(&t);
}

DS_TEST(test_query_is_point) {
    KDTree t;
    kd_init(&t);
    double pts[6][2] = {
        {1.0, 1.0}, {2.0, 5.0}, {3.0, 2.0}, {6.0, 4.0}, {7.0, 8.0}, {9.0, 3.0},
    };
    kd_build(&t, pts, 6);
    for (int i = 0; i < 6; i++) {
        double q[2] = {pts[i][0], pts[i][1]};
        const KDNode *n = kd_nearest(&t, q);
        DS_ASSERT(n->point[0] == pts[i][0] && n->point[1] == pts[i][1],
                  "query equals a stored point -> dist 0");
        DS_ASSERT(kd_dist_sq(n->point, q) == 0.0, "dist_sq == 0");
    }
    kd_free(&t);
}

DS_TEST(test_duplicate_points) {
    KDTree t;
    kd_init(&t);
    double pts[4][2] = {
        {1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}, {5.0, 5.0},
    };
    kd_build(&t, pts, 4);
    DS_ASSERT(t.size == 4, "size 4 with dups");
    double q[2] = {1.0, 1.0};
    const KDNode *n = kd_nearest(&t, q);
    DS_ASSERT(n->point[0] == 1.0 && n->point[1] == 1.0, "nearest is dup point");
    DS_ASSERT(kd_dist_sq(n->point, q) == 0.0, "dist 0");
    double q2[2] = {6.0, 6.0};
    const KDNode *n2 = kd_nearest(&t, q2);
    DS_ASSERT(n2->point[0] == 5.0 && n2->point[1] == 5.0, "nearest (5,5)");
    kd_free(&t);
}

DS_TEST(test_dist_sq) {
    double a[2] = {0.0, 0.0};
    double b[2] = {3.0, 4.0};
    DS_ASSERT(kd_dist_sq(a, b) == 25.0, "3-4-5 triangle dist_sq 25");
    double c[2] = {1.0, 1.0};
    double d[2] = {1.0, 1.0};
    DS_ASSERT(kd_dist_sq(c, d) == 0.0, "same point dist 0");
    double e[2] = {-1.0, -2.0};
    double f[2] = {2.0, 2.0};
    DS_ASSERT(kd_dist_sq(e, f) == 25.0, "(-1,-2)-(2,2) dist_sq 25");
}

DS_TEST(test_correctness_vs_brute) {
    srand(42);
    const int N = 200;
    double (*pts)[2] = malloc(N * sizeof(double[2]));
    for (int i = 0; i < N; i++) {
        pts[i][0] = rand_d();
        pts[i][1] = rand_d();
    }
    KDTree t;
    kd_init(&t);
    kd_build(&t, pts, N);

    KDNode *flat = malloc(N * sizeof(KDNode));
    for (int i = 0; i < N; i++) {
        flat[i].point[0] = pts[i][0];
        flat[i].point[1] = pts[i][1];
    }

    int ok = 1;
    for (int trial = 0; trial < 500; trial++) {
        double q[2] = {rand_d(), rand_d()};
        const KDNode *nk = kd_nearest(&t, q);
        const KDNode *nb = brute_nearest(flat, N, q);
        double dk = kd_dist_sq(nk->point, q);
        double db = kd_dist_sq(nb->point, q);
        if (fabs(dk - db) > 1e-9) {
            ok = 0;
            break;
        }
    }
    DS_ASSERT(ok, "kd_nearest matches brute force on 500 random queries (N=200)");
    free(pts);
    free(flat);
    kd_free(&t);
}

DS_TEST(test_stress_correctness) {
    srand(12345);
    const int N = 2000;
    double (*pts)[2] = malloc(N * sizeof(double[2]));
    for (int i = 0; i < N; i++) {
        pts[i][0] = rand_d();
        pts[i][1] = rand_d();
    }
    KDTree t;
    kd_init(&t);
    kd_build(&t, pts, N);
    DS_ASSERT(t.size == (size_t)N, "size N");

    KDNode *flat = malloc(N * sizeof(KDNode));
    for (int i = 0; i < N; i++) {
        flat[i].point[0] = pts[i][0];
        flat[i].point[1] = pts[i][1];
    }

    int ok = 1;
    for (int trial = 0; trial < 1000; trial++) {
        double q[2] = {rand_d(), rand_d()};
        const KDNode *nk = kd_nearest(&t, q);
        const KDNode *nb = brute_nearest(flat, N, q);
        double dk = kd_dist_sq(nk->point, q);
        double db = kd_dist_sq(nb->point, q);
        if (fabs(dk - db) > 1e-9) {
            ok = 0;
            break;
        }
    }
    DS_ASSERT(ok, "kd_nearest matches brute force on 1000 queries (N=2000)");
    free(pts);
    free(flat);
    kd_free(&t);
}

DS_TEST(test_axis_alternation) {
    KDTree t;
    kd_init(&t);
    double pts[7][2] = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0},
        {5.0, 5.0}, {6.0, 6.0}, {7.0, 7.0},
    };
    kd_build(&t, pts, 7);
    DS_ASSERT(t.root != NULL, "root not NULL");
    DS_ASSERT(t.root->split_axis == 0, "root splits on axis 0 (x)");
    int has_axis1 = 0;
    if (t.root->left && t.root->left->split_axis == 1) has_axis1 = 1;
    if (t.root->right && t.root->right->split_axis == 1) has_axis1 = 1;
    DS_ASSERT(has_axis1, "depth 1 splits on axis 1 (y)");
    kd_free(&t);
}

int main(void) {
    DS_RUN(test_empty);
    DS_RUN(test_single);
    DS_RUN(test_known_2d);
    DS_RUN(test_query_is_point);
    DS_RUN(test_duplicate_points);
    DS_RUN(test_dist_sq);
    DS_RUN(test_correctness_vs_brute);
    DS_RUN(test_stress_correctness);
    DS_RUN(test_axis_alternation);

    printf("--- 性能微基准 ---\n");
    srand(777);
    const int N = 100000;
    double (*pts)[2] = malloc(N * sizeof(double[2]));
    for (int i = 0; i < N; i++) {
        pts[i][0] = rand_d();
        pts[i][1] = rand_d();
    }
    KDTree t;
    kd_init(&t);
    double t0 = (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    kd_build(&t, pts, N);
    double t1 = (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    printf("[BENCH] kd_build N=%d: %.3f ms\n", N, t1 - t0);

    double q[2] = {0.5, 0.5};
    DS_BENCH(kd_nearest(&t, q), 100000);

    KDNode *flat = malloc(N * sizeof(KDNode));
    for (int i = 0; i < N; i++) {
        flat[i].point[0] = pts[i][0];
        flat[i].point[1] = pts[i][1];
    }
    DS_BENCH(brute_nearest(flat, N, q), 100000);

    const KDNode *nk = kd_nearest(&t, q);
    const KDNode *nb = brute_nearest(flat, N, q);
    printf("  kd nearest  = (%.4f, %.4f)\n", nk->point[0], nk->point[1]);
    printf("  brute nearest= (%.4f, %.4f)\n", nb->point[0], nb->point[1]);
    printf("  kd dist_sq = %.6f, brute dist_sq = %.6f\n",
           kd_dist_sq(nk->point, q), kd_dist_sq(nb->point, q));

    free(pts);
    free(flat);
    kd_free(&t);

    DS_TEST_SUMMARY();
}