#include "heap.h"
#include "../../common/c_utils/ds_test.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static bool is_min_heap(const MinHeap *h) {
    for (size_t i = 1; i < h->size; i++) {
        size_t p = (i - 1) / 2;
        if (h->data[i] < h->data[p]) {
            return false;
        }
    }
    return true;
}

DS_TEST(test_init_empty) {
    MinHeap h;
    heap_init(&h, 8);
    DS_ASSERT(heap_size(&h) == 0, "init size 0");
    DS_ASSERT(heap_empty(&h), "init empty");
    DS_ASSERT(!heap_peek(&h, NULL), "peek empty fail");
    DS_ASSERT(!heap_pop(&h, NULL), "pop empty fail");
    heap_free(&h);
}

DS_TEST(test_push_peek_pop_single) {
    MinHeap h;
    heap_init(&h, 4);
    DS_ASSERT(heap_push(&h, 42.0), "push 42");
    DS_ASSERT(heap_size(&h) == 1, "size 1");
    double v;
    DS_ASSERT(heap_peek(&h, &v) && v == 42.0, "peek 42");
    DS_ASSERT(heap_pop(&h, &v) && v == 42.0, "pop 42");
    DS_ASSERT(heap_size(&h) == 0, "size 0 after pop");
    heap_free(&h);
}

DS_TEST(test_min_property) {
    MinHeap h;
    heap_init(&h, 16);
    double vals[] = {5.0, 3.0, 8.0, 1.0, 9.0, 2.0, 7.0, 4.0, 6.0};
    size_t n = sizeof(vals) / sizeof(vals[0]);
    for (size_t i = 0; i < n; i++) {
        heap_push(&h, vals[i]);
    }
    DS_ASSERT(heap_size(&h) == n, "size n");
    DS_ASSERT(is_min_heap(&h), "heap order after pushes");
    double top;
    heap_peek(&h, &top);
    DS_ASSERT(top == 1.0, "min is 1.0");
    heap_free(&h);
}

DS_TEST(test_pop_sorted_ascending) {
    MinHeap h;
    heap_init(&h, 32);
    double vals[] = {7.3, 1.1, 9.9, 2.2, 5.5, 4.4, 8.8, 3.3, 6.6, 0.5};
    size_t n = sizeof(vals) / sizeof(vals[0]);
    for (size_t i = 0; i < n; i++) {
        heap_push(&h, vals[i]);
    }
    double prev = -INFINITY;
    int ok = 1;
    double v;
    while (heap_pop(&h, &v)) {
        if (v < prev) ok = 0;
        prev = v;
    }
    DS_ASSERT(ok, "pops in ascending order");
    DS_ASSERT(heap_empty(&h), "empty after all pops");
    heap_free(&h);
}

DS_TEST(test_duplicates) {
    MinHeap h;
    heap_init(&h, 8);
    heap_push(&h, 2.0);
    heap_push(&h, 2.0);
    heap_push(&h, 2.0);
    heap_push(&h, 1.0);
    heap_push(&h, 2.0);
    heap_push(&h, 1.0);
    DS_ASSERT(heap_size(&h) == 6, "size 6 with dups");
    DS_ASSERT(is_min_heap(&h), "heap order with dups");
    double v;
    heap_pop(&h, &v);
    DS_ASSERT(v == 1.0, "first pop 1.0");
    heap_pop(&h, &v);
    DS_ASSERT(v == 1.0, "second pop 1.0");
    heap_pop(&h, &v);
    DS_ASSERT(v == 2.0, "third pop 2.0");
    heap_free(&h);
}

DS_TEST(test_grow_capacity) {
    MinHeap h;
    heap_init(&h, 2);
    for (int i = 0; i < 1000; i++) {
        DS_ASSERT(heap_push(&h, (double)(1000 - i)), "push grow");
    }
    DS_ASSERT(heap_size(&h) == 1000, "size 1000");
    DS_ASSERT(is_min_heap(&h), "heap order after grow");
    double v;
    heap_peek(&h, &v);
    DS_ASSERT(v == 1.0, "min is 1.0 after grow");
    double prev = -INFINITY;
    int ok = 1;
    while (heap_pop(&h, &v)) {
        if (v < prev) ok = 0;
        prev = v;
    }
    DS_ASSERT(ok, "sorted after grow");
    heap_free(&h);
}

DS_TEST(test_negative_and_float) {
    MinHeap h;
    heap_init(&h, 16);
    heap_push(&h, -3.5);
    heap_push(&h, 2.7);
    heap_push(&h, -10.2);
    heap_push(&h, 0.0);
    heap_push(&h, 1.5);
    heap_push(&h, -7.1);
    DS_ASSERT(is_min_heap(&h), "heap order with negatives");
    double v;
    heap_pop(&h, &v);
    DS_ASSERT(v == -10.2, "min is -10.2");
    heap_pop(&h, &v);
    DS_ASSERT(v == -7.1, "next -7.1");
    heap_pop(&h, &v);
    DS_ASSERT(v == -3.5, "next -3.5");
    heap_pop(&h, &v);
    DS_ASSERT(v == 0.0, "next 0.0");
    heap_pop(&h, &v);
    DS_ASSERT(v == 1.5, "next 1.5");
    heap_pop(&h, &v);
    DS_ASSERT(v == 2.7, "next 2.7");
    heap_free(&h);
}

DS_TEST(test_build_heap) {
    MinHeap h;
    heap_init(&h, 0);
    double vals[] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};
    size_t n = sizeof(vals) / sizeof(vals[0]);
    DS_ASSERT(heap_build(&h, vals, n), "build heap");
    DS_ASSERT(heap_size(&h) == n, "size n after build");
    DS_ASSERT(is_min_heap(&h), "heap order after build");
    double v;
    heap_peek(&h, &v);
    DS_ASSERT(v == 0.0, "min is 0.0 after build");
    double prev = -INFINITY;
    int ok = 1;
    while (heap_pop(&h, &v)) {
        if (v < prev) ok = 0;
        prev = v;
    }
    DS_ASSERT(ok, "sorted after build");
    heap_free(&h);
}

DS_TEST(test_replace_top) {
    MinHeap h;
    heap_init(&h, 8);
    heap_push(&h, 1.0);
    heap_push(&h, 3.0);
    heap_push(&h, 2.0);
    heap_push(&h, 5.0);
    heap_push(&h, 4.0);
    double old;
    DS_ASSERT(heap_replace_top(&h, 10.0, &old), "replace top");
    DS_ASSERT(old == 1.0, "old top was 1.0");
    DS_ASSERT(is_min_heap(&h), "heap order after replace");
    double v;
    heap_peek(&h, &v);
    DS_ASSERT(v == 2.0, "new min is 2.0");
    DS_ASSERT(heap_size(&h) == 5, "size unchanged after replace");
    heap_free(&h);
}

DS_TEST(test_topk_pattern) {
    MinHeap h;
    heap_init(&h, 0);
    const int k = 5;
    double stream[] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
    size_t n = sizeof(stream) / sizeof(stream[0]);
    for (size_t i = 0; i < n; i++) {
        if (heap_size(&h) < (size_t)k) {
            heap_push(&h, stream[i]);
        } else {
            double top;
            heap_peek(&h, &top);
            if (stream[i] > top) {
                heap_replace_top(&h, stream[i], NULL);
            }
        }
    }
    DS_ASSERT(heap_size(&h) == (size_t)k, "heap size k");
    DS_ASSERT(is_min_heap(&h), "heap order");
    double result[5];
    for (int i = 0; i < k; i++) {
        heap_pop(&h, &result[i]);
    }
    double expected[] = {7, 8, 9, 9, 9};
    int ok = 1;
    for (int i = 0; i < k; i++) {
        if (result[i] != expected[i]) ok = 0;
    }
    DS_ASSERT(ok, "top-5 = [7,8,9,9,9]");
    heap_free(&h);
}

DS_TEST(test_stress_random) {
    MinHeap h;
    heap_init(&h, 0);
    srand(12345);
    const int N = 10000;
    double *ref = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        double v = (double)rand() / RAND_MAX * 1000.0;
        ref[i] = v;
        heap_push(&h, v);
    }
    DS_ASSERT(heap_size(&h) == (size_t)N, "size N");
    DS_ASSERT(is_min_heap(&h), "heap order after random push");
    qsort(ref, N, sizeof(double), cmp_double);
    int ok = 1;
    double v;
    for (int i = 0; i < N; i++) {
        if (!heap_pop(&h, &v) || v != ref[i]) {
            ok = 0;
            break;
        }
    }
    DS_ASSERT(ok, "pops match sorted reference");
    DS_ASSERT(heap_empty(&h), "empty after stress");
    free(ref);
    heap_free(&h);
}

int main(void) {
    DS_RUN(test_init_empty);
    DS_RUN(test_push_peek_pop_single);
    DS_RUN(test_min_property);
    DS_RUN(test_pop_sorted_ascending);
    DS_RUN(test_duplicates);
    DS_RUN(test_grow_capacity);
    DS_RUN(test_negative_and_float);
    DS_RUN(test_build_heap);
    DS_RUN(test_replace_top);
    DS_RUN(test_topk_pattern);
    DS_RUN(test_stress_random);

    printf("--- 性能微基准 ---\n");
    MinHeap h;
    heap_init(&h, 0);
    srand(7777);
    for (int i = 0; i < 100000; i++) {
        heap_push(&h, (double)rand());
    }
    double v;
    DS_BENCH({ heap_pop(&h, &v); heap_push(&h, v + 1.0); }, 100000);
    heap_free(&h);

    MinHeap hb;
    heap_init(&hb, 0);
    double *arr = (double *)malloc(100000 * sizeof(double));
    for (int i = 0; i < 100000; i++) {
        arr[i] = (double)rand();
    }
    DS_BENCH({ heap_build(&hb, arr, 100000); }, 100);
    free(arr);
    heap_free(&hb);

    DS_TEST_SUMMARY();
}