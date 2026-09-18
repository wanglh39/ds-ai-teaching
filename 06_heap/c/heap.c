#include "heap.h"
#include <stdlib.h>

static size_t heap_parent(size_t i) {
    return (i - 1) / 2;
}

static size_t heap_left(size_t i) {
    return 2 * i + 1;
}

static size_t heap_right(size_t i) {
    return 2 * i + 2;
}

static void heap_sift_up(MinHeap *h, size_t i) {
    while (i > 0) {
        size_t p = heap_parent(i);
        if (h->data[i] < h->data[p]) {
            double tmp = h->data[i];
            h->data[i] = h->data[p];
            h->data[p] = tmp;
            i = p;
        } else {
            break;
        }
    }
}

static void heap_sift_down(MinHeap *h, size_t i) {
    size_t n = h->size;
    while (1) {
        size_t l = heap_left(i);
        size_t r = heap_right(i);
        size_t smallest = i;
        if (l < n && h->data[l] < h->data[smallest]) {
            smallest = l;
        }
        if (r < n && h->data[r] < h->data[smallest]) {
            smallest = r;
        }
        if (smallest == i) {
            break;
        }
        double tmp = h->data[i];
        h->data[i] = h->data[smallest];
        h->data[smallest] = tmp;
        i = smallest;
    }
}

static bool heap_reserve(MinHeap *h, size_t need) {
    if (need <= h->capacity) return true;
    size_t new_cap = h->capacity == 0 ? 8 : h->capacity;
    while (new_cap < need) {
        new_cap *= 2;
    }
    double *new_data = (double *)realloc(h->data, new_cap * sizeof(double));
    if (!new_data) return false;
    h->data = new_data;
    h->capacity = new_cap;
    return true;
}

void heap_init(MinHeap *h, size_t capacity) {
    h->size = 0;
    h->capacity = capacity;
    if (capacity > 0) {
        h->data = (double *)malloc(capacity * sizeof(double));
    } else {
        h->data = NULL;
    }
}

void heap_free(MinHeap *h) {
    free(h->data);
    h->data = NULL;
    h->size = 0;
    h->capacity = 0;
}

bool heap_push(MinHeap *h, double v) {
    if (!heap_reserve(h, h->size + 1)) return false;
    h->data[h->size] = v;
    h->size++;
    heap_sift_up(h, h->size - 1);
    return true;
}

bool heap_pop(MinHeap *h, double *out) {
    if (h->size == 0) return false;
    if (out) *out = h->data[0];
    h->size--;
    if (h->size > 0) {
        h->data[0] = h->data[h->size];
        heap_sift_down(h, 0);
    }
    return true;
}

bool heap_peek(const MinHeap *h, double *out) {
    if (h->size == 0) return false;
    if (out) *out = h->data[0];
    return true;
}

size_t heap_size(const MinHeap *h) {
    return h->size;
}

size_t heap_capacity(const MinHeap *h) {
    return h->capacity;
}

bool heap_empty(const MinHeap *h) {
    return h->size == 0;
}

bool heap_build(MinHeap *h, const double *items, size_t n) {
    if (!heap_reserve(h, n)) return false;
    for (size_t i = 0; i < n; i++) {
        h->data[i] = items[i];
    }
    h->size = n;
    if (n <= 1) return true;
    size_t last_inner = heap_parent(n - 1);
    for (size_t i = last_inner + 1; i-- > 0;) {
        heap_sift_down(h, i);
    }
    return true;
}

bool heap_replace_top(MinHeap *h, double v, double *out_old) {
    if (h->size == 0) return false;
    if (out_old) *out_old = h->data[0];
    h->data[0] = v;
    heap_sift_down(h, 0);
    return true;
}