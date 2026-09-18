#ifndef DS_HEAP_H
#define DS_HEAP_H

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    double *data;
    size_t size;
    size_t capacity;
} MinHeap;

void heap_init(MinHeap *h, size_t capacity);
void heap_free(MinHeap *h);
bool heap_push(MinHeap *h, double v);
bool heap_pop(MinHeap *h, double *out);
bool heap_peek(const MinHeap *h, double *out);
size_t heap_size(const MinHeap *h);
size_t heap_capacity(const MinHeap *h);
bool heap_empty(const MinHeap *h);

bool heap_build(MinHeap *h, const double *items, size_t n);
bool heap_replace_top(MinHeap *h, double v, double *out_old);

#endif