#ifndef DQUEUE_H
#define DQUEUE_H

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    double *data;
    size_t head;
    size_t tail;
    size_t capacity;
} DQueue;

void dsq_init(DQueue *q, size_t capacity);
void dsq_free(DQueue *q);
bool dsq_enqueue(DQueue *q, double v);
bool dsq_dequeue(DQueue *q, double *out);
bool dsq_peek(const DQueue *q, double *out);
bool dsq_empty(const DQueue *q);
size_t dsq_size(const DQueue *q);
size_t dsq_capacity(const DQueue *q);

#endif