#include "queue.h"
#include <stdlib.h>

void dsq_init(DQueue *q, size_t capacity) {
    if (capacity == 0) capacity = 1;
    q->capacity = capacity + 1;
    q->data = (double *)malloc(q->capacity * sizeof(double));
    q->head = 0;
    q->tail = 0;
}

void dsq_free(DQueue *q) {
    free(q->data);
    q->data = NULL;
    q->head = 0;
    q->tail = 0;
    q->capacity = 0;
}

bool dsq_enqueue(DQueue *q, double v) {
    if ((q->tail + 1) % q->capacity == q->head) return false;
    q->data[q->tail] = v;
    q->tail = (q->tail + 1) % q->capacity;
    return true;
}

bool dsq_dequeue(DQueue *q, double *out) {
    if (q->head == q->tail) return false;
    if (out) *out = q->data[q->head];
    q->head = (q->head + 1) % q->capacity;
    return true;
}

bool dsq_peek(const DQueue *q, double *out) {
    if (q->head == q->tail) return false;
    if (out) *out = q->data[q->head];
    return true;
}

bool dsq_empty(const DQueue *q) {
    return q->head == q->tail;
}

size_t dsq_size(const DQueue *q) {
    return (q->tail - q->head + q->capacity) % q->capacity;
}

size_t dsq_capacity(const DQueue *q) {
    return q->capacity > 0 ? q->capacity - 1 : 0;
}