#ifndef DSTACK_H
#define DSTACK_H

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    double *data;
    size_t size;
    size_t capacity;
} DStack;

void ds_init(DStack *s);
void ds_free(DStack *s);
bool ds_push(DStack *s, double v);
bool ds_pop(DStack *s, double *out);
bool ds_peek(const DStack *s, double *out);
bool ds_empty(const DStack *s);
size_t ds_size(const DStack *s);

#endif