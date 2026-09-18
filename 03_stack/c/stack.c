#include "stack.h"
#include <stdlib.h>

#define DS_INIT_CAP 16

void ds_init(DStack *s) {
    s->data = NULL;
    s->size = 0;
    s->capacity = 0;
}

void ds_free(DStack *s) {
    free(s->data);
    s->data = NULL;
    s->size = 0;
    s->capacity = 0;
}

static bool ds_grow(DStack *s) {
    size_t nc = s->capacity == 0 ? DS_INIT_CAP : s->capacity * 2;
    double *p = realloc(s->data, nc * sizeof(double));
    if (!p) return false;
    s->data = p;
    s->capacity = nc;
    return true;
}

bool ds_push(DStack *s, double v) {
    if (s->size == s->capacity && !ds_grow(s)) return false;
    s->data[s->size++] = v;
    return true;
}

bool ds_pop(DStack *s, double *out) {
    if (s->size == 0) return false;
    *out = s->data[--s->size];
    return true;
}

bool ds_peek(const DStack *s, double *out) {
    if (s->size == 0) return false;
    *out = s->data[s->size - 1];
    return true;
}

bool ds_empty(const DStack *s) { return s->size == 0; }
size_t ds_size(const DStack *s) { return s->size; }