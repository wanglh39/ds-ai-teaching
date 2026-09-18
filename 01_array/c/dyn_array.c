#include "dyn_array.h"
#include <stdlib.h>
#include <string.h>

#define DA_INIT_CAP 8

void da_init(DynArray *a) {
    a->data = NULL;
    a->size = 0;
    a->capacity = 0;
}

void da_free(DynArray *a) {
    free(a->data);
    a->data = NULL;
    a->size = 0;
    a->capacity = 0;
}

bool da_resize(DynArray *a, size_t new_cap) {
    if (new_cap == 0) {
        free(a->data);
        a->data = NULL;
        a->size = 0;
        a->capacity = 0;
        return true;
    }
    int *p = realloc(a->data, new_cap * sizeof(int));
    if (!p) return false;
    a->data = p;
    a->capacity = new_cap;
    if (a->size > new_cap) a->size = new_cap;
    return true;
}

bool da_append(DynArray *a, int v) {
    if (a->size == a->capacity) {
        size_t new_cap = a->capacity == 0 ? DA_INIT_CAP : a->capacity * 2;
        if (!da_resize(a, new_cap)) return false;
    }
    a->data[a->size++] = v;
    return true;
}

int da_get(const DynArray *a, size_t i) {
    return a->data[i];
}

bool da_set(DynArray *a, size_t i, int v) {
    if (i >= a->size) return false;
    a->data[i] = v;
    return true;
}

bool da_pop(DynArray *a, int *out) {
    if (a->size == 0) return false;
    if (out) *out = a->data[--a->size];
    else a->size--;
    return true;
}

void da_clear(DynArray *a) {
    a->size = 0;
}

size_t da_size(const DynArray *a) { return a->size; }
size_t da_capacity(const DynArray *a) { return a->capacity; }