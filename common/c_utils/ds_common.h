#ifndef DS_COMMON_H
#define DS_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef int64_t ds_size_t;

#define DS_OK 0
#define DS_ERR_NULL -1
#define DS_ERR_OOM -2
#define DS_ERR_RANGE -3
#define DS_ERR_EMPTY -4
#define DS_ERR_NOTFOUND -5

static inline void *ds_xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p && n) {
        fprintf(stderr, "ds_xmalloc: out of memory (n=%zu)\n", n);
        abort();
    }
    return p;
}

static inline void *ds_xrealloc(void *q, size_t n) {
    void *p = realloc(q, n);
    if (!p && n) {
        fprintf(stderr, "ds_xrealloc: out of memory (n=%zu)\n", n);
        abort();
    }
    return p;
}

#endif