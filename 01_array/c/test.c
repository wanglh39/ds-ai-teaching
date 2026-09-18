#include "dyn_array.h"
#include "../../common/c_utils/ds_test.h"
#include <stdio.h>

DS_TEST(test_init_and_free) {
    DynArray a;
    da_init(&a);
    DS_ASSERT(da_size(&a) == 0, "init size 0");
    DS_ASSERT(da_capacity(&a) == 0, "init cap 0");
    da_free(&a);
}

DS_TEST(test_append_grow) {
    DynArray a;
    da_init(&a);
    for (int i = 0; i < 100; i++) {
        DS_ASSERT(da_append(&a, i), "append ok");
    }
    DS_ASSERT(da_size(&a) == 100, "size 100");
    DS_ASSERT(da_capacity(&a) >= 100, "cap >= 100");
    for (int i = 0; i < 100; i++) {
        DS_ASSERT(da_get(&a, i) == i, "value correct");
    }
    da_free(&a);
}

DS_TEST(test_pop) {
    DynArray a;
    da_init(&a);
    da_append(&a, 10);
    da_append(&a, 20);
    int v;
    DS_ASSERT(da_pop(&a, &v) && v == 20, "pop 20");
    DS_ASSERT(da_pop(&a, &v) && v == 10, "pop 10");
    DS_ASSERT(!da_pop(&a, &v), "pop empty fail");
    da_free(&a);
}

DS_TEST(test_set) {
    DynArray a;
    da_init(&a);
    da_append(&a, 1);
    da_append(&a, 2);
    DS_ASSERT(da_set(&a, 0, 99), "set ok");
    DS_ASSERT(da_get(&a, 0) == 99, "get 99");
    DS_ASSERT(!da_set(&a, 100, 0), "set out of range");
    da_free(&a);
}

int main(void) {
    DS_RUN(test_init_and_free);
    DS_RUN(test_append_grow);
    DS_RUN(test_pop);
    DS_RUN(test_set);

    printf("--- 性能微基准 ---\n");
    DynArray a;
    da_init(&a);
    DS_BENCH({ da_append(&a, 1); }, 1000000);
    da_free(&a);

    DS_TEST_SUMMARY();
}