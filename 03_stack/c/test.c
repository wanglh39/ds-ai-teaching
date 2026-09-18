#include "stack.h"
#include "../../common/c_utils/ds_test.h"
#include <stdio.h>

DS_TEST(test_push_pop) {
    DStack s;
    ds_init(&s);
    ds_push(&s, 1.5);
    ds_push(&s, 2.5);
    ds_push(&s, 3.5);
    DS_ASSERT(ds_size(&s) == 3, "size 3");
    double v;
    DS_ASSERT(ds_pop(&s, &v) && v == 3.5, "pop 3.5");
    DS_ASSERT(ds_pop(&s, &v) && v == 2.5, "pop 2.5");
    DS_ASSERT(ds_pop(&s, &v) && v == 1.5, "pop 1.5");
    DS_ASSERT(ds_empty(&s), "empty");
    ds_free(&s);
}

DS_TEST(test_peek) {
    DStack s;
    ds_init(&s);
    ds_push(&s, 10.0);
    double v;
    DS_ASSERT(ds_peek(&s, &v) && v == 10.0, "peek 10");
    DS_ASSERT(ds_size(&s) == 1, "peek no pop");
    ds_free(&s);
}

DS_TEST(test_grow) {
    DStack s;
    ds_init(&s);
    for (int i = 0; i < 1000; i++) {
        DS_ASSERT(ds_push(&s, i), "push ok");
    }
    DS_ASSERT(ds_size(&s) == 1000, "size 1000");
    for (int i = 999; i >= 0; i--) {
        double v;
        DS_ASSERT(ds_pop(&s, &v) && v == i, "pop order");
    }
    ds_free(&s);
}

int main(void) {
    DS_RUN(test_push_pop);
    DS_RUN(test_peek);
    DS_RUN(test_grow);

    printf("--- 性能微基准 ---\n");
    DStack s;
    ds_init(&s);
    DS_BENCH({ ds_push(&s, 1.0); ds_pop(&s, &(double){0}); }, 1000000);
    ds_free(&s);

    DS_TEST_SUMMARY();
}