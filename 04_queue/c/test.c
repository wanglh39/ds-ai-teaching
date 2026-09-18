#include "queue.h"
#include "../../common/c_utils/ds_test.h"
#include <stdio.h>

DS_TEST(test_basic_enqueue_dequeue) {
    DQueue q;
    dsq_init(&q, 4);
    DS_ASSERT(dsq_empty(&q), "init empty");
    DS_ASSERT(dsq_size(&q) == 0, "init size 0");
    DS_ASSERT(dsq_capacity(&q) == 4, "capacity 4");
    DS_ASSERT(dsq_enqueue(&q, 1.0), "enqueue 1");
    DS_ASSERT(dsq_enqueue(&q, 2.0), "enqueue 2");
    DS_ASSERT(dsq_enqueue(&q, 3.0), "enqueue 3");
    DS_ASSERT(dsq_size(&q) == 3, "size 3");
    double v;
    DS_ASSERT(dsq_dequeue(&q, &v) && v == 1.0, "dequeue 1 (FIFO)");
    DS_ASSERT(dsq_dequeue(&q, &v) && v == 2.0, "dequeue 2");
    DS_ASSERT(dsq_dequeue(&q, &v) && v == 3.0, "dequeue 3");
    DS_ASSERT(dsq_empty(&q), "empty after all dequeue");
    dsq_free(&q);
}

DS_TEST(test_peek) {
    DQueue q;
    dsq_init(&q, 4);
    dsq_enqueue(&q, 10.0);
    dsq_enqueue(&q, 20.0);
    double v;
    DS_ASSERT(dsq_peek(&q, &v) && v == 10.0, "peek front 10");
    DS_ASSERT(dsq_size(&q) == 2, "peek no dequeue");
    dsq_dequeue(&q, &v);
    DS_ASSERT(dsq_peek(&q, &v) && v == 20.0, "peek front 20 after dequeue");
    dsq_free(&q);
}

DS_TEST(test_wrap_around) {
    DQueue q;
    dsq_init(&q, 4);
    double v;
    for (int round = 0; round < 3; round++) {
        for (int i = 0; i < 4; i++) {
            DS_ASSERT(dsq_enqueue(&q, round * 10.0 + i), "enqueue wrap");
        }
        DS_ASSERT(dsq_size(&q) == 4, "full size 4");
        DS_ASSERT(!dsq_enqueue(&q, 999.0), "full reject");
        for (int i = 0; i < 4; i++) {
            DS_ASSERT(dsq_dequeue(&q, &v) && v == round * 10.0 + i, "dequeue wrap FIFO");
        }
        DS_ASSERT(dsq_empty(&q), "empty after round");
    }
    dsq_free(&q);
}

DS_TEST(test_full_boundary) {
    DQueue q;
    dsq_init(&q, 3);
    DS_ASSERT(dsq_enqueue(&q, 1.0), "enqueue 1");
    DS_ASSERT(dsq_enqueue(&q, 2.0), "enqueue 2");
    DS_ASSERT(dsq_enqueue(&q, 3.0), "enqueue 3");
    DS_ASSERT(!dsq_enqueue(&q, 4.0), "enqueue 4 fail (full)");
    DS_ASSERT(dsq_size(&q) == 3, "size 3 full");
    double v;
    DS_ASSERT(dsq_peek(&q, &v) && v == 1.0, "peek 1");
    DS_ASSERT(dsq_dequeue(&q, &v) && v == 1.0, "dequeue 1");
    DS_ASSERT(dsq_enqueue(&q, 4.0), "enqueue 4 after dequeue (wrap)");
    DS_ASSERT(dsq_size(&q) == 3, "size 3 again");
    DS_ASSERT(!dsq_enqueue(&q, 5.0), "enqueue 5 fail (full again)");
    DS_ASSERT(dsq_dequeue(&q, &v) && v == 2.0, "dequeue 2 (FIFO after wrap)");
    DS_ASSERT(dsq_dequeue(&q, &v) && v == 3.0, "dequeue 3");
    DS_ASSERT(dsq_dequeue(&q, &v) && v == 4.0, "dequeue 4 (wrapped element)");
    DS_ASSERT(dsq_empty(&q), "empty after all");
    dsq_free(&q);
}

DS_TEST(test_empty_boundary) {
    DQueue q;
    dsq_init(&q, 4);
    double v;
    DS_ASSERT(!dsq_dequeue(&q, &v), "dequeue empty fail");
    DS_ASSERT(!dsq_peek(&q, &v), "peek empty fail");
    DS_ASSERT(dsq_empty(&q), "empty");
    DS_ASSERT(dsq_size(&q) == 0, "size 0");
    dsq_enqueue(&q, 42.0);
    DS_ASSERT(!dsq_empty(&q), "not empty after enqueue");
    dsq_dequeue(&q, &v);
    DS_ASSERT(dsq_empty(&q), "empty after dequeue");
    DS_ASSERT(!dsq_dequeue(&q, &v), "dequeue empty again fail");
    dsq_free(&q);
}

DS_TEST(test_fifo_order_stress) {
    DQueue q;
    dsq_init(&q, 8);
    double expected = 0.0;
    double counter = 0.0;
    for (int i = 0; i < 1000; i++) {
        DS_ASSERT(dsq_enqueue(&q, counter), "enqueue stress");
        counter += 1.0;
        double v;
        DS_ASSERT(dsq_dequeue(&q, &v), "dequeue stress");
        DS_ASSERT(v == expected, "FIFO order");
        expected += 1.0;
    }
    DS_ASSERT(dsq_empty(&q), "empty after stress");
    dsq_free(&q);
}

DS_TEST(test_partial_fill_wrap) {
    DQueue q;
    dsq_init(&q, 5);
    double v;
    dsq_enqueue(&q, 1.0);
    dsq_enqueue(&q, 2.0);
    dsq_enqueue(&q, 3.0);
    dsq_dequeue(&q, &v);
    dsq_dequeue(&q, &v);
    dsq_enqueue(&q, 4.0);
    dsq_enqueue(&q, 5.0);
    dsq_enqueue(&q, 6.0);
    dsq_enqueue(&q, 7.0);
    DS_ASSERT(!dsq_enqueue(&q, 8.0), "full at capacity 5");
    DS_ASSERT(dsq_size(&q) == 5, "size 5");
    DS_ASSERT(dsq_dequeue(&q, &v) && v == 3.0, "dequeue 3");
    DS_ASSERT(dsq_dequeue(&q, &v) && v == 4.0, "dequeue 4");
    DS_ASSERT(dsq_dequeue(&q, &v) && v == 5.0, "dequeue 5");
    DS_ASSERT(dsq_dequeue(&q, &v) && v == 6.0, "dequeue 6");
    DS_ASSERT(dsq_dequeue(&q, &v) && v == 7.0, "dequeue 7 (wrapped)");
    DS_ASSERT(dsq_empty(&q), "empty after all");
    dsq_free(&q);
}

int main(void) {
    DS_RUN(test_basic_enqueue_dequeue);
    DS_RUN(test_peek);
    DS_RUN(test_wrap_around);
    DS_RUN(test_full_boundary);
    DS_RUN(test_empty_boundary);
    DS_RUN(test_fifo_order_stress);
    DS_RUN(test_partial_fill_wrap);

    printf("--- 性能微基准 ---\n");
    DQueue q;
    dsq_init(&q, 1024);
    DS_BENCH({ dsq_enqueue(&q, 1.0); dsq_dequeue(&q, &(double){0}); }, 1000000);
    dsq_free(&q);

    DS_TEST_SUMMARY();
}