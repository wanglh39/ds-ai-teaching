#include "dlist.h"
#include "../../common/c_utils/ds_test.h"
#include <stdio.h>
#include <assert.h>

DS_TEST(test_push_pop) {
    DList l;
    dl_init(&l);
    dl_push_front(&l, 1, 10);
    dl_push_front(&l, 2, 20);
    dl_push_back(&l, 3, 30);
    DS_ASSERT(dl_size(&l) == 3, "size 3");
    int k, v;
    DS_ASSERT(dl_pop_front(&l, &k, &v) && k == 2 && v == 20, "front 2,20");
    DS_ASSERT(dl_pop_back(&l, &k, &v) && k == 3 && v == 30, "back 3,30");
    DS_ASSERT(dl_pop_front(&l, &k, &v) && k == 1 && v == 10, "front 1,10");
    DS_ASSERT(dl_size(&l) == 0, "empty");
    dl_free(&l);
}

DS_TEST(test_move_to_front) {
    DList l;
    dl_init(&l);
    DListNode *a = dl_push_back(&l, 1, 10);
    DListNode *b = dl_push_back(&l, 2, 20);
    dl_push_back(&l, 3, 30);
    dl_move_to_front(&l, b);
    DS_ASSERT(l.head == b, "b is head");
    DS_ASSERT(l.head->next == a, "a is second");
    dl_free(&l);
}

DS_TEST(test_remove_middle) {
    DList l;
    dl_init(&l);
    dl_push_back(&l, 1, 10);
    DListNode *b = dl_push_back(&l, 2, 20);
    dl_push_back(&l, 3, 30);
    dl_remove(&l, b);
    DS_ASSERT(dl_size(&l) == 2, "size 2");
    DS_ASSERT(l.head->key == 1, "head 1");
    DS_ASSERT(l.tail->key == 3, "tail 3");
    dl_free(&l);
}

int main(void) {
    DS_RUN(test_push_pop);
    DS_RUN(test_move_to_front);
    DS_RUN(test_remove_middle);

    printf("--- 性能微基准 ---\n");
    DList l;
    dl_init(&l);
    DS_BENCH({ dl_push_front(&l, 1, 1); }, 1000000);
    dl_free(&l);

    DS_TEST_SUMMARY();
}