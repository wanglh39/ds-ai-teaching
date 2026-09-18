#include "hash_table.h"
#include "../../common/c_utils/ds_test.h"
#include <stdio.h>
#include <string.h>

DS_TEST(test_basic_put_get) {
    HashTable ht;
    ht_init(&ht, 8);
    DS_ASSERT(ht_size(&ht) == 0, "init size 0");
    DS_ASSERT(ht_capacity(&ht) == 8, "init capacity 8");

    DS_ASSERT(ht_put(&ht, "apple", 1), "put apple");
    DS_ASSERT(ht_put(&ht, "banana", 2), "put banana");
    DS_ASSERT(ht_put(&ht, "cherry", 3), "put cherry");
    DS_ASSERT(ht_size(&ht) == 3, "size 3");

    int64_t v;
    DS_ASSERT(ht_get(&ht, "apple", &v) && v == 1, "get apple=1");
    DS_ASSERT(ht_get(&ht, "banana", &v) && v == 2, "get banana=2");
    DS_ASSERT(ht_get(&ht, "cherry", &v) && v == 3, "get cherry=3");
    DS_ASSERT(!ht_get(&ht, "missing", &v), "get missing fail");

    DS_ASSERT(ht_has(&ht, "apple"), "has apple");
    DS_ASSERT(!ht_has(&ht, "missing"), "no missing");
    ht_free(&ht);
}

DS_TEST(test_update_existing_key) {
    HashTable ht;
    ht_init(&ht, 8);
    ht_put(&ht, "key", 100);
    DS_ASSERT(ht_size(&ht) == 1, "size 1 after first put");
    ht_put(&ht, "key", 200);
    DS_ASSERT(ht_size(&ht) == 1, "size still 1 after update");
    int64_t v;
    DS_ASSERT(ht_get(&ht, "key", &v) && v == 200, "value updated to 200");
    ht_free(&ht);
}

DS_TEST(test_collisions) {
    HashTable ht;
    ht_init(&ht, 4);
    DS_ASSERT(ht_put(&ht, "ab", 10), "put ab");
    DS_ASSERT(ht_put(&ht, "cd", 20), "put cd");
    DS_ASSERT(ht_put(&ht, "ef", 30), "put ef");
    DS_ASSERT(ht_put(&ht, "gh", 40), "put gh");
    DS_ASSERT(ht_put(&ht, "ij", 50), "put ij");
    DS_ASSERT(ht_put(&ht, "kl", 60), "put kl");

    int64_t v;
    DS_ASSERT(ht_get(&ht, "ab", &v) && v == 10, "get ab=10");
    DS_ASSERT(ht_get(&ht, "cd", &v) && v == 20, "get cd=20");
    DS_ASSERT(ht_get(&ht, "ef", &v) && v == 30, "get ef=30");
    DS_ASSERT(ht_get(&ht, "gh", &v) && v == 40, "get gh=40");
    DS_ASSERT(ht_get(&ht, "ij", &v) && v == 50, "get ij=50");
    DS_ASSERT(ht_get(&ht, "kl", &v) && v == 60, "get kl=60");
    DS_ASSERT(ht_size(&ht) == 6, "size 6 with collisions");
    ht_free(&ht);
}

DS_TEST(test_remove) {
    HashTable ht;
    ht_init(&ht, 8);
    ht_put(&ht, "one", 1);
    ht_put(&ht, "two", 2);
    ht_put(&ht, "three", 3);
    DS_ASSERT(ht_size(&ht) == 3, "size 3");

    DS_ASSERT(ht_remove(&ht, "two"), "remove two");
    DS_ASSERT(ht_size(&ht) == 2, "size 2 after remove");
    DS_ASSERT(!ht_has(&ht, "two"), "two gone");
    DS_ASSERT(ht_has(&ht, "one"), "one still there");
    DS_ASSERT(ht_has(&ht, "three"), "three still there");

    int64_t v;
    DS_ASSERT(ht_get(&ht, "one", &v) && v == 1, "one=1 after remove");
    DS_ASSERT(ht_get(&ht, "three", &v) && v == 3, "three=3 after remove");

    DS_ASSERT(!ht_remove(&ht, "two"), "remove two again fail");
    DS_ASSERT(!ht_remove(&ht, "missing"), "remove missing fail");
    DS_ASSERT(ht_size(&ht) == 2, "size still 2");
    ht_free(&ht);
}

DS_TEST(test_empty_boundary) {
    HashTable ht;
    ht_init(&ht, 4);
    int64_t v;
    DS_ASSERT(!ht_get(&ht, "anything", &v), "get from empty fail");
    DS_ASSERT(!ht_has(&ht, "anything"), "has on empty false");
    DS_ASSERT(!ht_remove(&ht, "anything"), "remove from empty fail");
    DS_ASSERT(ht_size(&ht) == 0, "empty size 0");
    DS_ASSERT(ht_load_factor(&ht) == 0.0, "empty load factor 0");
    ht_free(&ht);
}

DS_TEST(test_null_and_empty_key) {
    HashTable ht;
    ht_init(&ht, 8);
    DS_ASSERT(ht_put(&ht, "", 42), "put empty string key");
    int64_t v;
    DS_ASSERT(ht_get(&ht, "", &v) && v == 42, "get empty key=42");
    DS_ASSERT(ht_has(&ht, ""), "has empty key");
    DS_ASSERT(ht_remove(&ht, ""), "remove empty key");
    DS_ASSERT(!ht_has(&ht, ""), "empty key gone");
    ht_free(&ht);
}

DS_TEST(test_rehash_grows) {
    HashTable ht;
    ht_init(&ht, 4);
    size_t cap_before = ht_capacity(&ht);
    DS_ASSERT(cap_before == 4, "initial cap 4");

    char key[32];
    for (int i = 0; i < 100; i++) {
        sprintf(key, "key_%d", i);
        DS_ASSERT(ht_put(&ht, key, (int64_t)i), "put many");
    }
    DS_ASSERT(ht_size(&ht) == 100, "size 100");
    DS_ASSERT(ht_capacity(&ht) > cap_before, "capacity grew after rehash");
    DS_ASSERT(ht_load_factor(&ht) <= 0.75 + 0.01, "load factor within bound");

    for (int i = 0; i < 100; i++) {
        sprintf(key, "key_%d", i);
        int64_t v;
        DS_ASSERT(ht_get(&ht, key, &v) && v == (int64_t)i, "get after rehash");
    }
    ht_free(&ht);
}

DS_TEST(test_stress_large) {
    HashTable ht;
    ht_init(&ht, 16);
    char key[32];
    const int N = 5000;
    for (int i = 0; i < N; i++) {
        sprintf(key, "stress_%d", i);
        DS_ASSERT(ht_put(&ht, key, (int64_t)i * 7), "put stress");
    }
    DS_ASSERT(ht_size(&ht) == (size_t)N, "size N");

    for (int i = 0; i < N; i++) {
        sprintf(key, "stress_%d", i);
        int64_t v;
        DS_ASSERT(ht_get(&ht, key, &v) && v == (int64_t)i * 7, "get stress");
    }

    for (int i = 0; i < N; i += 2) {
        sprintf(key, "stress_%d", i);
        DS_ASSERT(ht_remove(&ht, key), "remove half");
    }
    DS_ASSERT(ht_size(&ht) == (size_t)(N / 2), "size N/2 after half remove");

    for (int i = 1; i < N; i += 2) {
        sprintf(key, "stress_%d", i);
        int64_t v;
        DS_ASSERT(ht_get(&ht, key, &v) && v == (int64_t)i * 7, "odd keys survive");
    }
    for (int i = 0; i < N; i += 2) {
        sprintf(key, "stress_%d", i);
        DS_ASSERT(!ht_has(&ht, key), "even keys gone");
    }
    ht_free(&ht);
}

DS_TEST(test_overwrite_after_remove) {
    HashTable ht;
    ht_init(&ht, 4);
    ht_put(&ht, "x", 1);
    ht_remove(&ht, "x");
    ht_put(&ht, "x", 2);
    int64_t v;
    DS_ASSERT(ht_get(&ht, "x", &v) && v == 2, "re-put after remove");
    DS_ASSERT(ht_size(&ht) == 1, "size 1");
    ht_free(&ht);
}

int main(void) {
    DS_RUN(test_basic_put_get);
    DS_RUN(test_update_existing_key);
    DS_RUN(test_collisions);
    DS_RUN(test_remove);
    DS_RUN(test_empty_boundary);
    DS_RUN(test_null_and_empty_key);
    DS_RUN(test_rehash_grows);
    DS_RUN(test_stress_large);
    DS_RUN(test_overwrite_after_remove);

    printf("--- 性能微基准 ---\n");
    HashTable ht;
    ht_init(&ht, 1024);
    char bk[32];
    for (int i = 0; i < 10000; i++) {
        sprintf(bk, "bench_%d", i);
        ht_put(&ht, bk, (int64_t)i);
    }
    int64_t bv;
    DS_BENCH({ ht_get(&ht, "bench_5000", &bv); }, 1000000);
    ht_free(&ht);

    DS_TEST_SUMMARY();
}