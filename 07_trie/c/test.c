#include "trie.h"
#include "../../common/c_utils/ds_test.h"
#include <stdio.h>
#include <string.h>

DS_TEST(test_init_empty) {
    Trie t;
    trie_init(&t);
    DS_ASSERT(trie_size(&t) == 0, "init size 0");
    DS_ASSERT(trie_node_count(&t) == 1, "init 1 node (root)");
    DS_ASSERT(!trie_search(&t, "a"), "search empty trie");
    DS_ASSERT(!trie_search(&t, "hello"), "search empty trie 2");
    DS_ASSERT(!trie_starts_with(&t, "a"), "starts_with empty trie");
    trie_free(&t);
}

DS_TEST(test_insert_search_basic) {
    Trie t;
    trie_init(&t);
    DS_ASSERT(trie_insert(&t, "apple"), "insert apple");
    DS_ASSERT(trie_insert(&t, "app"), "insert app");
    DS_ASSERT(trie_insert(&t, "banana"), "insert banana");
    DS_ASSERT(trie_size(&t) == 3, "size 3");
    DS_ASSERT(trie_search(&t, "apple"), "search apple");
    DS_ASSERT(trie_search(&t, "app"), "search app");
    DS_ASSERT(trie_search(&t, "banana"), "search banana");
    DS_ASSERT(!trie_search(&t, "ap"), "ap not a word");
    DS_ASSERT(!trie_search(&t, "appl"), "appl not a word");
    DS_ASSERT(!trie_search(&t, "ban"), "ban not a word");
    DS_ASSERT(!trie_search(&t, "bananaextra"), "bananaextra not a word");
    trie_free(&t);
}

DS_TEST(test_starts_with) {
    Trie t;
    trie_init(&t);
    trie_insert(&t, "apple");
    trie_insert(&t, "app");
    trie_insert(&t, "application");
    trie_insert(&t, "banana");
    trie_insert(&t, "band");
    DS_ASSERT(trie_starts_with(&t, "app"), "prefix app");
    DS_ASSERT(trie_starts_with(&t, "appl"), "prefix appl");
    DS_ASSERT(trie_starts_with(&t, "apple"), "prefix apple (full word is prefix)");
    DS_ASSERT(trie_starts_with(&t, "ban"), "prefix ban");
    DS_ASSERT(trie_starts_with(&t, "band"), "prefix band");
    DS_ASSERT(trie_starts_with(&t, "bana"), "bana is prefix of banana");
    DS_ASSERT(!trie_starts_with(&t, "bann"), "bann not a prefix");
    DS_ASSERT(!trie_starts_with(&t, "cat"), "cat not a prefix");
    DS_ASSERT(!trie_starts_with(&t, "applex"), "applex not a prefix");
    trie_free(&t);
}

DS_TEST(test_empty_string) {
    Trie t;
    trie_init(&t);
    DS_ASSERT(!trie_insert(&t, ""), "insert empty rejected");
    DS_ASSERT(trie_size(&t) == 0, "size still 0");
    DS_ASSERT(!trie_search(&t, ""), "search empty is false");
    DS_ASSERT(trie_starts_with(&t, ""), "empty is prefix of anything");
    DS_ASSERT(trie_node_count(&t) == 1, "only root node");
    trie_free(&t);
}

DS_TEST(test_duplicate_insert) {
    Trie t;
    trie_init(&t);
    trie_insert(&t, "hello");
    DS_ASSERT(trie_size(&t) == 1, "size 1");
    size_t nodes1 = trie_node_count(&t);
    DS_ASSERT(trie_insert(&t, "hello"), "insert hello again");
    DS_ASSERT(trie_size(&t) == 1, "size still 1 after dup");
    DS_ASSERT(trie_node_count(&t) == nodes1, "nodes unchanged after dup");
    trie_free(&t);
}

DS_TEST(test_invalid_char) {
    Trie t;
    trie_init(&t);
    DS_ASSERT(!trie_insert(&t, "Hello"), "uppercase rejected");
    DS_ASSERT(!trie_insert(&t, "ab1"), "digit rejected");
    DS_ASSERT(!trie_insert(&t, "ab c"), "space rejected");
    DS_ASSERT(trie_size(&t) == 0, "size 0 after all rejected");
    DS_ASSERT(!trie_search(&t, "Hello"), "search uppercase false");
    trie_free(&t);
}

DS_TEST(test_longest_prefix) {
    Trie t;
    trie_init(&t);
    trie_insert(&t, "a");
    trie_insert(&t, "app");
    trie_insert(&t, "apple");
    trie_insert(&t, "the");
    trie_insert(&t, "there");
    char buf[64];
    size_t len;
    len = trie_longest_prefix(&t, "applesauce", buf, sizeof(buf));
    DS_ASSERT(len == 5, "longest prefix of applesauce is apple(5)");
    DS_ASSERT(strcmp(buf, "apple") == 0, "buf is apple");
    len = trie_longest_prefix(&t, "application", buf, sizeof(buf));
    DS_ASSERT(len == 3, "longest prefix of application is app(3)");
    DS_ASSERT(strcmp(buf, "app") == 0, "buf is app");
    len = trie_longest_prefix(&t, "therefore", buf, sizeof(buf));
    DS_ASSERT(len == 5, "longest prefix of therefore is there(5)");
    DS_ASSERT(strcmp(buf, "there") == 0, "buf is there");
    len = trie_longest_prefix(&t, "xyz", buf, sizeof(buf));
    DS_ASSERT(len == 0, "no prefix for xyz");
    len = trie_longest_prefix(&t, "a", buf, sizeof(buf));
    DS_ASSERT(len == 1, "single char a");
    DS_ASSERT(strcmp(buf, "a") == 0, "buf is a");
    len = trie_longest_prefix(&t, "th", buf, sizeof(buf));
    DS_ASSERT(len == 0, "th not a word, no prefix");
    trie_free(&t);
}

DS_TEST(test_shared_prefix_memory) {
    Trie t;
    trie_init(&t);
    const char *words[] = {
        "a", "ab", "abc", "abcd", "abcde",
        "b", "bc", "bcd", "bcde", "bcdef"
    };
    size_t n = sizeof(words) / sizeof(words[0]);
    for (size_t i = 0; i < n; i++) {
        trie_insert(&t, words[i]);
    }
    DS_ASSERT(trie_size(&t) == n, "all words inserted");
    size_t total_chars = 0;
    for (size_t i = 0; i < n; i++) {
        total_chars += strlen(words[i]);
    }
    size_t nodes = trie_node_count(&t);
    DS_ASSERT(nodes < total_chars, "shared prefix: nodes < total chars");
    DS_ASSERT(nodes == 11, "11 nodes for these shared prefixes");
    trie_free(&t);
}

DS_TEST(test_no_false_positive) {
    Trie t;
    trie_init(&t);
    trie_insert(&t, "cat");
    trie_insert(&t, "car");
    trie_insert(&t, "care");
    DS_ASSERT(trie_search(&t, "cat"), "cat");
    DS_ASSERT(trie_search(&t, "car"), "car");
    DS_ASSERT(trie_search(&t, "care"), "care");
    DS_ASSERT(!trie_search(&t, "ca"), "ca not word");
    DS_ASSERT(!trie_search(&t, "cars"), "cars not word");
    DS_ASSERT(!trie_search(&t, "careful"), "careful not word");
    DS_ASSERT(!trie_search(&t, "c"), "c not word");
    DS_ASSERT(trie_starts_with(&t, "ca"), "ca is prefix");
    DS_ASSERT(trie_starts_with(&t, "car"), "car is prefix");
    DS_ASSERT(trie_starts_with(&t, "care"), "care is prefix");
    DS_ASSERT(!trie_starts_with(&t, "careful"), "careful not prefix");
    trie_free(&t);
}

DS_TEST(test_stress_insert_search) {
    Trie t;
    trie_init(&t);
    const int N = 5000;
    char buf[16];
    for (int i = 0; i < N; i++) {
        int len = 3 + (i % 10);
        int v = i;
        for (int j = 0; j < len; j++) {
            buf[j] = 'a' + (v % 26);
            v = v * 31 + 7;
            v = v < 0 ? -v : v;
            v %= 26;
        }
        buf[len] = '\0';
        trie_insert(&t, buf);
    }
    DS_ASSERT(trie_size(&t) > 0, "inserted some words");
    DS_ASSERT(trie_size(&t) <= (size_t)N, "size <= N (dups merged)");
    int found = 0;
    for (int i = 0; i < N; i++) {
        int len = 3 + (i % 10);
        int v = i;
        for (int j = 0; j < len; j++) {
            buf[j] = 'a' + (v % 26);
            v = v * 31 + 7;
            v = v < 0 ? -v : v;
            v %= 26;
        }
        buf[len] = '\0';
        if (trie_search(&t, buf)) {
            found++;
        }
    }
    DS_ASSERT(found == N, "all generated strings found");
    trie_free(&t);
}

DS_TEST(test_longest_prefix_bpe_like) {
    Trie t;
    trie_init(&t);
    const char *vocab[] = {
        "t", "h", "e", "th", "he", "the", "re", "the", "n", "d",
        "and", "ing", "ion", "tion", "ation", "i", "o", "r", "a", "s"
    };
    size_t nv = sizeof(vocab) / sizeof(vocab[0]);
    for (size_t i = 0; i < nv; i++) {
        trie_insert(&t, vocab[i]);
    }
    char buf[64];
    size_t len = trie_longest_prefix(&t, "thenation", buf, sizeof(buf));
    DS_ASSERT(len == 3, "longest prefix of thenation is the(3)");
    DS_ASSERT(strcmp(buf, "the") == 0, "buf is the");
    len = trie_longest_prefix(&t, "ationing", buf, sizeof(buf));
    DS_ASSERT(len == 5, "longest prefix of ationing is ation(5)");
    DS_ASSERT(strcmp(buf, "ation") == 0, "buf is ation");
    trie_free(&t);
}

int main(void) {
    DS_RUN(test_init_empty);
    DS_RUN(test_insert_search_basic);
    DS_RUN(test_starts_with);
    DS_RUN(test_empty_string);
    DS_RUN(test_duplicate_insert);
    DS_RUN(test_invalid_char);
    DS_RUN(test_longest_prefix);
    DS_RUN(test_shared_prefix_memory);
    DS_RUN(test_no_false_positive);
    DS_RUN(test_stress_insert_search);
    DS_RUN(test_longest_prefix_bpe_like);

    printf("--- 性能微基准 ---\n");
    Trie t;
    trie_init(&t);
    char w[12];
    for (int i = 0; i < 100000; i++) {
        int v = i;
        for (int j = 0; j < 8; j++) {
            w[j] = 'a' + (v % 26);
            v /= 26;
        }
        w[8] = '\0';
        trie_insert(&t, w);
    }
    char buf2[64];
    DS_BENCH(trie_search(&t, "abcdefgh"), 1000000);
    DS_BENCH(trie_starts_with(&t, "abcdefgh"), 1000000);
    DS_BENCH(trie_longest_prefix(&t, "abcdefghij", buf2, sizeof(buf2)), 1000000);
    printf("  trie size = %zu, nodes = %zu\n", trie_size(&t), trie_node_count(&t));
    trie_free(&t);

    DS_TEST_SUMMARY();
}