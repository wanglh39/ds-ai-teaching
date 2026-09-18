#ifndef DS_TRIE_H
#define DS_TRIE_H

#include <stddef.h>
#include <stdbool.h>

#define TRIE_ALPHA_SIZE 26

typedef struct TrieNode {
    struct TrieNode *children[TRIE_ALPHA_SIZE];
    bool is_end;
} TrieNode;

typedef struct {
    TrieNode *root;
    size_t word_count;
    size_t node_count;
} Trie;

void trie_init(Trie *t);
void trie_free(Trie *t);
bool trie_insert(Trie *t, const char *word);
bool trie_search(const Trie *t, const char *word);
bool trie_starts_with(const Trie *t, const char *prefix);
size_t trie_size(const Trie *t);
size_t trie_node_count(const Trie *t);
size_t trie_longest_prefix(const Trie *t, const char *s, char *out, size_t out_cap);

#endif