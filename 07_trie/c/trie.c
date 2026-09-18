#include "trie.h"
#include <stdlib.h>
#include <string.h>

static TrieNode *node_new(void) {
    TrieNode *n = (TrieNode *)calloc(1, sizeof(TrieNode));
    return n;
}

static int char_index(char c) {
    if (c < 'a' || c > 'z') return -1;
    return c - 'a';
}

void trie_init(Trie *t) {
    t->root = node_new();
    t->word_count = 0;
    t->node_count = 1;
}

static void node_free(TrieNode *n) {
    if (!n) return;
    for (int i = 0; i < TRIE_ALPHA_SIZE; i++) {
        node_free(n->children[i]);
        n->children[i] = NULL;
    }
    free(n);
}

void trie_free(Trie *t) {
    node_free(t->root);
    t->root = NULL;
    t->word_count = 0;
    t->node_count = 0;
}

bool trie_insert(Trie *t, const char *word) {
    if (!word || word[0] == '\0') return false;
    TrieNode *cur = t->root;
    for (size_t i = 0; word[i] != '\0'; i++) {
        int idx = char_index(word[i]);
        if (idx < 0) return false;
        if (!cur->children[idx]) {
            cur->children[idx] = node_new();
            if (!cur->children[idx]) return false;
            t->node_count++;
        }
        cur = cur->children[idx];
    }
    if (!cur->is_end) {
        cur->is_end = true;
        t->word_count++;
    }
    return true;
}

bool trie_search(const Trie *t, const char *word) {
    if (!word) return false;
    TrieNode *cur = t->root;
    for (size_t i = 0; word[i] != '\0'; i++) {
        int idx = char_index(word[i]);
        if (idx < 0) return false;
        cur = cur->children[idx];
        if (!cur) return false;
    }
    return cur->is_end;
}

bool trie_starts_with(const Trie *t, const char *prefix) {
    if (!prefix) return false;
    TrieNode *cur = t->root;
    for (size_t i = 0; prefix[i] != '\0'; i++) {
        int idx = char_index(prefix[i]);
        if (idx < 0) return false;
        cur = cur->children[idx];
        if (!cur) return false;
    }
    return true;
}

size_t trie_size(const Trie *t) {
    return t->word_count;
}

size_t trie_node_count(const Trie *t) {
    return t->node_count;
}

size_t trie_longest_prefix(const Trie *t, const char *s, char *out, size_t out_cap) {
    if (!s || !t->root) return 0;
    TrieNode *cur = t->root;
    size_t last_end = 0;
    size_t i = 0;
    for (; s[i] != '\0'; i++) {
        int idx = char_index(s[i]);
        if (idx < 0) break;
        cur = cur->children[idx];
        if (!cur) break;
        if (cur->is_end) {
            last_end = i + 1;
        }
    }
    if (out && last_end > 0 && last_end < out_cap) {
        memcpy(out, s, last_end);
        out[last_end] = '\0';
    }
    return last_end;
}