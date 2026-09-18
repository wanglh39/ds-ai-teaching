#ifndef DS_TEST_H
#define DS_TEST_H

#include <stdio.h>
#include <time.h>

#ifdef _WIN32
static double ds_now_ms(void) {
    return (double)clock() * 1000.0 / CLOCKS_PER_SEC;
}
#else
static double ds_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

#define DS_TEST(name) static void name(int *passed, int *failed)

#define DS_ASSERT(cond, msg) do { \
    if (cond) { (*passed)++; } \
    else { (*failed)++; printf("  [FAIL] %s:%d %s\n", __FILE__, __LINE__, msg); } \
} while (0)

#define DS_RUN(test) do { \
    int p = 0, f = 0; \
    printf("[RUN ] %s\n", #test); \
    test(&p, &f); \
    printf("[DONE] %s: %d passed, %d failed\n\n", #test, p, f); \
    g_total_passed += p; g_total_failed += f; \
} while (0)

static int g_total_passed = 0;
static int g_total_failed = 0;


#define DS_BENCH(name, iters) do { \
    double t0 = ds_now_ms(); \
    for (size_t _i = 0; _i < (iters); _i++) { name; } \
    double t1 = ds_now_ms(); \
    printf("[BENCH] %s: %.3f ms total, %.6f ms/iter (iters=%zu)\n", \
           #name, t1 - t0, (t1 - t0) / (iters), (size_t)(iters)); \
} while (0)

#define DS_TEST_SUMMARY() do { \
    printf("=== Summary: %d passed, %d failed ===\n", g_total_passed, g_total_failed); \
    return g_total_failed == 0 ? 0 : 1; \
} while (0)

#endif