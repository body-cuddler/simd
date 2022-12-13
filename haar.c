#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <immintrin.h>

#define ROWS 16
#define COLS 16

#define NO_INLINE __attribute__((noinline))
#define ALIGNED16 __attribute__((aligned(16)))

static inline uint8_t avg(uint8_t a, uint8_t b) {
    return (uint8_t) (((uint16_t) a + (uint16_t) b + 1) / 2);
}

static inline void haar_x_scalar(uint8_t *output, const uint8_t *input) {
    for (size_t y = 0; y < ROWS; y++) {
        uint8_t tmp_input_row[COLS];
        memcpy(tmp_input_row, &input[y * COLS], COLS);

        for (size_t lim = COLS; lim > 1; lim /= 2) {
            for (size_t x = 0; x < lim; x += 2) {
                uint8_t a = tmp_input_row[x];
                uint8_t b = tmp_input_row[x + 1];
                uint8_t s = avg(a, b);
                uint8_t d = avg(a, -b);
                tmp_input_row[x / 2] = s;
                output[y * COLS + (lim + x) / 2] = d;
            }
        }

        output[y * COLS] = tmp_input_row[0];
    }
}

static inline void haar_y_scalar(uint8_t *output, const uint8_t *input) {
    for (size_t x = 0; x < COLS; x++) {
        uint8_t tmp_input_col[ROWS];
        for (size_t y = 0; y < ROWS; y++) {
            tmp_input_col[y] = input[y * COLS + x];
        }

        for (size_t lim = COLS; lim > 1; lim /= 2) {
            for (size_t y = 0; y < lim; y += 2) {
                uint8_t a = tmp_input_col[y];
                uint8_t b = tmp_input_col[y + 1];
                uint8_t s = avg(a, b);
                uint8_t d = avg(a, -b);
                tmp_input_col[y / 2] = s;
                output[(lim + y) / 2 * COLS + x] = d;
            }
        }

        output[x] = tmp_input_col[0];
    }
}

NO_INLINE static void haar_scalar(uint8_t *output, const uint8_t *input) {
    uint8_t tmp[ROWS * COLS];
    haar_x_scalar(tmp, input);
    haar_y_scalar(output, tmp);
}

static inline void haar_x_simd(uint8_t *output, const uint8_t *input) {
    // TODO Vectorize me
    __m128i mask_l = _mm_set1_epi16(256);
    __m128i mask_r = _mm_set1_epi16(16);
    __m128i mask_l_r2 = _mm_set1_epi32(0x10000);
    __m128i mask_r_r2 = _mm_set1_epi32(0x10000);
    __m128 twos = _mm_setr_ps(2,2,2,2);

    for(int i = 0; i < ROWS; i++) {
        __m128i row_0 = _mm_loadu_si128((const __m128i_u *) &input[i]);

        // Row 1
        __m64 row_1_left = _mm_setzero_si64();
        __m64 row_1_right = _mm_setzero_si64();

        _mm_maskmoveu_si128(row_0, mask_l, (char *) &row_1_left); // get only a's
        _mm_maskmoveu_si128(row_0, mask_r, (char *) &row_1_right); // get only b's

        row_1_left = _mm_adds_pi8(row_1_left, row_1_right); // (a+b)
        row_1_right = _mm_subs_pi8(row_1_left, row_1_right); // (a-b)

        __m128 row_1_left_floats = _mm_cvtpi8_ps(row_1_left); // (a+b)/2
        __m128 row_1_right_floats = _mm_cvtpi8_ps(row_1_right); // (a-b)/2

        row_1_right = _mm_cvtps_pi32(row_1_right_floats); // Convert float to int
        _mm_stream_pi(&output[i*COLS+(COLS/2)],row_1_right); // Save Right Side

        // Row
        __m128 row_2_left_floats = _mm_div_ps(row_1_left_floats, twos);

    }

    haar_x_scalar(output, input);
}

static inline void haar_y_simd(uint8_t *output, const uint8_t *input) {
    // TODO Vectorize me
    haar_y_scalar(output, input);
}

NO_INLINE static void haar_simd(uint8_t *output, const uint8_t *input) {
    uint8_t tmp[ROWS * COLS] ALIGNED16;
    haar_x_simd(tmp, input);
    haar_y_simd(output, tmp);
}

static int64_t time_diff(struct timespec start, struct timespec end) {
    struct timespec temp;
    if (end.tv_nsec - start.tv_nsec < 0) {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp.tv_sec * 1000000000 + temp.tv_nsec;
}

static void benchmark(
        void (*fn)(uint8_t *, const uint8_t *),
        uint8_t *output, const uint8_t *input, size_t iterations, const char *msg) {
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    for (size_t i = 0; i < iterations; i++) {
        fn(output, input);
    }

    clock_gettime(CLOCK_REALTIME, &end);
    double avg = (double) time_diff(start, end) / iterations;
    printf("%10s:\t %.3f ns\n", msg, avg);
}

static uint8_t *alloc_matrix() {
    return memalign(16, ROWS * COLS);
}

static void init_matrix(uint8_t *matrix) {
    for (size_t y = 0; y < ROWS; y++) {
        for (size_t x = 0; x < COLS; x++) {
            matrix[y * COLS + x] = (uint8_t) (rand() & UINT8_MAX);
        }
    }
}

static bool compare_matrix(uint8_t *expected, uint8_t *actual) {
    bool correct = true;
    for (size_t y = 0; y < ROWS; y++) {
        for (size_t x = 0; x < COLS; x++) {
            uint8_t e = expected[y * COLS + x];
            uint8_t a = actual[y * COLS + x];
            if (e != a) {
                printf(
                        "Failed at (y=%zu, x=%zu): expected=%u, actual=%u\n",
                        y, x, e, a
                );
                correct = false;
            }
        }
    }
    return correct;
}

int main() {
    uint8_t *input = alloc_matrix();
    uint8_t *output_scalar = alloc_matrix();
    uint8_t *output_simd = alloc_matrix();

    /* Check for correctness */
    for (size_t n = 0; n < 100; n++) {
        init_matrix(input);
        haar_scalar(output_scalar, input);
        haar_simd(output_simd, input);
        if (!compare_matrix(output_scalar, output_simd)) {
            break;
        }
    }

    /* Benchmark */
    init_matrix(input);
    benchmark(haar_scalar, output_scalar, input, 3000000, "scalar");
    benchmark(haar_simd, output_simd, input, 3000000, "simd");
    benchmark(haar_x_scalar, output_scalar, input, 3000000, "scalar_x");
    benchmark(haar_x_simd, output_simd, input, 3000000, "simd_x");
    benchmark(haar_y_scalar, output_scalar, input, 3000000, "scalar_y");
    benchmark(haar_y_simd, output_simd, input, 3000000, "simd_y");

    return 1;
}