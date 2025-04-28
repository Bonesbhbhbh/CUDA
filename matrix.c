#include <stdio.h>

void matrixMul(int *a, int *b, int *c, int N) {
    //for each row of matrix a
    for (int i = 0; i < N; i++) {
        // for every column of matrix b
        for (int j = 0; j < N; j++) {
            // store result in c
            c[i * N + j] = 0;

            // Perform the dot product of row i of a and column j of b
            for (int k = 0; k < N; k++) {
                c[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}

int main() {
    int N = 3;
    int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int b[] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    int c[9];

    matrixMul(a, b, c, N);

    // print out result
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", c[i * N + j]);
        }
        printf("\n");
    }

    return 0;
}