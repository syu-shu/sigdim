#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// #define MAX_CHOICE 6

int recfun();

int vertices (int max, int pw, int ph, char P[ph][pw], int aw, int ah, char A[ah][aw]){
    char *choice = (char *)malloc(sizeof(char) * ph);
    int hoge = recfun(0, 0, 0, choice, 0, max, pw, ph, P, aw, ah, A);
    free(choice);
    return hoge;
}

int recfun(int pi, int ai, int column, char* choice, int count, int max, int pw, int ph, char P[ph][pw], int aw, int ah, char A[ah][aw]){
    if (pi == ph) {
        memcpy(A[ai], choice, aw * sizeof(char));
        return 1;
    }
    int diff = 0;
    if (count < max){
        for (int i = 0; i < pw; i++) {
            if (P[pi][i] > 0 && count + (((column & (1 << i)) == 0)) <= max) {
                choice[pi] = (char)i;
                diff += recfun(pi + 1, ai + diff, column | (1 << i), choice, count + (((column & (1 << i)) == 0)), max, pw, ph, P, aw, ah, A);
            }
        }
    } else if (count == max) {
        for (int i = 0; i < pw; i++) {
            if ((column & (1 << i)) > 0 && P[pi][i] > 0) {
                choice[pi] = (char)i;
                diff += recfun(pi + 1, ai + diff, column, choice, count, max, pw, ph, P, aw, ah, A);
            }
        }
    }
    return diff;
}