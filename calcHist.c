#include <stdio.h>

void calcHist(int *pixels, int rows, int *histo, int sz)
{
    for(int i = 0; i < rows; ++i){
        int r = pixels[i*3+0];
        int g = pixels[i*3+1];
        int b = pixels[i*3+2];
        *(histo + r*sz*sz + g*sz + b) += 1 ;
    }
}