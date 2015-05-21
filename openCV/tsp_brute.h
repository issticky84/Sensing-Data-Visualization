#ifndef TSP_BRUTE_H
#define TSP_BRUTE_H

# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <math.h>
# include <time.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <highgui.h>

using namespace cv;
using namespace std;

class tsp_brute{
public:
	void start(Mat,int,int,Mat&);
	void start(Mat, Mat&);
	char ch_cap ( char ch );
	int ch_eqi ( char ch1, char ch2 );
	int ch_to_digit ( char ch );
	int file_column_count ( char *input_filename );
	int file_row_count ( char *input_filename );
	void perm_next3 ( int n, int p[], int *more );
	double r8_huge ( );
	double *r8mat_data_read ( char *input_filename, int m, int n );
	void r8mat_header_read ( char *input_filename, int *m, int *n );
	void r8mat_print ( int m, int n, double a[], char *title );
	void r8mat_print_some ( int m, int n, double a[], int ilo, int jlo, int ihi, int jhi, char *title );
	int s_len_trim ( char *s );
	double s_to_r8 ( char *s, int *lchar, int *error );
	int s_to_r8vec ( char *s, int n, double rvec[] );
	int s_word_count ( char *s );
	void timestamp ( );
};

#endif