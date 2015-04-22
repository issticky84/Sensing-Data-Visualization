#ifndef Preprocessing_Data_H
#define Preprocessing_Data_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <highgui.h>
#include <vector>
#include "cv.h"

#define LENGTH 1000

using namespace std; 
using namespace cv;
//using namespace tapkee;
//using namespace Eigen;


class Preprocessing_Data
{
private:
	char file_csv_data[200];
	vector < vector<float> > raw_data;
	vector < vector<float> > lab_vertices;

	void output_mat_as_csv_file(char file_name[],Mat);
	void output_mat_as_csv_file_int(char file_name[],Mat);
	void output_mat_as_csv_file_double(char file_name[],Mat);
	void calcCovMat(Mat&, Mat&, Mat&);
	void reduceDimPCA(Mat&, int, Mat&, Mat&);
	void read_raw_data();
	float degtorad(float);
	float norm_value(float,float,float);
	float DistanceOfLontitudeAndLatitude(float,float,float,float);
	void set_hour_data(int time_title[]);
	Mat Gaussian_filter(int attribute_title[]);
	Mat set_matrix(int attribute_title[],int);
	void voting(int,Mat,int);
	Mat Position_by_MDS(Mat,Mat,Mat,int);
	Mat lab_alignment(Mat);
	Mat lab_alignment_dim1(Mat);
	Mat lab_alignment_dim2(Mat);
	Mat lab_alignment_new(Mat);
	void read_lab_csv();
	bool lab_boundary_test(float,float,float);
	Mat LAB2RGB(Mat);
	Mat RGB2LAB(Mat);
	Mat compute_centroid(Mat);
	void gray2rgb(float,float& ,float& ,float&);
	Mat normalize_column(Mat);
	void sort_by_color(int, Mat&, Mat&, Mat&);
	void interpolate_distance(Mat&,int);
	void distance_by_GMM(Mat&,Mat&,Mat,int);
	void distance_by_Euclidean(Mat&,Mat,int);
	void distance_by_mdg(Mat&,Mat,Mat,Mat,vector< vector<int> >);
	void distance_by_mdg2(Mat&,Mat,Mat,Mat,vector< vector<int> >);
	void distance_by_mdg3(Mat&,Mat,Mat,Mat,vector< vector<int> >);
	void distance_by_bh(Mat&,int);
	float Log2(float); 
	double mdg(Mat,int,Mat,Mat);
	void interpolate_latlon(Mat&,int);
	void adjust_histogram(Mat,Mat,Mat);
	Mat MDS(Mat,int); 
    void Position_by_histogram(Mat&);
	Mat lab_alignment_by_cube(Mat);
	void TSP();

public:
	Preprocessing_Data();

	void start();

	vector <int> hour_data;
	Mat histogram;//int
	Mat rgb_mat;//float
	Mat rgb_mat2;//float
	Mat position;//double
	Mat raw_data_3D;//float
	Mat lab;//float
	Mat MDS_1D;
	Mat Ev_PCA1D;
	float** adjust_weight;
};



#endif	