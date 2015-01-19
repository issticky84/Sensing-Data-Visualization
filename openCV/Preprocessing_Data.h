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

	void output_mat_as_txt_file(char file_name[],Mat);
	void output_mat_as_csv_file(char file_name[],Mat);
	void calcCovMat(Mat&, Mat&, Mat&);
	void reduceDimPCA(Mat&, int, Mat&, Mat&);
	void read_raw_data();
	float degtorad(float);
	float norm_value(float,float,float);
	float DistanceOfLontitudeAndLatitude(float,float,float,float);
	void set_hour_data(int time_title[]);
	Mat Gaussian_filter(int attribute_title[],int);
	Mat set_matrix(int attribute_title[],int);
	void voting(int,Mat,int);
	Mat Position_by_MDS(Mat,int ,float);
	Mat lab_alignment(Mat);
	Mat lab_alignment_dim1(Mat);
	Mat lab_alignment_dim2(Mat);
	void read_lab_csv();
	bool lab_boundary_test(float,float,float);
	Mat LAB2RGB(Mat);
	Mat compute_centroid(Mat);
	void mrdivide(const Mat &, const Mat &, Mat &);
public:
	Preprocessing_Data();

	void start();

	vector <int> hour_data;
	Mat histogram;//int
	Mat rgb_mat;//float
	Mat rgb_mat2;//float
	Mat position;//double
	Mat raw_data_3D;//float
};



#endif	