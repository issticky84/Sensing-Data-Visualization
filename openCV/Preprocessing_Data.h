#ifndef Preprocessing_Data_H
#define Preprocessing_Data_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <highgui.h>
#include <vector>
#include <string>
#include "cv.h"
#include "city_info.h"
#include "tsp_brute.h"

//////boost
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/timer.hpp>
#include <boost/integer_traits.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/simple_point.hpp>
#include <boost/graph/metric_tsp_approx.hpp>
#include <boost/graph/graphviz.hpp>
//////

#define LENGTH 1000

using namespace std; 
using namespace cv;
//using namespace tapkee;
//using namespace Eigen;

enum{
	gravity_x = 0,
	gravity_y,
	gravity_z,
	linear_acc_x,
	linear_acc_y,
	linear_acc_z,
	gyro_x,
	gyro_y,
	gyro_z,
	latitude,
	longitude
};

template<typename PointType>
struct cmpPnt
{
    bool operator()(const boost::simple_point<PointType>& l,
                    const boost::simple_point<PointType>& r) const
    { return (l.x > r.x); }
};

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
	Mat Gaussian_filter(int*);
	Mat set_matrix(int*,int);
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
    void Position_by_histogram(Mat&, Mat);
	void Position_by_histogram_TSP(Mat&, Mat);
	Mat lab_alignment_by_cube(Mat);
	void TSP_for_histogram(Mat);
	void TSP_for_lab_color(Mat);
	void TSP_Start(CITY_INFO *, int, double *, string &);
	bool SplitSet(const vector<CITY_INFO> &, vector< vector<CITY_INFO> > &);
	void TSP_Helper(vector<CITY_INFO> &, double *, string &, CITY_INFO &, CITY_INFO &);
	double CalculateDistance(CITY_INFO, CITY_INFO);
	string int2str(int);

	vector<vector<CITY_INFO> > mysplitset;

	template<typename VertexListGraph, typename PointContainer,typename WeightMap, typename VertexIndexMap>
	void connectAllEuclidean(VertexListGraph& ,const PointContainer&,WeightMap ,VertexIndexMap vmap);
	//void testScalability(unsigned );
	template <typename PositionVec> void checkAdjList(PositionVec);

	void TSP_boost_for_histogram(Mat, Mat&);
	void TSP_boost_for_histogram_coarse_to_fine(Mat, Mat&);
	void TSP_boost_for_histogram_coarse_to_fine2(Mat, Mat&);
	void TSP_boost_for_histogram_coarse_to_fine3(Mat, Mat&);
	void TSP_boost_for_lab_color(Mat, Mat&);
	void TSP_boost_for_lab_color_coarse_to_fine(Mat, Mat&);
	void TSP_boost_for_lab_color_coarse_to_fine2(Mat, Mat&);
	void TSP_boost_for_lab_color_coarse_to_fine3(Mat, Mat&);
	void TSP_path_for_lab_color_coarse_to_fine(Mat, Mat&);
	void TSP_path_for_lab_color_coarse_to_fine2(Mat, Mat&);
	double TSP_boost(Mat, Mat&);
	double TSP_boost(Mat, Mat&, Mat&);
	double TSP_path(Mat, Mat&);
	void sort_by_color_by_TSP(Mat, Mat&, Mat&, Mat&);

	double db_index(Mat, Mat, Mat);
	double compute_dist(Mat,Mat,int);

	int Find_Cluster_By_Elbow_Method(Mat);

	double TSP_boost_by_EdgeWeight(Mat, Mat&);
	double TSP_boost_by_EdgeWeight(Mat, Mat&, int, int);

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
	float** adjust_weight;

	vector< vector<int> > path_index_vec;
	int path_index;

	vector<int> attribute_index;
	int time_index;

};



#endif	