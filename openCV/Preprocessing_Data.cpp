#include "Preprocessing_Data.h"

#include <algorithm>
#include <math.h>
#include <ctime>
#include <fstream>
#include "tapkee/tapkee.hpp"
#include "tapkee/callbacks/dummy_callbacks.hpp"
#include <opencv2/core/eigen.hpp> //cv2eigen
#include <algorithm>

using namespace tapkee;
using namespace Eigen;
using namespace boost;

extern void cuda_kmeans(Mat, int, Mat& , Mat&);

Preprocessing_Data::Preprocessing_Data()
{
	read_lab_csv();
}

void Preprocessing_Data::start()
{
	//=================Read CSV file====================//
	clock_t begin1 = clock();
	strcpy(file_csv_data,"../../../../csv_data/BigData_20141121_0723_new.csv");
	read_raw_data(); 
	clock_t end1 = clock();
	printf("Read csv elapsed time: %f\n",double(end1 - begin1) / CLOCKS_PER_SEC);
	//==================================================//
	//int attribute_title[] = {4,5,6,7,8,9,10,11,12,22,23};//(gravity_x,gravity_y,gravity_z),(linear_acc_x,linear_acc_y,linear_acc_z),(gyro_x,gyro_y,gyro_z),(latitude,longitude)
	int* attribute_title = new int[attribute_index.size()];
	for(int i=0;i<attribute_index.size();i++)
	{
		attribute_title[i] = attribute_index[i] + 1;
	}
	
	//int time_title[] = {29,30,31,32,33};//hour(30),minute(31)
	int time_title[5];
	for(int i=0;i<5;i++)
	{
		time_title[i] = time_index + i + 1;
	}	
	//============Setting matrix for K-means============//
	clock_t begin2 = clock();
	set_hour_data(time_title);
	Mat model = set_matrix(attribute_title,attribute_index.size()).clone();
	clock_t end2 = clock();
	printf("Setting matrix elapsed time: %f\n",double(end2 - begin2) / CLOCKS_PER_SEC);

	output_mat_as_csv_file("model.csv",model);
	//==============K means clustering with no speed up==================//
	/*
    int k = 24; 
    Mat cluster_tag; //Tag:0~k-1
    int attempts = 2;//應該是執行次數
	Mat cluster_centers;
	//使用k means分群
	clock_t begin3 = clock();
	kmeans(model, k, cluster_tag,TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.0001), attempts,KMEANS_PP_CENTERS,cluster_centers);
	clock_t end3 = clock();
    //TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 1),  這裡有三個參數，決定k-means何時結束，第二個參數是指迭代最大次數，第三個參數是精確度多少，第一個參數是指依照前兩個參數的哪一個為準，以範例中就是兩者都參照，以 or 的方式決定
	printf("Kmeans (K = %d) elapsed time: %f\n",k,double(end3 - begin3) / CLOCKS_PER_SEC);
	*/
	//================K means clustering with cuda=====================//
	//int k = Find_Cluster_By_Elbow_Method(model) + 12;
	int k = 16;
	Mat cluster_tag = Mat::zeros(model.rows,1,CV_32S);
	Mat cluster_centers = Mat::zeros(k,model.cols,CV_32F);

	clock_t begin3 = clock();
	cuda_kmeans(model, k, cluster_tag, cluster_centers);
	clock_t end3 = clock();
	printf("Kmeans (K = %d) elapsed time: %f\n",k,double(end3 - begin3) / CLOCKS_PER_SEC);
	//Mat cluster_tag_2,cluster_centers_2;
	//for(int i=8;i<36;i+=4)
	//{
	//	cluster_tag_2 = Mat::zeros(model.rows,1,CV_32S);
	//	cluster_centers_2 = Mat::zeros(i,model.cols,CV_32F);
	//	cuda_kmeans(i, cluster_tag_2, cluster_centers_2);

	//	cout << "cluster " << i << endl;
	//	db_index(model,cluster_centers_2,cluster_tag_2);
	//}
	output_mat_as_csv_file("cluster_center_old.csv",cluster_centers);
	//=================LAB alignment====================//
	//////////////////////////////////////////
	//lab_alignment_by_cube(cluster_centers);
	/////////////////////////////////////////
	clock_t begin5 = clock();
	//if(cluster_centers.cols>=3) rgb_mat2 = lab_alignment(cluster_centers);
	if(cluster_centers.cols>=3) rgb_mat2 = lab_alignment_new(cluster_centers);
	else if(cluster_centers.cols==1) rgb_mat2 = lab_alignment_dim1(cluster_centers);
	else if(cluster_centers.cols==2) rgb_mat2 = lab_alignment_dim2 (cluster_centers);
	clock_t end5 = clock();
	printf("\nLAB alignment elapsed time: %f\n",double(end5 - begin5) / CLOCKS_PER_SEC);

	output_mat_as_csv_file("rgb_mat_old.csv",rgb_mat2);
	//=============TSP for lab color================//
	clock_t begin8 = clock();
	//TSP_for_lab_color(cluster_centers);
	Mat lab_color_sort_index = Mat::zeros(k,1,CV_32S);
	//TSP_boost_for_lab_color(lab,lab_color_sort_index);
	//TSP_boost_for_lab_color_coarse_to_fine(lab,lab_color_sort_index);
	TSP_path_for_lab_color_coarse_to_fine2(lab,lab_color_sort_index);
	clock_t end8 = clock();
	printf("TSP for lab color elapsed time: %f\n",double(end8 - begin8) / CLOCKS_PER_SEC);
	output_mat_as_csv_file_int("lab_color_sort_index.csv",lab_color_sort_index);

	//Mat TSP_path_color_sort_index = Mat::zeros(k,1,CV_32S);
	//Mat lab_data_2D(lab, Range(0,lab_data.rows), Range(0,2) );
	//TSP_boost(lab,TSP_path_color_sort_index);
	//cout << TSP_boost_by_EdgeWeight(lab,TSP_path_color_sort_index);
	//cout << TSP_boost_by_EdgeWeight(lab,TSP_path_color_sort_index,0,5);
	//TSP_path(lab, TSP_path_color_sort_index);
	//output_mat_as_csv_file_int("TSP_path_color_sort_index.csv",TSP_path_color_sort_index);

	//====================TSP Brute======================//
	//Mat TSP_brute_sort_index;
	//tsp_brute tsp;
	//tsp.start(lab,0,5,TSP_brute_sort_index);
	//tsp.start(lab,TSP_brute_sort_index);
	//====================sort pattern by color========================//
	//sort the cluster by color and generate new cluster tag and cluster center
	clock_t begin4 = clock();
	//sort_by_color(k,rgb_mat2,cluster_centers,cluster_tag);
	sort_by_color_by_TSP(lab_color_sort_index,cluster_centers,cluster_tag,rgb_mat2);
	clock_t end4 = clock();
	printf("Sort by Color elapsed time: %f\n",double(end4 - begin4) / CLOCKS_PER_SEC);

	output_mat_as_csv_file("rgb_mat_sort.csv",rgb_mat2);
	output_mat_as_csv_file("cluster_center_sort.csv",cluster_centers);
	//=======================voting=================//
	clock_t begin7 = clock();
	voting(k,cluster_tag,model.rows); // Type: int	
	output_mat_as_csv_file_int("histogram.csv",histogram);
	clock_t end7 = clock();
	printf("Histogram voting elapsed time: %f\n",double(end7 - begin7) / CLOCKS_PER_SEC);
	//===============Position (MDS)=====================//
	/*
	clock_t begin6 = clock();
	position = Position_by_MDS(cluster_centers,model,cluster_tag,k).clone(); //Type:double
	cluster_tag.release();
	output_mat_as_csv_file_double("position.csv",position);
	clock_t end6 = clock();
	printf("MDS Position elapsed time: %f\n",double(end6 - begin6) / CLOCKS_PER_SEC);
	*/
	//===============Position (neighbor distance)=====================//
	
	Mat histo_sort_index = Mat::zeros(histogram.rows,1,CV_32S); 
	//TSP_boost_for_histogram(cluster_centers,histo_sort_index);
	TSP_boost_for_histogram_coarse_to_fine3(cluster_centers,histo_sort_index);
	output_mat_as_csv_file_int("histo_sort_index.csv",histo_sort_index);
	
	
	Mat histo_position = Mat::zeros(histogram.rows,1,CV_64F);
	Position_by_histogram_TSP(histo_position,histo_sort_index);
	position = histo_position.clone();
	
	/*
	clock_t begin9 = clock();
	Mat histo_position = Mat::zeros(histogram.rows,1,CV_64F);
	Position_by_histogram(histo_position,cluster_centers);
	position = histo_position.clone();
	clock_t end9 = clock();
	printf("Histogram neighbor distance elapsed time: %f\n",double(end9 - begin9) / CLOCKS_PER_SEC);	
	*/

	cluster_centers.release();
	model.release();

}

int Preprocessing_Data::Find_Cluster_By_Elbow_Method(Mat model)
{
	
	int fit_k;
	for(int k=4;k<40;k+=4)
	{
		cout << "Cluster: " << k << endl;
		int dim = model.cols;
		Mat cluster_tag = Mat::zeros(model.rows,1,CV_32S);
		Mat cluster_centers = Mat::zeros(k,dim,CV_32F);
		cuda_kmeans(model, k, cluster_tag, cluster_centers);
	
		class cluster{
		public:
			int num;
			//Mat cov;
		};
		vector<cluster> cluster_vec(k);

		for(int i=0;i<cluster_vec.size();i++)
		{
			//cluster_vec[i].cov = Mat::zeros(dim,dim,CV_32F);
			cluster_vec[i].num = 0;
		}

		for(int i=0;i<model.rows;i++)
		{
			int tag = cluster_tag.at<int>(i,0);
			//cluster_vec[tag].cov += ( model.row(i) - cluster_centers.row(tag) ).t() * ( model.row(i) - cluster_centers.row(tag) );
			cluster_vec[tag].num++;
		}
		/*
		for(int i=0;i<k;i++)
		{
			cluster_vec[i].cov /= cluster_vec[i].num;
		}

		Mat within_class_scatter_matrix = Mat::zeros(dim,dim,CV_32F);
		for(int i=0;i<k;i++)
		{
			within_class_scatter_matrix += ((float)cluster_vec[i].num/model.rows) * cluster_vec[i].cov;
		}
		*/

		//Scalar trace_Sw = trace(within_class_scatter_matrix);
		//cout << "trace of within class " << trace_Sw.val[0,0] << endl;

		Mat mean;
		Mat mixture_scatter_matrix;
		calcCovMat(model, mean, mixture_scatter_matrix);

		Mat between_class_scatter_matrix = Mat::zeros(dim,dim,CV_32F);
		for(int i=0;i<k;i++)
		{
			between_class_scatter_matrix += ((float)cluster_vec[i].num/model.rows) * ( cluster_centers.row(i) - mean ).t() * ( cluster_centers.row(i) - mean );
		}

		Scalar trace_Sb = trace(between_class_scatter_matrix);
		printf("trace of between class: %f\n",trace_Sb.val[0,0]);

		Scalar trace_Sm = trace(mixture_scatter_matrix);
		printf("trace of mixutre class: %f\n",trace_Sm.val[0,0]);

		cout << "variance " << trace_Sb.val[0,0] / trace_Sm.val[0,0] << endl;
		if(trace_Sb.val[0,0] / trace_Sm.val[0,0] > 0.9)
		{
			fit_k = k;
			break;
		}
	}

	return fit_k;
	//return (fit_k=fit_k<12?12:fit_k);
	
}	

double Preprocessing_Data::compute_dist(Mat m1,Mat m2,int dim)
{
	double dist = 0.0;
	for(int i=0;i<m1.rows;i++)
	{
		for(int j=0;j<dim;j++)
		{
			dist += sqrt( (m1.at<double>(i,j) - m2.at<double>(i,j) )*(m1.at<double>(i,j) - m2.at<double>(i,j) ) );
		}
	}

	return dist;
}

double Preprocessing_Data::db_index(Mat model, Mat cluster_center, Mat cluster_tag)
{
	int k = cluster_center.rows;
	int dim = cluster_center.cols;
	double* Ti = new double[k];
	for(int i=0;i<k;i++) Ti[i] = 0.0;
	
	Mat Si = Mat::zeros(k,1,CV_64F);
	for(int i=0;i<model.rows;i++)
	{
		int c = cluster_tag.at<int>(i,0);
		Ti[c]++;
		Mat dist = Mat::zeros(1,1,CV_64F);
		dist.at<double>(0,0) = compute_dist(model.row(i), cluster_center.row(c), dim);
		//add(dist.row(0),Si.row(c),Si.row(c));
		Si.at<double>(c,0) += dist.at<double>(0,0);
	}

	for(int i=0;i<k;i++)
	{
		Si.at<double>(i,0) /= Ti[i];
	}
	
	output_mat_as_csv_file_double("Si.csv",Si);

	Mat Mij = Mat::zeros(k,k,CV_64F);
	for(int i=0;i<k;i++)
	{
		for(int j=0;j<k;j++)
		{
			if(i==j) continue;
			Mij.at<double>(i,j) = compute_dist(cluster_center.row(i),cluster_center.row(j), dim);
		}
	}
	output_mat_as_csv_file_double("Mij.csv",Mij);

	Mat Rij = Mat::zeros(k,k,CV_64F);
	for(int i=0;i<k;i++)
	{
		for(int j=0;j<k;j++)
		{
			if(i==j) continue;
			Rij.at<double>(i,j) = (Si.at<double>(i,0) + Si.at<double>(j,0)) / Mij.at<double>(i,j);
		}
	}
	output_mat_as_csv_file_double("Rij.csv",Rij);

	double* Di = new double[k];
	double DB_value = 0.0;
	double min, max;
	for(int i=0;i<k;i++)
	{
		minMaxLoc(Rij.row(i), &min, &max);
		Di[i] = max;
		DB_value += Di[i];
	}

	printf("DB value: %f\n",DB_value);

	return DB_value;
}

void Preprocessing_Data::sort_by_color_by_TSP(Mat lab_color_sort_index, Mat& cluster_center, Mat& cluster_tag, Mat& rgb_mat2)
{
	Mat cluster_center_old = cluster_center.clone();
	Mat cluster_tag_old = cluster_tag.clone();
	Mat rgb_mat2_old = rgb_mat2.clone();
	int k = cluster_center.rows;

	//注意:row的複製不能用rgb_mat2.row(i) = rgb_mat2_old.row(new_tag).clone();!!!!!!!
	for(int i=0;i<k;i++)
	{
		int key = lab_color_sort_index.at<int>(i,0);
		cluster_center_old.row(key).copyTo(cluster_center.row(i));
		rgb_mat2_old.row(key).copyTo(rgb_mat2.row(i));
	}

	//更新cluster的tag
	for(int i=0;i<raw_data.size();i++)
	{
		int find;
		for(int j=0;j<k;j++)
		{
			if(lab_color_sort_index.at<int>(j,0) == cluster_tag_old.at<int>(i,0))
			{
				find = j;
				break;
			}
		}
		cluster_tag.at<int>(i,0) = find;	
	}

}

void Preprocessing_Data::sort_by_color(int k, Mat& rgb_mat2, Mat& cluster_centers, Mat& cluster_tag)
{
	class cluster_info{
	public:
		int key;
		Mat rgb;
		Mat lab;
		Mat lab_1D;
		Mat cluster_center;
		
		cluster_info(int k, Mat rgb_mat, Mat lab_mat, Mat lab_1D_mat, Mat center){
			key = k;
			rgb = rgb_mat.clone();
			lab = lab_mat.clone();
			lab_1D = lab_1D_mat.clone();
			cluster_center = center.clone();
		} 
	};

	class sort_by_rgb{
	public:
		inline bool operator() (cluster_info& c1, cluster_info& c2)
		{
			/*
			float lab_1D_c1 = c1.lab_1D.at<double>(0,0);
			float lab_1D_c2 = c2.lab_1D.at<double>(0,0);
			return (lab_1D_c1 < lab_1D_c2);
			*/
			/*
			float L1 = c1.lab.at<float>(0,0);
			float A1 = c1.lab.at<float>(0,1);
			float B1 = c1.lab.at<float>(0,2);
			float L2 = c2.lab.at<float>(0,0);
			float A2 = c2.lab.at<float>(0,1);
			float B2 = c2.lab.at<float>(0,2);
			return (A1<A2 || (A1==A2 && B1>B2) || (A1==A2 && B1==B2 && L1>L2) );
			return (L1<L2 || (L1==L2 && A1>A2) || (L1==L2 && A1==A2 && B1>B2) );
			*/
			
			float R1 = c1.rgb.at<float>(0,0);
			float G1 = c1.rgb.at<float>(0,1);
			float B1 = c1.rgb.at<float>(0,2);
			float R2 = c2.rgb.at<float>(0,0);
			float G2 = c2.rgb.at<float>(0,1);
			float B2 = c2.rgb.at<float>(0,2);
			Mat rgb_color1(1, 1, CV_32FC3);
			Mat rgb_color2(1, 1, CV_32FC3);
			Mat hsv_color1(1, 1, CV_32FC3);
			Mat hsv_color2(1, 1, CV_32FC3);
			rgb_color1.at<Vec3f>(0,0) = Vec3f(R1,G1,B1);
			rgb_color2.at<Vec3f>(0,0) = Vec3f(R2,G2,B2);
			cvtColor(rgb_color1,hsv_color1,CV_RGB2HLS);
			cvtColor(rgb_color2,hsv_color2,CV_RGB2HLS);

			float H1 = hsv_color1.at<Vec3f>(0,0).val[0];
			float H2 = hsv_color2.at<Vec3f>(0,0).val[0];
			float L1 = hsv_color1.at<Vec3f>(0,0).val[1];
			float L2 = hsv_color2.at<Vec3f>(0,0).val[1];
			float S1 = hsv_color1.at<Vec3f>(0,0).val[2];
			float S2 = hsv_color2.at<Vec3f>(0,0).val[2];
			return (H1<H2 || (H1==H2 && L1>L2) || (H1==H2 && L1==L2 && S1>S2) );
			
			//return ( c1._cluster_center_1D.at<float>(0,0) < c2._cluster_center_1D.at<float>(0,0) );
		}
	};


	Mat lab_dist = Mat::zeros(lab.rows,lab.rows,CV_64F);
	for(int i=0;i<lab_dist.rows;i++)
	{
		for(int j=0;j<lab_dist.rows;j++)
		{
			for(int t=0;t<3;t++)
			{
				lab_dist.at<double>(i,j) += abs( lab.at<float>(i,t) - lab.at<float>(j,t) );
			}
		}
	}

	//output_mat_as_csv_file_double("lab_dist.csv",lab_dist);
	Mat lab_1D = MDS(lab_dist,1).clone();
	//output_mat_as_csv_file_double("lab_1D.csv",lab_1D);

	vector< cluster_info > cluster_vec;
	for(int i=0;i<k;i++)
	{
		cluster_vec.push_back( cluster_info( i,rgb_mat2.row(i),lab.row(i),lab_1D.row(i),cluster_centers.row(i) ) );
	}

	sort(cluster_vec.begin(), cluster_vec.end(), sort_by_rgb());


	//注意:row的複製不能用rgb_mat2.row(i) = rgb_mat2_old.row(new_tag).clone();!!!!!!!
	for(int i=0;i<k;i++)
	{
		cluster_vec[i].cluster_center.copyTo(cluster_centers.row(i));
		cluster_vec[i].rgb.copyTo(rgb_mat2.row(i));
	}

	//更新cluster的tag
	Mat cluster_tag_old = cluster_tag.clone();
	for(int i=0;i<raw_data.size();i++)
	{
		int find;
		for(int j=0;j<k;j++)
		{
			if(cluster_vec[j].key == cluster_tag_old.at<int>(i,0))
			{
				find = j;
				break;
			}
		}
		cluster_tag.at<int>(i,0) = find;	
	}
}

void Preprocessing_Data::output_mat_as_csv_file(char file_name[],Mat mat)
{
	ofstream fout(file_name); 
	for(int i=0;i<mat.rows;i++)
	{
		for(int j=0;j<mat.cols;j++)
		{
			if(j!=0) fout << ",";
			fout << mat.at<float>(i,j) ;
		}
		fout << endl;
	}
	fout.close();
}            

void Preprocessing_Data::output_mat_as_csv_file_double(char file_name[],Mat mat)
{
	ofstream fout(file_name); 
	for(int i=0;i<mat.rows;i++)
	{
		for(int j=0;j<mat.cols;j++)
		{
			if(j!=0) fout << ",";
			fout << mat.at<double>(i,j) ;
		}
		fout << endl;
	}
	fout.close();
}        

void Preprocessing_Data::output_mat_as_csv_file_int(char file_name[],Mat mat)
{
	ofstream fout(file_name); 
	for(int i=0;i<mat.rows;i++)
	{
		for(int j=0;j<mat.cols;j++)
		{
			if(j!=0) fout << ",";
			fout << mat.at<int>(i,j) ;
		}
		fout << endl;
	}

	fout.close();
}   

/**
 * 從data計算平均與covariance matrix
 * http://en.wikipedia.org/wiki/Covariance_matrix#Definition
 *
 */
void Preprocessing_Data::calcCovMat(Mat& data, Mat& mean, Mat& cov){
    // 初始化
	cov = Mat::zeros(data.cols, data.cols, CV_32F);

	// 計算資料點的重心(平均)
	mean = Mat::zeros(1, data.cols, CV_32F);
	for (int i = 0; i < data.rows; i++){
		mean += data.row(i);
	}
	mean /= double(data.rows);

	// 計算covariance matrix
	for (int i = 0; i < data.rows; i++){
		cov += (data.row(i) - mean).t() * (data.row(i) - mean);
	}
	cov /= double(data.rows);
}



/**
 * 用Principal Component Analysis (PCA) 做降維
 *
 */
void Preprocessing_Data::reduceDimPCA(Mat& data, int rDim, Mat& components, Mat& result){
	// 計算covariance matrix
	Mat cov, mean;
	calcCovMat(data, mean, cov);

	// 從covariance matrix計算eigenvectors
	// http://docs.opencv.org/modules/core/doc/operations_on_arrays.html?highlight=pca#eigen
	Mat eigenVal, eigenVec;
	eigen(cov, eigenVal, eigenVec);

	// 記錄前rDim個principal components
	components = Mat(rDim, data.cols, CV_32F);
	for (int i = 0; i < rDim; i++){
		// http://docs.opencv.org/modules/core/doc/basic_structures.html?highlight=mat%20row#mat-row
		eigenVec.row(i).copyTo(components.row(i));

		// http://docs.opencv.org/modules/core/doc/operations_on_arrays.html?highlight=normalize#normalize
		normalize(components.row(i), components.row(i));
	}

	// 計算結果
	result = Mat(data.rows, rDim, CV_32F);
	for (int i = 0; i < data.rows; i++){
		for (int j = 0; j < rDim; j++){
			// 內積(投影到principal component上)
			// http://docs.opencv.org/modules/core/doc/basic_structures.html?highlight=dot#mat-dot
			result.at<float>(i, j) = (data.row(i) - mean).dot(components.row(j));
			//result:代表第i個資料點的第j個係數cj
		}
	}
}

void Preprocessing_Data::read_raw_data()
{
	vector<string> title_name;

	FILE *csv_file;
	csv_file = fopen(file_csv_data,"r");
	if(!csv_file) 
	{
		cout << "Can't open config file!" << endl;
		system("pause");
		exit(1);
	}

	char line[LENGTH];
	char *token;
	int i,j;
	i = j = 0;
	fgets(line,LENGTH,csv_file); //ignore sep=
	fgets(line,LENGTH,csv_file); //ignore title
	
	token = strtok(line,";");
	while(token!=NULL)
	{
		title_name.push_back(token);
		token = strtok(NULL,";");
	}
	for(int i=0;i<title_name.size();i++)
	{
		//cout << title_name[i] << " ";
		if(title_name[i].compare("GRAVITY X (m/s簡)")==0){ attribute_index.push_back(i); }
		else if(title_name[i].compare("GRAVITY Y (m/s簡)")==0){ attribute_index.push_back(i); }
		else if(title_name[i].compare("GRAVITY Z (m/s簡)")==0){ attribute_index.push_back(i); }
		else if(title_name[i].compare("LINEAR ACCELERATION X (m/s簡)")==0){ attribute_index.push_back(i); }
		else if(title_name[i].compare("LINEAR ACCELERATION Y (m/s簡)")==0){ attribute_index.push_back(i); }
		else if(title_name[i].compare("LINEAR ACCELERATION Z (m/s簡)")==0){ attribute_index.push_back(i); }
		else if(title_name[i].compare("GYROSCOPE X (rad/s)")==0){ attribute_index.push_back(i); }
		else if(title_name[i].compare("GYROSCOPE Y (rad/s)")==0){ attribute_index.push_back(i); }
		else if(title_name[i].compare("GYROSCOPE Z (rad/s)")==0){ attribute_index.push_back(i); }
		else if(title_name[i].compare("LOCATION Latitude : ")==0){ attribute_index.push_back(i); }
		else if(title_name[i].compare("LOCATION Longitude : ")==0){ attribute_index.push_back(i); }
		else if(title_name[i].compare("YYYY-MO-DD HH-MI-SS_SSS\n")==0){ time_index = i; }
	}

	//for(int i=0;i<attribute_index.size();i++)
	//	cout << attribute_index[i] << endl;

	cout << "time_index " << time_index << endl;

	while(!feof(csv_file))
	{
		fgets(line,LENGTH,csv_file);
		token = strtok(line,";");
		raw_data.push_back(vector<float> (1));
		//printf("%s ",token);
		while(token!=NULL)
		{
			raw_data.back().push_back(atof(token));
			//token = strtok(NULL," ;:");
			token = strtok(NULL," ;:");
		}
	}

	//for(int i=0;i<33;i++)
	//{
	//	cout << raw_data[0][i] << " ";
	//}
	//cout << endl;
	//cout << raw_data[0][1] << " " << raw_data[0][29] << " " << raw_data[0][30] << " " << raw_data[0][31] << " " << raw_data[0][32]  << " " << raw_data[0][33] << endl;
	//cout << raw_data[0][1] << " " << raw_data[0][27] << " " << raw_data[0][28] << " " << raw_data[0][29] << " " << raw_data[0][30]  << " " << raw_data[0][31] << endl;
	//cout << raw_data[0][1] << " " << raw_data[0][31] << " " << raw_data[0][32] << " " << raw_data[0][33] << " " << raw_data[0][34]  << " " << raw_data[0][35] << endl;
	//29:Year,30:Hour,31:Minute,32:second,33:millisecond

	cout << "Csv Data Size: " << raw_data.size() <<endl;
	//cout << raw_data[0].size() << endl;

	fclose(csv_file);
}

float Preprocessing_Data::degtorad(float deg)
{
	float rad = deg *3.14159265 / 180;
	return rad;
}

float Preprocessing_Data::norm_value(float v1,float v2,float v3)
{
	return sqrt(v1*v1 + v2*v2 + v3*v3);
}

float Preprocessing_Data::DistanceOfLontitudeAndLatitude(float lat1,float lat2,float lon1,float lon2)
{
	float R = 6371; //km
	float theta1 = degtorad(lat1);
	float theta2 = degtorad(lat2);
	float delta_theta = degtorad(lat2-lat1);
	float delta_lumda = degtorad(lon2-lon1);
	float a = sin(delta_theta/2) * sin(delta_theta/2) + cos(theta1) * cos(theta2) * sin(delta_lumda/2) * sin(delta_lumda/2);
	float c = 2 * atan2((double)sqrt(a),(double)sqrt(1.0-a));
	float d = R * c;

	return d;
}

float Preprocessing_Data::Log2( float n )  
{  
    // log(n)/log(2) is log2.  
    return log( n ) / log( 2.0 );  
}

void Preprocessing_Data::set_hour_data(int time_title[])
{
	int hour_index = time_title[1];
	int time_step_amount = floor(raw_data.size()/600.0);
	hour_data.resize(time_step_amount);
	int t = 0;
	for(int i=0;i<time_step_amount;i++)
	{
		hour_data[i] = raw_data[t][hour_index]; 
		t += 600;
	}	
	Mat hour_mat = Mat::zeros(time_step_amount,1,CV_32S);
	for(int i=0;i<time_step_amount;i++)
		hour_mat.at<int>(i,0) = hour_data[i];
	output_mat_as_csv_file_int("hour_data.csv",hour_mat);
	/*
	int begin_hour,end_hour,num_of_begin_hour,num_of_end_hour;
	int num_of_five_minutes = hour_data.size();

	for(int i=0;i<hour_data.size();i++)
	{
		hour_map[hour_data[i]]++;
	}

	map<int,int>::iterator it;
	int start = 0;
	int hour_num;
	for(it = hour_map.begin(); it!=hour_map.end(); ++it)
	{
		hour_num = 11;
		if(it == hour_map.begin())
		{
			begin_hour = it->first;
			num_of_begin_hour = it->second;
			hour_num = num_of_begin_hour - 1;
		}
		else if(next(it,1)==hour_map.end())
		{
			end_hour = it->first;
			num_of_end_hour = it->second;
			hour_num = num_of_end_hour - 1 - 1;
		}

		vector2 temp;
		temp.x = start;
		temp.y = start + hour_num;
		hour_range.push_back(temp);
		hour_index.push_back(it->first);

		start += (hour_num+1);
		
	}
	*/
}

Mat Preprocessing_Data::Gaussian_filter(int* attribute_title)
{
	Mat Gaussian_filter_mat(raw_data.size(),attribute_index.size(), CV_32F);

	//Apply Gaussian filter to raw data(0~8)
	for(int i=0;i<raw_data.size();i++)
	{
		for(int j=0;j<attribute_index.size();j++)
		{
			Gaussian_filter_mat.at<float>(i,j) = raw_data[i][ attribute_title[j] ];
		}
	}
	//output_mat_as_csv_file("raw_data_mat.csv",Gaussian_filter_mat);
	

	//float sum, total = 163;
	//float weight[5] = {70, 56, 25, 8, 1}; // 參考自己 + 前4個時刻的data
	//for (int i = 0; i < Gaussian_filter_mat.cols; i++){
	//	for (int j = 4; j < Gaussian_filter_mat.rows; j++){
	//		sum = 0;
	//		for (int k = 0; k < 5; k++){
	//			sum += weight[k] * Gaussian_filter_mat.at<float>(j - k, i);
	//		}
	//		Gaussian_filter_mat.at<float>(j, i) = sum / total;
	//	}
	//}
	
	//GaussianBlur( Gaussian_filter_mat, Gaussian_filter_mat, Size( 3, 3 ), 0, 0 );

	
	int MAX_KERNEL_LENGTH = 5;
	for(int j=0;j<Gaussian_filter_mat.cols;j++)
	{
		GaussianBlur( Gaussian_filter_mat.col(j), Gaussian_filter_mat.col(j), Size(MAX_KERNEL_LENGTH, 1), 0.1, 0.1);
	}
	

	//int KERNEL_LENGTH = 600;
	//for(int i=1;i<MAX_KERNEL_LENGTH;i=i+2)
    //{ 
	//	GaussianBlur( Gaussian_filter_mat, Gaussian_filter_mat, Size( KERNEL_LENGTH, 1 ), 0.5, 0.5 );
	//}
	
	//output_mat_as_csv_file("Gaussian_filter_mat.csv",Gaussian_filter_mat);
	return Gaussian_filter_mat;
}

Mat Preprocessing_Data::set_matrix(int* attribute_title,int attribute_title_size)
{
	Mat handle_mat;
	Mat handle_mat_raw;

	Mat Gaussian_filter_mat = Gaussian_filter(attribute_title).clone();
	
	Mat norm_gravity(1, raw_data.size(), CV_32F);
	Mat norm_linear_acc(1, raw_data.size(), CV_32F);
	Mat norm_gyro(1, raw_data.size(), CV_32F);
	Mat latitude_mat(1, raw_data.size(), CV_32F);
	Mat longitude_mat(1, raw_data.size(), CV_32F);
	//Compute norm
	for(int i=0;i<raw_data.size();i++)
	{
		norm_gravity.at<float>(0,i) = norm_value(Gaussian_filter_mat.at<float>(i,0),Gaussian_filter_mat.at<float>(i,1),Gaussian_filter_mat.at<float>(i,2));
		norm_linear_acc.at<float>(0,i) = norm_value(Gaussian_filter_mat.at<float>(i,3),Gaussian_filter_mat.at<float>(i,4),Gaussian_filter_mat.at<float>(i,5));
		norm_gyro.at<float>(0,i) = norm_value(Gaussian_filter_mat.at<float>(i,6),Gaussian_filter_mat.at<float>(i,7),Gaussian_filter_mat.at<float>(i,8));
		latitude_mat.at<float>(0,i) = Gaussian_filter_mat.at<float>(i,9);
		longitude_mat.at<float>(0,i) = Gaussian_filter_mat.at<float>(i,10);
	}
	
	//for(int i=0;i<raw_data.size();i++)
	//{
	//	norm_gravity.at<float>(0,i) = norm_value(raw_data[i][attribute_title[0]],raw_data[i][attribute_title[1]],raw_data[i][attribute_title[2]]);
	//	norm_linear_acc.at<float>(0,i) = norm_value(raw_data[i][attribute_title[3]],raw_data[i][attribute_title[4]],raw_data[i][attribute_title[5]]);
	//	norm_gyro.at<float>(0,i) = norm_value(raw_data[i][attribute_title[6]],raw_data[i][attribute_title[7]],raw_data[i][attribute_title[8]]);
	//}

	//handle_mat.push_back(norm_gravity);
	//handle_mat_raw.push_back(norm_gravity);
	handle_mat.push_back(norm_linear_acc);
	handle_mat_raw.push_back(norm_linear_acc);
	handle_mat.push_back(norm_gyro);
	handle_mat_raw.push_back(norm_gyro);

	//output_mat_as_csv_file("Norm_Gaussian_filter_mat.csv",handle_mat.t());


	//Compute latitude & longitude
	//int lat_index = attribute_title[9];
	//int lon_index = attribute_title[10];
	//Mat latitude_mat = Mat::zeros(1,raw_data.size(),CV_32F);
	//Mat longitude_mat = Mat::zeros(1,raw_data.size(),CV_32F);
	//for(int i=0;i<raw_data.size();i++)
	//{
	//	latitude_mat.at<float>(0,i) = raw_data[i][lat_index];
	//	longitude_mat.at<float>(0,i) = raw_data[i][lon_index];
	//}

	//output_mat_as_csv_file("latitude_mat.csv",latitude_mat.t());
	//output_mat_as_csv_file("longitude_mat.csv",longitude_mat.t());
	//interpolate_latlon(latitude_mat,200);
	//interpolate_latlon(longitude_mat,200);
	//output_mat_as_csv_file("latitude_mat_after.csv",latitude_mat.t());
	//output_mat_as_csv_file("longitude_mat_after.csv",longitude_mat.t());

	Mat first_order_distance_mat = Mat::zeros(1, raw_data.size(), CV_32F);
	for(int i=0;i<raw_data.size();i++)
	{
		if(i==0)
			first_order_distance_mat.at<float>(0,i) = 0.0;
		else
		{
			//float dist = DistanceOfLontitudeAndLatitude(raw_data[i-1][lat_index],raw_data[i][lat_index],raw_data[i-1][lon_index],raw_data[i][lon_index]);
			float dist = DistanceOfLontitudeAndLatitude(latitude_mat.at<float>(0,i-1),latitude_mat.at<float>(0,i),longitude_mat.at<float>(0,i-1),longitude_mat.at<float>(0,i));
			first_order_distance_mat.at<float>(0,i) = dist;
		}
	}

	output_mat_as_csv_file("first_order_distance_mat.csv",first_order_distance_mat.t());
	interpolate_distance(first_order_distance_mat,3000);
	output_mat_as_csv_file("first_order_distance_mat_interpolation.csv",first_order_distance_mat.t());

	Mat first_order_distance_adjust_mat(1, raw_data.size(), CV_32F);
	for(int i=0;i<raw_data.size();i++)
	{
		float d = first_order_distance_mat.at<float>(0,i);
		float d1;
		if(d==0.0 || d<1.0e-4)
			d1 = 0.0;
		else if(d>0.05)
			d1 = 0.05;
		else if(d!=0.0)	
		{
			d1 = log(d);
		}

		first_order_distance_adjust_mat.at<float>(0,i) = d1;
	}	

	//interpolate_distance(first_order_distance_adjust_mat,100);

	//output_mat_as_csv_file("first_order_distance_mat_log.csv",first_order_distance_adjust_mat.t());
	double min, max;
	minMaxLoc(first_order_distance_adjust_mat, &min, &max);
	for(int i=0;i<first_order_distance_adjust_mat.cols;i++)
	{
		if(first_order_distance_adjust_mat.at<float>(0,i) != 0)
			first_order_distance_adjust_mat.at<float>(0,i) -= min;
	}

	handle_mat.push_back(first_order_distance_adjust_mat);
	handle_mat_raw.push_back(first_order_distance_mat);

	Mat handle_mat_transpose = handle_mat.t();
	Mat handle_mat_raw_transpose = handle_mat_raw.t();
	//output_mat_as_csv_file("handle_mat_transpose.csv",handle_mat_transpose);
	//output_mat_as_csv_file("handle_mat_raw_transpose.csv",handle_mat_raw_transpose);

	Mat normalize_mat = handle_mat_transpose;
	for(int i=0;i<handle_mat_transpose.cols;i++)///////////
		//normalize_mat.col(i) = normalize_column(handle_mat_transpose.col(i)).clone();
		normalize(handle_mat_transpose.col(i),normalize_mat.col(i),0,1,NORM_MINMAX);

	//output_mat_as_csv_file("normalize_mat.csv",normalize_mat);
	//normalize_mat.col(2) = normalize_mat.col(2).mul(100.0);//enhance the weighting of distance
	

	//output_mat_as_csv_file("normalize_mat2.csv",normalize_mat);

	return normalize_mat;

}

void Preprocessing_Data::interpolate_latlon(Mat& latlon_mat,int window_size)
{
	for(int i=0;i<raw_data.size()-1;i++)
	{
		if( abs( latlon_mat.at<float>(0,i) - latlon_mat.at<float>(0,i+1) ) > 2.0)
		{
			int index = i;
			vector<float> value_in_window(window_size);
			for(int j=index-window_size/2;j<index+window_size/2;j++)
			{
				value_in_window.push_back(latlon_mat.at<float>(0,j));
			}

			sort(value_in_window.begin(),value_in_window.end(),greater<float>());
			float median = value_in_window[window_size/2];
			value_in_window.clear();

			latlon_mat.at<float>(0,index+1) = median;
		}
	}
}

void Preprocessing_Data::interpolate_distance(Mat& first_order_distance_mat,int interval)
{
	for(int i=0;i<raw_data.size()-1;i++)
	{
		int start = i;
		int cur_index = i;
		int count = 0;
		if(first_order_distance_mat.at<float>(0,cur_index) != 0.0)
		{
			//cout << "i " << i << endl;
			cur_index++;
			while( first_order_distance_mat.at<float>(0,cur_index) == 0.0)
			{
				count++;
				cur_index++;
				//cout << "count " << count << " " << "current index " << cur_index << endl;
				if(count>interval) break;
			}
			if(count>0 && count<=interval)
			{
				//cout << "start " << first_order_distance_mat.at<float>(0,start) << " end " << first_order_distance_mat.at<float>(0,cur_index) << endl;
				float interpolation = ( first_order_distance_mat.at<float>(0,start) + first_order_distance_mat.at<float>(0,cur_index) ) / (count+2);
				//cout << "interpolation " << interpolation << endl;
				for(int j=0;j<count+2;j++)
				{
					first_order_distance_mat.at<float>(0,start+j) = interpolation;
				}
				//system("pause");
			}
		}
		//i = cur_index;
	}
}

void Preprocessing_Data::voting(int k,Mat cluster_tag,int row_size)
{
	int time_step_amount = floor(row_size/600.0);
	histogram = Mat::zeros(time_step_amount,k,CV_32S);
	int t = 0;
	for(int i=0;i<time_step_amount;i++)
	{
		for(int j=0;j<600;j++)
		{
			int index = cluster_tag.at<int>(t,0);
			histogram.at<int>(i,index)++;
			t++;
		}
	}

}

void Preprocessing_Data::distance_by_Euclidean(Mat& histo_coeff, Mat cluster_centers, int k)
{
	Mat cluster_centers_distance_mat = Mat::zeros(k,k, CV_32F);
	for(int i=0;i<k;i++)
	{
		for(int j=0;j<k;j++)
		{
			if(i!=j)
			{
				cluster_centers_distance_mat.at<float>(i,j) = norm(cluster_centers.row(i),cluster_centers.row(j)); 
			}
		}
	}
	Mat wi; //1xk
	reduce(cluster_centers_distance_mat,wi,0,CV_REDUCE_SUM); //kxk->1xk
	Mat total_distance_mat; //1x1
	reduce(wi,total_distance_mat,1,CV_REDUCE_SUM);//kx1->1x1
	float total_distance_of_cluster_centers = total_distance_mat.at<float>(0,0);
	wi = wi.mul(1.0/total_distance_of_cluster_centers);

	int time_step_amount = floor(raw_data.size()/600.0);
	for(int i=0;i<time_step_amount;i++)
	{
		for(int j=0;j<time_step_amount;j++)
		{
			for(int t=0;t<k;t++)
			{
				histo_coeff.at<double>(i,j) += wi.at<float>(0,t)*abs(histogram.at<int>(i,t)-histogram.at<int>(j,t));
			}
		}
	}
}

void Preprocessing_Data::distance_by_GMM(Mat& histo_coeff, Mat& Ev, Mat cluster_centers, int k)
{
	//GMM(Gaussian Mixutre Model)
	int time_step_amount = floor(raw_data.size()/600.0);
	for(int i=0;i<time_step_amount;i++)
	{
		for(int j=0;j<k;j++)
		{
			//Ev.row(i) += adjust_weight[i][j]*cluster_centers.row(j);
			Ev.row(i) += (histogram.at<int>(i,j)/600.0)*cluster_centers.row(j);
		}
	}
	output_mat_as_csv_file("Ev.csv",Ev);	

	for(int i=0;i<time_step_amount;i++)
	{
		for(int j=0;j<time_step_amount;j++)
		{
			if(i==j) continue;
			else if(i>j)
			{
				histo_coeff.at<double>(i,j) = histo_coeff.at<double>(j,i);
			}
			else
			{
				for(int t=0;t<cluster_centers.cols;t++)
				{
					histo_coeff.at<double>(i,j) += abs(Ev.at<float>(i,t)-Ev.at<float>(j,t));
					//histo_coeff.at<double>(i,j) += sqrt( Ev.at<float>(i,t) * Ev.at<float>(j,t) );
				}
				//histo_coeff.at<double>(i,j) = sqrt( 1.0 - histo_coeff.at<double>(i,j) );
			}
		}
	}
}

double Preprocessing_Data::mdg(Mat x,int d,Mat cov,Mat mean)
{
	//cout << x << endl << cov << endl << mean << endl;

	//防止sqrt(負數)=NaN
	double temp = sqrt( pow(2.0*3.14,d) * determinant(cov) );
	Mat eigen_value,eigen_vector;
	eigen(cov,eigen_value,eigen_vector);

	double coeff;
	if(temp==0) //防止除以0
		coeff = 0;
	else 
		coeff = 1.0 / temp;

	/*
	Mat diff;
	//absdiff(diff,x,mean);
	compare(x,mean,diff,CMP_EQ);
	bool same = true;
	for(int i=0;i<diff.rows;i++)
	{
		for(int j=0;j<diff.cols;j++)
		{
			//cout << diff.at<int>(i,j) << endl; system("pause");
			if(diff.at<float>(i,j) != 1)
				same = false;
		}
	}
	//if(same==true) result = 
	*/
	Mat result = -0.5*(x-mean)*cov.inv()*(x-mean).t();
	//cout << "temp: " << temp << endl;
	//cout << "result: " << result.at<float>(0,0) << endl;
	//cout << "coeff: " << coeff << endl;
	//cout << "exp: " << exp( result.at<float>(0,0) ) << endl;
	//cout << "ans: " << coeff*exp( result.at<float>(0,0) ) << endl;
	//system("pause");

	//return exp( result.at<float>(0,0) );
	return coeff*exp( result.at<float>(0,0) );
}

void Preprocessing_Data::adjust_histogram(Mat cluster_centers,Mat cluster_tag,Mat model)
{
	int k = cluster_centers.rows;
	int dim = cluster_centers.cols;
	int raw_size = cluster_tag.rows;
	Mat mean = cluster_centers.clone();
	int five_minutes_amount = histogram.rows;
	//Mat cov = Mat::zeros(dim,dim,CV_32F);
	class cluster
	{
	public:
		Mat cov;
		int num;
	};
	vector<cluster> c_list;
	c_list.resize(k);

	for(int i=0;i<c_list.size();i++)
	{
		c_list[i].cov = Mat::zeros(dim,dim,CV_32F);
		c_list[i].num = 0;
	}

	for(int i=0;i<raw_size;i++)
	{
		int tag = cluster_tag.at<int>(i,0);
		c_list[tag].cov += ( model.row(i)-mean.row(tag) ).t() *( model.row(i)-mean.row(tag) );
		c_list[tag].num++;
	}

	for(int i=0;i<k;i++)
	{
		c_list[i].cov /= c_list[i].num;
	}
	
	//Mat new_weight = Mat::zeros(k,1,CV_32F);
	float** new_weight = new float*[five_minutes_amount];
	for(int i=0;i<five_minutes_amount;i++) new_weight[i] = new float[k];
	for(int i=0;i<five_minutes_amount;i++) 
		for(int j=0;j<k;j++)
			new_weight[i][j] = 0.0;

	float* total_weight = new float[five_minutes_amount];

	//for(int i=0;i<five_minutes_amount;i++)
	//{
	//	for(int j=0;j<k;j++)
	//	{
	//		for(int t=0;t<k;t++)
	//		{
	//			new_weight[i][j] += (float)histogram.at<int>(i,t)/600.0 * mdg(cluster_centers.row(j),dim,c_list[t].cov,mean.row(t));	
	//			total_weight[i] += new_weight[i][j];
	//		}
	//	}
	//}
	for(int i=0;i<five_minutes_amount;i++)
	{
		for(int j=i*600;j<(i+1)*600;j++)
		{
			int t = cluster_tag.at<int>(j,0);
			//float sum_of_prob = 0.0;
			//for(int t=0;t<k;t++)
			//{
			//	sum_of_prob += ((float)c_list[t].num/raw_size)*mdg(model.row(j),dim,c_list[t].cov,mean.row(t));//p(ci)*G(x|ci)			
			//}
			//for(int t=0;t<k;t++)
			//{
				//new_weight[i][t] += ( ((float)c_list[t].num/raw_size)*mdg(model.row(j),dim,c_list[t].cov,mean.row(t)) / sum_of_prob);//p(ci)*G(x|ci)
				//new_weight[i][t] +=  ((float)c_list[t].num/raw_size)*mdg(model.row(j),dim,c_list[t].cov,mean.row(t));//p(ci)*G(x|ci)
				new_weight[i][t] +=  (histogram.at<int>(i,t)/600.0)*mdg(model.row(j),dim,c_list[t].cov,mean.row(t));//p(ci)*G(x|ci)
				total_weight[i] += new_weight[i][t];
			//}
		}
	}

	adjust_weight = new float*[five_minutes_amount];
	for(int i=0;i<five_minutes_amount;i++) adjust_weight[i] = new float[k];

	for(int i=0;i<five_minutes_amount;i++)
	{
		for(int j=0;j<k;j++)
		{
			adjust_weight[i][j] = new_weight[i][j]/total_weight[i];
		}
	}

	Mat histogram_adjust = Mat::zeros(five_minutes_amount,k,CV_32S);
	for(int i=0;i<five_minutes_amount;i++)
	{
		for(int j=0;j<k;j++)
		{
			//cout << adjust_weight[i][j] << endl;
			histogram_adjust.at<int>(i,j) = (int)(adjust_weight[i][j]*600.0);
			//cout << histogram_adjust.at<int>(i,j) << endl;
			//system("pause");
		}
	}

	output_mat_as_csv_file_int("histogram_adjust.csv",histogram_adjust);
}

void Preprocessing_Data::distance_by_mdg3(Mat& histo_coeff,Mat cluster_centers,Mat model,Mat cluster_tag,vector< vector<int> > voting_result)
{
	int k = cluster_centers.rows;
	int dim = cluster_centers.cols;
	int raw_data_size = model.rows;
	int five_minutes_amount = histo_coeff.rows;

	Mat mean = cluster_centers.clone();

	class Hist{
	public:	
		Mat* cov;
		//Mat mean;
	};
	vector<Hist> hist(five_minutes_amount);
	
	/*
	for(int i=0;i<hist.size();i++)
	{
		hist[i].mean = Mat::zeros(1,dim,CV_32F);
		for(int j=0;j<k;j++)
		{
			hist[i].mean += histogram.at<int>(i,j)*cluster_centers.row(j);
		}
		hist[i].mean /= 600.0;
	}
	*/

	//allocate covariance mat
	for(int i=0;i<hist.size();i++)
	{
		hist[i].cov = new Mat[k];
		for(int j=0;j<k;j++) hist[i].cov[j] = Mat::zeros(dim,dim,CV_32F);
	}
	//covariance mat for every histogram and for every k(cluster)
	for(int i=0;i<hist.size();i++)
	{
		for(int j=i*600;j<(i+1)*600;j++)
		{
			int c = cluster_tag.at<int>(j,0);
			hist[i].cov[c] += ( model.row(j) - mean.row(c) ).t() * ( model.row(j) - mean.row(c) );	
		}

		for(int j=0;j<k;j++)
			hist[i].cov[j] /= histogram.at<int>(i,j);
	}

	double *mdg_value = new double[five_minutes_amount];
	for(int index=0;index<five_minutes_amount;index++)
	{
		mdg_value[index] = 0.0;
		
		//for(int i=0;i<k;i++)
		//{
		//	double weight = histogram.at<int>(index,i)/600.0;
		//	if(weight!=0)
		//	{
		//		mdg_value[index] += weight * mdg(cluster_centers.row(i),dim,hist[index].cov[i],mean.row(i));	
		//		cout << cluster_centers.row(i) << endl << hist[index].cov[i] << endl << mean.row(i) << endl;
		//		cout <<  mdg(cluster_centers.row(i),dim,hist[index].cov[i],mean.row(i)) << endl;
		//		system("pause");
		//	}
		//}
		
		for(int t=index*600;t<(index+1)*600;t++)
		{
			int tag = cluster_tag.at<int>(t,0);
			mdg_value[index] += mdg(model.row(t),dim,hist[index].cov[tag],mean.row(tag));
		}
		
		cout << "GMM: " << mdg_value[index]  << endl;
	}

	//association
	float mdg_i,mdg_j;
	Mat mdg_association = Mat::zeros(five_minutes_amount,five_minutes_amount,CV_32F);
	for(int i=0;i<five_minutes_amount;i++)
	{
		for(int j=0;j<five_minutes_amount;j++)
		{
			if(i==j) continue;
			else if(i>j)//speed up to avoid duplicate computation
			{
				histo_coeff.at<double>(i,j)  = histo_coeff.at<double>(j,i);
			}
			else
			{
				mdg_i = mdg_j = 0.0;
				for(int t=0;t<k;t++)
				{
					int votes_i = histogram.at<int>(i,t);
					int votes_j = histogram.at<int>(j,t);
					float weight = (votes_i+votes_j)/1200.0;
					if(weight!=0)
					{
						mdg_i += weight * mdg(cluster_centers.row(t),dim,hist[i].cov[t],mean.row(i));
						mdg_j += weight * mdg(cluster_centers.row(t),dim,hist[j].cov[t],mean.row(j));
					}
				}

				//histo_coeff.at<double>(i,j) = -log( sqrt( (mdg_value[i]/mdg_i) * (mdg_value[j]/mdg_j) ) );
				histo_coeff.at<double>(i,j) = sqrt( (mdg_value[i]/mdg_i) * (mdg_value[j]/mdg_j) );
			}
		}
	}
	/*
	////////////////////
	float sum_GMM1,sum_GMM2;
	//float* sum_GMM;
	//sum_GMM = new float[hist.size()];
	//for(int i=0;i<hist.size();i++) sum_GMM[i] = 0.0;
	Mat mgd_value = Mat::zeros(hist.size(),hist.size(),CV_32F);
	//Mat sum_GMM = Mat::zeros(hist.size(),hist.size(),CV_32F);
	cout << "1";
	for(int index1=0;index1<hist.size();index1++)
	{
		for(int index2=0;index2<hist.size();index2++)
		{
			if(index1>index2) 
			{
				histo_coeff.at<double>(index1,index2) = histo_coeff.at<double>(index2,index1);
				continue;
			}
			else if(index1==index2)
			{
				continue;
			}
			vector<int> association_index;
			for(int t=index1*600;t<(index1+1)*600;t++)
			{
				association_index.push_back(t);
			}
			for(int t=index2*600;t<(index2+1)*600;t++)
			{
				association_index.push_back(t);
			}
			sum_GMM1 = sum_GMM2 = 0.0;
			float prob1[1200] = {0.0};
			float prob2[1200] = {0.0};	
			for(int i=0;i<1200;i++)
			{
				for(int j=0;j<k;j++)
				{
					float w1 = histogram.at<int>(index1,j)/600.0;
					float w2 = histogram.at<int>(index2,j)/600.0;
					prob1[i] += w1*mdg(model.row(association_index[i]),dim,hist[index1].cov[j],mean.row(j));
					prob2[i] += w2*mdg(model.row(association_index[i]),dim,hist[index2].cov[j],mean.row(j));
				}
				sum_GMM1 += prob1[i];
				sum_GMM2 += prob2[i];
			}
			for(int i=0;i<1200;i++)
			{
				prob1[i] /= sum_GMM1;
				prob2[i] /= sum_GMM2;
				histo_coeff.at<double>(index1,index2) += sqrt(prob1[i]*prob2[i]);//BC
			}
			
			histo_coeff.at<double>(index1,index2) = sqrt( 1.0 - histo_coeff.at<double>(index1,index2) );


			association_index.clear();
		}
	}
	cout << "2";
	*/
}


void Preprocessing_Data::distance_by_mdg2(Mat& histo_coeff,Mat cluster_centers,Mat model,Mat cluster_tag,vector< vector<int> > voting_result)
{
	int k = cluster_centers.rows;
	int raw_data_size = model.rows;
	int five_minutes_amount = histogram.rows;
	int dim = cluster_centers.cols;

	class Hist{
	public:
		Mat cov;
		Mat mean;
	};
	vector<Hist> hist(five_minutes_amount);


	for(int i=0;i<hist.size();i++)
	{
		hist[i].mean = Mat::zeros(1,dim,CV_32F);
		for(int j=0;j<k;j++)
		{
			hist[i].mean += histogram.at<int>(i,j)*cluster_centers.row(j);
		}
		//for(int t=i*600;t<(i+1)*600;t++)
		//{
		//	hist[i].mean += model.row(t);
		//}
		hist[i].mean /= 600.0;
	}

	for(int i=0;i<hist.size();i++)
	{
		hist[i].cov = Mat::zeros(dim,dim,CV_32F);

		for(int t=i*600;t<(i+1)*600;t++)
		{
			hist[i].cov += ( model.row(t) - hist[i].mean ).t() * ( model.row(t) - hist[i].mean );
		}
		//for(int t=i*600;t<(i+1)*600;t++)
		//{
		//	int tag = cluster_tag.at<int>(t,0);
		//	hist[i].cov += ( cluster_centers.row(tag) - hist[i].mean ).t() * ( cluster_centers.row(tag) - hist[i].mean );
		//}
		//for(int j=0;j<k;j++)
		//{
		//	for(int t=0;t<histogram.at<int>(i,j);t++)
		//	{
		//		hist[i].cov += ( cluster_centers.row(j) - hist[i].mean ).t() * ( cluster_centers.row(j) - hist[i].mean );
		//	}
		//}

		hist[i].cov /= 600.0;
		
	}

	double *mdg_value = new double[five_minutes_amount];
	for(int index=0;index<five_minutes_amount;index++)
	{
		mdg_value[index] = 0.0;
		for(int i=0;i<k;i++)
		{
			double weight = histogram.at<int>(index,i)/600.0;
			if(weight!=0)
			{
				mdg_value[index] += weight * mdg(cluster_centers.row(i),dim,hist[index].cov,hist[index].mean);	
			}
		}
		
		
		//for(int t=index*600;t<(index+1)*600;t++)
		//{
		//	mdg_value[index] += mdg(model.row(t),dim,hist[index].cov,hist[index].mean);
		//}		
		//cout << "GMM: " << mdg_value[index]  << endl;
	}
	
	//for(int i=0;i<five_minutes_amount;i++)
	//{
	//	double mdg_base = 0.0;
	//	for(int j=0;j<k;j++)
	//	{
	//		for(int t=0;t<voting_result[j].size();t++)
	//		{
	//			 mdg_base += mdg(cluster_centers.row(j),dim,hist[i].cov,hist[i].mean);
	//		}
	//	}
	//	mdg_base/=raw_data_size;
	//	mdg_value[i]/=mdg_base;
	//}

	//=======================association=======================//
	double mdg_i,mdg_j;
	Mat mdg_association = Mat::zeros(five_minutes_amount,five_minutes_amount,CV_32F);
	for(int i=0;i<five_minutes_amount;i++)
	{
		for(int j=0;j<five_minutes_amount;j++)
		{
			if(i==j) continue;
			else if(i>j)//speed up to avoid duplicate computation
			{
				histo_coeff.at<double>(i,j)  = histo_coeff.at<double>(j,i);
			}
			else
			{
				mdg_i = mdg_j = 0.0;
				for(int t=0;t<k;t++)
				{
					int votes_i = histogram.at<int>(i,t);
					int votes_j = histogram.at<int>(j,t);
					float weight = (votes_i+votes_j)/1200.0;
					if(weight!=0)
					{
						mdg_i += weight * mdg(cluster_centers.row(t),dim,hist[i].cov,hist[i].mean);
						mdg_j += weight * mdg(cluster_centers.row(t),dim,hist[j].cov,hist[j].mean);
					}
				}

				//double dist = 0.0;
				//if(mdg_i==0) mdg_i = e-5;
				//if(mdg_j==0) mdg_j = e-5;
				//if(mdg_i==0 && mdg_j!=0)
				//	dist = sqrt( (mdg_value[j]/mdg_j) );
				//else if(mdg_i!=0 && mdg_j==0)
				//	dist = sqrt((mdg_value[i]/mdg_i));
				//else if(mdg_i!=0 || mdg_j!=0)
				//histo_coeff.at<double>(i,j) = -log( sqrt( (mdg_value[i]/mdg_i) * (mdg_value[j]/mdg_j) ) );
				//histo_coeff.at<double>(i,j) = sqrt( (mdg_value[i]/mdg_i) * (mdg_value[j]/mdg_j) );
				
			}
		}
	}
	//=========================================================//
	/*
	float mdg_i,mdg_j;
	Mat mdg_association = Mat::zeros(five_minutes_amount,five_minutes_amount,CV_32F);
	for(int i=0;i<five_minutes_amount;i++)
	{
		for(int j=0;j<five_minutes_amount;j++)
		{
			if(i==j) continue;
			else if(i>j)//speed up to avoid duplicate computation
			{
				histo_coeff.at<double>(i,j)  = histo_coeff.at<double>(j,i);
			}
			else
			{
				mdg_i = mdg_j = 0.0;
				for(int t=i*600;t<(i+1)*600;t++)
				{
					mdg_i += mdg(model.row(t),dim,hist[i].cov,hist[i].mean);
					mdg_j += mdg(model.row(t),dim,hist[j].cov,hist[j].mean);
				}
				for(int t=j*600;t<(j+1)*600;t++)
				{
					mdg_i += mdg(model.row(t),dim,hist[i].cov,hist[i].mean);
					mdg_j += mdg(model.row(t),dim,hist[j].cov,hist[j].mean);
				}

				//histo_coeff.at<double>(i,j) = -log( sqrt( (mdg_value[i]/mdg_i) * (mdg_value[j]/mdg_j) ) );
				histo_coeff.at<double>(i,j) = sqrt( (mdg_value[i]/mdg_i) * (mdg_value[j]/mdg_j) );
			}
		}
	}*/



	//double min,max;
	//minMaxLoc(histo_coeff, &min, &max);
	//Mat max_mat = Mat::ones(histo_coeff.rows,histo_coeff.cols,CV_64F)*max;
	//subtract(max_mat,histo_coeff,histo_coeff);

	//for(int i=0;i<five_minutes_amount;i++)
	//{
	//	for(int j=0;j<five_minutes_amount;j++)
	//	{
	//		for(int t=0;t<k;t++)
	//		{
	//			histo_coeff.at<double>(i,j) += abs( mdg_value.at<float>(i,t) - mdg_value.at<float>(j,t) );
	//		}
	//	}
	//}

	//for(int i=0;i<five_minutes_amount;i++)
	//{
	//	for(int j=0;j<five_minutes_amount;j++)
	//	{
	//		//histo_coeff.at<double>(i,j) += abs(mdg_value[i] - mdg_value[j]);
	//		histo_coeff.at<double>(i,j) += sqrt(mdg_value[i] * mdg_value[j]);
	//		//histo_coeff.at<double>(i,j) = mdg_association.at<float>(i,j);
	//	}
	//}

	for(int i=0;i<five_minutes_amount;i++)
	{
		for(int j=0;j<five_minutes_amount;j++)
		{
			Mat cov_avg = (hist[i].cov + hist[j].cov)/2.0;
			Mat DB = Mat::zeros(1,1,CV_32F);
			DB = 0.125 * (hist[i].mean-hist[j].mean) * cov_avg * (hist[i].mean-hist[j].mean).t() 
				+ 0.5 * log( determinant(cov_avg) / sqrt( determinant(hist[i].cov) * determinant(hist[j].cov) ) );
			histo_coeff.at<double>(i,j) = DB.at<float>(0,0);
		}
	}

	delete [] mdg_value;
	
}

void Preprocessing_Data::distance_by_mdg(Mat& histo_coeff,Mat cluster_centers,Mat model,Mat cluster_tag,vector< vector<int> > voting_result)
{
	int k = cluster_centers.rows;
	int raw_data_size = model.rows;

	class cluster{
	public: 
		Mat cov;	
		int num;
	};
	vector<cluster> c_list(k);
	for(int i=0;i<k;i++)
	{
		c_list[i].cov = Mat::zeros(cluster_centers.cols, cluster_centers.cols, CV_32F);
		c_list[i].num = 0;
	}

	// 計算資料點的重心(平均)
	Mat mean = Mat::zeros(1, cluster_centers.cols, CV_32F);

	// 計算covariance matrix
	for (int i=0;i<k;i++){
		mean = cluster_centers.row(i);
		for(int j=0;j<voting_result[i].size();j++)
		{
			int index = voting_result[i][j];
			c_list[i].cov += ( model.row(index) - mean).t() * ( model.row(index) - mean );
			c_list[i].num++;
		}
	}

	for(int i=0;i<k;i++)
	{
		c_list[i].cov /= double(c_list[i].num);
	}
	
	//GMM for association of every 5 minutes
	//double *mdg_base = new double[histogram.rows];
	//for(int index=0;index<histogram.rows;index++)
	//{
	//	mdg_base[index] = 0.0;
	//	for(int i=0;i<k;i++)
	//	{
	//		float weight = (float)voting_result[i].size()/(float)raw_data_size;
	//		for(int j=0;j<voting_result[i].size();j++)
	//		{
	//			mdg_base[index] += weight * mdg(cluster_centers.row(i),model.cols,c_list[i].cov,cluster_centers.row(i));
	//		}
	//	}
	//}

	//Multivariate Gaussian Distribution
	//mdg_value[index]:GMM for every 5 minutes
	
	//double *mdg_value = new double[histogram.rows];
	//for(int index=0;index<histogram.rows;index++)
	//{
	//	mdg_value[index] = 0.0;
	//	for(int i=0;i<k;i++)
	//	{
	//		float weight = histogram.at<int>(index,i)/600.0;
	//		//mdg_value[index] += weight * mdg(cluster_centers.row(i),model.cols,c_list[i].cov,cluster_centers.row(i));
	//		for(int j=0;j<voting_result[i].size();j++)
	//		{
	//			mdg_value[index] += weight * mdg(model.row(i),model.cols,c_list[i].cov,cluster_centers.row(i));
	//		}
	//	}
	//	//cout << "GMM: " << mdg_value[index]  << endl;
	//}
	
	
	Mat mdg_value = Mat::zeros(histogram.rows,k,CV_32F);
	
	for(int index=0;index<histogram.rows;index++)
	{
		//mdg_value[index] = 0.0;
		for(int i=0;i<k;i++)
		{
			float weight = histogram.at<int>(index,i)/600.0;
			mdg_value.at<float>(index,i) = weight * mdg(cluster_centers.row(i),model.cols,c_list[i].cov,cluster_centers.row(i));
			//mdg_value[index] += weight * mdg(cluster_centers.row(i),model.cols,c_list[i].cov,cluster_centers.row(i));
			//for(int j=0;j<voting_result[i].size();j++)
			//{
			//	mdg_value[index] += weight * mdg(model.row(i),model.cols,c_list[i].cov,cluster_centers.row(i));
			//}
		}
		//cout << "GMM: " << mdg_value[index]  << endl;
	}

	//Mat mdg_value = Mat::zeros(histogram.rows,histogram.rows,CV_64F);
	//for(int i=0;i<histogram.rows;i++)
	//{
	//	for(int j=0;j,histogram.rows;j++)
	//	{	
	//		mdg_value.at<double>(i,j) = 0.0;
	//		if(i==j)
	//		{	
	//			continue;
	//		}
	//		double mdg_i,mdg_j;
	//		mdg_i = mdg_j = 0.0;
	//		for(int t=0;t<k;t++)
	//		{
	//			double wi = histogram.at<int>(i,t)/600.0;
	//			double wj = histogram.at<int>(j,t)/600.0;
	//			double joint_votes = (histogram.at<int>(i,t) + histogram.at<int>(j,t))/1200.0;
	//			mdg_i += wi*mdg(cluster_centers.row(t),model.cols,c_list[t].cov,cluster_centers.row(t));				
	//			mdg_j += wj*mdg(cluster_centers.row(t),model.cols,c_list[t].cov,cluster_centers.row(t));	
	//		}
	//	}
	//}

	
	int time_step_amount = floor(raw_data.size()/600.0);
	for(int i=0;i<time_step_amount;i++)
	{
		for(int j=0;j<time_step_amount;j++)
		{
			//if(i==j || i>j) continue;

			for(int t=0;t<k;t++)
			{
				//histo_coeff.at<double>(i,j) += abs( mdg_value.at<float>(i,t) - mdg_value.at<float>(j,t) );
				histo_coeff.at<double>(i,j) += sqrt( mdg_value.at<float>(i,t)*mdg_value.at<float>(j,t) );
			}
			//histo_coeff.at<double>(i,j) = 1.0 - histo_coeff.at<double>(i,j);
		}
	}
	//for(int i=0;i<time_step_amount;i++)
	//{
	//	for(int j=0;j<time_step_amount;j++)
	//	{
	//		//histo_coeff.at<double>(i,j) += abs(mdg_value[i] - mdg_value[j]);
	//		histo_coeff.at<double>(i,j) += sqrt(mdg_value[i] * mdg_value[j]);
	//	}
	//}

	//delete [] mdg_value;

}

void Preprocessing_Data::distance_by_bh(Mat& histo_coeff,int k)
{
	int time_step_amount = histo_coeff.rows;
	for(int i=0;i<time_step_amount;i++)
	{
		for(int j=0;j<time_step_amount;j++)
		{
			if(i==j) continue;
			else if(i>j)
			{
				histo_coeff.at<double>(i,j) = histo_coeff.at<double>(j,i);
			}
			else
			{
				double BC = 0.0;

				for(int t=0;t<k;t++)
				{
					BC += adjust_weight[i][t] * adjust_weight[j][t];
					//BC += (histogram.at<int>(i,t)/600.0) *  (histogram.at<int>(j,t)/600.0); 
				}
				BC = sqrt(BC);
				histo_coeff.at<double>(i,j) = -log( MAX(BC,0.0000001) );
				//histo_coeff.at<double>(i,j) = sqrt(1.0 - BC);
			}
		}
	}
}

Mat Preprocessing_Data::Position_by_MDS(Mat cluster_centers,Mat model,Mat cluster_tag,int k)
{
	int time_step_amount = floor(raw_data.size()/600.0);
	Mat histo_coeff = Mat::zeros(time_step_amount,time_step_amount,CV_64F);

	//==================GMM(Gaussian Mixture Model)======================//
	Mat Ev = Mat::zeros(time_step_amount,cluster_centers.cols,CV_32F);
	distance_by_GMM(histo_coeff,Ev,cluster_centers,k);
	
	//========Euclidean Distance + weighting by cluster distance========//
	//distance_by_Euclidean(histo_coeff,cluster_centers,k);

	//================Multivariate Gaussian Distribution================// 
	//distance_by_mdg3(histo_coeff,cluster_centers,model,cluster_tag,voting_result);
	//distance_by_mdg2(histo_coeff,cluster_centers,model,cluster_tag,voting_result);

	//distance_by_bh(histo_coeff,k);

	output_mat_as_csv_file_double("histo_coeff.csv",histo_coeff);

	//CompareHist
	//char* testArr1 = new char[k];
	//char *testArr2 = new char[k];
	//for(int i=0;i<k;i++) testArr1[i] = histogram.at<int>(50,i);
	//for(int i=0;i<k;i++) testArr2[i] = histogram.at<int>(100,i);
	////char testArr1[] = {4,23,0,12,0,0,0,0,1,0,0,0,201,35,0,52,50,14,66,47,67,0,6,22,0};
	////char testArr2[] = {57,51,66,22,0,0,4,0,0,0,1,0,0,1,0,0,1,11,15,7,48,102,0,5,209};
	//Mat M1 = Mat(1,k,CV_8UC1,testArr1);
	//Mat M2 = Mat(1,k,CV_8UC1,testArr2);
	////M1 = histogram.row(0);
	////M2 = histogram.row(1);

	//int histSize = k;
	//float range[] = {0, 600};
	//const float* histRange = {range};
	//bool uniform = true;
	//bool accumulate = false;
	//Mat a1_hist, a2_hist;

	//calcHist(&M1, 1, 0, cv::Mat(), a1_hist, 1, &histSize, &histRange, uniform, accumulate );
	//calcHist(&M2, 1, 0, cv::Mat(), a2_hist, 1, &histSize, &histRange, uniform, accumulate );
	//double compar_bh = compareHist(a1_hist, a2_hist, CV_COMP_BHATTACHARYYA);
	//cout << compar_bh << endl;
	//==========================================================================//

	/*
	Matrix<double,Dynamic,Dynamic> histo_coeff_EigenType;//The type pass to Tapkee must be "double" not "float"
	cv2eigen(histo_coeff,histo_coeff_EigenType);
	TapkeeOutput output = tapkee::initialize() 
						   .withParameters((method=MultidimensionalScaling,target_dimension=1))
						   .embedUsing(histo_coeff_EigenType);
	Mat MDS_mat; //Type:double  
	eigen2cv(output.embedding,MDS_mat);
	normalize(MDS_mat.col(0),MDS_mat.col(0),1,100,NORM_MINMAX);//normalize to 1~1000	
	//output_mat_as_txt_file("normalized_MDS_mat.txt",MDS_mat);
	MDS_mat = MDS_mat.mul(50.0);
	//output_mat_as_txt_file("normalized_MDS_mat_2.txt",MDS_mat);

	return MDS_mat; */
	
	Mat position_mat = MDS(histo_coeff,1).clone();

	return position_mat;
}

Mat Preprocessing_Data::MDS(Mat target_mat, int dim)
{
	Matrix<double,Dynamic,Dynamic> target_mat_EigenType;//The type pass to Tapkee must be "double" not "float"
	cv2eigen(target_mat,target_mat_EigenType);
	TapkeeOutput output = tapkee::initialize() 
						   .withParameters((method=MultidimensionalScaling,target_dimension=1))
						   .embedUsing(target_mat_EigenType);
	
	Mat MDS_mat; //Type:double  
	eigen2cv(output.embedding,MDS_mat);
	normalize(MDS_mat.col(0),MDS_mat.col(0),1,5000,NORM_MINMAX);//normalize to 1~1000	
	//MDS_mat = MDS_mat.mul(50.0);

	return MDS_mat;
}

Mat Preprocessing_Data::lab_alignment_by_cube(Mat cluster_center)
{
	int vTotal = lab_vertices.size();
	int k = cluster_center.rows;
	//Turn LAB vector into LAB mat
	Mat lab_mat = Mat::zeros(vTotal,3,CV_32F);
	for(int i=0;i<vTotal;i++)
	{
		for(int j=0;j<3;j++)
		{
			lab_mat.at<float>(i,j) = lab_vertices[i][j+1];
		}
	}

	//Compute centroid of cluster center
	Mat cluster_center_centroid = compute_centroid(cluster_center);
	//Align the centroid to the origin by subtract all points to centroid
	Mat cluster_center_alignment_mat = Mat::zeros(cluster_center.rows,cluster_center.cols,CV_32F);
	for(int i=0;i<cluster_center.rows;i++)
	{
		for(int j=0;j<cluster_center.cols;j++)
		{
			cluster_center_alignment_mat.at<float>(i,j) = cluster_center.at<float>(i,j) - cluster_center_centroid.at<float>(0,j);
		}
	}

	Mat lab_centroid = compute_centroid(lab_mat).clone();
	//Align the centroid to the origin by subtract all points to centroid
	Mat lab_alignment_mat = Mat::zeros(lab_mat.rows,lab_mat.cols,CV_32F);
	for(int i=0;i<lab_mat.rows;i++)
	{
		for(int j=0;j<lab_mat.cols;j++)
		{
			lab_alignment_mat.at<float>(i,j) = lab_mat.at<float>(i,j) - lab_centroid.at<float>(0,j);
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	Mat component,cluster_center_coeff;
	Mat cluster_center_component = Mat::zeros(3,3,CV_32F);
	int rDim = 3;
	reduceDimPCA(cluster_center_alignment_mat, rDim, component, cluster_center_coeff);//PCA 4dim->3dim (for dimension reduction)
	for(int i=0;i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
			cluster_center_component.at<float>(i,j) = component.at<float>(i,j);
		}
	}
	cout << "cluster_center_component " << endl << cluster_center_component << endl;

	//lab vertices 3 axis of PCA
	Mat lab_components,lab_coeff;
	rDim = 3;
	reduceDimPCA(lab_alignment_mat, rDim, lab_components, lab_coeff); //PCA 3dim->3dim (for principal component)
	cout << "lab_components " << endl << lab_components << endl;

	////////////////////////////////////
	double min,max;
	double cmax[3],cmin[3];
	double lmax[3],lmin[3];
	double lmax_const[3],lmin_const[3];
	for(int i=0;i<3;i++)
	{
		minMaxLoc(cluster_center_coeff.col(i), &min, &max);
		cmax[i] = max;
		cmin[i] = min;
		minMaxLoc(lab_coeff.col(i), &min, &max);
		lmax[i] = max;
		lmin[i] = min;
		lmax_const[i] = max;
		lmin_const[i] = min;
	}


	Mat max_lab_normalize_mat = Mat::zeros(k,3,CV_32F);
	bool flag = false;
	vector<double> scale_factor;
	for(double i=0.0;i<=80.0;i+=5.0)
		scale_factor.push_back(i);
	double min_scale_1 = 100.0;
	double min_scale_2 = 100.0;
	double min_scale_3 = 100.0;

	for(int u=0;u<scale_factor.size();u++)
	{
		for(int v=0;v<scale_factor.size();v++)
		{
			for(int w=0;w<scale_factor.size();w++)
			{
				//flag = false;
				//while(flag==false)
				//{
					flag = false;

					Mat lab_normalize_mat = Mat::zeros(k,3,CV_32F);
					for(int i=0;i<k;i++)
					{
						double c[3];
						for(int j=0;j<3;j++)
						{
							c[j] = cluster_center_coeff.at<float>(i,j);
							double f_c = lmin[j] + (c[j]-cmin[j])/(cmax[j]-cmin[j])*(lmax[j]-lmin[j]);
							//cout << "f_c " << f_c << endl;
							add(lab_components.row(j).mul(f_c),lab_normalize_mat.row(i),lab_normalize_mat.row(i));
						}
					}

					lmin[0] = lmin_const[0] + scale_factor[u];
					lmax[0] = lmax_const[0] - scale_factor[u];
					lmin[1] = lmin_const[1] + scale_factor[v];
					lmax[1] = lmax_const[1] - scale_factor[v];
					lmin[2] = lmin_const[2] + scale_factor[w];
					lmax[2] = lmax_const[2] - scale_factor[w];

					for(int i=0;i<lab_normalize_mat.rows;i++)
					{
						add(lab_centroid.row(0),lab_normalize_mat.row(i),lab_normalize_mat.row(i));
					}

					for(int i=0;i<lab_normalize_mat.rows;i++)
					{
						if(lab_boundary_test(lab_normalize_mat.at<float>(i,0),lab_normalize_mat.at<float>(i,1),lab_normalize_mat.at<float>(i,2))==false)
						{
							flag = false;
							break;
						}
						flag = true;
					}	

					if(flag)
					{
						if(scale_factor[u] < min_scale_1)
						{
							min_scale_1 = scale_factor[u];
							max_lab_normalize_mat = lab_normalize_mat.clone();
						}
						if(scale_factor[v] > min_scale_2)
						{
							min_scale_2 = scale_factor[v];
							max_lab_normalize_mat = lab_normalize_mat.clone();
						}		
						if(scale_factor[w] > min_scale_3)
						{
							min_scale_3 = scale_factor[w];
							max_lab_normalize_mat = lab_normalize_mat.clone();
						}
					}

					//cout << "flag " << flag << endl;
				//}
			}
		}
	}

	cout << "min scale 1 " << min_scale_1 << endl;
	cout << "min scale 2 " << min_scale_2 << endl;
	cout << "min scale 3 " << min_scale_3 << endl;

	output_mat_as_csv_file("lab_normalize_mat.csv",max_lab_normalize_mat);

	//lab = max_lab_normalize_mat.clone();
	Mat rgb = LAB2RGB(max_lab_normalize_mat).clone();
	output_mat_as_csv_file("rgb.csv",rgb);

	return rgb;
}

Mat Preprocessing_Data::lab_alignment_new(Mat cluster_center)
{
	int vTotal = lab_vertices.size();
	//Turn LAB vector into LAB mat
	Mat lab_mat = Mat::zeros(vTotal,3,CV_32F);
	for(int i=0;i<vTotal;i++)
	{
		for(int j=0;j<3;j++)
		{
			lab_mat.at<float>(i,j) = lab_vertices[i][j+1];
		}
	}
	//Compute centroid of cluster center
	Mat cluster_center_centroid = compute_centroid(cluster_center).clone();

	//Align the centroid to the origin by subtract all points to centroid
	Mat cluster_center_alignment_mat = Mat::zeros(cluster_center.rows,cluster_center.cols,CV_32F);
	for(int i=0;i<cluster_center.rows;i++)
	{
		for(int j=0;j<cluster_center.cols;j++)
		{
			cluster_center_alignment_mat.at<float>(i,j) = cluster_center.at<float>(i,j) - cluster_center_centroid.at<float>(0,j);
		}
	}
	Mat component,cluster_center_coeff;
	Mat cluster_center_component = Mat::zeros(3,3,CV_32F);
	int rDim = 3;
	reduceDimPCA(cluster_center_alignment_mat, rDim, component, cluster_center_coeff);//PCA 4dim->3dim (for dimension reduction)
	for(int i=0;i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
			cluster_center_component.at<float>(i,j) = component.at<float>(i,j);
		}
	}
	cout << "cluster_center_component " << endl << cluster_center_component << endl;
	//cluster center 3 axis of PCA


	output_mat_as_csv_file("cluster_center_coeff.csv",cluster_center_coeff);
	//compute centroid of LAB
	Mat lab_centroid = compute_centroid(lab_mat).clone();
	//Align the centroid to the origin by subtract all points to centroid
	Mat lab_alignment_mat = Mat::zeros(lab_mat.rows,lab_mat.cols,CV_32F);
	for(int i=0;i<lab_mat.rows;i++)
	{
		for(int j=0;j<lab_mat.cols;j++)
		{
			lab_alignment_mat.at<float>(i,j) = lab_mat.at<float>(i,j) - lab_centroid.at<float>(0,j);
		}
	}
	//lab vertices 3 axis of PCA
	Mat lab_components,lab_coeff;
	rDim = 3;
	reduceDimPCA(lab_alignment_mat, rDim, lab_components, lab_coeff); //PCA 3dim->3dim (for principal component)
	cout << "lab_components " << endl << lab_components << endl;

	//////////////////////////////////////////////////////////////////////////////////////////////
	vector<float> move_vector;
	for(float k=-5.0;k<=5.0;k+=1.0)
		move_vector.push_back(k);

	//vector<float> scale_vector;
	//for(float k=1.0;k<=100.0;k+=1.0)
	//	scale_vector.push_back(k);

	Mat align_mat = Mat::zeros(cluster_center.rows,3,CV_32F);
	Mat max_align_mat = Mat::zeros(cluster_center.rows,3,CV_32F);
	int start = 1;
	int luminance_threshold = 30;

	Mat cluster_center_coeff_const = cluster_center_coeff.clone();
	float scale = 1.0;
	float max_scale = 1.0;
	float max_move_x = 0.0;
	float max_move_y = 0.0;
	float max_move_z = 0.0;
	bool flag = true;

	for(int u=0;u<move_vector.size();u++)
	{
		for(int v=0;v<move_vector.size();v++)
		{
			for(int w=0;w<move_vector.size();w++)
			{
				float scale = max_scale;
				flag = true;
				while(flag)
				{			
						cluster_center_coeff = cluster_center_coeff_const.mul(scale);
						//for(int i=0;i<cluster_center_coeff_const.rows;i++)
						//{
						//	cluster_center_coeff.at<float>(i,0) = cluster_center_coeff_const.at<float>(i,0) * scale_vector[u];
						//	cluster_center_coeff.at<float>(i,1) = cluster_center_coeff_const.at<float>(i,1) * scale_vector[v];
						//	cluster_center_coeff.at<float>(i,2) = cluster_center_coeff_const.at<float>(i,2) * scale_vector[w];
						//}

						for(int i=0;i<cluster_center.rows;i++)
						{
							for(int j=0;j<3;j++)
							{
								Mat result = cluster_center_coeff.row(i)*lab_components.col(j);
								align_mat.at<float>(i,j) = result.at<float>(0,0);
							}
						}
						add(align_mat.col(0),move_vector[u],align_mat.col(0)); //move
						add(align_mat.col(1),move_vector[v],align_mat.col(1)); //move
						add(align_mat.col(2),move_vector[w],align_mat.col(2)); //move
				
						//把重心平移回去
						for(int i=0;i<align_mat.rows;i++)
						{
							for(int j=0;j<3;j++)
							{
								align_mat.at<float>(i,j) += lab_centroid.at<float>(0,j);
							}
						}

						//cout << align_mat << endl;
						//system("pause");

						for(int i=0;i<align_mat.rows;i++)
						{
							if( (align_mat.at<float>(i,0)<luminance_threshold) || (align_mat.at<float>(i,0)>85.0) )
							{
								flag = false;
								break;
							}
						}

						if(flag)
						{
							for(int i=0;i<align_mat.rows;i++)
							{
								if(lab_boundary_test(align_mat.at<float>(i,0),align_mat.at<float>(i,1),align_mat.at<float>(i,2))==false)
								{
									flag = false;
									break;
								}
							}	
						}

						if(flag)
						{
							if(scale>max_scale)
							{
								max_scale = scale;
								max_move_x = move_vector[u];
								max_move_y = move_vector[v];
								max_move_z = move_vector[w];
								max_align_mat = align_mat.clone();
							}	
						}

						scale+=1.0;
				}
			}
		}
	}


	cout << "max scale " << max_scale << endl;
	cout << "max move x " << max_move_x << endl;
	cout << "max move y " << max_move_y << endl;
	cout << "max move z " << max_move_z << endl;

	output_mat_as_csv_file("lab_raw_data.csv",max_align_mat);
	//printf("max_move : %f max_scale : %f\n",max_move,max_scale);
	
	lab = max_align_mat.clone();
	Mat rgb_mat = LAB2RGB(max_align_mat).clone();

	return rgb_mat;
}

Mat Preprocessing_Data::lab_alignment(Mat cluster_center)
{
	int vTotal = lab_vertices.size();
	//Turn LAB vector into LAB mat
	Mat lab_mat = Mat::zeros(vTotal,3,CV_32F);
	for(int i=0;i<vTotal;i++)
	{
		for(int j=0;j<3;j++)
		{
			lab_mat.at<float>(i,j) = lab_vertices[i][j+1];
		}
	}
	//Compute centroid of cluster center
	Mat cluster_center_centroid = compute_centroid(cluster_center);

	//Align the centroid to the origin by subtract all points to centroid
	Mat cluster_center_alignment_mat = Mat::zeros(cluster_center.rows,cluster_center.cols,CV_32F);
	for(int i=0;i<cluster_center.rows;i++)
	{
		for(int j=0;j<cluster_center.cols;j++)
		{
			cluster_center_alignment_mat.at<float>(i,j) = cluster_center.at<float>(i,j) - cluster_center_centroid.at<float>(0,j);
		}
	}
	Mat component,cluster_center_PCA;
	Mat cluster_center_component = Mat::zeros(3,3,CV_32F);
	Mat garbage;
	int rDim = 3;
	reduceDimPCA(cluster_center_alignment_mat, rDim, component, cluster_center_PCA);//PCA 4dim->3dim (for dimension reduction)
	//reduceDimPCA(cluster_center_PCA, rDim, cluster_center_component, garbage); //PCA 3dim->3dim (for principal component)
	for(int i=0;i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
			cluster_center_component.at<float>(i,j) = component.at<float>(i,j);
		}
	}
	cout << "cluster_center_component " << endl << cluster_center_component << endl;
	//cluster center 3 axis of PCA
	Mat cluster_center_axis = Mat::zeros(3,3,CV_32F);
	cluster_center_axis = cluster_center_component;

	output_mat_as_csv_file("cluster_center_PCA.csv",cluster_center_PCA);
	//compute centroid of LAB
	Mat lab_centroid = compute_centroid(lab_mat);
	//lab vertices 3 axis of PCA
	Mat lab_components,lab_PCA;
	rDim = 3;
	reduceDimPCA(lab_mat, rDim, lab_components, lab_PCA); //PCA 3dim->3dim (for principal component)
	cout << "lab_components " << endl << lab_components << endl;
	Mat lab_axis = Mat::zeros(3,3,CV_32F);
	lab_axis = lab_components;

	//////////////////////////////////////////////////////////////////
	Mat cluster_center_PCA_const = cluster_center_PCA;
	vector<float> move_vector;
	for(float k=-0.5;k<=0.5;k+=0.1)
		move_vector.push_back(k);

	float max_move = 0.0;
	float max_scale = 0.0;
	//Mat max_align_mat = cluster_center_PCA;
	Mat align_mat = Mat::zeros(cluster_center.rows,3,CV_32F);
	Mat max_align_mat = Mat::zeros(cluster_center.rows,3,CV_32F);
	int start = 1;
	int luminance_threshold = 30;
	vector<int> scale_vector;
	//binary search the best scale & convell hull for speed up
	for(int t=0;t<move_vector.size();t++)
	{	
		for(int i=start;i<=150;i++)
			scale_vector.push_back(i);
		
		float low = start;	
		float high = scale_vector.size();
		while(low <= high)
		{
			float mid = (low + high)/2; 
			//cout << mid << " " << high << " " << low << endl;
			Mat cluster_center_PCA_temp,cluster_center_PCA_weight,cluster_center_axis_invert;
			add(cluster_center_PCA_const,move_vector[t],cluster_center_PCA_temp); //move
			cluster_center_PCA_temp = cluster_center_PCA_temp.mul(mid); //scale
			cluster_center_axis_invert = cluster_center_axis.inv();
			cluster_center_PCA_weight = cluster_center_PCA_temp * cluster_center_axis_invert;
			align_mat = cluster_center_PCA_weight*lab_axis;

			//把重心平移回去
			for(int i=0;i<align_mat.rows;i++)
			{
				for(int j=0;j<3;j++)
				{
					align_mat.at<float>(i,j) += lab_centroid.at<float>(0,j);
				}
			}
	
			bool flag = true;
			for(int i=0;i<align_mat.rows;i++)
			{
				if( (align_mat.at<float>(i,0)<luminance_threshold) || (align_mat.at<float>(i,0)>90.0) )
				{
					flag = false;
					break;
				}
			}

			if(flag)
			{
				for(int i=0;i<align_mat.rows;i++)
				{
					if(lab_boundary_test(align_mat.at<float>(i,0),align_mat.at<float>(i,1),align_mat.at<float>(i,2))==false)
					{
						flag = false;
						break;
					}
				}
			}

			if(high<=low)
				break;
			else if(flag)
				low = mid + 1;
			else 
				high = mid - 1;

		}

		if(low>=max_scale)
		{		
			max_scale = low;
			max_move = move_vector[t];
			max_align_mat = align_mat.clone();
			start = max_scale;
		}

		scale_vector.clear();

	}

	if(max_scale==0)
	{
		for(int i=0;i<max_align_mat.rows;i++)
		{
			for(int j=0;j<max_align_mat.cols;j++)
			{
				max_align_mat.at<float>(i,j) += lab_centroid.at<float>(0,j);
			}
		}
	}

	output_mat_as_csv_file("lab_raw_data.csv",max_align_mat);
	printf("max_move : %f max_scale : %f\n",max_move,max_scale);
	
	lab = max_align_mat.clone();
	Mat rgb_mat = LAB2RGB(max_align_mat).clone();

	return rgb_mat;
}

Mat Preprocessing_Data::compute_centroid(Mat input_mat)
{
	Mat input_mat_centroid = Mat::zeros(1,input_mat.cols,CV_32F);
	for(int i=0;i<input_mat.rows;i++)
	{
		for(int j=0;j<input_mat.cols;j++)
		{
			input_mat_centroid.at<float>(0,j) += input_mat.at<float>(i,j);
		}
	}

	for(int j=0;j<input_mat.cols;j++) input_mat_centroid.at<float>(0,j)/=input_mat.rows;
	
	return input_mat_centroid;
}

bool Preprocessing_Data::lab_boundary_test(float p1,float p2,float p3)
{
	bool test = true;
	Mat lab_color(1, 1, CV_32FC3);
	Mat rgb_color(1, 1, CV_32FC3);
	lab_color.at<Vec3f>(0, 0) = Vec3f(p1, p2, p3);
	cvtColor(lab_color, rgb_color, CV_Lab2BGR);
	cvtColor(rgb_color, lab_color, CV_BGR2Lab);
	if(abs(lab_color.at<Vec3f>(0,0).val[0] - p1) > 1.0 || abs(lab_color.at<Vec3f>(0,0).val[1] - p2) > 1.0 || abs(lab_color.at<Vec3f>(0,0).val[2] - p3) > 1.0)
		test = false;
	return test;
}

Mat Preprocessing_Data::LAB2RGB(Mat lab_mat)
{
	Mat rgb_mat2 = lab_mat;
	for(int i=0;i<lab_mat.rows;i++)
	{
		Mat color(1, 1, CV_32FC3);
		color.at<Vec3f>(0, 0) = Vec3f(lab_mat.at<float>(i,0),lab_mat.at<float>(i,1),lab_mat.at<float>(i,2));		
		cvtColor(color, color, CV_Lab2BGR);
		rgb_mat2.at<float>(i,0) = color.at<Vec3f>(0,0).val[2];//R
		rgb_mat2.at<float>(i,1) = color.at<Vec3f>(0,0).val[1];//G
		rgb_mat2.at<float>(i,2) = color.at<Vec3f>(0,0).val[0];//B
	}

	return rgb_mat2;
}

Mat Preprocessing_Data::RGB2LAB(Mat rgb)
{
	Mat lab_mat = rgb.clone();
	for(int i=0;i<lab_mat.rows;i++)
	{
		Mat color(1, 1, CV_32FC3);
		color.at<Vec3f>(0, 0) = Vec3f(rgb.at<float>(i,2),rgb.at<float>(i,1),rgb.at<float>(i,0));		
		cvtColor(color, color, CV_BGR2Lab);
		lab_mat.at<float>(i,0) = color.at<Vec3f>(0,0).val[0];//L
		lab_mat.at<float>(i,1) = color.at<Vec3f>(0,0).val[1];//a
		lab_mat.at<float>(i,2) = color.at<Vec3f>(0,0).val[2];//b
	}

	return lab_mat;
}

void Preprocessing_Data::read_lab_csv()
{
	FILE *csv_file;
	csv_file = fopen("LAB_vertices.csv","r");
	if(!csv_file) 
	{
		cout << "Can't open config file!" << endl;
		exit(1);
	}

	char line[LENGTH];
	char *token;
	int i,j;
	i = j = 0;
	//fgets(line,LENGTH,csv_file); //ignore sep=
	//fgets(line,LENGTH,csv_file); //ignore title

	while(!feof(csv_file))
	{
		fgets(line,LENGTH,csv_file);
		//token = strtok(line,";");
		token = strtok(line,",");
		lab_vertices.push_back(vector<float> (1));
		//printf("%s ",token);
		while(token!=NULL)
		{
			lab_vertices.back().push_back(atof(token));
			//token = strtok(NULL," ;:");
			token = strtok(NULL," ,");
		}
	}
	
	//cout << lab_vertices.size() << " " << lab_vertices[0].size() << endl; //6146 x 4
	//cout << lab_vertices[3][1] << " " << lab_vertices[3][2] << lab_vertices[3][3] << endl;

	fclose(csv_file);
}

Mat Preprocessing_Data::lab_alignment_dim1(Mat cluster_center)
{
	//read_lab_csv();
	int vTotal = lab_vertices.size();
	//Turn LAB vector into LAB mat
	Mat lab_mat = Mat::zeros(vTotal,3,CV_32F);
	for(int i=0;i<vTotal;i++)
	{
		for(int j=0;j<3;j++)
		{
			lab_mat.at<float>(i,j) = lab_vertices[i][j+1];
		}
	}
	//Compute centroid of cluster center
	Mat cluster_center_centroid = compute_centroid(cluster_center);

	//Align the centroid to the origin by subtract all points to centroid
	Mat cluster_center_alignment_mat = Mat::zeros(cluster_center.rows,cluster_center.cols,CV_32F);
	for(int i=0;i<cluster_center.rows;i++)
	{
		for(int j=0;j<cluster_center.cols;j++)
		{
			cluster_center_alignment_mat.at<float>(i,j) = cluster_center.at<float>(i,j) - cluster_center_centroid.at<float>(0,j);
		}
	}
	Mat cluster_center_component,cluster_center_PCA;
	int rDim = 1;
	reduceDimPCA(cluster_center_alignment_mat, rDim, cluster_center_component, cluster_center_PCA);
	//cluster center 3 axis of PCA
	Mat cluster_center_axis = Mat::zeros(1,1,CV_32F);
	cluster_center_axis.at<float>(0,0) = cluster_center_component.at<float>(0,0);

	//compute centroid of LAB
	Mat lab_centroid = compute_centroid(lab_mat);
	
	//lab vertices 3 axis of PCA
	Mat lab_components,lab_PCA;
	rDim = 1;
	reduceDimPCA(lab_mat, rDim, lab_components, lab_PCA);
	Mat lab_axis = Mat::zeros(1,3,CV_32F);
	cout << lab_components << endl;
	for(int j=0;j<3;j++)
	{
		lab_axis.at<float>(0,j) = lab_components.at<float>(0,j);
	}
	
	//////////////////////////////////////////////////////////////////
	Mat cluster_center_PCA_const = cluster_center_PCA;
	vector<float> move_vector;
	for(float k=-0.5;k<=0.5;k+=0.1)
		move_vector.push_back(k);

	float max_move = 0.0;
	float max_scale = 0.0;
	Mat align_mat;
	Mat max_align_mat = cluster_center_PCA;
	int start = 1;
	int luminance_threshold = 30;
	vector<int> scale_vector;
	//binary search the best scale & convell hull for speed up
	for(int t=0;t<move_vector.size();t++)
	{	
		for(int i=start;i<=150;i++)
			scale_vector.push_back(i);
		
		int low = start;
		int high = scale_vector.size();
		while(low <= high)
		{
			int mid = (low + high)/2; 
			Mat cluster_center_PCA_temp,cluster_center_PCA_weight,cluster_center_axis_invert;
			add(cluster_center_PCA_const,move_vector[t],cluster_center_PCA_temp); //move
			cluster_center_PCA_temp = cluster_center_PCA_temp.mul(mid); //scale
			cluster_center_axis_invert = cluster_center_axis.inv();
			cluster_center_PCA_weight = cluster_center_PCA_temp * cluster_center_axis_invert;
			align_mat = cluster_center_PCA_weight*lab_axis;

			//把重心平移回去
			for(int i=0;i<align_mat.rows;i++)
			{
				for(int j=0;j<3;j++)
				{
					align_mat.at<float>(i,j) += lab_centroid.at<float>(0,j);
				}
			}

			//cout << "align_mat " << align_mat << endl;
	
			bool flag = true;
			for(int i=0;i<align_mat.rows;i++)
			{
				if(align_mat.at<float>(i,0)<luminance_threshold)
				{
					flag = false;
					break;
				}
			}

			if(flag)
			{
				for(int i=0;i<align_mat.rows;i++)
				{
					if(lab_boundary_test(align_mat.at<float>(i,0),align_mat.at<float>(i,1),align_mat.at<float>(i,2))==false)
					{
						flag = false;
						break;
					}
				}
			}

			if(high<=low)
				break;
			else if(flag)
				low = mid + 1;
			else 
				high = mid - 1;

		}

		if(low>max_scale)
		{
			max_scale = low;
			max_move = move_vector[t];
			max_align_mat = align_mat.clone();;
			start = max_scale;
		}

		scale_vector.clear();
	}

	if(max_scale==0)
	{
		for(int i=0;i<max_align_mat.rows;i++)
		{
			for(int j=0;j<max_align_mat.cols;j++)
			{
				max_align_mat.at<float>(i,j) += lab_centroid.at<float>(0,j);
			}
		}
	}

	printf("max_move : %f max_scale : %f\n",max_move,max_scale);
	lab = max_align_mat.clone();
	Mat rgb_mat2 = LAB2RGB(max_align_mat).clone();

	return rgb_mat2;
}

Mat Preprocessing_Data::lab_alignment_dim2(Mat cluster_center)
{
	//read_lab_csv();
	int vTotal = lab_vertices.size();
	//Turn LAB vector into LAB mat
	Mat lab_mat = Mat::zeros(vTotal,3,CV_32F);
	for(int i=0;i<vTotal;i++)
	{
		for(int j=0;j<3;j++)
		{
			lab_mat.at<float>(i,j) = lab_vertices[i][j+1];
		}
	}
	//Compute centroid of cluster center
	Mat cluster_center_centroid = compute_centroid(cluster_center);

	//Align the centroid to the origin by subtract all points to centroid
	Mat cluster_center_alignment_mat = Mat::zeros(cluster_center.rows,cluster_center.cols,CV_32F);
	for(int i=0;i<cluster_center.rows;i++)
	{
		for(int j=0;j<cluster_center.cols;j++)
		{
			cluster_center_alignment_mat.at<float>(i,j) = cluster_center.at<float>(i,j) - cluster_center_centroid.at<float>(0,j);
		}
	}
	Mat cluster_center_component,cluster_center_PCA;
	int rDim = 2;
	reduceDimPCA(cluster_center_alignment_mat, rDim, cluster_center_component, cluster_center_PCA);
	//cluster center 3 axis of PCA
	Mat cluster_center_axis = Mat::zeros(2,2,CV_32F);
	for(int i=0;i<2;i++)
	{
		for(int j=0;j<2;j++)
		{
			cluster_center_axis.at<float>(i,j) = cluster_center_component.at<float>(i,j);
		}
	}

	//compute centroid of LAB
	Mat lab_centroid = compute_centroid(lab_mat);
	
	//lab vertices 3 axis of PCA
	Mat lab_components,lab_PCA;
	rDim = 2;
	reduceDimPCA(lab_mat, rDim, lab_components, lab_PCA);
	Mat lab_axis = Mat::zeros(2,3,CV_32F);
	for(int i=0;i<2;i++)
	{
		for(int j=0;j<3;j++)
		{
			lab_axis.at<float>(i,j) = lab_components.at<float>(i,j);
		}
	}
	
	//////////////////////////////////////////////////////////////////
	Mat cluster_center_PCA_const = cluster_center_PCA;
	vector<float> move_vector;
	for(float k=-0.5;k<=0.5;k+=0.1)
		move_vector.push_back(k);

	float max_move = 0.0;
	float max_scale = 0.0;
	Mat align_mat;
	Mat max_align_mat = cluster_center_PCA;
	int start = 1;
	int luminance_threshold = 30;
	vector<int> scale_vector;
	//binary search the best scale & convell hull for speed up
	for(int t=0;t<move_vector.size();t++)
	{	
		for(int i=start;i<=150;i++)
			scale_vector.push_back(i);
		
		int low = start;
		int high = scale_vector.size();
		while(low <= high)
		{
			int mid = (low + high)/2; 
			Mat cluster_center_PCA_temp,cluster_center_PCA_weight,cluster_center_axis_invert;
			add(cluster_center_PCA_const,move_vector[t],cluster_center_PCA_temp); //move
			cluster_center_PCA_temp = cluster_center_PCA_temp.mul(mid); //scale
			cluster_center_axis_invert = cluster_center_axis.inv();
			cluster_center_PCA_weight = cluster_center_PCA_temp * cluster_center_axis_invert;

			align_mat = cluster_center_PCA_weight*lab_axis;

			//把重心平移回去
			for(int i=0;i<align_mat.rows;i++)
			{
				for(int j=0;j<3;j++)
				{
					align_mat.at<float>(i,j) += lab_centroid.at<float>(0,j);
				}
			}

	
			bool flag = true;
			for(int i=0;i<align_mat.rows;i++)
			{
				if(align_mat.at<float>(i,0)<luminance_threshold)
				{
					flag = false;
					break;
				}
			}

			if(flag)
			{
				for(int i=0;i<align_mat.rows;i++)
				{
					if(lab_boundary_test(align_mat.at<float>(i,0),align_mat.at<float>(i,1),align_mat.at<float>(i,2))==false)
					{
						flag = false;
						break;
					}
				}
			}

			if(high<=low)
				break;
			else if(flag)
				low = mid + 1;
			else 
				high = mid - 1;

		}

		if(low>max_scale)
		{
			max_scale = low;
			max_move = move_vector[t];
			max_align_mat = align_mat.clone();
			start = max_scale;
		}

		scale_vector.clear();
	}

	if(max_scale==0)
	{
		for(int i=0;i<max_align_mat.rows;i++)
		{
			for(int j=0;j<max_align_mat.cols;j++)
			{
				max_align_mat.at<float>(i,j) += lab_centroid.at<float>(0,j);
			}
		}
	}

	printf("max_move : %f max_scale : %f\n",max_move,max_scale);
	lab = max_align_mat.clone();
	Mat rgb_mat2 = LAB2RGB(max_align_mat).clone();

	return rgb_mat2;
}

void Preprocessing_Data::gray2rgb(float gray,float& r,float& g,float& b)
{
	r = g = b = 0.0;
	if(gray>1.0)
	{
		r = 1.0;
		g = 0.0;
		b = 0.0;
	}
	if(gray<0.0)
	{
		r = 0.0;
		g = 0.0;
		b = 1.0;
	}

	if(gray<0.33333)
	{
		b = 1.0 - gray*3.0;
		g = gray*3.0;
	}
	else if(gray<0.66666)
	{
		r = (gray-0.33333)*3.0;
		g = 1.0;
	}
	else
	{
		r = 1.0;
		g = 1.0 - (gray-0.66666)*3.0;
	}
}

Mat Preprocessing_Data::normalize_column(Mat col_mat)
{
	Mat output_mat = col_mat.clone();
	double min,max;
	minMaxLoc(output_mat, &min, &max);
	for(int i=0;i<col_mat.rows;i++)
	{
		//output_mat.at<float>(i,0) = ( col_mat.at<float>(i,0) - min ) / (max - min);
		output_mat.at<float>(i,0) = col_mat.at<float>(i,0) / max;
	}

	return output_mat;
}

void Preprocessing_Data::Position_by_histogram_TSP(Mat& histo_position,Mat histo_sort_index)
{
	int k = histogram.cols;
	int five_minutes = histogram.rows;
	Mat histogram_copy = histogram.clone();
	Mat dist = Mat::zeros(1,five_minutes-1,CV_64F);

	output_mat_as_csv_file_int("histo_sort_index.csv",histo_sort_index);

	for(int i=0;i<five_minutes-1;i++)
	{
		int index = histo_sort_index.at<int>(i,0);
		int index2 = histo_sort_index.at<int>(i+1,0);
		double vote_dist = 0.0;
		for(int j=0;j<k;j++)
		{
			vote_dist += abs( histogram_copy.at<int>(index,j) - histogram_copy.at<int>(index2,j) );
		}

		dist.at<double>(0,i) = pow(vote_dist,5);
	}
	output_mat_as_csv_file_double("vote_dist.csv",dist.t());
	
	Mat histo_position_sort = Mat::zeros(five_minutes,1,CV_64F);
	histo_position_sort.at<double>(0,0) = 0.0;
	double accumulate_dist = 0.0;
	for(int i=0;i<five_minutes-1;i++)
	{
		accumulate_dist += dist.at<double>(0,i);
		histo_position_sort.at<double>(i+1,0) = accumulate_dist;
	}
	//output_mat_as_csv_file_double("histo_position_sort.csv",histo_position_sort);

	normalize(histo_position_sort.col(0),histo_position_sort.col(0),0,7000,NORM_MINMAX); 

	for(int i=0;i<five_minutes;i++)
	{
		int index = histo_sort_index.at<int>(i,0);
		histo_position.at<double>(index,0) = histo_position_sort.at<double>(i,0);
	}
}

void Preprocessing_Data::Position_by_histogram(Mat& histo_position, Mat cluster_center)//histo_position:double
{
	int k = histogram.cols;
	int five_minutes = histogram.rows;
	//Mat histogram_copy = histogram.clone();
	Mat histogram_sort = histogram.clone();
	
	Mat Ev = Mat::zeros(five_minutes,cluster_center.cols,CV_32F);
	for(int i=0;i<five_minutes;i++)
	{
		float base = 0;
		for(int j=0;j<k;j++)
		{
			Ev.row(i) += (histogram.at<int>(i,j)/600.0)*cluster_center.row(j);
		}
	}
	output_mat_as_csv_file("Ev.csv",Ev);	

	Mat components, coeff;
	int rDim = 1;
	reduceDimPCA(Ev, rDim, components, coeff);
	Mat Ev_PCA1D = coeff * components;

	class histo_info{
	public:
		int* vote;
		int cols;
		int key;
		Mat histo;
		Mat Ev_PCA1D;
	};
	vector< histo_info > histo_vec(five_minutes);
	for(int i=0;i<five_minutes;i++)
	{
		Ev_PCA1D.row(i).copyTo(histo_vec[i].Ev_PCA1D);
		histogram.row(i).copyTo(histo_vec[i].histo);
		histo_vec[i].key = i;
		histo_vec[i].cols = k;
		histo_vec[i].vote = new int[k];
		for(int j=0;j<k;j++)
		{
			histo_vec[i].vote[j] = histogram.at<int>(i,j);
		}
	}
	
	class sort_by_votes{
	public:
		inline bool operator() (histo_info& h1, histo_info& h2)
		{
			return h1.Ev_PCA1D.at<float>(0,0) < h2.Ev_PCA1D.at<float>(0,0);
		}
	};

	sort(histo_vec.begin(), histo_vec.end(), sort_by_votes() );
	

	//for(int i=0;i<five_minutes;i++)
	//	cout << histo_vec[i].key << " " ;
	//cout << endl;

	Mat dist = Mat::zeros(1,five_minutes-1,CV_64F);

	for(int i=0;i<five_minutes-1;i++)
	{
		double vote_dist = 0.0;
		for(int j=0;j<k;j++)
		{
			vote_dist += abs( histo_vec[i].histo.at<int>(0,j) - histo_vec[i+1].histo.at<int>(0,j) );
			//vote_dist += sqrt( histo_vec[i].histo.at<int>(0,j)/600.0 * histo_vec[i+1].histo.at<int>(0,j)/600.0 );
		}
		//vote_dist = -log( MAX(vote_dist,0.0000001) );
		dist.at<double>(0,i) = pow(vote_dist,8);
	}
	output_mat_as_csv_file_double("vote_dist.csv",dist.t());
	
	Mat histo_position_sort = Mat::zeros(five_minutes,1,CV_64F);
	histo_position_sort.at<double>(0,0) = 0.0;
	double accumulate_dist = 0.0;
	for(int i=0;i<five_minutes-1;i++)
	{
		accumulate_dist += dist.at<double>(0,i);
		histo_position_sort.at<double>(i+1,0) = accumulate_dist;
	}

	normalize(histo_position_sort.col(0),histo_position_sort.col(0),0,6000,NORM_MINMAX); 
	//cout << endl << dist << endl << endl;
	//cout << endl << histo_position << endl;

	for(int i=0;i<five_minutes;i++)
	{
		int key = histo_vec[i].key;
		histo_position.at<double>(key,0) = histo_position_sort.at<double>(i,0);
	}

}

string Preprocessing_Data::int2str(int i)
{
		stringstream ss;
		ss << i;
		return ss.str();
}

double Preprocessing_Data::CalculateDistance(CITY_INFO c1, CITY_INFO c2)
{
    double result = sqrt( (double)(c1.x - c2.x)*(c1.x - c2.x) + (c1.y - c2.y)*(c1.y - c2.y));
    return result;	
}

void Preprocessing_Data::TSP_Helper(vector<CITY_INFO> &arr, double *dist, string &path, CITY_INFO &firstpoint, CITY_INFO &lastpoint)
{
    int len = arr.size();
    if(len > 5)
        return; // Not Supported
 
    double mindist = INT_MAX;
    std::string str = "01234";
    str = str.substr(0, len);
    do
    {
        double distance = 0;
        for(int i = 0; i < str.size() - 1; i++)
        {
            distance += CalculateDistance(arr[str[i]-'0'], arr[str[i+1] - '0']);
        }
        if( mindist > distance)
        {
            mindist = distance;
            *dist = mindist;
           
            path = "";
			path_index_vec[path_index].clear();
            for(int i = 0; i < str.size(); i++)
            {
                path += arr[str[i]-'0'].cityname;
				//path_index[index].push_back( str2int(arr[str[i]-'0'].cityname) );
				path_index_vec[path_index].push_back( arr[str[i]-'0'].index );
            }
 
            firstpoint = arr[str[0]-'0'];
            lastpoint = arr[str[str.size() - 1]-'0'];           
        }
        //std::cout << *dist << "\t" << str.c_str() << "\n";
 
    } while(std::next_permutation(str.begin(), str.end()) != false);

	path_index++;	
}

bool Preprocessing_Data::SplitSet(const vector<CITY_INFO> &myset, vector< vector<CITY_INFO> > &mysplitset)
{
    // Construct a grid
 
    std::vector<CITY_INFO>::const_iterator it = myset.begin();
 
    double minx = it->x;
    double maxx = it->x;
    double miny = it->y;
    double maxy = it->y;
 
    for(; it != myset.end(); ++it)
    {
        if(minx >= it->x)
            minx = it->x;
        if(maxx < it->x)
            maxx = it->x;
 
        if(miny >= it->y)
            miny = it->y;
        if(maxy < it->y)
            maxy = it->y;
    }
    double width = maxx - minx;
    double height = maxy - miny;
    double midx = width / 2 + minx;
    double midy = height / 2 + miny;
 
   
    std::vector<CITY_INFO> s1, s2, s3, s4;
    std::vector<CITY_INFO> *pset[] = { &s1, &s2, &s3, &s4 };
 
    it = myset.begin();
    for(; it != myset.end(); ++it)
    {
        // First Grid
        if(it->x < midx && it->y < midy)
            s1.push_back(*it);       
    
        // Second Grid
        if(it->x >= midx && it->y < midy)
            s2.push_back(*it);
   
        // Third Grid
        if(it->x < midx && it->y >= midy)
            s3.push_back(*it);
       
        // Fourth Grid
        if(it->x >= midx && it->y >= midy)
            s4.push_back(*it);
    }
 
    for(int i = 0; i < 4; i++)
    {
        if(pset[i]->size() <= 5)
        {
            if(pset[i]->size() > 0)
                mysplitset.push_back(*pset[i]);
        }
        else
        {
            std::vector<std::vector<CITY_INFO> > tempset;
            SplitSet(*pset[i], tempset);
            for(std::vector<std::vector<CITY_INFO> >::iterator tit = tempset.begin();
                tit != tempset.end(); ++tit)
            {
                if(tit->size() > 0)
                    mysplitset.push_back(*tit);
            }
        }
    }
    return true;	
}

void Preprocessing_Data::TSP_Start(CITY_INFO *parr, int len, double *dist, std::string &finalpath)
{
    int NumCities = len;
    if(NumCities <= 5)
    {
        std::vector<CITY_INFO> myset;
        for(int i = 0; i < len; i++)
        {
            myset.push_back(parr[i]);
        }
        CITY_INFO firstpoint, lastpoint;
        TSP_Helper(myset, dist, finalpath, firstpoint, lastpoint);
    }
    else
    {
        std::vector<CITY_INFO> myset;
        for(int i = 0; i < len; i++)
        {
            myset.push_back(parr[i]);
        }
 
        std::vector<std::vector<CITY_INFO> > mysplitset;
        SplitSet(myset, mysplitset);
 
        double distance = 0;
        std::string result = "";
       
 
        finalpath = "";
        CITY_INFO firstpoint, lastpoint;
        for(int i = 0; i < mysplitset.size(); i++)
        {
            std::vector<CITY_INFO> current = mysplitset[i];
           
            if(i == 0)
            {
                double distSP = 0;
                std::string path;
                TSP_Helper(current, &distSP, path, firstpoint, lastpoint);
 
                //std::cout << "Path: "<< distSP << "\t" << path.c_str() << "\n";
                distance = distSP;
                finalpath = path;

            }
            else
            {
                double distSP = 0;
                std::string path;
                CITY_INFO fp, lp;
                TSP_Helper(current, &distSP, path, fp, lp);
 
                distance += distSP;
                finalpath += path;


                // Previous iteration last and current first point distance
                std::vector<CITY_INFO> prev = mysplitset[i - 1];
                // optimization required on this line!!!
                // Distance between the last point and the closet point the current circuit would be taken here instead of first point
                distance += CalculateDistance(lastpoint, fp);
                lastpoint = lp;
                firstpoint = fp;
 
                //std::cout << "Path: " << distance << "\t" << finalpath.c_str() << "\n";

                *dist = distance;
            }
        }
    }
}

void Preprocessing_Data::TSP_for_histogram(Mat cluster_center)
{
	int k = cluster_center.rows;
	int five_minutes = histogram.rows;
	Mat Ev = Mat::zeros(five_minutes,cluster_center.cols,CV_32F);
	for(int i=0;i<five_minutes;i++)
	{
		for(int j=0;j<k;j++)
		{
			Ev.row(i) += (histogram.at<int>(i,j)/600.0)*cluster_center.row(j);
		}
	}

	Mat component, coeff;
	int rDim = 2;
	reduceDimPCA(Ev, rDim, component, coeff);
	Mat Ev_PCA2D = coeff * component;
	
	//cout << Ev_PCA2D.rows << " " << Ev_PCA2D.cols << endl;
	path_index = 0; //initialize TSP path index 

	CITY_INFO polypoints[250];
	for(int i=0;i<30;i++)
	{
		polypoints[i].set_info(int2str(i),i, Ev_PCA2D.at<float>(i,0),  Ev_PCA2D.at<float>(i,1));
	}

	int NumCities = 30;
	path_index_vec.resize(1000);
    double dist = 0;
    string path;
	
	TSP_Start(polypoints, NumCities, &dist, path);

	for(int i=0;i<path_index_vec.size();i++)
	{
		for(int j=0;j<path_index_vec[i].size();j++)
			cout << path_index_vec[i][j] << " ";
	}
	cout << endl;

}

int myrandom (int i) { return std::rand()%i;}


double Preprocessing_Data::TSP_boost_by_EdgeWeight(Mat input_mat, Mat& sort_index, int start_index, int end_index)
{
	int row = input_mat.rows;

	typedef vector<simple_point<double> > PositionVec;
	typedef adjacency_matrix<undirectedS, no_property,
	property <edge_weight_t, double> > Graph;
	typedef graph_traits<Graph>::vertex_descriptor Vertex;
	typedef graph_traits<Graph>::edge_descriptor edge_descriptor;
	typedef vector<Vertex> Container;
	typedef property_map<Graph, edge_weight_t>::type WeightMap;
	typedef property_map<Graph, vertex_index_t>::type VertexMap;

	/*	
	PositionVec position_vec;
	for(int i=0;i<row;i++)
	{
		simple_point<double> vertex;
		vertex.x = input_mat.at<float>(i,0);
		vertex.y = input_mat.at<float>(i,1);
		position_vec.push_back(vertex);
	}	
	*/	

    int num_nodes = row + 1; //dummy node
    int num_arcs = num_nodes * (num_nodes-1) / 2;
	//int num_arcs = num_nodes * (num_nodes-1) / 2 - (num_nodes-1-2);
	typedef std::pair<int, int> Edge;
	//Edge edge_array[] = { Edge(A, C), Edge(B, B), Edge(B, D), Edge(B, E),Edge(C, B), Edge(C, D), Edge(D, E), Edge(E, A), Edge(E, B) };
	//int weights[] = { 1, 2, 1, 2, 7, 3, 1, 1, 1 };
	Edge* edge_array = new Edge[num_arcs];
	float* weights = new float[num_arcs];
	float check_weight[50][50];
	int t = 0;
	int big_num = 500;

	float start_end_dist = 0.0;
	for(int u=0;u<input_mat.cols;u++)
	{
		start_end_dist += abs( input_mat.at<float>(start_index,u) - input_mat.at<float>(end_index,u) ); 
	}
	cout << "start_end_dist " << start_end_dist << endl;

	for(int i=0;i<num_nodes;i++)
	{
		for(int j=0;j<num_nodes;j++)
		{
			if(i==j) 
				continue;
			else if(i>j) 
				continue;
			else
			{	
				if(i==start_index && j==num_nodes-1)
				{
					weights[t] = start_end_dist/2.0;
				}
				else if(i==end_index && j==num_nodes-1)
				{
					weights[t] =  start_end_dist/2.0+1.0;
				}
				else if(j==num_nodes-1)
				{
					//continue;
					if(i!=num_nodes-1) 
						weights[t] = big_num;
				}
				else
				{
					weights[t] = 0.0;
					for(int u=0;u<input_mat.cols;u++)
					{
						weights[t] += abs( input_mat.at<float>(i,u) - input_mat.at<float>(j,u) ); 
					}
				}
				edge_array[t] = Edge(i,j);
				check_weight[i][j] = weights[t];
				t++;
			}
		}
	}

	/*
	ofstream fout("tsp_9.txt");
	int zero = 0;
	for(int i=0;i<num_nodes;i++)
	{
		for(int j=0;j<num_nodes;j++)
		{
			if(i==j) 
				fout << zero << " ";
			else if(i>j)
				fout << check_weight[j][i] << " ";
			else
				fout << check_weight[i][j] << " ";
		}
		fout << endl;
	}

	fout.close();
	*/

	/*
	Graph g(edge_array, edge_array + num_arcs, weights, num_nodes);
	WeightMap weight_map(get(edge_weight, g));
	VertexMap v_map = get(vertex_index, g);
	*/
	Graph g(num_nodes);
	WeightMap weight_map(get(edge_weight, g));
	VertexMap v_map = get(vertex_index, g);
	for (size_t j = 0; j < num_arcs; ++j) {
		edge_descriptor e; 
		bool inserted;
		boost::tie(e, inserted) = add_edge(edge_array[j].first, edge_array[j].second, g);
		weight_map[e] = weights[j];
		//cout << "weight[(" << edge_array[j].first << "," << edge_array[j].second << ")] = " << get(weight_map, e) << endl;
	}

	//boost::graph_traits<Graph>::vertex_descriptor u, v;
	//u = vertex(0, g);
 //   v = vertex(1, g);
 //   boost::graph_traits<Graph>::edge_descriptor e1, e2;
 //   bool found;
 //   boost::tie(e1, found) = edge(u, v, g);
 //   boost::tie(e2, found) = edge(v, u, g);
 //   cout << "weight[(u,v)] = " << get(weight_map, e1) << endl;
 //   cout << "weight[(v,u)] = " << get(weight_map, e2) << endl;

	Container c;
	//connectAllEuclidean(g, position_vec, weight_map, v_map);

	metric_tsp_approx_tour(g, back_inserter(c));

	cout << "Number of points: " << num_vertices(g) << endl;
	cout << "Number of edges: " << num_edges(g) << endl;

	int i = 0;
	for (vector<Vertex>::iterator itr = c.begin(); itr != c.end(); ++itr)
	{
		//cout << *itr << " ";
		sort_index.at<int>(i,0) = *itr;
		i++;
		if(i==num_nodes) break;
	}

	for (vector<Vertex>::iterator itr = c.begin(); itr != c.end(); ++itr)
	{
		cout << *itr << " ";
	}

	cout << endl << endl;

	c.clear();

	//checkAdjList(position_vec);

	//metric_tsp_approx_from_vertex(g, *vertices(g).first,
	//	get(edge_weight, g), get(vertex_index, g),
	//	tsp_tour_visitor<back_insert_iterator<vector<Vertex> > >
	//	(back_inserter(c)));

	//for (vector<Vertex>::iterator itr = c.begin(); itr != c.end(); ++itr)
	//{
	//	cout << *itr << " ";
	//}
	//cout << endl << endl;

	//c.clear();
   
	double len(0.0);
	try {
		metric_tsp_approx(g, make_tsp_tour_len_visitor(g, back_inserter(c), len, weight_map));
	}
	catch (const bad_graph& e) {
		cerr << "bad_graph: " << e.what() << endl;
		//return;
	}

	//for (vector<Vertex>::iterator itr = c.begin(); itr != c.end(); ++itr)
	//{
	//	cout << *itr << " ";
	//}
	cout << "Number of points: " << num_vertices(g) << endl;
	cout << "Number of edges: " << num_edges(g) << endl;
	cout << "Length of Tour: " << len << endl;

	return len;
}

double Preprocessing_Data::TSP_boost_by_EdgeWeight(Mat input_mat, Mat& sort_index)
{
	int row = input_mat.rows;

	typedef vector<simple_point<double> > PositionVec;
	typedef adjacency_matrix<undirectedS, no_property,
	property <edge_weight_t, double> > Graph;
	typedef graph_traits<Graph>::vertex_descriptor Vertex;
	typedef graph_traits<Graph>::edge_descriptor edge_descriptor;
	typedef vector<Vertex> Container;
	typedef property_map<Graph, edge_weight_t>::type WeightMap;
	typedef property_map<Graph, vertex_index_t>::type VertexMap;

	/*	
	PositionVec position_vec;
	for(int i=0;i<row;i++)
	{
		simple_point<double> vertex;
		vertex.x = input_mat.at<float>(i,0);
		vertex.y = input_mat.at<float>(i,1);
		position_vec.push_back(vertex);
	}	*/	

	//cout << "position_vec size " << position_vec.size() << endl;

    int num_nodes = row;
    int num_arcs = row * (row-1) / 2;
	typedef std::pair<int, int> Edge;
	//Edge edge_array[] = { Edge(A, C), Edge(B, B), Edge(B, D), Edge(B, E),Edge(C, B), Edge(C, D), Edge(D, E), Edge(E, A), Edge(E, B) };
	//int weights[] = { 1, 2, 1, 2, 7, 3, 1, 1, 1 };
	Edge* edge_array = new Edge[num_arcs];
	float* weights = new float[num_arcs];
	int t = 0;
	for(int i=0;i<row;i++)
	{
		for(int j=0;j<row;j++)
		{
			if(i==j) 
				continue;
			else if(i>j) 
				continue;
			else
			{
				edge_array[t] = Edge(i,j);
				weights[t] = 0.0;
				for(int u=0;u<input_mat.cols;u++)
				{
					weights[t] += abs( input_mat.at<float>(i,u) - input_mat.at<float>(j,u) ); 
				}
				t++;
			}
		}
	}

	Graph g(num_nodes);
	WeightMap weight_map(get(edge_weight, g));
	VertexMap v_map = get(vertex_index, g);
	edge_descriptor e; 
	for (size_t j = 0; j < num_arcs; ++j) {
		bool inserted;
		boost::tie(e, inserted) = add_edge(edge_array[j].first, edge_array[j].second, g);
		weight_map[e] = weights[j];
		//cout << "weight[(" << edge_array[j].first << "," << edge_array[j].second << ")] = " << get(weight_map, e) << endl;
		//if(edge_array[j].second == num_nodes-1) cout << endl;
	}


	//Graph g(edge_array, edge_array + num_arcs, weights, num_nodes);

	Container c;

	//connectAllEuclidean(g, position_vec, weight_map, v_map);

	metric_tsp_approx_tour(g, back_inserter(c));

	int i = 0;
	for (vector<Vertex>::iterator itr = c.begin(); itr != c.end(); ++itr)
	{
		//cout << *itr << " ";
		sort_index.at<int>(i,0) = *itr;
		i++;
		if(i==row) break;
	}

	//cout << "i " << i << endl;
	cout << endl << endl;

	c.clear();

	//checkAdjList(position_vec);

	//metric_tsp_approx_from_vertex(g, *vertices(g).first,
	//	get(edge_weight, g), get(vertex_index, g),
	//	tsp_tour_visitor<back_insert_iterator<vector<Vertex> > >
	//	(back_inserter(c)));

	//for (vector<Vertex>::iterator itr = c.begin(); itr != c.end(); ++itr)
	//{
	//	cout << *itr << " ";
	//}
	//cout << endl << endl;

	//c.clear();
   
	double len(0.0);
	try {
		metric_tsp_approx(g, make_tsp_tour_len_visitor(g, back_inserter(c), len, weight_map));
	}
	catch (const bad_graph& e) {
		cerr << "bad_graph: " << e.what() << endl;
		//return;
	}

	cout << "Number of points: " << num_vertices(g) << endl;
	cout << "Number of edges: " << num_edges(g) << endl;
	cout << "Length of Tour: " << len << endl;

	return len;
}

double Preprocessing_Data::TSP_boost(Mat input_mat, Mat& sort_index)
{
	//int five_minutes = histogram.rows;
	int row = input_mat.rows;

	typedef vector<simple_point<double> > PositionVec;
	typedef adjacency_matrix<undirectedS, no_property,
	property <edge_weight_t, double> > Graph;
	typedef graph_traits<Graph>::vertex_descriptor Vertex;
	typedef vector<Vertex> Container;
	typedef property_map<Graph, edge_weight_t>::type WeightMap;
	typedef property_map<Graph, vertex_index_t>::type VertexMap;

	/*
	vector<int> random_index(row);
	for(int i=0;i<row;i++) random_index[i] = i;
	
	srand ( unsigned ( time(0) ) );
	random_shuffle ( random_index.begin(), random_index.end(), myrandom );
	for(int i=0;i<random_index.size();i++) cout << random_index[i] << " ";
	cout << endl;
	*/
	
	PositionVec position_vec;
	for(int i=0;i<row;i++)
	{
		simple_point<double> vertex;
		vertex.x = input_mat.at<float>(i,0);
		vertex.y = input_mat.at<float>(i,1);
		position_vec.push_back(vertex);
	}		

	//cout << "position_vec size " << position_vec.size() << endl;

	Container c;
	Graph g(position_vec.size());
	WeightMap weight_map(get(edge_weight, g));
	VertexMap v_map = get(vertex_index, g);

	connectAllEuclidean(g, position_vec, weight_map, v_map);

	metric_tsp_approx_tour(g, back_inserter(c));

	int i = 0;
	for (vector<Vertex>::iterator itr = c.begin(); itr != c.end(); ++itr)
	{
		//cout << *itr << " ";
		sort_index.at<int>(i,0) = *itr;
		i++;
		if(i==row) break;
	}

	//cout << "i " << i << endl;
	//cout << endl << endl;

	c.clear();

	//checkAdjList(position_vec);

	//metric_tsp_approx_from_vertex(g, *vertices(g).first,
	//	get(edge_weight, g), get(vertex_index, g),
	//	tsp_tour_visitor<back_insert_iterator<vector<Vertex> > >
	//	(back_inserter(c)));

	//for (vector<Vertex>::iterator itr = c.begin(); itr != c.end(); ++itr)
	//{
	//	cout << *itr << " ";
	//}
	//cout << endl << endl;

	//c.clear();
   
	double len(0.0);
	try {
		metric_tsp_approx(g, make_tsp_tour_len_visitor(g, back_inserter(c), len, weight_map));
	}
	catch (const bad_graph& e) {
		cerr << "bad_graph: " << e.what() << endl;
		//return;
	}

	cout << "Number of points: " << num_vertices(g) << endl;
	cout << "Number of edges: " << num_edges(g) << endl;
	cout << "Length of Tour: " << len << endl;

	return len;

}

double Preprocessing_Data::TSP_boost(Mat input_mat, Mat& sort_index, Mat& guess_index)
{
	//int five_minutes = histogram.rows;
	int row = input_mat.rows;

	typedef vector<simple_point<double> > PositionVec;
	typedef adjacency_matrix<undirectedS, no_property,
	property <edge_weight_t, double> > Graph;
	typedef graph_traits<Graph>::vertex_descriptor Vertex;
	typedef vector<Vertex> Container;
	typedef property_map<Graph, edge_weight_t>::type WeightMap;
	typedef property_map<Graph, vertex_index_t>::type VertexMap;

	/*
	vector<int> random_index(row);
	for(int i=0;i<row;i++) random_index[i] = i;
	
	srand ( unsigned ( time(0) ) );
	random_shuffle ( random_index.begin(), random_index.end(), myrandom );
	for(int i=0;i<random_index.size();i++) cout << random_index[i] << " ";
	cout << endl;
	*/
	
	PositionVec position_vec;
	for(int i=0;i<row;i++)
	{
		simple_point<double> vertex;
		vertex.x = input_mat.at<float>(guess_index.at<int>(i,0),0);
		vertex.y = input_mat.at<float>(guess_index.at<int>(i,0),1);
		position_vec.push_back(vertex);
	}		

	//cout << "position_vec size " << position_vec.size() << endl;

	Container c;
	Graph g(position_vec.size());
	WeightMap weight_map(get(edge_weight, g));
	VertexMap v_map = get(vertex_index, g);

	connectAllEuclidean(g, position_vec, weight_map, v_map);

	metric_tsp_approx_tour(g, back_inserter(c));

	int i = 0;
	for (vector<Vertex>::iterator itr = c.begin(); itr != c.end(); ++itr)
	{
		cout << *itr << " ";
		sort_index.at<int>(i,0) = *itr;
		i++;
		if(i==row) break;
	}

	//cout << "i " << i << endl;
	//cout << endl << endl;

	c.clear();

	//checkAdjList(position_vec);

	//metric_tsp_approx_from_vertex(g, *vertices(g).first,
	//	get(edge_weight, g), get(vertex_index, g),
	//	tsp_tour_visitor<back_insert_iterator<vector<Vertex> > >
	//	(back_inserter(c)));

	//for (vector<Vertex>::iterator itr = c.begin(); itr != c.end(); ++itr)
	//{
	//	cout << *itr << " ";
	//}
	//cout << endl << endl;

	//c.clear();
   
	double len(0.0);
	try {
		metric_tsp_approx(g, make_tsp_tour_len_visitor(g, back_inserter(c), len, weight_map));
	}
	catch (const bad_graph& e) {
		cerr << "bad_graph: " << e.what() << endl;
		//return;
	}

	cout << "Number of points: " << num_vertices(g) << endl;
	cout << "Number of edges: " << num_edges(g) << endl;
	cout << "Length of Tour: " << len << endl;

	return len;

}

void Preprocessing_Data::TSP_boost_for_histogram(Mat cluster_center, Mat& histo_sort_index)
{
	int k = cluster_center.rows;
	int five_minutes = histogram.rows;
	Mat Ev = Mat::zeros(five_minutes,cluster_center.cols,CV_32F);
	for(int i=0;i<five_minutes;i++)
	{
		for(int j=0;j<k;j++)
		{
			Ev.row(i) += (histogram.at<int>(i,j)/600.0)*cluster_center.row(j);
		}
	}

	Mat component, coeff;
	int rDim = 2;
	reduceDimPCA(Ev, rDim, component, coeff);
	Mat Ev_PCA2D = coeff * component;

	double min_tour_len = 1000;
	Mat histo_sort_index_copy = histo_sort_index.clone();
	Mat guess_index = Mat::zeros(five_minutes,1,CV_32S);
	for(int i=0;i<five_minutes;i++)
		guess_index.at<int>(i,0) = i;
	//////////////////////////////////////////
	vector<int> random_index(five_minutes);
	for(int i=0;i<five_minutes;i++) random_index[i] = i;
	
	for(int i=0;i<20;i++)
	{
		double tour_len = TSP_boost(Ev_PCA2D, histo_sort_index_copy,guess_index);
		//guess_index = histo_sort_index_copy.clone();
		srand ( unsigned ( time(0) ) );
		random_shuffle ( random_index.begin(), random_index.end(), myrandom );
		for(int i=0;i<five_minutes;i++) guess_index.at<int>(i,0) = random_index[i];
		//histo_sort_index = histo_sort_index_copy.clone();
		if(tour_len <= min_tour_len)
		{
			min_tour_len = tour_len;
			histo_sort_index = histo_sort_index_copy.clone();
		}
	}
	cout << "min_tour_len " << min_tour_len << endl;
	
}

void Preprocessing_Data::TSP_boost_for_histogram_coarse_to_fine(Mat cluster_center, Mat& histo_sort_index)
{
	int k = cluster_center.rows;
	int five_minutes = histogram.rows; //k是?的倍數
	int dim = histogram.rows;

	Mat Ev = Mat::zeros(five_minutes,cluster_center.cols,CV_32F);
	for(int i=0;i<five_minutes;i++)
	{
		for(int j=0;j<k;j++)
		{
			Ev.row(i) += (histogram.at<int>(i,j)/600.0)*cluster_center.row(j);
		}
	}

	int group_num = 4;
	Mat cluster_tag = Mat::zeros(five_minutes,1,CV_32S);
	Mat cluster_centers = Mat::zeros(group_num,dim,CV_32F);
	cuda_kmeans(Ev, group_num, cluster_tag, cluster_centers);
	class group{
	public:
		vector<int> index;
	};
	vector<group> groups(group_num);

	for(int i=0;i<five_minutes;i++)
	{
		int tag = cluster_tag.at<int>(i,0);
		groups[tag].index.push_back(i);
	}

	//Mat component, coeff;
	//int rDim = 2;
	//reduceDimPCA(Ev, rDim, component, coeff);
	//Mat Ev_PCA2D = coeff * component;

	int t = 0;
	for(int i=0;i<group_num;i++)
	{
		Mat Ev_sub = Mat::zeros(groups[i].index.size(),3,CV_32F);
		Mat histo_sort_index_sub = Mat::zeros(groups[i].index.size(),1,CV_32S);
		for(int j=0;j<groups[i].index.size();j++)
		{
			int index = groups[i].index[j];
			//Ev_sub.at<float>(j,0) = Ev.at<float>(index,0);	
			//Ev_sub.at<float>(j,1) = Ev.at<float>(index,1);	
			Ev.row(index).copyTo( Ev_sub.row(j) );
		}
		TSP_boost_by_EdgeWeight(Ev_sub, histo_sort_index_sub);
		for(int j=0;j<groups[i].index.size();j++)
		{
			int index = histo_sort_index_sub.at<int>(j,0);
			histo_sort_index.at<int>(t,0) = groups[i].index[index];
			t++;
		}
	}

}

void Preprocessing_Data::TSP_boost_for_histogram_coarse_to_fine2(Mat cluster_center, Mat& histo_sort_index)
{
	int k = cluster_center.rows;
	int five_minutes = histogram.rows; //k是?的倍數
	int dim = histogram.rows;
	
	Mat Ev = Mat::zeros(five_minutes,cluster_center.cols,CV_32F);
	for(int i=0;i<five_minutes;i++)
	{
		for(int j=0;j<k;j++)
		{
			Ev.row(i) += (histogram.at<int>(i,j)/600.0)*cluster_center.row(j);
		}
	}
	int group_num = 4;
	Mat cluster_tag = Mat::zeros(five_minutes,1,CV_32S);
	Mat cluster_centers = Mat::zeros(group_num,dim,CV_32F);
	cuda_kmeans(Ev, group_num, cluster_tag, cluster_centers);
	//kmeans(Ev, group_num, cluster_tag, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.0001), 2,KMEANS_PP_CENTERS,cluster_centers);
	class group{
	public:
		vector<int> index;
		vector<int> sort_index;
		vector<int> adj_index;
	};
	vector<group> groups(group_num);

	for(int i=0;i<five_minutes;i++)
	{
		int tag = cluster_tag.at<int>(i,0);
		groups[tag].index.push_back(i);
	}

 //	for(int i=0;i<group_num;i++)
	//{
	//	for(int j=0;j<groups[i].index.size();j++)
	//	{
	//		cout << groups[i].index[j] << " ";
	//	}
	//	cout << endl;
	//}

	Mat component, coeff;
	int rDim = 2;
	reduceDimPCA(Ev, rDim, component, coeff);
	Mat Ev_PCA2D = coeff * component;

	//int t = 0;
	for(int i=0;i<group_num;i++)
	{
		Mat Ev_PCA2D_sub = Mat::zeros(groups[i].index.size(),2,CV_32F);
		Mat histo_sort_index_sub = Mat::zeros(groups[i].index.size(),1,CV_32S);
		for(int j=0;j<groups[i].index.size();j++)
		{
			int index = groups[i].index[j];
			Ev_PCA2D_sub.at<float>(j,0) = Ev_PCA2D.at<float>(index,0);	
			Ev_PCA2D_sub.at<float>(j,1) = Ev_PCA2D.at<float>(index,1);	
		}
		TSP_boost_by_EdgeWeight(Ev_PCA2D_sub, histo_sort_index_sub);
		//cout << "lab_color_sort_index_sub " << lab_color_sort_index_sub << endl;
		for(int j=0;j<groups[i].index.size();j++)
		{
			int index = histo_sort_index_sub.at<int>(j,0);
			groups[i].sort_index.push_back(groups[i].index[index]);
			//lab_color_sort_index.at<int>(t,0) = groups[i].index[index];
			//t++;
		}
	}

	cout << "sort index: " << endl;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].sort_index.size();j++)
		{
			if(i==0)
			{
				groups[i].adj_index.push_back(groups[i].sort_index[j]);
			}
			cout << groups[i].sort_index[j] << " ";
		}
		cout << endl;
	}

	for(int s=1;s<group_num;s++)
	{
		bool flag = false;
		for(int i=0;i<groups[s].sort_index.size();i++)
		{
			Mat Ev_PCA2D_sub = Mat::zeros(groups[s-1].sort_index.size()+1,2,CV_32F);
			Mat histo_sort_index_sub = Mat::zeros(groups[s-1].sort_index.size()+1,1,CV_32S);
			for(int j=0;j<groups[s-1].sort_index.size();j++)
			{
				int index = groups[s-1].sort_index[j];
				Ev_PCA2D_sub.at<float>(j,0) = Ev_PCA2D.at<float>(index,0);	
				Ev_PCA2D_sub.at<float>(j,1) = Ev_PCA2D.at<float>(index,1);	
			}
			int try_index = groups[s].sort_index[i];
			Ev_PCA2D_sub.at<float>(groups[s-1].sort_index.size(),0) = Ev_PCA2D.at<float>(try_index,0);
			TSP_boost_by_EdgeWeight(Ev_PCA2D_sub, histo_sort_index_sub);
			//cout << "lab_color_sort_index_sub " << lab_color_sort_index_sub << endl;
		
			if( histo_sort_index_sub.at<int>(groups[s-1].sort_index.size(),0) == groups[s-1].sort_index.size() )
			{
				flag = true;
				for(int t=i;t<groups[s].sort_index.size();t++)
				{
					groups[s].adj_index.push_back( groups[s].sort_index[t] );
					//cout << groups[s].sort_index[t] << " ";
				}
			
				for(int t=0;t<i;t++)
				{
					groups[s].adj_index.push_back( groups[s].sort_index[t] );
					//cout << groups[s].sort_index[t] << " ";
				}
				break;
			}
		}
		if(flag==false)
		{
			for(int t=0;t<groups[s].sort_index.size();t++)
			{
				groups[s].adj_index.push_back( groups[s].sort_index[t] );
				//cout << groups[s].sort_index[t] << " ";
			}
		}
		//cout << endl;
	}

	cout << "adj index: " << endl;
	int t = 0;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].adj_index.size();j++)
		{
			cout << groups[i].adj_index[j] << " ";
			histo_sort_index.at<int>(t,0) = groups[i].adj_index[j];
			t++;
		}
		cout << endl;
	}

}

void Preprocessing_Data::TSP_boost_for_histogram_coarse_to_fine3(Mat cluster_center, Mat& histo_sort_index)
{
	int k = cluster_center.rows;
	int five_minutes = histogram.rows; //k是?的倍數
	int dim = histogram.rows;
	
	Mat Ev = Mat::zeros(five_minutes,cluster_center.cols,CV_32F);
	for(int i=0;i<five_minutes;i++)
	{
		for(int j=0;j<k;j++)
		{
			Ev.row(i) += (histogram.at<int>(i,j)/600.0)*cluster_center.row(j);
		}
	}
	int group_num = 6;
	Mat cluster_tag = Mat::zeros(five_minutes,1,CV_32S);
	Mat group_cluster_centers = Mat::zeros(group_num,dim,CV_32F);
	cuda_kmeans(Ev, group_num, cluster_tag, group_cluster_centers);
	//kmeans(Ev, group_num, cluster_tag, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.0001), 2,KMEANS_PP_CENTERS,cluster_centers);
	class group{
	public:
		vector<int> index;
		vector<int> index2;
		vector<int> sort_index;
		//vector<int> adj_index;
	};
	vector<group> groups(group_num);

	for(int i=0;i<five_minutes;i++)
	{
		int tag = cluster_tag.at<int>(i,0);
		groups[tag].index.push_back(i);
	}


	cout << "index " << endl;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].index.size();j++)
		{
			cout << groups[i].index[j] << " ";
		}
		cout << endl;
	}

	Mat component, coeff;
	int rDim = 2;
	reduceDimPCA(group_cluster_centers, rDim, component, coeff);
	Mat group_cluster_center2D_ndim = coeff * component;
	Mat group_cluster_center2D = Mat::zeros(group_num,2,CV_32F);
	for(int i=0;i<rDim;i++)
	{
		group_cluster_center2D_ndim.col(i).copyTo( group_cluster_center2D.col(i) );
	}
	Mat group_sort_index = Mat::zeros(group_num,1,CV_32S);
	TSP_path(group_cluster_center2D_ndim, group_sort_index);
	//TSP_boost(group_cluster_center2D, group_sort_index);

	for(int i=0;i<group_num;i++)
	{
		int index = group_sort_index.at<int>(i,0);
		for(int j=0;j<groups[index].index.size();j++)
		{
			groups[i].index2.push_back( groups[index].index[j] );
		}
	}

	cout << "index2 :" << endl;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].index2.size();j++)
		{
			cout << groups[i].index2[j] << " ";
		}
		cout << endl;
	}

	//Mat component, coeff;
	rDim = 2;
	reduceDimPCA(Ev, rDim, component, coeff);
	Mat Ev_PCA2D = coeff * component;

	//int t = 0;
	for(int i=0;i<group_num;i++)
	{
		Mat Ev_PCA2D_sub = Mat::zeros(groups[i].index.size(),2,CV_32F);
		Mat histo_sort_index_sub = Mat::zeros(groups[i].index.size(),1,CV_32S);
		for(int j=0;j<groups[i].index.size();j++)
		{
			int index = groups[i].index[j];
			Ev_PCA2D_sub.at<float>(j,0) = Ev_PCA2D.at<float>(index,0);	
			Ev_PCA2D_sub.at<float>(j,1) = Ev_PCA2D.at<float>(index,1);	
		}
		TSP_boost(Ev_PCA2D_sub, histo_sort_index_sub);
		//cout << "lab_color_sort_index_sub " << lab_color_sort_index_sub << endl;
		for(int j=0;j<groups[i].index.size();j++)
		{
			int index = histo_sort_index_sub.at<int>(j,0);
			groups[i].sort_index.push_back(groups[i].index[index]);
			//lab_color_sort_index.at<int>(t,0) = groups[i].index[index];
			//t++;
		}
	}

	cout << "sort index: " << endl;
	int t = 0;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].sort_index.size();j++)
		{
			histo_sort_index.at<int>(t,0) = groups[i].sort_index[j];
			t++;
			//if(i==0)
			//{
			//	//groups[i].adj_index.push_back(groups[i].sort_index[j]);
			//}
			cout << groups[i].sort_index[j] << " ";
		}
		cout << endl;
	}

	/*
	for(int s=1;s<group_num;s++)
	{
		bool flag = false;
		for(int i=0;i<groups[s].sort_index.size();i++)
		{
			Mat Ev_PCA2D_sub = Mat::zeros(groups[s-1].sort_index.size()+1,2,CV_32F);
			Mat histo_sort_index_sub = Mat::zeros(groups[s-1].sort_index.size()+1,1,CV_32S);
			for(int j=0;j<groups[s-1].sort_index.size();j++)
			{
				int index = groups[s-1].sort_index[j];
				Ev_PCA2D_sub.at<float>(j,0) = Ev_PCA2D.at<float>(index,0);	
				Ev_PCA2D_sub.at<float>(j,1) = Ev_PCA2D.at<float>(index,1);	
			}
			int try_index = groups[s].sort_index[i];
			Ev_PCA2D_sub.at<float>(groups[s-1].sort_index.size(),0) = Ev_PCA2D.at<float>(try_index,0);
			TSP_boost(Ev_PCA2D_sub, histo_sort_index_sub);
			//cout << "lab_color_sort_index_sub " << lab_color_sort_index_sub << endl;
		
			if( histo_sort_index_sub.at<int>(groups[s-1].sort_index.size(),0) == groups[s-1].sort_index.size() )
			{
				flag = true;
				for(int t=i;t<groups[s].sort_index.size();t++)
				{
					groups[s].adj_index.push_back( groups[s].sort_index[t] );
					//cout << groups[s].sort_index[t] << " ";
				}
			
				for(int t=0;t<i;t++)
				{
					groups[s].adj_index.push_back( groups[s].sort_index[t] );
					//cout << groups[s].sort_index[t] << " ";
				}
				break;
			}
		}
		if(flag==false)
		{
			for(int t=0;t<groups[s].sort_index.size();t++)
			{
				groups[s].adj_index.push_back( groups[s].sort_index[t] );
				//cout << groups[s].sort_index[t] << " ";
			}
		}
		//cout << endl;
	}
	
	cout << "adj index: " << endl;
	int t = 0;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].adj_index.size();j++)
		{
			cout << groups[i].adj_index[j] << " ";
			histo_sort_index.at<int>(t,0) = groups[i].adj_index[j];
			t++;
		}
		cout << endl;
	}
	*/
}

void Preprocessing_Data::TSP_boost_for_lab_color(Mat cluster_center, Mat& lab_color_sort_index)
{
	int k = cluster_center.rows;

	Mat component, coeff;
	int rDim = 2;
	reduceDimPCA(lab, rDim, component, coeff);
	Mat lab_PCA2D = coeff * component;

	TSP_boost(lab_PCA2D, lab_color_sort_index);
}

void Preprocessing_Data::TSP_boost_for_lab_color_coarse_to_fine(Mat lab_data, Mat& lab_color_sort_index)
{
	int k = lab_data.rows;//k是4的倍數
	int dim = lab_data.rows;
	//int base = 4;
	//int group_num = k/base;
	int group_num = 3;
	Mat cluster_tag = Mat::zeros(k,1,CV_32S);
	Mat cluster_centers = Mat::zeros(group_num,dim,CV_32F);
	//cuda_kmeans(lab_data, group_num, cluster_tag, cluster_centers);
	kmeans(lab_data, group_num, cluster_tag, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.0001), 2,KMEANS_PP_CENTERS,cluster_centers);
	class group{
	public:
		vector<int> index;
	};
	vector<group> groups(group_num);

	for(int i=0;i<k;i++)
	{
		int tag = cluster_tag.at<int>(i,0);
		groups[tag].index.push_back(i);
	}

 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].index.size();j++)
		{
			cout << groups[i].index[j] << " ";
		}
		cout << endl;
	}

	Mat component, coeff;
	int rDim = 2;
	reduceDimPCA(lab, rDim, component, coeff);
	Mat lab_PCA2D = coeff * component;

	int t = 0;
	for(int i=0;i<group_num;i++)
	{
		Mat lab_PCA2D_sub = Mat::zeros(groups[i].index.size(),2,CV_32F);
		Mat lab_color_sort_index_sub = Mat::zeros(groups[i].index.size(),1,CV_32S);
		for(int j=0;j<groups[i].index.size();j++)
		{
			int index = groups[i].index[j];
			//lab_PCA2D.row(index).copyTo( lab_PCA2D_sub.row(j) );
			lab_PCA2D_sub.at<float>(j,0) = lab_PCA2D.at<float>(index,0);	
			lab_PCA2D_sub.at<float>(j,1) = lab_PCA2D.at<float>(index,1);	
		}
		//TSP_boost(lab_PCA2D_sub, lab_color_sort_index_sub);
		double tour_len = TSP_path(lab_PCA2D_sub, lab_color_sort_index_sub);
		cout << "tour_len: " << tour_len << endl;
		for(int j=0;j<groups[i].index.size();j++)
		{
			int index = lab_color_sort_index_sub.at<int>(j,0);
			lab_color_sort_index.at<int>(t,0) = groups[i].index[index];
			t++;
		}
	}
}

void Preprocessing_Data::TSP_boost_for_lab_color_coarse_to_fine2(Mat lab_data, Mat& lab_color_sort_index)
{
	int k = lab_data.rows;//k是4的倍數
	int dim = lab_data.rows;
	int group_num = 3;
	Mat cluster_tag = Mat::zeros(k,1,CV_32S);
	Mat cluster_centers = Mat::zeros(group_num,dim,CV_32F);
	//cuda_kmeans(lab_data, group_num, cluster_tag, cluster_centers);
	kmeans(lab_data, group_num, cluster_tag, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.0001), 2,KMEANS_PP_CENTERS,cluster_centers);
	class group{
	public:
		vector<int> index;
		vector<int> sort_index;
		vector<int> adj_index;
	};
	vector<group> groups(group_num);

	for(int i=0;i<k;i++)
	{
		int tag = cluster_tag.at<int>(i,0);
		groups[tag].index.push_back(i);
	}

 //	for(int i=0;i<group_num;i++)
	//{
	//	for(int j=0;j<groups[i].index.size();j++)
	//	{
	//		cout << groups[i].index[j] << " ";
	//	}
	//	cout << endl;
	//}

	Mat component, coeff;
	int rDim = 2;
	reduceDimPCA(lab, rDim, component, coeff);
	Mat lab_PCA2D = coeff * component;

	//int t = 0;
	for(int i=0;i<group_num;i++)
	{
		Mat lab_PCA2D_sub = Mat::zeros(groups[i].index.size(),2,CV_32F);
		Mat lab_color_sort_index_sub = Mat::zeros(groups[i].index.size(),1,CV_32S);
		for(int j=0;j<groups[i].index.size();j++)
		{
			int index = groups[i].index[j];
			lab_PCA2D_sub.at<float>(j,0) = lab_PCA2D.at<float>(index,0);	
			lab_PCA2D_sub.at<float>(j,1) = lab_PCA2D.at<float>(index,1);	
		}
		TSP_boost(lab_PCA2D_sub, lab_color_sort_index_sub);
		//cout << "lab_color_sort_index_sub " << lab_color_sort_index_sub << endl;
		for(int j=0;j<groups[i].index.size();j++)
		{
			int index = lab_color_sort_index_sub.at<int>(j,0);
			groups[i].sort_index.push_back(groups[i].index[index]);
			//lab_color_sort_index.at<int>(t,0) = groups[i].index[index];
			//t++;
		}
	}

	cout << "sort index: " << endl;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].sort_index.size();j++)
		{
			if(i==0)
			{
				groups[i].adj_index.push_back(groups[i].sort_index[j]);
			}
			cout << groups[i].sort_index[j] << " ";
		}
		cout << endl;
	}

	for(int s=1;s<group_num;s++)
	{
		bool flag = false;
		for(int i=0;i<groups[s].sort_index.size();i++)
		{
			Mat lab_PCA2D_sub = Mat::zeros(groups[s-1].sort_index.size()+1,2,CV_32F);
			Mat lab_color_sort_index_sub = Mat::zeros(groups[s-1].sort_index.size()+1,1,CV_32S);
			for(int j=0;j<groups[s-1].sort_index.size();j++)
			{
				int index = groups[s-1].sort_index[j];
				lab_PCA2D_sub.at<float>(j,0) = lab_PCA2D.at<float>(index,0);	
				lab_PCA2D_sub.at<float>(j,1) = lab_PCA2D.at<float>(index,1);	
			}
			int try_index = groups[s].sort_index[i];
			lab_PCA2D_sub.at<float>(groups[s-1].sort_index.size(),0) = lab_PCA2D.at<float>(try_index,0);
			TSP_boost(lab_PCA2D_sub, lab_color_sort_index_sub);
			//cout << "lab_color_sort_index_sub " << lab_color_sort_index_sub << endl;
		
			if( lab_color_sort_index_sub.at<int>(groups[s-1].sort_index.size(),0) == groups[s-1].sort_index.size() )
			{
				flag = true;
				for(int t=i;t<groups[s].sort_index.size();t++)
				{
					groups[s].adj_index.push_back( groups[s].sort_index[t] );
					//cout << groups[s].sort_index[t] << " ";
				}
			
				for(int t=0;t<i;t++)
				{
					groups[s].adj_index.push_back( groups[s].sort_index[t] );
					//cout << groups[s].sort_index[t] << " ";
				}
				break;
			}
		}
		if(flag==false)
		{
			for(int t=0;t<groups[s].sort_index.size();t++)
			{
				groups[s].adj_index.push_back( groups[s].sort_index[t] );
				//cout << groups[s].sort_index[t] << " ";
			}
		}
		//cout << endl;
	}

	cout << "adj index: " << endl;
	int t = 0;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].adj_index.size();j++)
		{
			cout << groups[i].adj_index[j] << " ";
			lab_color_sort_index.at<int>(t,0) = groups[i].adj_index[j];
			t++;
		}
		cout << endl;
	}

}

void Preprocessing_Data::TSP_boost_for_lab_color_coarse_to_fine3(Mat lab_data, Mat& lab_color_sort_index)
{
	int k = lab_data.rows;//k是4的倍數
	int dim = lab_data.rows;
	int group_num = 5;
	Mat cluster_tag = Mat::zeros(k,1,CV_32S);
	Mat cluster_centers = Mat::zeros(group_num,dim,CV_32F);
	//cuda_kmeans(lab_data, group_num, cluster_tag, cluster_centers);
	kmeans(lab_data, group_num, cluster_tag, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.0001), 2, KMEANS_PP_CENTERS,cluster_centers);
	class group{
	public:
		vector<int> index;
		vector<int> index2;
		vector<int> sort_index;
		vector<int> adj_index;
	};
	vector<group> groups(group_num);

	for(int i=0;i<k;i++)
	{
		int tag = cluster_tag.at<int>(i,0);
		groups[tag].index.push_back(i);
	}

	cout << "index :" << endl;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].index.size();j++)
		{
			cout << groups[i].index[j] << " ";
		}
		cout << endl;
	}

	Mat component, coeff;
	int rDim = 2;
	reduceDimPCA(cluster_centers, rDim, component, coeff);
	Mat lab_cluster_center2D_ndim = coeff * component;
	Mat lab_cluster_center2D = Mat::zeros(group_num,2,CV_32F);
	for(int i=0;i<rDim;i++)
	{
		lab_cluster_center2D_ndim.col(i).copyTo( lab_cluster_center2D.col(i) );
	}
	Mat color_sort_index = Mat::zeros(group_num,1,CV_32S);
	//TSP_boost(lab_cluster_center2D, color_sort_index);
	TSP_path(lab_cluster_center2D, color_sort_index);

	for(int i=0;i<group_num;i++)
	{
		int index = color_sort_index.at<int>(i,0);
		for(int j=0;j<groups[index].index.size();j++)
		{
			groups[i].index2.push_back( groups[index].index[j] );
		}
	}

	cout << "index2 :" << endl;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].index2.size();j++)
		{
			cout << groups[i].index2[j] << " ";
		}
		cout << endl;
	}

	//Mat component, coeff;
	rDim = 2;
	reduceDimPCA(lab, rDim, component, coeff);
	Mat lab_PCA2D = coeff * component;

	//int t = 0;
	for(int i=0;i<group_num;i++)
	{
		Mat lab_PCA2D_sub = Mat::zeros(groups[i].index2.size(),2,CV_32F);
		Mat lab_color_sort_index_sub = Mat::zeros(groups[i].index2.size(),1,CV_32S);
		for(int j=0;j<groups[i].index2.size();j++)
		{
			int index = groups[i].index2[j];
			lab_PCA2D_sub.at<float>(j,0) = lab_PCA2D.at<float>(index,0);	
			lab_PCA2D_sub.at<float>(j,1) = lab_PCA2D.at<float>(index,1);	
		}
		//TSP_boost(lab_PCA2D_sub, lab_color_sort_index_sub);
		TSP_path(lab_PCA2D_sub, lab_color_sort_index_sub);
		for(int j=0;j<groups[i].index2.size();j++)
		{
			int index = lab_color_sort_index_sub.at<int>(j,0);
			groups[i].sort_index.push_back(groups[i].index2[index]);
			//lab_color_sort_index.at<int>(t,0) = groups[i].index2[index];
			//t++;
		}
	}

	cout << "sort index: " << endl;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].sort_index.size();j++)
		{
			if(i==0)
			{
				groups[i].adj_index.push_back(groups[i].sort_index[j]);
			}
			cout << groups[i].sort_index[j] << " ";
		}
		cout << endl;
	}

	for(int s=1;s<group_num;s++)
	{
		bool flag = false;
		for(int i=0;i<groups[s].sort_index.size();i++)
		{
			Mat lab_PCA2D_sub = Mat::zeros(groups[s-1].sort_index.size()+1,2,CV_32F);
			Mat lab_color_sort_index_sub = Mat::zeros(groups[s-1].sort_index.size()+1,1,CV_32S);
			for(int j=0;j<groups[s-1].sort_index.size();j++)
			{
				int index = groups[s-1].sort_index[j];
				lab_PCA2D_sub.at<float>(j,0) = lab_PCA2D.at<float>(index,0);	
				lab_PCA2D_sub.at<float>(j,1) = lab_PCA2D.at<float>(index,1);	
			}
			int try_index = groups[s].sort_index[i];
			lab_PCA2D_sub.at<float>(groups[s-1].sort_index.size(),0) = lab_PCA2D.at<float>(try_index,0);
			lab_PCA2D_sub.at<float>(groups[s-1].sort_index.size(),1) = lab_PCA2D.at<float>(try_index,1);
			TSP_boost(lab_PCA2D_sub, lab_color_sort_index_sub);
			//TSP_path(lab_PCA2D_sub, lab_color_sort_index_sub);
		
			if( lab_color_sort_index_sub.at<int>(groups[s-1].sort_index.size(),0) == groups[s-1].sort_index.size() )
			{
				flag = true;
				for(int t=i;t<groups[s].sort_index.size();t++)
				{
					groups[s].adj_index.push_back( groups[s].sort_index[t] );
					//cout << groups[s].sort_index[t] << " ";
				}
			
				for(int t=0;t<i;t++)
				{
					groups[s].adj_index.push_back( groups[s].sort_index[t] );
					//cout << groups[s].sort_index[t] << " ";
				}
				break;
			}
		}
		if(flag==false)
		{
			for(int t=0;t<groups[s].sort_index.size();t++)
			{
				groups[s].adj_index.push_back( groups[s].sort_index[t] );
				//cout << groups[s].sort_index[t] << " ";
			}
		}
		//cout << endl;
	}

	cout << "adj index: " << endl;
	int t = 0;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].adj_index.size();j++)
		{
			cout << groups[i].adj_index[j] << " ";
			lab_color_sort_index.at<int>(t,0) = groups[i].adj_index[j];
			t++;
		}
		cout << endl;
	}

	groups.clear();
}

void Preprocessing_Data::TSP_path_for_lab_color_coarse_to_fine2(Mat lab_data, Mat& lab_color_sort_index)
{
	int k = lab_data.rows;//k是4的倍數
	int dim = lab_data.rows;
	int group_num = 4;
	Mat cluster_tag = Mat::zeros(k,1,CV_32S);
	Mat cluster_centers = Mat::zeros(group_num,dim,CV_32F);
	//cuda_kmeans(lab_data, group_num, cluster_tag, cluster_centers);
	kmeans(lab_data, group_num, cluster_tag, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.0001), 2, KMEANS_PP_CENTERS,cluster_centers);
	class group{
	public:
		vector<int> index;
		vector<int> index2;
		vector<int> sort_index;
		vector<int> adj_index;
	};
	vector<group> groups(group_num);

	for(int i=0;i<k;i++)
	{
		int tag = cluster_tag.at<int>(i,0);
		groups[tag].index.push_back(i);
	}

	cout << "index :" << endl;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].index.size();j++)
		{
			cout << groups[i].index[j] << " ";
		}
		cout << endl;
	}

	Mat component, coeff;
	int rDim = 2;
	reduceDimPCA(cluster_centers, rDim, component, coeff);
	Mat lab_cluster_center2D_ndim = coeff * component;
	Mat lab_cluster_center2D = Mat::zeros(group_num,2,CV_32F);
	for(int i=0;i<rDim;i++)
	{
		lab_cluster_center2D_ndim.col(i).copyTo( lab_cluster_center2D.col(i) );
	}
	Mat color_sort_index = Mat::zeros(group_num,1,CV_32S);
	//TSP_boost(lab_cluster_center2D, color_sort_index);
	TSP_path(lab_cluster_center2D, color_sort_index);

	for(int i=0;i<group_num;i++)
	{
		int index = color_sort_index.at<int>(i,0);
		for(int j=0;j<groups[index].index.size();j++)
		{
			groups[i].index2.push_back( groups[index].index[j] );
		}
	}

	cout << "index2 :" << endl;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].index2.size();j++)
		{
			cout << groups[i].index2[j] << " ";
		}
		cout << endl;
	}

	rDim = 2;
	reduceDimPCA(lab_data, rDim, component, coeff);
	Mat lab_PCA2D = coeff * component;
	////////////////////////////////////////////////////////////////////////////////////
	Mat lab_PCA2D_sub = Mat::zeros(groups[0].index2.size()+1,2,CV_32F);
	Mat lab_color_sort_index_sub = Mat::zeros(groups[0].index2.size()+1,1,CV_32S);
	Mat optimal_sort_index;
	for(int i=0;i<groups[0].index2.size();i++)
	{
		int index = groups[0].index2[i];
		lab_PCA2D_sub.at<float>(i,0) = lab_PCA2D.at<float>(index,0);
		lab_PCA2D_sub.at<float>(i,1) = lab_PCA2D.at<float>(index,1);
	}
	double min_len = 10000;
	double min_target = 0;
	for(int j=0;j<groups[1].index2.size();j++)
	{
		int index = groups[1].index2[j];
		lab_PCA2D_sub.at<float>(groups[0].index2.size(),0) = lab_PCA2D.at<float>(index,0);
		lab_PCA2D_sub.at<float>(groups[0].index2.size(),1) = lab_PCA2D.at<float>(index,1);		
		double len = TSP_path(lab_PCA2D_sub, lab_color_sort_index_sub);
		if(len < min_len) 
		{
			min_len = len;
			min_target = j;
			optimal_sort_index = lab_color_sort_index_sub.clone();
		}
	}
	
	cout << "optimal_sort_index " << optimal_sort_index << endl;
	for(int i=0;i<optimal_sort_index.rows;i++)
	{
		int index = optimal_sort_index.at<int>(i,0);
		if( index!=groups[0].index2.size() ) //以防start點不是出現在最後面
		{
			groups[0].sort_index.push_back( groups[0].index2[index] );
		}
	}

	groups[0].sort_index.push_back( groups[1].index2[min_target] );
	int start = groups[1].index2[min_target];
	cout << "start " << start << endl;

	cout << "temp: ";
	vector<int> temp = groups[1].index2;
	groups[1].index2.clear();
	for(int i=0;i<temp.size();i++)
	{
		if(temp[i]!=start)
		{
			groups[1].index2.push_back( temp[i] );
			cout << temp[i] << " ";
		}
	}
	cout << endl;
	////////////////////////////////////////////////////////////////////////////////////
	lab_PCA2D_sub = Mat::zeros(groups[0].sort_index.size()+1,2,CV_32F);
	lab_color_sort_index_sub = Mat::zeros(groups[0].sort_index.size()+1,1,CV_32S);	
	for(int i=0;i<groups[0].sort_index.size();i++)
	{
		int index = groups[0].sort_index[i];
		lab_PCA2D_sub.at<float>(i,0) = lab_PCA2D.at<float>(index,0);
		lab_PCA2D_sub.at<float>(i,1) = lab_PCA2D.at<float>(index,1);		
	}
	min_len = 10000;
	min_target = 0;
	for(int j=0;j<groups[2].index2.size();j++)
	{
		int index = groups[2].index2[j];
		lab_PCA2D_sub.at<float>(groups[0].sort_index.size(),0) = lab_PCA2D.at<float>(index,0);
		lab_PCA2D_sub.at<float>(groups[0].sort_index.size(),1) = lab_PCA2D.at<float>(index,1);		
		double len = TSP_path(lab_PCA2D_sub, lab_color_sort_index_sub);
		if(len < min_len) 
		{
			min_len = len;
			min_target = j;
			optimal_sort_index = lab_color_sort_index_sub.clone();
		}
	}

	int end = groups[2].index2[min_target];
	cout << "end " << end << endl;

	Mat TSP_brute_sort_index = Mat::zeros(groups[1].index2.size()+2,1,CV_32S);
	Mat lab_sub = Mat::zeros(groups[1].index2.size()+2,3,CV_32F);
	int t=0;
	lab_data.row( start ).copyTo( lab_sub.row(t++) );
	for(int i=0;i<groups[1].index2.size();i++)
	{
		lab_data.row( groups[1].index2[i] ).copyTo( lab_sub.row(t++) );
	}
	lab_data.row( end ).copyTo( lab_sub.row(t++) );

	tsp_brute tsp;
	tsp.start(lab_sub,0,t-1,TSP_brute_sort_index);
	cout << "TSP_brute_sort_index " << TSP_brute_sort_index << endl;
	for(int i=1;i<TSP_brute_sort_index.rows-1;i++)
	{
		int index = TSP_brute_sort_index.at<int>(i,0) - 1;
		groups[1].sort_index.push_back( groups[1].index2[ index ] );
	}
	groups[1].sort_index.push_back(end);

	cout << "temp: ";
	temp.clear();
	temp = groups[2].index2;
	groups[2].index2.clear();
	for(int i=0;i<temp.size();i++)
	{
		if(temp[i]!=end)
		{
			groups[2].index2.push_back( temp[i] );
			cout << temp[i] << " ";
		}
	}
	cout << endl;

	start = end;
	cout << "start " << start << endl;
	///////////////////////////////////////////////////////////////////////
	lab_PCA2D_sub = Mat::zeros(groups[2].index2.size()+1,2,CV_32F);
	lab_color_sort_index_sub = Mat::zeros(groups[2].index2.size()+1,1,CV_32S);	
	for(int i=0;i<groups[2].index2.size();i++)
	{
		int index = groups[2].index2[i];
		lab_PCA2D_sub.at<float>(i,0) = lab_PCA2D.at<float>(index,0);
		lab_PCA2D_sub.at<float>(i,1) = lab_PCA2D.at<float>(index,1);		
	}
	min_len = 10000;
	min_target = 0;
	for(int j=0;j<groups[3].index2.size();j++)
	{
		int index = groups[3].index2[j];
		lab_PCA2D_sub.at<float>(groups[2].index2.size(),0) = lab_PCA2D.at<float>(index,0);
		lab_PCA2D_sub.at<float>(groups[2].index2.size(),1) = lab_PCA2D.at<float>(index,1);		
		double len = TSP_path(lab_PCA2D_sub, lab_color_sort_index_sub);
		if(len < min_len) 
		{
			min_len = len;
			min_target = j;
			optimal_sort_index = lab_color_sort_index_sub.clone();
		}
	}

	end = groups[3].index2[min_target];
	cout << "end " << end << endl;

	TSP_brute_sort_index = Mat::zeros(groups[2].index2.size()+2,1,CV_32S);
	lab_sub = Mat::zeros(groups[2].index2.size()+2,3,CV_32F);
	t=0;
	lab_data.row( start ).copyTo( lab_sub.row(t++) );
	for(int i=0;i<groups[2].index2.size();i++)
	{
		lab_data.row( groups[2].index2[i] ).copyTo( lab_sub.row(t++) );
	}
	lab_data.row( end ).copyTo( lab_sub.row(t++) );

	//tsp_brute tsp;
	tsp.start(lab_sub,0,t-1,TSP_brute_sort_index);
	cout << "TSP_brute_sort_index " << TSP_brute_sort_index << endl;
	for(int i=1;i<TSP_brute_sort_index.rows-1;i++)
	{
		int index = TSP_brute_sort_index.at<int>(i,0) - 1;
		groups[2].sort_index.push_back( groups[2].index2[ index ] );
	}
	groups[2].sort_index.push_back(end);
	
	cout << "temp: ";
	temp.clear();
	temp = groups[3].index2;
	groups[3].index2.clear();
	for(int i=0;i<temp.size();i++)
	{
		if(temp[i]!=end)
		{
			groups[3].index2.push_back( temp[i] );
			cout << temp[i] << " ";
		}
	}
	cout << endl;	

	start = end;
	cout << "start " << start << endl;
	/////////////////////////////////////////////////////////
	lab_PCA2D_sub = Mat::zeros(groups[3].index2.size()+1,2,CV_32F);
	lab_color_sort_index_sub = Mat::zeros(groups[3].index2.size()+1,1,CV_32S);	
	lab_PCA2D_sub.at<float>(0,0) = lab_PCA2D.at<float>(start,0);
	lab_PCA2D_sub.at<float>(0,1) = lab_PCA2D.at<float>(start,1);
	for(int i=0;i<groups[3].index2.size();i++)
	{
		int index = groups[3].index2[i];
		lab_PCA2D_sub.at<float>(i+1,0) = lab_PCA2D.at<float>(index,0);
		lab_PCA2D_sub.at<float>(i+1,1) = lab_PCA2D.at<float>(index,1);		
	}
	TSP_path(lab_PCA2D_sub, lab_color_sort_index_sub);
	cout << "lab_color_sort_index_sub " << lab_color_sort_index_sub << endl;

	for(int i=0;i<lab_color_sort_index_sub.rows;i++)
	{
		int index = lab_color_sort_index_sub.at<int>(i,0);
		if( index==0 )
		{
			continue;
		}
		else
		{
			groups[3].sort_index.push_back( groups[3].index2[index] );
		}
	}

	t = 0;
	cout << "sort index :" << endl;
 	for(int i=0;i<4;i++)
	{
		for(int j=0;j<groups[i].sort_index.size();j++)
		{
			cout << groups[i].sort_index[j] << " ";
			lab_color_sort_index.at<int>(t,0) = groups[i].sort_index[j];
			t++;
		}
		cout << endl;
	}

}

void Preprocessing_Data::TSP_path_for_lab_color_coarse_to_fine(Mat lab_data, Mat& lab_color_sort_index)
{
	int k = lab_data.rows;//k是4的倍數
	int dim = lab_data.rows;
	int group_num = 4;
	Mat cluster_tag = Mat::zeros(k,1,CV_32S);
	Mat cluster_centers = Mat::zeros(group_num,dim,CV_32F);
	//cuda_kmeans(lab_data, group_num, cluster_tag, cluster_centers);
	//Mat lab_data_2D(lab_data, Range(0,lab_data.rows), Range(1,3) );
	kmeans(lab_data, group_num, cluster_tag, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.0001), 10, KMEANS_RANDOM_CENTERS,cluster_centers);
	class group{
	public:
		vector<int> index;
		vector<int> index2;
		vector<int> sort_index;
		vector<int> adj_index;
	};
	vector<group> groups(group_num);

	for(int i=0;i<k;i++)
	{
		int tag = cluster_tag.at<int>(i,0);
		groups[tag].index.push_back(i);
	}

	cout << "index :" << endl;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].index.size();j++)
		{
			cout << groups[i].index[j] << " ";
		}
		cout << endl;
	}

	Mat component, coeff;
	int rDim = 2;
	reduceDimPCA(cluster_centers, rDim, component, coeff);
	Mat lab_cluster_center2D_ndim = coeff * component;
	Mat lab_cluster_center2D = Mat::zeros(group_num,2,CV_32F);
	for(int i=0;i<rDim;i++)
	{
		lab_cluster_center2D_ndim.col(i).copyTo( lab_cluster_center2D.col(i) );
	}
	Mat color_sort_index = Mat::zeros(group_num,1,CV_32S);
	//TSP_boost(lab_cluster_center2D, color_sort_index);
	TSP_path(lab_cluster_center2D, color_sort_index);

	for(int i=0;i<group_num;i++)
	{
		int index = color_sort_index.at<int>(i,0);
		for(int j=0;j<groups[index].index.size();j++)
		{
			groups[i].index2.push_back( groups[index].index[j] );
		}
	}

	cout << "index2 :" << endl;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].index2.size();j++)
		{
			cout << groups[i].index2[j] << " ";
		}
		cout << endl;
	}

	//Mat component, coeff;
	rDim = 2;
	reduceDimPCA(lab, rDim, component, coeff);
	Mat lab_PCA2D = coeff * component;

	//int t = 0;
	for(int i=0;i<group_num;i++)
	{
		Mat lab_PCA2D_sub = Mat::zeros(groups[i].index2.size(),2,CV_32F);
		Mat lab_color_sort_index_sub = Mat::zeros(groups[i].index2.size(),1,CV_32S);
		for(int j=0;j<groups[i].index2.size();j++)
		{
			int index = groups[i].index2[j];
			lab_PCA2D_sub.at<float>(j,0) = lab_PCA2D.at<float>(index,0);	
			lab_PCA2D_sub.at<float>(j,1) = lab_PCA2D.at<float>(index,1);	
		}
		//TSP_boost(lab_PCA2D_sub, lab_color_sort_index_sub);
		TSP_path(lab_PCA2D_sub, lab_color_sort_index_sub);
		for(int j=0;j<groups[i].index2.size();j++)
		{
			int index = lab_color_sort_index_sub.at<int>(j,0);
			groups[i].sort_index.push_back(groups[i].index2[index]);
			//lab_color_sort_index.at<int>(t,0) = groups[i].index2[index];
			//t++;
		}
	}

	cout << "sort index: " << endl;
	//int t = 0;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].sort_index.size();j++)
		{
			if(i==0)
			{
				groups[i].adj_index.push_back(groups[i].sort_index[j]);
			}
			cout << groups[i].sort_index[j] << " ";
			//lab_color_sort_index.at<int>(t,0) = groups[i].sort_index[j];
			//t++;
		}
		cout << endl;
	}
	
	for(int i=0;i<group_num-1;i++)
	{
		double tour_len1,tour_len2;
		if(groups[i].sort_index.size()==1)
		{
			tour_len1 = sqrt( pow(lab_PCA2D.at<float>(groups[i].adj_index[0],0) - lab_PCA2D.at<float>(groups[i+1].sort_index[0],0), 2)
				           +  pow(lab_PCA2D.at<float>(groups[i].adj_index[0],1) - lab_PCA2D.at<float>(groups[i+1].sort_index[0],1), 2));
			tour_len2 = sqrt( pow(lab_PCA2D.at<float>(groups[i].adj_index[0],0) - lab_PCA2D.at<float>(groups[i+1].sort_index[ groups[i+1].sort_index.size()-1 ],0), 2)
				           +  pow(lab_PCA2D.at<float>(groups[i].adj_index[0],1) - lab_PCA2D.at<float>(groups[i+1].sort_index[ groups[i+1].sort_index.size()-1],1), 2));
			cout << "tour_len1 " << tour_len1 << endl;
			cout << "tour_len2 " << tour_len2 << endl;
		}
		//else if(groups[i+1].sort_index.size()==1)
		//{
		//	tour_len1 = sqrt( pow(lab_PCA2D.at<float>(groups[i].adj_index[0],0) - lab_PCA2D.at<float>(groups[i+1].sort_index[0],0), 2)
		//		           +  pow(lab_PCA2D.at<float>(groups[i].adj_index[0],1) - lab_PCA2D.at<float>(groups[i+1].sort_index[0],1), 2));
		//	tour_len2 = sqrt( pow(lab_PCA2D.at<float>(groups[i].adj_index[ groups[i].adj_index.size()-1 ],0) - lab_PCA2D.at<float>(groups[i+1].sort_index[0],0), 2)
		//		           +  pow(lab_PCA2D.at<float>(groups[i].adj_index[ groups[i].adj_index.size()-1 ],1) - lab_PCA2D.at<float>(groups[i+1].sort_index[0],1), 2));
		//	cout << "tour_len1 " << tour_len1 << endl;
		//	cout << "tour_len2 " << tour_len2 << endl;			
		//}
		else
		{
			Mat lab_PCA2D_sub = Mat::zeros(groups[i].sort_index.size()+1,2,CV_32F);
			Mat lab_color_sort_index_sub = Mat::zeros(groups[i].sort_index.size(),1,CV_32S);
			for(int j=0;j<groups[i].adj_index.size();j++)
			{
				int index = groups[i].adj_index[j];
				lab_PCA2D_sub.at<float>(j,0) = lab_PCA2D.at<float>(index,0);	
				lab_PCA2D_sub.at<float>(j,1) = lab_PCA2D.at<float>(index,1);	
				//cout << index << " " ;
			}
			//cout << groups[i+1].sort_index[0] << endl;
			lab_PCA2D_sub.at<float>(groups[i].sort_index.size(),0) = lab_PCA2D.at<float>(groups[i+1].sort_index[0],0);	
			lab_PCA2D_sub.at<float>(groups[i].sort_index.size(),1) = lab_PCA2D.at<float>(groups[i+1].sort_index[0],1);	
			tour_len1 = TSP_path(lab_PCA2D_sub, lab_color_sort_index_sub);
			cout << "tour_len1 " << tour_len1 << endl;
			lab_PCA2D_sub.at<float>(groups[i].sort_index.size(),0) = lab_PCA2D.at<float>(groups[i+1].sort_index[ groups[i+1].sort_index.size()-1 ],0);	
			lab_PCA2D_sub.at<float>(groups[i].sort_index.size(),1) = lab_PCA2D.at<float>(groups[i+1].sort_index[ groups[i+1].sort_index.size()-1 ],1);	
			tour_len2 = TSP_path(lab_PCA2D_sub, lab_color_sort_index_sub);
			cout << "tour_len2 " << tour_len2 << endl;
		}
		if(tour_len1<tour_len2)
		{
			for(int j=0;j<groups[i+1].sort_index.size();j++)
			{
				groups[i+1].adj_index.push_back( groups[i+1].sort_index[j] );
			}			
		}
		else
		{
			for(int j=groups[i+1].sort_index.size()-1;j>=0;j--)
			{
				groups[i+1].adj_index.push_back( groups[i+1].sort_index[j] );
			}			
		}

	}

	cout << "adj index: " << endl;
	int t = 0;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].adj_index.size();j++)
		{
			cout << groups[i].adj_index[j] << " ";
			lab_color_sort_index.at<int>(t,0) = groups[i].adj_index[j];
			t++;
		}
		cout << endl;
	}


	/*
	for(int s=1;s<group_num;s++)
	{
		bool flag = false;
		for(int i=0;i<groups[s].sort_index.size();i++)
		{
			Mat lab_PCA2D_sub = Mat::zeros(groups[s-1].sort_index.size()+1,2,CV_32F);
			Mat lab_color_sort_index_sub = Mat::zeros(groups[s-1].sort_index.size()+1,1,CV_32S);
			for(int j=0;j<groups[s-1].sort_index.size();j++)
			{
				int index = groups[s-1].sort_index[j];
				lab_PCA2D_sub.at<float>(j,0) = lab_PCA2D.at<float>(index,0);	
				lab_PCA2D_sub.at<float>(j,1) = lab_PCA2D.at<float>(index,1);	
			}
			int try_index = groups[s].sort_index[i];
			lab_PCA2D_sub.at<float>(groups[s-1].sort_index.size(),0) = lab_PCA2D.at<float>(try_index,0);
			lab_PCA2D_sub.at<float>(groups[s-1].sort_index.size(),1) = lab_PCA2D.at<float>(try_index,1);
			TSP_boost(lab_PCA2D_sub, lab_color_sort_index_sub);
			//TSP_path(lab_PCA2D_sub, lab_color_sort_index_sub);
		
			if( lab_color_sort_index_sub.at<int>(groups[s-1].sort_index.size(),0) == groups[s-1].sort_index.size() )
			{
				flag = true;
				for(int t=i;t<groups[s].sort_index.size();t++)
				{
					groups[s].adj_index.push_back( groups[s].sort_index[t] );
					//cout << groups[s].sort_index[t] << " ";
				}
			
				for(int t=0;t<i;t++)
				{
					groups[s].adj_index.push_back( groups[s].sort_index[t] );
					//cout << groups[s].sort_index[t] << " ";
				}
				break;
			}
		}
		if(flag==false)
		{
			for(int t=0;t<groups[s].sort_index.size();t++)
			{
				groups[s].adj_index.push_back( groups[s].sort_index[t] );
				//cout << groups[s].sort_index[t] << " ";
			}
		}
		//cout << endl;
	}

	cout << "adj index: " << endl;
	int t = 0;
 	for(int i=0;i<group_num;i++)
	{
		for(int j=0;j<groups[i].adj_index.size();j++)
		{
			cout << groups[i].adj_index[j] << " ";
			lab_color_sort_index.at<int>(t,0) = groups[i].adj_index[j];
			t++;
		}
		cout << endl;
	}
	*/

	groups.clear();
}

double Preprocessing_Data::TSP_path(Mat input_mat, Mat& sort_index)
{
	int NumCities = input_mat.rows;
	int cols = input_mat.cols;
	Mat input_mat_2D = Mat::zeros(NumCities,2,CV_32F);
	if(cols>2)
	{
		Mat component, coeff;
		int rDim = 2;
		reduceDimPCA(input_mat, rDim, component, coeff);
		Mat reduceMat = coeff * component;
		reduceMat.col(0).copyTo( input_mat_2D.col(0) );
		reduceMat.col(1).copyTo( input_mat_2D.col(1) );
	}
	else if(cols==2)
	{
		input_mat_2D = input_mat.clone();
	}

	path_index = 0; //initialize TSP path index 

	CITY_INFO polypoints[300];
	for(int i=0;i<NumCities;i++)
	{
		polypoints[i].set_info(int2str(i),i, input_mat_2D.at<float>(i,0),  input_mat_2D.at<float>(i,1));
	}

	//int NumCities;
	path_index_vec.resize(NumCities);
    double dist = 0;
    string path;

	TSP_Start(polypoints, NumCities, &dist, path);
	//cout << "dist " << dist << endl;

	//for(int i=0;i<path_index_vec.size();i++)
	//{
	//	for(int j=0;j<path_index_vec[i].size();j++)
	//		cout << path_index_vec[i][j] << " ";
	//}
	//cout << endl;
	int t = 0;
	for(int i=0;i<path_index_vec.size();i++)
	{
		for(int j=0;j<path_index_vec[i].size();j++)
		{
			sort_index.at<int>(t,0) = path_index_vec[i][j];	
			t++;
		}
	}

	double tour_len = 0.0;
	for(int i=0;i<sort_index.rows-1;i++)
	{
		int index = sort_index.at<int>(i,0);
		double dist = sqrt( ( input_mat_2D.at<float>(index,0) - input_mat_2D.at<float>(index+1,0) ) * ( input_mat_2D.at<float>(index,0) - input_mat_2D.at<float>(index+1,0) ) 
		                   + ( input_mat_2D.at<float>(index,1) - input_mat_2D.at<float>(index+1,1) ) * ( input_mat_2D.at<float>(index,1) - input_mat_2D.at<float>(index+1,1) ) );
		tour_len += dist;
	}

	path_index_vec.clear();

	return tour_len;
}

void Preprocessing_Data::TSP_for_lab_color(Mat cluster_center)
{
	int k = cluster_center.rows;

	Mat component, coeff;
	int rDim = 2;
	reduceDimPCA(lab, rDim, component, coeff);
	Mat lab_PCA2D = coeff * component;
	
	path_index = 0; //initialize TSP path index 

	CITY_INFO polypoints[100];
	for(int i=0;i<k;i++)
	{
		polypoints[i].set_info(int2str(i),i, lab_PCA2D.at<float>(i,0),  lab_PCA2D.at<float>(i,1));
	}

	int NumCities = k;
	path_index_vec.resize(1000);
    double dist = 0;
    string path;
	
	TSP_Start(polypoints, NumCities, &dist, path);

	for(int i=0;i<path_index_vec.size();i++)
	{
		for(int j=0;j<path_index_vec[i].size();j++)
			cout << path_index_vec[i][j] << " ";
	}
	cout << endl;

}

//add edges to the graph (for each node connect it to all other nodes)
template<typename VertexListGraph, typename PointContainer,
    typename WeightMap, typename VertexIndexMap>
void Preprocessing_Data::connectAllEuclidean(VertexListGraph& g,
                        const PointContainer& points,
                        WeightMap wmap,            // Property maps passed by value
                        VertexIndexMap vmap       // Property maps passed by value
                        )
{
    using namespace boost;
    using namespace std;
    typedef typename graph_traits<VertexListGraph>::edge_descriptor Edge;
    typedef typename graph_traits<VertexListGraph>::vertex_iterator VItr;

    Edge e;
    bool inserted;

    pair<VItr, VItr> verts(vertices(g));
    for (VItr src(verts.first); src != verts.second; src++)
    {
        for (VItr dest(src); dest != verts.second; dest++)
        {
            if (dest != src)
            {
                double weight(sqrt(pow(
                    static_cast<double>(points[vmap[*src]].x -
                        points[vmap[*dest]].x), 2.0) +
                    pow(static_cast<double>(points[vmap[*dest]].y -
                        points[vmap[*src]].y), 2.0)));

                boost::tie(e, inserted) = add_edge(*src, *dest, g);

                wmap[e] = weight;
            }

        }

    }
}

template <typename PositionVec>
void Preprocessing_Data::checkAdjList(PositionVec v)
{
    using namespace std;
    using namespace boost;

    typedef adjacency_list<listS, listS, undirectedS> Graph;
    typedef graph_traits<Graph>::vertex_descriptor Vertex;
    typedef graph_traits <Graph>::edge_descriptor Edge;
    typedef vector<Vertex> Container;
    typedef map<Vertex, std::size_t> VertexIndexMap;
    typedef map<Edge, double> EdgeWeightMap;
    typedef associative_property_map<VertexIndexMap> VPropertyMap;
    typedef associative_property_map<EdgeWeightMap> EWeightPropertyMap;
    typedef graph_traits<Graph>::vertex_iterator VItr;

    Container c;
    EdgeWeightMap w_map;
    VertexIndexMap v_map;
    VPropertyMap v_pmap(v_map);
    EWeightPropertyMap w_pmap(w_map);

    Graph g(v.size());

    //create vertex index map
    VItr vi, ve;
    int idx(0);
    for (boost::tie(vi, ve) = vertices(g); vi != ve; ++vi)
    {
        Vertex v(*vi);
        v_pmap[v] = idx;
        idx++;
    }

    connectAllEuclidean(g, v, w_pmap,
        v_pmap);

    metric_tsp_approx_from_vertex(g,
        *vertices(g).first,
        w_pmap,
        v_pmap,
        tsp_tour_visitor<back_insert_iterator<Container > >
        (back_inserter(c)));

    //cout << "adj_list" << endl;
    //for (Container::iterator itr = c.begin(); itr != c.end(); ++itr) {
    //    cout << v_map[*itr] << " ";
    //}
    //cout << endl << endl;

    c.clear();
}