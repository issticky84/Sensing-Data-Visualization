#include "Preprocessing_Data.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include "tapkee/tapkee.hpp"
#include "tapkee/callbacks/dummy_callbacks.hpp"
#include <opencv2/core/eigen.hpp> //cv2eigen

using namespace tapkee;
using namespace Eigen;

void Preprocessing_Data::start()
{
	
	//=================Read CSV file====================//
	clock_t begin = clock();
	strcpy(file_csv_data,"../../../../csv_data/BigData_20141121_0723_new.csv");
	read_raw_data(); 
	clock_t end = clock();
	printf("Read csv elapsed time: %f\n",double(end - begin) / CLOCKS_PER_SEC);
	//==================================================//

	int attribute_title_size = 11;
	int attribute_title[] = {4,5,6,7,8,9,10,11,12,22,23};//(gravity_x,gravity_y,gravity_z,linear_acc_x),(linear_acc_x,linear_acc_y,linear_acc_z),(gyro_x,gyro_y,gyro_z),(latitude,longitude)
	int time_title[] = {29,30,31,32,33};//hour(30),minute(31)
	//int time_title[] = {27,28,29,30,31};//hour(30),minute(31)
	//============Setting matrix for K-means============//
	clock_t begin2 = clock();
	set_hour_data(time_title);
	Mat model = set_matrix(attribute_title,attribute_title_size).clone();
	clock_t end2 = clock();
	printf("Setting matrix elapsed time: %f\n",double(end2 - begin2) / CLOCKS_PER_SEC);
	//==================================================//
	output_mat_as_txt_file("model.txt",model);
    int k = 25; 
    Mat cluster_tag; //Tag:0~k-1
    int attempts = 2;//應該是執行次數
	Mat cluster_centers;
	//==============K means clustering==================//
	//使用k means分群
	clock_t begin3 = clock();
	kmeans(model, k, cluster_tag,TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 0.0001), attempts,KMEANS_PP_CENTERS,cluster_centers);
	clock_t end3 = clock();
    //TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 1),  這裡有三個參數，決定k-means何時結束，第二個參數是指迭代最大次數，第三個參數是精確度多少，第一個參數是指依照前兩個參數的哪一個為準，以範例中就是兩者都參照，以 or 的方式決定
	printf("Kmeans (K = %d) elapsed time: %f\n",k,double(end3 - begin3) / CLOCKS_PER_SEC);
	//==================================================//
	output_mat_as_csv_file("cluster_center.csv",cluster_centers);
	voting(k,cluster_tag,model.rows); // Type: int
	cluster_tag.release();
	//===================PCA RGB=======================//
	Mat components, result;
	int rDim = 3;
	clock_t begin4 = clock();
	reduceDimPCA(cluster_centers, rDim, components, result);
	clock_t end4 = clock();
	printf("PCA RGB elapsed time: %f\n",double(end4 - begin4) / CLOCKS_PER_SEC);
	rgb_mat = result.clone();
	for(int i=0;i<result.cols;i++)
		normalize(result.col(i),rgb_mat.col(i),0,1,NORM_MINMAX); //normalize to 0-1
	output_mat_as_txt_file("rgb_mat.txt",rgb_mat);
	//=================LAB alignment====================//
	rgb_mat2 = lab_alignment(cluster_centers);
	output_mat_as_csv_file("lab_mat.csv",rgb_mat2);
	//===============Position (MDS)=====================//
	clock_t begin5 = clock();
	position = Position_by_MDS(cluster_centers,k,20).clone(); //Type:double
	clock_t end5 = clock();
	printf("MDS Position elapsed time: %f\n",double(end5 - begin5) / CLOCKS_PER_SEC);
	cluster_centers.release();
	//===================PCA raw data 3 dim=======================//
	reduceDimPCA(model, rDim, components, result);
	raw_data_3D = result.clone();
	for(int i=0;i<result.cols;i++)
		normalize(result.col(i),raw_data_3D.col(i),0,1,NORM_MINMAX); //normalize to 0-1
	model.release();
	//output_mat_as_txt_file("raw_data_3D.txt",raw_data_3D);
}

void Preprocessing_Data::output_mat_as_txt_file(char file_name[],Mat mat)
{
	ofstream fout(file_name);
	fout << mat << endl;
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
		exit(1);
	}

	char line[LENGTH];
	char *token;
	int i,j;
	i = j = 0;
	fgets(line,LENGTH,csv_file); //ignore sep=
	fgets(line,LENGTH,csv_file); //ignore title
	
	//token = strtok(line,";");
	//while(token!=NULL)
	//{
	//	title_name.push_back(token);
	//	token = strtok(NULL,";");
	//}
	//for(int i=0;i<title_name.size();i++) cout << title_name[i] << " ";
	//cout << "title size: " << title_name.size() << endl;

	while(!feof(csv_file))
	{
		fgets(line,LENGTH,csv_file);
		//token = strtok(line,";");
		token = strtok(line,";");
		raw_data.push_back(vector<float> (1));
		//printf("%s ",token);
		while(token!=NULL)
		{
			raw_data.back().push_back(atof(token));
			//token = strtok(NULL," ;:");
			token = strtok(NULL," ;:/");
		}
	}

	//cout << raw_data[0][1] << " " << raw_data[0][27] << " " << raw_data[0][28] << " " << raw_data[0][29] << " " << raw_data[0][30]  << " " << raw_data[0][31] << endl;
	//cout << raw_data[0][1] << " " << raw_data[0][29] << " " << raw_data[0][30] << " " << raw_data[0][31] << " " << raw_data[0][32]  << " " << raw_data[0][33] << endl;
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

	if(d>20.0) 
		return 0.01;
	else 
		return d;
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

Mat Preprocessing_Data::Gaussian_filter(int attribute_title[],int MAX_KERNEL_LENGTH)
{
	Mat Gaussian_filter_mat(raw_data.size(),9, CV_32F);

	//Apply Gaussian filter to raw data(0~8)
	for(int i=0;i<raw_data.size();i++)
	{
		for(int j=0;j<9;j++)
		{
			Gaussian_filter_mat.at<float>(i,j) = raw_data[i][attribute_title[j]];
		}
	}

	//int MAX_KERNEL_LENGTH = 20;
    for(int i=1;i<MAX_KERNEL_LENGTH;i=i+2)
    { 
		GaussianBlur( Gaussian_filter_mat, Gaussian_filter_mat, Size( i, i ), 0.5, 0.5 );
	}

	return Gaussian_filter_mat;
}

Mat Preprocessing_Data::set_matrix(int attribute_title[],int attribute_title_size)
{
	//Mat handle_mat(raw_data.size(),4, CV_32F);
	Mat handle_mat;

	Mat Gaussian_filter_mat = Gaussian_filter(attribute_title,20).clone();

	Mat norm_gravity(1, raw_data.size(), CV_32F);
	Mat norm_linear_acc(1, raw_data.size(), CV_32F);
	Mat norm_gyro(1, raw_data.size(), CV_32F);
	//Compute norm
	for(int i=0;i<raw_data.size();i++)
	{
		norm_gravity.at<float>(0,i) = norm_value(Gaussian_filter_mat.at<float>(i,attribute_title[0]),Gaussian_filter_mat.at<float>(i,attribute_title[1]),Gaussian_filter_mat.at<float>(i,attribute_title[2]));
		norm_linear_acc.at<float>(0,i) = norm_value(Gaussian_filter_mat.at<float>(i,attribute_title[3]),Gaussian_filter_mat.at<float>(i,attribute_title[4]),Gaussian_filter_mat.at<float>(i,attribute_title[5]));
		norm_gyro.at<float>(0,i) = norm_value(Gaussian_filter_mat.at<float>(i,attribute_title[6]),Gaussian_filter_mat.at<float>(i,attribute_title[7]),Gaussian_filter_mat.at<float>(i,attribute_title[8]));
	}

	handle_mat.push_back(norm_gravity);
	handle_mat.push_back(norm_linear_acc);
	handle_mat.push_back(norm_gyro);
	//Compute norm
	//for(int i=0;i<raw_data.size();i++)
	//{
	//	handle_mat.at<float>(i,0) = norm_value(Gaussian_filter_mat.at<float>(i,attribute_title[0]),Gaussian_filter_mat.at<float>(i,attribute_title[1]),Gaussian_filter_mat.at<float>(i,attribute_title[2]));
	//	handle_mat.at<float>(i,1) = norm_value(Gaussian_filter_mat.at<float>(i,attribute_title[3]),Gaussian_filter_mat.at<float>(i,attribute_title[4]),Gaussian_filter_mat.at<float>(i,attribute_title[5]));
	//	handle_mat.at<float>(i,2) = norm_value(Gaussian_filter_mat.at<float>(i,attribute_title[6]),Gaussian_filter_mat.at<float>(i,attribute_title[7]),Gaussian_filter_mat.at<float>(i,attribute_title[8]));
	//}
	//for(int i=0;i<raw_data.size();i++)
	//{
	//	handle_mat.at<float>(i,0) = norm_value(raw_data[i][attribute_title[0]],raw_data[i][attribute_title[1]],raw_data[i][attribute_title[2]]);
	//	handle_mat.at<float>(i,1) = norm_value(raw_data[i][attribute_title[3]],raw_data[i][attribute_title[4]],raw_data[i][attribute_title[5]]);
	//	handle_mat.at<float>(i,2) = norm_value(raw_data[i][attribute_title[6]],raw_data[i][attribute_title[7]],raw_data[i][attribute_title[8]]);
	//}

	//Compute latitude & longitude
	int lat_index = attribute_title[9];
	int lon_index = attribute_title[10];
	Mat first_order_distance_mat(1, raw_data.size(), CV_32F);
	for(int i=0;i<raw_data.size();i++)
	{
		if(i==0)
			first_order_distance_mat.at<float>(0,i) = 0.0;
		else
		{
			first_order_distance_mat.at<float>(0,i) = DistanceOfLontitudeAndLatitude(raw_data[i-1][lat_index],raw_data[i][lat_index],raw_data[i-1][lon_index],raw_data[i][lon_index]);
		}
	}

	handle_mat.push_back(first_order_distance_mat);
	Mat handle_mat2 = handle_mat.t();

	//cout << handle_mat2.rows + " " + handle_mat2.cols << endl;

	//output_mat_as_csv_file("handle_mat.csv",handle_mat2);
	//cout << handle_mat << endl;
	Mat normalize_mat = handle_mat2;
	for(int i=0;i<handle_mat2.cols;i++)
		normalize(handle_mat2.col(i),normalize_mat.col(i),0,1,NORM_MINMAX);
	//cout << normalize_mat << endl;

	output_mat_as_csv_file("normalize_mat.csv",normalize_mat);

	return normalize_mat;

}

void Preprocessing_Data::voting(int k,Mat cluster_tag,int row_size)
{
	int time_step_amount = floor(row_size/600.0);
	histogram = Mat::zeros(time_step_amount,k,CV_32S);
	int t = 1;
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

Mat Preprocessing_Data::Position_by_MDS(Mat cluster_centers,int k,float larger_weighting)
{
	//test
	int time_step = floor(raw_data.size()/600.0);
	Mat Ev = Mat::zeros(time_step,cluster_centers.cols,CV_32F);
	for(int i=0;i<time_step;i++)
	{
		float base = 0;
		for(int j=0;j<k;j++)
		{
			Ev.row(i) += histogram.at<int>(i,j)*cluster_centers.row(j);
			base += histogram.at<int>(i,j);
		}
		Ev.row(i)/=base;
	}
	output_mat_as_txt_file("Ev.txt",Ev);
	//
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
	wi *= larger_weighting;

	int time_step_amount = histogram.rows;
	Mat histo_coeff = Mat::zeros(time_step_amount,time_step_amount,CV_64F);
	//for(int i=0;i<time_step_amount;i++)
	//	for(int j=0;j<time_step_amount;j++)
	//		for(int t=0;t<k;t++)
	//		{
	//			histo_coeff.at<double>(i,j) += wi.at<float>(0,t)*abs(histogram.at<int>(i,t)-histogram.at<int>(j,t));
	//		}
	for(int i=0;i<time_step_amount;i++)
		for(int j=0;j<time_step_amount;j++)
			for(int t=0;t<cluster_centers.cols;t++)
			{
				histo_coeff.at<double>(i,j) += abs(Ev.at<float>(i,t)-Ev.at<float>(j,t));
			}
	histo_coeff.mul(1000);
	output_mat_as_txt_file("histo_coeff.txt",histo_coeff);
	Matrix<double,Dynamic,Dynamic> histo_coeff_EigenType;//The type pass to Tapkee must be "double" not "float"
	cv2eigen(histo_coeff,histo_coeff_EigenType);
	TapkeeOutput output = tapkee::initialize() 
						   .withParameters((method=MultidimensionalScaling,target_dimension=1))
						   .embedUsing(histo_coeff_EigenType);
	Mat MDS_mat; //Type:double  
	eigen2cv(output.embedding,MDS_mat);
	normalize(MDS_mat.col(0),MDS_mat.col(0),1,100,NORM_MINMAX);//normalize to 1~1000	
	output_mat_as_txt_file("normalized_MDS_mat.txt",MDS_mat);
	MDS_mat = MDS_mat.mul(50);
	output_mat_as_txt_file("normalized_MDS_mat_2.txt",MDS_mat);

	return MDS_mat; 
}

void Preprocessing_Data::mrdivide(const cv::Mat &A, const cv::Mat &b, cv::Mat &x) {
     cv::solve(A.t(), b.t(), x);
     x.t();
}

Mat Preprocessing_Data::lab_alignment(Mat cluster_center)
{
	read_lab_csv();
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
			cluster_center_alignment_mat.at<float>(i,j) = cluster_center.at<float>(i,j) - cluster_center_centroid.at<float>(1,j);
		}
	}
	Mat cluster_center_component,cluster_center_PCA;
	int rDim = 3;
	reduceDimPCA(cluster_center_alignment_mat, rDim, cluster_center_component, cluster_center_PCA);
	//cluster center 3 axis of PCA
	Mat cluster_center_axis = Mat::zeros(3,3,CV_32F);
	for(int i=0;i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
			cluster_center_axis.at<float>(i,j) = cluster_center_component.at<float>(j,i);
		}
	}
	output_mat_as_csv_file("cluster_center_PCA.csv",cluster_center_PCA);
	//compute centroid of LAB
	Mat lab_centroid = compute_centroid(lab_mat);
	
	//lab vertices 3 axis of PCA
	Mat lab_components,lab_PCA;
	rDim = 3;
	reduceDimPCA(lab_mat, rDim, lab_components, lab_PCA);
	Mat lab_axis = Mat::zeros(3,3,CV_32F);
	for(int i=0;i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
			lab_axis.at<float>(i,j) = lab_components.at<float>(j,i);
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
			//cout << "inverse " << cluster_center_axis.inv() << endl;
			cluster_center_axis_invert = cluster_center_axis.inv();
			//cluster_center_axis_invert = cluster_center_axis_invert.mul(0.1);
			cluster_center_PCA_weight = cluster_center_PCA_temp * cluster_center_axis_invert;
			//cout << "cluster_center_PCA_weight " << cluster_center_PCA_weight << endl;
			//cout << "cluster_center_axis_invert" << cluster_center_axis_invert << endl;
			align_mat = cluster_center_PCA_weight*lab_axis;
			//cout << "align_mat " << align_mat << endl;
			//把重心平移回去
			for(int i=0;i<align_mat.rows;i++)
			{
				for(int j=0;j<3;j++)
				{
					align_mat.at<float>(i,j) += lab_centroid.at<float>(1,j);
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
			max_align_mat = align_mat;
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
				max_align_mat.at<float>(i,j) += lab_centroid.at<float>(1,j);
			}
		}
	}

	output_mat_as_txt_file("max_align_mat.txt",max_align_mat);
	printf("max_move : %f max_scale : %f\n",max_move,max_scale);
	Mat rgb_mat2 = LAB2RGB(max_align_mat).clone();

	return rgb_mat2;
}

Mat Preprocessing_Data::compute_centroid(Mat input_mat)
{
	Mat input_mat_centroid = Mat::zeros(1,input_mat.cols,CV_32F);
	for(int i=0;i<input_mat.rows;i++)
	{
		for(int j=0;j<input_mat.cols;j++)
		{
			input_mat_centroid.at<float>(1,j) += input_mat.at<float>(i,j);
		}
	}

	for(int j=0;j<input_mat.cols;j++) input_mat_centroid.at<float>(1,j)/=input_mat.rows;
	
	return input_mat_centroid;
}

bool Preprocessing_Data::lab_boundary_test(float p1,float p2,float p3)
{
	//cout << p1 << " " << p2 << " " << p3 << endl;
	bool test = true;
	Mat lab_color(1, 1, CV_32FC3);
	Mat rgb_color(1, 1, CV_32FC3);
	lab_color.at<Vec3f>(0, 0) = Vec3f(p1, p2, p3);
	cvtColor(lab_color, rgb_color, CV_Lab2BGR);
	cvtColor(rgb_color, lab_color, CV_BGR2Lab);
	//cout << lab_color.at<Vec3f>(0,0).val[0] << " " << lab_color.at<Vec3f>(0,0).val[1] << " " << lab_color.at<Vec3f>(0,0).val[2] << endl << endl;
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
