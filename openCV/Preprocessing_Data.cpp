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

Preprocessing_Data::Preprocessing_Data()
{
	read_lab_csv();
}

void Preprocessing_Data::start()
{
	//=================Read CSV file====================//
	clock_t begin1 = clock();
	strcpy(file_csv_data,"../../../../csv_data/BigData_20150104_2314_new.csv");
	read_raw_data(); 
	clock_t end1 = clock();
	printf("Read csv elapsed time: %f\n",double(end1 - begin1) / CLOCKS_PER_SEC);
	//==================================================//

	int attribute_title_size = 11;
	int attribute_title[] = {4,5,6,7,8,9,10,11,12,22,23};//(gravity_x,gravity_y,gravity_z,linear_acc_x),(linear_acc_x,linear_acc_y,linear_acc_z),(gyro_x,gyro_y,gyro_z),(latitude,longitude)
	int time_title[] = {29,30,31,32,33};//hour(30),minute(31)
	//int time_title[] = {27,28,29,30,31};//hour(30),minute(31) //0104_2314
	//============Setting matrix for K-means============//
	clock_t begin2 = clock();
	set_hour_data(time_title);
	Mat model = set_matrix(attribute_title,attribute_title_size).clone();
	clock_t end2 = clock();
	printf("Setting matrix elapsed time: %f\n",double(end2 - begin2) / CLOCKS_PER_SEC);
	//==================================================//
	output_mat_as_csv_file("model.csv",model);
    int k = 25; 
    Mat cluster_tag; //Tag:0~k-1
    int attempts = 2;//應該是執行次數
	Mat cluster_centers;
	//==============K means clustering==================//
	//使用k means分群
	clock_t begin3 = clock();
	kmeans(model, k, cluster_tag,TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.0001), attempts,KMEANS_PP_CENTERS,cluster_centers);
	clock_t end3 = clock();
    //TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 1),  這裡有三個參數，決定k-means何時結束，第二個參數是指迭代最大次數，第三個參數是精確度多少，第一個參數是指依照前兩個參數的哪一個為準，以範例中就是兩者都參照，以 or 的方式決定
	printf("Kmeans (K = %d) elapsed time: %f\n",k,double(end3 - begin3) / CLOCKS_PER_SEC);
	//=================LAB alignment====================//
	clock_t begin5 = clock();
	//if(cluster_centers.cols>=3) rgb_mat2 = lab_alignment(cluster_centers);
	if(cluster_centers.cols>=3) rgb_mat2 = lab_alignment_new(cluster_centers);
	else if(cluster_centers.cols==1) rgb_mat2 = lab_alignment_dim1(cluster_centers);
	else if(cluster_centers.cols==2) rgb_mat2 = lab_alignment_dim2 (cluster_centers);
	clock_t end5 = clock();
	printf("\nLAB alignment elapsed time: %f\n",double(end5 - begin5) / CLOCKS_PER_SEC);
	output_mat_as_csv_file("lab_mat.csv",rgb_mat2);
	//==================================================//
	//sort the cluster by color and generate new cluster tag and cluster center
	clock_t begin4 = clock();
	sort_by_color(k,rgb_mat2,cluster_centers,cluster_tag);
	clock_t end4 = clock();
	printf("\nSort by Color elapsed time: %f\n",double(end4 - begin4) / CLOCKS_PER_SEC);

	output_mat_as_csv_file("lab_mat2.csv",rgb_mat2);
	output_mat_as_csv_file("cluster_center.csv",cluster_centers);
	output_mat_as_csv_file_int("cluster_tag.csv",cluster_tag);
	//==================================================//
	voting(k,cluster_tag,model.rows); // Type: int	
	//adjust_histogram(cluster_centers,cluster_tag,model);
	////////////
	//test(cluster_centers,model,cluster_tag);
	///////////
	//===================PCA RGB=======================//
	Mat components, result;
	int rDim = 3;
	//clock_t begin4 = clock();
	//reduceDimPCA(cluster_centers, rDim, components, result);
	//clock_t end4 = clock();
	//printf("PCA RGB elapsed time: %f\n",double(end4 - begin4) / CLOCKS_PER_SEC);
	//rgb_mat = result.clone();
	//for(int i=0;i<result.cols;i++)
	//	normalize(result.col(i),rgb_mat.col(i),0,1,NORM_MINMAX); //normalize to 0-1
	//output_mat_as_txt_file("rgb_mat.txt",rgb_mat);
	//================voting result=====================//
	//clock_t begin5 = clock();
	//vector< vector<int> > voting_result(k);
	//for(int i=0;i<cluster_tag.rows;i++)
	//{
	//	int c = cluster_tag.at<int>(i,0);
	//	voting_result[c].push_back(i);
	//}
	//clock_t end5 = clock();
	//printf("Voting Result elapsed time: %f\n",double(end5 - begin5) / CLOCKS_PER_SEC);
	//===============Position (MDS)=====================//
	clock_t begin6 = clock();
	position = Position_by_MDS(cluster_centers,model,cluster_tag,k).clone(); //Type:double
	cluster_tag.release();
	output_mat_as_csv_file_double("position.csv",position);
	clock_t end6 = clock();
	printf("MDS Position elapsed time: %f\n",double(end6 - begin6) / CLOCKS_PER_SEC);
	cluster_centers.release();
	//===================PCA raw data 3 dim=======================//
	rDim = 1;
	reduceDimPCA(model, rDim, components, result);
	normalize(result.col(0),result.col(0),0,1,NORM_MINMAX); //normalize to 0-1
	Mat raw_data_3D = Mat::zeros(result.rows,3,CV_32F);
	for(int i=0;i<result.rows;i++)
	{
		float r,g,b;
		gray2rgb(result.at<float>(i,0),r,g,b);
		raw_data_3D.at<float>(i,0) = r;
		raw_data_3D.at<float>(i,1) = g;
		raw_data_3D.at<float>(i,2) = b;
	}
	output_mat_as_txt_file("raw_data_3D.txt",raw_data_3D);
	//raw_data_3D = result.clone();
	//for(int i=0;i<result.cols;i++)
	//	normalize(result.col(i),raw_data_3D.col(i),0,1,NORM_MINMAX); //normalize to 0-1
	model.release();

}

void Preprocessing_Data::sort_by_color(int k,Mat& rgb_mat2,Mat& cluster_centers, Mat& cluster_tag)
{
	vector< vector<float> > rgb_vector;
	for(int i=0;i<k;i++)
	{
		vector<float> rgb;
		for(int j=0;j<3;j++)
		{
			rgb.push_back(rgb_mat2.at<float>(i,j));
		}
		rgb_vector.push_back(rgb);
	}

	class cluster_info{
	public:
		int key;
		vector<float> rgb_vec;
		Mat lab;
		Mat lab_1D;
		
		cluster_info(int k,vector<float> rgb,Mat lab_mat,Mat lab_1D_mat){
			key = k;
			rgb_vec = rgb;
			lab = lab_mat.clone();
			lab_1D = lab_1D_mat.clone();
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

			float R1 = c1.rgb_vec[0];
			float G1 = c1.rgb_vec[1];
			float B1 = c1.rgb_vec[2];
			float R2 = c2.rgb_vec[0];
			float G2 = c2.rgb_vec[1];
			float B2 = c2.rgb_vec[2];
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
				///////////////////////////////////////////////
				lab_dist.at<double>(i,j) += abs( lab.at<float>(i,t) - lab.at<float>(j,t) );
			}
		}
	}

	output_mat_as_csv_file_double("lab_dist.csv",lab_dist);
	Mat lab_1D = MDS(lab_dist,1).clone();
	output_mat_as_csv_file_double("lab_1D.csv",lab_1D);

	vector< cluster_info > cluster_vec;
	for(int i=0;i<k;i++)
	{
		cluster_vec.push_back( cluster_info( i,rgb_vector[i],lab.row(i),lab_1D.row(i) ) );
	}

	sort(cluster_vec.begin(), cluster_vec.end(), sort_by_rgb());

	
	//for(int i=0;i<cluster_vec.size();i++)
	//{
	//	cout << cluster_vec[i].key << " ";
	//}
	
	output_mat_as_csv_file("cluster_center_old.csv",cluster_centers);
	output_mat_as_csv_file_int("cluster_tag_old.csv",cluster_tag);
	Mat cluster_centers_old = cluster_centers.clone();
	Mat rgb_mat2_old = rgb_mat2.clone();
	Mat cluster_tag_old = cluster_tag.clone();
	for(int i=0;i<k;i++)
	{
		int new_tag = cluster_vec[i].key;
		cluster_centers_old.row(new_tag).copyTo(cluster_centers.row(i));
		rgb_mat2_old.row(new_tag).copyTo(rgb_mat2.row(i));
		//注意:row的複製不能用rgb_mat2.row(i) = rgb_mat2_old.row(new_tag).clone();!!!!!!!
	}
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

void Preprocessing_Data::output_mat_as_txt_file(char file_name[],Mat mat)
{
	ofstream fout(file_name);
	fout << mat << endl;

	fout.close();
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

Mat Preprocessing_Data::Gaussian_filter(int attribute_title[])
{
	Mat Gaussian_filter_mat(raw_data.size(),11, CV_32F);

	//Apply Gaussian filter to raw data(0~8)
	for(int i=0;i<raw_data.size();i++)
	{
		for(int j=0;j<11;j++)
		{
			Gaussian_filter_mat.at<float>(i,j) = raw_data[i][attribute_title[j]];
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

Mat Preprocessing_Data::set_matrix(int attribute_title[],int attribute_title_size)
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
		normalize_mat.col(i) = normalize_column(handle_mat_transpose.col(i)).clone();
		//normalize(handle_mat2.col(i),normalize_mat.col(i),0,1,NORM_MINMAX);

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

	output_mat_as_csv_file_int("histogram.csv",histogram);

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
		float base = 0;
		for(int j=0;j<k;j++)
		{
			//Ev.row(i) += adjust_weight[i][j]*cluster_centers.row(j);
			Ev.row(i) += (histogram.at<int>(i,j)/600.0)*cluster_centers.row(j);
		}
		//Ev.row(i)/=600;
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
			new_weight[i][t] += mdg(model.row(j),dim,c_list[t].cov,mean.row(t));
			total_weight[i] += new_weight[i][t];
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
				histo_coeff.at<double>(i,j) = sqrt(1.0-BC);
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
	Mat lab_centroid = compute_centroid(lab_mat);
	//lab vertices 3 axis of PCA
	Mat lab_components,lab_coeff;
	rDim = 3;
	reduceDimPCA(lab_mat, rDim, lab_components, lab_coeff); //PCA 3dim->3dim (for principal component)
	cout << "lab_components " << endl << lab_components << endl;

	//////////////////////////////////////////////////////////////////////////////////////////////
	vector<float> move_vector;
	for(float k=-20.0;k<=20.0;k+=1.0)
		move_vector.push_back(k);

	vector<float> scale_vector;
	for(float k=1.0;k<=100.0;k+=1.0)
		scale_vector.push_back(k);

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

						//for(int i=0;i<align_mat.rows;i++)
						//{
						//	if( (align_mat.at<float>(i,0)<luminance_threshold) || (align_mat.at<float>(i,0)>90.0) )
						//	{
						//		flag = false;
						//		break;
						//	}
						//}

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
	Mat output_mat = col_mat;
	double min,max;
	minMaxLoc(output_mat, &min, &max);
	for(int i=0;i<col_mat.rows;i++)
	{
		//output_mat.at<float>(i,0) = ( col_mat.at<float>(i,0) - min ) / (max - min);
		output_mat.at<float>(i,0) = col_mat.at<float>(i,0) / max;
	}

	return output_mat;
}
