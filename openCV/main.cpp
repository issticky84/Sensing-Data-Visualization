#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include <highgui.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include "cv.h"
#include <tapkee/tapkee.hpp>
#include <tapkee/callbacks/dummy_callbacks.hpp>
#include <opencv2/core/eigen.hpp> //cv2eigen
#include <FTGL/ftgl.h>
#include "glut.h"
#include <fstream>

using namespace std; 
using namespace cv;
using namespace tapkee;
using namespace Eigen;


#define LENGTH 1000
#define FONT_FILE "arial.ttf"
FTPixmapFont font(FONT_FILE);

char file_csv_data[] = "../../../../csv_data/BigData_20141121_0723_new.csv";
int windowSize[2];
char quote[24][80];
float scale_amount = 0.0;
float x_amount=0.0,y_amount=0.0,z_amount=0.0;
vector<float> draw_color(3);
vector < vector<float> > raw_data;
vector <int> hour_data;
Mat histogram;//int
Mat rgb_mat;//float
Mat position;//double

struct  RECTANGLE
{
	float x,y,h,w;
};

void output_mat_as_file(char file_name[],Mat mat)
{
	ofstream fout(file_name);
	fout << mat;
}

/**
 * 從data計算平均與covariance matrix
 * http://en.wikipedia.org/wiki/Covariance_matrix#Definition
 *
 */
void calcCovMat(Mat& data, Mat& mean, Mat& cov){
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
void reduceDimPCA(Mat& data, int rDim, Mat& components, Mat& result){
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

void read_raw_data()
{
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
	//29:Year,30:Hour,31:Minute,32:second,33:millisecond

	cout << "Csv Data Size: " << raw_data.size() <<endl;
	//cout << raw_data[0].size() << endl;

	fclose(csv_file);
}

float degtorad(float deg)
{
	float rad = deg *3.14159265 / 180;
	return rad;
}

float norm_value(float v1,float v2,float v3)
{
	return sqrt(v1*v1 + v2*v2 + v3*v3);
}

float DistanceOfLontitudeAndLatitude(float lat1,float lat2,float lon1,float lon2)
{
	float R = 6371; //km
	float theta1 = degtorad(lat1);
	float theta2 = degtorad(lat2);
	float delta_theta = degtorad(lat2-lat1);
	float delta_lumda = degtorad(lon2-lon1);
	float a = sin(delta_theta/2) * sin(delta_theta/2) + cos(theta1) * cos(theta2) * sin(delta_lumda/2) * sin(delta_lumda/2);
	float c = 2 * atan2((double)sqrt(a),(double)sqrt(1.0-a));
	float d = R * c;

	if(d>1.0) 
		return 0.01;
	else 
		return d;
}

Mat set_matrix(int attribute_title[],int time_title[],int attribute_title_size)
{
	Mat handle_mat(raw_data.size(),4, CV_32F);

	//time data & hour data
	int hour_index = time_title[1];
	int time_step_amount = floor(raw_data.size()/600.0);
	hour_data.resize(time_step_amount);
	int t = 0;
	for(int i=0;i<time_step_amount;i++)
	{
		hour_data[i] = raw_data[t][hour_index];
		t += 600;
	}

	//Compute norm
	for(int i=0;i<raw_data.size();i++)
	{
		handle_mat.at<float>(i,0) = norm_value(raw_data[i][attribute_title[0]],raw_data[i][attribute_title[1]],raw_data[i][attribute_title[2]]);
		handle_mat.at<float>(i,1) = norm_value(raw_data[i][attribute_title[3]],raw_data[i][attribute_title[4]],raw_data[i][attribute_title[5]]);
		handle_mat.at<float>(i,2) = norm_value(raw_data[i][attribute_title[6]],raw_data[i][attribute_title[7]],raw_data[i][attribute_title[8]]);
	}

	//Compute latitude & longitude
	int lat_index = attribute_title[9];
	int lon_index = attribute_title[10];
	for(int i=0;i<raw_data.size();i++)
	{
		if(i==0)
			handle_mat.at<float>(i,3) = 0.0;
		else
		{
			handle_mat.at<float>(i,3) = DistanceOfLontitudeAndLatitude(raw_data[i-1][lat_index],raw_data[i][lat_index],raw_data[i-1][lon_index],raw_data[i][lon_index]);
		}
	}

	//cout << handle_mat << endl;
	Mat normalize_mat = handle_mat;
	for(int i=0;i<handle_mat.cols;i++)
		normalize(handle_mat.col(i),normalize_mat.col(i),0,1,NORM_MINMAX);
	//cout << normalize_mat << endl;

	return normalize_mat;

}

void voting(int k,Mat cluster_tag,int row_size)
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

	//return histogram;
}

Mat Position_by_MDS(Mat cluster_centers,int k)
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

	int time_step_amount = histogram.rows;
	Mat histo_coeff = Mat::zeros(time_step_amount,time_step_amount,CV_64F);
	for(int i=0;i<time_step_amount;i++)
		for(int j=0;j<time_step_amount;j++)
			for(int t=0;t<k;t++)
			{
				histo_coeff.at<double>(i,j) += wi.at<float>(0,t)*abs(histogram.at<int>(i,t)-histogram.at<int>(j,t));
			}
	Matrix<double,Dynamic,Dynamic> histo_coeff_EigenType;//The type pass to Tapkee must be "double" not "float"
	cv2eigen(histo_coeff,histo_coeff_EigenType);
	TapkeeOutput output = tapkee::initialize() 
						   .withParameters((method=MultidimensionalScaling,target_dimension=1))
						   .embedUsing(histo_coeff_EigenType);
	Mat MDS_mat; //Type:double  
	eigen2cv(output.embedding,MDS_mat);
	normalize(MDS_mat.col(0),MDS_mat.col(0),0,1000,NORM_MINMAX);//normalize to 0~1000	
	//MDS_mat = MDS_mat.mul(10);

	return MDS_mat;
}

int move_state = 0;
int old_x,old_y;
void mouse(int button, int state, int x, int y)
{ 
	if(state == GLUT_DOWN)
	{
		if(button == 3)
		{
			scale_amount+=0.1;
			glutPostRedisplay();
		}
		else if(button == 4)
		{
			scale_amount-=0.1;
			glutPostRedisplay();
		}

		else if(button == GLUT_LEFT_BUTTON)
		{
			old_x = x;
			old_y = y;
		}
	}
	else if(state == GLUT_UP)
	{
			old_x = x;
			old_y = y;	
	}
}

void motion(int x, int y)
{
		x_amount = (x - old_x);
		y_amount = (y - old_y);
}

void DrawRectWithOpenGL(RECTANGLE* pRect)
{
	glPushMatrix();

	glBegin(GL_QUADS);
	glColor3f(draw_color[0],draw_color[1],draw_color[2]);
	glVertex3f(pRect->x,pRect->y,0);
	glVertex3f(pRect->x+pRect->w,pRect->y,0);
	glVertex3f(pRect->x+pRect->w,pRect->y+pRect->h,0);
	glVertex3f(pRect->x,pRect->y+pRect->h,0);
	glEnd();
	
	glPopMatrix();
}

void DrawText_FTGL(int l,int x, int y)
{
	glPushMatrix();

	glTranslatef(x, y-30, 0);
	glScalef(1.0+scale_amount,1.0+scale_amount,1.0+scale_amount);
	font.FaceSize(20);
	glColor3f(1.0, 1.0, 1.0);
	glRasterPos2f(0 , 0 + font.LineHeight());
	font.Render(quote[l]);

	glPopMatrix();
}

void reshape(GLsizei w, GLsizei h)
{
	windowSize[0] = w;
	windowSize[1] = h;
}

void time_string()
{
    strcpy(quote[0],"00:00");
	strcpy(quote[1],"01:00");
	strcpy(quote[2],"02:00");
	strcpy(quote[3],"03:00");
	strcpy(quote[4],"04:00");
	strcpy(quote[5],"05:00");
	strcpy(quote[6],"06:00");
	strcpy(quote[7],"07:00");
	strcpy(quote[8],"08:00");
	strcpy(quote[9],"09:00");
	strcpy(quote[10],"10:00");
    strcpy(quote[11],"11:00");
	strcpy(quote[12],"12:00");
	strcpy(quote[13],"13:00");
	strcpy(quote[14],"14:00");
	strcpy(quote[15],"15:00");
	strcpy(quote[16],"16:00");
	strcpy(quote[17],"17:00");
	strcpy(quote[18],"18:00");
	strcpy(quote[19],"19:00");
	strcpy(quote[20],"20:00");
    strcpy(quote[21],"21:00");
	strcpy(quote[22],"22:00");
	strcpy(quote[23],"23:00");
}

void display()
{
	glClearColor(0.0, 0.0, 0.0, 0.0);  //Set the cleared screen colour to black
	glViewport(0, 0, windowSize[0], windowSize[1]);   //This sets up the viewport so that the coordinates (0, 0) are at the top left of the window

	//Set up the orthographic projection so that coordinates (0, 0) are in the top left
	//and the minimum and maximum depth is -10 and 10. To enable depth just put in
	//glEnable(GL_DEPTH_TEST)
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, windowSize[0], windowSize[1], 0, -10, 10);

	//Back to the modelview so we can draw stuff 
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //Clear the screen and depth buffer

	// draw histogram visualization
	glTranslatef(0.0+x_amount,0.0+y_amount,0.0+z_amount);
	glScalef(0.6+scale_amount,0.6+scale_amount,0.6+scale_amount);

	int y_coord = 1400;
	int pixels;
	int current_hour;
	int last_hour = -1;
	int t=0;
	for(int i=0;i<position.rows;++i)
	{
		current_hour = hour_data[i];
		if(current_hour!=last_hour)
		{
			draw_color[0] = 1; draw_color[1] = 1; draw_color[2] = 1;
			RECTANGLE *line;
			line = new RECTANGLE();
			line->h = 3;
			line->w = 1050;
			line->x = 0;
			line->y = y_coord - 5;
			DrawRectWithOpenGL(line);
			DrawText_FTGL(current_hour,20,y_coord-15);
			delete(line);
			t++;
			y_coord-=10;
		}		

		int start = 0;
		for(int k=0;k<histogram.cols;++k)
		{   
			if(histogram.at<int>(i,k)!=0)
			{
				pixels = histogram.at<int>(i,k);
				draw_color[0] = rgb_mat.at<float>(k,0); draw_color[1] = rgb_mat.at<float>(k,1); draw_color[2] = rgb_mat.at<float>(k,2);
				for(int u=start;u<start+pixels;++u)
				{
					RECTANGLE *rect;
					rect = new RECTANGLE();
					rect->h = 3;
					rect->w = 0.1;
					rect->x = 110 + position.at<double>(i,1)/10.0 + (double)u*0.1;
					rect->y = y_coord;
					DrawRectWithOpenGL(rect);	
					delete(rect);
				}
				start += pixels;
			}		
		}

		y_coord-=5;

		last_hour = current_hour;
	}

	glutSwapBuffers(); // Swap front and back buffers
	
} 

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitWindowSize(900, 900);
	glutInitWindowPosition(0, 0);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB); // GLUT_DOUBLE : Bit mask to select a double buffered window
	glutCreateWindow("Visualization");
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	//glutKeyboardFunc(keyboard);
	//glutMouseFunc(mouse);
	//glutMotionFunc(motion);

	//=================Read CSV file====================//
	clock_t begin = clock();
	read_raw_data();
	clock_t end = clock();
	printf("Read csv elapsed time: %f\n",double(end - begin) / CLOCKS_PER_SEC);
	//==================================================//

	int attribute_title_size = 11;
	int attribute_title[] = {4,5,6,7,8,9,10,11,12,22,23};//(gravity_x,gravity_y,gravity_z,linear_acc_x),(linear_acc_x,linear_acc_y,linear_acc_z),(gyro_x,gyro_y,gyro_z),(latitude,longitude)
	int time_title[] = {29,30,31,32,33};//hour(30),minute(31)
	
	//============Setting matrix for K-means============//
	clock_t begin2 = clock();
	Mat model = set_matrix(attribute_title,time_title,attribute_title_size).clone();
	clock_t end2 = clock();
	printf("Setting matrix elapsed time: %f\n",double(end2 - begin2) / CLOCKS_PER_SEC);
	//==================================================//
	output_mat_as_file("model.txt",model);
    int k = 50; 
    Mat cluster_tag; //Tag:0~k-1
    int attempts = 2;//應該是執行次數
	Mat cluster_centers;
	//==============K means clustering==================//
    
	//使用k means分群
	clock_t begin3 = clock();
	kmeans(model, k, cluster_tag,TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 0.0001), attempts,KMEANS_PP_CENTERS,cluster_centers);
	clock_t end3 = clock();
    //TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 1),  這裡有三個參數，決定k-means何時結束，第二個參數是指迭代最大次數，第三個參數是精確度多少，第一個參數是指依照前兩個參數的哪一個為準，以範例中就是兩者都參照，以 or 的方式決定
	printf("Kmeans elapsed time: %f\n",double(end3 - begin3) / CLOCKS_PER_SEC);
	//==================================================//
	
	voting(k,cluster_tag,model.rows); // Type: int
	cluster_tag.release();
	model.release();
	//===================PCA RGB=======================//
	Mat components, result;
	int rDim = 3;
	reduceDimPCA(cluster_centers, rDim, components, result);
	rgb_mat = result.clone();
	for(int i=0;i<result.cols;i++)
		normalize(result.col(i),rgb_mat.col(i),0,1,NORM_MINMAX); //normalize to 0-1
	output_mat_as_file("rgb_mat.txt",rgb_mat);
	//===============Position (MDS)=====================//
	clock_t begin4 = clock();
	position = Position_by_MDS(cluster_centers,k).clone(); //Type:double
	clock_t end4 = clock();
	printf("MDS Position elapsed time: %f\n",double(end4 - begin4) / CLOCKS_PER_SEC);
	cluster_centers.release();
	//==================================================//

	time_string();

	raw_data.clear();

	glutMainLoop();

	return 0;
}