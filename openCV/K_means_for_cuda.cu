
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cv.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <math.h>

using namespace std;
#define MAX_DATA_DIMENSION 40
#define MAX_ITERATION     500
#define BLOCK_DIM         1024
//#define CLUSTER_NUM       2
#define THRESHOLD    0.000001

void DataSizeRead(char* fileName,int &dataNum,int &dimNum);
void FileRead(char* fileName,float *point,float *minDist,int *clusterIdx);
void Init(float *point,int dataNum,float *cluster,float *old_cluster,int *clusterSz,int *old_clusterSz,int clusterNum,int dimNum);
void initKmeans(float *point,int *pClusterIdx,float *pMinDistance,int dataNum,float *cluster,int clusterNum,int dimNum);
__global__ void Kmeans(float *point,int *pClusterIdx,float *pMinDistance,int dataNum,float *cluster,int clusterNum,int dimNum);
void CheckConverge(float *cluster,float *old_cluster,int *clusterDataSize,int *old_clusterDataSize,int clusterNum,bool &conv,int dimNum);
void AdjustCentroid(float *point,int *pClusterIdx,float *pMinDistance,int dataNum,float * cluster,int *clusterDataSize,int clusterNum,int dimNum);
__host__ __device__ double Distance_Compute(float *lhs,float *rhs,int indexl,int indexr,int dimNum);
void printResult(float* point,int *pclusterIdx,int dataNum,int clusterNum,int dimNum);
double ms_time(){ return (double)clock()/CLOCKS_PER_SEC*1000.0;}

__device__ float Distance2(float* objects, float *clusters, int numPoint, int dimNum, int numClusters,int objectId, int clusterId);

texture<float, 1, cudaReadModeElementType> texRefX;
//int main(int argc, char* argv[])
void cuda_kmeans(int k, cv::Mat& cluster_index, cv::Mat& cluster_center)
{
	char *path = "model.csv" ;
	int dataPointSize;
	int dataDimension;
	float  *DataPoints;
	float  *MinDistance;
	int    *ClusterIndex;
	float  *NewClusterCenter;
	float  *OldClusterCenter;
	int    *NewClusterDataSize;
	int    *OldClusterDataSize;
	clock_t start,end;
	bool converge=0;
	int iteration=0;
	int GRID_DIM;  
	int clusterNum = k; //
	//if(argc >=2) 
	//{
	//	path = argv[1];
	//	if(argc>2)
	//		clusterNum = atoi(argv[2]);
	//}
	// pre-processing 
	//cout << "loading file ..." << endl;
	DataSizeRead(path,dataPointSize,dataDimension); // check right 
	GRID_DIM = dataPointSize % BLOCK_DIM ? dataPointSize/BLOCK_DIM +1: dataPointSize/BLOCK_DIM;  
	//cout << "initializing data ..." << endl;
	// points and clusters memory cpy ;
	DataPoints   = new float[dataPointSize*dataDimension]; 
	MinDistance  = new float[dataPointSize];
	ClusterIndex = new int  [dataPointSize];
	// [][dataDimension]   for min_distance
	// [][dataDimension+1] for cluster index
	NewClusterCenter = new float[clusterNum*dataDimension];
	OldClusterCenter = new float[clusterNum*dataDimension];
	NewClusterDataSize  = new int  [clusterNum];
	OldClusterDataSize  = new int  [clusterNum];
	// [][dataDimension]   for point size ;
	
	//cout <<" reading file ... " << endl;
	//  file read ; 
	FileRead(path,DataPoints,MinDistance,ClusterIndex);// checked; 
	
	
	float  *GPU_DataPoints;
	float  *GPU_MinDistance;
	int    *GPU_ClusterIndex;
	float  *GPU_NewClusterCenter;
	float  *GPU_OldClusterCenter;
	int    *GPU_NewClusterDataSize;
	int    *GPU_OldClusterDataSize;
	int    *GPU_iteration;
	bool   *GPU_converge;
	
	
	cudaMalloc((void**)&GPU_iteration       , sizeof(int)*1);
	cudaMalloc((void**)&GPU_converge   		, sizeof(bool)*1);
	cudaMalloc((void**)&GPU_DataPoints 		, sizeof(float)*dataPointSize*dataDimension);
	cudaMalloc((void**)&GPU_MinDistance 	, sizeof(float)*dataPointSize*1			   );
	cudaMalloc((void**)&GPU_ClusterIndex 	, sizeof(int)  *dataPointSize*1			   );
	cudaMalloc((void**)&GPU_NewClusterCenter, sizeof(float)*clusterNum*dataDimension);
	cudaMalloc((void**)&GPU_OldClusterCenter, sizeof(float)*clusterNum*dataDimension);
	cudaMalloc((void**)&GPU_NewClusterDataSize, sizeof(int)*clusterNum);
	cudaMalloc((void**)&GPU_OldClusterDataSize, sizeof(int)*clusterNum);
	
	
	
	//cout << "initializing cluster center ... " << endl;
	Init(DataPoints,dataPointSize,NewClusterCenter,OldClusterCenter,NewClusterDataSize,OldClusterDataSize,clusterNum,dataDimension);
	cudaMemcpy(GPU_DataPoints			,DataPoints			,sizeof(float)*dataPointSize*dataDimension,cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_MinDistance			,MinDistance		,sizeof(float)*dataPointSize*1			  ,cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_ClusterIndex			,ClusterIndex		,sizeof(int)*dataPointSize*1			  ,cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_NewClusterCenter		,NewClusterCenter	,sizeof(float)*clusterNum*dataDimension	  ,cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_OldClusterCenter		,OldClusterCenter	,sizeof(float)*clusterNum*dataDimension	  ,cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_NewClusterDataSize	,NewClusterDataSize	,sizeof(float)*clusterNum*1				  ,cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_OldClusterDataSize	,OldClusterDataSize	,sizeof(float)*clusterNum*1				  ,cudaMemcpyHostToDevice);
	
	cudaBindTexture(0, texRefX, GPU_DataPoints, dataPointSize*dataDimension*sizeof(float));
	

	const unsigned int clusterBlockSharedDataSize =
        BLOCK_DIM * sizeof(unsigned char) +
        clusterNum * dataDimension * sizeof(float);

    cudaDeviceProp deviceProp;
    int deviceNum;
    cudaGetDevice(&deviceNum);
    cudaGetDeviceProperties(&deviceProp, deviceNum);

    if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
        printf("WARNING: Your CUDA hardware has insufficient block shared memory.\n");
    }



	//cout << " starting k-means ... " << endl;
	start = ms_time();
	initKmeans(DataPoints,ClusterIndex,MinDistance,dataPointSize,NewClusterCenter,clusterNum,dataDimension);
	
	while(iteration++ <MAX_ITERATION)
	{
		//cout << "iter:" << iteration << endl;
		AdjustCentroid(DataPoints,ClusterIndex,MinDistance,dataPointSize,NewClusterCenter,NewClusterDataSize,clusterNum,dataDimension);
		CheckConverge(NewClusterCenter,OldClusterCenter,NewClusterDataSize,OldClusterDataSize,clusterNum,converge,dataDimension);
		if(converge)
		{
			//cout << "clustering converged " << endl;
			break;
		}
		cudaMemcpy(GPU_NewClusterCenter,NewClusterCenter,sizeof(float)*clusterNum*dataDimension,cudaMemcpyHostToDevice);
		Kmeans<<<GRID_DIM,BLOCK_DIM, clusterBlockSharedDataSize>>>(GPU_DataPoints,GPU_ClusterIndex,GPU_MinDistance,dataPointSize,GPU_NewClusterCenter,clusterNum,dataDimension);
		cudaMemcpy(ClusterIndex,GPU_ClusterIndex,sizeof(int)*dataPointSize*1 ,cudaMemcpyDeviceToHost);
	}
	end = ms_time();
	cout << "total iteration = " << iteration << endl;
	cout << " execution time " << double(end-start)  << "ms"<<endl;
	if(clusterNum <= 20)
	{
		printResult(DataPoints,ClusterIndex ,dataPointSize,clusterNum,dataDimension);
	}

	//ofstream fout("ClusterIndex.csv");
	//for(int i=0;i<dataPointSize;i++)
	//	fout<<ClusterIndex[i]<<endl;
	//fout.close();


	//ofstream fout2("ClusterCenter.txt");
	//for(int i=0;i<clusterNum;i++)
	//{
	//	for(int j=0;j<dataDimension;j++)
	//	{
	//		fout2<<NewClusterCenter[i*dataDimension+j]<<" ";
	//	}
	//    fout2<<endl;
	//}
	//fout2.close();

	for(int i=0;i<dataPointSize;i++)
		cluster_index.at<int>(i,0) = ClusterIndex[i];

	for(int i=0;i<clusterNum;i++)
	{
		for(int j=0;j<dataDimension;j++)
		{
			cluster_center.at<float>(i,j) = NewClusterCenter[i*dataDimension+j];
		}
	}

	//  release memory 
	delete [] (DataPoints);
	delete [] (MinDistance);
	delete [] (ClusterIndex);
	delete [] (NewClusterCenter);
	delete [] (OldClusterCenter);
	delete [] (NewClusterDataSize);
	delete [] (OldClusterDataSize);
	
	//system("pause");
	//return 0;
}

void DataSizeRead(char* fileName,int &dataNum,int &dimNum)
{
	int lineNum =0;
	int paraNum =0;
	char input[1024];
	fstream fs;
	char *pch;
	const char *token = ",\n";
	fs.open(fileName,ios::in);
	if(!fs.is_open()) cerr << "can't open file" << endl;
	while( fs.getline(input,1024))
	{
		if(lineNum==0)
		{
			pch = strtok(input,token);
			while(pch!=NULL)
			{
				paraNum++;
				pch = strtok(NULL,token);
			}
		}
		lineNum++;
	}
	fs.close();
	dataNum = lineNum;
	dimNum = paraNum;
}
void FileRead(char* fileName,float *point,float *minDist,int *clusterIdx)
{
	fstream file;
	const char *token = ",\n";
	char * pch;
	char input[1024];
	int numIndex=0;
	int lineCount=0;
	file.open(fileName,ios::in);
	while(file.getline(input,1024))
	{
		pch = strtok(input,token);
		while(pch != NULL)
		{
			point[numIndex++] = atof(pch);
			pch = strtok(NULL,token);
		}
		minDist[lineCount] = FLT_MAX;
		clusterIdx[lineCount] = -1;
		lineCount++;
	}
}
void Init(float *point,int dataNum,float *cluster,float *old_cluster,int *clusterSz,int *old_clusterSz,int clusterNum,int dimNum)
{
	if( dataNum < clusterNum) 
		cout << " your data number is less than your cluster number .. \n";
	int indexP,indexC;
	for(int pIndex =0;pIndex < clusterNum; pIndex++)
	{
		for(int dimIndex=0;dimIndex < dimNum; dimIndex++)
		{
			indexP = pIndex*dimNum + dimIndex;
			indexC = pIndex*dimNum + dimIndex;
			cluster[indexC]     =  point[indexP];
			old_cluster[indexC] =  point[indexP];
		}
		clusterSz[pIndex]     = 0.0;
		old_clusterSz[pIndex] = 0.0;
	}
}
void initKmeans(float *point,int *pClusterIdx,float *pMinDistance,int dataNum,float *cluster,int clusterNum,int dimNum)
{
	// minDistanceIndex = dimNum;
	// cluster index    = dimNum+1; 
	int indexC,indexP;
	for(int pIndex=0;pIndex <dataNum;pIndex++)
	{
		float minDistance = FLT_MAX;
		int clusterIndex  =     -1 ;
		
		indexP = pIndex*dimNum;
		// each row contain dimNum+2 col 
		for(int cIndex=0;cIndex < clusterNum;cIndex++)
		{
			indexC = cIndex*dimNum;
			// each row contain dimNum+1 col
			float distance = Distance_Compute(point,cluster,indexP,indexC,dimNum);
			if(minDistance > distance)
			{
				minDistance  = distance;
				clusterIndex = cIndex;
			}
		}
		if(clusterIndex!=-1)
		{
			pMinDistance[pIndex] = minDistance;
			pClusterIdx[pIndex] = clusterIndex;
		}
	}
}


__global__ void Kmeans(float *point,int *pClusterIdx,float *pMinDistance,int dataNum,float *cluster,int clusterNum,int dimNum)
{
	#define xx_point(k) tex1Dfetch(texRefX, k)
	
	extern __shared__ char sharedMemory[];
	float *share_clusters = (float *)(sharedMemory + clusterNum*dimNum);
	//__shared__ float share_clusters[11*16];

	/*for (int i = threadIdx.x; i < clusterNum; i += blockDim.x) {
        for (int j = 0; j < dimNum; j++) {
            share_clusters[dimNum * j + i] = cluster[dimNum * j + i];
        }
    }*/
    int pIndex = blockIdx.x*blockDim.x+threadIdx.x;
    int count = 0;
    for (int i = threadIdx.x; i < clusterNum*dimNum; i += blockDim.x)
    {
    	share_clusters[i] = cluster[i];
    	//count++;
    }
    __syncthreads();
    /*for (int i = 0; i < clusterNum*dimNum; i++)
    {
    	share_clusters[i] = cluster[i];
    }*/
    //__syncthreads();

	int indexC,indexP;
	//int gtid = blockIdx.x*blockDim.x+threadIdx.x;
	//int pIndex = gtid;
	//int pIndex = blockIdx.x*blockDim.x+threadIdx.x;
//printf("share_clusters[0] = %f, cluster[0] = %f\n", share_clusters[0], cluster[0]);

	if(pIndex < dataNum)
	//for(int pIndex=0;pIndex <dataNum;pIndex++)
	{
		float minDistance = FLT_MAX;
		int clusterIndex  =     -1 ;
		
		indexP = pIndex*dimNum;
		// each row contain dimNum+2 col 
		for(int cIndex=0;cIndex < clusterNum;cIndex++)
		{
			indexC = cIndex*dimNum;
			// each row contain dimNum+1 col
			//float distance = Distance(point,cluster,indexP,indexC,dimNum);
			float distance = 0.0;
			for(int dimIdx=0;dimIdx <dimNum;dimIdx++)
			{
				//distance += (point[indexP+dimIdx]-cluster[indexC+dimIdx])*
				//			(point[indexP+dimIdx]-cluster[indexC+dimIdx]);
				distance += (xx_point(indexP+dimIdx)-share_clusters[indexC+dimIdx])*
							(xx_point(indexP+dimIdx)-share_clusters[indexC+dimIdx]);
			}
							//(float* objects, float *clusters, int numPoint, int dimNum, int numClusters,int objectId, int clusterId)
			//float distance = Distance2(point, cluster, dataNum, dimNum, clusterNum, pIndex, cIndex);
			//float distance = Distance(point, share_clusters, indexP, indexC, dimNum);
			//float distance = Distance(xx_point, cluster, indexP, indexC, dimNum);
			//printf("share_clusters = %f, clusters = %f\n", distance, distance2);
			if(minDistance > distance)
			{
				minDistance  = distance;
				clusterIndex = cIndex;
			}
		}
		if(clusterIndex!=-1)
		{
			pMinDistance[pIndex] = minDistance;
			pClusterIdx[pIndex] = clusterIndex;
		}
	}
	#undef xx
	__syncthreads();
}
void CheckConverge(float *cluster,float *old_cluster,int *clusterDataSize,int *old_clusterDataSize,int clusterNum,bool &conv,int dimNum)
{
	conv = 1;
	int indexC;
	for(int cIndex=0;cIndex < clusterNum;cIndex++)
	{
		indexC = cIndex*dimNum;
		// each row contains dimNum+1 col;
	
		//  data size 不同 一定不同
		float sizeDiff =   clusterDataSize[cIndex]>old_clusterDataSize[cIndex]
						?clusterDataSize[cIndex]-old_clusterDataSize[cIndex]
						:old_clusterDataSize[cIndex]-clusterDataSize[cIndex];
		double distance = Distance_Compute(cluster,old_cluster,indexC,indexC,dimNum);

		if( sizeDiff > 0.5  ||
			 distance > THRESHOLD )
		{
			conv =0;
			break;
		}
		
	}
	if(conv) return;

	for(int cIndex=0;cIndex < clusterNum;cIndex++)
	{
		indexC = cIndex*dimNum;
		// each row contains dimNum+1 col;
		for(int dimIndex =0;dimIndex <dimNum;dimIndex++)
		{
			old_cluster[indexC+dimIndex] = cluster[indexC+dimIndex];
		}
		old_clusterDataSize[cIndex] = clusterDataSize[cIndex];
	}
	// check 完 沒有converge 再update old cluster ; 
}
void AdjustCentroid(float *point,int *pClusterIdx,float *pMinDistance,int dataNum,float * cluster,int *clusterDataSize,int clusterNum,int dimNum)
{
	int indexC,indexP;
	for(int cIndex=0;cIndex < clusterNum ;cIndex++)
	{
		indexC = cIndex*dimNum;
		//for each row contains dimNum+1 cols;
		for(int dimIndex=0;dimIndex< dimNum; dimIndex++)
		{
			// reset all values 
			cluster[indexC+dimIndex] = 0.0;
		}
		clusterDataSize[cIndex] = 0;
	}
	for(int pIndex=0;pIndex < dataNum;pIndex++)
	{
		
		int cIndex = pClusterIdx[pIndex];
		indexC     =  cIndex *dimNum;
		indexP = pIndex * dimNum;
		// for each row contains dimNum+1 cols;
		for(int dimIndex=0;dimIndex< dimNum; dimIndex++)
		{
			cluster[indexC+dimIndex] += point[indexP+dimIndex];
		}
		clusterDataSize[cIndex] += 1 ;
	}
	for(int cIndex=0;cIndex < clusterNum ;cIndex++)
	{
		indexC = cIndex*dimNum;
		// each row contains dimNum+1 cols; 
		for(int dimIndex=0;dimIndex< dimNum; dimIndex++)
		{
			cluster[indexC+dimIndex] /= clusterDataSize[cIndex];
		}
	}
	/*
	for(int pIndex =0;pIndex < dataNum;pIndex++)
	{
		int cIndex = pClusterIdx[pIndex];
		indexP     = pIndex*dimNum;
		indexC     = cIndex *dimNum;
		// for each row contains dimNum+1 cols;
		float distance = Distance(point,cluster,indexP,indexC,dimNum);
		pMinDistance[pIndex]=distance;
	}*/

}
__host__ __device__ double Distance_Compute(float* lhs,float *rhs,int indexl,int indexr,int dimNum)
{
	double distance =0.0;
	for(int i=0;i<dimNum;i++)
	{
		distance += (lhs[indexl+i]-rhs[indexr+i]) * (lhs[indexl+i]-rhs[indexr+i]);
	}
	return (distance);
}
__device__ float Distance2(float* objects, float *clusters, int numPoint, int dimNum, int numClusters,int objectId, int clusterId)
{
	int i;

	float ans=0.0;

    for (i = 0; i < dimNum; i++) {
        ans += (objects[numPoint * i + objectId] - clusters[numClusters * i + clusterId]) *
               (objects[numPoint * i + objectId] - clusters[numClusters * i + clusterId]);
    }

    return (ans);
}

void printResult(float* point,int *pclusterIdx,int dataNum,int clusterNum,int dimNum)
{
	char path[256];
	fstream *fs;
	fs = new fstream[clusterNum];
	int indexP;
	for(int i=0;i<clusterNum;i++)
	{
		sprintf(path,"Clustering/Cluster%d.txt",i);
		fs[i].open(path,ios::out);
	}
	for(int pIndex=0;pIndex <dataNum;pIndex++)
	{
		indexP = pIndex*dimNum;
		//for each row contains dimNum+2 cols
		int cIndex= pclusterIdx[pIndex];
		
		for(int dimIndex=0;dimIndex < dimNum-1;dimIndex++)
		{
			fs[cIndex] << point[indexP++]<<",";
		}
		fs[cIndex] << point[indexP]<< endl;
	}
	for(int i=0;i<clusterNum;i++)
		fs[i].close();

	delete [] (fs);
}