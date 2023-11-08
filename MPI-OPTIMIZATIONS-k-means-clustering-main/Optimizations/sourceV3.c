#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <math.h>
#include "mpi.h"
#define MASTER 0
/**
 * Point structure for point data
 */
typedef struct 
{
	double _x;
	double _y;
} Point;
	
/**
 * Data structure for essential data
 */
typedef struct
{
	int num_points;
	int job_size;
	int num_clusters;
} Data;

typedef struct 
{
	Point *points;
	Point *centroids;
    int *former_clusters;
    int *latter_clusters;
    int num_points;
	int num_clusters;
	int job_done;
} Parameters;

/**
 * reader function of the input file's first(for number of clusters)  
 * & second(for number of points) line
 * @param input input file handler
 * @param num_clusters pointer to return number of clusters
 * @param num_points pointer to return number of points	
 */
void readHeaders(FILE *input,int* num_clusters,int* num_points)
{
	fscanf(input,"%d\n",num_clusters);
	//printf("%d\n",*num_clusters);

	fscanf(input,"%d\n",num_points);
	//printf("%d\n",*num_points);
}

int compare(const void *x, const void *y) {
  
    Point *pointA = (Point *)x;
    Point *pointB = (Point *)y;
  
	if(pointA->_y != pointB->_y) return pointA->_y > pointB->_y;
}

/**
 * reader function of the points in the input file
 * This function must be called after  readHeaders(...) function
 * @param input input file handler
 * @param points pointer to return the array of points
 * @param num_points number of points to read
 */
void readPoints(FILE* input,Point *points,int num_points)
{
	int dex;
	for(dex=0;dex<num_points;dex++)
	{
		fscanf(input,"%lf,%lf",&points[dex]._x,&points[dex]._y);
	}
	qsort(points, num_points, sizeof(Point), compare);
}

/**
 * initializer function that randomly initialize the centroids
 * @param centroids pointer to return array of centroids
 * @param num_cluster number of clusters(so number of centroids, too)
 */
void initialize(Point* points, Point* centroids,int num_clusters)
{
	for(int dex=0;dex<num_clusters;dex++)
	{
		centroids[dex]._x=points[dex]._x;
		centroids[dex]._y=points[dex]._y;
	}
}

/**
 * initializer function that initializes the all cluster array values to -1
 * @param data pointer to return array of cluster data
 * @param num_points number of points to initialize
 */
int resetData(int *data,int num_points)
{
	int dex;
	for(dex=0;dex<num_points;dex++)
	{
		data[dex]=-1;
	}		
}
/**
 * calculate distance between two points
 * @param point1 first point
 * @param point2 second point
 * @return distance in double precision
 */
double calculateDistance(Point point1,Point point2)
{
	return (pow((point1._x-point2._x)*100,2)+pow((point1._y-point2._y)*100,2));	
}
/**
 * Wierd name but essential function; decides witch centroid is closer to the given point
 * @param point point given
 * @param centroids pointer to centroids array
 * @param num_centroids number of centroids to check
 * @return closest centroid's index in centroids array(2nd param)
 */
int whoIsYourDaddy(Point point,Point* centroids,int num_centroids)
{
	int daddy=0;
	double distance=0;
	double minDistance=calculateDistance(point,centroids[0]);
	int dex;
	
	for(dex=1;dex<num_centroids;dex++)
	{	
		distance=calculateDistance(point,centroids[dex]);
		if(minDistance>=distance)
		{
			daddy=dex;
			minDistance=distance;
		}
	}
	return daddy;
}
/**
 * Cumulative function that must be called after the closest centroid for each point is found
 * Calculates new centroids as describen in kmeans algorithm
 * @param points array of points
 * @param data array of cluster assignments
 * @param centroids return array of centroids
 * @param num_clusters number of clusters(so number of centroids)
 * @param num_points number of points 
 */
void calculateNewCentroids(Point* points,int* data,Point* centroids,int num_clusters,int num_points)
{
	Point* newCentroids=malloc(sizeof(Point)*num_clusters);
	int* population=malloc(sizeof(int)*num_clusters);
	int dex;

	for(dex=0;dex<num_clusters;dex++)
	{
		population[dex]=0;
		newCentroids[dex]._x=0;
		newCentroids[dex]._y=0;
	}	
	for(dex=0;dex<num_points;dex++)
	{
		population[data[dex]]++;
		newCentroids[data[dex]]._x+=points[dex]._x;
		newCentroids[data[dex]]._y+=points[dex]._y;
	}
	for(dex=0;dex<num_clusters;dex++)
	{
		if(population[dex]!=0.0)
		{
			newCentroids[dex]._x/=population[dex];
			newCentroids[dex]._y/=population[dex];
		}
	}
	for(dex=0;dex<num_clusters;dex++)
	{
		centroids[dex]._x=newCentroids[dex]._x;
		centroids[dex]._y=newCentroids[dex]._y;
	}	
}
/**
 * Convergence checker (see project description for further info)
 * @param former_clusters pointer to array of older cluster assignments
 * @param latter_clusters pointer to array of newer cluster assignments
 * @param num_points number of points 
 * @return -1 if not converged, 0 if converged.
 */
int checkConvergence(int *former_clusters,int *latter_clusters,int num_points)
{
	int dex;
	for(dex=0;dex<num_points;dex++)
		if(former_clusters[dex]!=latter_clusters[dex])
			return -1;
	return 0;
}

void *flagged_check(void* arg){
	Parameters* p = (Parameters*)arg;

    if(checkConvergence(p->latter_clusters,p->former_clusters, p->num_points)==0)
    {
        //printf("Converged!\n");
        p->job_done=1;
    }
    else    
    {
        //printf("Not converged!\n");
        for(int dex=0;dex<p->num_points;dex++)
			p->former_clusters[dex] = p->latter_clusters[dex];
    }
}

void *new_centroids(void* arg){
	Parameters* p = (Parameters*)arg;
	calculateNewCentroids(p->points ,p->latter_clusters ,p->centroids ,p->num_clusters ,p->num_points);
}

/**
 * main function
 * divided to two brances for master & slave processors respectively
 * @param argc commandline argument count
 * @param argv array of commandline arguments
 * @return 0 if success
 */
int main(int argc, char* argv[])
{
    int rank;
	int size;
    int num_clusters;
    int num_points;
	int dex;
	int job_size;
	int job_done=0;
	
	Point* centroids;
	Point* points;
	
	int  * slave_clusters;
	int  * former_clusters;
	int  * latter_clusters;
    	
	MPI_Init(&argc, &argv);
	
	MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	//creation of derived MPI structure
	MPI_Datatype MPI_POINT, MPI_DATA;
	MPI_Datatype type=MPI_DOUBLE;
	MPI_Datatype data=MPI_INT;
	int blocklenPoint=2, blocklenData=3;
	MPI_Aint disp=0;
	MPI_Type_create_struct(1,&blocklenPoint,&disp,&type,&MPI_POINT);
	MPI_Type_create_struct(1,&blocklenData,&disp,&data,&MPI_DATA);
	MPI_Type_commit(&MPI_POINT);
	MPI_Type_commit(&MPI_DATA);

/******************* MASTER PROCESSOR WORKS HERE******************************************************/ 
    
    double start_time = MPI_Wtime();
   	if(rank==MASTER)
  	{
        pthread_t thread1;
        pthread_t thread2;

		int flag = 1;

		//Reading the file
		Point* received_points;
		FILE *input;
		input=fopen("input.txt","r");
		readHeaders(input,&num_clusters,&num_points);
		points=(Point*)malloc(sizeof(Point)*num_points);
		readPoints(input,points,num_points);
		fclose(input);
		
		//other needed memory locations
		former_clusters=malloc(sizeof(int)*num_points);
		latter_clusters=malloc(sizeof(int)*num_points);
		slave_clusters=malloc(sizeof(int)*num_points);
		job_size=num_points/(size);
		centroids=malloc(sizeof(Point)*num_clusters);
		received_points=malloc(sizeof(Point)*num_points);
		
		//reseting and initializing to default behaviour		
		initialize(points, centroids,num_clusters);

		resetData(former_clusters,num_points);
		resetData(latter_clusters,num_points);

		//initalizing and populating the type Data
		Data data;
		data.num_points = num_points;
		data.job_size = job_size;
		data.num_clusters = num_clusters;		

		Parameters* parameters = (Parameters*) malloc(sizeof(Parameters));

		//Sending the essential data to slave processors
		MPI_Bcast(&data, 1, MPI_DATA, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(centroids, num_clusters, MPI_POINT, MASTER, MPI_COMM_WORLD);
		MPI_Scatter(points, job_size, MPI_POINT, received_points, job_size, MPI_POINT, MASTER, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);
		//printf("Sent!\n");
	
		while(flag == 1)
		{	
			latter_clusters=malloc(sizeof(int)*num_points);
            MPI_Barrier(MPI_COMM_WORLD);
			//printf("Calculation of new clusters [%d]\n",rank);
			for(dex=0;dex<job_size;dex++)
			{
				slave_clusters[dex]=whoIsYourDaddy(received_points[dex],centroids,num_clusters);
			}
            //printf("Done! [%d]\n",rank);

            MPI_Gather(slave_clusters, job_size, MPI_INT, latter_clusters, job_size, MPI_INT, MASTER, MPI_COMM_WORLD);
			//printf("Gathering clusters [%d]\n",rank);
			//MPI_Barrier(MPI_COMM_WORLD);
            //printf("Master Received\n");

			parameters->points = points;
			parameters->centroids = centroids;
			parameters->latter_clusters = latter_clusters;
			parameters->former_clusters = former_clusters;
			parameters->num_points = num_points;
			parameters->num_clusters = num_clusters;
			parameters->job_done = 0;

            pthread_create(&thread1, NULL, flagged_check, (void*)parameters);
			pthread_create(&thread2, NULL, new_centroids, (void*)parameters);

			pthread_join(thread1, NULL);
			pthread_join(thread2, NULL);

            //printf("New Centroids are done!\n");

			job_done = parameters->job_done;
			centroids = parameters->centroids;

            //Informing slaves that no more job to be done
            MPI_Bcast(&job_done, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
            if(job_done==1)
                flag = 0;

            //Sending the recently created centroids
			//printf("Sending new centroids!\n");		
            MPI_Bcast(centroids, num_clusters, MPI_POINT, MASTER, MPI_COMM_WORLD);
			//printf("Centroids sent!\n");
		}
		
		//Outputting to the output file		
		FILE* output=fopen("output.txt","w");
		fprintf(output,"%d\n",num_clusters);
		fprintf(output,"%d\n",num_points);
		for(dex=0;dex<num_clusters;dex++)
			fprintf(output,"%lf,%lf\n",centroids[dex]._x,centroids[dex]._y);
		for(dex=0;dex<num_points;dex++)
			fprintf(output,"%lf,%lf,%d\n",points[dex]._x,points[dex]._y,latter_clusters[dex]+1);
		fclose(output);
	}

/*************END OF MASTER PROCESSOR'S BRANCH -- SLAVE PROCESSORS' JOB IS TO FOLLOW ************************/
	else
	{
		//Receiving the essential data
		int flag = 1;
		Point* received_points;
		Data data;
		
		//printf("Receiving Data from master [%d]\n", rank);
		MPI_Bcast(&data    	,1           ,MPI_DATA  ,MASTER ,MPI_COMM_WORLD);

		centroids=malloc(sizeof(Point)*data.num_clusters);
		MPI_Bcast(centroids    ,data.num_clusters,MPI_POINT,MASTER, MPI_COMM_WORLD);

		//printf("part_size =%d\n",data.job_size);

		received_points=malloc(sizeof(Point)*data.job_size);
		slave_clusters=malloc(sizeof(int)*data.job_size);
		MPI_Scatter(NULL, data.job_size, MPI_POINT, received_points, data.job_size, MPI_POINT, MASTER, MPI_COMM_WORLD);
		//printf("Received [%d]\n",rank);

		MPI_Barrier(MPI_COMM_WORLD);
		
		while(flag == 1)
		{
			//Let all processes enter their WHILE loop
			MPI_Barrier(MPI_COMM_WORLD);
			//printf("Calculation of new clusters [%d]\n",rank);
			for(dex=0;dex<data.job_size;dex++)
			{
				slave_clusters[dex]=whoIsYourDaddy(received_points[dex],centroids,data.num_clusters);
			}
			//printf("Done! [%d]\n",rank);

			MPI_Gather(slave_clusters, data.job_size, MPI_INT, NULL, data.job_size, MPI_INT, MASTER, MPI_COMM_WORLD);
			//printf("Gathering clusters [%d]\n",rank);
			//MPI_Barrier(MPI_COMM_WORLD);

			//Control if job is done
			MPI_Bcast(&job_done, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

			if(job_done==1) //No more work to be done
				flag = 0;
			
			//Receiving recently created centroids from master
			MPI_Bcast(centroids, data.num_clusters, MPI_POINT, MASTER, MPI_COMM_WORLD);
		}
	}

	//End of all	
    double end_time = MPI_Wtime();
	printf("MPI: Elapsed time: %f\n", end_time - start_time);
	MPI_Finalize();
    	return 0;
}
/* EOF */
