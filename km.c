#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define N 1000000// Number of data points
#define K 5 // Number of clusters
#define MAX_ITER 5 // Maximum number of iterations

// Structure to represent a 2D point
typedef struct Point
{
    float x, y;
} Point;

// Function to calculate the Euclidean distance between two points
float distance(Point p1, Point p2)
{
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

// Function to assign each data point to the nearest cluster
void assignToClusters(Point *data, Point *centroids, int *assignments, int n,int p)
{
#pragma omp parallel for if(p==1) num_threads(5)
    for (int i = 0; i < n; i++)
    {
        float minDist = distance(data[i], centroids[0]);
        int cluster = 0;
        for (int j = 1; j < K; j++)
        {
            float dist = distance(data[i], centroids[j]);
            if (dist < minDist)
            {
                minDist = dist;
                cluster = j;
            }
        }
        assignments[i] = cluster;
    }
}

// Function to recalculate cluster centroids
void updateCentroids(Point *data, int *assignments, Point *centroids, int n, int p)
{
    int counts[K] = {0};
    for (int i = 0; i < K; i++)
    {
        centroids[i].x = 0;
        centroids[i].y = 0;
    }
    long iter=n*n;
    #pragma omp parallel for if(p==1) num_threads(5)
    for (int i = 0; i < iter; i++)
    {
        int cluster = assignments[i];
        centroids[cluster].x += data[i].x;
        centroids[cluster].y += data[i].y;
        counts[cluster]++;
    }
    for (int i = 0; i < K; i++)
    {
        if (counts[i] > 0)
        {
            centroids[i].x /= counts[i];
            centroids[i].y /= counts[i];
        }
    }
}
static unsigned long get_nsecs(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000UL + ts.tv_nsec;
}
int main()
{
    // Generate random data points
    srand(time(NULL));
    Point* data;
    data=(Point*)malloc(sizeof(Point)*N);
    for (int i = 0; i < N; i++)
    {
        data[i].x = (float)(rand() % 100) + 1;
       
        data[i].y = (float)(rand() % 100) + 1;
       
    }

    // Initialize cluster centroids randomly
    Point centroids[K];
    for (int i = 0; i < K; i++)
    {
        centroids[i] = data[i];
    }
   
 long  start, end;
    /* Store start time here */
    start = get_nsecs();
    // Perform k-means clustering
    int assignments[N];
    for (int iter = 0; iter < MAX_ITER; iter++)
    {
        // Assign data points to clusters
        assignToClusters(data, centroids, assignments, N,0);
        // Update cluster centroids
        updateCentroids(data, assignments, centroids, N, 0);
    }
    // Print cluster as0signments
   
    end = get_nsecs();
    /* Get the time taken by program to execute in seconds */
    double duration = ((double)end - start)/1000000000;
    printf("Time taken to execute in seconds without openmp: %f\n", duration);


   
    /* Store start time here */
    start = get_nsecs();
    // Perform k-means clustering
    for (int iter = 0; iter < MAX_ITER; iter++)
    {
        // Assign data points to clusters
        assignToClusters(data, centroids, assignments, N,1);
        // Update cluster centroids
        updateCentroids(data, assignments, centroids, N, 1);
    }
    // Print cluster assignments
   
    end = get_nsecs();
    /* Get the time taken by program to execute in seconds */
    duration = ((double)end - start)/1000000000;
    printf("Time taken to execute in seconds with openmp: %f", duration);

    return 0;
}