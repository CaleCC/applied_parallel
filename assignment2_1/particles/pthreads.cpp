#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include <vector>
#include <iostream>
#include <pthread.h>
using namespace std;
//
//  tuned constants copied from common.cpp
//
#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

//
//  global variables
//
int n, n_threads,no_output=0;
int num_bins, num_bin_row;
particle_t *particles;
vector<vector<particle_t*> > bins;
FILE *fsave,*fsum;
pthread_barrier_t barrier;
pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;
double gabsmin=1.0,gabsavg=0.0;

//
//  check that pthreads routine call was successful
//
#define P( condition ) {if( (condition) != 0 ) { printf( "\n FAILURE in %s, line %d\n", __FILE__, __LINE__ );exit( 1 );}}

//
//create bins with length of cutoff
//
void create_bins() {
	//size = sqrt(density * n)
	double binsize = 2 * cutoff;
	//the number of bins in a row = size / binlength. Round up.
	double size = sqrt(density * n); //This is from common.cpp
	num_bin_row = ceil(size / binsize);
	//printf("num_bin_row is %d\n", num_bin_row);
	//resize the vector to the exact size of bins.
	//printf("bins.resize\n");
	bins.resize(num_bin_row * num_bin_row);
	//put particles in bins according to their locations
	for (int j = 0; j < n; j++) {
		int x = floor(particles[j].x / binsize);
		int y = floor(particles[j].y / binsize);
		//printf("Pushing back x: %d, y: %d, into %d\n", x, y, x + y*num_bin_row);
		bins[x + y * num_bin_row].push_back(&particles[j]);
	}
	//printf("create_bins completed\n");
}

//
//  This is where the action happens
//  We want each thread to handle a block of local particles, as well as its halo region.
//
void *thread_routine( void *pthread_id )
{
    int navg,nabsavg=0;
    double dmin,absmin=1.0,davg,absavg=0.0;
    int thread_id = *(int*)pthread_id;
	vector<particle_t*> temp_move;

    int bins_per_thread = (num_bins + n_threads - 1) / n_threads;
    int first = min(  thread_id    * bins_per_thread, num_bins);
    int last  = min( (thread_id+1) * bins_per_thread, num_bins);
    
    //
    //  simulate a number of time steps
    //
    for( int step = 0; step < NSTEPS; step++ )
    {
        dmin = 1.0;
        navg = 0;
        davg = 0.0;
        //
        //  compute forces
        //
        vector<int> x_range;
        vector<int> y_range;
		for (int i = first; i < last; i++)
		{
			vector<particle_t*> binQ = bins[i];
			int particles_per_bin = binQ.size();
			//printf("binQ size: %d\t", particles_per_bin);
			for (int j = 0; j < particles_per_bin; j++) {
				binQ[j]->ax = binQ[j]->ay = 0;
			}

			//Search within current bin and a 'halo region'
			//Halo region may be defined as the region around the bin of size 'cutoff'
			//For ease in computation, the neighboring bins may be used as the halo.

			int location = i;
			x_range.push_back(0);
			y_range.push_back(0);
			if (location >= num_bin_row) {
				y_range.push_back(-1);
			}
			if (location < num_bin_row*(num_bin_row - 1)) {
				y_range.push_back(1);
			}
			if (location % num_bin_row != 0) {
				x_range.push_back(-1);
			}
			if (location % num_bin_row != num_bin_row - 1) {
				x_range.push_back(1);
			}
			//printf("x and y ranges initialized.\n");
			//This should manage the ranges such that the halo region is searched.
			for (int a = 0; a < x_range.size(); a++) {
				for (int b = 0; b < y_range.size(); b++) {
					int bin_num = i + x_range[a] + num_bin_row*y_range[b];
					//printf("i: %d, bin_num: %d, bins[i].size(): %d, bins[bin_num].size(): %d\n",i, bin_num, bins[i].size(), bins[bin_num].size());

					for (int c = 0; c < binQ.size(); c++) {
						for (int d = 0; d < bins[bin_num].size(); d++) {
							apply_force(*binQ[c], *bins[bin_num][d], &dmin, &davg, &navg);
						}
					}
				}
			}
            x_range.clear();
            y_range.clear();
		}
        
        pthread_barrier_wait( &barrier );
        
        if( no_output == 0 )
        {
          //
          // Computing statistical data
          // 
          if (navg) {
            absavg +=  davg/navg;
            nabsavg++;
          }
          if (dmin < absmin) absmin = dmin;
	}

        //
        //  move particles
        //
		double binsize = cutoff * 2;

		for (int b = first; b < last; b++)
		{
			int size = bins[b].size();
			for (int p = 0; p < size;) {
				//printf("Moving particle in bin %d, p = %d\n", b, p);
				move(*bins[b][p]);

				int x = floor(bins[b][p]->x / binsize);
				int y = floor(bins[b][p]->y / binsize);
				if (y * num_bin_row + x != b)
				{
					//printf("p is %d: Moving particles from bin %d to %d... \n", p, b, x+y*num_bin_row);
					temp_move.push_back(bins[b][p]);
                    size--;
                    bins[b][p] = bins[b][size];
                    //We move the last particle address into the current one, then reduce size by one.
				}
				else {
					p++;
				}
			}
            bins[b].resize(size);
		}
        pthread_barrier_wait( &barrier );
        
        int tempsize = temp_move.size();
		for (int i = 0; i < tempsize; i++) {
			int x = floor(temp_move[i]->x / binsize);
			int y = floor(temp_move[i]->y / binsize);
			//printf("Pushing particle into bin[%d]... ", x+y*num_bin_row);
			int index = x + y * num_bin_row;
			pthread_mutex_lock(&mutex);
			bins[index].push_back(temp_move[i]);
			pthread_mutex_unlock(&mutex);
			//printf("Pushback complete\n");
		}
		temp_move.clear();
		//printf("temp_move cleared\n");
        
        pthread_barrier_wait( &barrier );
        
        //
        //  save if necessary
        //
        if (no_output == 0) 
          if( thread_id == 0 && fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );
        
    }
     
    if (no_output == 0 )
    {
      absavg /= nabsavg; 	
      //printf("Thread %d has absmin = %lf and absavg = %lf\n",thread_id,absmin,absavg);
      pthread_mutex_lock(&mutex);
      gabsavg += absavg;
      if (absmin < gabsmin) gabsmin = absmin;
      pthread_mutex_unlock(&mutex);    
    }

    return NULL;
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    //
    //  process command line
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-p <int> to set the number of threads\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");        
        return 0;
    }
    
    n = read_int( argc, argv, "-n", 1000 );
    n_threads = read_int( argc, argv, "-p", 2 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    fsave = savename ? fopen( savename, "w" ) : NULL;
    fsum = sumname ? fopen ( sumname, "a" ) : NULL;

    if( find_option( argc, argv, "-no" ) != -1 )
      no_output = 1;

    //
    //  allocate resources
    //
    particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );

	//create the bins to contain the particles
	num_bin_row = 0;
	create_bins();
	num_bins = num_bin_row * num_bin_row;

    pthread_attr_t attr;
    P( pthread_attr_init( &attr ) );
    P( pthread_barrier_init( &barrier, NULL, n_threads ) );

    int *thread_ids = (int *) malloc( n_threads * sizeof( int ) );
    for( int i = 0; i < n_threads; i++ ) 
        thread_ids[i] = i;

    pthread_t *threads = (pthread_t *) malloc( n_threads * sizeof( pthread_t ) );
    
    //
    //  do the parallel work
    //
    double simulation_time = read_timer( );
    for( int i = 1; i < n_threads; i++ ) 
        P( pthread_create( &threads[i], &attr, thread_routine, &thread_ids[i] ) );
    
    thread_routine( &thread_ids[0] );
    
    for( int i = 1; i < n_threads; i++ ) 
        P( pthread_join( threads[i], NULL ) );
    simulation_time = read_timer( ) - simulation_time;
   
    printf( "n = %d, simulation time = %g seconds", n, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
      gabsavg /= (n_threads*1.0);
      // 
      //  -the minimum distance absmin between 2 particles during the run of the simulation
      //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
      //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
      //
      //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
      //
      printf( ", absmin = %lf, absavg = %lf", gabsmin, gabsavg);
      if (gabsmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting ");
      if (gabsavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting ");
    }
    printf("\n");

    //
    // Printing summary data
    //
    if( fsum)
        fprintf(fsum,"%d %d %g\n",n,n_threads,simulation_time); 
    
    //
    //  release resources
    //
    P( pthread_barrier_destroy( &barrier ) );
    P( pthread_attr_destroy( &attr ) );
    free( thread_ids );
    free( threads );
    free( particles );
    if( fsave )
        fclose( fsave );
    if( fsum )
        fclose ( fsum );
    
    return 0;
}
