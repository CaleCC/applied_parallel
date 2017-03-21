#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include <vector>
#include <iostream>
#include <string.h>

#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005
#define binsize (cutoff)

using namespace std;
//
//create bins with length of cutoff
//
void create_bins(vector<particle_t> bins[], particle_t* particles, int n, int num_bin_row, int first) {

    //put particles in bins according to their locations
    for (int j = 0; j < n; j++) {
        int x = floor(particles[j].x / binsize);
        int y = floor(particles[j].y / binsize);
        bins[x + (y-first) * num_bin_row].push_back(particles[j]);
    }
    printf("from %d, Bins created\n", first);
}
//Partition particles into n_proc bins based on their row location. 
void partition_bins(vector<particle_t> bins[], particle_t* particles, int* particles_per_process, int n, int num_bin_row, int n_proc) {
    
    memset ( particles_per_process, 0, sizeof(int)*n_proc);
    int rows_per_proc = (num_bin_row + n_proc -1) / n_proc;
    //put particles in bins according to their locations
    for (int j = 0; j < n; j++) {
        //int x = floor(particles[j].x / binsize);
        int y = floor(particles[j].y / binsize);
        int procdex = floor(y / rows_per_proc);
        bins[procdex].push_back(particles[j]);
        particles_per_process[procdex]++;
        //If this particle is in a halo bin, we must also import it twice.
        int boundcheck = y % rows_per_proc;
        if(boundcheck == 0 && y != 0 ){
            //We now know that y is on a boundary, and must be included in procdex-1.
            bins[procdex-1].push_back(particles[j]);
            particles_per_process[procdex-1]++;
        }
        else if(boundcheck == rows_per_proc-1 && y != num_bin_row-1){
            //We now know that y is on a boundary, and must be included in procdex+1.
            bins[procdex+1].push_back(particles[j]);
            particles_per_process[procdex+1]++;
        }
    }
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg; 
    int* partition_sizes;
    int* partition_offsets;
    particle_t *sendBuf;
 
    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    partition_offsets = (int*) malloc(sizeof(int) * n_proc);
    partition_sizes = (int*) malloc(sizeof(int)*n_proc);

    vector<particle_t> bins[n_proc];
    
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );


    int num_bin_row;
    //size = sqrt(density * n)
    //the number of bins in a row = size / binlength. Round up.
    double size = sqrt(density * n); //This is from common.cpp
    num_bin_row = ceil(size / binsize);
    int num_bins = num_bin_row * num_bin_row;
    int rows_per_proc = (num_bin_row + n_proc -1) / n_proc;
    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size( n );
    if( rank == 0 ){
        init_particles( n, particles );
        //partition_bins will sort the particles into n_proc bins, based on their bin row index
        //While we do this, we want to also count the number of particles in each level;
        //We use this info to malloc a particle_t array to scatterv across all processes.
        partition_bins(bins, particles, partition_sizes, n, num_bin_row, n_proc);
        int totalSize = 0;
        partition_offsets[0] = 0;
        for (int i = 0; i < n_proc-1; i++){
            totalSize += partition_sizes[i];
            partition_offsets[i+1] = partition_offsets[i] + partition_sizes[i]; 
        }
        totalSize += partition_sizes[n_proc-1];
        //Initialize the large array of particles we will be sending
        sendBuf = (particle_t*) malloc( totalSize * sizeof(particle_t) );
        //This loop is meant to fill sendBuf with contiguous particles.
        for( int i = 0; i < n_proc; i++){
            memcpy(&sendBuf[partition_offsets[i]], bins[i].data(), partition_sizes[i] * sizeof(particle_t));
        }
        printf("Particles and send buffer initialized\n");
    }
        //MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast( partition_sizes, n_proc, MPI_INT, 0, MPI_COMM_WORLD );
        printf("rank %d: partition sizes initialized\n",rank);
        MPI_Bcast( partition_offsets, n_proc, MPI_INT, 0, MPI_COMM_WORLD );
        //At this point, we expect every worker to have a complete set of knowledge regarding the sizes and offsets.
        printf("rank %d: partition offsets initialized\n",rank);
    //
    //  allocate storage for local partition
    //
    int nlocal = partition_sizes[rank];
    particle_t *local = (particle_t*) malloc( nlocal * sizeof(particle_t) );
    particle_t *fromAbove = (particle_t*) malloc( n/(2*n_proc) * sizeof(particle_t) );
    particle_t *fromBelow = (particle_t*) malloc( n/(2*n_proc) * sizeof(particle_t) );
    particle_t* movingup = (particle_t*) malloc(sizeof(particle_t) * n/(2*n_proc));
    particle_t* movingdown = (particle_t*) malloc(sizeof(particle_t) * n/(2*n_proc));
    //It is reasonable to assume that no more than half the total particles will travel between a local partition.

    
    MPI_Scatterv( sendBuf, partition_sizes, partition_offsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD );
    printf("rank %d: particles scattered\n",rank);
    //
    //  Create bins for local rows.
    //
    int first = min(  rank    * rows_per_proc, num_bin_row);
    int last  = min( (rank+1) * rows_per_proc, num_bin_row);
    first--;
    last++;
    if(rank == 0){
        first++; //On the top and bottom, we will have one less row (smaller halo)
    }
    else if(rank == n_proc){
        last--;
    }
    int bins_proc = last - first; //General case

    int local_bin_size = bins_proc * num_bin_row;

    vector<particle_t> localBins[local_bin_size];
    //vector<particle_t> throwaway;
    //localBins.push_back(throwaway);
    //localBins.resize(local_bin_size); //Bins per row * number of rows.

    create_bins(localBins, local, nlocal, num_bin_row, first);

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        printf("Timestep: %d\n", step);
        navg = 0;
        dmin = 1.0;
        davg = 0.0;
        // 
        //  collect all global data locally (not good idea to do)
        //
        //MPI_Allgatherv( local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, MPI_COMM_WORLD );
        //Rather than updating our local copy by collecting global data, we will send and recv 
        //any and all updates of particles entering or exiting our area, halo region included.
        //We will do this at the end of each time step, as we initialize our data proper.

        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if( find_option( argc, argv, "-no" ) == -1 )
          if( fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );
        
        //
        //  compute all forces
        //
        //We must account for halo regions- we do not want to calculate forces for the halo rows.
        //These will generally be 0 to num_bin_row and local_bin_size+num_bin_row to local_bin_size+2*num_bin_row
        //, except for the 0 rank worker process and n_proc rank process. We only must adjust for the starting value.

        int biter = num_bin_row;
        if(rank == 0){
            biter = 0;
        }
        int endBins = local_bin_size-biter; //Localbinsize includes the halo region.
        //  Do one set of computations for each bin.
        for (; biter < endBins; biter++)
        {
            vector<particle_t> binQ = localBins[biter];
            int particles_per_bin = binQ.size();

            for (int j = 0; j < particles_per_bin; j++) {
                particles[biter].ax = particles[biter].ay = 0;
            }
            int location = biter;
            vector<int> x_range;
            vector<int> y_range;
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
                    int bin_num = biter + x_range[a] + num_bin_row*y_range[b];
                    //printf("i: %d, bin_num: %d, bins[i].size(): %d, bins[bin_num].size(): %d\n",i, bin_num, bins[i].size(), bins[bin_num].size());

                    for (int c = 0; c < binQ.size(); c++) {
                        for (int d = 0; d < localBins[bin_num].size(); d++) {
                            apply_force(binQ[c], localBins[bin_num][d], &dmin, &davg, &navg);
                        }
                    }
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if( find_option( argc, argv, "-no" ) == -1 )
        {
          
          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

 
          if (rank == 0){
            //
            // Computing statistical data
            //
            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) absmin = rdmin;
          }
        }

        //
        //  move particles
        //
        //  We must first move particles as per usual, except our temp array of particles to exit our system
        //  must now be sent to the relevant folks.
        vector<particle_t> moveUp;
        vector<particle_t> moveDown;
        biter = num_bin_row;
        if(rank == 0){
            biter = 0;
        }
        endBins = local_bin_size-biter;

        for (;biter < endBins; biter++)
        {//Insert logic here
            int size = localBins[biter].size();
            for (int p = 0; p < size;) {
                //printf("Moving particle in bin %d, p = %d\n", biter, p);
                move(localBins[biter][p]);

                int x = floor(localBins[biter][p].x / binsize);
                int y = floor(localBins[biter][p].y / binsize);
                if (y * num_bin_row + x != biter)
                {
                    if(biter < num_bin_row)
                        moveUp.push_back(localBins[biter][p]);
                    else 
                        moveDown.push_back(localBins[biter][p]);

                    localBins[biter].erase(localBins[biter].begin() + p);
                    size--;
                }
                else{
                    p++;
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        int fa, fb;
        if(rank > 0){
            MPI_Send(moveUp.data(), moveUp.size(), PARTICLE, rank-1, rank, MPI_COMM_WORLD);
        }
        if(rank < n_proc-1){
            MPI_Send(moveDown.data(), moveDown.size(), PARTICLE, rank+1, rank, MPI_COMM_WORLD);
        }

        if(rank > 0){
            MPI_Recv(fromAbove, n/n_proc, PARTICLE, rank+1, rank+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Get_count(MPI_STATUS_IGNORE, PARTICLE, &fa);
        }
        if(rank < n_proc-1){
            MPI_Recv(fromBelow, n/n_proc, PARTICLE, rank-1, rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Get_count(MPI_STATUS_IGNORE, PARTICLE, &fb);
        }
        //Now we wish to recieve the data of all processes which have moved particles into our system.
        //We also wish to send the data of particles which have exited our system to our neighboring processes.
        MPI_Barrier(MPI_COMM_WORLD);
        for(int i = 0; i < fa; i++){
            int x = floor(fromAbove[i].x / binsize);
            int y = floor(fromAbove[i].y / binsize);
            localBins[x + (y-first) * num_bin_row].push_back(fromAbove[i]);
        }
        for(int i = 0; i < fb; i++){
            int x = floor(fromBelow[i].x / binsize);
            int y = floor(fromBelow[i].y / binsize);
            localBins[x + (y-first) * num_bin_row].push_back(fromBelow[i]);
        }


//                                                                      //
        ///                                                     ///
        ///     We must also update the halo region for each.   ///
        ///                                                     ///
//                                                                      //

//We accomplish this by clearing our topmost and bottommost bins, 
//With the exception of the rank 0 and rank n_proc processes, handle those with care,
//And then sending the second topmost and second bottommost bin rows to our neighbors
//And then receiving the updated halo regions.

        biter = num_bin_row;
        if(rank == 0){
            biter = 0;
        }
        endBins = local_bin_size-biter;

        for(int i = 0; i < biter; i++){
            localBins[i].clear();
        }
        for(int i = endBins; i < last*num_bin_row; i++){
            localBins[i].clear();
        }

        //Send the first REAL row to our neighbors.
        moveUp.clear();
        moveDown.clear();
        int upsize = 0;
        int downsize = 0;
        if(rank > 0){
            for(int eob = biter; eob < biter + num_bin_row; eob++){
                //for(int i = 0; i < localBins[eob].size(); i++)
                //    moveUp.push_back(localBins[eob][i]);
                memcpy(&movingup[upsize], localBins[eob].data(), localBins[eob].size());
                upsize += localBins[eob].size();
            }   
        }
        if(rank < n_proc-1){
            for(int boe = local_bin_size; boe < endBins; boe++){
                //for(int i = 0; i < localBins[boe].size(); i++)
                //    moveDown.push_back(localBins[boe][i]);
                memcpy(&movingdown[downsize], localBins[boe].data(), localBins[boe].size());
                downsize += localBins[boe].size();
            }   
        }
        MPI_Barrier(MPI_COMM_WORLD);
        //Same as above, we send data up and down
        if(rank > 0){
            MPI_Send(movingup, upsize, PARTICLE, rank-1, rank, MPI_COMM_WORLD);
        }
        if(rank < n_proc-1){
            MPI_Send(movingdown, downsize, PARTICLE, rank+1, rank, MPI_COMM_WORLD);
        }

        if(rank > 0){
            MPI_Recv(fromAbove, n/n_proc, PARTICLE, rank+1, rank+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Get_count(MPI_STATUS_IGNORE, PARTICLE, &fa);
        }
        if(rank < n_proc-1){
            MPI_Recv(fromBelow, n/n_proc, PARTICLE, rank-1, rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Get_count(MPI_STATUS_IGNORE, PARTICLE, &fb);
        }
        //We must now handle the data received differently though; we have to rebin it into our halo regions.
        if(rank > 0)
            for (int j = 0; j < fa; j++) {
                int x = floor(fromAbove[j].x / binsize);
                int y = floor(fromAbove[j].y / binsize);
                localBins[x + (y-first) * num_bin_row].push_back(fromAbove[j]);
            }
        if(rank < n_proc-1)
            for (int j = 0; j < fb; j++) {
                int x = floor(fromBelow[j].x / binsize);
                int y = floor(fromBelow[j].y / binsize);
                localBins[x + (y-first) * num_bin_row].push_back(fromBelow[j]);
            }
        MPI_Barrier(MPI_COMM_WORLD);
//End of time step.
    }
    simulation_time = read_timer( ) - simulation_time;
  
    if (rank == 0) {  
      printf( "n = %d, simulation time = %g seconds", n, simulation_time);

      if( find_option( argc, argv, "-no" ) == -1 )
      {
        if (nabsavg) absavg /= nabsavg;
      // 
      //  -the minimum distance absmin between 2 particles during the run of the simulation
      //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
      //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
      //
      //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
      //
      printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
      if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
      if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
      }
      printf("\n");     
        
      //  
      // Printing summary data
      //  
      if( fsum)
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
  
    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    free( partition_offsets );
    free( partition_sizes );
    free( local );
    free( particles );
    free( sendBuf );
    free( fromAbove );
    free( fromBelow );
    free( movingup );
    free( movingdown );


    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}
