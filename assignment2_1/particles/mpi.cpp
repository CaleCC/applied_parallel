#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "common.h"
#include <vector>
#include<math.h>
#include <iostream>

using namespace std;
#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

void create_bins(vector<vector<particle_t*> > &bins, int & num_bins) {

	bins.resize(num_bins);
	//put particles in bins according to their locations
	// for (int j = 0; j < n; j++) {
	// 	int x = floor(particles[j].x / binsize);
	// 	int y = floor(particles[j].y / binsize);
	// 	//printf("Pushing back x: %d, y: %d, into %d\n", x, y, x + y*num_bin_row);
	// 	bins[x + y * num_bin_row].push_back(&particles[j]);
	// }
	printf("create_bins completed\n");
	//return bins;
}

//return the processors number according to the location of the particle
int num_Proc(particle_t particle, int n_proc, double p_size_x, double p_size_y){
  //int location = floor(particle.x/p_size_x) + n_proc * floor(particle.y / p_size_y);
	  int location = floor(particle.x/p_size_x);
  return location;
}

//return the bin's location according to the loaction of the particles
int bin_loc(particle_t &particle, int bin_num_row,double bin_size, double off_set_x,double off_set_y){
  int location = floor((particle.x - off_set_x)/bin_size) + bin_num_row*floor((particle.y-off_set_y)/bin_size);
  return location;
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

    set_size(n);
    double size = sqrt(density*n);// from common.cpp
    //double bin_size = size / (n_proc*floor(floor(size/cutoff)/n_proc));//must ensure that bin fits in the processor
    int num_bin_row = n_proc*floor(floor(size/cutoff)/n_proc);
		double bin_size = size/num_bin_row;
    //seperate the grid into rectanlge so only need to commucinate between left and right
    int p_bin_num_x = num_bin_row / n_proc ;
    int p_bin_num_y = num_bin_row;

    double p_size_x = p_bin_num_x * bin_size;
    double p_size_y = p_bin_num_y * bin_size;

    //calculate the halo area of each rectangle
    int halo_left = 0;//the  haloare on the left
    int halo_right = 0; // the halo area on the right
    if(rank%n_proc != 0){//not the left most one
      halo_left = 1;
    }
    if(rank%n_proc != n_proc - 1){//not the right most one
      halo_right = 1;
    }
    int proc_bin_width = halo_right+halo_left+p_bin_num_x;


    int bin_num = num_bin_row * (proc_bin_width); //the number of bins in each processor

    //set up the buffer to send and receive data in halo
    particle_t *send_l = (particle_t*) malloc( 3*num_bin_row*sizeof(particle_t));
    particle_t *send_r = (particle_t*) malloc( 3*num_bin_row*sizeof(particle_t));
    particle_t *receive_l = (particle_t*) malloc( 3*num_bin_row*sizeof(particle_t));
    particle_t *receive_r = (particle_t*) malloc( 3*num_bin_row*sizeof(particle_t));
    int send_l_count = 0;
    int send_r_count = 0;
    int rec_l_count = 0;
    int rec_r_count = 0;
    MPI_Request rec_req_l;
    MPI_Request rec_req_r;
    MPI_Status r_st_l;
    MPI_Status r_st_r;
    MPI_Request req_l;
    MPI_Request req_r;

    //allocate resources for proc
    //particle_t *processor_particles = (particle_t*) malloc ( 2 * n * sizeof(particle_t));

    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;


    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );

    //
    //  set up the data partitioning across processors
    //
    int particle_per_proc = 2 * n / n_proc;
    int *partition_offsets = (int*) malloc( n_proc * sizeof(int) );

    for( int i = 0; i < n_proc+1; i++ )
        partition_offsets[i] =  i * particle_per_proc;

    int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
    for( int i = 0; i < n_proc; i++ )
        partition_sizes[i] = particle_per_proc;




    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size( n );
    if( rank == 0 )
        init_particles( n, particles );

    //assign particles to the location of the array that belongs to the
    particle_t *assign_particles_to_p = (particle_t*)malloc(2*n*sizeof(particle_t));
    int *num_partic_proc = (int*)malloc(n_proc*sizeof(int));//an array to keep the number of particles in each processors
    for(int i= 0; i< n_proc; i++){//set to zero
      num_partic_proc[i] = 0;
    }
    if(rank == 0){
      for(int i=0; i < n; i++){
          int proc_num = num_Proc(particles[i], n_proc, p_size_x, p_size_y);
          assign_particles_to_p[proc_num * particle_per_proc + num_partic_proc[proc_num]] = particles[i];
          num_partic_proc[proc_num]++;
      }
    }
    //  allocate storage for local partition
    //
    int nlocal = 0;
    MPI_Scatter(num_partic_proc,1,MPI_INT,&nlocal,1,MPI_INT,0,MPI_COMM_WORLD);
		printf( "scatter number of particles: %d\n",nlocal);
    particle_t *local = (particle_t*) malloc( nlocal * sizeof(particle_t) );
    vector< vector<particle_t*> > bins;
    create_bins(bins, bin_num);
    //MPI_Scatter(num_partic_proc,1,MPI_INT,&nlocal,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Scatterv( assign_particles_to_p, partition_sizes, partition_offsets, PARTICLE, local, particle_per_proc, PARTICLE, 0, MPI_COMM_WORLD );


    // the offset that needs to be subtracted when calculating the location of bin where the particle stays in
    double off_set_x = rank%n_proc*p_size_x;
    double off_set_y = 0;
		printf( "proc: %d  . number of bins: %d. number of bins in a row of processor: %d, p_bin_number_y: %d, left %d, right %d\n",rank,bin_num,p_bin_num_x,p_bin_num_y,halo_left,halo_right);

    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;

        for(int i = 0; i < bin_num; i++){
          bins[i].clear();
        }
				printf( "cleared the bins\n");
        for(int i = 0; i < nlocal;i++){
          int partic_loc = bin_loc(local[i], proc_bin_width,bin_size,off_set_x-halo_left*bin_size,off_set_y);
          bins[partic_loc].push_back(local+i);
        }
				printf( "push particle into bins\n");
        send_l_count = 0;
        send_r_count = 0;

        for(int i = 0; i < p_bin_num_y; i++){
          if(halo_left){
						int bin_left_most = halo_left+i*(proc_bin_width);
						if(rank == 0) printf( "proc %d left bin location %d\n",rank,bin_left_most);
            for(int j = 0; j < bins[bin_left_most].size();j++){
                send_l[send_l_count] = *bins[bin_left_most][j];
                send_l_count++;
            }
          }//if halo left exist
					//printf( "send_l_cout %d\n",send_l_cout);
          if(halo_right){//if halo right exists
						int bin_right_most = p_bin_num_x - 1+halo_left+i*(proc_bin_width);
						if(rank == 0) printf( "proc %d right bin location %d\n",rank,bin_right_most);
            for(int j = 0; j <bins[bin_right_most].size(); j++){
              send_r[send_r_count] = *bins[bin_right_most][j];
              send_r_count++;
            }
          }
        }
				printf( "proc %d start receive\n",rank);
        //receive the number of particle from neighbour

        rec_r_count = 0;
				rec_l_count = 0;
        if(halo_left){
          MPI_Irecv(&rec_l_count,1,MPI_INT,rank-1,rank,MPI_COMM_WORLD,&rec_req_l);
        }

        if(halo_right){
          MPI_Irecv(&rec_r_count,1,MPI_INT,rank+1,rank,MPI_COMM_WORLD,&rec_req_r);
        }

        //send the number of particles to neighbour
         if(halo_left){
           MPI_Isend(&send_l_count,1,MPI_INT,rank-1,rank-1,MPI_COMM_WORLD,&req_l);
         }
         if(halo_right){
           MPI_Isend(&send_r_count,1,MPI_INT,rank+1,rank+1,MPI_COMM_WORLD,&req_r);
         }


         //wait for receive of those numbers
         if(halo_left){
           MPI_Wait(&rec_req_l,&r_st_l);
					  printf( "proc %d   receive count left %d\n",rank,rec_l_count);
         }
         if(halo_right){
           MPI_Wait(&rec_req_r, &r_st_r);
					 printf( "proc %d   receive count right %d\n",rank,rec_l_count);
         }
				 printf( "proc %d spoped  receive count\n",rank);

        //receive for halo area
        if(halo_left && rec_l_count){
          MPI_Irecv(receive_l,rec_l_count,PARTICLE,rank-1,rank,MPI_COMM_WORLD,&rec_req_l);
        }
        if(halo_right && rec_r_count){
          MPI_Irecv(receive_r,rec_r_count,PARTICLE,rank+1,rank,MPI_COMM_WORLD,&rec_req_r);
        }

        //send halo area to other processors
        if(halo_left && send_l_count){
          MPI_Isend(send_l,send_l_count,PARTICLE,rank-1,rank-1,MPI_COMM_WORLD,&req_l);
        }
        if(halo_right && send_r_count){
          MPI_Isend(send_r,send_r_count,PARTICLE,rank+1,rank+1,MPI_COMM_WORLD,&req_r);
        }

        //wait to receive the area
        if(halo_left&&rec_l_count){
          MPI_Wait(&rec_req_l,&r_st_l);
        }
        if(halo_right&&rec_r_count){
          MPI_Wait(&rec_req_r,&r_st_r);
        }
				 printf( "proc %d spoped  receive halo area\n",rank);

        //push the received particles in halo area into the bins
        for(int i = 0; i < rec_l_count; i++){
          bins[bin_loc(receive_l[i],proc_bin_width,bin_size,off_set_x - halo_left*bin_size,off_set_y)].push_back(receive_l+i);
        }
        for(int i = 0;i<rec_r_count;i++){
          bins[bin_loc(receive_r[i],proc_bin_width,bin_size,off_set_x - halo_left*bin_size,off_set_y)].push_back(receive_r+i);
        }

        //compute all forces
        for(int p = 0; p < nlocal; p++){
          local[p].ax = 0;
          local[p].ay = 0;
          int location = bin_loc(local[p],proc_bin_width,bin_size, off_set_x-halo_left*bin_size,off_set_y );
          vector<int> x_range;
          vector<int> y_range;
          x_range.push_back(0);
          y_range.push_back(0);
          if (location >= (proc_bin_width)) {
            y_range.push_back(-1);
          }
          if (location < (proc_bin_width)*(p_bin_num_y-1)) {
            y_range.push_back(1);
          }
          if ((location % (proc_bin_width) != 0)&&(halo_left != 0)) {
            x_range.push_back(-1);
          }
          if ((location%(proc_bin_width) !=  (proc_bin_width)-1)&&(halo_right != 0)) {
            x_range.push_back(1);
          }


        for (int a = 0; a < x_range.size(); a++) {
          for (int b = 0; b < y_range.size(); b++) {
            int nbin= location + x_range[a] + (proc_bin_width)*y_range[b];
            //printf("i: %d, bin_num: %d, bins[i].size(): %d, bins[bin_num].size(): %d\n",i, bin_num, bins[i].size(), bins[bin_num].size());

            for (int c = 0; c < bins[nbin].size(); c++) {

                apply_force(local[p], *bins[nbin][c], &dmin, &davg, &navg);
                //printf("apply_force end\n");
            }
          }
        }
      }
			 printf( "proc %d finished force compute\n",rank);
			if( find_option( argc, argv, "-no" ) == -1 ){
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
      for( int i = 0; i < nlocal; i++ )
          move( local[i] );
			printf( "proc %d finished move\n",rank);
      //check particles that has left current processor
      send_r_count = 0;
      send_l_count = 0;
      rec_r_count = 0;
      rec_l_count = 0;
      int change_flag = 0;
      for(int i = 0; i < nlocal;i++){
          change_flag = 0;
          if(local[i].x < off_set_x){
            send_l[send_l_count] = local[i];
            send_l_count++;
            change_flag=1;
          }

					if(local[i].x >= off_set_x+p_size_x){
            send_r[send_r_count] = local[i];
            send_r_count++;
            change_flag = 1;
          }

          if(change_flag == 1){//delete it from local
            local[i] = local[nlocal-1];
            nlocal--;
            i--;
          }
      }
			printf("proc %d finished check the moved particles \n",rank);
			if(halo_left){
				MPI_Irecv(&rec_l_count,1,MPI_INT,rank-1,rank,MPI_COMM_WORLD,&rec_req_l);
			}
			if(rank == 1) printf("proc %d post for receive left \n",rank);
			if(halo_right){
				MPI_Irecv(&rec_r_count,1,MPI_INT,rank+1,rank,MPI_COMM_WORLD,&rec_req_r);
			}
			if(rank == 1) printf("proc %d post for receive right \n",rank);

			//send the number of particles to neighbour
			 if(halo_left){
				 MPI_Isend(&send_l_count,1,MPI_INT,rank-1,rank-1,MPI_COMM_WORLD,&req_l);
			 }
			 if(rank == 1) printf("proc %d post to send left \n",rank);
			 if(halo_right){
				 MPI_Isend(&send_r_count,1,MPI_INT,rank+1,rank+1,MPI_COMM_WORLD,&req_r);
			 }
			 if(rank == 1) printf("proc %d post to send right \n",rank);


			 //wait for receive of those numbers
			 if(halo_left){
				 if(rank == 1) printf("proc %d wait left wrong? \n",rank);
				 MPI_Wait(&rec_req_l,&r_st_l);
					printf( "after move proc %d   receive count from left %d\n",rank,rec_l_count);
			 }
			 if(halo_right){
				 MPI_Wait(&rec_req_r, &r_st_r);
				 printf( "after move proc %d   receive count from right %d\n",rank,rec_r_count);
			 }
			printf( "proc %d spoped  receive count\n",rank);

			printf( "proc %d actually received\n",rank);
      //receive for halo area
      if(halo_left && rec_l_count){
        MPI_Irecv(receive_l,rec_l_count,PARTICLE,rank-1,rank,MPI_COMM_WORLD,&rec_req_l);
      }
      if(halo_right && rec_r_count){
        MPI_Irecv(receive_r,rec_r_count,PARTICLE,rank+1,rank,MPI_COMM_WORLD,&rec_req_r);
      }
			printf( "proc %d post for  receive array\n",rank);
      //send halo area to other processors
      if(halo_left && send_l_count){
        MPI_Isend(send_l,send_l_count,PARTICLE,rank-1,rank-1,MPI_COMM_WORLD,&req_l);
      }
      if(halo_right && send_r_count){
        MPI_Isend(send_r,send_r_count,PARTICLE,rank+1,rank+1,MPI_COMM_WORLD,&req_r);
      }
			printf( "proc %d post for  send array\n",rank);
      //wait to receive the area
      if(halo_left&&rec_l_count){
        MPI_Wait(&rec_req_l,&r_st_l);
      }
      if(halo_right&&rec_r_count){
        MPI_Wait(&rec_req_r,&r_st_r);
      }
			printf( "proc %d finished send left particles\n",rank);
      //add those received particles to local array
      for(int i = 0; i < rec_l_count;i++){
        local[nlocal] = receive_l[i];
        nlocal++;
      }
      for(int i = 0; i < rec_r_count;i++){
        local[nlocal] = receive_r[i];
        nlocal++;
      }
			printf( "proc %d finished add new particles to local\n",rank);
			if( find_option( argc, argv, "-no" ) == -1 ){
					if( (step%SAVEFREQ) == 0 )
				{
				MPI_Gather(&nlocal,1,MPI_INT,num_partic_proc,1,MPI_INT,0,MPI_COMM_WORLD);
				if (rank == 0)
				{
					int tmp_nsum=0;
					for(int i=0;i<n_proc;i++)
					{
						partition_offsets[i]=tmp_nsum;
						tmp_nsum+=num_partic_proc[i];
					}
				}

				MPI_Gatherv( local, nlocal, PARTICLE, particles,num_partic_proc, partition_offsets, PARTICLE,0, MPI_COMM_WORLD );
				if (rank == 0)
					save( fsave, n, particles );
				}
			 }


      MPI_Barrier(MPI_COMM_WORLD);
			printf( "proc %d one step complete\n",rank);
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
    if( fsave )
        fclose( fsave );

    MPI_Finalize( );

    return 0;
}
