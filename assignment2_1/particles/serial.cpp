#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include <vector>
#include <iostream>
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
//create bins with length of cutoff
//
void create_bins(vector<vector<particle_t*> > &bins, particle_t* particles, int n, int & num_bin_row) {
	//size = sqrt(density * n)
	double binsize = 2 * cutoff;
	//the number of bins in a row = size / binlength. Round up.
	double size = sqrt(density * n); //This is from common.cpp
	num_bin_row = ceil(size / binsize);
	printf("num_bin_row is %d\n", num_bin_row);
	//resize the vector to the exact size of bins.
	printf("bins.resize\n");
	bins.resize(num_bin_row * num_bin_row);
	//put particles in bins according to their locations
	for (int j = 0; j < n; j++) {
		int x = floor(particles[j].x / binsize);
		int y = floor(particles[j].y / binsize);
		printf("Pushing back x: %d, y: %d, into %d\n", x, y, x + y*num_bin_row);
		bins[x + y * num_bin_row].push_back(&particles[j]);
	}
	printf("create_bins completed\n");
}
//
//  benchmarking program
//
int main(int argc, char **argv)
{
	int navg, nabsavg = 0;
	double davg, dmin, absmin = 1.0, absavg = 0.0;

	if (find_option(argc, argv, "-h") >= 0)
	{
		printf("Options:\n");
		printf("-h to see this help\n");
		printf("-n <int> to set the number of particles\n");
		printf("-o <filename> to specify the output file name\n");
		printf("-s <filename> to specify a summary file name\n");
		printf("-no turns off all correctness checks and particle output\n");
		return 0;
	}

	//int n = read_int(argc, argv, "-n", 1000);

	char *savename = read_string(argc, argv, "-o", NULL);
	char *sumname = read_string(argc, argv, "-s", NULL);


	FILE *fsave = savename ? fopen(savename, "w") : NULL;
	FILE *fsum = sumname ? fopen(sumname, "a") : NULL;

	int n = 10;

	particle_t *particles;

	vector<particle_t*> temp_move;

	if ((particles = (particle_t*)malloc(n * sizeof(particle_t))) == NULL) {
		printf("particles malloc NULL\n");
		return 0;
	}
	set_size(n);
	init_particles(n, particles);
	//create the bins to contain the particles
	vector<vector<particle_t*> > bins;
	int num_bin_row = 0;
	create_bins(bins, particles, n, num_bin_row);
	int num_bins = num_bin_row * num_bin_row;
	printf("Bins Created\n");
	//
	//  simulate a number of time steps
	//

	double simulation_time = read_timer();

	for (int step = 0; step < NSTEPS; step++)
	{
		printf("Step: %d\t", step);
		navg = 0;
		davg = 0.0;
		dmin = 1.0;
		//
		//  compute forces
		//
		//  Do one set of computations for each bin.
		for (int i = 0; i < num_bins; i++)
		{
			vector<particle_t*> binQ = bins[i];
			int particles_per_bin = binQ.size();
			printf("binQ size: %d\t", particles_per_bin);
			for (int j = 0; j < particles_per_bin; j++) {
				binQ[j]->ax = binQ[j]->ay = 0;
			}
			//printf("Accelerations zeroed\n");

			//Search within current bin and a 'halo region'
			//Halo region may be defined as the region around the bin of size 'cutoff'
			//For ease in computation, the neighboring bins may be used as the halo.

			//int cx = floor(particles[i].x/cutoff);
			//int cy = floor(particles[i].y/cutoff);
			//int location = cx + num_bin_row * cy;
			int location = i;
			vector<int> x_range;
			vector<int> y_range;
			// int startx = -1; // the begin row of bin
			// int endx = 1;
			// int starty = -1;
			// int endy = 1;
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

					for (int c = 0; c < bins[i].size(); c++) {
						for (int d = 0; d < bins[bin_num].size(); d++) {
							//printf("apply_force begin");
							apply_force(*bins[i][c], *bins[bin_num][d], &dmin, &davg, &navg);
							//printf("apply_force end\n");
						}
					}
				}
			}
			//printf("Single bin completed\n");
		}
		printf("\nAll bins completed\n");
		//
		//  move particles
		//  The particles must also be moved between bins as necessary.
		//
		double binsize = cutoff * 2;

		for (int b = 0; b < num_bins; b++)
		{//Insert logic here
			int size = bins[b].size();
			for (int p = 0; p < size;) {
				//printf("Moving particle in bin %d, p = %d\n", b, p);
				move(*bins[b][p]);

				int x = floor(bins[b][p]->x / binsize);
				int y = floor(bins[b][p]->y / binsize);
				if (y * num_bin_row + x != b)
				{
					printf("p is %d\n", p);
					printf("Moving particles from bin %d to %d\n", b, x+y*num_bin_row);
					temp_move.push_back(bins[b][p]);
					bins[b].erase(bins[b].begin() + p);
					size--;
				}
				else{
					p++;
				}
			}
		}
		for(int i = 0; i < temp_move.size(); i++){
			int x = floor(temp_move[i]->x / binsize);
			int y = floor(temp_move[i]->y / binsize);
			printf("Pushing particle into bin[%d]\n", x+y*num_bin_row);
					
			bins[x + y * num_bin_row].push_back(temp_move[i]);
			printf("Pushback complete\n");
		}
		temp_move.clear();
		printf("temp_move cleared\n");

		if (find_option(argc, argv, "-no") == -1)
		{
			//
			// Computing statistical data
			//
			if (navg) {
				absavg += davg / navg;
				nabsavg++;
			}
			if (dmin < absmin) absmin = dmin;

			//
			//  save if necessary
			//
			if (fsave && (step%SAVEFREQ) == 0)
				save(fsave, n, particles);
		}
		printf("Nextloop. Step: %d\n", step);
	}
	simulation_time = read_timer() - simulation_time;

	printf("n = %d, simulation time = %g seconds", n, simulation_time);

	if (find_option(argc, argv, "-no") == -1)
	{
		if (nabsavg) absavg /= nabsavg;
		//
		//  -the minimum distance absmin between 2 particles during the run of the simulation
		//  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
		//  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
		//
		//  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
		//
		printf(", absmin = %lf, absavg = %lf", absmin, absavg);
		if (absmin < 0.4) printf("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
		if (absavg < 0.8) printf("\nThe average distance is below 0.8 meaning that most particles are not interacting");
	}
	printf("\n");

	//
	// Printing summary data
	//
	if (fsum)
		fprintf(fsum, "%d %g\n", n, simulation_time);

	//
	// Clearing space
	//
	if (fsum)
		fclose(fsum);
	free(particles);
	if (fsave)
		fclose(fsave);
	scanf("Press any key to return\n");
	return 0;
}
