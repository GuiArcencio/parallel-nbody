#include "generate.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// #define DEBUG
#define BLOCKSIZE 256
#define ETA 0.05
#define EPSILON 0.001

__global__
void compute_accelerations(Particle *p, unsigned int N, float *accels, float epsilon) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float rx, ry, rz, r_squared;

    if (i < N) {
        float epsilon_squared = epsilon * epsilon;
        p[i].ax = 0;
        p[i].ay = 0;
        p[i].az = 0;
        for (int j = 0; j < N; j++)
            if (i != j) {
                rx = p[i].x - p[j].x;
                ry = p[i].y - p[j].y;
                rz = p[i].z - p[j].z;
                r_squared = rx*rx + ry*ry + rz*rz;

                float coef = p[j].mass / pow(r_squared + epsilon_squared, 1.5);
                p[i].ax -= coef * rx;
                p[i].ay -= coef * ry;
                p[i].az -= coef * rz;
            }    

        accels[i] = p[i].ax*p[i].ax + p[i].ay*p[i].ay + p[i].az*p[i].az;
    }
}

__global__
void first_step_particles(Particle *p, unsigned int N, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        p[i].x_old = p[i].x;
        p[i].y_old = p[i].y;
        p[i].z_old = p[i].z;
        p[i].x = p[i].x_old + p[i].vx * dt + 0.5 * p[i].ax * dt * dt;
        p[i].y = p[i].y_old + p[i].vy * dt + 0.5 * p[i].ay * dt * dt;
        p[i].z = p[i].z_old + p[i].vz * dt + 0.5 * p[i].az * dt * dt;
    }
}

__global__
void step_particles(Particle *p, unsigned int N, float dt, float dt_old) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float new_x = p[i].x + (p[i].x - p[i].x_old) * (dt/dt_old) + p[i].ax * dt * (dt + dt_old) / 2.0;
        float new_y = p[i].y + (p[i].y - p[i].y_old) * (dt/dt_old) + p[i].ay * dt * (dt + dt_old) / 2.0;
        float new_z = p[i].z + (p[i].z - p[i].z_old) * (dt/dt_old) + p[i].az * dt * (dt + dt_old) / 2.0;

        p[i].x_old = p[i].x; 
        p[i].y_old = p[i].y;
        p[i].z_old = p[i].z;
        p[i].x = new_x;
        p[i].y = new_y;
        p[i].z = new_z;
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Use: %s <num. of particles> <num. of steps> <seed>\n", argv[0]);
        exit(1);
    }
    int arg1 = atoi(argv[1]);
    int arg2 = atoi(argv[2]);
    int arg3 = atoi(argv[3]);
    if (arg1 <= 0 || arg2 <= 0 || arg3 <= 0) {
        fprintf(stderr, "Use: %s <num. of particles> <num. of steps> <seed>\n", argv[0]);
        exit(1);
    }

    unsigned int N = (unsigned int) arg1;
    unsigned int N_steps = (unsigned int) arg2;
    unsigned int seed = (unsigned int) arg3;
    float dt, dt_old, eta, epsilon, max_a;
    Particle *p = generate_bodies(N, seed); 

    eta = ETA;
    epsilon = EPSILON;
  
    struct timespec start, finish;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &start); 

    Particle *d_p;
    float *accels = (float*) malloc(N*sizeof(float));
    float *d_accels;
    cudaMalloc(&d_p, N*sizeof(Particle));
    cudaMalloc(&d_accels, N*sizeof(float));
    cudaMemcpyAsync(d_p, p, N*sizeof(Particle), cudaMemcpyHostToDevice);

    int grid_size = (N + BLOCKSIZE - 1) / BLOCKSIZE;

    // First step: use velocities
    compute_accelerations<<<grid_size, BLOCKSIZE>>>(
        d_p, N, d_accels, epsilon
    );
    cudaMemcpy(accels, d_accels, N*sizeof(float), cudaMemcpyDeviceToHost);
    max_a = accels[0];
    for (int i = 1; i < N; i++)
        if (accels[i] > max_a) max_a = accels[i];
    max_a = sqrt(max_a);
    dt = sqrt(eta * epsilon / max_a);

    first_step_particles<<<grid_size, BLOCKSIZE>>>(
        d_p, N, dt
    );
    
    dt_old = dt;

    for (int step = 1; step < N_steps; step++) {
        compute_accelerations<<<grid_size, BLOCKSIZE>>>(
            d_p, N, d_accels, epsilon
        );
        cudaMemcpy(accels, d_accels, N*sizeof(float), cudaMemcpyDeviceToHost);
        max_a = accels[0];
        for (int i = 1; i < N; i++)
            if (accels[i] > max_a) max_a = accels[i];
        max_a = sqrt(max_a);
        dt = sqrt(eta * epsilon / max_a);
   
        step_particles<<<grid_size, BLOCKSIZE>>>(
            d_p, N, dt, dt_old
        );

        dt_old = dt;
    }

    cudaMemcpy(p, d_p, N*sizeof(Particle), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = finish.tv_sec - start.tv_sec;
    elapsed += (finish.tv_nsec - start.tv_nsec) * 1e-9;

    printf("%lf\n", elapsed);

#ifdef DEBUG
    for (int i = 0; i < N; i++) 
        printf("%.6f,%.6f,%.6f\n", p[i].x, p[i].y, p[i].z);
#endif

    return 0;
}

