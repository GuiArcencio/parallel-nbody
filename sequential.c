#include "generate.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// #define DEBUG
#define ETA 0.05
#define EPSILON 0.001

void compute_accelerations(Particle *p, unsigned int N, float *max_a, float epsilon);

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
    float elapsed;

    clock_gettime(CLOCK_MONOTONIC, &start); 

    // First step: use velocities
    compute_accelerations(p, N, &max_a, epsilon);
    dt = sqrt(eta * epsilon / max_a);
    for (int i = 0; i < N; i++) {
        p[i].x_old = p[i].x;
        p[i].y_old = p[i].y;
        p[i].z_old = p[i].z;
        p[i].x = p[i].x_old + p[i].vx * dt + 0.5 * p[i].ax * dt * dt;
        p[i].y = p[i].y_old + p[i].vy * dt + 0.5 * p[i].ay * dt * dt;
        p[i].z = p[i].z_old + p[i].vz * dt + 0.5 * p[i].az * dt * dt;
    }
    dt_old = dt;

    for (int step = 1; step < N_steps; step++) {
        compute_accelerations(p, N, &max_a, epsilon);
        dt = sqrt(eta * epsilon / max_a);
        
        for (int i = 0; i < N; i++) {
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

        dt_old = dt;
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = finish.tv_sec - start.tv_sec;
    elapsed += (finish.tv_nsec - start.tv_nsec) * 1e-9;

    printf("%lf\n", elapsed);

#ifdef DEBUG
    for (int i = 0; i < N; i++) 
        printf("%.6lf,%.6lf,%.6lf\n", p[i].x, p[i].y, p[i].z);
#endif

    return 0;
}

void compute_accelerations(Particle *p, unsigned int N, float *max_a, float epsilon) {
    float epsilon_squared = epsilon * epsilon;
    *max_a = -1.0;

    for (int i = 0; i < N; i++) {
        float rx, ry, rz, r_squared;
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

        float a = p[i].ax*p[i].ax + p[i].ay*p[i].ay + p[i].az*p[i].az;
        if (a > *max_a)
            *max_a = a;
    }
    
    *max_a = sqrt(*max_a);
}

