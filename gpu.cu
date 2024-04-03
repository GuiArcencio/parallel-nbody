#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// #define DEBUG
#define BLOCKSIZE 256
#define ETA 0.05
#define EPSILON 0.001

void generate_bodies(float4 *particles, float3 *velocities, unsigned int N, unsigned int seed) {
    srand(seed);

    for (int i = 0; i < N; i++) {
        particles[i].x = ((float) rand()) / RAND_MAX;
        particles[i].y = ((float) rand()) / RAND_MAX;
        particles[i].z = ((float) rand()) / RAND_MAX;
        velocities[i].x = ((float) rand()) / RAND_MAX;
        velocities[i].y = ((float) rand()) / RAND_MAX;
        velocities[i].z = ((float) rand()) / RAND_MAX;
        particles[i].w = (1 * ((float) rand()) / RAND_MAX) + 0.001;
    }
}

__device__
float3 interaction(float3 a_i, float4 p_i, float4 p_j, float epsilon_squared) {
    float3 r;   
    r.x = p_i.x - p_j.x;
    r.y = p_i.y - p_j.y;
    r.z = p_i.z - p_j.z;  
    float r_squared = r.x * r.x + r.y * r.y + r.z * r.z + epsilon_squared;

    float coef = p_j.w / sqrt(r_squared*r_squared*r_squared);  
    a_i.x -= coef * r.x;
    a_i.y -= coef * r.y;
    a_i.z -= coef * r.z;
    return a_i;
}

__global__
void compute_accelerations(float4 *p, float3 *a, unsigned int N, float *accels, float epsilon) {
    __shared__ float4 p_buffer[BLOCKSIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float epsilon_squared = epsilon * epsilon;
        float4 p_i = p[i];
        float3 a_i = {0.0, 0.0, 0.0};

        // Data tiling using shared memory
        for (int tile = 0; tile < gridDim.x; tile++) {
            p_buffer[threadIdx.x] = p[tile*BLOCKSIZE + threadIdx.x];

            __syncthreads();

            for (int j = 0; j < BLOCKSIZE && tile*BLOCKSIZE + j < N; j++)
                if (tile*BLOCKSIZE + j != i)
                    a_i = interaction(a_i, p_i, p_buffer[j], epsilon_squared);

            __syncthreads();
        }

        a[i] = a_i;
        accels[i] = sqrt(a_i.x*a_i.x + a_i.y*a_i.y + a_i.z*a_i.z);
    }
}

__global__
void first_step_particles(
    float4 *p,
    float3 *p_old,
    float3 *v,
    float3 *a,
    float *accels,
    float *dt_old_p,
    unsigned int N,
    float eta,
    float epsilon
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float dt = sqrt(eta * epsilon / accels[0]);

        p_old[i].x = p[i].x;
        p_old[i].y = p[i].y;
        p_old[i].z = p[i].z;
        p[i].x = p_old[i].x + v[i].x * dt + 0.5 * a[i].x * dt * dt;
        p[i].y = p_old[i].y + v[i].y * dt + 0.5 * a[i].y * dt * dt;
        p[i].z = p_old[i].z + v[i].z * dt + 0.5 * a[i].z * dt * dt;

        if (i == 0) *dt_old_p = dt;
    }
}

__global__
void step_particles(
    float4 *p,
    float3 *p_old,
    float3 *a,
    float * accels,
    unsigned int N,
    float eta,
    float epsilon,
    float *dt_old_p
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float dt = sqrt(eta * epsilon / accels[0]);
        float dt_old = *dt_old_p;

        float new_x = p[i].x + (p[i].x - p_old[i].x) * (dt/dt_old) + a[i].x * dt * (dt + dt_old) / 2.0;
        float new_y = p[i].y + (p[i].y - p_old[i].y) * (dt/dt_old) + a[i].y * dt * (dt + dt_old) / 2.0;
        float new_z = p[i].z + (p[i].z - p_old[i].z) * (dt/dt_old) + a[i].z * dt * (dt + dt_old) / 2.0;

        p_old[i].x = p[i].x;
        p_old[i].y = p[i].y;
        p_old[i].z = p[i].z;
        p[i].x = new_x;
        p[i].y = new_y;
        p[i].z = new_z;

        if (i == 0) *dt_old_p = dt;
    }
}

__global__
void parallel_max(float *accels, unsigned int N, int step_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_threads = N >> 1;

    if (i < n_threads) {
        int first_index = i * step_size * 2;
        int second_index = first_index + step_size;
        float first = accels[first_index];
        float second = accels[second_index];
        float biggest = first > second ? first : second;

        if (N % 2 != 0 && i == n_threads - 1) {
            float third = accels[second_index + step_size];
            biggest = third > biggest ? third : biggest;
        }

        accels[first_index] = biggest;
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
    float eta, epsilon;
    float4 *p = (float4*) malloc(N*sizeof(float4));
    float3 *v = (float3*) malloc(N*sizeof(float3));
    generate_bodies(p, v, N, seed);

    eta = ETA;
    epsilon = EPSILON;

    struct timespec start, finish;
    float elapsed;

    clock_gettime(CLOCK_MONOTONIC, &start);

    float4 *d_p;
    float3 *d_p_old, *d_v, *d_a;
    float *d_accels, *d_dt_old;
    float *accels = (float*) malloc(N*sizeof(float));
    cudaMalloc(&d_p, N*sizeof(float4));
    cudaMalloc(&d_p_old, N*sizeof(float3));
    cudaMalloc(&d_v, N*sizeof(float3));
    cudaMalloc(&d_a, N*sizeof(float3));
    cudaMalloc(&d_accels, N*sizeof(float));
    cudaMalloc(&d_dt_old, sizeof(float));
    cudaMemcpyAsync(d_p, p, N*sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_v, v, N*sizeof(float3), cudaMemcpyHostToDevice);

    int grid_size = (N + BLOCKSIZE - 1) / BLOCKSIZE;

    // First step: use velocities
    compute_accelerations<<<grid_size, BLOCKSIZE>>>(
        d_p, d_a, N, d_accels, epsilon
    );
    
    // Max-reduction on GPU
    int num_accels = N;
    int step_size = 1;
    int accel_grid_size;
    while (num_accels > 1) {
        accel_grid_size = ((num_accels >> 1) + BLOCKSIZE - 1) / BLOCKSIZE;
        parallel_max<<<accel_grid_size, BLOCKSIZE>>>(
            d_accels, num_accels, step_size
        );
        num_accels >>= 1;
        step_size <<= 1;
    }

    first_step_particles<<<grid_size, BLOCKSIZE>>>(
        d_p, d_p_old, d_v, d_a, d_accels, d_dt_old, N, eta, epsilon
    );

    for (int step = 1; step < N_steps; step++) {
        compute_accelerations<<<grid_size, BLOCKSIZE>>>(
            d_p, d_a, N, d_accels, epsilon
        );

        // Max-reduction on GPU
        num_accels = N;
        step_size = 1;
        while (num_accels > 1) {
            accel_grid_size = (num_accels + BLOCKSIZE - 1) / BLOCKSIZE;
            parallel_max<<<accel_grid_size, BLOCKSIZE>>>(
                d_accels, num_accels, step_size
            );
            num_accels >>= 1;
            step_size <<= 1;
        }

        step_particles<<<grid_size, BLOCKSIZE>>>(
            d_p, d_p_old, d_a, d_accels, N, eta, epsilon, d_dt_old
        );
    }

    cudaMemcpy(p, d_p, N*sizeof(float4), cudaMemcpyDeviceToHost);

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