#include "generate.h"

#include <stdlib.h>

Particle *generate_bodies(unsigned int N, unsigned int seed) {
    Particle *particles = (Particle*) malloc(N * sizeof(Particle));
    srand(seed);

    for (int i = 0; i < N; i++) {
        particles[i].x = ((double) rand()) / RAND_MAX;
        particles[i].y = ((double) rand()) / RAND_MAX;
        particles[i].z = ((double) rand()) / RAND_MAX;
        particles[i].vx = ((double) rand()) / RAND_MAX;
        particles[i].vy = ((double) rand()) / RAND_MAX;
        particles[i].vz = ((double) rand()) / RAND_MAX;
        particles[i].mass = (1 * ((double) rand()) / RAND_MAX) + 0.001;
    }

    return particles;
}
