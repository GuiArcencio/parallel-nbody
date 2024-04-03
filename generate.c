#include "generate.h"

#include <stdlib.h>

Particle *generate_bodies(unsigned int N, unsigned int seed) {
    Particle *particles = (Particle*) malloc(N * sizeof(Particle));
    srand(seed);

    for (int i = 0; i < N; i++) {
        particles[i].x = ((float) rand()) / RAND_MAX;
        particles[i].y = ((float) rand()) / RAND_MAX;
        particles[i].z = ((float) rand()) / RAND_MAX;
        particles[i].vx = ((float) rand()) / RAND_MAX;
        particles[i].vy = ((float) rand()) / RAND_MAX;
        particles[i].vz = ((float) rand()) / RAND_MAX;
        particles[i].mass = (1 * ((float) rand()) / RAND_MAX) + 0.001;
    }

    return particles;
}
