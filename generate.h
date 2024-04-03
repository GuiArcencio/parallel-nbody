#ifndef GENERATE_H
#define GENERATE_H

typedef struct {
    float x, y, z;
    float x_old, y_old, z_old;
    float vx, vy, vz;
    float ax, ay, az;
    float mass;
} Particle;

Particle *generate_bodies(unsigned int N, unsigned int seed);

#endif 
