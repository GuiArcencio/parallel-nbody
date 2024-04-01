#ifndef GENERATE_H
#define GENERATE_H

typedef struct {
    double x, y, z;
    double x_old, y_old, z_old;
    double vx, vy, vz;
    double ax, ay, az;
    double mass;
} Particle;

Particle *generate_bodies(unsigned int N, unsigned int seed);

#endif 
