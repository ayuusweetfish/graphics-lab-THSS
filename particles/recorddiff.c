#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 8888

typedef struct particle_header {
  float radius;
  float mass;
  float elas;
  float body;
} particle_header;

typedef struct particle {
  float force[5][3];
  float pos[3];
  float vel[3];
  float contact[3];
} particle;

void printparticle(const particle p)
{
  for (int i = 0; i < 5; i++)
    printf("f[%d] = (%.9f, %.9f, %.9f)\n", i, p.force[i][0], p.force[i][1], p.force[i][2]);
  printf("pos = (%.9f, %.9f, %.9f)\n", p.pos[0], p.pos[1], p.pos[2]);
  printf("vel = (%.9f, %.9f, %.9f)\n", p.vel[0], p.vel[1], p.vel[2]);
  printf("contact = %d, %d, %d\n", (int)p.contact[0], (int)p.contact[1], (int)p.contact[2]);
}

int main(int argc, char *argv[])
{
  int framesize = sizeof(particle) * N;
  int headersize = sizeof(particle_header) * N;

  FILE *f = fopen("r1.bin", "rb");
  fseek(f, 0, SEEK_END);
  int nframes = (ftell(f) - headersize) / framesize;
  fseek(f, 0, SEEK_SET);
  printf("%d particles, %d frames\n", N, nframes);

  particle_header *phs1 = (particle_header *)malloc(sizeof(particle_header) * N);
  fread(phs1, sizeof(particle_header) * N, 1, f);
  particle *ps1 = (particle *)malloc(nframes * sizeof(particle) * N);
  fread(ps1, nframes * sizeof(particle) * N, 1, f);

  fclose(f);
  f = fopen("r2.bin", "rb");

  particle_header *phs2 = (particle_header *)malloc(sizeof(particle_header) * N);
  fread(phs2, sizeof(particle_header) * N, 1, f);
  particle *ps2 = (particle *)malloc(nframes * sizeof(particle) * N);
  fread(ps2, nframes * sizeof(particle) * N, 1, f);

  bool diff = false;
  for (int i = 0; !diff && i < nframes; i++) {
    for (int j = 0; j < N; j++) {
      if (memcmp(&ps1[i * N + j], &ps2[i * N + j], sizeof(particle) - sizeof(float)*3) != 0) {
        printf("frame %d, particle %d:\n", i, j);
        printf("== A ==\n");
        printparticle(ps1[i * N + j]);
        printf("== B ==\n");
        printparticle(ps2[i * N + j]);
        diff = true;
      }
    }
  }

  if (!diff) puts("identical");

  return 0;
}
