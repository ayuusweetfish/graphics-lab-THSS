#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "playback_utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 888

typedef struct particle_header {
  float radius;
  float body;
} particle_header;

typedef struct particle {
  float force[5][3];
  float pos[3];
} particle;

void MyDrawSphereWires(Vector3 centerPos, float radius, int rings, int slices, Color color);

int main()
{
  int framesize = sizeof(particle) * N;
  int headersize = sizeof(particle_header) * N;

  FILE *f = fopen("record.bin", "rb");
  fseek(f, 0, SEEK_END);
  int nframes = (ftell(f) - headersize) / framesize;
  fseek(f, 0, SEEK_SET);
  printf("%d particles, %d frames\n", N, nframes);

  particle_header *phs = (particle_header *)malloc(sizeof(particle_header) * N);
  fread(phs, sizeof(particle_header) * N, 1, f);
  particle *ps = (particle *)malloc(nframes * sizeof(particle) * N);
  fread(ps, nframes * sizeof(particle) * N, 1, f);

  Color *bodycolour = (Color *)malloc(sizeof(Color) * N);
  for (int i = 0; i < N; i++) {
    int r = rand() % 64;
    int g = rand() % 64;
    int b = rand() % 64;
    int base = 160 + (64 - (r * 2 + g * 5 + b) / 8) / 2;
    bodycolour[i] = (Color){r + base, g + base, b + base, 255};
  }

  InitWindow(1280, 720, "Playback");
  SetTargetFPS(60);

  Camera3D camera = (Camera3D){
    (Vector3){4, 5, 6},
    (Vector3){0, 0, 0},
    (Vector3){0, 1, 0},
    30,
    CAMERA_PERSPECTIVE
  };
  // SetCameraMode(camera, CAMERA_FIRST_PERSON);
  // SetCameraMode(camera, CAMERA_ORBITAL);

  int frame = 0;

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(RAYWHITE);

    if (IsKeyDown(KEY_SPACE))
      frame = (frame + (IsKeyDown(KEY_LEFT_SHIFT) ? (nframes - 10) : 10)) % nframes;
    else if (IsKeyPressed(KEY_DOWN))
      frame = (frame + (IsKeyDown(KEY_LEFT_SHIFT) ? 1 : 10)) % nframes;
    else if (IsKeyPressed(KEY_UP))
      frame = (frame + nframes - (IsKeyDown(KEY_LEFT_SHIFT) ? 1 : 10)) % nframes;
    int framebase = frame * N;

    if (IsKeyDown(KEY_A) || IsKeyDown(KEY_D)) {
      int rate = 0;
      if (IsKeyDown(KEY_A)) rate -= 1;
      if (IsKeyDown(KEY_D)) rate += 1;
      Vector3 delta = Vector3Scale(
        Vector3Normalize(Vector3CrossProduct(
          Vector3Subtract(camera.target, camera.position),
          camera.up
        )),
        rate * 0.03
      );
      camera.position = Vector3Add(camera.position, delta);
      camera.target = Vector3Add(camera.target, delta);
    }
    if (IsKeyDown(KEY_S) || IsKeyDown(KEY_W)) {
      int rate = 0;
      if (IsKeyDown(KEY_S)) rate -= 1;
      if (IsKeyDown(KEY_W)) rate += 1;
      Vector3 delta = Vector3Scale(
        Vector3Normalize(Vector3Subtract(camera.target, camera.position)),
        rate * 0.12
      );
      camera.position = Vector3Add(camera.position, delta);
      camera.target = Vector3Add(camera.target, delta);
    }
    if (IsKeyDown(KEY_Q) || IsKeyDown(KEY_Z)) {
      int rate = 0;
      if (IsKeyDown(KEY_Z)) rate -= 1;
      if (IsKeyDown(KEY_Q)) rate += 1;
      Vector3 delta = Vector3Scale(
        Vector3Normalize(camera.up),
        rate * 0.03
      );
      camera.position = Vector3Add(camera.position, delta);
      camera.target = Vector3Add(camera.target, delta);
    }

    int forcemask = 0;
    if (IsKeyDown(KEY_ONE)) forcemask |= (1 << 0);
    if (IsKeyDown(KEY_TWO)) forcemask |= (1 << 1);
    if (IsKeyDown(KEY_THREE)) forcemask |= (1 << 2);
    if (IsKeyDown(KEY_FOUR)) forcemask |= (1 << 3);
    if (IsKeyDown(KEY_FIVE)) forcemask |= (1 << 4);

    // UpdateCamera(&camera);
    if (IsKeyPressed(KEY_ZERO))
      camera.target = (Vector3){0, 0, 0};

    BeginMode3D(camera);
    for (int i = 0; i < N; i++) {
      Vector3 position = (Vector3){
        ps[framebase + i].pos[0],
        ps[framebase + i].pos[1],
        ps[framebase + i].pos[2],
      };
      /*DrawCircle3D(
        position,
        phs[i].radius,
        (Vector3){0, 1, 0},
        atan2(camera.position.z, -camera.position.x) * RAD2DEG + 90,
        BEIGE
      );*/
      MyDrawSphereWires(
        position,
        phs[i].radius,
        8, 8,
        bodycolour[(int)phs[i].body]
      );
      for (int j = 0; j < 5; j++) if (forcemask & (1 << j)) {
        Vector3 force = (Vector3){
          ps[framebase + i].force[j][0],
          ps[framebase + i].force[j][1],
          ps[framebase + i].force[j][2],
        };
        DrawLine3D(position,
          Vector3Add(position, Vector3Scale(force, 5e-4)),
          (Color){
            bodycolour[(int)phs[i].body].r * 2 / 3,
            bodycolour[(int)phs[i].body].g * 2 / 3,
            bodycolour[(int)phs[i].body].b * 2 / 3,
            255
          }
        );
      }
    }
    EndMode3D();

    static const char *forcenames[5] = {
      "repulsive", "damping", "shear", "floor collision", "floor friction"
    };
    for (int j = 0, k = 0; j < 5; j++) if (forcemask & (1 << j)) {
      DrawText(forcenames[j], 10, 10 + (k++) * 20, 16, BLACK);
    }
    EndDrawing();
  }

  CloseWindow();

  return 0;
}
