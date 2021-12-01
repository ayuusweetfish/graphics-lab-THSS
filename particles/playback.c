#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct particle_header {
  float radius;
  float mass;
  float elas;
  float body;
} particle_header;

typedef struct particle {
  float pos[3];
#if RECDEBUG
  float vel[3];
  float force[5][3];
  float contact[3];
#endif
} particle;

static void MyDrawSphereWires(Vector3 centerPos, float radius, Color color);
static void MyDrawCircleFilled3D(Vector3 center, float radius, Color color);

Font font, fontlarge, fontxlarge, fontxxlarge;
static void MyDrawText(const char *text, float posX, float posY, int fontSize, Color color)
{
  Font curfont = (
    fontSize == 16 ? font :
    fontSize == 24 ? fontlarge :
    fontSize == 40 ? fontxlarge :
    fontxxlarge);
  DrawTextEx(curfont, text, (Vector2){posX, posY}, fontSize, 0, color);
}
static void MyDrawTextCen(const char *text, float posX, float posY, int fontSize, Color color)
{
  Font curfont = (
    fontSize == 16 ? font :
    fontSize == 24 ? fontlarge :
    fontSize == 40 ? fontxlarge :
    fontxxlarge);
  Vector2 dims = MeasureTextEx(curfont, text, fontSize, 0);
  MyDrawText(text, posX - dims.x / 2, posY - dims.y / 2, fontSize, color);
}

#define norm3d(_v) \
  (sqrtf(((_v)[0])*((_v)[0]) + ((_v)[1])*((_v)[1]) + ((_v)[2])*((_v)[2])))

static void draw_frame(
  int N, int M,
  particle_header *phs, particle *ps,
  Camera camera, Color *bodycolour,
  int stepnum, int forcemask, int tintby, int selparticle
);

int main(int argc, char *argv[])
{
  const char *path = "record.bin";
  if (argc > 1) path = argv[1];

  FILE *f = fopen(path, "rb");

  int32_t intbuf[2];
  fread(intbuf, sizeof(int32_t), 2, f);
  int N = intbuf[0];  // Number of particles
  int M = intbuf[1];  // Number of bodies

  int framesize = sizeof(particle) * N;
  int headersize = sizeof(particle_header) * N + sizeof(int32_t) * 2;

  fseek(f, 0, SEEK_END);
  int nframes = (ftell(f) - headersize) / framesize;
  printf("%d particles, %d bodies, %d frames\n", N, M, nframes);

  fseek(f, sizeof(int32_t) * 2, SEEK_SET);
  particle_header *phs = (particle_header *)malloc(sizeof(particle_header) * N);
  fread(phs, sizeof(particle_header) * N, 1, f);
  particle *ps = (particle *)malloc(nframes * sizeof(particle) * N);
  fread(ps, nframes * sizeof(particle) * N, 1, f);

  Color *bodycolour = (Color *)malloc(sizeof(Color) * M);
  int randseed = 20211128;
  for (int i = 0; i < M; i++) {
    int r = ((randseed = ((randseed * 1103515245 + 12345) & 0x7fffffff)) >> 17) % 64;
    int g = ((randseed = ((randseed * 1103515245 + 12345) & 0x7fffffff)) >> 18) % 64;
    int b = ((randseed = ((randseed * 1103515245 + 12345) & 0x7fffffff)) >> 19) % 64;
    int base = 160 + (64 - (r * 2 + g * 5 + b) / 8) / 2;
    bodycolour[i] = (Color){r + base, g + base, b + base, 255};
  }

  int W = 1280;
  int H = 720;

  int titlemaxlen = 16 + strlen(path);
  char *title = (char *)malloc(titlemaxlen);
  snprintf(title, titlemaxlen, "Playback [%s]", path);
  InitWindow(W, H, title);
  SetTargetFPS(60);

  const char *fontpath = "AnonymousPro-Regular.ttf";
  font = LoadFontEx(fontpath, 32, 0, 256);
  fontlarge = LoadFontEx(fontpath, 48, 0, 256);
  fontxlarge = LoadFontEx(fontpath, 80, 0, 256);
  fontxxlarge = LoadFontEx(fontpath, 108, 0, 256);

  Camera3D camera = (Camera3D){
    (Vector3){4, 5, 6},
    (Vector3){0, 0, 0},
    (Vector3){0, 1, 0},
    45,
    CAMERA_PERSPECTIVE
  };

  // Temporarily trick
  void *data = malloc(W*2*H*2*4);
  Texture2D rendertex1 = LoadTextureFromImage((Image){
    data, W*2, H*2, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
  });
  free(data);

  int frame = 0;
  int forcemask = 0;
  int tintby = -1;
  int immerse = 1;

  while (!WindowShouldClose()) {
    int amount = (IsKeyDown(KEY_LEFT_SHIFT) ? 1 : 10);
    if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_SPACE)) frame = (frame + amount) % nframes;
    if (IsKeyDown(KEY_LEFT)) frame = (frame + nframes - amount) % nframes;
    if (IsKeyPressed(KEY_DOWN)) frame = (frame + amount) % nframes;
    if (IsKeyPressed(KEY_UP)) frame = (frame + nframes - amount) % nframes;
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
      Vector3 dir = Vector3Subtract(camera.target, camera.position);
      Vector3 delta = Vector3Scale(
        Vector3Normalize(   // Assume camera.up is unit
          Vector3Subtract(dir,
            Vector3Scale(camera.up, Vector3DotProduct(dir, camera.up)))
        ),
        rate * 0.03
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
    if (IsKeyDown(KEY_E) || IsKeyDown(KEY_R)) {
      int rate = 0;
      if (IsKeyDown(KEY_E)) rate -= 1;
      if (IsKeyDown(KEY_R)) rate += 1;
      camera.target = Vector3Add(
        camera.position,
        Vector3Transform(
          Vector3Subtract(camera.target, camera.position),
          MatrixRotate(camera.up, rate * -0.01))
      );
    }

    if (IsKeyPressed(KEY_ONE)) forcemask ^= (1 << 0);
    if (IsKeyPressed(KEY_TWO)) forcemask ^= (1 << 1);
    if (IsKeyPressed(KEY_THREE)) forcemask ^= (1 << 2);
    if (IsKeyPressed(KEY_FOUR)) forcemask ^= (1 << 3);
    if (IsKeyPressed(KEY_FIVE)) forcemask ^= (1 << 4);

    if (IsKeyPressed(KEY_ZERO)) tintby = (tintby + 2) % 4 - 1;

    if (IsKeyPressed(KEY_GRAVE)) immerse ^= 1;

    // Find the particle being pointed at
    float bestdistance = 1e24;
    int selparticle = -1;
    if (!immerse) {
      Ray ray = GetMouseRay(GetMousePosition(), camera);
      for (int i = 0; i < N; i++) {
        Vector3 position = (Vector3){
          ps[framebase + i].pos[0],
          ps[framebase + i].pos[1],
          ps[framebase + i].pos[2],
        };
        RayCollision colli = GetRayCollisionSphere(ray, position, phs[i].radius);
        if (colli.hit && colli.distance < bestdistance && colli.distance >= 0.2) {
          bestdistance = colli.distance;
          selparticle = i;
        }
      }
    }

    // Actual drawing
    BeginDrawing();
    // raylib's render textures behaves differently from default render target,
    // hence raw pixels are read back. Performance effects are negligible
    // as pixels need to be read back anyway.
    // Draw the frame
    draw_frame(
      N, M,
      phs, ps + framebase,
      camera, bodycolour,
      frame, forcemask, tintby, selparticle
    );
    rlDrawRenderBatchActive();  // Flush
    // Read back pixels
    // XXX: Hack: directly reads pixels, bypassing raylib's internals
    unsigned char *pixels = rlReadScreenPixels(W*2, H*2);
    UpdateTexture(rendertex1, pixels);
    RL_FREE(pixels);

    ClearBackground((Color){48, 48, 48});
    DrawTexturePro(
      rendertex1,
      (Rectangle){0, 0, W*2, H*2},
      (Rectangle){W*0.5, H*0.25, W*0.5, H*0.5},
      (Vector2){0, 0},
      0, WHITE);
    MyDrawTextCen("Mass", W/2, H*0.15, 54, WHITE);
    MyDrawTextCen("Particles with large masses push others away", W/2, H*0.85, 40, WHITE);
    EndDrawing();

    if (IsKeyPressed(KEY_ENTER)) {
      char s[32];
      snprintf(s, sizeof s, "wf%02d.png", frame);
      TakeScreenshot(s);
    }
  }

  CloseWindow();

  return 0;
}

// Draw a frame
void draw_frame(
  int N, int M,
  particle_header *phs, particle *ps,
  Camera camera, Color *bodycolour,
  int stepnum, int forcemask, int tintby, int selparticle
) {
  ClearBackground(RAYWHITE);

  // Draw 3D models
  BeginMode3D(camera);
  for (int i = 0; i < N; i++) {
    Vector3 position = (Vector3){
      ps[i].pos[0],
      ps[i].pos[1],
      ps[i].pos[2],
    };
    Color tint = bodycolour[(int)phs[i].body];
    switch (tintby) {
      case 0: // mass
        tint.a = Remap(phs[i].mass, 0, 200, 16, 255);
        break;
      case 1: // elasticity
        tint.a = Remap(phs[i].elas, 0.9, 0.3, 240, 255);
        tint.r = Remap(phs[i].elas, 0.9, 0.3, 240, tint.r);
        tint.g = Remap(phs[i].elas, 0.9, 0.3, 240, tint.g);
        tint.b = Remap(phs[i].elas, 0.9, 0.3, 240, tint.b);
        break;
      #if RECDEBUG
      case 2: // contact force
      {
        // tint.a = (ps[i].contact[0] == -1 ? 16 : 255);
        float contactf[3] = {0, 0, 0};
        for (int j = 0; j < 3; j++)
          for (int k = 0; k < 3; k++)
            contactf[k] += ps[i].force[j][k];
        tint.a = Clamp(Remap(norm3d(contactf), 0, 500, 16, 255), 16, 255);
        break;
      }
      #endif
      default: break;
    }
    if (i != selparticle)
      MyDrawSphereWires(position, phs[i].radius, tint);
    else
      DrawSphereEx(position, phs[i].radius, 12, 12, tint);
    // Shadow
    if (position.y <= phs[i].radius * 4) {
      float radius = Clamp((phs[i].radius * 4 - position.y) / 3, 0, phs[i].radius);
      int alpha = Clamp(Remap(position.y, phs[i].radius * 4, phs[i].radius * 2, 0, 255), 0, 255);
      MyDrawCircleFilled3D(
        (Vector3){position.x, -1e-4, position.z},
        radius,
        (Color){236, 236, 236, alpha}
      );
    }
  #if RECDEBUG
    for (int j = 0; j < 5; j++) if (forcemask & (1 << j)) {
      Vector3 force = (Vector3){
        ps[i].force[j][0],
        ps[i].force[j][1],
        ps[i].force[j][2],
      };
      DrawLine3D(position,
        Vector3Add(position, Vector3Scale(force, 5e-4)),
        (Color){
          (int)bodycolour[(int)phs[i].body].r * 3 / 4,
          (int)bodycolour[(int)phs[i].body].g * 3 / 4,
          (int)bodycolour[(int)phs[i].body].b * 3 / 4,
          255
        }
      );
    }
  #endif
  }
  // Floor
  const int gridsize = 30;
  const float cellsize = 0.5;
  for (int x = -gridsize; x <= gridsize; x++)
    for (int z = -gridsize; z <= gridsize; z++) {
      DrawLine3D(
        (Vector3){x * cellsize, 0, -gridsize * cellsize},
        (Vector3){x * cellsize, 0, +gridsize * cellsize},
        (Color){192, 192, 192, 64});
      DrawLine3D(
        (Vector3){-gridsize * cellsize, 0, z * cellsize},
        (Vector3){+gridsize * cellsize, 0, z * cellsize},
        (Color){192, 192, 192, 64});
    }
  EndMode3D();

  char s[256];
  snprintf(s, sizeof s, "step %04d", stepnum);
  MyDrawText(s, 10, 10, 24, BLACK);

  if (tintby != -1) {
    static const char *tinttypes[3] = {
      "mass", "elasticity", "contact force"
    };
    snprintf(s, sizeof s, "tint by %s", tinttypes[tintby]);
    MyDrawText(s, 10, 40, 16, BLACK);
  }

  static const char *forcenames[5] = {
    "repulsive", "damping", "shear", "groundup", "friction"
  };
  if (forcemask != 0) {
    char *ss = s;
    char *end = s + sizeof s;
    *ss = '\0';
    for (int j = 0, first = 1; j < 5; j++) if (forcemask & (1 << j)) {
      ss += strlcat(ss, (first ? "forces: " : ", "), end - ss);
      ss += strlcat(ss, forcenames[j], end - ss);
      first = 0;
    }
    MyDrawText(s, 10, 60, 16, BLACK);
  }

  int ybase = 70;
  int yskip = 20;
  if (selparticle != -1) {
    Color tint = (Color){
      (int)bodycolour[(int)phs[selparticle].body].r * 3 / 4,
      (int)bodycolour[(int)phs[selparticle].body].g * 3 / 4,
      (int)bodycolour[(int)phs[selparticle].body].b * 3 / 4,
      255
    };
    snprintf(s, sizeof s, "body %d  particle %d",
      (int)phs[selparticle].body, selparticle);
    MyDrawText(s, 10, (ybase += yskip), 16, tint);
    snprintf(s, sizeof s, "pos     (%.4f, %.4f, %.4f)",
      ps[selparticle].pos[0],
      ps[selparticle].pos[1],
      ps[selparticle].pos[2]);
    MyDrawText(s, 10, (ybase += yskip), 16, tint);
  #if RECDEBUG
    snprintf(s, sizeof s, "vel     (%.4f, %.4f, %.4f)",
      ps[selparticle].vel[0],
      ps[selparticle].vel[1],
      ps[selparticle].vel[2]);
    MyDrawText(s, 10, (ybase += yskip), 16, tint);
  #endif
    snprintf(s, sizeof s, "radius  %.4f\n", phs[selparticle].radius);
    MyDrawText(s, 10, (ybase += yskip), 16, tint);
    snprintf(s, sizeof s, "mass    %.4f\n", phs[selparticle].mass);
    MyDrawText(s, 10, (ybase += yskip), 16, tint);
    snprintf(s, sizeof s, "elast   %.4f\n", phs[selparticle].elas);
    MyDrawText(s, 10, (ybase += yskip), 16, tint);
    ybase += 10;
  #if RECDEBUG
    for (int j = 0; j < 5; j++) {
      snprintf(s, sizeof s, "%-10s %.5f", forcenames[j],
        norm3d(ps[selparticle].force[j]));
      MyDrawText(s, 10, (ybase += yskip), 16, (Color){
        (int)bodycolour[(int)phs[selparticle].body].r * 2 / 3,
        (int)bodycolour[(int)phs[selparticle].body].g * 2 / 3,
        (int)bodycolour[(int)phs[selparticle].body].b * 2 / 3,
        255
      });
    }
    ybase += 10;
    if (ps[selparticle].contact[0] != -1) {
      char *ss = s;
      char *end = s + sizeof s;
      for (int j = 0; j < 3; j++)
        if (ps[selparticle].contact[j] != -1) {
          ss += snprintf(ss, end - ss, "%s%d",
            j == 0 ? "contact: " : ", ",
            (int)ps[selparticle].contact[j]
          );
        }
      MyDrawText(s, 10, (ybase += yskip), 16, (Color){64, 64, 64, 255});
    }
  #endif
  }
}

// Drawing subroutines
// Copied from rmodels.c, raysan5/raylib@ed125f27b01053dfd814a0d847ce7534c0a3ea8d
// Draw sphere wires
void MyDrawSphereWires(Vector3 centerPos, float radius, Color color)
{
    int numVertex = 30 * 4;
    // XXX: rlBegin pads vertices to alignment, may break prior checks?
    rlCheckRenderBatchLimit(numVertex * 16);

#define rlVertex3f(_x, _y, _z) rlVertex3f( \
  (_x) * radius + centerPos.x, \
  (_y) * radius + centerPos.y, \
  (_z) * radius + centerPos.z  \
)
    rlBegin(RL_LINES);
        rlColor4ub(color.r, color.g, color.b, color.a);
#define Ax 0.525731112
#define Bx 0.850650808
#define Cx 0.000000000
#define Dx 0.809016994
#define Ex 0.500000000
#define Fx 0.309016994
#define Gx 1.000000000
#define i(a, b, c, d, e, f) rlVertex3f(a##x, b##x, c##x); rlVertex3f(d##x, e##x, f##x);
        i(-A,B,C,-D,E,F)i(-D,E,F,-F,D,E)i(-B,C,A,-E,F,D)i(C,A,B,-F,D,E)i(-D,E,F,-E,F,D)i(-E,F,D,-F,D,E)i(-A,B,C,-F,D,E)i(-F,D,E,C,G,C)i(C,A,B,F,D,E)i(A,B,C,C,G,C)i(-F,D,E,F,D,E)i(F,D,E,C,G,C)i(-A,B,C,C,G,C)i(C,G,C,-F,D,-E)i(A,B,C,F,D,-E)i(C,A,-B,-F,D,-E)i(C,G,C,F,D,-E)i(F,D,-E,-F,D,-E)i(-A,B,C,-F,D,-E)i(-F,D,-E,-D,E,-F)i(C,A,-B,-E,F,-D)i(-B,C,-A,-D,E,-F)i(-F,D,-E,-E,F,-D)i(-E,F,-D,-D,E,-F)i(-A,B,C,-D,E,-F)i(-D,E,-F,-D,E,F)i(-B,C,-A,-G,C,C)i(-B,C,A,-D,E,F)i(-D,E,-F,-G,C,C)i(-G,C,C,-D,E,F)i(A,B,C,F,D,E)i(F,D,E,D,E,F)i(C,A,B,E,F,D)i(B,C,A,D,E,F)i(F,D,E,E,F,D)i(E,F,D,D,E,F)i(C,A,B,-E,F,D)i(-E,F,D,C,C,G)i(-B,C,A,-E,-F,D)i(C,-A,B,C,C,G)i(-E,F,D,-E,-F,D)i(-E,-F,D,C,C,G)i(-B,C,A,-G,C,C)i(-G,C,C,-D,-E,F)i(-B,C,-A,-D,-E,-F)i(-A,-B,C,-D,-E,F)i(-G,C,C,-D,-E,-F)i(-D,-E,-F,-D,-E,F)i(-B,C,-A,-E,F,-D)i(-E,F,-D,-E,-F,-D)i(C,A,-B,C,C,-G)i(C,-A,-B,-E,-F,-D)i(-E,F,-D,C,C,-G)i(C,C,-G,-E,-F,-D)i(C,A,-B,F,D,-E)i(F,D,-E,E,F,-D)i(A,B,C,D,E,-F)i(B,C,-A,E,F,-D)i(F,D,-E,D,E,-F)i(D,E,-F,E,F,-D)i(A,-B,C,D,-E,F)i(D,-E,F,F,-D,E)i(B,C,A,E,-F,D)i(C,-A,B,F,-D,E)i(D,-E,F,E,-F,D)i(E,-F,D,F,-D,E)i(A,-B,C,F,-D,E)i(F,-D,E,C,-G,C)i(C,-A,B,-F,-D,E)i(-A,-B,C,C,-G,C)i(F,-D,E,-F,-D,E)i(-F,-D,E,C,-G,C)i(A,-B,C,C,-G,C)i(C,-G,C,F,-D,-E)i(-A,-B,C,-F,-D,-E)i(C,-A,-B,F,-D,-E)i(C,-G,C,-F,-D,-E)i(-F,-D,-E,F,-D,-E)i(A,-B,C,F,-D,-E)i(F,-D,-E,D,-E,-F)i(C,-A,-B,E,-F,-D)i(B,C,-A,D,-E,-F)i(F,-D,-E,E,-F,-D)i(E,-F,-D,D,-E,-F)i(A,-B,C,D,-E,-F)i(D,-E,-F,D,-E,F)i(B,C,-A,G,C,C)i(B,C,A,D,-E,F)i(D,-E,-F,G,C,C)i(G,C,C,D,-E,F)i(C,-A,B,E,-F,D)i(E,-F,D,C,C,G)i(B,C,A,E,F,D)i(C,A,B,C,C,G)i(E,-F,D,E,F,D)i(E,F,D,C,C,G)i(-A,-B,C,-F,-D,E)i(-F,-D,E,-D,-E,F)i(C,-A,B,-E,-F,D)i(-B,C,A,-D,-E,F)i(-F,-D,E,-E,-F,D)i(-E,-F,D,-D,-E,F)i(C,-A,-B,-F,-D,-E)i(-F,-D,-E,-E,-F,-D)i(-A,-B,C,-D,-E,-F)i(-B,C,-A,-E,-F,-D)i(-F,-D,-E,-D,-E,-F)i(-D,-E,-F,-E,-F,-D)i(B,C,-A,E,-F,-D)i(E,-F,-D,E,F,-D)i(C,-A,-B,C,C,-G)i(C,A,-B,E,F,-D)i(E,-F,-D,C,C,-G)i(C,C,-G,E,F,-D)i(B,C,A,G,C,C)i(G,C,C,D,E,F)i(B,C,-A,D,E,-F)i(A,B,C,D,E,F)i(G,C,C,D,E,-F)i(D,E,-F,D,E,F)
#undef i
#undef Ax
#undef Bx
#undef Cx
#undef Dx
#undef Ex
#undef Fx
#undef Gx
    rlEnd();
#undef rlVertex3f
}

void MyDrawCircleFilled3D(Vector3 center, float radius, Color color)
{
    rlCheckRenderBatchLimit(3*12 * 16);

#define rlVertex3f(_x, _y, _z) rlVertex3f( \
  (_x) + center.x, \
  (_y) + center.y, \
  (_z) + center.z  \
)
    rlBegin(RL_TRIANGLES);
        for (int i = 0; i < 360; i += 30)
        {
            rlColor4ub(color.r, color.g, color.b, color.a);
            rlVertex3f(0.0f, 0.0f, 0.0f);
            rlVertex3f(sinf(DEG2RAD*i)*radius, 0.0f, cosf(DEG2RAD*i)*radius);
            rlVertex3f(sinf(DEG2RAD*(i + 30))*radius, 0.0f, cosf(DEG2RAD*(i + 30))*radius);
        }
    rlEnd();
#undef rlVertex3f
}
