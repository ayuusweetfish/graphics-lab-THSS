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
static Texture2D CachedScreenshotTexture(const char *prefix, int frame);

Font font, fontlarge, fontxlarge, fontxxlarge;
static void MyDrawText(const char *text, float posX, float posY, int fontSize, Color color)
{
  Font curfont = (
    fontSize == 1 ? font :
    fontSize == 2 ? fontlarge :
    fontSize == 3 ? fontxlarge :
    fontxxlarge);
  int fontSizePts = (
    fontSize == 1 ? 36 :
    fontSize == 2 ? 54 :
    fontSize == 3 ? 90 :
    120) / 2;
  DrawTextEx(curfont, text, (Vector2){posX, posY}, fontSizePts, 0, color);
}
static void MyDrawTextCen(const char *text, float posX, float posY, int fontSize, Color color)
{
  Font curfont = (
    fontSize == 1 ? font :
    fontSize == 2 ? fontlarge :
    fontSize == 3 ? fontxlarge :
    fontxxlarge);
  int fontSizePts = (
    fontSize == 1 ? 36 :
    fontSize == 2 ? 54 :
    fontSize == 3 ? 90 :
    120) / 2;
  Vector2 dims = MeasureTextEx(curfont, text, fontSizePts, 0);
  MyDrawText(text, posX - dims.x / 2, posY - dims.y / 2, fontSize, color);
}

static inline unsigned char try_range(int *x, int dur)
{
  if (*x < dur) return 1;
  *x -= dur;
  return 0;
}

#define norm3d(_v) \
  (sqrtf(((_v)[0])*((_v)[0]) + ((_v)[1])*((_v)[1]) + ((_v)[2])*((_v)[2])))

static void draw_frame(
  int N, int M,
  particle_header *phs, particle *ps,
  Camera camera, Color *bodycolour,
  int stepnum, int forcemask, int tintby, int selparticle
);

// Hack, OpenGL functions without including headers and recompiling raylib
typedef unsigned int GLenum;
typedef unsigned int GLbitfield;
typedef int GLint;
typedef unsigned int GLuint;
#define GL_READ_FRAMEBUFFER 0x8CA8
#define GL_DRAW_FRAMEBUFFER 0x8CA9
#define GL_COLOR_BUFFER_BIT 0x00004000
#define GL_NEAREST 0x2600
void glBindFramebuffer(GLenum target, GLuint framebuffer);
void glBlitFramebuffer(
  GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1,
  GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1,
  GLbitfield mask, GLenum filter);
// End of hack

int main(int argc, char *argv[])
{
  const char *path = "record.bin";
  if (argc > 1) path = argv[1];

  // Find directory of record file
  int pathlen = strlen(path);
  char *screenshotprefix = NULL;
  for (int i = pathlen - 1; i >= 0; i--) {
    if (path[i] == '/') {
      screenshotprefix = (char *)malloc(i + 1);
      memcpy(screenshotprefix, path, i);
      screenshotprefix[i] = '\0';
      break;
    } else if (i == 0) {
      screenshotprefix = strdup(".");
    }
  }

  int globalmode = 0;
  if (argc > 2) globalmode = (int)strtol(argv[2], NULL, 10);
  int globalframe = 0;  // Absolute timestamp; only used when globalmode is 1

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

  const char *fontpathr = "playback_fonts/NotoSansMono-Regular.ttf";
  const char *fontpathl = "playback_fonts/NotoSansMono-Light.ttf";
  font = LoadFontEx(fontpathr, 36, 0, 256);
  fontlarge = LoadFontEx(fontpathr, 54, 0, 256);
  fontxlarge = LoadFontEx(fontpathl, 90, 0, 256);
  fontxxlarge = LoadFontEx(fontpathr, 120, 0, 256);

  Camera3D camera = (Camera3D){
    (Vector3){4, 5, 6},
    (Vector3){0, 0, 0},
    (Vector3){0, 1, 0},
    45,
    CAMERA_PERSPECTIVE
  };

  RenderTexture rentex1 = LoadRenderTexture(W*2, H*2);

  int frame = 0;
  int forcemask = 0;
  int tintby = -1;
  int immerse = (globalmode == 0 ? 0 : 1);

  while (!WindowShouldClose()) {
    int amount = (IsKeyDown(KEY_LEFT_SHIFT) ? 1 : 10);
    if (globalmode == 0) {
      if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_SPACE)) frame = (frame + amount) % nframes;
      if (IsKeyDown(KEY_LEFT)) frame = (frame + nframes - amount) % nframes;
      if (IsKeyPressed(KEY_DOWN)) frame = (frame + amount) % nframes;
      if (IsKeyPressed(KEY_UP)) frame = (frame + nframes - amount) % nframes;
    }
    int framebase = frame * N;

    if (globalmode == 0 && (IsKeyDown(KEY_A) || IsKeyDown(KEY_D))) {
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
    if (globalmode == 0 && (IsKeyDown(KEY_S) || IsKeyDown(KEY_W))) {
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
    if (globalmode == 0 && (IsKeyDown(KEY_Q) || IsKeyDown(KEY_Z))) {
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
    if (globalmode == 0 && (IsKeyDown(KEY_E) || IsKeyDown(KEY_R))) {
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
    if (globalmode == 0 && !immerse) {
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

    // 0 - wireframe
    // 1 - mass
    // 2 - elasticity
    // 3 - screenshot only
    // 4 - large text under small text
    // 5 - large text above small text
    int presentmode = 0;
    // Used when presentmode is 4 or 5
    const char *presentsubtext = NULL;
    const char *presenttext = NULL;

    if (globalmode != 0) {
      int T = globalframe;

      // Animation!
      if (globalmode == 1) {
        if (try_range(&T, 120)) {
          presentmode = 4;
          presentsubtext = "Simulation 1";
          presenttext = "Spheric Particles";
        } else if (try_range(&T, 600)) {
          presentmode = 3;
          frame = T * 10;
        } else if (try_range(&T, 600)) {
          presentmode = 1;
          frame = T * 10;
        } else if (try_range(&T, 600)) {
          presentmode = 2;
          frame = T * 10;
        } else if (try_range(&T, 2)) {
          presentmode = 4;
          presentsubtext = presenttext = "--";
        } else {
          break;
        }
      } else if (globalmode == 2) {
        if (try_range(&T, 120)) {
          presentmode = 4;
          presentsubtext = "Simulation 2";
          presenttext = "Antichlorobenzene";
        } else if (try_range(&T, 600)) {
          presentmode = 3;
          frame = T * 10;
        } else if (try_range(&T, 600)) {
          presentmode = 0;
          tintby = (T < 300 ? 1 : 0);
          frame = T * 10;
        } else if (try_range(&T, 120)) {
          presentmode = 5;
          presenttext = "The End";
          presentsubtext = "Thanks for watching";
        } else if (try_range(&T, 2)) {
          presentmode = 4;
          presentsubtext = presenttext = "--";
        } else {
          break;
        }
      }

      // XXX: For debug use
      //if (!IsKeyDown(KEY_Z)) globalframe += 9;
      //if (IsKeyPressed(KEY_X)) globalframe++;
      globalframe++;
    }

    if (presentmode == 0) {
      BeginDrawing();
      draw_frame(
        N, M,
        phs, ps + framebase,
        camera, bodycolour,
        frame, forcemask, tintby, selparticle
      );
      EndDrawing();
    }

    if (presentmode >= 1 && presentmode <= 2) {
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
        frame, 0, presentmode == 1 ? 0 : 1, selparticle
      );
      rlDrawRenderBatchActive();  // Flush

      // Copy to framebuffer
      glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, rentex1.id);
      glBlitFramebuffer(
        0, 0, W*2, H*2,
        0, 0, W*2, H*2,
        GL_COLOR_BUFFER_BIT, GL_NEAREST);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
      int _mipmaps;
      rlGenTextureMipmaps(rentex1.texture.id,
        W*2, H*2, RL_PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, &_mipmaps);

      ClearBackground((Color){48, 48, 48});
      // Draw texture over filled rectangle to support non-opaque pixels
      Rectangle rentex1dst = (Rectangle){W*0.5, H*0.25, W*0.5, H*0.5};
      DrawRectangleRec(rentex1dst, RAYWHITE);
      DrawTexturePro(
        rentex1.texture,
        (Rectangle){0, H*2, W*2, -H*2},
        rentex1dst,
        (Vector2){0, 0},
        0, WHITE);
      // Screenshot
      DrawTexturePro(
        CachedScreenshotTexture(screenshotprefix, frame),
        (Rectangle){0, H*2, W*2, H*2},
        (Rectangle){W*0.0, H*0.25, W*0.5, H*0.5},
        (Vector2){0, 0},
        0, WHITE);
      // Text
      MyDrawTextCen(
        presentmode == 1 ? "Effect of Mass" : "Effect of Elasticity",
        W/2, H*0.15, 4, WHITE);
      MyDrawTextCen(
        presentmode == 1 ?
          "Particles with large mass push others away" :
          "Particles with high elasticity values bounce higher",
        W/2, H*0.84, 3, WHITE);
      EndDrawing();
    }
    if (presentmode == 3) {
      BeginDrawing();
      DrawTexturePro(
        CachedScreenshotTexture(screenshotprefix, frame),
        (Rectangle){0, H*2, W*2, H*2},
        (Rectangle){0, 0, W, H},
        (Vector2){0, 0},
        0, WHITE);
      EndDrawing();
    }
    if (presentmode == 4 || presentmode == 5) {
      BeginDrawing();
      ClearBackground((Color){48, 48, 48});
      int y1, y2;
      if (presentmode == 4) {
        y1 = H * 0.425;
        y2 = H * 0.525;
      } else {
        y1 = H * 0.55;
        y2 = H * 0.45;
      }
      MyDrawTextCen(presentsubtext, W/2, y1, 3, WHITE);
      MyDrawTextCen(presenttext, W/2, y2, 4, WHITE);
      EndDrawing();
    }

    if (globalmode != 0 || IsKeyPressed(KEY_ENTER)) {
      char s[32];
      snprintf(s, sizeof s, "wf%04d.png", globalmode != 0 ? globalframe : frame);
      // TakeScreenshot(s);
      unsigned char *imgData = rlReadScreenPixels(W*2, H*2);
      Image image = { imgData, W*2, H*2, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 };
      ExportImage(image, s);
      RL_FREE(imgData);
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
  MyDrawText(s, 10, 10, 2, BLACK);

  if (tintby != -1) {
    static const char *tinttypes[3] = {
      "mass", "elasticity", "contact force"
    };
    snprintf(s, sizeof s, "tint by %s", tinttypes[tintby]);
    MyDrawText(s, 10, 40, 1, BLACK);
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
    MyDrawText(s, 10, 60, 1, BLACK);
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
    MyDrawText(s, 10, (ybase += yskip), 1, tint);
    snprintf(s, sizeof s, "pos     (%.4f, %.4f, %.4f)",
      ps[selparticle].pos[0],
      ps[selparticle].pos[1],
      ps[selparticle].pos[2]);
    MyDrawText(s, 10, (ybase += yskip), 1, tint);
  #if RECDEBUG
    snprintf(s, sizeof s, "vel     (%.4f, %.4f, %.4f)",
      ps[selparticle].vel[0],
      ps[selparticle].vel[1],
      ps[selparticle].vel[2]);
    MyDrawText(s, 10, (ybase += yskip), 1, tint);
  #endif
    snprintf(s, sizeof s, "radius  %.4f\n", phs[selparticle].radius);
    MyDrawText(s, 10, (ybase += yskip), 1, tint);
    snprintf(s, sizeof s, "mass    %.4f\n", phs[selparticle].mass);
    MyDrawText(s, 10, (ybase += yskip), 1, tint);
    snprintf(s, sizeof s, "elast   %.4f\n", phs[selparticle].elas);
    MyDrawText(s, 10, (ybase += yskip), 1, tint);
    ybase += 10;
  #if RECDEBUG
    for (int j = 0; j < 5; j++) {
      snprintf(s, sizeof s, "%-10s %.5f", forcenames[j],
        norm3d(ps[selparticle].force[j]));
      MyDrawText(s, 10, (ybase += yskip), 1, (Color){
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
      MyDrawText(s, 10, (ybase += yskip), 1, (Color){64, 64, 64, 255});
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

static Texture2D CachedScreenshotTexture(const char *prefix, int frame)
{
  static char ptr = -1;
  static struct entry {
    int frame;
    Texture2D tex;
  } cache[32];
  const int cache_size = sizeof cache / sizeof cache[0];

  // Initialization
  if (ptr == -1) {
    for (int i = 0; i < cache_size; i++) cache[i].frame = -1;
    ptr = 0;
  }

  // Lookup
  for (int i = 0; i < cache_size; i++)
    if (cache[i].frame == frame)
      return cache[i].tex;

  // Miss. Load and replace
  char s[64];
  snprintf(s, sizeof s, "%s/ti%02d.png", prefix, frame);
  Texture2D tex = LoadTexture(s);

  if (cache[ptr].frame != -1)
    UnloadTexture(cache[ptr].tex);
  cache[ptr].frame = frame;
  cache[ptr].tex = tex;
  ptr = (ptr + 1) % cache_size;

  return tex;
}
