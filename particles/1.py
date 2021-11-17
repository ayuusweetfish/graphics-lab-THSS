import taichi as ti
import taichi_glsl as ts
ti.init(arch=ti.vulkan)

dt = 1.0 / 600
R = 0.1
G = 0.4

K = 100

N = 11
x = ti.Vector.field(3, float, (N,))
v = ti.Vector.field(3, float, (N,))
f = ti.Vector.field(3, float, (N,))

@ti.kernel
def init():
  for i in range(N):
    x[i] = ti.Vector([-0.5 + R * 0.9 * i, R * 0.2 * i, 0])
    v[i] = ti.Vector([0, ti.random() * 0.5, 0])

@ti.func
def step():
  # Gravity
  for i in range(N):
    v[i].y -= G * dt
  # Collision
  for i in range(N):
    f[i] = ti.Vector([0, 0, 0])
    for j in range(N):
      if i == j: continue
      d = (x[i] - x[j]).norm()
      if d < R:
        # Repulsive
        f[i] += K * (R * 2 - d) * (x[i] - x[j]).normalized()
  for i in range(N):
    v[i] += f[i] * 0.2 * dt
  # Integrate
  for i in range(N):
    x[i] += v[i] * dt
    if x[i].y < -0.5 + R:
      v[i].y *= -1
      x[i].y = -0.5 + R
    if x[i].x < -0.8 + R or x[i].x > 0.8 - R:
      v[i].x *= -1
      x[i].x = ts.clamp(x[i].x, -0.8 + R, 0.8 - R)
    if x[i].z < -0.8 + R or x[i].z > 0.8 - R:
      v[i].z *= -1
      x[i].z = ts.clamp(x[i].z, -0.8 + R, 0.8 - R)

@ti.kernel
def frame():
  for i in range(10):
    step()

init()

window = ti.ui.Window('Collision', (600, 600), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

while window.running:
  frame()

  camera.position(0, 0, 2)
  camera.lookat(0, 0, 0)
  scene.set_camera(camera)

  scene.point_light(pos=(0.5, 1, 2), color=(0.5, 0.5, 0.5))
  scene.ambient_light(color=(0.5, 0.5, 0.5))
  scene.particles(x, radius=R, color=(0.6, 0.7, 1))
  canvas.scene(scene)
  window.show()
