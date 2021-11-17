import taichi as ti
import taichi_glsl as ts
ti.init(arch=ti.vulkan)

dt = 1.0 / 600
R = 0.1
G = 0.4

Ks = 200
Eta = 1
Kt = 1

N = 33
x0 = ti.Vector.field(3, float, (N,))
x = ti.Vector.field(3, float, (N,))
v = ti.Vector.field(3, float, (N,))
f = ti.Vector.field(3, float, (N,))
body = ti.field(int, (N,))

M = 11
bodyIdx = ti.Vector.field(2, int, (M,))   # (start, end)
bodyPos = ti.Vector.field(3, float, (M,))
bodyVel = ti.Vector.field(3, float, (M,))
bodyAng = ti.Vector.field(3, float, (M,))
bodyOri = ti.Vector.field(4, float, (M,)) # quaternion

count = ti.field(int, ())

@ti.kernel
def init():
  for i in range(M):
    x0[i * 3 + 0] = ti.Vector([R * 0.1, -R * 0.9, 0])
    x0[i * 3 + 1] = ti.Vector([0, 0, 0])
    x0[i * 3 + 2] = ti.Vector([-R * 0.1, R * 0.9, 0])
    body[i * 3 + 0] = i
    body[i * 3 + 1] = i
    body[i * 3 + 2] = i
    bodyPos[i] = ti.Vector([-0.5 + R * 0.9 * i, R * 0.2 * i + R, 0])
    bodyIdx[i] = ti.Vector([i * 3, i * 3 + 3])
    bodyVel[i] = ti.Vector([-0.2, ti.random() * 0.5, 0])
    bodyAng[i] = ti.Vector([0, 0, 0])
    bodyOri[i] = ti.Vector([0, 0, 0, 0])

@ti.func
def step():
  # Gravity
  for i in range(M):
    bodyVel[i].y -= G * dt

  # Calculate particle position and velocity
  for i in range(M):
    for j in range(bodyIdx[i][0], bodyIdx[i][1]):
      x[j] = bodyPos[i] + x0[j]
      v[j] = bodyVel[i]

  # Force
  for i in range(M): f[i] = ti.Vector([0, 0, 0])
  # Collision
  for i in range(N):
    f[i] = ti.Vector([0, 0, 0])
    for j in range(N):
      if body[i] == body[j]: continue
      d = (x[i] - x[j]).norm()
      if d < R:
        # Repulsive
        dirUnit = (x[i] - x[j]).normalized()
        f[i] += Ks * (R * 2 - d) * dirUnit
        # Damping
        relVel = v[j] - v[i]
        f[i] += Eta * relVel
        # Shear
        f[i] += Kt * (relVel - (relVel.dot(dirUnit) * dirUnit))
    # Impulse from boundaries
    if x[i].y < -0.5 + R and v[i].y < 0:
      f[i].y += -v[i].y / (0.2 * dt) * 2
    if ((x[i].x < -0.8 + R and v[i].x < 0) or
        (x[i].x >  0.8 - R and v[i].x > 0)):
      f[i].x += -v[i].x / (0.2 * dt) * 2

  # Integrate
  for i in range(M):
    bodyPos[i] += bodyVel[i] * dt
    fSum = ti.Vector([0.0, 0.0, 0.0])
    # for j in range(bodyIdx[i][0], bodyIdx[i][1]):
    # Crash?
    j = bodyIdx[i][0]
    while j < bodyIdx[i][1]:
      fSum += f[j]
      j += 1
    bodyVel[i] += fSum * 0.2 * dt

@ti.kernel
def frame():
  for i in ti.static(range(10)):
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
