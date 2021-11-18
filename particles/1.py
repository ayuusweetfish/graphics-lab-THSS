import taichi as ti
import taichi_glsl as ts
ti.init(arch=ti.vulkan)

dt = 1.0 / 600
R = 0.05
G = 1.5

Ks = 3000
Eta = 60  # 1
Kt = 1
Mu = 0.2
KsB = 10000

N = 33
x0 = ti.Vector.field(3, float)
m = ti.field(float)
x = ti.Vector.field(3, float)
v = ti.Vector.field(3, float)
ti.root.dense(ti.i, N).place(x0, m, x, v)

M = 11
bodyIdx = ti.Vector.field(2, int)   # (start, end)
bodyPos = ti.Vector.field(3, float)
bodyVel = ti.Vector.field(3, float)
bodyMas = ti.field(float)
bodyAcc = ti.Vector.field(3, float)
bodyOri = ti.Vector.field(4, float)     # Rotation quaternion (x, y, z, w)
bodyIne = ti.Matrix.field(3, 3, float)  # Inverse inertia tensor
bodyAng = ti.Vector.field(3, float)     # Angular momentum
ti.root.dense(ti.i, M).place(
  bodyIdx, bodyPos, bodyVel, bodyMas, bodyAcc, bodyOri, bodyIne, bodyAng
)

count = ti.field(float, ())

@ti.kernel
def init():
  for i in range(M):
    x0[i * 3 + 0] = ti.Vector([R * 0.8, -R * 1.4, 0])
    x0[i * 3 + 1] = ti.Vector([0, 0, 0])
    x0[i * 3 + 2] = ti.Vector([-R * 0.8, R * 1.4, 0])
    bodyPos[i] = ti.Vector([-0.5 + R * 1.8 * i, R * 0.4 * i + R, 0])
    bodyIdx[i] = ti.Vector([i * 3, i * 3 + 3])
    bodyVel[i] = ti.Vector([-0.2, ti.random() * 0.5, 0])
    bodyAcc[i] = ti.Vector([0, 0, 0])
    bodyAng[i] = ti.Vector([0, 0, 0])
    bodyOri[i] = ti.Vector([0, 0, 0, 1])
    # Mass and inverse inertia tensor
    bodyMas[i] = 0
    ine = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for j in range(i * 3, i * 3 + 3):
      m[j] = 5
      bodyMas[i] += m[j]
      for p, q in ti.static(ti.ndrange(3, 3)):
        ine[p, q] -= m[j] * x0[j][p] * x0[j][q]
        if p == q:
          ine[p, q] += m[j] * x0[j].norm() ** 2
    bodyIne[i] = ine.inverse()

@ti.func
def quat_mul(a, b):
  av = a.xyz
  bv = b.xyz
  rv = a.w * bv + b.w * av + av.cross(bv)
  w = a.w * b.w - av.dot(bv)
  return ti.Vector([rv.x, rv.y, rv.z, w])

@ti.func
def quat_rot(v, q):
  return quat_mul(quat_mul(
    q,
    ti.Vector([ v.x,  v.y,  v.z, 0])),
    ti.Vector([-q.x, -q.y, -q.z, q.w])
  ).xyz

@ti.func
def quat_mat(q):
  x, y, z, w = q.x, q.y, q.z, q.w
  return ti.Matrix([
    [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
    [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
    [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
  ])

@ti.kernel
def step():
  # Verlet integration pre-step
  for i in range(M):
    bodyPos[i] += bodyVel[i] * dt + bodyAcc[i] * (0.5*dt*dt)

  # Calculate particle position and velocity
  for i in range(M):
    for j in range(bodyIdx[i][0], bodyIdx[i][1]):
      r = quat_rot(x0[j], bodyOri[i])
      x[j] = bodyPos[i] + r
      v[j] = bodyVel[i] + bodyAng[i].cross(r)

  # Collision
  for b in range(M):
    fSum = ti.Vector([0.0, 0.0, 0.0])
    tSum = ti.Vector([0.0, 0.0, 0.0])
    boundaryX = 0.0
    boundaryXPart = 0
    boundaryY = 0.0
    boundaryYPart = 0
    for i in range(bodyIdx[b][0], bodyIdx[b][1]):
      f = ti.Vector([0.0, 0.0, 0.0])
      for j in range(N):
        if j >= bodyIdx[b][0] and j < bodyIdx[b][1]: continue
        d = (x[i] - x[j]).norm()
        if d < R * 2:
          # Repulsive
          dirUnit = (x[i] - x[j]).normalized()
          f += Ks * (R * 2 - d) * dirUnit
          # Damping
          relVel = v[j] - v[i]
          f += Eta * relVel
          # Shear
          f += Kt * (relVel - (relVel.dot(dirUnit) * dirUnit))
      # Impulse from boundaries
      if (abs(v[i].y) > abs(boundaryY) and
          x[i].y < -0.5 + R and v[i].y < 0):
        boundaryY = v[i].y
        boundaryYPart = i
      if (abs(v[i].x) > abs(boundaryX) and
          (x[i].x < -0.8 + R and v[i].x < 0) or
          (x[i].x >  0.8 - R and v[i].x > 0)):
        boundaryX = v[i].x
        boundaryXPart = i
      f.y -= m[i] * G
      fSum += f
      tSum += (x[i] - bodyPos[b]).cross(f)

    if boundaryX != 0:
      i = boundaryXPart
      impF = ti.Vector([0.0, 0.0, 0.0])
      impF.x = -boundaryX * bodyMas[b] / dt * 1.2 # 2
      fSum += impF
      tSum += (x[i] - bodyPos[b]).cross(impF)
    if boundaryY != 0:
      i = boundaryYPart
      impF = ti.Vector([0.0, 0.0, 0.0])
      impF.y = -boundaryY * bodyMas[b] / dt * 1.2 # 2
      impF.y += KsB * (-0.5 + R - x[i].y)
      paraV = v[i].xz.norm()
      fricF = max(-fSum.y, 0) * Mu
      if paraV >= 1e-5:
        impF.x -= v[i].x / paraV * fricF
        impF.z -= v[i].z / paraV * fricF
      fSum += impF
      tSum += (x[i] - bodyPos[b]).cross(impF)

    # Translational
    # Verlet integration post-step
    newAcc = fSum / bodyMas[b]
    bodyVel[b] += (bodyAcc[b] + newAcc) * 0.5 * dt
    bodyAcc[b] = newAcc
    # Rotational
    bodyAng[b] += tSum * dt
    rotMat = quat_mat(bodyOri[b])
    angVel = (rotMat * bodyIne[b] * rotMat.transpose()).__matmul__(bodyAng[b])
    if bodyAng[b].norm() >= 1e-5:
      theta = bodyAng[b].norm() * dt
      dqw = ti.cos(theta / 2)
      dqv = ti.sin(theta / 2) * bodyAng[b].normalized()
      dq = ti.Vector([dqv.x, dqv.y, dqv.z, dqw])
      bodyOri[b] = quat_mul(dq, bodyOri[b])

boundVertsL = [
  [-0.8, -0.5, -1],
  [ 0.8, -0.5, -1],
  [ 0.8, -0.5,  1],
  [-0.8, -0.5,  1],

  [-0.8,  0.5, -1],
  [ 0.8,  0.5, -1],
  [ 0.8,  0.5,  1],
  [-0.8,  0.5,  1],
]
boundInds0L = [0, 1, 2, 2, 3, 0]  # bottom
boundInds1L = [1, 2, 5, 2, 6, 5, 0, 3, 7, 0, 7, 4]  # right, left
boundVerts = ti.Vector.field(3, float, len(boundVertsL))
boundInds0 = ti.field(int, len(boundInds0L))
boundInds1 = ti.field(int, len(boundInds1L))
import numpy as np
boundVerts.from_numpy(np.array(boundVertsL, dtype=np.float32))
boundInds0.from_numpy(np.array(boundInds0L, dtype=np.int32))
boundInds1.from_numpy(np.array(boundInds1L, dtype=np.int32))

init()

window = ti.ui.Window('Collision', (600, 600), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

while window.running:
  for i in range(10): step()

  camera.position(0, 0, 2)
  camera.lookat(0, 0, 0)
  scene.set_camera(camera)

  scene.point_light(pos=(0.5, 1, 2), color=(0.5, 0.5, 0.5))
  scene.ambient_light(color=(0.5, 0.5, 0.5))
  scene.mesh(boundVerts, indices=boundInds0, color=(0.65, 0.65, 0.5), two_sided=True)
  scene.mesh(boundVerts, indices=boundInds1, color=(0.7, 0.68, 0.6), two_sided=True)
  scene.particles(x, radius=R*2, color=(0.6, 0.7, 1))
  canvas.scene(scene)
  window.show()
