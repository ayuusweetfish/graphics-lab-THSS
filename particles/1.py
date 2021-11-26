import taichi as ti
import taichi_glsl as ts
ti.init(arch=ti.vulkan)

import math

dt = 1.0 / 600
R = 0.05
G = 1.5

Ks = 3000
Eta = 60
Kt = 1
Mu = 0.2
KsB = 10000
EtaB = 0.3

Vmax = 3

N = 8888#8
x0 = ti.Vector.field(3, float)
m = ti.field(float)
x = ti.Vector.field(3, float)
v = ti.Vector.field(3, float)
elas = ti.field(float)
radius = ti.field(float)
body = ti.field(int)
ti.root.dense(ti.i, N).place(x0, m, x, v, elas, radius, body)

projIdx = ti.field(int)
projPos = ti.field(float)
ti.root.dense(ti.i, N * 2).place(projPos, projIdx)
rsThreads = int(N**0.5) + 1
rsRadixW = 8
rsRadix = 1 << rsRadixW
rsBlockSum = ti.field(int, (rsThreads,))
rsCount = ti.field(int)
ti.root.dense(ti.i, rsThreads).dense(ti.j, rsRadix).place(rsCount)
rsTempProjIdx = ti.field(int)
rsTempProjPos = ti.field(float)
ti.root.dense(ti.i, N * 2).place(rsTempProjPos, rsTempProjIdx)

M = 1111#1
bodyIdx = ti.Vector.field(2, int)   # (start, end)
bodyPos = ti.Vector.field(3, float)
bodyVel = ti.Vector.field(3, float)
bodyMas = ti.field(float)
bodyAcc = ti.Vector.field(3, float)
bodyOri = ti.Vector.field(4, float)     # Rotation quaternion (x, y, z, w)
bodyIne = ti.Matrix.field(3, 3, float)  # Inverse inertia tensor
bodyAng = ti.Vector.field(3, float)     # Angular momentum
fSum = ti.Vector.field(3, float)  # Accumulated force
tSum = ti.Vector.field(3, float)  # Accumulated torque
ti.root.dense(ti.i, M).place(
  bodyIdx, bodyPos, bodyVel, bodyMas, bodyAcc, bodyOri, bodyIne, bodyAng,
  fSum, tSum,
)

gridSize = R * 10
maxCoord = 1000

debug = ti.field(float, (10,))
ldebug = ti.field(float, (512,))

@ti.kernel
def init():
  for i in range(M):
    x0[i * 8 + 0] = ti.Vector([2.0, 0.0, 0.0]) * R
    x0[i * 8 + 1] = ti.Vector([-2.0, 0.0, 0.0]) * R
    x0[i * 8 + 2] = ti.Vector([1.0, 3.0**0.5, 0.0]) * R
    x0[i * 8 + 3] = ti.Vector([1.0, -3.0**0.5, 0.0]) * R
    x0[i * 8 + 4] = ti.Vector([-1.0, 3.0**0.5, 0.0]) * R
    x0[i * 8 + 5] = ti.Vector([-1.0, -3.0**0.5, 0.0]) * R
    x0[i * 8 + 6] = ti.Vector([4.0, 0.0, 0.0]) * R
    x0[i * 8 + 7] = ti.Vector([-4.0, 0.0, 0.0]) * R
    bodyPos[i] = ti.Vector([
      -0.5 + R * 1.8 * (i % 11),
      R * 6.4 * float(i // 11) + R,
      R * 0.4 * (i % 7),
    ])
    # ** For 2D testing **
    #bodyPos[i] = ti.Vector([
    #  -0.5 + R * 5.4 * (i % 3),
    #  R * 4 * float(i // 3) + R,
    #  0,#R * 3 * (i % 3),
    #])
    bodyIdx[i] = ti.Vector([i * 8, i * 8 + 8])
    bodyVel[i] = ti.Vector([-0.2, ti.random() * 0.5, 0])
    bodyAcc[i] = ti.Vector([0, 0, 0])
    bodyAng[i] = ti.Vector([0, 0, 0])
    bodyOri[i] = ti.Vector([0, 0, 0, 1])
    # Mass and inverse inertia tensor
    bodyMas[i] = 0
    ine = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # Normalize position
    cm = ti.Vector([0.0, 0.0, 0.0])
    for j in range(i * 8, i * 8 + 8):
      body[j] = i
      elas[j] = 1
      m[j] = 5
      radius[j] = R
      bodyMas[i] += m[j]
      cm += m[j] * x0[j]
    cm /= 8
    for j in range(i * 8, i * 8 + 8): x0[j] -= cm
    for j in range(i * 8, i * 8 + 8):
      for p, q in ti.static(ti.ndrange(3, 3)):
        ine[p, q] -= m[j] * x0[j][p] * x0[j][q]
        if p == q:
          ine[p, q] += m[j] * x0[j].norm() ** 2
    bodyIne[i] = ine.inverse()
  for i in range(N): projIdx[i] = i

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

@ti.func
def colliResp(i, j, bodyi, radiusi, xi, vi, Etaelasi):
  bodyj = body[j]
  if bodyi != bodyj:
    xj = x[j]
    r = xi - xj
    dsq = r.x * r.x + r.y * r.y
    dist = radiusi + radius[j]
    if dsq < dist * dist:
      f = ti.Vector([0.0, 0.0, 0.0])
      # Repulsive
      dirUnit = (xi - xj).normalized()
      f += Ks * (dist - dsq ** 0.5) * dirUnit
      # Damping
      relVel = v[j] - vi
      f += Etaelasi * elas[j] * relVel
      # Shear
      f += Kt * (relVel - (relVel.dot(dirUnit) * dirUnit))
      fSum[bodyi] += f
      tSum[bodyi] += (xi - bodyPos[bodyi]).cross(f)
      fSum[bodyj] -= f
      tSum[bodyj] -= (xj - bodyPos[bodyj]).cross(f)

@ti.func
def sortPass(N, sortRound, pos1, idx1, pos2, idx2):
  for t, i in ti.ndrange(rsThreads, rsRadix): rsCount[t, i] = 0
  # Count
  for t in range(rsThreads):
    tA = N * t // rsThreads
    tB = N * (t+1) // rsThreads
    for i in range(tA, tB):
      bucket = (ti.bit_cast(pos1[i], ti.uint32) >> (sortRound * rsRadixW)) % rsRadix
      rsCount[t, bucket] += 1

  # Accumulate counts
  # Each thread processes `rsRadix` elements
  for t in range(rsThreads):
    tA = rsRadix * t
    x, y = tA // rsThreads, tA % rsThreads
    s = 0
    for i in range(rsRadix):
      s += rsCount[y, x]
      y += 1
      if y == rsThreads: x, y = x + 1, 0
    rsBlockSum[t] = s
  for _ in range(1):
    s = 0
    for t in range(rsThreads):
      rsBlockSum[t], s = s, s + rsBlockSum[t]
  for t in range(rsThreads):
    tA = rsRadix * t
    x, y = tA // rsThreads, tA % rsThreads
    s = rsBlockSum[t]
    for i in range(rsRadix):
      rsCount[y, x], s = s, s + rsCount[y, x]
      y += 1
      if y == rsThreads: x, y = x + 1, 0

  # Place
  for t in range(rsThreads):
    tA = N * t // rsThreads
    tB = N * (t+1) // rsThreads
    for i in range(tA, tB):
      bucket = (ti.bit_cast(pos1[i], ti.uint32) >> (sortRound * rsRadixW)) % rsRadix
      pos = rsCount[t, bucket]
      idx2[pos] = idx1[i]
      pos2[pos] = pos1[i]
      rsCount[t, bucket] += 1

@ti.func
def sortProj(N):
  # Flip
  # https://github.com/liufububai/GPU-Sweep-Prune-Collision-Detection/blob/e86fca4a5418884a6694383f4391e23b389d07e8/sapDetection/radixsort.cu#L111
  for i in range(N):
    f = ti.bit_cast(projPos[i], ti.uint32)
    mask = -int(f >> 31) | 0x80000000
    projPos[i] = ti.bit_cast(f ^ mask, ti.float32)

  # Radix sort
  for sortRound in ti.static(range(0, (32 + rsRadixW - 1) // rsRadixW, 2)):
    sortPass(N, sortRound + 0, projPos, projIdx, rsTempProjPos, rsTempProjIdx)
    sortPass(N, sortRound + 1, rsTempProjPos, rsTempProjIdx, projPos, projIdx)

  # Unflip
  for i in range(N):
    f = ti.bit_cast(projPos[i], ti.uint32)
    mask = (int(f >> 31) - 1) | 0x80000000
    projPos[i] = ti.bit_cast(f ^ mask, ti.float32)

@ti.kernel
def stepsort(sz: ti.i32):
  sortProj(sz)

@ti.func
def cantor(x, y):
  return (x + y) * (x + y + 1) / 2 + y
@ti.func
def cantorOmni(x, y):
  base = cantor(abs(x), abs(y)) * 4
  if x < 0: base += 2.0
  if y < 0: base += 1.0
  return base

@ti.kernel
def step():
  # Calculate particle position and velocity
  for i in range(M):
    for j in range(bodyIdx[i][0], bodyIdx[i][1]):
      r = quat_rot(x0[j], bodyOri[i])
      x[j] = bodyPos[i] + r
      v[j] = bodyVel[i] + bodyAng[i].cross(r)

  for b in range(M):
    fSum[b] = ti.Vector([0.0, 0.0, 0.0])
    tSum[b] = ti.Vector([0.0, 0.0, 0.0])

  # Collisions
  # Find axis with PCA
  pcaThreads = 128
  # Zero-centre
  meanPos = ti.Vector([0.0, 0.0, 0.0])
  for t in range(pcaThreads):
    tA = N * t // pcaThreads
    tB = N * (t + 1) // pcaThreads
    tlsMeanPos = ti.Vector([0.0, 0.0, 0.0])
    for i in range(tA, tB): tlsMeanPos += x[i]
    meanPos += tlsMeanPos
  meanPos /= N
  # Covariance matrix
  pcaC = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
  for t in range(pcaThreads):
    tA = N * t // pcaThreads
    tB = N * (t + 1) // pcaThreads
    tlsPcaC = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for i in range(tA, tB):
      x1 = x[i] - meanPos
      for p in ti.static(range(3)):
        for q in ti.static(range(p, 3)):
          tlsPcaC[p, q] += x1[p] * x1[q]
    pcaC += tlsPcaC
  for p in ti.static(range(2)):
    for q in ti.static(range(p + 1, 3)):
      pcaC[q, p] = pcaC[p, q]
  # Dominant eigenvector by power iteration
  eigenvector = ti.Vector([ti.random(), ti.random(), ti.random()])
  for _ in range(1):
    for it in range(10):
      eigenvector = (pcaC @ eigenvector).normalized()

  # Sweep and prune
  axis = eigenvector
  coaxisP = ti.Vector([axis.y, -axis.x, 0]) if axis.z == 0 \
       else ti.Vector([axis.z, 0, -axis.x])
  coaxisP = coaxisP.normalized()
  coaxisQ = axis.cross(coaxisP) # Already unit vector
  extraIdx = N
  for i in range(N):
    xi = x[i]
    ri = radius[i]
    Z = xi.dot(axis)
    P = xi.dot(coaxisP)
    Q = xi.dot(coaxisQ)
    minP = ti.floor((P - ri) / gridSize)
    maxP = ti.ceil((P + ri) / gridSize)
    minQ = ti.floor((Q - ri) / gridSize)
    maxQ = ti.ceil((Q + ri) / gridSize)
    for curP in range(minP, maxP):
      for curQ in range(minQ, maxQ):
        #idx = i if curP == minP and curQ == minQ else ti.atomic_add(extraIdx, 1)
        idx = i
        if curP > minP or curQ > minQ: idx = ti.atomic_add(extraIdx, 1)
        flag = 0 if curP == minP and curQ == minQ else N
        gridId = cantorOmni(curP, curQ)
        projPos[idx] = Z + maxCoord * gridId
        projIdx[idx] = i + flag
  sortProj(extraIdx)
  debug[4] = extraIdx

  # Responses
  for pi in range(extraIdx):
    i = projIdx[pi]
    if i < N:
      upLimit = projPos[pi] + R * 2
      lwLimit = projPos[pi] - R * 2
      bodyi, radiusi, xi, vi = body[i], radius[i], x[i], v[i]
      Etaelasi = Eta * elas[i]
      for pj in range(pi + 1, extraIdx):
        if projPos[pj] > upLimit: break
        j = projIdx[pj]
        if j >= N: j -= N
        colliResp(i, j, bodyi, radiusi, xi, vi, Etaelasi)
      #for pj in range(pi - 1, -1, -1):
      for dpj in range(1, pi + 1):
        pj = pi - dpj
        if projPos[pj] < lwLimit: break
        j = projIdx[pj]
        if j >= N:
          j -= N
          colliResp(i, j, bodyi, radiusi, xi, vi, Etaelasi)

  for b in range(M):
    f = fSum[b]
    t = tSum[b]

    # Gravity
    for i in range(bodyIdx[b][0], bodyIdx[b][1]):
      grav = ti.Vector([0.0, -m[i] * G, 0.0])
      f += grav
      t += (x[i] - bodyPos[b]).cross(grav)

    # Impulse from boundaries
    boundaryY = 0.0
    boundaryYPart = 0
    for i in range(bodyIdx[b][0], bodyIdx[b][1]):
      if (abs(v[i].y) > abs(boundaryY) and
          x[i].y < radius[i] and v[i].y < 0):
        boundaryY = v[i].y
        boundaryYPart = i

    if boundaryY != 0:
      i = boundaryYPart
      impF = ti.Vector([0.0, 0.0, 0.0])
      impF.y = -boundaryY * bodyMas[b] / dt * EtaB
      impF.y += KsB * (radius[i] - x[i].y)
      paraV = v[i].xz.norm()
      fricF = max(-f.y, 0) * Mu
      if paraV >= 1e-5:
        impF.x -= v[i].x / paraV * fricF
        impF.z -= v[i].z / paraV * fricF
      f += impF
      t += (x[i] - bodyPos[b]).cross(impF)

    # Integration
    # Translational: Verlet integration
    newAcc = f / bodyMas[b]
    bodyVel[b] += (bodyAcc[b] + newAcc) * 0.5 * dt
    Vnorm = bodyVel[b].norm()
    if Vnorm >= Vmax: bodyVel[b] *= Vmax / Vnorm
    bodyAcc[b] = newAcc
    bodyPos[b] += bodyVel[b] * dt + bodyAcc[b] * (0.5*dt*dt)
    # Rotational
    bodyAng[b] += t * dt
    rotMat = quat_mat(bodyOri[b])
    angVel = (rotMat @ bodyIne[b] @ rotMat.transpose()).__matmul__(bodyAng[b])
    if bodyAng[b].norm() >= 1e-5:
      theta = bodyAng[b].norm() * dt
      dqw = ti.cos(theta / 2)
      dqv = ti.sin(theta / 2) * bodyAng[b].normalized()
      dq = ti.Vector([dqv.x, dqv.y, dqv.z, dqw])
      bodyOri[b] = quat_mul(dq, bodyOri[b])

boundVertsL = [
  [-100, -0, -100],
  [ 100, -0, -100],
  [ 100, -0,  100],
  [-100, -0,  100],

  [-100,  0, -100],
  [ 100,  0, -100],
  [ 100,  0,  100],
  [-100,  0,  100],
]
boundInds0L = [0, 1, 2, 2, 3, 0]  # bottom
boundVerts = ti.Vector.field(3, float, len(boundVertsL))
boundInds0 = ti.field(int, len(boundInds0L))
import numpy as np
boundVerts.from_numpy(np.array(boundVertsL, dtype=np.float32))
boundInds0.from_numpy(np.array(boundInds0L, dtype=np.int32))

TorusPlSd = 12
TorusOrSd = 12
HemisPlSd = 6
HemisOrSd = 4
PDCBNumVerts = TorusPlSd*TorusOrSd + 2 * (HemisPlSd*(HemisOrSd+1) + 1)
PDCBNumTris = TorusPlSd*TorusOrSd*2 + 2 * (HemisPlSd*(HemisOrSd*2+1))
NumVerts = M * PDCBNumVerts
NumTris = M * PDCBNumTris
particleVerts = ti.Vector.field(3, float, NumVerts)
particleVertStart = ti.field(int, M)
particleVertInds = ti.field(int, NumTris * 3)

pdcbInitPos = ti.Vector.field(3, float, PDCBNumVerts)

@ti.kernel
def buildMesh():
  # Torus
  for p in range(TorusPlSd):
    theta = math.pi*2 / TorusPlSd * p
    sinTheta = ti.sin(theta)
    cosTheta = ti.cos(theta)
    centre = ti.Vector([cosTheta * R*2, sinTheta * R*2, 0])
    radial = ti.Vector([cosTheta * R, sinTheta * R, 0])
    up = ti.Vector([0, 0, R])
    for q in range(TorusOrSd):
      phi = math.pi*2 / TorusOrSd * q
      sinPhi = ti.sin(phi)
      cosPhi = ti.cos(phi)
      pdcbInitPos[p*TorusOrSd + q] = centre + radial * cosPhi + up * sinPhi
  # Handles
  for h in range(2):
    base = TorusPlSd*TorusOrSd + h * (HemisPlSd*(HemisOrSd+1) + 1)
    for p in range(-1, HemisOrSd + 1):
      radius = (1 if p == -1 else ti.cos(math.pi/2 * p/HemisOrSd)) * R
      radial = ti.Vector([0, 0, radius])
      up = ti.Vector([0, radius, 0])
      cx = (2 if p == -1 else 3 + ti.sin(p/HemisOrSd)) * R
      centre = ti.Vector([cx, 0, 0])
      for q in range(HemisPlSd if p < HemisOrSd else 1):
        phi = math.pi*2 / HemisPlSd * q
        sinPhi = ti.sin(phi)
        cosPhi = ti.cos(phi)
        pdcbInitPos[base + (p+1)*HemisPlSd + q] = (
          centre + radial * cosPhi + up * sinPhi
        ) * (h*2 - 1)
  vertCount = 0
  indCount = 0
  for i in range(M):
    vertStart = ti.atomic_add(vertCount, PDCBNumVerts)
    indStart = ti.atomic_add(indCount, PDCBNumTris*3)
    particleVertStart[i] = vertStart
    indIdx = 0
    for p, q in ti.ndrange(TorusPlSd, TorusOrSd):
      particleVertInds[indStart + indIdx + 0] = vertStart + p*TorusOrSd + q
      particleVertInds[indStart + indIdx + 1] = vertStart + p*TorusOrSd + (q+1)%TorusOrSd
      particleVertInds[indStart + indIdx + 2] = vertStart + ((p+1)%TorusPlSd)*TorusOrSd + q
      particleVertInds[indStart + indIdx + 4] = vertStart + ((p+1)%TorusPlSd)*TorusOrSd + q
      particleVertInds[indStart + indIdx + 3] = vertStart + ((p+1)%TorusPlSd)*TorusOrSd + (q+1)%TorusOrSd
      particleVertInds[indStart + indIdx + 5] = vertStart + p*TorusOrSd + (q+1)%TorusOrSd
      indIdx += 6
    for h in range(2):
      base = vertStart + TorusPlSd*TorusOrSd + h * (HemisPlSd*(HemisOrSd+1) + 1)
      for p in range(-1, HemisOrSd):
        for q in range(HemisPlSd):
          nextLevelQ0 = q if p < HemisOrSd - 1 else 0
          nextLevelQ1 = (q+1)%HemisPlSd if p < HemisOrSd - 1 else 0
          particleVertInds[indStart + indIdx + 0] = base + (p+1)*HemisPlSd + q
          particleVertInds[indStart + indIdx + 1] = base + (p+1)*HemisPlSd + (q+1)%HemisPlSd
          particleVertInds[indStart + indIdx + 2] = base + (p+2)*HemisPlSd + nextLevelQ0
          if p < HemisOrSd - 1:
            particleVertInds[indStart + indIdx + 4] = base + (p+2)*HemisPlSd + nextLevelQ0
            particleVertInds[indStart + indIdx + 3] = base + (p+2)*HemisPlSd + nextLevelQ1
            particleVertInds[indStart + indIdx + 5] = base + (p+1)*HemisPlSd + (q+1)%HemisPlSd
            indIdx += 3
          indIdx += 3

@ti.kernel
def updateMesh():
  for i in range(M):
    vertStart = particleVertStart[i]
    for j in range(PDCBNumVerts):
      particleVerts[vertStart + j] = (
        quat_rot(pdcbInitPos[j], bodyOri[i]) + bodyPos[i]
      )

init()
buildMesh()

window = ti.ui.Window('Collision', (1280, 720), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

step()
while window.running:
  for i in range(10): step()
  #for i in range(50): stepsort(N)
  updateMesh()

  #camera.position(10, 1, 20)
  camera.position(4, 5, 6)
  #camera.position(0, 1, 6)
  camera.lookat(0, 0, 0)
  scene.set_camera(camera)

  scene.point_light(pos=(0.5, 1, 2), color=(0.5, 0.5, 0.5))
  scene.ambient_light(color=(0.5, 0.5, 0.5))
  scene.mesh(boundVerts, indices=boundInds0, color=(0.65, 0.65, 0.5), two_sided=True)
  #scene.particles(x, radius=R*2, color=(0.6, 0.7, 1))
  scene.mesh(particleVerts, indices=particleVertInds, color=(0.7, 0.8, 0.7), two_sided=True)
  canvas.scene(scene)
  window.show()
  # print(debug)
  # print(ldebug)
