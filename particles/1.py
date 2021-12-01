import taichi as ti
import taichi_glsl as ts
ti.init(arch=ti.vulkan)

import math

dt = 1.0 / 600  # Time step

R = 0.05  # Maximum particle radius
G = 1.5   # Gravity

Ks = 30000  # Repulsive force coefficient
Eta = 1500  # Damping force coefficient
Kt = 100    # Shearing force coefficient
Mu = 0.2    # Friction coefficient
KsB = 5000  # Repulsive force coefficient for the floor
EtaB = 50   # Damping force coefficient for the floor

# Maximum linear velocity and angular velocity
Vmax = 3
Amax = math.pi * 2

# N = number of particles
N = 1111*8 + 111*19
x0 = ti.Vector.field(3, float)  # Position relative to body origin
m = ti.field(float)             # Mass
x = ti.Vector.field(3, float)   # World position
v = ti.Vector.field(3, float)   # Velocity
elas = ti.field(float)          # Elasticity
radius = ti.field(float)        # Radius
body = ti.field(int)            # Index of corresponding rigid body
ti.root.dense(ti.i, N).place(x0, m, x, v, elas, radius, body)

# Radix sorting
projIdx = ti.field(int)     # Particle index, see processing logic in step()
projPos = ti.field(float)   # Particle coordinate along sweep line
ti.root.dense(ti.i, N * 2).place(projPos, projIdx)

rsThreads = 2 * int(N**0.5) + 1   # Number of threads during sorting

rsRadixW = 8            # Radix width, in binary bits
rsRadix = 1 << rsRadixW # Radix
rsBlockSum = ti.field(int, (rsThreads,))  # Scratch space, see sorting subroutine
rsCount = ti.field(int)   # Bucket counter for each thread
ti.root.dense(ti.i, rsThreads).dense(ti.j, rsRadix).place(rsCount)
rsTempProjIdx = ti.field(int)   # Double buffering, see sorting subroutine
rsTempProjPos = ti.field(float)
ti.root.dense(ti.i, N * 2).place(rsTempProjPos, rsTempProjIdx)

M = 1111 + 111
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

# Space subdivision parameters
gridSize = R * 10
maxCoord = 1000

# Array of 3 input buttons
pullCloseInput = ti.field(int, (3,))

# Repulsive, damping, shear, floor collision, floor friction
particleF = ti.Vector.field(3, float, (N, 5))
particleFContact = ti.field(float, (N, 3))
# Particle sum, external
bodyF = ti.Vector.field(3, float, (M, 2))
bodyT = ti.Vector.field(3, float, (M,))

# For passing values out during debugging sessions
debug = ti.field(float, (10,))

# Set up rigid bodies
@ti.kernel
def init():
  for i in range(M):
    scale = 0.6 + 0.4 * (i % 8) / 7
    nv = 0      # Number of vertices
    nstart = 0  # Start index of vertices
    if i < 1111:
      # Represent the torus-with-handles shape with eight particles
      nv = 8
      nstart = i * 8
      x0[nstart + 0] = ti.Vector([2.0, 0.0, 0.0]) * (R * scale)
      x0[nstart + 1] = ti.Vector([-2.0, 0.0, 0.0]) * (R * scale)
      x0[nstart + 2] = ti.Vector([1.0, 3.0**0.5, 0.0]) * (R * scale)
      x0[nstart + 3] = ti.Vector([1.0, -3.0**0.5, 0.0]) * (R * scale)
      x0[nstart + 4] = ti.Vector([-1.0, 3.0**0.5, 0.0]) * (R * scale)
      x0[nstart + 5] = ti.Vector([-1.0, -3.0**0.5, 0.0]) * (R * scale)
      x0[nstart + 6] = ti.Vector([4.0, 0.0, 0.0]) * (R * scale)
      x0[nstart + 7] = ti.Vector([-4.0, 0.0, 0.0]) * (R * scale)
    elif i < 1111 + 111:
      # Represent the stick shape with seven particles
      nv = 19
      nstart = 1111 * 8 + (i - 1111) * 19
      for k in ti.static(range(7)):
        x0[nstart + k] = ti.Vector([float(k)*2 - 6.0, 0.0, 0.0]) * (R * scale)
      for k in ti.static(range(7, 10)):
        x0[nstart + k] = ti.Vector([0.0, float(k-7)*2 - 6.0, 0.0]) * (R * scale)
      for k in ti.static(range(10, 13)):
        x0[nstart + k] = ti.Vector([0.0, float(k-6)*2 - 6.0, 0.0]) * (R * scale)
      for k in ti.static(range(13, 16)):
        x0[nstart + k] = ti.Vector([0.0, 0.0, float(k-13)*2 - 6.0]) * (R * scale)
      for k in ti.static(range(16, 19)):
        x0[nstart + k] = ti.Vector([0.0, 0.0, float(k-12)*2 - 6.0]) * (R * scale)
    bodyPos[i] = ti.Vector([
      -0.5 + R * 6.0 * (i % 71 - 35),
      R * 6.4 * float(i // 71 + 1) + R,
      R * 7.5 * (i % 31 - 15 + 0.009 * (i % 97)),
    ])
    bodyIdx[i] = ti.Vector([nstart, nstart + nv])
    rand = ((i * (i % 4 + i * (i // 3) % 17 + 2) + 24) % 97) / 97
    bodyVel[i] = ti.Vector([-0.2, rand * 0.5, 0])
    bodyAcc[i] = ti.Vector([0, 0, 0])
    bodyAng[i] = ti.Vector([0, 0, 0])
    bodyOri[i] = ti.Vector([0, 0, 0, 1])
    # Mass and inverse inertia tensor
    bodyMas[i] = 0
    ine = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # Normalize position
    cm = ti.Vector([0.0, 0.0, 0.0])
    for j in range(nstart, nstart + nv):
      body[j] = i
      elas[j] = 0.9 - 0.6 * (i % 8) / 7
      m[j] = 5 if i % 50 != 0 else 200
      radius[j] = R * scale
      bodyMas[i] += m[j]
      cm += m[j] * x0[j]
    cm /= nv
    for j in range(nstart, nstart + nv): x0[j] -= cm
    # Inertia tensor
    for j in range(nstart, nstart + nv):
      for p, q in ti.static(ti.ndrange(3, 3)):
        ine[p, q] -= m[j] * x0[j][p] * x0[j][q]
        if p == q:
          ine[p, q] += m[j] * x0[j].norm() ** 2
    # Fix anomalies: for axes for which the moment of inertia is zero,
    # remove the corresponding row and column from consideration
    anom = ti.Vector([0, 0, 0])
    for p in ti.static(range(3)):
      if abs(ine[p, p]) <= 1e-6:
        for q in ti.static(range(3)): ine[p, q] = ine[q, p] = 0
        ine[p, p] = 1
        anom[p] = 1
    ine = ine.inverse()
    for p in ti.static(range(3)):
      if anom[p]: ine[p, p] = 0
    bodyIne[i] = ine

# Quaternion multiplication
@ti.func
def quat_mul(a, b):
  av = a.xyz
  bv = b.xyz
  rv = a.w * bv + b.w * av + av.cross(bv)
  w = a.w * b.w - av.dot(bv)
  return ti.Vector([rv.x, rv.y, rv.z, w])

# Rotation application to vector
@ti.func
def quat_rot(v, q):
  return quat_mul(quat_mul(
    q,
    ti.Vector([ v.x,  v.y,  v.z, 0])),
    ti.Vector([-q.x, -q.y, -q.z, q.w])
  ).xyz

# Matrix representation of rotation
@ti.func
def quat_mat(q):
  x, y, z, w = q.x, q.y, q.z, q.w
  return ti.Matrix([
    [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
    [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
    [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
  ])

# Collision response between two particles `i` and `j`
# All other arguments for particle `i` are prefetched for performance's sake,
# as this does not change throughout an inner loop over `j`.
@ti.func
def colliResp(i, j, bodyi, radiusi, xi, vi, bodyMasi):
  bodyj = body[j]
  if bodyi != bodyj:
    xj = x[j]
    r = xi - xj
    dsq = r.x * r.x + r.y * r.y + r.z * r.z
    dist = radiusi + radius[j]
    if dsq < dist * dist:
      factori = bodyMasi*2 / (bodyMasi + bodyMas[bodyj])
      factorj = 2 - factori
      # Repulsive
      dirUnit = r.normalized()
      repul = Ks * (dist - dsq ** 0.5) * dirUnit
      particleF[i, 0] += repul * factori
      particleF[j, 0] -= repul * factorj
      # Damping
      relVel = v[j] - vi
      damp = Eta * relVel
      particleF[i, 1] += damp * factori
      particleF[j, 1] -= damp * factorj
      # Shear
      shear = Kt * (relVel - (relVel.dot(dirUnit) * dirUnit))
      particleF[i, 2] += shear * factori
      particleF[j, 2] -= shear * factorj
      # Accumulate force and torque to respective bodies
      f = repul + damp + shear
      fSum[bodyi] += f * factori
      tSum[bodyi] += (xi - bodyPos[bodyi]).cross(f * factori)
      fSum[bodyj] -= f * factorj
      tSum[bodyj] -= (xj - bodyPos[bodyj]).cross(f * factorj)
      # Record contacting particles
      for k in range(3):
        if ti.atomic_add(particleFContact[i, k], float(j + 1)) == -1: break
        ti.atomic_add(particleFContact[i, k], -float(j + 1))
      for k in range(3):
        if ti.atomic_add(particleFContact[j, k], float(i + 1)) == -1: break
        ti.atomic_add(particleFContact[j, k], -float(i + 1))

# A pass of radix sort
# `N`: number of elements
# `sortRound`: round number, i.e. sorting on the `sortRound`-th digit (base `rsRadix`)
# `pos1`: values to be sorted (as u32)
# `idx1`: tags to be moved around with values
# `pos2`, `idx2`: buffer for sorted sequence
@ti.func
def sortPass(N, sortRound, pos1, idx1, pos2, idx2):
  for t, i in ti.ndrange(rsThreads, rsRadix): rsCount[t, i] = 0
  # Bucket counting
  for t in range(rsThreads):
    tA = N * t // rsThreads
    tB = N * (t+1) // rsThreads
    for i in range(tA, tB):
      bucket = (ti.bit_cast(pos1[i], ti.uint32) >> (sortRound * rsRadixW)) % rsRadix
      rsCount[t, bucket] += 1

  # Prefix sum over `rsCount`, traversing digit-major.
  # Each thread processes `rsRadix` elements as a continuous block.
  # To prevent sequential dependency, calculates sums within each thread first
  # and later distribute over threads.
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

  # Permute all elements according to accumulated starting positions
  for t in range(rsThreads):
    tA = N * t // rsThreads
    tB = N * (t+1) // rsThreads
    for i in range(tA, tB):
      bucket = (ti.bit_cast(pos1[i], ti.uint32) >> (sortRound * rsRadixW)) % rsRadix
      pos = rsCount[t, bucket]
      idx2[pos] = idx1[i]
      pos2[pos] = pos1[i]
      rsCount[t, bucket] += 1

# Sort `projPos`, simultaneously permuting the tag `projIdx`
@ti.func
def sortProj(N):
  # f32 values are processed prior to sorting
  # so that radix sort can be performed directly on bit representation.
  # For positive values, flips sign bit only;
  # for negative values, flips all bits.

  # Flip
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

# Cantor pairing function and unrigourous extension to the entire plane
@ti.func
def cantor(x, y):
  return (x + y) * (x + y + 1) / 2 + y
@ti.func
def cantorOmni(x, y):
  base = cantor(abs(x), abs(y)) * 2
  if x < 0: base += 1.0
  if y < 0: base = -base
  return base

# One step of simulation
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

  for i in range(N):
    for j in ti.static(range(5)): particleF[i, j].fill(0)
    for j in ti.static(range(3)): particleFContact[i, j] = -1

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
  extraIdx = N  # Index to place the next repeated element
  for i in range(N):
    xi = x[i]
    ri = radius[i]
    # Z: coordinate along the sweep line
    # P, Q: coordinate along auxiliary 2D space perpendicular to the sweep line
    Z = xi.dot(axis)
    P = xi.dot(coaxisP)
    Q = xi.dot(coaxisQ)
    # Space subdivision
    minP = ti.floor((P - ri) / gridSize)
    maxP = ti.ceil((P + ri) / gridSize)
    minQ = ti.floor((Q - ri) / gridSize)
    maxQ = ti.ceil((Q + ri) / gridSize)
    for curP in range(minP, maxP):
      for curQ in range(minQ, maxQ):
        # For the first copy, save at index `i`
        # For subsequent copies, save at `extraIdx` and mark as non-starting
        # XXX: The following appears to be incorrectly interpreted. Report upstream?
        # idx = i if curP == minP and curQ == minQ else ti.atomic_add(extraIdx, 1)
        idx = i
        if curP > minP or curQ > minQ: idx = ti.atomic_add(extraIdx, 1)
        flag = 0 if curP == minP and curQ == minQ else N  # Non-starting flag
        gridId = cantorOmni(curP, curQ)
        projPos[idx] = Z + maxCoord * gridId
        projIdx[idx] = i + flag
  # Sort
  sortProj(extraIdx)

  # Responses
  for pi in range(extraIdx):
    i = projIdx[pi]
    # Skip non-starting objects (>= `N`)
    if i < N:
      upLimit = projPos[pi] + R * 2
      lwLimit = projPos[pi] - R * 2
      bodyi, radiusi, xi, vi = body[i], radius[i], x[i], v[i]
      bodyMasi = bodyMas[bodyi]
      # Forward scan
      for pj in range(pi + 1, extraIdx):
        if projPos[pj] > upLimit: break
        j = projIdx[pj]
        if j >= N: j -= N
        colliResp(i, j, bodyi, radiusi, xi, vi, bodyMasi)
      # Backward scan, only handling non-starting objects
      # Taichi does not allow `range` with step argument
      # for pj in range(pi - 1, -1, -1):
      for dpj in range(1, pi + 1):
        pj = pi - dpj
        if projPos[pj] < lwLimit: break
        j = projIdx[pj]
        if j >= N:
          j -= N
          colliResp(i, j, bodyi, radiusi, xi, vi, bodyMasi)

  for b in range(M):
    f = fSum[b]
    t = tSum[b]

    # Gravity
    for i in range(bodyIdx[b][0], bodyIdx[b][1]):
      grav = ti.Vector([0.0, -m[i] * G, 0.0])
      f += grav
      t += (x[i] - bodyPos[b]).cross(grav)

    # Forces from the floor
    # Weight is distributed among those intersecting the floor
    # by distance of penetration
    penSum = 0.0
    for i in range(bodyIdx[b][0], bodyIdx[b][1]):
      pen = radius[i] - x[i].y
      if pen > 0: penSum += pen

    if penSum != 0:
      for i in range(bodyIdx[b][0], bodyIdx[b][1]):
        pen = radius[i] - x[i].y
        if pen <= 0: continue
        # Weights of all affected particles sum to 1
        weight = pen / penSum
        # Repulsive force
        impF = ti.Vector([0.0, 0.0, 0.0])
        impF.y += KsB * pen               # Hooke's law
        impF.y -= v[i].y * EtaB * elas[i] # Damping
        if impF.y < 0: impF.y = 0
        impF.y *= bodyMas[b]  # Scale with body mass (gravitational weight)
        particleF[i, 3] += impF
        # Friction
        paraV = v[i].xz.norm()
        if paraV >= 1e-5:
          fricF = max(-f.y, 0) * Mu * weight
          impF.x -= v[i].x / paraV * fricF
          impF.z -= v[i].z / paraV * fricF
          particleF[i, 4].x -= v[i].x / paraV * fricF
          particleF[i, 4].z -= v[i].z / paraV * fricF
        f += impF
        t += (x[i] - bodyPos[b]).cross(impF)

    # External pulling force
    pullF = ti.Vector([0.0, 0.0, 0.0])
    if pullCloseInput[0] > 0: pullF.z += bodyPos[b].z * -2.0
    if pullCloseInput[1] > 0: pullF.x += bodyPos[b].x * -2.0
    if pullCloseInput[2] > 0:
      pullF += ti.Vector([0.0, 1.0, 0.0]).cross(bodyPos[b]) * 2.0
      pullF += -bodyPos[b] * 1.5
    pullF *= bodyMas[b]
    f += pullF

    bodyF[b, 0] = f - pullF
    bodyF[b, 1] = pullF
    bodyT[b] = t

    # Integration
    # Translational: Verlet integration
    newAcc = f / bodyMas[b]
    bodyVel[b] += (bodyAcc[b] + newAcc) * 0.5 * dt
    Vnorm = bodyVel[b].norm()
    if Vnorm > Vmax: bodyVel[b] *= Vmax / Vnorm
    bodyAcc[b] = newAcc
    bodyPos[b] += bodyVel[b] * dt + bodyAcc[b] * (0.5*dt*dt)
    # Rotational: forward Euler
    bodyAng[b] += t * dt
    rotMat = quat_mat(bodyOri[b])
    angVel = (rotMat @ bodyIne[b] @ rotMat.transpose()) @ bodyAng[b]
    Anorm = angVel.norm()
    if Anorm > Amax:
      bodyAng[b] *= Amax / Anorm
      angVel *= Amax / Anorm
      Anorm = Amax
    if Anorm >= 1e-5:
      theta = Anorm * dt
      dqw = ti.cos(theta / 2)
      dqv = ti.sin(theta / 2) * angVel.normalized()
      dq = ti.Vector([dqv.x, dqv.y, dqv.z, dqw])
      bodyOri[b] = quat_mul(dq, bodyOri[b])

# Vertices for floor
boundVertsL = [
  [-100, -0, -100],
  [ 100, -0, -100],
  [ 100, -0,  100],
  [-100, -0,  100],
]
boundIndsL = [0, 1, 2, 2, 3, 0]
boundVerts = ti.Vector.field(3, float, len(boundVertsL))
boundInds = ti.field(int, len(boundIndsL))
import numpy as np
boundVerts.from_numpy(np.array(boundVertsL, dtype=np.float32))
boundInds.from_numpy(np.array(boundIndsL, dtype=np.int32))

# Builder for the torus-with-handles shape,
# which is abbreviated as PDCB (p-dichlorobenzene)
# For details, plot the vertices in 3D and all should be visualized!

# PlSd = Planar subdivision
# OrSd = Orbital subdivision
TorusPlSd = 12  # Approximate the skeleton with 12-sided polygon
TorusOrSd = 12  # Approximate the section with 12-sided polygon
HemisPlSd = 6
HemisOrSd = 4
PDCBNumVerts = TorusPlSd*TorusOrSd + 6 * (HemisPlSd*(HemisOrSd+1) + 1)
PDCBNumTris = TorusPlSd*TorusOrSd*2 + 6 * (HemisPlSd*(HemisOrSd*2+1))
# Initial positions (relative to local origin) of the shape
pdcbInitPos = ti.Vector.field(3, float, PDCBNumVerts)

# Stick/cylinder shape
StickSd = 12
StickEndOrSd = 3
StickNumVerts = 2 * (StickSd * StickEndOrSd + 1)
StickNumTris = (2*StickSd) * (2*StickEndOrSd)
CrossNumVerts = StickNumVerts * 3
CrossNumTris = StickNumTris * 3
crossInitPos = ti.Vector.field(3, float, CrossNumVerts)

# All vertices and triangles
NumVerts = 1111 * PDCBNumVerts + 111 * CrossNumTris
NumTris = 1111 * PDCBNumTris + 111 * CrossNumTris
particleVerts = ti.Vector.field(3, float, NumVerts)
particleVertStart = ti.field(int, M)
particleVertInds = ti.field(int, NumTris * 3)
particleColours = ti.Vector.field(3, float, NumVerts)

# Linear congruent PRNG at step `n`
@ti.func
def LCG(n):
  seed = 20211128
  a = 1103515245
  b = 12345
  while n > 0:
    if n % 2 == 1: seed = seed * a + b
    a, b = a*a, (a+1)*b
    n >>= 1
  return seed & 0x7fffffff

@ti.kernel
def buildMesh():
  # PDCB
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
  for h in range(6):
    base = TorusPlSd*TorusOrSd + h * (HemisPlSd*(HemisOrSd+1) + 1)
    sinTheta = ti.sin(math.pi*2 * h/6)
    cosTheta = ti.cos(math.pi*2 * h/6)
    for p in range(-1, HemisOrSd + 1):
      radius = (1 if p == -1 else ti.cos(math.pi/2 * p/HemisOrSd)) * R
      cx = (2 if p == -1 else 4 + ti.sin(math.pi/2 * p/HemisOrSd)) * R
      # Shrink handles at positions 1, 2, 4, 5
      if h != 0 and h != 3:
        radius *= 0.8
        if p == -1: cx = R * 2.1
        else: cx = R * 2.1 + (cx - R*3) * 0.8
      radial = ti.Vector([0, 0, radius])
      # up = ti.Vector([0, radius, 0])
      # centre = ti.Vector([cx, 0, 0])
      # Rotate `up` and `centre` vectors around Z axis
      # `radial` is parallel to the Z axis and does not need to be processed
      up = ti.Vector([sinTheta*radius, cosTheta*radius, 0])
      centre = ti.Vector([cosTheta*cx, -sinTheta*cx, 0])
      for q in range(HemisPlSd if p < HemisOrSd else 1):
        phi = math.pi*2 / HemisPlSd * q
        sinPhi = ti.sin(phi)
        cosPhi = ti.cos(phi)
        pdcbInitPos[base + (p+1)*HemisPlSd + q] = (
          centre + radial * cosPhi + up * sinPhi
        )
  # Mesh in local coordinates
  vertCount = 0
  indCount = 0
  for i in range(1111):
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
    for h in range(6):
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

  # Stick
  # StickNumVerts = 2 * (StickSd * StickEndOrSd + 1)
  for s in range(StickEndOrSd * 2):
    # A ring perpendicular to the X axis
    s1 = s
    flip = -1
    if s >= StickEndOrSd:
      s1 = StickEndOrSd * 2 - 1 - s
      flip = 1
    x = (6 + ti.cos(math.pi/2 * (s1+1)/StickEndOrSd)) * flip
    radius = ti.sin(math.pi/2 * (s1+1)/StickEndOrSd)
    for t in range(StickSd):
      theta = math.pi*2 / StickSd * t
      z = ti.cos(theta) * radius
      y = ti.sin(theta) * radius
      crossInitPos[s*StickSd + t] = ti.Vector([x, y, z]) * R
  # Extreme points
  crossInitPos[StickSd*StickEndOrSd*2 + 0] = ti.Vector([-R*7, 0.0, 0.0])
  crossInitPos[StickSd*StickEndOrSd*2 + 1] = ti.Vector([ R*7, 0.0, 0.0])
  # Copy a stick two times
  for j in range(StickNumVerts):
    crossInitPos[StickNumVerts + j] = ti.Vector([
      crossInitPos[j].y,
      -crossInitPos[j].x,
      crossInitPos[j].z,
    ])
    crossInitPos[StickNumVerts*2 + j] = ti.Vector([
      crossInitPos[j].z,
      crossInitPos[j].y,
      -crossInitPos[j].x,
    ])
  for i in range(1111, 1111 + 111):
    vertStart = ti.atomic_add(vertCount, CrossNumVerts)
    indStart = ti.atomic_add(indCount, CrossNumTris*3)
    particleVertStart[i] = vertStart
    # StickNumTris = (2*StickSd) * (2*StickEndOrSd)
    # Strips
    for s in range(2*StickEndOrSd - 1):
      for t in range(StickSd):
        particleVertInds[indStart + 0] = vertStart + s*StickSd + t
        particleVertInds[indStart + 1] = vertStart + s*StickSd + (t+1)%StickSd
        particleVertInds[indStart + 2] = vertStart + (s+1)*StickSd + t
        particleVertInds[indStart + 3] = vertStart + (s+1)*StickSd + t
        particleVertInds[indStart + 4] = vertStart + s*StickSd + (t+1)%StickSd
        particleVertInds[indStart + 5] = vertStart + (s+1)*StickSd + (t+1)%StickSd
        indStart += 6
    # Ends
    for t in range(StickSd):
      # Negative X
      particleVertInds[indStart + 1] = vertStart + t
      particleVertInds[indStart + 0] = vertStart + (t+1)%StickSd
      particleVertInds[indStart + 2] = vertStart + StickSd*StickEndOrSd*2 + 0
      # Positive X
      particleVertInds[indStart + 3] = vertStart + (2*StickEndOrSd - 1)*StickSd + t
      particleVertInds[indStart + 4] = vertStart + (2*StickEndOrSd - 1)*StickSd + (t+1)%StickSd
      particleVertInds[indStart + 5] = vertStart + StickSd*StickEndOrSd*2 + 1
      indStart += 6
    # Copy a stick
    for j in range(StickNumTris * 3):
      particleVertInds[indStart + j] = (
        particleVertInds[indStart + j - StickNumTris * 3] +
        StickNumVerts
      )
      particleVertInds[indStart + StickNumTris*3 + j] = (
        particleVertInds[indStart + j - StickNumTris * 3] +
        StickNumVerts*2
      )

  # Colours
  for i in range(M):
    vertStart = particleVertStart[i]
    vertCount = PDCBNumVerts if i < 1111 else CrossNumVerts
    basetint = ti.Vector([0.95, 0.8, 0.65]) if i < 1111 else ti.Vector([0.9, 0.65, 0.6])
    # Random colours
    r = (LCG(i*3+1) >> 17) % 64
    g = (LCG(i*3+2) >> 18) % 64
    b = (LCG(i*3+3) >> 19) % 64
    base = 160 + (64 - (r * 2 + g * 5 + b) // 8) // 2
    tint = ti.Vector([
      (base + r) / 255.0,
      (base + g) / 255.0,
      (base + b) / 255.0,
    ])
    tint = basetint * 0.7 + tint * 0.3
    for j in range(vertCount):
      particleColours[vertStart + j] = tint

# Build the mesh for the entire scene
# according to body position and orientation, and the precalculated shape
@ti.kernel
def updateMesh():
  for i in range(1111):
    vertStart = particleVertStart[i]
    scale = 0.6 + 0.4 * (i % 8) / 7
    for j in range(PDCBNumVerts):
      particleVerts[vertStart + j] = (
        quat_rot(pdcbInitPos[j] * scale, bodyOri[i]) + bodyPos[i]
      )
  for i in range(1111, 1111 + 111):
    vertStart = particleVertStart[i]
    scale = 0.6 + 0.4 * (i % 8) / 7
    for j in range(CrossNumVerts):
      particleVerts[vertStart + j] = (
        quat_rot(crossInitPos[j] * scale, bodyOri[i]) + bodyPos[i]
      )

init()
buildMesh()

# Display
window = ti.ui.Window('Collision', (1280, 720), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

step()

# For recording
import os
record = False
recordFrames = 0
recordFile = None
try:
  f = os.environ['REC'].split(',')
  recordFrames = int(f[0])
  recordFile = f[1] if len(f) >= 2 else 'record.bin'
  record = True
except:
  pass
recordScreenshot = (os.environ.get('SCR') == '1')
# Flattens a Taichi field into a NumPy array of (N, ?)
def npFlatten(field):
  arr = field.to_numpy()
  shape = arr.shape
  count = 1
  for i in range(1, len(shape)): count *= shape[i]
  return np.resize(arr, (shape[0], count))

frameCount = 0
if record:
  recordFile = open(
    os.path.join(
      os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))),
      recordFile
    ),
    'wb'
  )
  print('recording to ' + os.path.realpath(recordFile.name))
  # Write recording header
  recordFile.write(np.array([N, M], dtype='int32').tobytes())
  recordFile.write(np.concatenate((
    npFlatten(radius),
    npFlatten(m),
    npFlatten(elas),
    npFlatten(body)
  ), axis=1, dtype='float32').tobytes())

# Show the window, saving a screenshot if requested
def render(skipScrenshot=False):
  updateMesh()

  camera.position(4, 5, 6)
  camera.lookat(0, 0, 0)
  scene.set_camera(camera)

  # Change floor colour according to input button states
  floorR, floorG, floorB = 0.8, 0.8, 0.8
  if pullCloseInput[0] == 1: floorR += 0.1
  if pullCloseInput[1] == 1: floorG += 0.07
  if pullCloseInput[2] == 1: floorB += 0.15

  scene.point_light(pos=(0, 4, 6), color=(0.4, 0.4, 0.4))
  scene.ambient_light(color=(0.7, 0.7, 0.7))
  scene.mesh(boundVerts, indices=boundInds, color=(floorR, floorG, floorB), two_sided=True)
  scene.mesh(particleVerts, indices=particleVertInds, per_vertex_color=particleColours, two_sided=True)
  canvas.scene(scene)
  if recordScreenshot and not skipScrenshot:
    fileName = 'ti%02d.png' % frameCount
    window.write_image(
      os.path.join(
        os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))),
        fileName
      )
    )
  window.show()

# Workaround for Taichi 0.8.5 rendering lighting incorrectly on the first frame
render(skipScrenshot=True)

while window.running:
  render()
  if record and frameCount >= recordFrames: break

  pullCloseInput[0] = 1 if window.is_pressed(ti.ui.UP) else 0
  pullCloseInput[1] = 1 if window.is_pressed(ti.ui.LEFT) else 0
  pullCloseInput[2] = 1 if window.is_pressed(ti.ui.SPACE) else 0

  if frameCount >= 7500: break
  if frameCount in range(3000, 3300): pullCloseInput[2] = 1
  if frameCount in range(4500, 4700): pullCloseInput[1] = 1
  if frameCount in range(4600, 4800): pullCloseInput[0] = 1

  for i in range(10):
    if record:
      # Dump relevant data of the current step
      recordFile.write(np.concatenate((
        npFlatten(x),
        npFlatten(v),
        npFlatten(particleF),
        npFlatten(particleFContact),
      ), axis=1, dtype='float32').tobytes())
    frameCount += 1
    step()

if record: recordFile.close()
