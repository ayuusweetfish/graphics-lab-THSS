import taichi as ti
import taichi_glsl as ts
ti.init(arch=ti.vulkan)

import math

dt = 1.0 / 600  # Time step

R = 0.05  # Maximum particle radius
G = 1.5   # Gravity

Ks = 30000  # Repulsive force coefficient
Eta = 1500  # Damping force coefficient
Kt = 1      # Shearing force coefficient
Mu = 0.2    # Friction coefficient
KsB = 2e5   # Repulsive force coefficient for the floor
EtaB = 2000 # Damping force coefficient for the floor

# Maximum linear velocity and angular velocity
Vmax = 3
Amax = math.pi * 2

# N = number of particles
N = 888#88
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

M = 111#11
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
    # Represent the torus-with-handles shape with eight particles
    scale = 0.6 + 0.4 * (i % 8) / 7
    x0[i * 8 + 0] = ti.Vector([2.0, 0.0, 0.0]) * (R * scale)
    x0[i * 8 + 1] = ti.Vector([-2.0, 0.0, 0.0]) * (R * scale)
    x0[i * 8 + 2] = ti.Vector([1.0, 3.0**0.5, 0.0]) * (R * scale)
    x0[i * 8 + 3] = ti.Vector([1.0, -3.0**0.5, 0.0]) * (R * scale)
    x0[i * 8 + 4] = ti.Vector([-1.0, 3.0**0.5, 0.0]) * (R * scale)
    x0[i * 8 + 5] = ti.Vector([-1.0, -3.0**0.5, 0.0]) * (R * scale)
    x0[i * 8 + 6] = ti.Vector([4.0, 0.0, 0.0]) * (R * scale)
    x0[i * 8 + 7] = ti.Vector([-4.0, 0.0, 0.0]) * (R * scale)
    bodyPos[i] = ti.Vector([
      -0.5 + R * 4.8 * (i % 71 - 35),
      R * 6.4 * float(i // 71 + 1) + R,
      R * 7.5 * (i % 31 - 15 + 0.009 * (i % 97)),
    ])
    bodyIdx[i] = ti.Vector([i * 8, i * 8 + 8])
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
    for j in range(i * 8, i * 8 + 8):
      body[j] = i
      elas[j] = 0.9 - 0.6 * (i % 8) / 7
      m[j] = 5 if i % 200 != 0 else 200
      radius[j] = R * scale
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
def colliResp(i, j, bodyi, radiusi, xi, vi):
  bodyj = body[j]
  if bodyi != bodyj:
    xj = x[j]
    r = xi - xj
    dsq = r.x * r.x + r.y * r.y + r.z * r.z
    dist = radiusi + radius[j]
    if dsq < dist * dist:
      f = ti.Vector([0.0, 0.0, 0.0])
      # Repulsive
      dirUnit = r.normalized()
      f += Ks * (dist - dsq ** 0.5) * dirUnit
      particleF[i, 0] += Ks * (dist - dsq ** 0.5) * dirUnit
      particleF[j, 0] -= Ks * (dist - dsq ** 0.5) * dirUnit
      # Damping
      relVel = v[j] - vi
      f += Eta * relVel
      particleF[i, 1] += Eta * relVel
      particleF[j, 1] -= Eta * relVel
      # Shear
      f += Kt * (relVel - (relVel.dot(dirUnit) * dirUnit))
      particleF[i, 2] += Kt * (relVel - (relVel.dot(dirUnit) * dirUnit))
      particleF[j, 2] -= Kt * (relVel - (relVel.dot(dirUnit) * dirUnit))
      # Accumulate force and torque to respective bodies
      fSum[bodyi] += f
      tSum[bodyi] += (xi - bodyPos[bodyi]).cross(f)
      fSum[bodyj] -= f
      tSum[bodyj] -= (xj - bodyPos[bodyj]).cross(f)
      for k in range(3):
        if particleFContact[i, k] == -1:
          particleFContact[i, k] = j
          break
      for k in range(3):
        if particleFContact[j, k] == -1:
          particleFContact[j, k] = i
          break

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
        # XXX: The following appears to be incorrectly interpreted. Report upstream?
        # idx = i if curP == minP and curQ == minQ else ti.atomic_add(extraIdx, 1)
        idx = i
        if curP > minP or curQ > minQ: idx = ti.atomic_add(extraIdx, 1)
        flag = 0 if curP == minP and curQ == minQ else N
        gridId = cantorOmni(curP, curQ)
        projPos[idx] = Z + maxCoord * gridId
        projIdx[idx] = i + flag
  # Sort
  sortProj(extraIdx)

  # Responses
  for pi in range(extraIdx):
    i = projIdx[pi]
    if i < N:
      upLimit = projPos[pi] + R * 2
      lwLimit = projPos[pi] - R * 2
      bodyi, radiusi, xi, vi = body[i], radius[i], x[i], v[i]
      for pj in range(pi + 1, extraIdx):
        if projPos[pj] > upLimit: break
        j = projIdx[pj]
        if j >= N: j -= N
        colliResp(i, j, bodyi, radiusi, xi, vi)
      # Taichi does not allow `range` with step argument
      # for pj in range(pi - 1, -1, -1):
      for dpj in range(1, pi + 1):
        pj = pi - dpj
        if projPos[pj] < lwLimit: break
        j = projIdx[pj]
        if j >= N:
          j -= N
          colliResp(i, j, bodyi, radiusi, xi, vi)

  for b in range(M):
    f = fSum[b]
    t = tSum[b]

    # Gravity
    for i in range(bodyIdx[b][0], bodyIdx[b][1]):
      grav = ti.Vector([0.0, -m[i] * G, 0.0])
      f += grav
      t += (x[i] - bodyPos[b]).cross(grav)

    # Impulse from the floor
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
        weight = pen / penSum
        impF = ti.Vector([0.0, 0.0, 0.0])
        impF.y += KsB * pen
        impF.y -= v[i].y * EtaB * elas[i]
        particleF[i, 3] += impF
        # Friction
        paraV = v[i].xz.norm()
        fricF = max(-f.y, 0) * Mu * weight
        if paraV >= 1e-5:
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
    # Rotational
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
PDCBNumVerts = TorusPlSd*TorusOrSd + 2 * (HemisPlSd*(HemisOrSd+1) + 1)
PDCBNumTris = TorusPlSd*TorusOrSd*2 + 2 * (HemisPlSd*(HemisOrSd*2+1))
NumVerts = M * PDCBNumVerts
NumTris = M * PDCBNumTris
particleVerts = ti.Vector.field(3, float, NumVerts)
particleVertStart = ti.field(int, M)
particleVertInds = ti.field(int, NumTris * 3)

# Initial positions (relative to local origin) of the shape
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

# Build the mesh for the entire scene
# according to body position and orientation, and the precalculated shape
@ti.kernel
def updateMesh():
  for i in range(M):
    vertStart = particleVertStart[i]
    scale = 0.6 + 0.4 * (i % 8) / 7
    for j in range(PDCBNumVerts):
      particleVerts[vertStart + j] = (
        quat_rot(pdcbInitPos[j] * scale, bodyOri[i]) + bodyPos[i]
      )

init()
buildMesh()

window = ti.ui.Window('Collision', (1280, 720), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

step()

record = True
def npFlatten(field):
  arr = field.to_numpy()
  shape = arr.shape
  count = 1
  for i in range(1, len(shape)): count *= shape[i]
  return np.resize(arr, (shape[0], count))

import os
frameCount = 0
if record:
  recordFile = open(
    os.path.join(
      os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))),
      'record.bin'
    ),
    'wb'
  )
  print(os.path.realpath(recordFile.name))
  # Write records
  combined = np.concatenate((
    npFlatten(radius),
    npFlatten(m),
    npFlatten(elas),
    npFlatten(body)
  ), axis=1, dtype='float32')
  recordFile.write(combined.tobytes())

while window.running:
  frameCount += 1
  if record and frameCount > 400: break

  pullCloseInput[0] = 1 if window.is_pressed(ti.ui.UP) else 0
  pullCloseInput[1] = 1 if window.is_pressed(ti.ui.LEFT) else 0
  pullCloseInput[2] = 1 if window.is_pressed(ti.ui.SPACE) else 0

  for i in range(10):
    step()
    if record:
      # Write file
      combined = np.concatenate((
        npFlatten(particleF),
        npFlatten(x),
        npFlatten(v),
        npFlatten(particleFContact),
      ), axis=1, dtype='float32')
      recordFile.write(combined.tobytes())
  updateMesh()

  camera.position(4, 5, 6)
  camera.lookat(0, 0, 0)
  scene.set_camera(camera)

  # Change floor colour according to input button states
  floorR, floorG, floorB = 0.7, 0.7, 0.7
  if pullCloseInput[0] == 1: floorR += 0.1
  if pullCloseInput[1] == 1: floorG += 0.07
  if pullCloseInput[2] == 1: floorB += 0.15

  scene.point_light(pos=(0.5, 1, 2), color=(0.4, 0.4, 0.4))
  scene.ambient_light(color=(0.6, 0.6, 0.6))
  scene.mesh(boundVerts, indices=boundInds, color=(floorR, floorG, floorB), two_sided=True)
  scene.mesh(particleVerts, indices=particleVertInds, color=(0.85, 0.7, 0.55), two_sided=True)
  canvas.scene(scene)
  window.show()
  # print(debug)

if record: recordFile.close()
