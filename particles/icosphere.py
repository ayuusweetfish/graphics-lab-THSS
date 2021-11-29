import taichi as ti

# Icosphere calculation identical to 0.py
t = (1 + 5**0.5) / 2
icosphereVerts = [
  ti.Vector([-1,  t,  0]).normalized(),
  ti.Vector([ 1,  t,  0]).normalized(),
  ti.Vector([-1, -t,  0]).normalized(),
  ti.Vector([ 1, -t,  0]).normalized(),
  ti.Vector([ 0, -1,  t]).normalized(),
  ti.Vector([ 0,  1,  t]).normalized(),
  ti.Vector([ 0, -1, -t]).normalized(),
  ti.Vector([ 0,  1, -t]).normalized(),
  ti.Vector([ t,  0, -1]).normalized(),
  ti.Vector([ t,  0,  1]).normalized(),
  ti.Vector([-t,  0, -1]).normalized(),
  ti.Vector([-t,  0,  1]).normalized(),
]
icosphereTris = [
  [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
]
# Subdivide
for subdivision in range(1):
  newTris = []
  for [a, b, c] in icosphereTris:
    icosphereVerts += [
      ((icosphereVerts[a] + icosphereVerts[b]) / 2).normalized(),
      ((icosphereVerts[b] + icosphereVerts[c]) / 2).normalized(),
      ((icosphereVerts[c] + icosphereVerts[a]) / 2).normalized(),
    ]
    n = len(icosphereVerts)
    d, e, f = n-3, n-2, n-1
    newTris += [[a, d, f], [b, e, d], [c, f, e], [d, e, f]]
  icosphereTris = newTris

# Output
mapping = {}
count = 0
def add(x):
  global count
  if mapping.get(x) == None:
    mapping[abs(x)] = chr(65 + count)
    if x != 0: mapping[-abs(x)] = '-' + chr(65 + count)
    count += 1
for p in icosphereVerts:
  add(p.x)
  add(p.y)
  add(p.z)

for k, v in mapping.items():
  if k >= 0: print('#define %sx %.9f' % (v, k))
print('#define i(a, b, c, d, e, f) ' +
  'rlVertex3f(a##x, b##x, c##x); rlVertex3f(d##x, e##x, f##x);')

def output(a, b):
  print('i(%s,%s,%s,%s,%s,%s)' %
    (mapping[a[0]], mapping[a[1]], mapping[a[2]],
     mapping[b[0]], mapping[b[1]], mapping[b[2]]),
    end='')
  # print('rlVertex3f(%2.8f, %2.8f, %2.8f);' % tuple(a))
  # print('rlVertex3f(%2.8f, %2.8f, %2.8f);' % tuple(b))
for [a, b, c] in icosphereTris:
  va = icosphereVerts[a]
  vb = icosphereVerts[b]
  vc = icosphereVerts[c]
  if a < b: output(va, vb)
  if b < c: output(vb, vc)
  if c < a: output(vc, va)

print('\n#undef i')
for k, v in mapping.items():
  if k >= 0: print('#undef %sx' % v)
