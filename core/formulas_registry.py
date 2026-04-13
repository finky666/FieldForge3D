# core/formulas_registry.py
# Auto-generated from plugin FORMULA strings.
# You may still edit entries manually if you want custom overrides.

FORMULAS = {
    "fibo_nested_cubes": r"""\
Fibonacci Nested Cubes (wireframe / edges)
Plugin: fibo_nested_cubes  (fibo_nested_cubes.py)

Idea:
We place multiple nested wireframe cubes inside [-1..1]^3.
Cube scale grows by the golden ratio φ, so cube size shrinks roughly as ~ 1 / φ^lvl.
A small rotation per level creates the swirling / woven look.

Golden ratio:
- φ = (1 + sqrt(5)) / 2
- scale_0 = 1
- scale_(lvl+1) = scale_lvl * φ
- halfsize(hs) = base_hs / scale_lvl

Rotation per level (Euler-like):
(x, y, z) -> rot3(x, y, z, axk, ayk, azk)
axk = ROTX + TWIST * 0.73 * lvl
ayk = ROTY + TWIST * 0.57 * lvl
azk = ROTZ + TWIST * 1.00 * lvl

Wire thickness:
- th = THICK * hs
- th >= 1.2 * dx    (dx = 2 / (N-1)  -> minimum "voxel" thickness)
- th <= 0.30 * hs   (avoid turning the cube into a solid volume)
""",

    "galaxy_swirl": r"""\
Galaxy Swirl (scalar density)
Plugin: galaxy_swirl  (galaxy_swirl.py)

Idea:
A simple procedural "galaxy" density field: a bright core + spiral arms, with a thickness
falloff along the Z axis.

Formulas:
- r = sqrt(x^2 + y^2),  θ = atan2(y, x)
- core density:    d_core = exp( -(r^2 + z^2) / core^2 )
- arms modulation: arm    = 0.5 + 0.5 * cos( arms*θ + twist*(3r) )
- z thickness:     d_z    = exp( -(z^2) / thick^2 )
- val = d_core + arm * d_z * exp(-2r)
- output: v = 1 - exp( -gain * val )

Notes:
- gain increases contrast (exponential compression)
""",

    "golden_lattice": r"""\
Golden lattice (φ-mix)
Plugin: golden_lattice  (golden_lattice.py)

Idea:
Two periodic wave-fields are phase-shifted by the golden ratio φ and blended into one
interference lattice. The result looks like a "woven" crystal with φ-flavored symmetry.

Formulas:
- φ = (1 + sqrt(5)) / 2
- a = sin(φx + y) * cos(φy + z)
- b = sin(φz + x) * cos(φx + z)
- g = mix * a + (1 - mix) * b
- v = 0.5 + 0.5 * g   (optionally clamped to [0..1])

Notes:
- freq scales space: (x, y, z) := freq * (x, y, z)
- mix blends the two φ-shifted terms into a single interference field
- clamp keeps values inside [0..1] for stable isosurfaces
""",

    "gyroid": r"""\
Gyroid (TPMS)
Plugin: gyroid  (gyroid.py)

Idea:
The classic triply-periodic minimal surface (TPMS) "gyroid". The isosurface of F(x,y,z)=0
creates a continuous labyrinth-like sheet.

Formula:
- F(x,y,z) = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)
- optional "abs/sym" modes may fold / symmetrize the field
- output mapping typically uses v = clamp(0.5 + 0.5 * F)

Notes:
- freq scales space: (x, y, z) := freq * (x, y, z)
- ISO selects the shell thickness around F=0
""",

    "gyroid_twist_tunnel": r"""\
Gyroid + Twist Tunnel (TPMS style)
Plugin: gyroid_twist_tunnel  (gyroid_twist_tunnel.py)

Gyroid (implicit surface)
  g = sin(fx)cos(fy) + sin(fy)cos(fz) + sin(fz)cos(fx)
  where fx = FREQ*x, fy = FREQ*y, fz = FREQ*z

Twist around Z
- Before evaluating g, the XY plane is rotated depending on z:
  (x,y) := rotZ(x, y, TWIST * z / r_norm)

Tunnel bias
- We add a radial term to carve a tunnel-like feature:
  r = sqrt(x^2 + y^2) / r_norm
  tun = (r - TUN_R)
  F = g + TUN_GAIN * tun

Shell thickness
- We convert the implicit F into a "band" around F=0:
  t = smoothstep( clamp(1 - |F|/THICK) )

Output mapping
  value = 1 - exp(-GAIN * t)

Tips
- Increase THICK for a wider band.
- Increase TUN_GAIN to make the tunnel dominate the shape.
""",

    "heart_implicit": r"""\
Heart (implicit 3D)
Classic implicit heart surface:
  F(x,y,z) = (x^2 + 9/4 y^2 + z^2 - 1)^3 - x^2 z^3 - 9/80 y^2 z^3

We map F to [0..1] around the surface with thickness/softness.
""",

    "hopf_fibration_rings": r"""\
Hopf Fibration Rings (field)
Plugin: hopf_fibration_rings  (hopf_fibration_rings.py)

Idea
- A stylized 3D projection inspired by the Hopf fibration:
  linked circles / tori-like rings that "thread" through each other.

Implementation sketch
- We build a field from distances to a set of rings.
- Each ring contributes a smooth radial profile.
- Multiple rings are combined to form a linked structure.

Output mapping
  value = 1 - exp(-GAIN * field)

Tips
- Increase RINGS for more links.
- If it becomes too dense: reduce THICK or reduce GAIN.
""",

    "klein_bottle_field": r"""\
Klein bottle (implicit, shell)
Plugin: klein_bottle_field  (klein_bottle_field.py)

Formulas:
  - r² = x² + y² + z²
  - F(x,y,z) = (r² + 2y - 1)² · (r² - 2y - 1) - 8z²·(r² + 2y - 1)

  - shell around F = 0:
      t = smoothstep( clamp(1 - |F| / THICK) )
      v = 1 - exp(-GAIN · t)

Notes:
""",

    "lorenz": r"""\
Lorenz Attractor (density)
Plugin: lorenz_density  (lorenz_density.py)

Idea:
Simulate the Lorenz system and accumulate a 3D density histogram. This produces a
volumetric "butterfly" attractor that can be rendered as an isosurface.

ODE:
- dx/dt = σ (y - x)
- dy/dt = x (ρ - z) - y
- dz/dt = x y - β z

Density:
- positions are normalized into [-1..1]^3, then binned into a voxel grid
- hist[ix,iy,iz] += 1
- optional smoothing (few passes)
- normalize and compress: v = 1 - exp(-GAIN * hist_norm)

Notes:
- STEPS controls quality vs. time
- DT controls numerical stability (keep it small)
""",

    "mandelbox_like": r"""\
Mandelbox-like Fractal (distance-ish field)
Plugin: mandelbox_like  (mandelbox_like.py)

Idea
- A "Mandelbox-inspired" iterative fold + scale process in 3D.
- Not a strict distance estimator, but produces characteristic boxy fractal structures.

Typical steps
- Box fold: reflect coordinates into a cube.
- Sphere fold: clamp radius into a range.
- Scale + translate: p = p * SCALE + offset

Field
- The iteration produces a value that correlates with "escape / detail".
- Mapped through an exponential curve for a usable [0..1] scalar field.

Tips
- Higher ITER increases detail but costs time.
- Adjust SCALE/SHIFT to explore different shapes.
""",

    "mandelbulb": r"""\
Mandelbulb (3D fractal, distance-ish)
Plugin: mandelbulb  (mandelbulb.py)

Concept
- 3D analogue of the Mandelbrot set using spherical coordinates.
- Iteration: z_{n+1} = z_n^POWER + c

Spherical transform (sketch)
  r = ||z||
  θ = acos(z_z / r)
  φ = atan2(z_y, z_x)

  r' = r^POWER
  θ' = θ * POWER
  φ' = φ * POWER

  z' = r' * (sinθ'cosφ', sinθ'sinφ', cosθ') + c

Field output
- Uses an escape-based estimate mapped to [0..1].
- Higher POWER changes the symmetry and "spikiness".

Tips
- Increase MAX_ITER for more detail (slower).
- If it looks empty: lower ISO or reduce BOUNDS.
""",

    "mandelbulb_de": r"""\
Mandelbulb (distance estimator)
Plugin: mandelbulb_de  (mandelbulb_de.py)

Idea:
A 3D Mandelbulb rendered via a distance estimator (DE). Instead of a simple escape mask,
we estimate distance to the fractal surface and convert it to a smooth density shell.

Core:
- iterate z in spherical coordinates with power P:
  (r, θ, φ) -> (r^P, θ*P, φ*P)
- track derivative dr to compute DE

Distance estimator:
- DE ≈ 0.5 * log(r) * r / dr
- we create a shell around DE=ISO:
  v = exp( - (DE-ISO)^2 / (2*sigma^2) )   (conceptually)

Notes:
- ISO sets the shell radius; adjust with BOUNDS and ISO together
- MAX_ITER and POWER change the shape and detail
""",

    "menger_sponge": r"""\
Menger Sponge (nested cubes)
Plugin: menger_sponge  (menger_sponge.py)

Idea:
A 3D fractal built by repeatedly removing the center cross from a cube (Menger sponge).
We evaluate membership by iterating coordinates in base-3 style (or equivalent rules).

Rule of thumb:
At each level, if two or more coordinates fall into the middle third, the point is removed.

Output:
- v in [0..1] represents membership / proximity
- use GAIN to increase contrast when needed

Notes:
- LEVELS controls detail (and compute cost)
""",

    "metaballs": r"""\
Metaballs (Σ 1/r²)
Plugin: metaballs  (metaballs.py)

Idea:
A smooth implicit surface created by summing radial fields from several centers.

Formula:
- s(p) = Σ_i w_i / (||p - c_i||^2 + ε)
- output: v = 1 - exp(-STRENGTH * s)

Notes:
- BALLS controls number of centers
- STRENGTH increases contrast (stronger "blobs")
""",

    "phyllotaxis": r"""\
Phyllotaxis (Fibonacci spiral field)
Plugin: phyllotaxis_fibo  (phyllotaxis_fibo.py)

Idea
- A sunflower-like distribution using the golden angle.
- Points are placed on a spiral with approximately uniform packing.

Golden angle
  γ = 2π * (1 - 1/φ)  ≈ 137.507°
  θ_k = k * γ
  r_k = sqrt(k / K) * R

Field
- Each point contributes a smooth radial kernel.
- Combined points form shells / seeds / spiral patterns.

Tips
- Increase COUNT for denser packing.
- Increase GAIN for higher contrast.
""",

    "phyllotaxis_shell": r"""\
Phyllotaxis Shell (sunflower)
Plugin: phyllotaxis_shell  (phyllotaxis_shell.py)

Idea:
A phyllotaxis pattern wrapped into a 3D shell: points placed using the golden angle
create a natural spiral packing (sunflower / pinecone style).

Core:
- golden angle: ga = π * (3 - sqrt(5))
- for n = 0..N:
    θ = n * ga
    r = sqrt(n / N)
    point = (r*cosθ, r*sinθ, z(r))

Output:
A smooth density / field around the point set, suitable for isosurface rendering.

Notes:
- adjust ISO to control shell thickness
- higher N gives smoother results but costs more
""",

    "quat_julia": r"""\
Quaternion Julia (4D Julia slice, projected to 3D)
Plugin: quaternion_julia  (quaternion_julia.py)

Quaternion iteration
- We iterate a quaternion q in 4D:
    q_{n+1} = q_n^2 + C
- We visualize a 3D slice by fixing one component (W) and using (X,Y,Z) as space.

Field
- Escape-based field mapped to [0..1].
- MAX_ITER and bailout control the sharpness.

Tips
- Small changes in C create wildly different structures.
- If the set disappears: adjust ISO and BOUNDS first.
""",

    "rossler_attractor": r"""\
Rössler Attractor (density histogram)
Plugin: rossler_attractor  (rossler_attractor.py)

Differential equations
  dx/dt = -y - z
  dy/dt =  x + a*y
  dz/dt =  b + z*(x - c)

Euler integration
  x_{n+1} = x_n + dt * (-y_n - z_n)
  y_{n+1} = y_n + dt * ( x_n + a*y_n)
  z_{n+1} = z_n + dt * ( b + z_n*(x_n - c))

Density histogram
- The trajectory is normalized into the cube [-1..1]^3:
    xn = x/sx, yn = y/sy, zn = (z-10)/sz
- Voxel indices:
    ix = floor((xn+1)/2 * (N-1)), similarly iy, iz
- hist[ix,iy,iz] += 1

Smoothing (SMOOTH)
- A few passes of a 7-point stencil (center + 6 neighbors) with edge clamping.
- Makes the density look like continuous smoke instead of speckle noise.

Output mapping
  hist_norm = hist / max(hist)
  value = 1 - exp(-GAIN * hist_norm)

Tips
- Increase STEPS for denser attractor (slower).
- Increase SMOOTH for a softer look.
- If it becomes empty: reduce BOUNDS or reduce ISO.
""",

    "diamond_tpms": r"""\
Schwarz D (Diamond TPMS)
Plugin: schwarz_diamond  (schwarz_diamond.py)

Idea:
The Schwarz D (diamond) triply-periodic minimal surface.
A symmetric TPMS with diamond-like channels.

Formula (one common approximation):
- F = sin(x)*sin(y)*sin(z) + sin(x)*cos(y)*cos(z) + cos(x)*sin(y)*cos(z) + cos(x)*cos(y)*sin(z)

Mapping:
- output v = clamp(0.5 + 0.5 * F) (or a shell around F=0)

Notes:
- freq scales space
- ISO selects the shell thickness around F=0
""",

    "sierpinski_cube": r"""\
Sierpinski Cube (recursive voids / fractal)
Plugin: sierpinski_cube  (sierpinski_cube.py)

Idea
- A 3D Sierpinski-like structure built by recursively removing sub-cubes.
- Equivalent to a Menger-style rule with cube partitions.

Field
- For each point we evaluate whether it falls into a "kept" region after LEVELS steps.
- The result is mapped into a usable [0..1] scalar field.

Tips
- More LEVELS => more detail but slower.
- Adjust ISO to select the visible shell thickness.
""",

    "sierpinski_cube_octree": r"""\
Sierpinski Cube (octree variant)
Plugin: sierpinski_cube_octree  (sierpinski_cube_octree.py)

Idea
- Same visual family as Sierpinski/Menger, but implemented with an octree-style recursion.
- Uses subdivision and occupancy rules to produce a crisp blocky fractal.

Notes
- This version is usually faster for deeper recursion levels.
- Works well with moderate N and LEVELS.

Tips
- If it looks too "solid": lower ISO or reduce LEVELS.
""",

    "sierpinski_tetra": r"""\
Sierpinski tetra (membership + smoothing)
Plugin: sierpinski_tetra  (sierpinski_tetra.py)

Formulas:
  - works in [0,1]^3 and keeps only the tetra region x+y+z ≤ 1 1
      (x,y,z) := 2·(x,y,z)
      ix=int(x), iy=int(y), iz=int(z), potom frac part

Notes:
""",

    "superformula_3d": r"""\
Superformula 3D (Gielis-style implicit surface)
Plugin: superformula_3d  (superformula_3d.py)

Superformula (2D)
  r(φ) = [ (|cos(mφ/4)/a|^n2 + |sin(mφ/4)/b|^n3) ]^(-1/n1)

3D extension
- Build two superformula radii: one for latitude θ and one for longitude φ.
- Combine them into a 3D surface.

Field output
- Produces a smooth, highly parameterized family of shapes:
  stars, flowers, spiky blobs, shells, etc.

Tips
- Small parameter changes can be dramatic.
- Use GAIN/ISO to control how "thick" the resulting surface looks.
""",

    "superquadric": r"""\
Superquadric (implicit)
Plugin: superquadric  (superquadric.py)

Idea:
An implicit family of shapes between cubes and spheres, controlled by exponents.

Formula:
- (|x/a|^n + |y/b|^n)^(m/n) + |z/c|^m = 1

Notes:
- larger exponents -> more boxy
- smaller exponents -> more round / star-like
""",

    "torus": r"""\
Torus (SDF-like implicit)
Plugin: torus  (torus.py)

Idea:
Classic torus defined by major radius R and minor radius r.

Implicit / SDF-style:
- q = (sqrt(x^2 + y^2) - R, z)
- d = sqrt(qx^2 + qy^2) - r
- we convert distance to a smooth field around d=0

Notes:
- ISO controls the shell thickness around the torus surface
""",

    "tpms_schwarz_p": r"""\
TPMS Schwarz P (variant)
Plugin: tpms_schwarz_p  (tpms_schwarz_p.py)

Idea:
A Schwarz P TPMS variant used as an isosurface-based lattice.

Formula:
- F = cos(x) + cos(y) + cos(z)   (with optional warps / modes)

Notes:
- freq scales space
- ISO selects the shell thickness around F=0
""",

    "trefoil_knot": r"""\
Trefoil knot (implicit tube)
Plugin: trefoil_knot

We build an implicit "tube around a curve".
Curve (trefoil) parameterization:
  x(t) = (2 + cos(3t)) * cos(2t)
  y(t) = (2 + cos(3t)) * sin(2t)
  z(t) = sin(3t)
Then for each point P we approximate distance to the curve by sampling t.
Field = 1 - smoothstep(dist / thickness).

Notes:
- This is a sampled distance field (not exact). Increase SAMPLES for smoother tube.
- Keep N moderate; SAMPLES makes it heavier.
""",

    "twisted_ribbon": r"""\
Twisted Ribbon (implicit)
Plugin: twisted_ribbon  (twisted_ribbon.py)

Idea:
An implicit ribbon / band that twists along an axis. Useful as a clean "sculpture" field.

Concept:
- start with a band around a curve (or axis)
- apply twist as a function of position
- output a smooth field around the ribbon centerline

Notes:
- ISO controls thickness
- TWIST controls how fast the ribbon rotates
""",

    "wave_lattice": r"""\
Wave lattice (sin + cos "turbulence")
Plugin: wave_lattice  (wave_lattice.py)

Idea:
A smooth lattice made from three coupled trigonometric waves. The TWIST term adds a mild
cross-axis modulation (turbulence), producing organic "folds" while staying periodic.

Formulas:
- v0 = sin(x + TWIST*cos(y))
     + sin(y + TWIST*cos(z))
     + sin(z + TWIST*cos(x))
- v  = clamp( 0.5 + 0.5 * tanh( GAIN * v0 / 3 ) )

Notes:
- freq scales space: (x, y, z) := freq * (x, y, z)
- higher GAIN increases contrast (sharper walls)
""",

}