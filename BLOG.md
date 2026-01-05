# Raytracing a Black Hole with WebGPU: A Deep Dive

*How to render scientifically-accurate gravitational lensing in real-time using Three.js and WebGPU*

---

## Introduction

When *Interstellar* hit theaters in 2014, audiences were captivated by the hauntingly beautiful imagery of Gargantua, the supermassive black hole at the film's center. What made these visuals remarkable wasn't just their aesthetic appeal - they were based on real physics, with gravitational lensing calculations performed by a team led by physicist Kip Thorne.

In this tutorial, we'll recreate this effect in real-time using WebGPU and Three.js's new Shading Language (TSL). By the end, you'll understand both the physics behind black hole visualization and how to implement it efficiently in a browser.

**What we'll build:**
- A raymarched black hole with gravitational lensing
- An accretion disk with temperature-based coloring
- Doppler beaming (relativistic brightness variation)
- A procedural star field that gets distorted by gravity
- Interactive controls for all parameters

[Live Demo](#) | [Source Code](https://github.com/dgreenheck/webgpu-galaxy)

---

## Part 1: The Physics of Black Holes

### 1.1 Schwarzschild Spacetime

A black hole is a region of spacetime where gravity is so strong that nothing - not even light - can escape. For a non-rotating black hole (called a Schwarzschild black hole), the key radius is the **event horizon**:

```
rs = 2GM/c²
```

Where:
- `G` is the gravitational constant
- `M` is the black hole's mass
- `c` is the speed of light

In our simulation, we use geometric units where `G = c = 1`, so `rs = 2M`.

There's another critical radius called the **photon sphere** at `r = 1.5rs`. This is where photons can orbit the black hole in unstable circular orbits. Light passing just outside this radius can loop around the black hole multiple times before escaping.

### 1.2 How Light Bends Around a Black Hole

In general relativity, massive objects curve spacetime, and light follows the curves - called **geodesics**. Near a black hole, these curves can be dramatic.

The key concept is the **impact parameter** `b` - the perpendicular distance at which a light ray would pass the black hole if space were flat. There's a critical value:

```
b_critical = rs × sqrt(27)/2 ≈ 2.6 × rs
```

Rays with `b < b_critical` fall into the black hole. Rays with `b > b_critical` escape, but are bent. The closer to the critical value, the more bending occurs.

This bending creates several visual effects:
1. **Einstein rings** - Background stars appear as rings around the black hole
2. **Multiple images** - We can see the same object from different angles
3. **The shadow** - A dark region where light has fallen in

### 1.3 The Accretion Disk

Matter falling into a black hole doesn't drop straight in. Due to conservation of angular momentum, it forms a flat, rotating disk called an **accretion disk**.

**Temperature Profile:**
The inner disk is hotter because gravitational potential energy converts to heat as matter falls inward. The standard thin disk model (Shakura-Sunyaev) predicts:

```
T ∝ r^(-3/4)
```

This means the inner edge glows white-hot while the outer edge is cooler (red/orange).

**Doppler Beaming:**
Because the disk rotates, material on one side moves toward us while the other side moves away. Relativistic effects make the approaching side appear brighter and bluer - this is called **Doppler beaming**.

---

## Part 2: The Rendering Approach

### 2.1 Why Raymarching?

Traditional rasterization renders objects by projecting triangles onto the screen. But gravitational lensing bends light in ways that triangles can't represent. We need to trace each ray's path through curved spacetime.

**Raymarching** is perfect for this:
1. For each pixel, create a ray from the camera
2. Step the ray forward in small increments
3. At each step, bend the ray toward the black hole
4. Check if the ray hits the disk, falls into the hole, or escapes

### 2.2 Algorithm Overview

```
for each pixel:
    ray = createRay(camera, pixelCoordinates)
    color = black

    for step in range(maxSteps):
        // Bend ray toward black hole
        ray.direction += gravitationalAcceleration

        // Step forward
        ray.position += ray.direction * stepSize

        // Check disk intersection
        if crossedDiskPlane():
            color += getDiskColor(hitPosition)

        // Check termination
        if ray.position.length < eventHorizon:
            break  // Captured
        if ray.position.length > maxDistance:
            color += getBackgroundColor(ray.direction)
            break  // Escaped

    return color
```

---

## Part 3: Implementation Deep Dive

### 3.1 Setting Up Three.js with TSL

Three.js recently introduced **Three Shading Language (TSL)**, a JavaScript-based shader language that compiles to WGSL (WebGPU) or GLSL (WebGL). This lets us write shaders in a familiar syntax.

```javascript
import * as THREE from 'three/webgpu';
import { screenUV, vec3, float, Fn, Loop } from 'three/tsl';

// Create a fullscreen shader
const material = new THREE.MeshBasicNodeMaterial();
material.colorNode = Fn(() => {
    // Our raymarching shader goes here
    return vec4(1, 0, 0, 1);  // Red for testing
})();
```

### 3.2 The Geodesic Integrator

The core of our simulation is bending light rays. We use a simplified model based on the Schwarzschild metric:

```javascript
// Schwarzschild radius (event horizon)
const rs = blackHoleMass.mul(2.0);

// Direction from ray to black hole center
const toCenter = rayPos.negate().normalize();

// Bend strength: stronger when closer
// Based on: a ≈ -rs/(2r²)
const r = length(rayPos);
const bendStrength = rs.div(r.mul(r)).mul(stepSize).mul(1.5);

// Apply bending to ray direction
rayDir.addAssign(toCenter.mul(bendStrength));
rayDir.assign(normalize(rayDir));
```

**Adaptive Step Size:**
We use smaller steps near the black hole for accuracy, larger steps far away for performance:

```javascript
const distFromHorizon = r.sub(rs);
const adaptiveStep = baseStepSize.mul(
    smoothstep(0.0, rs.mul(5.0), distFromHorizon)
        .mul(0.8).add(0.2)
);
```

### 3.3 Disk Intersection and Rendering

The accretion disk lies in the equatorial plane (y = 0). We detect crossings by checking for sign changes in the y-coordinate:

```javascript
const prevY = rayPos.y;
// ... step ray forward ...
const currY = rayPos.y;

// Did we cross the plane?
If(sign(prevY).notEqual(sign(currY)), () => {
    // Interpolate exact crossing point
    const t = abs(prevY).div(abs(prevY).add(abs(currY)));
    const hitX = rayPos.x.sub(rayDir.x.mul(stepSize.mul(1.0 - t)));
    const hitZ = rayPos.z.sub(rayDir.z.mul(stepSize.mul(1.0 - t)));
    const hitR = sqrt(hitX.mul(hitX).add(hitZ.mul(hitZ)));

    // Check if within disk bounds
    If(hitR.greaterThan(innerRadius).and(hitR.lessThan(outerRadius)), () => {
        // Calculate disk color
        const hitAngle = atan(hitZ, hitX);
        const diskColor = accretionDiskColor(hitR, hitAngle, time);
        color.addAssign(diskColor);
    });
});
```

**Ring Structure:**
Real accretion disks show concentric rings due to density variations. We simulate this with layered sine waves:

```javascript
const ring1 = sin(hitR.mul(ringCount).mul(0.8)).mul(0.5).add(0.5);
const ring2 = sin(hitR.mul(ringCount).mul(2.0).add(time)).mul(0.3).add(0.7);
const ring3 = sin(hitR.mul(ringCount).mul(5.0)).mul(0.2).add(0.8);
const ringPattern = ring1.mul(ring2).mul(ring3);
```

**Turbulence:**
For realistic swirling patterns, we add Fractal Brownian Motion (FBM) noise:

```javascript
const turbCoord = vec3(
    cos(hitAngle).mul(hitR.mul(0.3)),
    sin(hitAngle).mul(hitR.mul(0.3)),
    time.mul(0.1)
);
const turbulence = fbm(turbCoord).mul(turbulenceAmount);
```

### 3.4 Relativistic Effects

**Doppler Beaming:**
Material in the disk orbits the black hole. The side moving toward us appears brighter:

```javascript
// Orbital velocity (Keplerian)
const orbitalSpeed = sqrt(blackHoleMass.div(hitR)).mul(0.4);

// Velocity direction (perpendicular to radius)
const velDir = vec3(sin(hitAngle).negate(), 0.0, cos(hitAngle));

// Doppler factor
const dopplerFactor = 1.0 + dot(velDir, rayDir.negate()) * orbitalSpeed * strength;
const doppler = pow(clamp(dopplerFactor, 0.5, 2.0), 3.0);

diskColor.mulAssign(doppler);
```

**Gravitational Redshift:**
Light loses energy climbing out of a gravity well:

```javascript
const redshift = sqrt(1.0 - rs / hitR);
diskColor.mulAssign(redshift);
```

### 3.5 Procedural Background

Our star field uses a grid-based approach for consistent star positions:

```javascript
const starField = Fn(([rayDir]) => {
    // Convert to spherical coordinates
    const theta = atan(rayDir.z, rayDir.x);
    const phi = asin(rayDir.y);

    // Grid cell
    const cell = vec2(theta, phi).mul(gridScale).floor();

    // Hash determines if this cell has a star
    const cellHash = hash21(cell);
    const hasStar = cellHash < starDensity;

    // Star position within cell
    const starPos = hash33(cell).xy.mul(0.8).add(0.1);
    const distToStar = length(fract(cell) - starPos);

    // Brightness based on distance
    const brightness = smoothstep(starSize, 0.0, distToStar) * hasStar;

    return starColor.mul(brightness);
});
```

Because we apply this to the *bent* ray direction, stars near the black hole appear distorted - exactly as physics predicts.

---

## Part 4: Performance Optimization

Raymarching is expensive. Here's how we achieve real-time performance:

### 4.1 Quality Presets

We expose ray count and step size as parameters:

| Preset | Ray Steps | Step Size | Target FPS |
|--------|-----------|-----------|------------|
| Low    | 64        | 0.4       | 60         |
| Medium | 100       | 0.3       | 30-60      |
| High   | 150       | 0.2       | 30         |
| Ultra  | 256       | 0.15      | 15-30      |

### 4.2 Adaptive Stepping

Small steps near the black hole, large steps far away:

```javascript
const step = baseStep * smoothstep(0, 5*rs, distFromHorizon);
```

### 4.3 Early Termination

Exit the loop as soon as we know the ray's fate:
- Captured: `r < rs`
- Escaped: `totalDistance > maxDistance`
- Opaque: `alpha > 0.99`

---

## Part 5: Results

The final simulation achieves:
- **Real-time performance** (30-60 FPS) on modern GPUs
- **Physically-based** gravitational lensing
- **Interactive** camera controls and parameters
- **Beautiful** accretion disk with rings and turbulence

The effect is most dramatic when you orbit the camera around the black hole - you can see how the disk bends above and below, creating the iconic "Interstellar" look.

---

## Conclusion

We've built a real-time black hole visualization using WebGPU and Three.js. Along the way, we learned:

1. **Schwarzschild spacetime** - How black holes curve light
2. **Raymarching** - Why it's ideal for curved spacetime
3. **Accretion disks** - Temperature profiles and Doppler beaming
4. **TSL shaders** - Writing GPU code in JavaScript

**Potential extensions:**
- Spinning (Kerr) black holes with frame dragging
- Wormholes connecting two regions of space
- Volumetric accretion disk rendering
- Relativistic jets

---

## References

1. James, O., von Tunzelmann, E., Franklin, P., & Thorne, K. S. (2015). *Gravitational lensing by spinning black holes in astrophysics, and in the movie Interstellar*. Classical and Quantum Gravity.

2. Schwarzschild, K. (1916). *On the gravitational field of a mass point according to Einstein's theory*.

3. Shakura, N. I., & Sunyaev, R. A. (1973). *Black holes in binary systems. Observational appearance*.

4. Three.js TSL Documentation: https://threejs.org/docs/#api/en/tsl/

---

*Built with Three.js WebGPU and TSL. [View source on GitHub](https://github.com/dgreenheck/webgpu-galaxy)*
