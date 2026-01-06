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

### 3.3 First Attempt: Fixed Step Size

Let's start with the simplest raymarching implementation - a fixed step size:

```javascript
const stepSize = 0.3;

Loop(256, () => {
    // Bend ray
    const toCenter = rayPos.negate().normalize();
    const bendStrength = rs.div(r.mul(r)).mul(stepSize).mul(1.5);
    rayDir.addAssign(toCenter.mul(bendStrength));
    rayDir.assign(normalize(rayDir));

    // Step forward
    rayPos.addAssign(rayDir.mul(stepSize));

    // Sample disk...
});
```

This works! We get a black hole with gravitational lensing and an accretion disk. But look closely at the inner regions of the disk:

**The Problem:** The inner disk is thin (we're simulating a realistic disk that gets thinner toward the center). With a fixed step size of 0.3 units, rays can skip right over the thinnest parts. This causes:
- Missing portions of the inner disk
- Aliased, jaggy edges
- Inconsistent brightness

The issue is fundamental: our step size is larger than the geometry we're trying to sample.

### 3.4 First Fix: Adaptive Stepping Near the Black Hole

Our first instinct might be to use smaller steps everywhere, but that's expensive. Instead, we can adapt the step size based on distance from the event horizon:

```javascript
const distFromHorizon = r.sub(rs);
const adaptiveStep = baseStepSize.mul(
    smoothstep(0.0, rs.mul(5.0), distFromHorizon)
        .mul(0.8).add(0.2)
);
```

This gives us:
- Step size = 20% of base near the event horizon
- Step size = 100% of base far from the black hole
- Smooth interpolation between

Better! The geodesic integration near the photon sphere is now more accurate. But we still have a problem: the thin inner disk regions are still getting skipped because our step size is based on distance from the *black hole*, not distance from the *disk*.

### 3.5 Second Fix: Disk-Aware Adaptive Stepping

The insight is that we need smaller steps when approaching the disk plane, especially where the disk is thin. Here's our improved adaptive stepper:

```javascript
// Factor 1: Distance from event horizon (for geodesic accuracy)
const distFromHorizon = r.sub(rs);
const horizonFactor = smoothstep(float(0.0), rs.mul(5.0), distFromHorizon)
    .mul(0.8).add(0.2);

// Factor 2 & 3: Disk proximity with thickness awareness
// Calculate horizontal distance (radius in disk plane)
const rHoriz = sqrt(rayPos.x.mul(rayPos.x).add(rayPos.z.mul(rayPos.z)));

// Check if we're within the disk's radial extent
const inDiskRegion = rHoriz.greaterThan(innerR.sub(diskMargin))
    .and(rHoriz.lessThan(outerR.add(diskMargin)));

// Calculate local disk thickness at this radius
// (inner disk is thinner than outer disk)
const normRForThickness = clamp(
    rHoriz.sub(innerR).div(outerR.sub(innerR)),
    float(0.0), float(1.0)
);
const localThickness = mix(
    diskInnerThickness,
    diskOuterThickness,
    normRForThickness
);

// When approaching the disk plane, reduce step size
// Scale based on distance to plane vs local thickness
const distToPlane = abs(rayPos.y);
const thicknessScale = localThickness.max(0.05);
const diskProximity = distToPlane.div(thicknessScale.mul(3.0));
const diskFactor = smoothstep(float(0.0), float(1.0), diskProximity)
    .mul(0.85).add(0.15);

// Combine: horizon factor always applies, disk factor only in disk region
const combinedDiskFactor = mix(float(1.0), diskFactor, step(float(0.5), inDiskRegion));
const adaptiveStep = stepSize.mul(horizonFactor).mul(combinedDiskFactor);
```

Now our step size accounts for:
1. **Black hole proximity** - accurate geodesics near the photon sphere
2. **Disk plane proximity** - smaller steps when approaching y = 0
3. **Local disk thickness** - even smaller steps in thin regions

The thin inner disk is now rendered correctly!

### 3.6 The Banding Problem

With our adaptive stepper, the disk shape is correct. But if you look carefully, you might notice another artifact: **banding**. The disk has visible stripes where the sampling is uniform.

This happens because every ray at a similar depth takes steps at the same positions. When those positions align with our noise functions or color gradients, we get coherent patterns instead of smooth gradients.

### 3.7 Third Fix: Step Jitter

The solution is to add controlled randomness to our sampling positions:

```javascript
// Apply jitter to sample position (not ray path) to break up banding
// This keeps the ray path deterministic for stable background stars
const sampleNoise = hash33(rayPos.add(vec3(frameIndex.mul(0.1))));
const jitterOffset = sampleNoise.sub(0.5).mul(adaptiveStep).mul(stepJitter);
const samplePos = rayPos.add(jitterOffset);

// Use samplePos (not rayPos) for disk sampling
const hitR = sqrt(samplePos.x.mul(samplePos.x).add(samplePos.z.mul(samplePos.z)));
```

Key insight: we jitter the **sampling position**, not the **ray position**. If we jittered the ray itself, background stars would flicker because each frame would trace a different path. By keeping the ray path deterministic and only jittering where we sample the disk, we get:
- Smooth, band-free disk rendering
- Stable, flicker-free background stars

The `stepJitter` parameter (typically 0.15-0.3) controls how much randomness to add. More jitter breaks up banding better but can introduce noise at low sample counts.

### 3.8 Disk Color and Opacity

With our sampling working correctly, we can focus on the disk appearance:

```javascript
const accretionDiskColor = Fn(([hitR, hitAngle, time]) => {
    // Normalized radius (0 at inner edge, 1 at outer edge)
    const normR = clamp(hitR.sub(innerR).div(outerR.sub(innerR)), 0.0, 1.0);

    // Temperature profile: T ~ r^(-3/4)
    const temperature = pow(normR.add(0.05), float(-0.75)).mul(diskTemperature);

    // Color based on temperature
    const colorMix = smoothstep(float(0.5), float(2.5), temperature);
    const diskColor = mix(outerColor, innerColor, colorMix);

    // Turbulence for realistic swirling patterns
    const orbitalPhase = hitAngle.add(time.mul(rotationSpeed).div(sqrt(hitR.add(0.5))));
    const turbCoord = vec3(
        cos(orbitalPhase).mul(hitR),
        sin(orbitalPhase).mul(hitR),
        hitR.mul(0.5)
    );
    const turbulence = fbm(turbCoord.mul(turbulenceScale));

    return vec4(diskColor, turbulence);  // rgb = color, a = opacity
});
```

### 3.9 Creating Ring Patterns

Real accretion disks aren't uniformly bright - they have concentric ring structures caused by density waves, orbital resonances, and variations in the flow of material. Let's add these patterns to make our disk more realistic.

#### First Attempt: Sine Waves

The most obvious approach is to use sine waves based on radius:

```javascript
// Simple sine wave rings
const ringPattern = sin(hitR.mul(ringScale)).mul(0.5).add(0.5);
const opacity = ringPattern.mul(baseOpacity);
```

This creates perfectly regular, evenly-spaced rings:

The problem? Nature isn't this regular. Real accretion disk features vary in spacing, intensity, and sharpness. Our sine wave rings look artificial - like the grooves on a vinyl record rather than the chaotic flow of superheated plasma.

#### The Fix: 1D Noise

Instead of regular sine waves, we can use 1D noise to create irregular ring patterns. The key insight is that we only need to vary by *radius* - we want concentric rings, not random blobs.

```javascript
/**
 * 1D Value noise for ring patterns.
 * Takes a single float input and returns smooth noise in [0,1].
 */
const noise1D = Fn(([x]) => {
    const i = floor(x);
    const f = fract(x);
    // Quintic interpolation for smooth results
    const u = f.mul(f).mul(f).mul(f.mul(f.mul(6.0).sub(15.0)).add(10.0));
    // Hash the integer positions
    const a = fract(sin(i.mul(127.1)).mul(43758.5453));
    const b = fract(sin(i.add(1.0).mul(127.1)).mul(43758.5453));
    return mix(a, b, u);
});
```

But a single octave of noise creates rings that are all similar in thickness. For more natural variation, we use Fractal Brownian Motion (FBM) - multiple octaves of noise at different scales:

```javascript
/**
 * 1D Fractal Brownian Motion for ring patterns.
 */
const fbm1D = Fn(([x, octaves, lacunarity, persistence]) => {
    const value = float(0.0).toVar();
    const amplitude = float(1.0).toVar();
    const frequency = float(1.0).toVar();
    const maxValue = float(0.0).toVar();

    // Add multiple octaves of noise
    for (let i = 0; i < octaves; i++) {
        value.addAssign(noise1D(x.mul(frequency)).mul(amplitude));
        maxValue.addAssign(amplitude);
        amplitude.mulAssign(persistence);  // Each octave is quieter
        frequency.mulAssign(lacunarity);   // Each octave is higher frequency
    }

    return value.div(maxValue);  // Normalize to [0, 1]
});
```

The parameters control the character of the rings:
- **Scale**: How many rings across the disk (3-10 works well)
- **Octaves**: How many layers of detail (3-4 is typical)
- **Lacunarity**: Frequency multiplier between octaves (2.0 = each octave is twice the frequency)
- **Persistence**: Amplitude multiplier between octaves (0.5 = each octave is half as loud)

Now we apply this to our disk opacity:

```javascript
// Sample 1D FBM noise based on radius
const noiseInput = hitR.mul(ringNoiseScale);
const ringNoise = fbm1D(
    noiseInput,
    ringNoiseOctaves,
    ringNoiseLacunarity,
    ringNoisePersistence
);

// Apply amplitude, offset, and sharpness
const rawRing = ringNoise.mul(ringNoiseAmplitude).add(ringNoiseOffset);
const ringOpacity = pow(clamp(rawRing, 0.0, 1.0), ringNoiseSharpness);
```

The **sharpness** parameter is particularly useful - higher values create sharper, more defined ring boundaries, while lower values give softer, more diffuse patterns.

The result is much more convincing: rings that vary in spacing, thickness, and intensity, creating the kind of structure we see in actual astronomical observations.

### 3.10 Relativistic Effects

**Gravitational Redshift:**
Light loses energy climbing out of a gravity well:

```javascript
const redshift = sqrt(1.0 - rs / hitR);
diskColor.mulAssign(redshift);
```

### 3.11 Procedural Background

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

### 4.2 The Adaptive Stepping Performance Benefit

Our disk-aware adaptive stepping isn't just about quality - it's also about performance. By using larger steps in empty space, we:
- Take fewer total steps to traverse the same distance
- Spend our step budget where it matters (near geometry)
- Maintain quality while improving frame rate

### 4.3 Early Termination

Exit the loop as soon as we know the ray's fate:
- **Captured**: `r < rs` - ray fell into black hole
- **Escaped**: `totalDistance > maxDistance` - ray left the scene
- **Opaque**: `alpha > 0.99` - accumulated enough disk material

---

## Part 5: Results

The final simulation achieves:
- **Real-time performance** (30-60 FPS) on modern GPUs
- **Physically-based** gravitational lensing
- **Interactive** camera controls and parameters
- **Beautiful** accretion disk with turbulence and proper thin-disk handling

The effect is most dramatic when you orbit the camera around the black hole - you can see how the disk bends above and below, creating the iconic "Interstellar" look.

---

## Conclusion

We've built a real-time black hole visualization using WebGPU and Three.js. Along the way, we learned:

1. **Schwarzschild spacetime** - How black holes curve light
2. **Raymarching** - Why it's ideal for curved spacetime
3. **Adaptive stepping** - Solving aliasing with geometry-aware step sizes
4. **Jitter** - Breaking up banding artifacts while keeping background stable
5. **TSL shaders** - Writing GPU code in JavaScript

The key lesson is that building graphics involves an iterative process: implement something simple, observe where it breaks, then fix it with targeted solutions. Each "fix" deepens our understanding of both the problem and the underlying physics.

**Potential extensions:**
- Spinning (Kerr) black holes with frame dragging
- Wormholes connecting two regions of space
- Relativistic jets
- Gravitational waves affecting the spacetime

---

## References

1. James, O., von Tunzelmann, E., Franklin, P., & Thorne, K. S. (2015). *Gravitational lensing by spinning black holes in astrophysics, and in the movie Interstellar*. Classical and Quantum Gravity.

2. Schwarzschild, K. (1916). *On the gravitational field of a mass point according to Einstein's theory*.

3. Shakura, N. I., & Sunyaev, R. A. (1973). *Black holes in binary systems. Observational appearance*.

4. Three.js TSL Documentation: https://threejs.org/docs/#api/en/tsl/

---

*Built with Three.js WebGPU and TSL. [View source on GitHub](https://github.com/dgreenheck/webgpu-galaxy)*
