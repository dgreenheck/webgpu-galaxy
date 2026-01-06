/**
 * ============================================================================
 * BLACK HOLE SIMULATION WITH RAYMARCHED GRAVITATIONAL LENSING
 * ============================================================================
 *
 * This simulation renders a Schwarzschild (non-rotating) black hole with:
 * - Gravitational lensing of light rays through curved spacetime
 * - Accretion disk with temperature-based coloring and turbulence
 * - Doppler beaming (relativistic brightness variation)
 * - Procedural star field and nebula background
 * - Photon ring at the critical impact parameter
 *
 * The code is organized pedagogically for use in a blog post explaining
 * the physics and implementation of black hole rendering.
 *
 * @author Daniel Greenheck
 * @see BLOG.md for detailed explanation
 */

import * as THREE from 'three/webgpu';
import {
  uniform,
  vec2,
  vec3,
  vec4,
  float,
  Fn,
  length,
  normalize,
  cross,
  dot,
  sin,
  cos,
  atan,
  asin,
  sqrt,
  abs,
  pow,
  exp,
  fract,
  clamp,
  smoothstep,
  mix,
  floor,
  step,
  Loop,
  Break,
  If,
  screenUV
} from 'three/tsl';

// ============================================================================
// SECTION 1: PHYSICAL CONSTANTS AND CONFIGURATION
// ============================================================================

/**
 * Quality presets for balancing visual quality vs performance.
 * These control the raymarching parameters.
 */
export const QUALITY_PRESETS = {
  low: {
    raySteps: 64,
    stepSize: 0.4,
    diskDetail: 1,
    starsEnabled: false,
    nebulaEnabled: false,
    rayJitter: 1.0,      // Higher jitter compensates for fewer steps
    stepJitter: 0.3
  },
  medium: {
    raySteps: 100,
    stepSize: 0.3,
    diskDetail: 2,
    starsEnabled: true,
    nebulaEnabled: false,
    rayJitter: 1.0,
    stepJitter: 0.25
  },
  high: {
    raySteps: 150,
    stepSize: 0.2,
    diskDetail: 3,
    starsEnabled: true,
    nebulaEnabled: true,
    rayJitter: 0.8,
    stepJitter: 0.2
  },
  ultra: {
    raySteps: 256,
    stepSize: 0.15,
    diskDetail: 4,
    starsEnabled: true,
    nebulaEnabled: true,
    rayJitter: 0.6,      // Less jitter needed with more steps
    stepJitter: 0.15
  }
};

// ============================================================================
// SECTION 2: BLACK HOLE SIMULATION CLASS
// ============================================================================

export class BlackHoleSimulation {
  constructor(scene, config) {
    this.scene = scene;
    this.config = config;
    this.blackHoleMesh = null;
    this.initializeUniforms(config);
  }

  /**
   * Initialize all shader uniforms with default values.
   * These can be updated in real-time via the UI.
   */
  initializeUniforms(config) {
    this.uniforms = {
      // === Physics ===
      blackHoleMass: uniform(config.blackHoleMass ?? 1.0),

      // === Accretion Disk Geometry ===
      diskInnerRadius: uniform(config.diskInnerRadius ?? 3.0),
      diskOuterRadius: uniform(config.diskOuterRadius ?? 12.0),

      // === Accretion Disk Appearance ===
      diskTemperature: uniform(config.diskTemperature ?? 1.5),
      diskBrightness: uniform(config.diskBrightness ?? 2.0),
      diskRotationSpeed: uniform(config.diskRotationSpeed ?? 0.3),
      diskInnerThickness: uniform(config.diskInnerThickness ?? 0.1),
      diskOuterThickness: uniform(config.diskOuterThickness ?? 0.8),

      // === Ring Pattern Controls ===
      ringEnabled: uniform(config.ringEnabled ? 1.0 : 0.0),
      ringScale: uniform(config.ringScale ?? 1.0),
      ringContrast: uniform(config.ringContrast ?? 1.5),
      ringBrightness: uniform(config.ringBrightness ?? 0.3),
      ringSharpness: uniform(config.ringSharpness ?? 1.0),
      ringTwist: uniform(config.ringTwist ?? 0.5),
      diskDifferentialRotation: uniform(config.diskDifferentialRotation ?? 0.8),
      noiseEvolutionSpeed: uniform(config.noiseEvolutionSpeed ?? 1.0),

      // === Disk Edge Falloff ===
      diskEdgeSoftnessInner: uniform(config.diskEdgeSoftnessInner ?? 0.15),
      diskEdgeSoftnessOuter: uniform(config.diskEdgeSoftnessOuter ?? 0.15),
      diskRadialFalloff: uniform(config.diskRadialFalloff ?? 0.5),

      // === Disk Color (User Configurable) ===
      diskInnerColor: uniform(new THREE.Color(config.diskInnerColor ?? '#ffffee')),
      diskOuterColor: uniform(new THREE.Color(config.diskOuterColor ?? '#ff4400')),

      // === Relativistic Effects ===
      gravitationalLensing: uniform(config.gravitationalLensing ?? 1.5),

      // === Volumetric Rendering ===
      diskDensity: uniform(config.diskDensity ?? 0.25),
      diskOpacityFalloff: uniform(config.diskOpacityFalloff ?? 0.8),

      // === Performance ===
      raySteps: uniform(config.raySteps ?? 100),
      stepSize: uniform(config.stepSize ?? 0.3),
      adaptiveMinStep: uniform(config.adaptiveMinStep ?? 0.15),

      // === Anti-Aliasing ===
      rayJitter: uniform(config.rayJitter ?? 1.0),
      stepJitter: uniform(config.stepJitter ?? 0.25),
      frameIndex: uniform(0),

      // === Stars ===
      starsEnabled: uniform(config.starsEnabled ? 1.0 : 0.0),
      starBackgroundColor: uniform(new THREE.Color(config.starBackgroundColor ?? '#000000')),
      starDensity: uniform(config.starDensity ?? 0.003),
      starSize: uniform(config.starSize ?? 2.0),
      starBrightness: uniform(config.starBrightness ?? 1.0),

      // === Nebula ===
      nebulaEnabled: uniform(config.nebulaEnabled ? 1.0 : 0.0),
      nebulaBrightness: uniform(config.nebulaBrightness ?? 0.15),
      nebulaColor1: uniform(new THREE.Color(config.nebulaColor1 ?? '#1a0033')),
      nebulaColor2: uniform(new THREE.Color(config.nebulaColor2 ?? '#4d1a26')),
      nebulaScale1: uniform(config.nebulaScale1 ?? 2.0),
      nebulaScale2: uniform(config.nebulaScale2 ?? 6.0),
      nebulaBlend: uniform(config.nebulaBlend ?? 0.3),
      nebulaSpeed: uniform(config.nebulaSpeed ?? 0.01),
      nebulaDensity: uniform(config.nebulaDensity ?? 0.5),

      // === Animation State ===
      time: uniform(0),

      // === Camera ===
      resolution: uniform(new THREE.Vector2(window.innerWidth, window.innerHeight)),
      cameraPosition: uniform(new THREE.Vector3(0, 5, 20)),
      cameraTarget: uniform(new THREE.Vector3(0, 0, 0))
    };
  }

  /**
   * Create the black hole visualization mesh.
   * Uses a large inverted sphere as the render surface with a custom TSL shader.
   */
  createBlackHole() {
    // Clean up existing mesh
    if (this.blackHoleMesh) {
      this.scene.remove(this.blackHoleMesh);
      this.blackHoleMesh.material?.dispose();
      this.blackHoleMesh.geometry?.dispose();
    }

    // Create inverted sphere geometry (renders from inside)
    const geometry = new THREE.SphereGeometry(100, 32, 32);
    geometry.scale(-1, 1, 1);

    // Create material with our raymarching shader
    const material = new THREE.MeshBasicNodeMaterial();
    material.colorNode = this.createRaymarchingShader();

    this.blackHoleMesh = new THREE.Mesh(geometry, material);
    this.blackHoleMesh.frustumCulled = false;
    this.scene.add(this.blackHoleMesh);
  }

  /**
   * Main shader creation method.
   * Builds the complete raymarching shader using Three.js TSL.
   */
  createRaymarchingShader() {
    const uniforms = this.uniforms;

    // ========================================================================
    // SECTION 3: UTILITY FUNCTIONS
    // ========================================================================

    /**
     * Hash function for pseudo-random number generation.
     * Used for procedural star and noise generation.
     */
    const hash21 = Fn(([p]) => {
      const n = sin(dot(p, vec2(127.1, 311.7))).mul(43758.5453);
      return fract(n);
    });

    const hash31 = Fn(([p]) => {
      const n = sin(dot(p, vec3(127.1, 311.7, 74.7))).mul(43758.5453);
      return fract(n);
    });

    const hash33 = Fn(([p]) => {
      const px = fract(sin(dot(p, vec3(127.1, 311.7, 74.7))).mul(43758.5453));
      const py = fract(sin(dot(p, vec3(269.5, 183.3, 246.1))).mul(43758.5453));
      const pz = fract(sin(dot(p, vec3(113.5, 271.9, 124.6))).mul(43758.5453));
      return vec3(px, py, pz);
    });

    /**
     * 3D Value noise for turbulence effects.
     */
    const noise3D = Fn(([p]) => {
      const i = floor(p);
      const f = fract(p);

      // Smooth interpolation
      const u = f.mul(f).mul(float(3.0).sub(f.mul(2.0)));

      // Hash corners
      const a = hash31(i);
      const b = hash31(i.add(vec3(1, 0, 0)));
      const c = hash31(i.add(vec3(0, 1, 0)));
      const d = hash31(i.add(vec3(1, 1, 0)));
      const e = hash31(i.add(vec3(0, 0, 1)));
      const f2 = hash31(i.add(vec3(1, 0, 1)));
      const g = hash31(i.add(vec3(0, 1, 1)));
      const h = hash31(i.add(vec3(1, 1, 1)));

      // Trilinear interpolation
      return mix(
        mix(mix(a, b, u.x), mix(c, d, u.x), u.y),
        mix(mix(e, f2, u.x), mix(g, h, u.x), u.y),
        u.z
      );
    });

    /**
     * Fractal Brownian Motion - layered noise for natural-looking turbulence.
     * @param p - 3D position
     * @param octaves - Number of noise layers (more = more detail but slower)
     */
    const fbm = Fn(([p]) => {
      const value = float(0.0).toVar();
      const amplitude = float(0.5).toVar();
      const pos = p.toVar();

      // 4 octaves of noise
      value.addAssign(noise3D(pos).mul(amplitude));
      pos.mulAssign(2.0);
      amplitude.mulAssign(0.5);

      value.addAssign(noise3D(pos).mul(amplitude));
      pos.mulAssign(2.0);
      amplitude.mulAssign(0.5);

      value.addAssign(noise3D(pos).mul(amplitude));
      pos.mulAssign(2.0);
      amplitude.mulAssign(0.5);

      value.addAssign(noise3D(pos).mul(amplitude));

      return value;
    });

    /**
     * 1D Value noise for ring patterns.
     * Takes a single float input and returns smooth noise in [0,1].
     */
    const noise1D = Fn(([x]) => {
      const i = floor(x);
      const f = fract(x);
      // Smooth interpolation (quintic for smoother results)
      const u = f.mul(f).mul(f).mul(f.mul(f.mul(6.0).sub(15.0)).add(10.0));
      // Hash the integer positions
      const a = fract(sin(i.mul(127.1)).mul(43758.5453));
      const b = fract(sin(i.add(1.0).mul(127.1)).mul(43758.5453));
      return mix(a, b, u);
    });

    /**
     * 1D Fractal Brownian Motion for ring patterns.
     * Produces natural-looking variation with configurable octaves.
     */
    const fbm1D = Fn(([x, octaves, lacunarity, persistence]) => {
      const value = float(0.0).toVar();
      const amplitude = float(1.0).toVar();
      const frequency = float(1.0).toVar();
      const maxValue = float(0.0).toVar();
      const pos = x.toVar();

      // Unrolled loop for up to 4 octaves (controlled by octaves uniform)
      // Octave 1
      If(octaves.greaterThanEqual(1.0), () => {
        value.addAssign(noise1D(pos.mul(frequency)).mul(amplitude));
        maxValue.addAssign(amplitude);
        amplitude.mulAssign(persistence);
        frequency.mulAssign(lacunarity);
      });
      // Octave 2
      If(octaves.greaterThanEqual(2.0), () => {
        value.addAssign(noise1D(pos.mul(frequency)).mul(amplitude));
        maxValue.addAssign(amplitude);
        amplitude.mulAssign(persistence);
        frequency.mulAssign(lacunarity);
      });
      // Octave 3
      If(octaves.greaterThanEqual(3.0), () => {
        value.addAssign(noise1D(pos.mul(frequency)).mul(amplitude));
        maxValue.addAssign(amplitude);
        amplitude.mulAssign(persistence);
        frequency.mulAssign(lacunarity);
      });
      // Octave 4
      If(octaves.greaterThanEqual(4.0), () => {
        value.addAssign(noise1D(pos.mul(frequency)).mul(amplitude));
        maxValue.addAssign(amplitude);
      });

      // Normalize to [0, 1]
      return value.div(maxValue.max(0.001));
    });

    // ========================================================================
    // SECTION 4: PROCEDURAL BACKGROUND
    // ========================================================================

    /**
     * Generate procedural star field.
     * Stars are placed using a grid-based hash function for consistent positions.
     */
    const starField = Fn(([rayDir]) => {
      // Convert ray direction to spherical coordinates for grid
      const theta = atan(rayDir.z, rayDir.x);
      const phi = asin(clamp(rayDir.y, float(-1.0), float(1.0)));

      // Create grid cells - lower scale = larger cells = bigger stars
      const gridScale = float(60.0).div(uniforms.starSize);
      const cell = vec2(theta, phi).mul(gridScale).floor();
      const cellUV = fract(vec2(theta, phi).mul(gridScale));

      // Hash for this cell
      const cellHash = hash21(cell);

      // Star probability - most cells are empty
      const starProb = step(float(1.0).sub(uniforms.starDensity), cellHash);

      // Star position within cell
      const starPos = hash33(vec3(cell.x, cell.y, float(42.0))).xy.mul(0.8).add(0.1);
      const distToStar = length(cellUV.sub(starPos));

      // Star size with variation - base size scaled by uniform
      const baseSizeVar = hash21(cell.add(100.0)).mul(0.03).add(0.01);
      const finalStarSize = baseSizeVar.mul(uniforms.starSize);

      // Star brightness with soft glow
      const starCore = smoothstep(finalStarSize, float(0.0), distToStar);
      const starGlow = smoothstep(finalStarSize.mul(3.0), float(0.0), distToStar).mul(0.3);
      const starIntensity = starCore.add(starGlow).mul(starProb);

      // Star color variation (blue to yellow)
      const colorTemp = hash21(cell.add(200.0));
      const starColor = mix(
        vec3(0.8, 0.9, 1.0),  // Blue-white
        vec3(1.0, 0.95, 0.8), // Yellow-white
        colorTemp
      );

      return starColor.mul(starIntensity).mul(uniforms.starBrightness);
    });

    /**
     * Generate procedural nebula clouds.
     * Uses two noise layers at different frequencies for depth.
     * Noise amplitude is -1 to 1, density offsets to control visibility.
     */
    const nebulaField = Fn(([rayDir, time]) => {
      // Layer 1: Large scale structures
      const noisePos1 = rayDir.mul(uniforms.nebulaScale1);
      const n1 = fbm(noisePos1.add(time.mul(uniforms.nebulaSpeed))).mul(2.0).sub(1.0); // Remap to [-1, 1]

      // Layer 2: Higher frequency detail
      const noisePos2 = rayDir.mul(uniforms.nebulaScale2);
      const n2 = fbm(noisePos2.sub(time.mul(uniforms.nebulaSpeed.mul(0.5)))).mul(2.0).sub(1.0); // Remap to [-1, 1]

      // Combine layers - blend controls mix, density offsets for visibility threshold
      const layer1Weight = float(1.0).sub(uniforms.nebulaBlend);
      const combined = n1.mul(layer1Weight).add(n2.mul(uniforms.nebulaBlend)).add(uniforms.nebulaDensity);
      const nebula = clamp(combined, float(0.0), float(1.0));

      // Color gradient based on first noise layer
      const colorMix = n1.mul(0.5).add(0.5); // Remap to [0, 1] for color mixing
      const nebulaColor = mix(uniforms.nebulaColor1, uniforms.nebulaColor2, colorMix);

      return nebulaColor.mul(nebula).mul(uniforms.nebulaBrightness);
    });

    // ========================================================================
    // SECTION 5: ACCRETION DISK
    // ========================================================================

    /**
     * Calculate the color and opacity of the accretion disk at a given point.
     * Returns vec4(color.rgb, opacity) where ring patterns control opacity.
     */
    const accretionDiskColor = Fn(([hitR, hitAngle, time]) => {
      const innerR = uniforms.diskInnerRadius;
      const outerR = uniforms.diskOuterRadius;

      // Normalized radius (0 at inner edge, 1 at outer edge)
      const normR = clamp(hitR.sub(innerR).div(outerR.sub(innerR)), float(0.0), float(1.0));

      // === BLACKBODY DISK COLOR (pure radial temperature profile) ===
      // Shakura-Sunyaev thin disk: T ~ r^(-3/4)
      // Hotter near the black hole, cooler at edges
      const temperature = pow(normR.add(0.05), float(-0.75)).mul(uniforms.diskTemperature);

      // Interpolate between user-defined inner/outer colors based on temperature
      const colorMix = smoothstep(float(0.5), float(2.5), temperature);
      const diskColor = mix(uniforms.diskOuterColor, uniforms.diskInnerColor, colorMix);

      // Edge falloff - disk fades at boundaries
      const edgeFalloff = smoothstep(float(0.0), uniforms.diskEdgeSoftnessInner, normR)
        .mul(smoothstep(float(1.0), float(1.0).sub(uniforms.diskEdgeSoftnessOuter), normR));

      // === RING PATTERN ===
      // Creates stretched ring structures using 2D noise with radius-dependent twist
      // The twist creates elongated features, cartesian coords ensure seamless wrapping
      const ringOpacity = float(1.0).toVar('ringOpacity');

      If(uniforms.ringEnabled.greaterThan(0.5), () => {
        // Uniform rotation - keeps pattern structure stable (no winding)
        const rotation = time.mul(uniforms.diskRotationSpeed);

        // Static twist based on radius - creates the sheared/elongated appearance
        // This is what makes features look stretched, not the time animation
        const staticTwist = hitR.add(1.0).log().mul(uniforms.ringTwist);

        // Combined angle - uniform rotation + static twist
        const sampleAngle = hitAngle.add(rotation).add(staticTwist);

        // Cartesian coordinates for noise - naturally seamless
        const noiseX = hitR.mul(cos(sampleAngle));
        const noiseY = hitR.mul(sin(sampleAngle));

        // === DIFFERENTIAL ROTATION via Z-axis evolution ===
        // Inner disk moves through 3D noise faster than outer disk
        // This creates visual appearance of differential rotation
        // without pattern winding (steady-state solution)
        //
        // Physical motivation: Keplerian orbits have omega ~ r^(-3/2)
        // We use sqrt (alpha=0.5) for subtle visual effect
        const referenceRadius = uniforms.diskInnerRadius.add(uniforms.diskOuterRadius).mul(0.5);
        const clampedR = hitR.max(uniforms.diskInnerRadius.mul(0.5));
        const differentialFactor = pow(referenceRadius.div(clampedR), float(0.5));

        // Z coordinate: moves faster at inner radii
        // When diskDifferentialRotation = 0: z = 0, pure 2D noise (original behavior)
        // When diskDifferentialRotation > 0: differential z-evolution
        // noiseEvolutionSpeed controls how fast the noise pattern evolves in time
        const zCoord = time.mul(uniforms.diskRotationSpeed)
          .mul(uniforms.diskDifferentialRotation)
          .mul(uniforms.noiseEvolutionSpeed)
          .mul(differentialFactor);

        // Sample 3D FBM noise
        const noiseCoord = vec3(noiseX, noiseY, zCoord).mul(uniforms.ringScale);
        const ringNoise = fbm(noiseCoord);

        // Apply contrast, brightness, and sharpness
        const rawRing = ringNoise.mul(uniforms.ringContrast).add(uniforms.ringBrightness);
        ringOpacity.assign(pow(clamp(rawRing, float(0.0), float(1.0)), uniforms.ringSharpness));
      });

      const finalOpacity = ringOpacity;

      // Return vec4: rgb = disk color (with edge falloff and brightness), a = opacity
      const finalColor = diskColor.mul(edgeFalloff).mul(uniforms.diskBrightness);
      return vec4(finalColor, finalOpacity);
    });

    // ========================================================================
    // SECTION 6: MAIN RAYMARCHING SHADER
    // ========================================================================

    return Fn(() => {
      // === SCHWARZSCHILD PARAMETERS ===
      // rs = 2GM/c^2 (in geometric units where G=c=1, rs = 2M)
      const rs = uniforms.blackHoleMass.mul(2.0);

      // === CAMERA SETUP ===
      const uv = screenUV.sub(0.5).mul(2.0);
      const aspect = uniforms.resolution.x.div(uniforms.resolution.y);
      const screenPos = vec2(uv.x.mul(aspect), uv.y);

      const camPos = uniforms.cameraPosition;
      const camTarget = uniforms.cameraTarget;

      // Build camera coordinate system
      const camForward = normalize(camTarget.sub(camPos));
      const worldUp = vec3(0.0, 1.0, 0.0);
      const camRight = normalize(cross(worldUp, camForward));
      const camUp = cross(camForward, camRight);

      // Generate ray direction through pixel
      const fov = float(1.0);
      const rayDir = normalize(
        camForward.mul(fov)
          .add(camRight.mul(screenPos.x))
          .add(camUp.mul(screenPos.y))
      ).toVar('rayDir');

      // === INITIALIZE RAY STATE ===
      const rayPos = camPos.toVar('rayPos');

      // Note: Initial ray jitter was removed because it affects the gravitational
      // lensing path, causing background stars/nebula to flicker. Per-step jitter
      // (applied in the raymarching loop) is sufficient for anti-banding on the disk.

      // Accumulated color with alpha for blending
      const color = vec3(0.0, 0.0, 0.0).toVar('color');
      const alpha = float(0.0).toVar('alpha');

      // Ray status
      const escaped = float(0.0).toVar('escaped');
      const captured = float(0.0).toVar('captured');

      // Disk parameters
      const innerR = uniforms.diskInnerRadius;
      const outerR = uniforms.diskOuterRadius;

      // === RAYMARCHING LOOP ===
      // Trace ray through curved spacetime
      // Max iterations is hardcoded (shader requirement), but raySteps uniform controls early exit
      const iterCount = float(0.0).toVar('iterCount');
      Loop(512, () => {
        // Check iteration limit (allows dynamic control via UI)
        If(iterCount.greaterThanEqual(uniforms.raySteps), () => {
          escaped.assign(1.0);
          Break();
        });
        iterCount.addAssign(1.0);

        // Check if we've already terminated
        If(escaped.greaterThan(0.5).or(captured.greaterThan(0.5)).or(alpha.greaterThan(0.99)), () => {
          Break();
        });

        const r = length(rayPos);

        // === TERMINATION: CAPTURED BY BLACK HOLE ===
        If(r.lessThan(rs.mul(1.01)), () => {
          captured.assign(1.0);
          Break();
        });

        // === ADAPTIVE STEP SIZE ===
        // Three factors control step size for accurate sampling:
        //
        // 1. Black hole proximity: Smaller steps near event horizon
        //    for accurate geodesic integration
        //
        // 2. Disk proximity: Smaller steps near the disk plane (y≈0)
        //    to avoid stepping over thin disk regions
        //
        // 3. Disk thickness awareness: The inner disk is thinner than
        //    the outer disk, requiring proportionally smaller steps

        // Factor 1: Distance from event horizon
        const distFromHorizon = r.sub(rs);
        const horizonFactor = smoothstep(float(0.0), rs.mul(5.0), distFromHorizon)
          .mul(0.8).add(0.2);

        // Factor 2 & 3: Disk proximity with thickness awareness
        // Calculate horizontal distance (radius in disk plane)
        const rHoriz = sqrt(rayPos.x.mul(rayPos.x).add(rayPos.z.mul(rayPos.z)));

        // Check if we're within the disk's radial extent (with some margin)
        // Use smoothstep for soft boundary instead of hard boolean
        const diskMargin = float(2.0);
        const inDiskRegion = smoothstep(innerR.sub(diskMargin.mul(2.0)), innerR.sub(diskMargin), rHoriz)
          .mul(smoothstep(outerR.add(diskMargin.mul(2.0)), outerR.add(diskMargin), rHoriz));

        // Calculate local disk thickness at this radius
        const normRForThickness = clamp(
          rHoriz.sub(innerR).div(outerR.sub(innerR)),
          float(0.0), float(1.0)
        );
        const localThickness = mix(
          uniforms.diskInnerThickness,
          uniforms.diskOuterThickness,
          normRForThickness
        );

        // When approaching the disk plane, reduce step size
        // Scale step reduction based on distance to disk vs local thickness
        // At |y| = 3*thickness, factor = 1.0 (no reduction)
        // At |y| = 0, factor = adaptiveMinStep (minimum step for thin regions)
        const approachDistance = float(3.0);
        const distToPlane = abs(rayPos.y);
        const thicknessScale = localThickness.max(0.05); // Prevent division issues
        const diskProximity = distToPlane.div(thicknessScale.mul(approachDistance));
        const minStep = uniforms.adaptiveMinStep;
        const diskFactor = smoothstep(float(0.0), float(1.0), diskProximity)
          .mul(float(1.0).sub(minStep)).add(minStep);

        // Combine factors: use disk factor only when in disk region
        // inDiskRegion is already 0-1 from smoothstep, so use it directly
        const combinedDiskFactor = mix(float(1.0), diskFactor, inDiskRegion);
        const adaptiveStep = uniforms.stepSize.mul(horizonFactor).mul(combinedDiskFactor);

        // === GRAVITATIONAL LIGHT BENDING ===
        // Simplified geodesic: acceleration toward black hole
        // Based on Schwarzschild metric: a ≈ -rs/(2r^2) * r_hat
        const toCenter = rayPos.negate().normalize();
        const bendStrength = rs.div(r.mul(r)).mul(adaptiveStep).mul(uniforms.gravitationalLensing);

        // Apply bending to ray direction
        rayDir.addAssign(toCenter.mul(bendStrength));
        rayDir.assign(normalize(rayDir));

        // Step ray forward (deterministic - no jitter here to keep background stable)
        rayPos.addAssign(rayDir.mul(adaptiveStep));

        // === VOLUMETRIC DISK SAMPLING ===
        // Apply jitter to sample position (not ray path) to break up banding
        // This keeps the ray path deterministic for stable background stars
        const sampleNoise = hash33(rayPos.add(vec3(uniforms.frameIndex.mul(0.1))));
        const jitterOffset = sampleNoise.sub(0.5).mul(adaptiveStep).mul(uniforms.stepJitter);
        const samplePos = rayPos.add(jitterOffset);

        // Check if ray is inside the disk volume (using jittered sample position)
        const hitR = sqrt(samplePos.x.mul(samplePos.x).add(samplePos.z.mul(samplePos.z)));

        // Normalized radius for tapering (0 at inner, 1 at outer)
        const normR = clamp(hitR.sub(innerR).div(outerR.sub(innerR)), float(0.0), float(1.0));

        // === SOFT RADIAL FALLOFF (prevents sawblade artifacts) ===
        // Use smooth density falloff instead of hard cutoffs
        const radialFalloffWidth = uniforms.diskRadialFalloff;
        const innerFalloff = smoothstep(
          innerR.sub(radialFalloffWidth),
          innerR.add(radialFalloffWidth),
          hitR
        );
        const outerFalloff = smoothstep(
          outerR.add(radialFalloffWidth),
          outerR.sub(radialFalloffWidth),
          hitR
        );
        const radialDensity = innerFalloff.mul(outerFalloff);

        // === DISK THICKNESS PROFILE ===
        // Linearly interpolate thickness from inner to outer radius
        const innerHalf = uniforms.diskInnerThickness.mul(0.5);
        const outerHalf = uniforms.diskOuterThickness.mul(0.5);
        const localHalfThickness = mix(innerHalf, outerHalf, normR);

        // === SOFT HEIGHT FALLOFF ===
        // Smooth density falloff from center to edge of disk
        // heightRatio: 0 at disk midplane, 1 at disk surface, >1 outside
        const heightRatio = abs(samplePos.y).div(localHalfThickness.max(0.01));

        // Smooth, fast rolloff using smoothstep for clean edges
        // Density is 1 at midplane, falls to 0 at surface
        const heightDensity = smoothstep(float(1.0), float(0.0), heightRatio);

        // Combined density - no hard cutoffs
        const totalDensity = radialDensity.mul(heightDensity);

        // Only process if density is significant
        If(totalDensity.greaterThan(0.001).and(alpha.lessThan(0.99)), () => {
          const hitAngle = atan(samplePos.z, samplePos.x);

          // Get disk color and turbulence opacity at this point
          const diskResult = accretionDiskColor(hitR, hitAngle, uniforms.time);
          const diskCol = diskResult.xyz;
          const turbOpacity = diskResult.w;

          // === GRAVITATIONAL REDSHIFT ===
          // Light loses energy climbing out of gravity well
          const redshift = sqrt(clamp(float(1.0).sub(rs.div(hitR)), float(0.1), float(1.0)));

          // Volumetric accumulation - turbulence opacity affects both color and alpha
          const sampleDensity = totalDensity.mul(uniforms.diskDensity).mul(turbOpacity);
          const contribution = diskCol.mul(redshift).mul(sampleDensity);
          const remainingAlpha = float(1.0).sub(alpha);
          color.addAssign(contribution.mul(remainingAlpha));
          alpha.addAssign(remainingAlpha.mul(sampleDensity.mul(uniforms.diskOpacityFalloff)));
        });
      });

      // === BACKGROUND (for escaped rays) ===
      If(escaped.greaterThan(0.5).and(alpha.lessThan(0.99)), () => {
        const bgColor = uniforms.starBackgroundColor.toVar('bgColor');

        // Add stars if enabled
        If(uniforms.starsEnabled.greaterThan(0.5), () => {
          const stars = starField(rayDir);
          bgColor.addAssign(stars);
        });

        // Add nebula if enabled
        If(uniforms.nebulaEnabled.greaterThan(0.5), () => {
          const nebula = nebulaField(rayDir, uniforms.time);
          bgColor.addAssign(nebula);
        });

        // Blend background with accumulated disk color
        color.addAssign(bgColor.mul(float(1.0).sub(alpha)));
      });

      // === TONE MAPPING (ACES Filmic) ===
      const a = float(2.51);
      const b = float(0.03);
      const c = float(2.43);
      const d = float(0.59);
      const e = float(0.14);

      const toneMapped = clamp(
        color.mul(color.mul(a).add(b))
          .div(color.mul(color.mul(c).add(d)).add(e)),
        vec3(0.0),
        vec3(1.0)
      );

      // === GAMMA CORRECTION ===
      const finalColor = pow(toneMapped, vec3(1.0 / 2.2));

      return vec4(finalColor, 1.0);
    })();
  }

  // ==========================================================================
  // SECTION 7: PUBLIC API
  // ==========================================================================

  /**
   * Update uniform values from config object.
   * Called when UI controls change.
   */
  updateUniforms(config) {
    const u = this.uniforms;

    // Physics
    if (config.blackHoleMass !== undefined) u.blackHoleMass.value = config.blackHoleMass;

    // Disk geometry
    if (config.diskInnerRadius !== undefined) u.diskInnerRadius.value = config.diskInnerRadius;
    if (config.diskOuterRadius !== undefined) u.diskOuterRadius.value = config.diskOuterRadius;
    if (config.diskInnerThickness !== undefined) u.diskInnerThickness.value = config.diskInnerThickness;
    if (config.diskOuterThickness !== undefined) u.diskOuterThickness.value = config.diskOuterThickness;

    // Disk appearance
    if (config.diskTemperature !== undefined) u.diskTemperature.value = config.diskTemperature;
    if (config.diskBrightness !== undefined) u.diskBrightness.value = config.diskBrightness;
    if (config.diskRotationSpeed !== undefined) u.diskRotationSpeed.value = config.diskRotationSpeed;

    // Ring pattern
    if (config.ringEnabled !== undefined) u.ringEnabled.value = config.ringEnabled ? 1.0 : 0.0;
    if (config.ringScale !== undefined) u.ringScale.value = config.ringScale;
    if (config.ringContrast !== undefined) u.ringContrast.value = config.ringContrast;
    if (config.ringBrightness !== undefined) u.ringBrightness.value = config.ringBrightness;
    if (config.ringSharpness !== undefined) u.ringSharpness.value = config.ringSharpness;
    if (config.ringTwist !== undefined) u.ringTwist.value = config.ringTwist;
    if (config.noiseEvolutionSpeed !== undefined) u.noiseEvolutionSpeed.value = config.noiseEvolutionSpeed;

    // Disk edge falloff
    if (config.diskEdgeSoftnessInner !== undefined) u.diskEdgeSoftnessInner.value = config.diskEdgeSoftnessInner;
    if (config.diskEdgeSoftnessOuter !== undefined) u.diskEdgeSoftnessOuter.value = config.diskEdgeSoftnessOuter;
    if (config.diskRadialFalloff !== undefined) u.diskRadialFalloff.value = config.diskRadialFalloff;

    // Relativistic effects
    if (config.gravitationalLensing !== undefined) u.gravitationalLensing.value = config.gravitationalLensing;

    // Volumetric rendering
    if (config.diskDensity !== undefined) u.diskDensity.value = config.diskDensity;
    if (config.diskOpacityFalloff !== undefined) u.diskOpacityFalloff.value = config.diskOpacityFalloff;

    // Performance
    if (config.raySteps !== undefined) u.raySteps.value = config.raySteps;
    if (config.stepSize !== undefined) u.stepSize.value = config.stepSize;
    if (config.adaptiveMinStep !== undefined) u.adaptiveMinStep.value = config.adaptiveMinStep;

    // Anti-aliasing
    if (config.rayJitter !== undefined) u.rayJitter.value = config.rayJitter;
    if (config.stepJitter !== undefined) u.stepJitter.value = config.stepJitter;
    if (config.frameIndex !== undefined) u.frameIndex.value = config.frameIndex;

    // Star uniforms
    if (config.starsEnabled !== undefined) u.starsEnabled.value = config.starsEnabled ? 1.0 : 0.0;
    if (config.starBackgroundColor !== undefined) u.starBackgroundColor.value.set(config.starBackgroundColor);
    if (config.starDensity !== undefined) u.starDensity.value = config.starDensity;
    if (config.starSize !== undefined) u.starSize.value = config.starSize;
    if (config.starBrightness !== undefined) u.starBrightness.value = config.starBrightness;

    // Nebula uniforms
    if (config.nebulaEnabled !== undefined) u.nebulaEnabled.value = config.nebulaEnabled ? 1.0 : 0.0;
    if (config.nebulaBrightness !== undefined) u.nebulaBrightness.value = config.nebulaBrightness;
    if (config.nebulaScale1 !== undefined) u.nebulaScale1.value = config.nebulaScale1;
    if (config.nebulaScale2 !== undefined) u.nebulaScale2.value = config.nebulaScale2;
    if (config.nebulaBlend !== undefined) u.nebulaBlend.value = config.nebulaBlend;
    if (config.nebulaSpeed !== undefined) u.nebulaSpeed.value = config.nebulaSpeed;
    if (config.nebulaDensity !== undefined) u.nebulaDensity.value = config.nebulaDensity;

    // Color uniforms
    if (config.diskInnerColor !== undefined) {
      u.diskInnerColor.value.set(config.diskInnerColor);
    }
    if (config.diskOuterColor !== undefined) {
      u.diskOuterColor.value.set(config.diskOuterColor);
    }
    if (config.nebulaColor1 !== undefined) {
      u.nebulaColor1.value.set(config.nebulaColor1);
    }
    if (config.nebulaColor2 !== undefined) {
      u.nebulaColor2.value.set(config.nebulaColor2);
    }
  }

  /**
   * Apply a quality preset.
   */
  applyQualityPreset(presetName) {
    const preset = QUALITY_PRESETS[presetName];
    if (!preset) return;

    this.updateUniforms({
      raySteps: preset.raySteps,
      stepSize: preset.stepSize,
      starsEnabled: preset.starsEnabled,
      nebulaEnabled: preset.nebulaEnabled,
      rayJitter: preset.rayJitter,
      stepJitter: preset.stepJitter
    });
  }

  /**
   * Update camera position uniform from Three.js camera.
   */
  updateCamera(camera) {
    this.uniforms.cameraPosition.value.copy(camera.position);

    // Calculate camera target from view direction
    const direction = new THREE.Vector3(0, 0, -1);
    direction.applyQuaternion(camera.quaternion);
    const target = camera.position.clone().add(direction.multiplyScalar(10));
    this.uniforms.cameraTarget.value.copy(target);
  }

  /**
   * Main update method - called each frame.
   */
  update(deltaTime, camera) {
    this.uniforms.time.value += deltaTime;
    this.updateCamera(camera);
  }

  /**
   * Handle window resize.
   */
  onResize(width, height) {
    this.uniforms.resolution.value.set(width, height);
  }

  /**
   * Regenerate the black hole mesh (e.g., after config changes).
   */
  regenerate() {
    this.createBlackHole();
  }
}
