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
  fract,
  clamp,
  smoothstep,
  mix,
  sign,
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
    nebulaEnabled: false
  },
  medium: {
    raySteps: 100,
    stepSize: 0.3,
    diskDetail: 2,
    starsEnabled: true,
    nebulaEnabled: false
  },
  high: {
    raySteps: 150,
    stepSize: 0.2,
    diskDetail: 3,
    starsEnabled: true,
    nebulaEnabled: true
  },
  ultra: {
    raySteps: 256,
    stepSize: 0.15,
    diskDetail: 4,
    starsEnabled: true,
    nebulaEnabled: true
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
      diskTurbulence: uniform(config.diskTurbulence ?? 0.5),
      diskRingCount: uniform(config.diskRingCount ?? 8.0),
      diskRotationSpeed: uniform(config.diskRotationSpeed ?? 0.3),

      // === Disk Color (User Configurable) ===
      diskInnerColor: uniform(new THREE.Color(config.diskInnerColor ?? '#ffffee')),
      diskOuterColor: uniform(new THREE.Color(config.diskOuterColor ?? '#ff4400')),

      // === Relativistic Effects ===
      dopplerStrength: uniform(config.dopplerStrength ?? 0.8),
      photonRingIntensity: uniform(config.photonRingIntensity ?? 1.0),

      // === Performance ===
      raySteps: uniform(config.raySteps ?? 100),
      stepSize: uniform(config.stepSize ?? 0.3),

      // === Background ===
      starsEnabled: uniform(config.starsEnabled ? 1.0 : 0.0),
      starDensity: uniform(config.starDensity ?? 0.003),
      nebulaEnabled: uniform(config.nebulaEnabled ? 1.0 : 0.0),
      nebulaBrightness: uniform(config.nebulaBrightness ?? 0.15),

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
     * 1D hash function for noise
     */
    const hash11 = Fn(([p]) => {
      const n = fract(sin(p.mul(127.1)).mul(43758.5453));
      return n;
    });

    /**
     * 1D value noise for radial band variation
     */
    const noise1D = Fn(([p]) => {
      const i = floor(p);
      const f = fract(p);
      // Smooth interpolation
      const u = f.mul(f).mul(float(3.0).sub(f.mul(2.0)));
      return mix(hash11(i), hash11(i.add(1.0)), u);
    });

    /**
     * Multi-octave 1D noise for irregular band patterns
     * Creates varying line widths - some thin, some thick
     */
    const irregularBands = Fn(([r, scale]) => {
      const value = float(0.0).toVar();
      const amplitude = float(1.0).toVar();
      const frequency = float(1.0).toVar();
      const pos = r.mul(scale).toVar();

      // Multiple octaves with different frequencies for varied line widths
      // High frequency = fine lines, low frequency = thick bands
      Loop(6, () => {
        value.addAssign(noise1D(pos.mul(frequency)).mul(amplitude));
        frequency.mulAssign(2.17); // Non-integer for less repetition
        amplitude.mulAssign(0.5);
      });

      return value;
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

      // Create grid cells
      const gridScale = float(50.0);
      const cell = vec2(theta, phi).mul(gridScale).floor();
      const cellUV = fract(vec2(theta, phi).mul(gridScale));

      // Hash for this cell
      const cellHash = hash21(cell);

      // Star probability - most cells are empty
      const starProb = step(float(1.0).sub(uniforms.starDensity), cellHash);

      // Star position within cell
      const starPos = hash33(vec3(cell.x, cell.y, float(42.0))).xy.mul(0.8).add(0.1);
      const distToStar = length(cellUV.sub(starPos));

      // Star brightness with size variation
      const starSize = hash21(cell.add(100.0)).mul(0.015).add(0.005);
      const starBrightness = smoothstep(starSize, float(0.0), distToStar).mul(starProb);

      // Star color variation (blue to yellow)
      const colorTemp = hash21(cell.add(200.0));
      const starColor = mix(
        vec3(0.8, 0.9, 1.0),  // Blue-white
        vec3(1.0, 0.95, 0.8), // Yellow-white
        colorTemp
      );

      return starColor.mul(starBrightness).mul(0.8);
    });

    /**
     * Generate procedural nebula clouds.
     * Uses layered 3D noise for volumetric appearance.
     */
    const nebulaField = Fn(([rayDir, time]) => {
      const noisePos = rayDir.mul(2.0);

      // Multiple noise layers for depth
      const n1 = fbm(noisePos.add(time.mul(0.01)));
      const n2 = fbm(noisePos.mul(2.0).sub(time.mul(0.005)));

      // Combine noise layers
      const nebula = n1.mul(n2).mul(2.0);

      // Color gradient (purple/red/blue nebula colors)
      const nebulaColor = mix(
        vec3(0.1, 0.0, 0.2),  // Deep purple
        vec3(0.3, 0.1, 0.15), // Dusty red
        n1
      ).add(vec3(0.05, 0.05, 0.1).mul(n2));

      return nebulaColor.mul(nebula).mul(uniforms.nebulaBrightness);
    });

    // ========================================================================
    // SECTION 5: ACCRETION DISK
    // ========================================================================

    /**
     * Calculate the color and intensity of the accretion disk at a given point.
     * Implements temperature-based coloring with irregular radial bands.
     */
    const accretionDiskColor = Fn(([hitR, hitAngle, time]) => {
      const innerR = uniforms.diskInnerRadius;
      const outerR = uniforms.diskOuterRadius;

      // Normalized radius (0 at inner edge, 1 at outer edge)
      const normR = clamp(hitR.sub(innerR).div(outerR.sub(innerR)), float(0.0), float(1.0));

      // === IRREGULAR RADIAL BANDS ===
      // Use multi-octave 1D noise for natural-looking irregular bands
      // The bands should vary in width - some thin lines, some thick bands
      const bandScale = uniforms.diskRingCount.mul(3.0);

      // Multiple layers of noise at different scales for band variation
      const bands1 = irregularBands(hitR, bandScale);
      const bands2 = irregularBands(hitR.add(17.3), bandScale.mul(1.7));
      const bands3 = irregularBands(hitR.add(31.7), bandScale.mul(0.5));

      // Create sharp transitions for distinct band edges
      // Use pow to sharpen the bands and create more defined lines
      const sharpBands = pow(bands1.mul(0.5).add(0.5), float(2.0));

      // Combine bands with variation in intensity
      const bandIntensity = sharpBands.mul(bands2.mul(0.3).add(0.7)).mul(bands3.mul(0.4).add(0.6));

      // Add some extra fine detail bands
      const fineDetail = noise1D(hitR.mul(bandScale).mul(8.0)).mul(0.15).add(0.85);

      // Final ring pattern combines all band layers
      const ringPattern = bandIntensity.mul(fineDetail);

      // === TURBULENCE ===
      // Swirling patterns using noise (reduced to keep bands prominent)
      const turbCoord = vec3(
        cos(hitAngle).mul(hitR.mul(0.3)),
        sin(hitAngle).mul(hitR.mul(0.3)),
        time.mul(0.1)
      );
      const turb = fbm(turbCoord).mul(uniforms.diskTurbulence).mul(0.5);

      // === TEMPERATURE PROFILE ===
      // Shakura-Sunyaev thin disk: T ~ r^(-3/4)
      // Hotter near the black hole, cooler at edges
      const baseTemp = pow(normR.add(0.05), float(-0.75)).mul(uniforms.diskTemperature);
      const tempVariation = ringPattern.mul(0.2).add(turb.mul(0.1));
      const finalTemp = clamp(baseTemp.mul(float(0.8).add(tempVariation)), float(0.3), float(4.0));

      // === COLOR FROM TEMPERATURE ===
      // Interpolate between user-defined inner/outer colors based on temperature
      const colorMix = smoothstep(float(0.5), float(2.5), finalTemp);
      const baseColor = mix(uniforms.diskOuterColor, uniforms.diskInnerColor, colorMix);

      // === INTENSITY MODULATION ===
      const intensity = ringPattern.mul(float(0.9).add(turb.mul(0.2)));

      // Edge falloff - disk fades at boundaries
      const edgeFalloff = smoothstep(float(0.0), float(0.1), normR)
        .mul(smoothstep(float(1.0), float(0.9), normR));

      return baseColor.mul(intensity).mul(edgeFalloff).mul(uniforms.diskBrightness);
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
      const totalDist = float(0.0).toVar('totalDist');
      const maxDist = float(100.0);

      // Accumulated color with alpha for blending
      const color = vec3(0.0, 0.0, 0.0).toVar('color');
      const alpha = float(0.0).toVar('alpha');

      // Track previous position for disk intersection
      const prevY = rayPos.y.toVar('prevY');

      // Ray status
      const escaped = float(0.0).toVar('escaped');
      const captured = float(0.0).toVar('captured');

      // Disk parameters
      const innerR = uniforms.diskInnerRadius;
      const outerR = uniforms.diskOuterRadius;

      // === RAYMARCHING LOOP ===
      // Trace ray through curved spacetime
      Loop(256, () => {
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

        // === TERMINATION: ESCAPED TO INFINITY ===
        If(totalDist.greaterThan(maxDist), () => {
          escaped.assign(1.0);
          Break();
        });

        // === ADAPTIVE STEP SIZE ===
        // Smaller steps near the black hole for accuracy
        const distFromHorizon = r.sub(rs);
        const adaptiveStep = uniforms.stepSize.mul(
          smoothstep(float(0.0), rs.mul(5.0), distFromHorizon)
            .mul(0.8).add(0.2)
        );

        // === GRAVITATIONAL LIGHT BENDING ===
        // Simplified geodesic: acceleration toward black hole
        // Based on Schwarzschild metric: a ≈ -rs/(2r^2) * r_hat
        const toCenter = rayPos.negate().normalize();
        const bendStrength = rs.div(r.mul(r)).mul(adaptiveStep).mul(1.5);

        // Apply bending to ray direction
        rayDir.addAssign(toCenter.mul(bendStrength));
        rayDir.assign(normalize(rayDir));

        // Store previous Y for disk intersection detection
        prevY.assign(rayPos.y);

        // Step ray forward
        rayPos.addAssign(rayDir.mul(adaptiveStep));
        totalDist.addAssign(adaptiveStep);

        // === DISK INTERSECTION DETECTION ===
        // Check if ray crossed the disk plane (y = 0)
        const currY = rayPos.y;
        const crossed = sign(prevY).notEqual(sign(currY));

        If(crossed.and(alpha.lessThan(0.95)), () => {
          // Interpolate exact crossing point
          const t = abs(prevY).div(abs(prevY).add(abs(currY)).max(0.0001));
          const hitX = rayPos.x.sub(rayDir.x.mul(adaptiveStep.mul(float(1.0).sub(t))));
          const hitZ = rayPos.z.sub(rayDir.z.mul(adaptiveStep.mul(float(1.0).sub(t))));
          const hitR = sqrt(hitX.mul(hitX).add(hitZ.mul(hitZ)));

          // Check if within disk bounds
          If(hitR.greaterThan(innerR).and(hitR.lessThan(outerR)), () => {
            const hitAngle = atan(hitZ, hitX);

            // Get disk color at this point
            const diskCol = accretionDiskColor(hitR, hitAngle, uniforms.time);

            // === DOPPLER BEAMING ===
            // Material orbits the black hole - approaching side is brighter
            const orbitalSpeed = sqrt(uniforms.blackHoleMass.div(hitR)).mul(0.4);
            const velDir = vec3(sin(hitAngle).negate(), float(0.0), cos(hitAngle));
            const dopplerFactor = float(1.0).add(
              dot(velDir, rayDir.negate()).mul(orbitalSpeed).mul(uniforms.dopplerStrength)
            );
            const doppler = pow(clamp(dopplerFactor, float(0.5), float(2.0)), float(3.0));

            // === GRAVITATIONAL REDSHIFT ===
            // Light loses energy climbing out of gravity well
            const redshift = sqrt(clamp(float(1.0).sub(rs.div(hitR)), float(0.1), float(1.0)));

            // Accumulate color with alpha blending
            const contribution = diskCol.mul(doppler).mul(redshift);
            const remainingAlpha = float(1.0).sub(alpha);
            color.addAssign(contribution.mul(remainingAlpha));
            alpha.addAssign(remainingAlpha.mul(0.85));
          });
        });
      });

      // === BACKGROUND (for escaped rays) ===
      If(escaped.greaterThan(0.5).and(alpha.lessThan(0.99)), () => {
        const bgColor = vec3(0.0, 0.0, 0.0).toVar('bgColor');

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

    if (config.blackHoleMass !== undefined) u.blackHoleMass.value = config.blackHoleMass;
    if (config.diskInnerRadius !== undefined) u.diskInnerRadius.value = config.diskInnerRadius;
    if (config.diskOuterRadius !== undefined) u.diskOuterRadius.value = config.diskOuterRadius;
    if (config.diskTemperature !== undefined) u.diskTemperature.value = config.diskTemperature;
    if (config.diskBrightness !== undefined) u.diskBrightness.value = config.diskBrightness;
    if (config.diskTurbulence !== undefined) u.diskTurbulence.value = config.diskTurbulence;
    if (config.diskRingCount !== undefined) u.diskRingCount.value = config.diskRingCount;
    if (config.diskRotationSpeed !== undefined) u.diskRotationSpeed.value = config.diskRotationSpeed;
    if (config.dopplerStrength !== undefined) u.dopplerStrength.value = config.dopplerStrength;
    if (config.photonRingIntensity !== undefined) u.photonRingIntensity.value = config.photonRingIntensity;
    if (config.raySteps !== undefined) u.raySteps.value = config.raySteps;
    if (config.stepSize !== undefined) u.stepSize.value = config.stepSize;
    if (config.starsEnabled !== undefined) u.starsEnabled.value = config.starsEnabled ? 1.0 : 0.0;
    if (config.starDensity !== undefined) u.starDensity.value = config.starDensity;
    if (config.nebulaEnabled !== undefined) u.nebulaEnabled.value = config.nebulaEnabled ? 1.0 : 0.0;
    if (config.nebulaBrightness !== undefined) u.nebulaBrightness.value = config.nebulaBrightness;

    // Color uniforms
    if (config.diskInnerColor !== undefined) {
      u.diskInnerColor.value.set(config.diskInnerColor);
    }
    if (config.diskOuterColor !== undefined) {
      u.diskOuterColor.value.set(config.diskOuterColor);
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
      nebulaEnabled: preset.nebulaEnabled
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
