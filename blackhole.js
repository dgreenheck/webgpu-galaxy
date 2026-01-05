/**
 * Black Hole Simulation with Ray-Traced Gravitational Lensing
 *
 * Features implemented:
 * - Gravitational lensing using Schwarzschild geodesics
 * - Accretion disk with temperature-based coloring
 * - Doppler beaming (relativistic brightness variation)
 * - Photon ring from multiple disk intersections
 * - Black hole shadow
 * - Gravitational redshift
 * - Image of disk's far side and underside via lensing
 */

import * as THREE from 'three/webgpu';
import {
  uniform,
  attribute,
  vec2,
  vec3,
  vec4,
  float,
  int,
  Fn,
  mix,
  length,
  normalize,
  cross,
  dot,
  sin,
  cos,
  atan2,
  sqrt,
  abs,
  max,
  min,
  pow,
  exp,
  floor,
  fract,
  clamp,
  smoothstep,
  step,
  Loop,
  Break,
  Continue,
  If,
  uv
} from 'three/tsl';

// ==============================================================================
// BLACK HOLE SIMULATION CLASS
// ==============================================================================

export class BlackHoleSimulation {
  constructor(scene, config) {
    this.scene = scene;
    this.config = config;

    // Scene objects
    this.blackHoleMesh = null;

    // Initialize uniforms
    this.initializeUniforms(config);
  }

  /**
   * Initialize all shader uniforms
   */
  initializeUniforms(config) {
    this.uniforms = {
      // Black hole physics
      blackHoleMass: uniform(config.blackHoleMass || 1.0),

      // Accretion disk parameters
      diskInnerRadius: uniform(config.diskInnerRadius || 2.6),
      diskOuterRadius: uniform(config.diskOuterRadius || 12.0),

      // Visual parameters
      diskTemperature: uniform(config.diskTemperature || 1.5),
      diskBrightness: uniform(config.diskBrightness || 2.0),
      dopplerStrength: uniform(config.dopplerStrength || 0.8),

      // Camera/view
      time: uniform(0),
      resolution: uniform(new THREE.Vector2(window.innerWidth, window.innerHeight)),
      cameraPos: uniform(new THREE.Vector3(0, 8, 20)),
      cameraTarget: uniform(new THREE.Vector3(0, 0, 0))
    };
  }

  /**
   * Creates the black hole visualization using a full-screen shader
   */
  createBlackHole() {
    // Clean up old mesh
    if (this.blackHoleMesh) {
      this.scene.remove(this.blackHoleMesh);
      if (this.blackHoleMesh.material) {
        this.blackHoleMesh.material.dispose();
      }
      if (this.blackHoleMesh.geometry) {
        this.blackHoleMesh.geometry.dispose();
      }
    }

    // Create a large sphere that will act as our render surface
    // Using a sphere ensures we can look around the black hole
    const geometry = new THREE.SphereGeometry(100, 64, 64);

    // Flip normals to render inside
    geometry.scale(-1, 1, 1);

    // Create custom shader material using TSL
    const material = new THREE.MeshBasicNodeMaterial();
    material.side = THREE.BackSide;

    // Ray-traced black hole shader
    const blackHoleShader = this.createBlackHoleShader();
    material.colorNode = blackHoleShader;

    this.blackHoleMesh = new THREE.Mesh(geometry, material);
    this.blackHoleMesh.frustumCulled = false;

    this.scene.add(this.blackHoleMesh);
  }

  /**
   * Creates the main ray-tracing shader for the black hole
   * Implements gravitational lensing with multiple disk crossings
   */
  createBlackHoleShader() {
    const uniforms = this.uniforms;

    // Blackbody color function (approximation)
    const blackbodyColor = Fn(([temperature]) => {
      const t = clamp(temperature, float(0.1), float(4.0));

      // Hot = blue-white, medium = yellow-orange, cool = red
      // Based on Wien's displacement law approximation
      const red = clamp(
        float(1.0).sub(exp(t.sub(1.5).mul(-2.0))).add(
          exp(t.sub(0.5).mul(-3.0)).mul(0.5)
        ),
        float(0.0),
        float(1.0)
      );

      const green = clamp(
        exp(t.sub(1.2).pow(2.0).mul(-2.0)).mul(0.9).add(
          smoothstep(float(2.0), float(4.0), t).mul(0.8)
        ),
        float(0.0),
        float(1.0)
      );

      const blue = clamp(
        smoothstep(float(1.5), float(3.0), t).mul(1.2),
        float(0.0),
        float(1.0)
      );

      return vec3(red, green, blue);
    });

    // Main shader function
    return Fn(() => {
      // Schwarzschild radius (event horizon) rs = 2GM/c² (using c=1, G=1)
      const rs = uniforms.blackHoleMass.mul(2.0);

      // Get UV coordinates and convert to ray direction
      const screenUV = uv().sub(0.5).mul(2.0);
      const aspect = uniforms.resolution.x.div(uniforms.resolution.y);
      const adjustedUV = vec2(screenUV.x.mul(aspect), screenUV.y);

      // Camera setup
      const camPos = uniforms.cameraPos;
      const camTarget = uniforms.cameraTarget;

      // Build camera coordinate system
      const camForward = normalize(camTarget.sub(camPos));
      const worldUp = vec3(0, 1, 0);
      const camRight = normalize(cross(worldUp, camForward));
      const camUp = cross(camForward, camRight);

      // Ray direction from camera through pixel
      const fov = float(1.2);
      const rayDir = normalize(
        camForward.mul(fov)
          .add(camRight.mul(adjustedUV.x))
          .add(camUp.mul(adjustedUV.y))
      );

      // Initialize ray state
      const rayPos = camPos.toVar('rayPos');
      const rayVel = rayDir.toVar('rayVel');

      // Accumulated color
      const accumulatedColor = vec3(0.0, 0.0, 0.0).toVar('accColor');
      const accumulatedAlpha = float(0.0).toVar('accAlpha');

      // Track previous Y for disk crossing detection
      const prevY = rayPos.y.toVar('prevY');

      // Track if ray is still active
      const rayActive = float(1.0).toVar('rayActive');

      // Ray marching with gravitational deflection
      Loop(300, () => {
        // Early exit if ray is no longer active
        If(rayActive.lessThan(0.5), () => {
          Break();
        });

        const r = length(rayPos);

        // Check if ray fell into black hole (inside event horizon)
        If(r.lessThan(rs.mul(1.01)), () => {
          // Black hole - no light escapes
          rayActive.assign(0.0);
          Break();
        });

        // Check if ray escaped to infinity
        If(r.greaterThan(60.0), () => {
          // Add subtle background stars
          const starNoise = fract(sin(dot(rayDir.xz, vec2(12.9898, 78.233))).mul(43758.5453));
          const star = step(float(0.998), starNoise).mul(0.3);
          accumulatedColor.addAssign(vec3(star, star, star.mul(1.2)));
          rayActive.assign(0.0);
          Break();
        });

        // Gravitational bending of light (Schwarzschild geodesic approximation)
        // Using the effective potential for light: d²u/dφ² + u = 3GMu²/c² (with u = 1/r)
        // This creates the gravitational lensing effect
        const hVec = cross(rayPos, rayVel);
        const h2 = dot(hVec, hVec); // Angular momentum squared

        // Acceleration towards black hole with relativistic correction
        // a = -GM/r² * r_hat + correction for light bending
        const rNorm = normalize(rayPos);
        const baseAccel = rs.div(r.mul(r)).mul(0.5);

        // Relativistic correction term (stronger near photon sphere at 1.5rs)
        const correction = rs.mul(h2).mul(1.5).div(r.mul(r).mul(r).mul(r));

        const accelMag = baseAccel.add(correction);
        const accel = rNorm.mul(accelMag.negate());

        // Adaptive step size (smaller steps closer to black hole)
        const adaptiveStep = float(0.15).mul(
          clamp(r.div(rs.mul(3.0)), float(0.3), float(2.0))
        );

        // Velocity Verlet integration
        const halfAccel = accel.mul(adaptiveStep.mul(0.5));
        rayVel.addAssign(halfAccel);
        rayPos.addAssign(rayVel.mul(adaptiveStep));
        rayVel.addAssign(halfAccel);

        // Renormalize velocity (light always travels at c=1)
        rayVel.assign(normalize(rayVel));

        // Check for disk intersection (disk is at y=0 plane)
        const currY = rayPos.y;
        const crossed = prevY.mul(currY).lessThan(0.0);

        If(crossed, () => {
          // Calculate crossing point via linear interpolation
          const tCross = abs(prevY).div(abs(prevY).add(abs(currY)).max(0.0001));
          const crossX = rayPos.x.sub(rayVel.x.mul(adaptiveStep.mul(float(1.0).sub(tCross))));
          const crossZ = rayPos.z.sub(rayVel.z.mul(adaptiveStep.mul(float(1.0).sub(tCross))));
          const crossR = sqrt(crossX.mul(crossX).add(crossZ.mul(crossZ)));

          // Check if within disk bounds
          const innerR = uniforms.diskInnerRadius;
          const outerR = uniforms.diskOuterRadius;

          If(crossR.greaterThan(innerR).and(crossR.lessThan(outerR)), () => {
            // Calculate disk emission at this point

            // Normalized radius (0 at inner edge, 1 at outer edge)
            const normR = crossR.sub(innerR).div(outerR.sub(innerR));

            // Temperature profile: T ~ r^(-3/4) for standard thin disk
            // Hottest near inner edge
            const temp = pow(normR.add(0.05), float(-0.75)).mul(uniforms.diskTemperature);

            // Get base color from temperature
            const baseColor = blackbodyColor(temp);

            // === DOPPLER BEAMING ===
            // Disk rotates counter-clockwise when viewed from above
            // Material moving towards us appears brighter (blue-shifted)
            // Material moving away appears dimmer (red-shifted)
            const diskAngle = atan2(crossZ, crossX);
            const rotPhase = diskAngle.add(uniforms.time.mul(0.3));

            // Keplerian orbital velocity: v = sqrt(GM/r)
            // For this visualization, we use a simplified model
            const orbitalSpeed = sqrt(uniforms.blackHoleMass.div(crossR)).mul(0.5);

            // Disk velocity direction (perpendicular to radius, in rotation direction)
            const diskVelX = sin(diskAngle).negate().mul(orbitalSpeed);
            const diskVelZ = cos(diskAngle).mul(orbitalSpeed);
            const diskVel = vec3(diskVelX, 0.0, diskVelZ);

            // View direction (where light is going)
            const viewDir = rayVel.negate();

            // Doppler factor: D = 1 / (γ(1 - β·n))
            // For relativistic beaming, intensity ~ D^3 (or D^4 for spectral)
            const beta = dot(diskVel, viewDir).mul(uniforms.dopplerStrength);
            const dopplerBoost = pow(float(1.0).add(beta).max(0.1), float(3.0));

            // === GRAVITATIONAL REDSHIFT ===
            // Light loses energy climbing out of gravity well
            // Redshift factor = sqrt(1 - rs/r)
            const gravRedshift = sqrt(float(1.0).sub(rs.div(crossR)).max(0.01));

            // === LIMB DARKENING ===
            // Edge of disk appears slightly dimmer
            const limbDark = sqrt(float(1.0).sub(normR.mul(0.3)));

            // === PHOTON RING ENHANCEMENT ===
            // Light that has orbited the black hole creates bright thin rings
            // These appear very close to the photon sphere projection
            const distFromCenter = crossR.div(rs.mul(1.5)); // Distance from photon sphere
            const nearPhotonSphere = smoothstep(float(1.0), float(1.8), distFromCenter)
              .mul(smoothstep(float(3.0), float(2.0), distFromCenter));

            // Ring structure from multiple images
            const ringBoost = float(1.0).add(nearPhotonSphere.mul(0.5));

            // Combine all effects
            const brightness = uniforms.diskBrightness
              .mul(dopplerBoost)
              .mul(gravRedshift)
              .mul(limbDark)
              .mul(ringBoost)
              .mul(clamp(temp.mul(0.7), float(0.5), float(3.0)));

            // Add noise for turbulence in the disk
            const turbulence = fract(
              sin(crossX.mul(12.9898).add(crossZ.mul(78.233)).add(uniforms.time.mul(2.0)))
                .mul(43758.5453)
            ).mul(0.15).add(0.92);

            const diskEmission = baseColor.mul(brightness).mul(turbulence);

            // Accumulate with diminishing contribution for secondary images
            const contribution = float(1.0).sub(accumulatedAlpha).mul(0.9);
            accumulatedColor.addAssign(diskEmission.mul(contribution));
            accumulatedAlpha.addAssign(contribution.mul(0.7));

            // Stop if we've accumulated enough
            If(accumulatedAlpha.greaterThan(0.95), () => {
              rayActive.assign(0.0);
            });
          });
        });

        prevY.assign(currY);
      });

      // Apply tone mapping (ACES-like)
      const a = float(2.51);
      const b = float(0.03);
      const c = float(2.43);
      const d = float(0.59);
      const e = float(0.14);

      const toneMapped = clamp(
        accumulatedColor.mul(accumulatedColor.mul(a).add(b))
          .div(accumulatedColor.mul(accumulatedColor.mul(c).add(d)).add(e)),
        vec3(0.0),
        vec3(1.0)
      );

      // Gamma correction
      const gamma = vec3(1.0 / 2.2);
      const finalColor = pow(toneMapped, gamma);

      return vec4(finalColor, 1.0);
    })();
  }

  /**
   * Update uniforms each frame
   */
  updateUniforms(configUpdate) {
    if (configUpdate.blackHoleMass !== undefined)
      this.uniforms.blackHoleMass.value = configUpdate.blackHoleMass;
    if (configUpdate.diskInnerRadius !== undefined)
      this.uniforms.diskInnerRadius.value = configUpdate.diskInnerRadius;
    if (configUpdate.diskOuterRadius !== undefined)
      this.uniforms.diskOuterRadius.value = configUpdate.diskOuterRadius;
    if (configUpdate.diskTemperature !== undefined)
      this.uniforms.diskTemperature.value = configUpdate.diskTemperature;
    if (configUpdate.diskBrightness !== undefined)
      this.uniforms.diskBrightness.value = configUpdate.diskBrightness;
    if (configUpdate.dopplerStrength !== undefined)
      this.uniforms.dopplerStrength.value = configUpdate.dopplerStrength;
  }

  /**
   * Update camera position from Three.js camera
   */
  updateCamera(camera) {
    this.uniforms.cameraPos.value.copy(camera.position);

    // Get camera look-at direction
    const dir = new THREE.Vector3(0, 0, -1);
    dir.applyQuaternion(camera.quaternion);

    // Set target as point along view direction
    const target = camera.position.clone().add(dir.multiplyScalar(10));
    this.uniforms.cameraTarget.value.copy(target);
  }

  /**
   * Main update loop
   */
  update(renderer, deltaTime, camera) {
    this.uniforms.time.value += deltaTime;
    this.updateCamera(camera);
  }

  /**
   * Handle window resize
   */
  onResize(width, height) {
    this.uniforms.resolution.value.set(width, height);
  }

  /**
   * Regenerate the black hole
   */
  regenerate() {
    this.createBlackHole();
  }
}
