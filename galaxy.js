/**
 * Galaxy Simulation - Main Orchestrator
 *
 * This module contains the main GalaxySimulation class that manages:
 * - Uniform initialization and updates
 * - Star particle system creation and physics
 * - Cloud particle system to simulate dust
 * - WebGPU compute shader execution
 */

import * as THREE from 'three/webgpu';
import {
  uniform,
  instancedArray,
  instanceIndex,
  vec3,
  vec4,
  float,
  Fn,
  mix,
  length,
  sin,
  cos,
  uv,
  smoothstep,
  texture
} from 'three/tsl';

import {
  hash,
  applyDifferentialRotation,
  applyMouseForce,
  applySpringForce
} from './helpers.js';

/**
 * Spiral Galaxy Position Generation
 *
 * Note: TSL Fn() functions can only return single TSL types (vec3, float, etc.),
 * not JavaScript objects. Since spiral position generation needs to return multiple
 * values (position, normalizedRadius, angle, etc.), we inline this logic in both
 * the star and cloud initialization shaders below.
 *
 * The pattern is consistent between stars and clouds:
 * 1. Generate radius using hash with power function (controls distribution)
 * 2. Select spiral arm and calculate spiral angle
 * 3. Add randomness for natural appearance
 * 4. Convert to Cartesian coordinates
 * 5. Apply vertical thickness (thicker at center, thinner at edges)
 */

// ==============================================================================
// GALAXY SIMULATION CLASS
// ==============================================================================

/**
 * GPU-accelerated galaxy simulation with stars and dust clouds
 * Uses WebGPU compute shaders for particle physics and rendering
 */
export class GalaxySimulation {
  constructor(scene, config, cloudTexture = null) {
    this.scene = scene;
    this.config = config;
    this.COUNT = config.starCount;
    this.cloudTexture = cloudTexture;

    // Storage buffers
    this.spawnPositionBuffer = null;
    this.originalPositionBuffer = null;
    this.velocityBuffer = null;
    this.densityFactorBuffer = null;

    // Compute shaders
    this.computeInit = null;
    this.computeUpdate = null;
    this.cloudInit = null;
    this.cloudUpdate = null;

    // Scene objects
    this.galaxy = null;
    this.cloudPlane = null;

    // Initialize uniforms organized by category
    this.initializeUniforms(config);

    // State
    this.initialized = false;
    this.cloudInitialized = false;
  }

  /**
   * Initialize all shader uniforms organized into logical groups
   */
  initializeUniforms(config) {
    // Compute state uniforms (time, mouse interaction)
    this.uniforms = {
      compute: {
        time: uniform(0),
        deltaTime: uniform(0.016),
        mouse: uniform(new THREE.Vector3(0, 0, 0)),
        mouseActive: uniform(0.0),
        mouseForce: uniform(config.mouseForce),
        mouseRadius: uniform(config.mouseRadius),
        rotationSpeed: uniform(config.rotationSpeed)
      },

      // Galaxy structure uniforms (shape, size, distribution)
      galaxy: {
        radius: uniform(config.galaxyRadius),
        thickness: uniform(config.galaxyThickness || 0.1),
        spiralTightness: uniform(config.spiralTightness),
        armCount: uniform(config.armCount),
        armWidth: uniform(config.armWidth),
        randomness: uniform(config.randomness)
      },

      // Visual appearance uniforms (colors, sizes, opacity)
      visual: {
        particleSize: uniform(config.particleSize),
        cloudSize: uniform(config.cloudSize),
        cloudOpacity: uniform(config.cloudOpacity !== undefined ? config.cloudOpacity : 0.5),
        starBrightness: uniform(config.starBrightness !== undefined ? config.starBrightness : 1.0),
        denseStarColor: uniform(new THREE.Color(config.denseStarColor || '#99ccff')),
        sparseStarColor: uniform(new THREE.Color(config.sparseStarColor || '#ffb380')),
        cloudTintColor: uniform(new THREE.Color(config.cloudTintColor || '#6ba8cc'))
      }
    };
  }

  /**
   * Creates the star particle system with spiral galaxy structure
   */
  createGalaxySystem() {
    // Clean up old galaxy
    if (this.galaxy) {
      this.scene.remove(this.galaxy);
      if (this.galaxy.material) {
        this.galaxy.material.dispose();
      }
    }

    // Create storage buffers for star particles
    this.spawnPositionBuffer = instancedArray(this.COUNT, 'vec3');
    this.originalPositionBuffer = instancedArray(this.COUNT, 'vec3');
    this.velocityBuffer = instancedArray(this.COUNT, 'vec3');
    this.densityFactorBuffer = instancedArray(this.COUNT, 'float');

    // Initialize stars with spiral arm distribution
    this.computeInit = Fn(() => {
      const idx = instanceIndex;
      const seed = idx.toFloat();

      // Distance from center (square root for even distribution)
      const radius = hash(seed.add(1)).pow(0.5).mul(this.uniforms.galaxy.radius);
      const normalizedRadius = radius.div(this.uniforms.galaxy.radius);

      // Choose which spiral arm this particle belongs to
      const armIndex = hash(seed.add(2)).mul(this.uniforms.galaxy.armCount).floor();
      const armAngle = armIndex.mul(6.28318).div(this.uniforms.galaxy.armCount);

      // Spiral angle based on distance (logarithmic spiral)
      const spiralAngle = normalizedRadius.mul(this.uniforms.galaxy.spiralTightness).mul(6.28318);

      // Add randomness to create natural appearance
      const angleOffset = hash(seed.add(3)).sub(0.5).mul(this.uniforms.galaxy.randomness);
      const radiusOffset = hash(seed.add(4)).sub(0.5).mul(this.uniforms.galaxy.armWidth);

      // Final angle and radius
      const angle = armAngle.add(spiralAngle).add(angleOffset);
      const offsetRadius = radius.add(radiusOffset);

      // Convert to Cartesian coordinates
      const x = cos(angle).mul(offsetRadius);
      const z = sin(angle).mul(offsetRadius);

      // Vertical position: thicker at center, thinner at edges
      const thicknessFactor = float(1.0).sub(normalizedRadius).add(0.2); // 1.2 at center, 0.2 at edge
      const y = hash(seed.add(5)).sub(0.5).mul(this.uniforms.galaxy.thickness).mul(thicknessFactor);

      const position = vec3(x, y, z);

      // Store initial positions
      this.spawnPositionBuffer.element(idx).assign(position);
      this.originalPositionBuffer.element(idx).assign(position);

      // Calculate orbital velocity (faster closer to center)
      const orbitalSpeed = float(1.0).div(offsetRadius.add(0.5)).mul(5.0);
      const vx = sin(angle).mul(orbitalSpeed).negate();
      const vz = cos(angle).mul(orbitalSpeed);
      this.velocityBuffer.element(idx).assign(vec3(vx, 0, vz));

      // Calculate density factor for coloring (0 = dense/center, 1 = sparse/edge)
      const radialSparsity = radiusOffset.abs().div(this.uniforms.galaxy.armWidth.mul(0.5).add(0.01));
      const angularSparsity = angleOffset.abs().div(this.uniforms.galaxy.randomness.mul(0.5).add(0.01));
      const sparsityFactor = radialSparsity.add(angularSparsity).mul(0.5).min(1.0);

      this.densityFactorBuffer.element(idx).assign(sparsityFactor);
    })().compute(this.COUNT);

    // Update shader: applies rotation, mouse interaction, and spring forces
    this.computeUpdate = Fn(() => {
      const idx = instanceIndex;
      const position = this.spawnPositionBuffer.element(idx).toVar();
      const originalPos = this.originalPositionBuffer.element(idx);

      // Apply differential rotation
      const rotatedPos = applyDifferentialRotation(
        position,
        this.uniforms.compute.rotationSpeed,
        this.uniforms.compute.deltaTime
      );
      position.assign(rotatedPos);

      // Rotate original position to maintain spring force target
      const rotatedOriginal = applyDifferentialRotation(
        originalPos,
        this.uniforms.compute.rotationSpeed,
        this.uniforms.compute.deltaTime
      );
      this.originalPositionBuffer.element(idx).assign(rotatedOriginal);

      // Apply mouse repulsion force
      const mouseForce = applyMouseForce(
        position,
        this.uniforms.compute.mouse,
        this.uniforms.compute.mouseActive,
        this.uniforms.compute.mouseForce,
        this.uniforms.compute.mouseRadius,
        this.uniforms.compute.deltaTime
      );
      position.addAssign(mouseForce);

      // Apply spring force to restore to original position
      const springForce = applySpringForce(
        position,
        rotatedOriginal,
        float(2.0), // Spring strength
        this.uniforms.compute.deltaTime
      );
      position.addAssign(springForce);

      this.spawnPositionBuffer.element(idx).assign(position);
    })().compute(this.COUNT);

    // Create star visualization material
    const spriteMaterial = new THREE.SpriteNodeMaterial();
    spriteMaterial.transparent = false;
    spriteMaterial.depthWrite = false;
    spriteMaterial.blending = THREE.AdditiveBlending;

    const starPos = this.spawnPositionBuffer.toAttribute();
    const densityFactor = this.densityFactorBuffer.toAttribute();

    // Smooth circular star shape
    const circleShape = Fn(() => {
      const center = uv().sub(0.5).mul(2.0);
      const dist = length(center);
      const alpha = smoothstep(1.0, 0.0, dist).mul(smoothstep(1.0, 0.3, dist));
      return alpha;
    })();

    // Color based on density: blue for dense regions, orange for sparse
    const starColorNode = mix(
      vec3(this.uniforms.visual.denseStarColor),
      vec3(this.uniforms.visual.sparseStarColor),
      densityFactor
    ).mul(this.uniforms.visual.starBrightness);

    spriteMaterial.positionNode = starPos;
    spriteMaterial.colorNode = vec4(starColorNode.x, starColorNode.y, starColorNode.z, float(1.0));
    spriteMaterial.opacityNode = circleShape;
    spriteMaterial.scaleNode = this.uniforms.visual.particleSize;

    this.galaxy = new THREE.Sprite(spriteMaterial);
    this.galaxy.count = this.COUNT;
    this.galaxy.frustumCulled = false;

    this.scene.add(this.galaxy);
  }

  /**
   * Creates cloud particles that follow the galaxy structure
   */
  createClouds() {
    // Clean up old clouds
    if (this.cloudPlane) {
      this.scene.remove(this.cloudPlane);
      if (this.cloudPlane.material) this.cloudPlane.material.dispose();
    }

    const CLOUD_COUNT = this.config.cloudCount;

    // Create cloud particle buffers
    const cloudPositionBuffer = instancedArray(CLOUD_COUNT, 'vec3');
    const cloudOriginalPositionBuffer = instancedArray(CLOUD_COUNT, 'vec3');
    const cloudColorBuffer = instancedArray(CLOUD_COUNT, 'vec3');
    const cloudSizeBuffer = instancedArray(CLOUD_COUNT, 'float');
    const cloudRotationBuffer = instancedArray(CLOUD_COUNT, 'float');

    // Initialize cloud particles
    this.cloudInit = Fn(() => {
      const idx = instanceIndex;
      const seed = idx.toFloat().add(10000); // Offset seed from stars

      // Distance from center (power = 0.7 for more even distribution to avoid center oversaturation)
      const radius = hash(seed.add(1)).pow(0.7).mul(this.uniforms.galaxy.radius);
      const normalizedRadius = radius.div(this.uniforms.galaxy.radius);

      // Choose spiral arm
      const armIndex = hash(seed.add(2)).mul(this.uniforms.galaxy.armCount).floor();
      const armAngle = armIndex.mul(6.28318).div(this.uniforms.galaxy.armCount);

      // Spiral angle based on distance (logarithmic spiral)
      const spiralAngle = normalizedRadius.mul(this.uniforms.galaxy.spiralTightness).mul(6.28318);

      // Add randomness (same pattern as stars)
      const angleOffset = hash(seed.add(3)).sub(0.5).mul(this.uniforms.galaxy.randomness);
      const radiusOffset = hash(seed.add(4)).sub(0.5).mul(this.uniforms.galaxy.armWidth);

      // Final angle and radius
      const angle = armAngle.add(spiralAngle).add(angleOffset);
      const offsetRadius = radius.add(radiusOffset);

      // Convert to Cartesian coordinates
      const x = cos(angle).mul(offsetRadius);
      const z = sin(angle).mul(offsetRadius);

      // Vertical position: slightly thinner than stars
      const thicknessFactor = float(1.0).sub(normalizedRadius).add(0.15); // 1.15 at center, 0.15 at edge
      const y = hash(seed.add(5)).sub(0.5).mul(this.uniforms.galaxy.thickness).mul(thicknessFactor);

      const position = vec3(x, y, z);

      // Store positions
      cloudPositionBuffer.element(idx).assign(position);
      cloudOriginalPositionBuffer.element(idx).assign(position);

      // Cloud color: tinted and darker towards edges
      const tintColor = vec3(this.uniforms.visual.cloudTintColor);
      const cloudColor = tintColor.mul(float(1.0).sub(normalizedRadius.mul(0.3)));
      cloudColorBuffer.element(idx).assign(cloudColor);

      // Size variation: larger clouds in denser regions
      const densityFactor = float(1.0).sub(normalizedRadius.mul(0.5));
      const size = hash(seed.add(6)).mul(0.5).add(0.7).mul(densityFactor);
      cloudSizeBuffer.element(idx).assign(size);

      // Random rotation for visual variation
      const rotation = hash(seed.add(7)).mul(6.28318); // 0 to 2π
      cloudRotationBuffer.element(idx).assign(rotation);
    })().compute(CLOUD_COUNT);

    // Update cloud particles (same physics as stars but weaker spring)
    this.cloudUpdate = Fn(() => {
      const idx = instanceIndex;
      const position = cloudPositionBuffer.element(idx).toVar();
      const originalPos = cloudOriginalPositionBuffer.element(idx);

      // Apply differential rotation
      const rotatedPos = applyDifferentialRotation(
        position,
        this.uniforms.compute.rotationSpeed,
        this.uniforms.compute.deltaTime
      );
      position.assign(rotatedPos);

      // Rotate original position
      const rotatedOriginal = applyDifferentialRotation(
        originalPos,
        this.uniforms.compute.rotationSpeed,
        this.uniforms.compute.deltaTime
      );
      cloudOriginalPositionBuffer.element(idx).assign(rotatedOriginal);

      // Apply mouse force
      const mouseForce = applyMouseForce(
        position,
        this.uniforms.compute.mouse,
        this.uniforms.compute.mouseActive,
        this.uniforms.compute.mouseForce,
        this.uniforms.compute.mouseRadius,
        this.uniforms.compute.deltaTime
      );
      position.addAssign(mouseForce);

      // Apply spring force (weaker than stars for more fluid movement)
      const springForce = applySpringForce(
        position,
        rotatedOriginal,
        float(1.0), // Weaker spring strength
        this.uniforms.compute.deltaTime
      );
      position.addAssign(springForce);

      cloudPositionBuffer.element(idx).assign(position);
    })().compute(CLOUD_COUNT);

    // Store cloud state
    this.cloudCount = CLOUD_COUNT;

    // Create cloud sprite material
    const cloudMaterial = new THREE.SpriteNodeMaterial();
    cloudMaterial.transparent = true;
    cloudMaterial.depthWrite = false;
    cloudMaterial.blending = THREE.AdditiveBlending; // Efficient for overlapping particles

    const cloudPos = cloudPositionBuffer.toAttribute();
    const cloudColor = cloudColorBuffer.toAttribute();
    const cloudSize = cloudSizeBuffer.toAttribute();
    const cloudRotation = cloudRotationBuffer.toAttribute();

    cloudMaterial.positionNode = cloudPos;
    cloudMaterial.colorNode = vec4(cloudColor.x, cloudColor.y, cloudColor.z, float(1.0));
    cloudMaterial.scaleNode = cloudSize.mul(this.uniforms.visual.cloudSize);
    cloudMaterial.rotationNode = cloudRotation;

    // Use texture for soft cloud appearance
    if (this.cloudTexture) {
      const cloudTextureNode = texture(this.cloudTexture, uv());
      cloudMaterial.opacityNode = cloudTextureNode.a.mul(this.uniforms.visual.cloudOpacity);
    } else {
      cloudMaterial.opacityNode = this.uniforms.visual.cloudOpacity;
    }

    this.cloudPlane = new THREE.Sprite(cloudMaterial);
    this.cloudPlane.count = CLOUD_COUNT;
    this.cloudPlane.frustumCulled = false;
    this.cloudPlane.renderOrder = -1; // Render clouds before stars

    this.scene.add(this.cloudPlane);

    // Reset initialization flag so clouds get initialized on next update
    this.cloudInitialized = false;
  }

  /**
   * Updates star count and regenerates galaxy
   */
  updateStarCount(newCount) {
    this.COUNT = newCount;
    this.config.starCount = newCount;
    this.createGalaxySystem();
    this.initialized = false;
  }

  /**
   * Updates uniform values from config changes
   */
  updateUniforms(configUpdate) {
    // Galaxy structure uniforms
    if (configUpdate.galaxyRadius !== undefined)
      this.uniforms.galaxy.radius.value = configUpdate.galaxyRadius;
    if (configUpdate.galaxyThickness !== undefined)
      this.uniforms.galaxy.thickness.value = configUpdate.galaxyThickness;
    if (configUpdate.spiralTightness !== undefined)
      this.uniforms.galaxy.spiralTightness.value = configUpdate.spiralTightness;
    if (configUpdate.armCount !== undefined)
      this.uniforms.galaxy.armCount.value = configUpdate.armCount;
    if (configUpdate.armWidth !== undefined)
      this.uniforms.galaxy.armWidth.value = configUpdate.armWidth;
    if (configUpdate.randomness !== undefined)
      this.uniforms.galaxy.randomness.value = configUpdate.randomness;

    // Compute uniforms
    if (configUpdate.rotationSpeed !== undefined)
      this.uniforms.compute.rotationSpeed.value = configUpdate.rotationSpeed;
    if (configUpdate.mouseForce !== undefined)
      this.uniforms.compute.mouseForce.value = configUpdate.mouseForce;
    if (configUpdate.mouseRadius !== undefined)
      this.uniforms.compute.mouseRadius.value = configUpdate.mouseRadius;

    // Visual uniforms
    if (configUpdate.particleSize !== undefined)
      this.uniforms.visual.particleSize.value = configUpdate.particleSize;
    if (configUpdate.cloudSize !== undefined)
      this.uniforms.visual.cloudSize.value = configUpdate.cloudSize;
    if (configUpdate.cloudOpacity !== undefined)
      this.uniforms.visual.cloudOpacity.value = configUpdate.cloudOpacity;
    if (configUpdate.starBrightness !== undefined)
      this.uniforms.visual.starBrightness.value = configUpdate.starBrightness;
    if (configUpdate.denseStarColor !== undefined)
      this.uniforms.visual.denseStarColor.value.set(configUpdate.denseStarColor);
    if (configUpdate.sparseStarColor !== undefined)
      this.uniforms.visual.sparseStarColor.value.set(configUpdate.sparseStarColor);
    if (configUpdate.cloudTintColor !== undefined)
      this.uniforms.visual.cloudTintColor.value.set(configUpdate.cloudTintColor);

    // Config state
    if (configUpdate.cloudCount !== undefined) {
      this.config.cloudCount = configUpdate.cloudCount;
    }
  }

  /**
   * Main update loop - runs compute shaders and updates uniforms
   */
  async update(renderer, deltaTime, mouse3D, mousePressed) {
    // Initialize stars on first frame
    if (!this.initialized) {
      await renderer.computeAsync(this.computeInit);
      this.initialized = true;
    }

    // Initialize clouds on first frame
    if (!this.cloudInitialized && this.cloudInit) {
      await renderer.computeAsync(this.cloudInit);
      this.cloudInitialized = true;
    }

    // Update compute uniforms
    this.uniforms.compute.time.value += deltaTime;
    this.uniforms.compute.deltaTime.value = deltaTime;
    this.uniforms.compute.mouse.value.copy(mouse3D);
    this.uniforms.compute.mouseActive.value = mousePressed ? 1.0 : 0.0;

    // Run physics computations
    await renderer.computeAsync(this.computeUpdate);

    if (this.cloudUpdate) {
      await renderer.computeAsync(this.cloudUpdate);
    }
  }

  /**
   * Marks galaxy for regeneration on next update
   */
  regenerate() {
    this.initialized = false;
    this.cloudInitialized = false;
  }
}
