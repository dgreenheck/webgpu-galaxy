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
  normalize,
  sin,
  cos,
  uv,
  smoothstep,
  fract,
  texture
} from 'three/tsl';

// Improved hash function for pseudo-random number generation
// Avoids precision loss with large seed values by normalizing first
const hash = Fn(([seed]) => {
  // Normalize seed to avoid precision issues with large values
  const p = fract(seed.mul(0.1031));
  // Apply triple32 hash algorithm
  const h = p.add(19.19);
  const x = fract(h.mul(h.add(47.43)).mul(p));
  return x;
});

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
    this.baseColorBuffer = null;
    this.colorBuffer = null;

    // Compute shaders
    this.computeInit = null;
    this.computeUpdate = null;
    this.galaxy = null;
    this.cloudPlane = null;

    // Compute uniforms
    this.computeTime = uniform(0);
    this.computeDeltaTime = uniform(0.016);
    this.computeMouse = uniform(new THREE.Vector3(0, 0, 0));
    this.computeMouseForce = uniform(config.mouseForce);
    this.computeMouseRadius = uniform(config.mouseRadius);
    this.computeMouseActive = uniform(0.0);
    this.computeRotationSpeed = uniform(config.rotationSpeed);

    // Galaxy generation uniforms
    this.galaxyRadiusUniform = uniform(config.galaxyRadius);
    this.galaxyThicknessUniform = uniform(config.galaxyThickness || 0.1);
    this.spiralTightnessUniform = uniform(config.spiralTightness);
    this.armCountUniform = uniform(config.armCount);
    this.armWidthUniform = uniform(config.armWidth);
    this.randomnessUniform = uniform(config.randomness);

    // Material uniforms
    this.particleSizeUniform = uniform(config.particleSize);
    this.cloudSizeUniform = uniform(config.cloudSize);
    this.cloudOpacityUniform = uniform(config.cloudOpacity !== undefined ? config.cloudOpacity : 0.5);
    this.starBrightnessUniform = uniform(config.starBrightness !== undefined ? config.starBrightness : 1.0);

    // Star color uniforms
    this.denseColorUniform = uniform(new THREE.Color(config.denseStarColor || '#99ccff'));
    this.sparseColorUniform = uniform(new THREE.Color(config.sparseStarColor || '#ffb380'));

    // Cloud color uniform
    this.cloudTintColorUniform = uniform(new THREE.Color(config.cloudTintColor || '#6ba8cc'));

    this.initialized = false;
  }

  createGalaxySystem() {
    // Remove old galaxy from scene if it exists
    if (this.galaxy) {
      this.scene.remove(this.galaxy);
      if (this.galaxy.material) {
        this.galaxy.material.dispose();
      }
    }

    // Create storage buffers
    this.spawnPositionBuffer = instancedArray(this.COUNT, 'vec3');
    this.originalPositionBuffer = instancedArray(this.COUNT, 'vec3');
    this.velocityBuffer = instancedArray(this.COUNT, 'vec3');
    this.densityFactorBuffer = instancedArray(this.COUNT, 'float');

    // Initialize galaxy with spiral arms
    this.computeInit = Fn(() => {
      const idx = instanceIndex;
      const seed = idx.toFloat();

      // Distance from center (square root for better distribution)
      const radius = hash(seed.add(1)).pow(0.5).mul(this.galaxyRadiusUniform);

      // Choose which spiral arm this particle belongs to
      const armIndex = hash(seed.add(2)).mul(this.armCountUniform).floor();
      const armAngle = armIndex.mul(6.28318).div(this.armCountUniform);

      // Spiral angle based on distance
      const spiralAngle = radius.div(this.galaxyRadiusUniform).mul(this.spiralTightnessUniform).mul(6.28318);

      // Add randomness
      const angleOffset = hash(seed.add(3)).sub(0.5).mul(this.randomnessUniform);
      const radiusOffset = hash(seed.add(4)).sub(0.5).mul(this.armWidthUniform);

      const angle = armAngle.add(spiralAngle).add(angleOffset);
      const offsetRadius = radius.add(radiusOffset);

      // Convert to cartesian coordinates
      const x = cos(angle).mul(offsetRadius);
      const z = sin(angle).mul(offsetRadius);

      // Thickness: maximum at center, tapering off towards edges
      // Use (1 - normalizedRadius) to make it thicker at center
      const normalizedRadius = radius.div(this.galaxyRadiusUniform);
      const thicknessFactor = float(1.0).sub(normalizedRadius).add(0.2); // 1.2 at center, 0.2 at edge
      const y = hash(seed.add(5)).sub(0.5).mul(this.galaxyThicknessUniform).mul(thicknessFactor);

      const initialPos = vec3(x, y, z);
      this.spawnPositionBuffer.element(idx).assign(initialPos);
      this.originalPositionBuffer.element(idx).assign(initialPos);

      // Simple orbital velocity
      const orbitalSpeed = float(1.0).div(offsetRadius.add(0.5)).mul(5.0);
      const vx = sin(angle).mul(orbitalSpeed).negate();
      const vz = cos(angle).mul(orbitalSpeed);
      this.velocityBuffer.element(idx).assign(vec3(vx, 0, vz));

      // Store sparsity factor (distance from arm center) for dynamic coloring
      // 0 = dense (at arm center), 1 = sparse (far from arm)
      // The offsets are already scaled by the uniforms, so we normalize them differently
      // Divide by half the range to map [-uniform/2, +uniform/2] to [0, 1]
      const radialSparsity = radiusOffset.abs().div(this.armWidthUniform.mul(0.5).add(0.01));
      const angularSparsity = angleOffset.abs().div(this.randomnessUniform.mul(0.5).add(0.01));
      const sparsityFactor = radialSparsity.add(angularSparsity).mul(0.5).min(1.0);

      this.densityFactorBuffer.element(idx).assign(sparsityFactor);
    })().compute(this.COUNT);

    // Update galaxy orbital motion and mouse interaction
    this.computeUpdate = Fn(() => {
      const idx = instanceIndex;
      const position = this.spawnPositionBuffer.element(idx).toVar();
      const originalPos = this.originalPositionBuffer.element(idx);
      const dt = this.computeDeltaTime;

      // Differential rotation - inner stars rotate faster than outer stars
      const distFromCenter = length(vec3(position.x, 0, position.z));
      const rotationFactor = float(1.0).div(distFromCenter.mul(0.1).add(1.0)); // Faster rotation for inner stars
      const angularSpeed = this.computeRotationSpeed.mul(rotationFactor).mul(dt);
      const cosTheta = cos(angularSpeed.negate());
      const sinTheta = sin(angularSpeed.negate());

      const newX = position.x.mul(cosTheta).sub(position.z.mul(sinTheta));
      const newZ = position.x.mul(sinTheta).add(position.z.mul(cosTheta));
      position.x.assign(newX);
      position.z.assign(newZ);

      // Rotate original position with same differential rotation
      const originalDistFromCenter = length(vec3(originalPos.x, 0, originalPos.z));
      const originalRotationFactor = float(1.0).div(originalDistFromCenter.mul(0.1).add(1.0));
      const originalAngularSpeed = this.computeRotationSpeed.mul(originalRotationFactor).mul(dt);
      const originalCosTheta = cos(originalAngularSpeed.negate());
      const originalSinTheta = sin(originalAngularSpeed.negate());

      const rotatedOriginalX = originalPos.x.mul(originalCosTheta).sub(originalPos.z.mul(originalSinTheta));
      const rotatedOriginalZ = originalPos.x.mul(originalSinTheta).add(originalPos.z.mul(originalCosTheta));
      const rotatedOriginalPos = vec3(rotatedOriginalX, originalPos.y, rotatedOriginalZ);

      this.originalPositionBuffer.element(idx).assign(rotatedOriginalPos);

      // Mouse interaction
      const toMouse = this.computeMouse.sub(position);
      const distToMouse = length(toMouse);
      const mouseInfluence = this.computeMouseActive.mul(
        float(1.0).sub(distToMouse.div(this.computeMouseRadius)).max(0.0)
      );

      const mouseDir = normalize(toMouse);
      const pushForce = mouseDir.mul(this.computeMouseForce).mul(mouseInfluence).mul(dt).negate();
      position.addAssign(pushForce);

      // Spring force back to original position
      const toOriginal = rotatedOriginalPos.sub(position);
      const springForce = toOriginal.mul(2.0).mul(dt);
      position.addAssign(springForce);

      this.spawnPositionBuffer.element(idx).assign(position);
    })().compute(this.COUNT);

    // Create galaxy visualization
    const spriteMaterial = new THREE.SpriteNodeMaterial();
    spriteMaterial.transparent = false;
    spriteMaterial.depthWrite = false;
    spriteMaterial.blending = THREE.AdditiveBlending;

    const starPos = this.spawnPositionBuffer.toAttribute();
    const densityFactor = this.densityFactorBuffer.toAttribute();

    const circleShape = Fn(() => {
      const center = uv().sub(0.5).mul(2.0);
      const dist = length(center);
      const alpha = smoothstep(1.0, 0.0, dist).mul(smoothstep(1.0, 0.3, dist));
      return alpha;
    })();

    // Apply colors dynamically based on density factor
    const starColorNode = mix(
      vec3(this.denseColorUniform),
      vec3(this.sparseColorUniform),
      densityFactor
    ).mul(this.starBrightnessUniform);

    spriteMaterial.positionNode = starPos;
    spriteMaterial.colorNode = vec4(starColorNode.x, starColorNode.y, starColorNode.z, float(1.0));
    spriteMaterial.opacityNode = circleShape;
    spriteMaterial.scaleNode = this.particleSizeUniform;

    this.galaxy = new THREE.Sprite(spriteMaterial);
    this.galaxy.count = this.COUNT;
    this.galaxy.frustumCulled = false;

    this.scene.add(this.galaxy);
  }

  createVolumetricClouds() {
    // Remove old cloud particles if they exist
    if (this.cloudPlane) {
      this.scene.remove(this.cloudPlane);
      if (this.cloudPlane.material) this.cloudPlane.material.dispose();
    }

    // Number of cloud particles
    const CLOUD_COUNT = this.config.cloudCount;

    // Create cloud particle buffers
    const cloudPositionBuffer = instancedArray(CLOUD_COUNT, 'vec3');
    const cloudOriginalPositionBuffer = instancedArray(CLOUD_COUNT, 'vec3');
    const cloudColorBuffer = instancedArray(CLOUD_COUNT, 'vec3');
    const cloudSizeBuffer = instancedArray(CLOUD_COUNT, 'float');
    const cloudRotationBuffer = instancedArray(CLOUD_COUNT, 'float');

    // Initialize cloud particles
    const cloudInit = Fn(() => {
      const idx = instanceIndex;
      const seed = idx.toFloat().add(10000); // Offset seed from stars

      // Distance from center - more even distribution to avoid center oversaturation
      const radius = hash(seed.add(1)).pow(0.7).mul(this.galaxyRadiusUniform);

      // Choose spiral arm
      const armIndex = hash(seed.add(2)).mul(this.armCountUniform).floor();
      const armAngle = armIndex.mul(6.28318).div(this.armCountUniform);

      // Spiral angle
      const spiralAngle = radius.div(this.galaxyRadiusUniform).mul(this.spiralTightnessUniform).mul(6.28318);

      // Match star distribution with same randomness pattern
      const angleOffset = hash(seed.add(3)).sub(0.5).mul(this.randomnessUniform);
      const radiusOffset = hash(seed.add(4)).sub(0.5).mul(this.armWidthUniform);

      const angle = armAngle.add(spiralAngle).add(angleOffset);
      const offsetRadius = radius.add(radiusOffset);

      // Position
      const x = cos(angle).mul(offsetRadius);
      const z = sin(angle).mul(offsetRadius);

      // Vertical distribution (clouds follow same thickness pattern as stars but slightly thinner)
      const normalizedRadius = radius.div(this.galaxyRadiusUniform);
      const thicknessFactor = float(1.0).sub(normalizedRadius).add(0.15); // 1.15 at center, 0.15 at edge (slightly thinner than stars)
      const y = hash(seed.add(5)).sub(0.5).mul(this.galaxyThicknessUniform).mul(thicknessFactor);

      const initialPos = vec3(x, y, z);
      cloudPositionBuffer.element(idx).assign(initialPos);
      cloudOriginalPositionBuffer.element(idx).assign(initialPos);

      // Color based on radius with tint from uniform
      const tintColor = vec3(this.cloudTintColorUniform);
      const cloudColor = tintColor.mul(float(1.0).sub(normalizedRadius.mul(0.3))); // Slightly darker towards edges
      cloudColorBuffer.element(idx).assign(cloudColor);

      // Size variation - larger clouds in denser regions
      const densityFactor = float(1.0).sub(normalizedRadius.mul(0.5));
      const size = hash(seed.add(6)).mul(0.5).add(0.7).mul(densityFactor);
      cloudSizeBuffer.element(idx).assign(size);

      // Random rotation for each cloud particle
      const rotation = hash(seed.add(7)).mul(6.28318); // 0 to 2*PI
      cloudRotationBuffer.element(idx).assign(rotation);
    })().compute(CLOUD_COUNT);

    // Update cloud particles (rotation and mouse interaction)
    const cloudUpdate = Fn(() => {
      const idx = instanceIndex;
      const position = cloudPositionBuffer.element(idx).toVar();
      const originalPos = cloudOriginalPositionBuffer.element(idx);

      const dt = this.computeDeltaTime;

      // Differential rotation for clouds (same as stars)
      const distFromCenter = length(vec3(position.x, 0, position.z));
      const rotationFactor = float(1.0).div(distFromCenter.mul(0.1).add(1.0));
      const angularSpeed = this.computeRotationSpeed.mul(rotationFactor).mul(dt);
      const cosTheta = cos(angularSpeed.negate());
      const sinTheta = sin(angularSpeed.negate());

      const newX = position.x.mul(cosTheta).sub(position.z.mul(sinTheta));
      const newZ = position.x.mul(sinTheta).add(position.z.mul(cosTheta));
      position.x.assign(newX);
      position.z.assign(newZ);

      // Rotate original position with differential rotation
      const originalDistFromCenter = length(vec3(originalPos.x, 0, originalPos.z));
      const originalRotationFactor = float(1.0).div(originalDistFromCenter.mul(0.1).add(1.0));
      const originalAngularSpeed = this.computeRotationSpeed.mul(originalRotationFactor).mul(dt);
      const originalCosTheta = cos(originalAngularSpeed.negate());
      const originalSinTheta = sin(originalAngularSpeed.negate());

      const rotatedOriginalX = originalPos.x.mul(originalCosTheta).sub(originalPos.z.mul(originalSinTheta));
      const rotatedOriginalZ = originalPos.x.mul(originalSinTheta).add(originalPos.z.mul(originalCosTheta));
      const rotatedOriginalPos = vec3(rotatedOriginalX, originalPos.y, rotatedOriginalZ);
      cloudOriginalPositionBuffer.element(idx).assign(rotatedOriginalPos);

      // Mouse interaction
      const toMouse = this.computeMouse.sub(position);
      const distToMouse = length(toMouse);
      const mouseInfluence = this.computeMouseActive.mul(
        float(1.0).sub(distToMouse.div(this.computeMouseRadius)).max(0.0)
      );

      const mouseDir = normalize(toMouse);
      const pushForce = mouseDir.mul(this.computeMouseForce).mul(mouseInfluence).mul(dt).negate();
      position.addAssign(pushForce);

      // Spring force back (slower than stars)
      const toOriginal = rotatedOriginalPos.sub(position);
      const springForce = toOriginal.mul(1.0).mul(dt);
      position.addAssign(springForce);

      cloudPositionBuffer.element(idx).assign(position);
    })().compute(CLOUD_COUNT);

    // Store compute shaders
    this.cloudInit = cloudInit;
    this.cloudUpdate = cloudUpdate;
    this.cloudCount = CLOUD_COUNT;

    // Create cloud sprite material
    const cloudMaterial = new THREE.SpriteNodeMaterial();
    cloudMaterial.transparent = true;
    cloudMaterial.depthWrite = false;
    cloudMaterial.blending = THREE.AdditiveBlending; // Much faster than NormalBlending for overlapping particles

    const cloudPos = cloudPositionBuffer.toAttribute();
    const cloudColor = cloudColorBuffer.toAttribute();
    const cloudSize = cloudSizeBuffer.toAttribute();
    const cloudRotation = cloudRotationBuffer.toAttribute();

    cloudMaterial.positionNode = cloudPos;
    cloudMaterial.colorNode = vec4(cloudColor.x, cloudColor.y, cloudColor.z, float(1.0));
    cloudMaterial.scaleNode = cloudSize.mul(this.cloudSizeUniform);
    cloudMaterial.rotationNode = cloudRotation;

    // Use preloaded texture for opacity
    if (this.cloudTexture) {
      const cloudTextureNode = texture(this.cloudTexture, uv());
      cloudMaterial.opacityNode = cloudTextureNode.a.mul(this.cloudOpacityUniform);
    } else {
      cloudMaterial.opacityNode = this.cloudOpacityUniform;
    }

    this.cloudPlane = new THREE.Sprite(cloudMaterial);
    this.cloudPlane.count = CLOUD_COUNT;
    this.cloudPlane.frustumCulled = false;
    this.cloudPlane.renderOrder = -1;

    this.scene.add(this.cloudPlane);

    // Initialize clouds
    this.cloudInitialized = false;
  }

  updateStarCount(newCount) {
    this.COUNT = newCount;
    this.config.starCount = newCount;
    this.createGalaxySystem();
    this.initialized = false;
  }

  updateUniforms(configUpdate) {
    if (configUpdate.galaxyRadius !== undefined) this.galaxyRadiusUniform.value = configUpdate.galaxyRadius;
    if (configUpdate.galaxyThickness !== undefined) this.galaxyThicknessUniform.value = configUpdate.galaxyThickness;
    if (configUpdate.spiralTightness !== undefined) this.spiralTightnessUniform.value = configUpdate.spiralTightness;
    if (configUpdate.armCount !== undefined) this.armCountUniform.value = configUpdate.armCount;
    if (configUpdate.armWidth !== undefined) this.armWidthUniform.value = configUpdate.armWidth;
    if (configUpdate.randomness !== undefined) this.randomnessUniform.value = configUpdate.randomness;
    if (configUpdate.rotationSpeed !== undefined) this.computeRotationSpeed.value = configUpdate.rotationSpeed;
    if (configUpdate.mouseForce !== undefined) this.computeMouseForce.value = configUpdate.mouseForce;
    if (configUpdate.mouseRadius !== undefined) this.computeMouseRadius.value = configUpdate.mouseRadius;
    if (configUpdate.particleSize !== undefined) this.particleSizeUniform.value = configUpdate.particleSize;
    if (configUpdate.cloudSize !== undefined) this.cloudSizeUniform.value = configUpdate.cloudSize;
    if (configUpdate.cloudOpacity !== undefined) this.cloudOpacityUniform.value = configUpdate.cloudOpacity;
    if (configUpdate.starBrightness !== undefined) this.starBrightnessUniform.value = configUpdate.starBrightness;
    if (configUpdate.denseStarColor !== undefined) this.denseColorUniform.value.set(configUpdate.denseStarColor);
    if (configUpdate.sparseStarColor !== undefined) this.sparseColorUniform.value.set(configUpdate.sparseStarColor);
    if (configUpdate.cloudTintColor !== undefined) this.cloudTintColorUniform.value.set(configUpdate.cloudTintColor);
    if (configUpdate.cloudCount !== undefined) {
      this.config.cloudCount = configUpdate.cloudCount;
    }
  }

  async update(renderer, deltaTime, mouse3D, mousePressed) {
    // Initialize galaxy once
    if (!this.initialized) {
      await renderer.computeAsync(this.computeInit);
      this.initialized = true;
    }

    // Initialize clouds once
    if (!this.cloudInitialized && this.cloudInit) {
      await renderer.computeAsync(this.cloudInit);
      this.cloudInitialized = true;
    }

    // Update uniforms
    this.computeTime.value += deltaTime;
    this.computeDeltaTime.value = deltaTime;
    this.computeMouse.value.copy(mouse3D);
    this.computeMouseActive.value = mousePressed ? 1.0 : 0.0;

    // Update galaxy
    await renderer.computeAsync(this.computeUpdate);

    // Update clouds
    if (this.cloudUpdate) {
      await renderer.computeAsync(this.cloudUpdate);
    }
  }

  regenerate() {
    this.initialized = false;
    this.cloudInitialized = false;
  }
}
