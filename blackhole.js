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
import { uniform } from 'three/tsl';
import { createBlackHoleShader } from './blackhole-shader.js';

// ============================================================================
// SECTION 1: PHYSICAL CONSTANTS AND CONFIGURATION
// ============================================================================

/**
 * Quality presets for balancing visual quality vs performance.
 * These control the raymarching parameters.
 */
export const QUALITY_PRESETS = {
  low: {
    stepSize: 0.4,
    starsEnabled: false,
    nebulaEnabled: false,
    stepJitter: 0.3
  },
  medium: {
    stepSize: 0.3,
    starsEnabled: true,
    nebulaEnabled: false,
    stepJitter: 0.25
  },
  high: {
    stepSize: 0.2,
    starsEnabled: true,
    nebulaEnabled: true,
    stepJitter: 0.2
  },
  ultra: {
    stepSize: 0.15,
    starsEnabled: true,
    nebulaEnabled: true,
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
      // Inner radius constrained to ISCO (Innermost Stable Circular Orbit)
      // For Schwarzschild black hole: ISCO = 3 × rs (where rs = 2M = 2.0 in our units)
      diskInnerRadius: uniform(config.diskInnerRadius ?? 3.0),
      diskOuterRadius: uniform(config.diskOuterRadius ?? 12.0),

      // === Accretion Disk Appearance ===
      // Peak temperature in thousands of Kelvin (at inner edge)
      // Typical values: 5-50 (5,000K - 50,000K)
      diskTemperature: uniform(config.diskTemperature ?? 10.0),
      // Temperature falloff exponent: 0.75 = physical (Shakura-Sunyaev), higher = steeper
      temperatureFalloff: uniform(config.temperatureFalloff ?? 0.75),
      diskBrightness: uniform(config.diskBrightness ?? 2.0),
      diskRotationSpeed: uniform(config.diskRotationSpeed ?? 0.3),
      diskInnerThickness: uniform(config.diskInnerThickness ?? 0.1),
      diskOuterThickness: uniform(config.diskOuterThickness ?? 0.8),

      // === Ring Pattern Controls (Turbulence with Keplerian Advection) ===
      ringEnabled: uniform(config.ringEnabled ? 1.0 : 0.0),
      ringScale: uniform(config.ringScale ?? 1.0),
      ringContrast: uniform(config.ringContrast ?? 1.5),
      ringBrightness: uniform(config.ringBrightness ?? 0.3),
      ringSharpness: uniform(config.ringSharpness ?? 1.0),
      ringTwist: uniform(config.ringTwist ?? 5.0),
      noiseAnimFrequency: uniform(config.noiseAnimFrequency ?? 1.0),
      noiseAnimAmplitude: uniform(config.noiseAnimAmplitude ?? 0.5),

      // === Disk Edge Falloff ===
      diskEdgeSoftnessInner: uniform(config.diskEdgeSoftnessInner ?? 0.15),
      diskEdgeSoftnessOuter: uniform(config.diskEdgeSoftnessOuter ?? 0.15),
      diskRadialFalloff: uniform(config.diskRadialFalloff ?? 0.5),

      // Note: Disk color is now computed from blackbody radiation based on temperature

      // === Relativistic Effects ===
      gravitationalLensing: uniform(config.gravitationalLensing ?? 1.5),

      // === Volumetric Rendering ===
      diskDensity: uniform(config.diskDensity ?? 0.25),
      diskOpacityFalloff: uniform(config.diskOpacityFalloff ?? 0.8),

      // === Performance ===
      stepSize: uniform(config.stepSize ?? 0.3),
      adaptiveMinStep: uniform(config.adaptiveMinStep ?? 0.15),

      // === Anti-Aliasing ===
      stepJitter: uniform(config.stepJitter ?? 0.25),
      frameIndex: uniform(0),

      // === Stars ===
      starsEnabled: uniform(config.starsEnabled ? 1.0 : 0.0),
      starBackgroundColor: uniform(new THREE.Color(config.starBackgroundColor ?? '#000000')),
      starDensity: uniform(config.starDensity ?? 0.003),
      starSize: uniform(config.starSize ?? 2.0),
      starBrightness: uniform(config.starBrightness ?? 1.0),

      // === Nebula Layer 1 ===
      nebulaEnabled: uniform(config.nebulaEnabled ? 1.0 : 0.0),
      nebula1Scale: uniform(config.nebula1Scale ?? 2.0),
      nebula1Density: uniform(config.nebula1Density ?? 0.5),
      nebula1Brightness: uniform(config.nebula1Brightness ?? 0.15),
      nebula1Color: uniform(new THREE.Color(config.nebula1Color ?? '#1a0033')),

      // === Nebula Layer 2 ===
      nebula2Scale: uniform(config.nebula2Scale ?? 6.0),
      nebula2Density: uniform(config.nebula2Density ?? 0.5),
      nebula2Brightness: uniform(config.nebula2Brightness ?? 0.15),
      nebula2Color: uniform(new THREE.Color(config.nebula2Color ?? '#4d1a26')),

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
    return createBlackHoleShader(this.uniforms);
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
    if (config.temperatureFalloff !== undefined) u.temperatureFalloff.value = config.temperatureFalloff;
    if (config.diskBrightness !== undefined) u.diskBrightness.value = config.diskBrightness;
    if (config.diskRotationSpeed !== undefined) u.diskRotationSpeed.value = config.diskRotationSpeed;

    // Ring pattern (with Keplerian advection)
    if (config.ringEnabled !== undefined) u.ringEnabled.value = config.ringEnabled ? 1.0 : 0.0;
    if (config.ringScale !== undefined) u.ringScale.value = config.ringScale;
    if (config.ringContrast !== undefined) u.ringContrast.value = config.ringContrast;
    if (config.ringBrightness !== undefined) u.ringBrightness.value = config.ringBrightness;
    if (config.ringSharpness !== undefined) u.ringSharpness.value = config.ringSharpness;
    if (config.ringTwist !== undefined) u.ringTwist.value = config.ringTwist;
    if (config.noiseAnimFrequency !== undefined) u.noiseAnimFrequency.value = config.noiseAnimFrequency;
    if (config.noiseAnimAmplitude !== undefined) u.noiseAnimAmplitude.value = config.noiseAnimAmplitude;

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
    if (config.stepSize !== undefined) u.stepSize.value = config.stepSize;
    if (config.adaptiveMinStep !== undefined) u.adaptiveMinStep.value = config.adaptiveMinStep;

    // Anti-aliasing
    if (config.stepJitter !== undefined) u.stepJitter.value = config.stepJitter;
    if (config.frameIndex !== undefined) u.frameIndex.value = config.frameIndex;

    // Star uniforms
    if (config.starsEnabled !== undefined) u.starsEnabled.value = config.starsEnabled ? 1.0 : 0.0;
    if (config.starBackgroundColor !== undefined) u.starBackgroundColor.value.set(config.starBackgroundColor);
    if (config.starDensity !== undefined) u.starDensity.value = config.starDensity;
    if (config.starSize !== undefined) u.starSize.value = config.starSize;
    if (config.starBrightness !== undefined) u.starBrightness.value = config.starBrightness;

    // Nebula Layer 1 uniforms
    if (config.nebulaEnabled !== undefined) u.nebulaEnabled.value = config.nebulaEnabled ? 1.0 : 0.0;
    if (config.nebula1Scale !== undefined) u.nebula1Scale.value = config.nebula1Scale;
    if (config.nebula1Density !== undefined) u.nebula1Density.value = config.nebula1Density;
    if (config.nebula1Brightness !== undefined) u.nebula1Brightness.value = config.nebula1Brightness;
    if (config.nebula1Color !== undefined) u.nebula1Color.value.set(config.nebula1Color);

    // Nebula Layer 2 uniforms
    if (config.nebula2Scale !== undefined) u.nebula2Scale.value = config.nebula2Scale;
    if (config.nebula2Density !== undefined) u.nebula2Density.value = config.nebula2Density;
    if (config.nebula2Brightness !== undefined) u.nebula2Brightness.value = config.nebula2Brightness;
    if (config.nebula2Color !== undefined) u.nebula2Color.value.set(config.nebula2Color);

    // Note: Disk color is computed from blackbody radiation (no color uniforms needed)
  }

  /**
   * Apply a quality preset.
   */
  applyQualityPreset(presetName) {
    const preset = QUALITY_PRESETS[presetName];
    if (!preset) return;

    this.updateUniforms({
      stepSize: preset.stepSize,
      starsEnabled: preset.starsEnabled,
      nebulaEnabled: preset.nebulaEnabled,
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
