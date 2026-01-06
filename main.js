/**
 * Black Hole Simulation - Main Entry Point
 *
 * Sets up Three.js WebGPU renderer, camera controls, and post-processing.
 * Connects the simulation to the UI controls.
 */

import * as THREE from 'three/webgpu';
import { pass } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { BlackHoleSimulation } from './blackhole.js';
import { BlackHoleUI } from './ui.js';

// ============================================================================
// LOCAL STORAGE
// ============================================================================

const STORAGE_KEY = 'blackhole-simulation-config';

/**
 * Load configuration from localStorage, merging with defaults.
 */
function loadConfig(defaults) {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      // Merge saved values with defaults (defaults provide any missing keys)
      return { ...defaults, ...parsed };
    }
  } catch (e) {
    console.warn('Failed to load config from localStorage:', e);
  }
  return defaults;
}

/**
 * Save current configuration to localStorage.
 */
function saveConfig(config) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
    console.log('Configuration saved to localStorage');
  } catch (e) {
    console.warn('Failed to save config to localStorage:', e);
  }
}

/**
 * Clear configuration from localStorage.
 */
function clearConfig() {
  try {
    localStorage.removeItem(STORAGE_KEY);
    console.log('Configuration cleared from localStorage');
  } catch (e) {
    console.warn('Failed to clear config from localStorage:', e);
  }
}

// ============================================================================
// CONFIGURATION
// ============================================================================

const defaultConfig = {
  // Black hole physics
  blackHoleMass: 1.0,

  // Accretion disk geometry
  diskInnerRadius: 3.0,
  diskOuterRadius: 12.0,
  diskInnerThickness: 0.1,
  diskOuterThickness: 0.8,

  // Accretion disk appearance
  diskTemperature: 1.5,
  diskBrightness: 2.0,
  diskRotationSpeed: 0.3,

  // Ring pattern
  ringEnabled: true,
  ringScale: 1.0,
  ringContrast: 1.5,
  ringBrightness: 0.3,
  ringSharpness: 1.0,
  ringTwist: 0.5,
  diskDifferentialRotation: 0.8,

  // Disk edge falloff
  diskEdgeSoftnessInner: 0.15,
  diskEdgeSoftnessOuter: 0.15,
  diskRadialFalloff: 0.5,

  // Disk colors (user configurable)
  diskInnerColor: '#ffffee',
  diskOuterColor: '#ff4400',

  // Relativistic effects
  gravitationalLensing: 1.5,

  // Volumetric rendering
  diskDensity: 0.25,
  diskOpacityFalloff: 0.8,

  // Performance
  qualityPreset: 'medium',
  raySteps: 100,
  stepSize: 0.3,
  adaptiveMinStep: 0.15,
  stepJitter: 0.25,

  // Stars
  starsEnabled: true,
  starDensity: 0.003,
  starSize: 2.0,
  starBrightness: 1.0,

  // Nebula
  nebulaEnabled: false,
  nebulaBrightness: 0.15,
  nebulaColor1: '#1a0033',
  nebulaColor2: '#4d1a26',
  nebulaScale: 2.0,
  nebulaDetailScale: 2.0,
  nebulaSpeed: 0.01,
  nebulaDensity: 2.0,
  nebulaOffsetX: 0.0,
  nebulaOffsetY: 0.0,
  nebulaOffsetZ: 0.0,

  // Bloom post-processing
  bloomStrength: 0.8,
  bloomRadius: 0.5,
  bloomThreshold: 0.2
};

// Load config from localStorage (merges with defaults)
const config = loadConfig(defaultConfig);

// ============================================================================
// SCENE SETUP
// ============================================================================

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(0, 5, 20);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGPURenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

// ============================================================================
// ORBIT CONTROLS
// ============================================================================

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 5;
controls.maxDistance = 50;
controls.target.set(0, 0, 0);

// ============================================================================
// POST-PROCESSING
// ============================================================================

let postProcessing = null;
let bloomPassNode = null;

function setupBloom() {
  if (!postProcessing) return;

  const scenePass = pass(scene, camera);
  const scenePassColor = scenePass.getTextureNode();

  bloomPassNode = bloom(scenePassColor);
  bloomPassNode.threshold.value = config.bloomThreshold;
  bloomPassNode.strength.value = config.bloomStrength;
  bloomPassNode.radius.value = config.bloomRadius;

  postProcessing.outputNode = scenePassColor.add(bloomPassNode);
}

// ============================================================================
// BLACK HOLE SIMULATION
// ============================================================================

const blackHoleSimulation = new BlackHoleSimulation(scene, config);
blackHoleSimulation.createBlackHole();

// ============================================================================
// UI CONTROLS
// ============================================================================

const ui = new BlackHoleUI(config, {
  // Handle individual uniform changes
  onUniformChange: (key, value) => {
    blackHoleSimulation.updateUniforms({ [key]: value });
  },

  // Handle bloom changes
  onBloomChange: (property, value) => {
    if (bloomPassNode) {
      bloomPassNode[property].value = value;
    }
  },

  // Handle quality preset changes
  onQualityPreset: (presetName) => {
    blackHoleSimulation.applyQualityPreset(presetName);
  },

  // Handle regeneration (e.g., after major config changes)
  onRegenerate: () => {
    blackHoleSimulation.updateUniforms(config);
    blackHoleSimulation.regenerate();
  },

  // Save current config to localStorage
  onSaveConfig: () => {
    saveConfig(config);
  },

  // Clear localStorage and reload with defaults
  onClearConfig: () => {
    clearConfig();
    window.location.reload();
  },

  // Reset to defaults without clearing localStorage
  onResetToDefaults: () => {
    Object.assign(config, defaultConfig);
    blackHoleSimulation.updateUniforms(config);
    if (bloomPassNode) {
      bloomPassNode.threshold.value = config.bloomThreshold;
      bloomPassNode.strength.value = config.bloomStrength;
      bloomPassNode.radius.value = config.bloomRadius;
    }
  }
});

// ============================================================================
// FPS COUNTER
// ============================================================================

let frameCount = 0;
let lastTime = performance.now();
let fps = 60;

function updateFPS() {
  frameCount++;
  const currentTime = performance.now();
  const deltaTime = currentTime - lastTime;

  if (deltaTime >= 1000) {
    fps = Math.round((frameCount * 1000) / deltaTime);
    frameCount = 0;
    lastTime = currentTime;

    // Update UI
    const fpsElement = document.getElementById('fps');
    if (fpsElement) {
      fpsElement.textContent = fps;
    }
    ui.updateFPS(fps);
  }
}

// ============================================================================
// ANIMATION LOOP
// ============================================================================

let lastFrameTime = performance.now();

async function animate() {
  requestAnimationFrame(animate);

  const currentTime = performance.now();
  const deltaTime = Math.min((currentTime - lastFrameTime) / 1000, 0.033);
  lastFrameTime = currentTime;

  // Update controls
  controls.update();

  // Update black hole simulation
  blackHoleSimulation.update(deltaTime, camera);

  // Render
  if (postProcessing) {
    postProcessing.render();
  } else {
    renderer.render(scene, camera);
  }

  updateFPS();
}

// ============================================================================
// WINDOW RESIZE
// ============================================================================

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  blackHoleSimulation.onResize(window.innerWidth, window.innerHeight);
});

// ============================================================================
// INITIALIZATION
// ============================================================================

renderer.init().then(() => {
  postProcessing = new THREE.PostProcessing(renderer);
  setupBloom();
  ui.setBloomNode(bloomPassNode);
  animate();
}).catch(err => {
  console.error('Failed to initialize WebGPU renderer:', err);
  // Show fallback message
  document.body.innerHTML = `
    <div style="color: white; padding: 20px; text-align: center;">
      <h1>WebGPU Not Supported</h1>
      <p>This demo requires a browser with WebGPU support.</p>
      <p>Try Chrome 113+ or Edge 113+ on desktop.</p>
    </div>
  `;
});
