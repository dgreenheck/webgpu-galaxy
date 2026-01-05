import * as THREE from 'three/webgpu';
import { pass } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { BlackHoleSimulation } from './blackhole.js';
import { BlackHoleUI } from './ui.js';

// Configuration
const config = {
  // Black hole physics
  blackHoleMass: 1.0,

  // Accretion disk
  diskInnerRadius: 2.6,
  diskOuterRadius: 12.0,
  diskTemperature: 1.5,
  diskBrightness: 2.0,
  dopplerStrength: 0.8,

  // Bloom post-processing
  bloomStrength: 0.6,
  bloomRadius: 0.5,
  bloomThreshold: 0.2
};

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(0, 8, 20);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGPURenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

// Orbit controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 5;
controls.maxDistance = 50;
controls.target.set(0, 0, 0);

// Post-processing
let postProcessing = null;
let bloomPassNode = null;

// Create black hole simulation
const blackHoleSimulation = new BlackHoleSimulation(scene, config);
blackHoleSimulation.createBlackHole();

// Setup bloom
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

// Create UI with callbacks
const ui = new BlackHoleUI(config, {
  onUniformChange: (key, value) => blackHoleSimulation.updateUniforms({ [key]: value }),

  onBloomChange: (property, value) => {
    if (bloomPassNode) bloomPassNode[property].value = value;
  },

  onRegenerate: () => {
    blackHoleSimulation.updateUniforms(config);
    blackHoleSimulation.regenerate();
  }
});

// FPS counter
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

    document.getElementById('fps').textContent = fps;
    ui.updateFPS(fps);
  }
}

// Animation loop
let lastFrameTime = performance.now();

async function animate() {
  requestAnimationFrame(animate);

  const currentTime = performance.now();
  const deltaTime = Math.min((currentTime - lastFrameTime) / 1000, 0.033);
  lastFrameTime = currentTime;

  // Update controls
  controls.update();

  // Update black hole simulation
  blackHoleSimulation.update(renderer, deltaTime, camera);

  // Render
  if (postProcessing) {
    postProcessing.render();
  } else {
    renderer.render(scene, camera);
  }

  updateFPS();
}

// Handle resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  blackHoleSimulation.onResize(window.innerWidth, window.innerHeight);
});

// Initialize
renderer.init().then(() => {
  postProcessing = new THREE.PostProcessing(renderer);
  setupBloom();
  ui.setBloomNode(bloomPassNode);
  animate();
}).catch(err => {
  console.error('Failed to initialize renderer:', err);
});
