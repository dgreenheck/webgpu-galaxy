import * as THREE from 'three/webgpu';
import { pass } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GalaxySimulation } from './galaxy.js';
import { GalaxyUI } from './ui.js';

// Configuration
const config = {
  starCount: 750000,
  rotationSpeed: 0.1,
  spiralTightness: 1.75,
  mouseForce: 7.0,
  mouseRadius: 10.0,
  galaxyRadius: 13.0,
  galaxyThickness: 3,
  armCount: 2,
  armWidth: 2.25,
  randomness: 1.8,
  particleSize: 0.06,
  starBrightness: 0.3,
  denseStarColor: '#1885ff',
  sparseStarColor: '#ffb28a',
  bloomStrength: 0.2,
  bloomRadius: 0.2,
  bloomThreshold: 0.1,
  cloudCount: 5000,
  cloudSize: 3,
  cloudOpacity: 0.02,
  cloudTintColor: '#ffdace'
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
camera.position.set(0, 12, 17);
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
controls.maxDistance = 30;
controls.target.set(0, -2, 0);

// Post-processing
let postProcessing = null;
let bloomPassNode = null;

// Mouse tracking
const mouse3D = new THREE.Vector3(0, 0, 0);
const raycaster = new THREE.Raycaster();
const intersectionPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
let mousePressed = false;

window.addEventListener('mousedown', () => mousePressed = true);
window.addEventListener('mouseup', () => mousePressed = false);
window.addEventListener('mousemove', (event) => {
  const mouse = new THREE.Vector2(
    (event.clientX / window.innerWidth) * 2 - 1,
    -(event.clientY / window.innerHeight) * 2 + 1
  );
  raycaster.setFromCamera(mouse, camera);
  raycaster.ray.intersectPlane(intersectionPlane, mouse3D);
});

/**
 * Creates a starry background with random colored stars distributed on a sphere
 * @param {THREE.Scene} scene - Scene to add stars to
 * @param {number} count - Number of background stars
 * @returns {THREE.Points} - The star points object
 */
function createStarryBackground(scene, count = 5000) {
  const starGeometry = new THREE.BufferGeometry();
  const starPositions = new Float32Array(count * 3);
  const starColors = new Float32Array(count * 3);

  // Distribute stars randomly on a sphere
  for (let i = 0; i < count; i++) {
    // Spherical coordinates for uniform distribution
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const radius = 100 + Math.random() * 100;

    // Convert to Cartesian coordinates
    starPositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
    starPositions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
    starPositions[i * 3 + 2] = radius * Math.cos(phi);

    // Add color variation (mostly white, some blue/orange tinted)
    const color = 0.8 + Math.random() * 0.2;
    const tint = Math.random();
    if (tint < 0.1) {
      // Blue tint
      starColors[i * 3] = color * 0.8;
      starColors[i * 3 + 1] = color * 0.9;
      starColors[i * 3 + 2] = color;
    } else if (tint < 0.2) {
      // Orange tint
      starColors[i * 3] = color;
      starColors[i * 3 + 1] = color * 0.8;
      starColors[i * 3 + 2] = color * 0.6;
    } else {
      // White
      starColors[i * 3] = color;
      starColors[i * 3 + 1] = color;
      starColors[i * 3 + 2] = color;
    }
  }

  starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
  starGeometry.setAttribute('color', new THREE.BufferAttribute(starColors, 3));

  const starMaterial = new THREE.PointsMaterial({
    size: 0.3,
    vertexColors: true,
    transparent: true,
    opacity: 0.8,
    sizeAttenuation: true
  });

  const stars = new THREE.Points(starGeometry, starMaterial);
  scene.add(stars);

  return stars;
}

// Preload cloud texture
const textureLoader = new THREE.TextureLoader();
const cloudTexture = textureLoader.load('/cloud.png');

// Create galaxy simulation with preloaded texture
const galaxySimulation = new GalaxySimulation(scene, config, cloudTexture);
galaxySimulation.createGalaxySystem();
galaxySimulation.createClouds();

// Create starry background
createStarryBackground(scene);

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
const ui = new GalaxyUI(config, {
  onUniformChange: (key, value) => galaxySimulation.updateUniforms({ [key]: value }),

  onBloomChange: (property, value) => {
    if (bloomPassNode) bloomPassNode[property].value = value;
  },

  onStarCountChange: (newCount) => {
    galaxySimulation.updateStarCount(newCount);
    document.getElementById('star-count').textContent = newCount.toLocaleString();
  },

  onCloudCountChange: (newCount) => {
    galaxySimulation.updateUniforms({ cloudCount: newCount });
    galaxySimulation.createClouds();
  },

  onCloudTintChange: (color) => {
    galaxySimulation.updateUniforms({ cloudTintColor: color });
    galaxySimulation.createClouds();
  },

  onRegenerate: () => {
    galaxySimulation.updateUniforms(config);
    galaxySimulation.createClouds();
    galaxySimulation.regenerate();
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

  // Update galaxy
  await galaxySimulation.update(renderer, deltaTime, mouse3D, mousePressed);

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
});

// Initialize
renderer.init().then(() => {
  postProcessing = new THREE.PostProcessing(renderer);
  setupBloom();
  ui.setBloomNode(bloomPassNode);

  document.getElementById('star-count').textContent = config.starCount.toLocaleString();
  animate();
}).catch(err => {
  console.error('Failed to initialize renderer:', err);
});
