# 🌌 WebGPU Galaxy Simulation

A real-time GPU-accelerated galaxy simulation using WebGPU, Three.js, and TSL (Three.js Shading Language). Experience an interactive spiral galaxy with up to 750,000 particles, dynamic bloom effects, and customizable parameters.

## ✨ Features

- **GPU-Accelerated Physics** - Particle simulation runs entirely on the GPU using WebGPU compute shaders
- **Interactive Controls** - Click and drag to interact with the galaxy using mouse forces
- **Real-time Parameters** - Adjust galaxy properties in real-time with Tweakpane UI
- **Bloom Post-Processing** - Beautiful HDR bloom effects for enhanced visuals
- **Procedural Generation** - Spiral arm generation with configurable parameters
- **Dust Clouds** - Realistic nebula clouds with alpha-blended particles
- **Starfield Background** - Spherical starfield with color variation

## 🚀 Live Demo

Visit the live demo at: `https://dgreenheck.github.io/webgpu-galaxy/`

## 🛠️ Technologies

- **Three.js (WebGPU)** - 3D rendering engine with WebGPU backend
- **TSL** - Three.js Shading Language for GPU compute shaders
- **Vite** - Fast build tool and dev server
- **Tweakpane** - UI controls for parameter adjustment

## 📋 Requirements

- A browser with WebGPU support (Chrome 113+, Edge 113+, or other compatible browsers)
- GPU with WebGPU capabilities

## 🏃 Getting Started

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open your browser to `http://localhost:5173` (or the port shown in the terminal).

### Build

```bash
npm run build
```

The built files will be in the `dist` directory.

### Preview Build

```bash
npm run preview
```

## 🎮 Controls

- **Left Mouse Drag** - Orbit camera around galaxy
- **Right Mouse Drag** - Pan camera
- **Mouse Wheel** - Zoom in/out
- **Click & Drag on Galaxy** - Apply force to particles
- **Right Panel** - Adjust galaxy parameters in real-time

## ⚙️ Configurable Parameters

### Galaxy Properties

- Star count
- Rotation speed
- Spiral tightness
- Arm count
- Arm width
- Randomness
- Galaxy radius and thickness

### Visual Effects

- Particle size and brightness
- Color gradients (dense vs sparse regions)
- Bloom strength, radius, and threshold
- Cloud count, size, and opacity

### Interaction

- Mouse force strength
- Mouse interaction radius

## 📝 License

MIT

## 🙏 Acknowledgments

Built with [Three.js](https://threejs.org/) and [WebGPU](https://www.w3.org/TR/webgpu/)
