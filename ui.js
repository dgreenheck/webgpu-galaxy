import { Pane } from 'tweakpane';

export class GalaxyUI {
  constructor(config, callbacks) {
    this.config = config;
    this.callbacks = callbacks;
    this.pane = new Pane({ title: '🌌 Galaxy Controls' });
    this.bloomPassNode = null;
    this.perfParams = { fps: 60 };

    this.setupUI();
  }

  setupUI() {
    this.setupPerformanceFolder();
    this.setupAppearanceFolder();
    this.setupCloudsFolder();
    this.setupBloomFolder();
    this.setupGalaxyFolder();
    this.setupMouseFolder();
  }

  setupPerformanceFolder() {
    const perfFolder = this.pane.addFolder({ title: 'Performance' });
    perfFolder.addBinding(this.perfParams, 'fps', { readonly: true, label: 'FPS' });

    // Star count control
    perfFolder.addBinding(this.config, 'starCount', {
      min: 1000,
      max: 1000000,
      step: 1000,
      label: 'Star Count'
    }).on('change', () => this.callbacks.onStarCountChange(this.config.starCount));
  }

  setupAppearanceFolder() {
    const appearanceFolder = this.pane.addFolder({ title: 'Appearance' });

    appearanceFolder.addBinding(this.config, 'particleSize', {
      min: 0.05,
      max: 0.5,
      step: 0.01,
      label: 'Star Size'
    }).on('change', () => this.callbacks.onUniformChange('particleSize', this.config.particleSize));

    appearanceFolder.addBinding(this.config, 'starBrightness', {
      min: 0.0,
      max: 2.0,
      step: 0.01,
      label: 'Star Brightness'
    }).on('change', () => this.callbacks.onUniformChange('starBrightness', this.config.starBrightness));

    appearanceFolder.addBinding(this.config, 'denseStarColor', {
      label: 'Dense Color',
      view: 'color'
    }).on('change', () => this.callbacks.onUniformChange('denseStarColor', this.config.denseStarColor));

    appearanceFolder.addBinding(this.config, 'sparseStarColor', {
      label: 'Sparse Color',
      view: 'color'
    }).on('change', () => this.callbacks.onUniformChange('sparseStarColor', this.config.sparseStarColor));
  }

  setupCloudsFolder() {
    const cloudsFolder = this.pane.addFolder({ title: 'Clouds' });

    cloudsFolder.addBinding(this.config, 'cloudCount', {
      min: 0,
      max: 100000,
      step: 1000,
      label: 'Count'
    }).on('change', () => this.callbacks.onCloudCountChange(this.config.cloudCount));

    cloudsFolder.addBinding(this.config, 'cloudSize', {
      min: 0.5,
      max: 10.0,
      step: 0.1,
      label: 'Size'
    }).on('change', () => this.callbacks.onUniformChange('cloudSize', this.config.cloudSize));

    cloudsFolder.addBinding(this.config, 'cloudOpacity', {
      min: 0.0,
      max: 1.0,
      step: 0.01,
      label: 'Opacity'
    }).on('change', () => this.callbacks.onUniformChange('cloudOpacity', this.config.cloudOpacity));

    cloudsFolder.addBinding(this.config, 'cloudTintColor', {
      label: 'Tint Color',
      view: 'color'
    }).on('change', () => this.callbacks.onCloudTintChange(this.config.cloudTintColor));
  }

  setupBloomFolder() {
    const bloomFolder = this.pane.addFolder({ title: 'Bloom' });

    bloomFolder.addBinding(this.config, 'bloomStrength', {
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Strength'
    }).on('change', () => this.callbacks.onBloomChange('strength', this.config.bloomStrength));

    bloomFolder.addBinding(this.config, 'bloomRadius', {
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Radius'
    }).on('change', () => this.callbacks.onBloomChange('radius', this.config.bloomRadius));

    bloomFolder.addBinding(this.config, 'bloomThreshold', {
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Threshold'
    }).on('change', () => this.callbacks.onBloomChange('threshold', this.config.bloomThreshold));
  }

  setupGalaxyFolder() {
    const galaxyFolder = this.pane.addFolder({ title: 'Galaxy Structure' });

    galaxyFolder.addBinding(this.config, 'rotationSpeed', {
      min: 0,
      max: 2,
      step: 0.1,
      label: 'Rotation Speed'
    }).on('change', () => this.callbacks.onUniformChange('rotationSpeed', this.config.rotationSpeed));

    galaxyFolder.addBinding(this.config, 'spiralTightness', {
      min: 0,
      max: 10,
      step: 0.1,
      label: 'Spiral Tightness'
    }).on('change', () => this.callbacks.onRegenerate());

    galaxyFolder.addBinding(this.config, 'armCount', {
      min: 1,
      max: 4,
      step: 1,
      label: 'Arm Count'
    }).on('change', () => this.callbacks.onRegenerate());

    galaxyFolder.addBinding(this.config, 'armWidth', {
      min: 1,
      max: 5,
      step: 0.1,
      label: 'Arm Width'
    }).on('change', () => this.callbacks.onRegenerate());

    galaxyFolder.addBinding(this.config, 'randomness', {
      min: 0,
      max: 5,
      step: 0.1,
      label: 'Randomness'
    }).on('change', () => this.callbacks.onRegenerate());

    galaxyFolder.addBinding(this.config, 'galaxyRadius', {
      min: 5,
      max: 20,
      step: 1,
      label: 'Galaxy Radius'
    }).on('change', () => this.callbacks.onRegenerate());

    galaxyFolder.addBinding(this.config, 'galaxyThickness', {
      min: 0.1,
      max: 10,
      step: 0.1,
      label: 'Thickness'
    }).on('change', () => this.callbacks.onRegenerate());
  }

  setupMouseFolder() {
    const mouseFolder = this.pane.addFolder({ title: 'Mouse Interaction' });

    mouseFolder.addBinding(this.config, 'mouseForce', {
      min: 0,
      max: 10,
      step: 0.5,
      label: 'Force'
    }).on('change', () => this.callbacks.onUniformChange('mouseForce', this.config.mouseForce));

    mouseFolder.addBinding(this.config, 'mouseRadius', {
      min: 1,
      max: 15,
      step: 0.5,
      label: 'Radius'
    }).on('change', () => this.callbacks.onUniformChange('mouseRadius', this.config.mouseRadius));
  }

  updateFPS(fps) {
    this.perfParams.fps = fps;
    this.pane.refresh();
  }

  setBloomNode(bloomNode) {
    this.bloomPassNode = bloomNode;
  }
}
