import { Pane } from 'tweakpane';

export class BlackHoleUI {
  constructor(config, callbacks) {
    this.config = config;
    this.callbacks = callbacks;
    this.pane = new Pane({ title: 'Black Hole Controls' });
    this.bloomPassNode = null;
    this.perfParams = { fps: 60 };

    this.setupUI();
  }

  setupUI() {
    this.setupPerformanceFolder();
    this.setupBlackHoleFolder();
    this.setupAccretionDiskFolder();
    this.setupEffectsFolder();
    this.setupBloomFolder();
  }

  setupPerformanceFolder() {
    const perfFolder = this.pane.addFolder({ title: 'Performance' });
    perfFolder.addBinding(this.perfParams, 'fps', { readonly: true, label: 'FPS' });
  }

  setupBlackHoleFolder() {
    const bhFolder = this.pane.addFolder({ title: 'Black Hole' });

    bhFolder.addBinding(this.config, 'blackHoleMass', {
      min: 0.5,
      max: 3.0,
      step: 0.1,
      label: 'Mass'
    }).on('change', () => {
      this.callbacks.onUniformChange('blackHoleMass', this.config.blackHoleMass);
      this.callbacks.onRegenerate();
    });
  }

  setupAccretionDiskFolder() {
    const diskFolder = this.pane.addFolder({ title: 'Accretion Disk' });

    diskFolder.addBinding(this.config, 'diskInnerRadius', {
      min: 1.5,
      max: 5.0,
      step: 0.1,
      label: 'Inner Radius'
    }).on('change', () => this.callbacks.onUniformChange('diskInnerRadius', this.config.diskInnerRadius));

    diskFolder.addBinding(this.config, 'diskOuterRadius', {
      min: 5.0,
      max: 20.0,
      step: 0.5,
      label: 'Outer Radius'
    }).on('change', () => this.callbacks.onUniformChange('diskOuterRadius', this.config.diskOuterRadius));

    diskFolder.addBinding(this.config, 'diskTemperature', {
      min: 0.5,
      max: 3.0,
      step: 0.1,
      label: 'Temperature'
    }).on('change', () => this.callbacks.onUniformChange('diskTemperature', this.config.diskTemperature));

    diskFolder.addBinding(this.config, 'diskBrightness', {
      min: 0.5,
      max: 5.0,
      step: 0.1,
      label: 'Brightness'
    }).on('change', () => this.callbacks.onUniformChange('diskBrightness', this.config.diskBrightness));
  }

  setupEffectsFolder() {
    const effectsFolder = this.pane.addFolder({ title: 'Relativistic Effects' });

    effectsFolder.addBinding(this.config, 'dopplerStrength', {
      min: 0.0,
      max: 2.0,
      step: 0.1,
      label: 'Doppler Beaming'
    }).on('change', () => this.callbacks.onUniformChange('dopplerStrength', this.config.dopplerStrength));
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

  updateFPS(fps) {
    this.perfParams.fps = fps;
    this.pane.refresh();
  }

  setBloomNode(bloomNode) {
    this.bloomPassNode = bloomNode;
  }
}
