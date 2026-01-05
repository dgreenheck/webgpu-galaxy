/**
 * UI Controls for Black Hole Simulation
 *
 * Uses Tweakpane for real-time parameter adjustment.
 * Organized into logical folders matching the simulation components.
 */

import { Pane } from 'tweakpane';
import { QUALITY_PRESETS } from './blackhole.js';

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
    this.setupDiskColorFolder();
    this.setupEffectsFolder();
    this.setupBackgroundFolder();
    this.setupBloomFolder();
  }

  // ==========================================================================
  // PERFORMANCE CONTROLS
  // ==========================================================================

  setupPerformanceFolder() {
    const perfFolder = this.pane.addFolder({ title: 'Performance' });

    // FPS monitor
    perfFolder.addBinding(this.perfParams, 'fps', {
      readonly: true,
      label: 'FPS'
    });

    // Quality preset dropdown
    perfFolder.addBinding(this.config, 'qualityPreset', {
      options: {
        'Low (60 FPS)': 'low',
        'Medium (30-60 FPS)': 'medium',
        'High (30 FPS)': 'high',
        'Ultra (15-30 FPS)': 'ultra'
      },
      label: 'Quality'
    }).on('change', () => {
      this.applyQualityPreset(this.config.qualityPreset);
    });

    // Advanced performance controls (collapsed by default)
    const advancedFolder = perfFolder.addFolder({
      title: 'Advanced',
      expanded: false
    });

    advancedFolder.addBinding(this.config, 'raySteps', {
      min: 32,
      max: 512,
      step: 16,
      label: 'Ray Steps'
    }).on('change', () => {
      this.callbacks.onUniformChange('raySteps', this.config.raySteps);
    });

    advancedFolder.addBinding(this.config, 'stepSize', {
      min: 0.1,
      max: 0.5,
      step: 0.05,
      label: 'Step Size'
    }).on('change', () => {
      this.callbacks.onUniformChange('stepSize', this.config.stepSize);
    });
  }

  applyQualityPreset(presetName) {
    const preset = QUALITY_PRESETS[presetName];
    if (!preset) return;

    // Update config
    this.config.raySteps = preset.raySteps;
    this.config.stepSize = preset.stepSize;
    this.config.starsEnabled = preset.starsEnabled;
    this.config.nebulaEnabled = preset.nebulaEnabled;

    // Apply to simulation
    this.callbacks.onQualityPreset(presetName);

    // Refresh UI to reflect changes
    this.pane.refresh();
  }

  // ==========================================================================
  // BLACK HOLE PHYSICS
  // ==========================================================================

  setupBlackHoleFolder() {
    const bhFolder = this.pane.addFolder({ title: 'Black Hole' });

    bhFolder.addBinding(this.config, 'blackHoleMass', {
      min: 0.5,
      max: 3.0,
      step: 0.1,
      label: 'Mass'
    }).on('change', () => {
      this.callbacks.onUniformChange('blackHoleMass', this.config.blackHoleMass);
    });
  }

  // ==========================================================================
  // ACCRETION DISK GEOMETRY & APPEARANCE
  // ==========================================================================

  setupAccretionDiskFolder() {
    const diskFolder = this.pane.addFolder({ title: 'Accretion Disk' });

    diskFolder.addBinding(this.config, 'diskInnerRadius', {
      min: 2.0,
      max: 5.0,
      step: 0.1,
      label: 'Inner Radius'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskInnerRadius', this.config.diskInnerRadius);
    });

    diskFolder.addBinding(this.config, 'diskOuterRadius', {
      min: 6.0,
      max: 20.0,
      step: 0.5,
      label: 'Outer Radius'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskOuterRadius', this.config.diskOuterRadius);
    });

    diskFolder.addBinding(this.config, 'diskBrightness', {
      min: 0.5,
      max: 5.0,
      step: 0.1,
      label: 'Brightness'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskBrightness', this.config.diskBrightness);
    });

    diskFolder.addBinding(this.config, 'diskRingCount', {
      min: 2,
      max: 20,
      step: 1,
      label: 'Ring Count'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskRingCount', this.config.diskRingCount);
    });

    diskFolder.addBinding(this.config, 'diskTurbulence', {
      min: 0.0,
      max: 1.0,
      step: 0.05,
      label: 'Turbulence'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskTurbulence', this.config.diskTurbulence);
    });

    diskFolder.addBinding(this.config, 'diskTemperature', {
      min: 0.5,
      max: 3.0,
      step: 0.1,
      label: 'Temperature'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskTemperature', this.config.diskTemperature);
    });
  }

  // ==========================================================================
  // DISK COLOR CUSTOMIZATION
  // ==========================================================================

  setupDiskColorFolder() {
    const colorFolder = this.pane.addFolder({
      title: 'Disk Colors',
      expanded: false
    });

    colorFolder.addBinding(this.config, 'diskInnerColor', {
      label: 'Inner (Hot)'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskInnerColor', this.config.diskInnerColor);
    });

    colorFolder.addBinding(this.config, 'diskOuterColor', {
      label: 'Outer (Cool)'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskOuterColor', this.config.diskOuterColor);
    });
  }

  // ==========================================================================
  // RELATIVISTIC EFFECTS
  // ==========================================================================

  setupEffectsFolder() {
    const effectsFolder = this.pane.addFolder({ title: 'Relativistic Effects' });

    effectsFolder.addBinding(this.config, 'dopplerStrength', {
      min: 0.0,
      max: 2.0,
      step: 0.1,
      label: 'Doppler Beaming'
    }).on('change', () => {
      this.callbacks.onUniformChange('dopplerStrength', this.config.dopplerStrength);
    });

    effectsFolder.addBinding(this.config, 'photonRingIntensity', {
      min: 0.0,
      max: 2.0,
      step: 0.1,
      label: 'Photon Ring'
    }).on('change', () => {
      this.callbacks.onUniformChange('photonRingIntensity', this.config.photonRingIntensity);
    });
  }

  // ==========================================================================
  // BACKGROUND (STARS & NEBULA)
  // ==========================================================================

  setupBackgroundFolder() {
    const bgFolder = this.pane.addFolder({
      title: 'Background',
      expanded: false
    });

    bgFolder.addBinding(this.config, 'starsEnabled', {
      label: 'Enable Stars'
    }).on('change', () => {
      this.callbacks.onUniformChange('starsEnabled', this.config.starsEnabled);
    });

    bgFolder.addBinding(this.config, 'starDensity', {
      min: 0.001,
      max: 0.01,
      step: 0.001,
      label: 'Star Density'
    }).on('change', () => {
      this.callbacks.onUniformChange('starDensity', this.config.starDensity);
    });

    bgFolder.addBinding(this.config, 'nebulaEnabled', {
      label: 'Enable Nebula'
    }).on('change', () => {
      this.callbacks.onUniformChange('nebulaEnabled', this.config.nebulaEnabled);
    });

    bgFolder.addBinding(this.config, 'nebulaBrightness', {
      min: 0.0,
      max: 0.5,
      step: 0.05,
      label: 'Nebula Brightness'
    }).on('change', () => {
      this.callbacks.onUniformChange('nebulaBrightness', this.config.nebulaBrightness);
    });
  }

  // ==========================================================================
  // BLOOM POST-PROCESSING
  // ==========================================================================

  setupBloomFolder() {
    const bloomFolder = this.pane.addFolder({ title: 'Bloom' });

    bloomFolder.addBinding(this.config, 'bloomStrength', {
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Strength'
    }).on('change', () => {
      this.callbacks.onBloomChange('strength', this.config.bloomStrength);
    });

    bloomFolder.addBinding(this.config, 'bloomRadius', {
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Radius'
    }).on('change', () => {
      this.callbacks.onBloomChange('radius', this.config.bloomRadius);
    });

    bloomFolder.addBinding(this.config, 'bloomThreshold', {
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Threshold'
    }).on('change', () => {
      this.callbacks.onBloomChange('threshold', this.config.bloomThreshold);
    });
  }

  // ==========================================================================
  // PUBLIC METHODS
  // ==========================================================================

  updateFPS(fps) {
    this.perfParams.fps = fps;
    this.pane.refresh();
  }

  setBloomNode(bloomNode) {
    this.bloomPassNode = bloomNode;
  }
}
