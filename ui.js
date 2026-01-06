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
    this.setupConfigFolder();
    this.setupPerformanceFolder();
    this.setupBlackHoleFolder();
    this.setupAccretionDiskFolder();
    // Note: Disk color folder removed - colors are now computed from blackbody radiation
    this.setupEffectsFolder();
    this.setupStarsFolder();
    this.setupNebulaFolder();
    this.setupBloomFolder();
  }

  // ==========================================================================
  // CONFIGURATION MANAGEMENT
  // ==========================================================================

  setupConfigFolder() {
    const configFolder = this.pane.addFolder({
      title: 'Save/Load',
      expanded: true
    });

    // Button params (Tweakpane buttons need a dummy object)
    const buttonParams = {
      save: () => {
        this.callbacks.onSaveConfig?.();
        this.showNotification('Settings saved!');
      },
      clear: () => {
        if (confirm('Clear saved settings and reload with defaults?')) {
          this.callbacks.onClearConfig?.();
        }
      },
      reset: () => {
        if (confirm('Reset all settings to defaults?')) {
          this.callbacks.onResetToDefaults?.();
          this.pane.refresh();
          this.showNotification('Reset to defaults');
        }
      }
    };

    configFolder.addButton({
      title: 'Save Settings'
    }).on('click', buttonParams.save);

    configFolder.addButton({
      title: 'Clear & Reload'
    }).on('click', buttonParams.clear);

    configFolder.addButton({
      title: 'Reset to Defaults'
    }).on('click', buttonParams.reset);
  }

  /**
   * Show a temporary notification message.
   */
  showNotification(message) {
    // Create notification element
    const notification = document.createElement('div');
    notification.textContent = message;
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(0, 0, 0, 0.8);
      color: #0f0;
      padding: 12px 24px;
      border-radius: 4px;
      font-family: monospace;
      font-size: 14px;
      z-index: 10000;
      pointer-events: none;
      opacity: 1;
      transition: opacity 0.3s ease;
    `;
    document.body.appendChild(notification);

    // Fade out and remove
    setTimeout(() => {
      notification.style.opacity = '0';
      setTimeout(() => notification.remove(), 300);
    }, 1500);
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

    advancedFolder.addBinding(this.config, 'stepSize', {
      min: 0.1,
      max: 1,
      step: 0.05,
      label: 'Step Size'
    }).on('change', () => {
      this.callbacks.onUniformChange('stepSize', this.config.stepSize);
    });

    advancedFolder.addBinding(this.config, 'adaptiveMinStep', {
      min: 0.05,
      max: 0.5,
      step: 0.05,
      label: 'Min Step Factor'
    }).on('change', () => {
      this.callbacks.onUniformChange('adaptiveMinStep', this.config.adaptiveMinStep);
    });

    advancedFolder.addBinding(this.config, 'stepJitter', {
      min: 0,
      max: 1.0,
      step: 0.05,
      label: 'Sample Jitter'
    }).on('change', () => {
      this.callbacks.onUniformChange('stepJitter', this.config.stepJitter);
    });

  }

  applyQualityPreset(presetName) {
    const preset = QUALITY_PRESETS[presetName];
    if (!preset) return;

    // Update config
    this.config.stepSize = preset.stepSize;
    this.config.starsEnabled = preset.starsEnabled;
    this.config.nebulaEnabled = preset.nebulaEnabled;
    this.config.stepJitter = preset.stepJitter ?? 0.25;

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
      min: 0.1,
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

    // === Geometry ===
    const geometryFolder = diskFolder.addFolder({
      title: 'Geometry',
      expanded: false
    });

    geometryFolder.addBinding(this.config, 'diskInnerRadius', {
      min: 2.0,
      max: 5.0,
      step: 0.1,
      label: 'Inner Radius'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskInnerRadius', this.config.diskInnerRadius);
    });

    geometryFolder.addBinding(this.config, 'diskOuterRadius', {
      min: 6.0,
      max: 20.0,
      step: 0.5,
      label: 'Outer Radius'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskOuterRadius', this.config.diskOuterRadius);
    });

    geometryFolder.addBinding(this.config, 'diskInnerThickness', {
      min: 0.05,
      max: 5.0,
      step: 0.05,
      label: 'Inner Thickness'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskInnerThickness', this.config.diskInnerThickness);
    });

    geometryFolder.addBinding(this.config, 'diskOuterThickness', {
      min: 0.1,
      max: 2.0,
      step: 0.05,
      label: 'Outer Thickness'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskOuterThickness', this.config.diskOuterThickness);
    });

    // === Appearance ===
    const appearanceFolder = diskFolder.addFolder({
      title: 'Appearance',
      expanded: true
    });

    appearanceFolder.addBinding(this.config, 'diskBrightness', {
      min: 0.5,
      max: 5.0,
      step: 0.1,
      label: 'Brightness'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskBrightness', this.config.diskBrightness);
    });

    appearanceFolder.addBinding(this.config, 'diskTemperature', {
      min: 1,
      max: 50,
      step: 1,
      label: 'Peak Temp (kK)',
      format: (v) => `${v.toFixed(0)}k K`
    }).on('change', () => {
      this.callbacks.onUniformChange('diskTemperature', this.config.diskTemperature);
    });

    appearanceFolder.addBinding(this.config, 'temperatureFalloff', {
      min: 0.25,
      max: 5.0,
      step: 0.01,
      label: 'Temp Falloff'
    }).on('change', () => {
      this.callbacks.onUniformChange('temperatureFalloff', this.config.temperatureFalloff);
    });

    // === Ring Pattern ===
    const ringFolder = diskFolder.addFolder({
      title: 'Ring Pattern',
      expanded: true
    });

    ringFolder.addBinding(this.config, 'ringEnabled', {
      label: 'Enable'
    }).on('change', () => {
      this.callbacks.onUniformChange('ringEnabled', this.config.ringEnabled);
    });

    ringFolder.addBinding(this.config, 'ringScale', {
      min: 0.1,
      max: 1.0,
      step: 0.01,
      label: 'Scale'
    }).on('change', () => {
      this.callbacks.onUniformChange('ringScale', this.config.ringScale);
    });

    ringFolder.addBinding(this.config, 'ringContrast', {
      min: 0.0,
      max: 2.0,
      step: 0.05,
      label: 'Contrast'
    }).on('change', () => {
      this.callbacks.onUniformChange('ringContrast', this.config.ringContrast);
    });

    ringFolder.addBinding(this.config, 'ringBrightness', {
      min: -1.0,
      max: 2.0,
      step: 0.05,
      label: 'Brightness'
    }).on('change', () => {
      this.callbacks.onUniformChange('ringBrightness', this.config.ringBrightness);
    });

    ringFolder.addBinding(this.config, 'ringSharpness', {
      min: 0.1,
      max: 10.0,
      step: 0.1,
      label: 'Sharpness'
    }).on('change', () => {
      this.callbacks.onUniformChange('ringSharpness', this.config.ringSharpness);
    });

    // === Rotational Dynamics ===
    const rotationFolder = diskFolder.addFolder({
      title: 'Rotational Dynamics',
      expanded: true
    });

    rotationFolder.addBinding(this.config, 'diskRotationSpeed', {
      min: -20.0,
      max: 20.0,
      step: 0.01,
      label: 'Keplerian Speed'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskRotationSpeed', this.config.diskRotationSpeed);
    });

    rotationFolder.addBinding(this.config, 'ringTwist', {
      min: 1.0,
      max: 20.0,
      step: 0.5,
      label: 'Arc Stretch'
    }).on('change', () => {
      this.callbacks.onUniformChange('ringTwist', this.config.ringTwist);
    });

    rotationFolder.addBinding(this.config, 'noiseAnimFrequency', {
      min: 0.0,
      max: 5.0,
      step: 0.1,
      label: 'Anim Frequency'
    }).on('change', () => {
      this.callbacks.onUniformChange('noiseAnimFrequency', this.config.noiseAnimFrequency);
    });

    rotationFolder.addBinding(this.config, 'noiseAnimAmplitude', {
      min: 0.0,
      max: 2.0,
      step: 0.1,
      label: 'Anim Amplitude'
    }).on('change', () => {
      this.callbacks.onUniformChange('noiseAnimAmplitude', this.config.noiseAnimAmplitude);
    });

    // === Edge Falloff Controls ===
    const edgeFolder = diskFolder.addFolder({
      title: 'Edge Falloff',
      expanded: false
    });

    edgeFolder.addBinding(this.config, 'diskEdgeSoftnessInner', {
      min: 0.0,
      max: 0.5,
      step: 0.01,
      label: 'Inner Softness'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskEdgeSoftnessInner', this.config.diskEdgeSoftnessInner);
    });

    edgeFolder.addBinding(this.config, 'diskEdgeSoftnessOuter', {
      min: 0.0,
      max: 0.5,
      step: 0.01,
      label: 'Outer Softness'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskEdgeSoftnessOuter', this.config.diskEdgeSoftnessOuter);
    });

    edgeFolder.addBinding(this.config, 'diskRadialFalloff', {
      min: 0.1,
      max: 2.0,
      step: 0.1,
      label: 'Radial Falloff'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskRadialFalloff', this.config.diskRadialFalloff);
    });

    // === Volumetric Rendering ===
    const volFolder = diskFolder.addFolder({
      title: 'Volumetric',
      expanded: false
    });

    volFolder.addBinding(this.config, 'diskDensity', {
      min: 0.05,
      max: 1,
      step: 0.01,
      label: 'Density'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskDensity', this.config.diskDensity);
    });

    volFolder.addBinding(this.config, 'diskOpacityFalloff', {
      min: 0.3,
      max: 1.0,
      step: 0.05,
      label: 'Opacity Falloff'
    }).on('change', () => {
      this.callbacks.onUniformChange('diskOpacityFalloff', this.config.diskOpacityFalloff);
    });
  }

  // Note: Disk color is now computed from blackbody radiation based on temperature

  // ==========================================================================
  // RELATIVISTIC EFFECTS
  // ==========================================================================

  setupEffectsFolder() {
    const effectsFolder = this.pane.addFolder({ title: 'Relativistic Effects' });

    effectsFolder.addBinding(this.config, 'gravitationalLensing', {
      min: 0.5,
      max: 3.0,
      step: 0.1,
      label: 'Grav. Lensing'
    }).on('change', () => {
      this.callbacks.onUniformChange('gravitationalLensing', this.config.gravitationalLensing);
    });
  }

  // ==========================================================================
  // STARS
  // ==========================================================================

  setupStarsFolder() {
    const starsFolder = this.pane.addFolder({
      title: 'Stars',
      expanded: false
    });

    starsFolder.addBinding(this.config, 'starsEnabled', {
      label: 'Enable Stars'
    }).on('change', () => {
      this.callbacks.onUniformChange('starsEnabled', this.config.starsEnabled);
    });

    starsFolder.addBinding(this.config, 'starBackgroundColor', {
      label: 'Background'
    }).on('change', () => {
      this.callbacks.onUniformChange('starBackgroundColor', this.config.starBackgroundColor);
    });

    starsFolder.addBinding(this.config, 'starDensity', {
      min: 0.001,
      max: 0.1,
      step: 0.001,
      label: 'Density'
    }).on('change', () => {
      this.callbacks.onUniformChange('starDensity', this.config.starDensity);
    });

    starsFolder.addBinding(this.config, 'starSize', {
      min: 0.5,
      max: 5.0,
      step: 0.1,
      label: 'Size'
    }).on('change', () => {
      this.callbacks.onUniformChange('starSize', this.config.starSize);
    });

    starsFolder.addBinding(this.config, 'starBrightness', {
      min: 0.1,
      max: 3.0,
      step: 0.1,
      label: 'Brightness'
    }).on('change', () => {
      this.callbacks.onUniformChange('starBrightness', this.config.starBrightness);
    });
  }

  // ==========================================================================
  // NEBULA
  // ==========================================================================

  setupNebulaFolder() {
    const nebulaFolder = this.pane.addFolder({
      title: 'Nebula',
      expanded: false
    });

    nebulaFolder.addBinding(this.config, 'nebulaEnabled', {
      label: 'Enable Nebula'
    }).on('change', () => {
      this.callbacks.onUniformChange('nebulaEnabled', this.config.nebulaEnabled);
    });

    // Layer 1 subfolder
    const layer1Folder = nebulaFolder.addFolder({
      title: 'Layer 1',
      expanded: false
    });

    layer1Folder.addBinding(this.config, 'nebula1Scale', {
      min: 0.5,
      max: 10.0,
      step: 0.5,
      label: 'Scale'
    }).on('change', () => {
      this.callbacks.onUniformChange('nebula1Scale', this.config.nebula1Scale);
    });

    layer1Folder.addBinding(this.config, 'nebula1Density', {
      min: -1.0,
      max: 1.0,
      step: 0.05,
      label: 'Density'
    }).on('change', () => {
      this.callbacks.onUniformChange('nebula1Density', this.config.nebula1Density);
    });

    layer1Folder.addBinding(this.config, 'nebula1Brightness', {
      min: 0.0,
      max: 1.0,
      step: 0.01,
      label: 'Brightness'
    }).on('change', () => {
      this.callbacks.onUniformChange('nebula1Brightness', this.config.nebula1Brightness);
    });

    layer1Folder.addBinding(this.config, 'nebula1Color', {
      label: 'Color'
    }).on('change', () => {
      this.callbacks.onUniformChange('nebula1Color', this.config.nebula1Color);
    });

    // Layer 2 subfolder
    const layer2Folder = nebulaFolder.addFolder({
      title: 'Layer 2',
      expanded: false
    });

    layer2Folder.addBinding(this.config, 'nebula2Scale', {
      min: 0.5,
      max: 20.0,
      step: 0.5,
      label: 'Scale'
    }).on('change', () => {
      this.callbacks.onUniformChange('nebula2Scale', this.config.nebula2Scale);
    });

    layer2Folder.addBinding(this.config, 'nebula2Density', {
      min: -1.0,
      max: 1.0,
      step: 0.05,
      label: 'Density'
    }).on('change', () => {
      this.callbacks.onUniformChange('nebula2Density', this.config.nebula2Density);
    });

    layer2Folder.addBinding(this.config, 'nebula2Brightness', {
      min: 0.0,
      max: 1.0,
      step: 0.01,
      label: 'Brightness'
    }).on('change', () => {
      this.callbacks.onUniformChange('nebula2Brightness', this.config.nebula2Brightness);
    });

    layer2Folder.addBinding(this.config, 'nebula2Color', {
      label: 'Color'
    }).on('change', () => {
      this.callbacks.onUniformChange('nebula2Color', this.config.nebula2Color);
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
