/**
 * ============================================================================
 * BLACK HOLE TSL SHADER
 * ============================================================================
 *
 * Raymarching shader for black hole visualization using Three.js TSL.
 * Separated from the main simulation class for better code organization.
 *
 * @author Daniel Greenheck
 * @see BLOG.md for detailed explanation
 */

import {
  vec2,
  vec3,
  vec4,
  float,
  Fn,
  length,
  normalize,
  cross,
  dot,
  sin,
  cos,
  atan,
  asin,
  sqrt,
  abs,
  pow,
  fract,
  clamp,
  smoothstep,
  mix,
  floor,
  step,
  Loop,
  Break,
  If,
  screenUV
} from 'three/tsl';

// ============================================================================
// SECTION 1: UTILITY FUNCTIONS
// ============================================================================

/**
 * Hash function for pseudo-random number generation.
 * Used for procedural star and noise generation.
 */
const hash21 = Fn(([p]) => {
  const n = sin(dot(p, vec2(127.1, 311.7))).mul(43758.5453);
  return fract(n);
});

const hash31 = Fn(([p]) => {
  const n = sin(dot(p, vec3(127.1, 311.7, 74.7))).mul(43758.5453);
  return fract(n);
});

const hash33 = Fn(([p]) => {
  const px = fract(sin(dot(p, vec3(127.1, 311.7, 74.7))).mul(43758.5453));
  const py = fract(sin(dot(p, vec3(269.5, 183.3, 246.1))).mul(43758.5453));
  const pz = fract(sin(dot(p, vec3(113.5, 271.9, 124.6))).mul(43758.5453));
  return vec3(px, py, pz);
});

/**
 * 3D Value noise for turbulence effects.
 */
const noise3D = Fn(([p]) => {
  const i = floor(p);
  const f = fract(p);

  // Smooth interpolation
  const u = f.mul(f).mul(float(3.0).sub(f.mul(2.0)));

  // Hash corners
  const a = hash31(i);
  const b = hash31(i.add(vec3(1, 0, 0)));
  const c = hash31(i.add(vec3(0, 1, 0)));
  const d = hash31(i.add(vec3(1, 1, 0)));
  const e = hash31(i.add(vec3(0, 0, 1)));
  const f2 = hash31(i.add(vec3(1, 0, 1)));
  const g = hash31(i.add(vec3(0, 1, 1)));
  const h = hash31(i.add(vec3(1, 1, 1)));

  // Trilinear interpolation
  return mix(
    mix(mix(a, b, u.x), mix(c, d, u.x), u.y),
    mix(mix(e, f2, u.x), mix(g, h, u.x), u.y),
    u.z
  );
});

/**
 * Fractal Brownian Motion - layered noise for natural-looking turbulence.
 * @param p - 3D position
 */
const fbm = Fn(([p]) => {
  const value = float(0.0).toVar();
  const amplitude = float(0.5).toVar();
  const pos = p.toVar();

  // 4 octaves of noise
  value.addAssign(noise3D(pos).mul(amplitude));
  pos.mulAssign(2.0);
  amplitude.mulAssign(0.5);

  value.addAssign(noise3D(pos).mul(amplitude));
  pos.mulAssign(2.0);
  amplitude.mulAssign(0.5);

  value.addAssign(noise3D(pos).mul(amplitude));
  pos.mulAssign(2.0);
  amplitude.mulAssign(0.5);

  value.addAssign(noise3D(pos).mul(amplitude));

  return value;
});

// ============================================================================
// SECTION 2: PROCEDURAL BACKGROUND
// ============================================================================

/**
 * Generate procedural star field.
 * Stars are placed using a grid-based hash function for consistent positions.
 */
const createStarField = (uniforms) => Fn(([rayDir]) => {
  // Convert ray direction to spherical coordinates for grid
  const theta = atan(rayDir.z, rayDir.x);
  const phi = asin(clamp(rayDir.y, float(-1.0), float(1.0)));

  // Create grid cells - lower scale = larger cells = bigger stars
  const gridScale = float(60.0).div(uniforms.starSize);
  const cell = vec2(theta, phi).mul(gridScale).floor();
  const cellUV = fract(vec2(theta, phi).mul(gridScale));

  // Hash for this cell
  const cellHash = hash21(cell);

  // Star probability - most cells are empty
  const starProb = step(float(1.0).sub(uniforms.starDensity), cellHash);

  // Star position within cell
  const starPos = hash33(vec3(cell.x, cell.y, float(42.0))).xy.mul(0.8).add(0.1);
  const distToStar = length(cellUV.sub(starPos));

  // Star size with variation - base size scaled by uniform
  const baseSizeVar = hash21(cell.add(100.0)).mul(0.03).add(0.01);
  const finalStarSize = baseSizeVar.mul(uniforms.starSize);

  // Star brightness with soft glow
  const starCore = smoothstep(finalStarSize, float(0.0), distToStar);
  const starGlow = smoothstep(finalStarSize.mul(3.0), float(0.0), distToStar).mul(0.3);
  const starIntensity = starCore.add(starGlow).mul(starProb);

  // Star color variation (blue to yellow)
  const colorTemp = hash21(cell.add(200.0));
  const starColor = mix(
    vec3(0.8, 0.9, 1.0),  // Blue-white
    vec3(1.0, 0.95, 0.8), // Yellow-white
    colorTemp
  );

  return starColor.mul(starIntensity).mul(uniforms.starBrightness);
});

/**
 * Generate procedural nebula clouds.
 * Uses two noise layers at different frequencies for depth.
 */
const createNebulaField = (uniforms) => Fn(([rayDir, time]) => {
  // Layer 1: Large scale structures
  const noisePos1 = rayDir.mul(uniforms.nebulaScale1);
  const n1 = fbm(noisePos1.add(time.mul(uniforms.nebulaSpeed))).mul(2.0).sub(1.0);

  // Layer 2: Higher frequency detail
  const noisePos2 = rayDir.mul(uniforms.nebulaScale2);
  const n2 = fbm(noisePos2.sub(time.mul(uniforms.nebulaSpeed.mul(0.5)))).mul(2.0).sub(1.0);

  // Combine layers - blend controls mix, density offsets for visibility threshold
  const layer1Weight = float(1.0).sub(uniforms.nebulaBlend);
  const combined = n1.mul(layer1Weight).add(n2.mul(uniforms.nebulaBlend)).add(uniforms.nebulaDensity);
  const nebula = clamp(combined, float(0.0), float(1.0));

  // Color gradient based on first noise layer
  const colorMix = n1.mul(0.5).add(0.5);
  const nebulaColor = mix(uniforms.nebulaColor1, uniforms.nebulaColor2, colorMix);

  return nebulaColor.mul(nebula).mul(uniforms.nebulaBrightness);
});

// ============================================================================
// SECTION 3: ACCRETION DISK
// ============================================================================

/**
 * Calculate the color and opacity of the accretion disk at a given point.
 * Returns vec4(color.rgb, opacity) where ring patterns control opacity.
 */
const createAccretionDiskColor = (uniforms) => Fn(([hitR, hitAngle, time]) => {
  const innerR = uniforms.diskInnerRadius;
  const outerR = uniforms.diskOuterRadius;

  // Normalized radius (0 at inner edge, 1 at outer edge)
  const normR = clamp(hitR.sub(innerR).div(outerR.sub(innerR)), float(0.0), float(1.0));

  // === BLACKBODY DISK COLOR (pure radial temperature profile) ===
  // Shakura-Sunyaev thin disk: T ~ r^(-3/4)
  const temperature = pow(normR.add(0.05), float(-0.75)).mul(uniforms.diskTemperature);

  // Interpolate between user-defined inner/outer colors based on temperature
  const colorMix = smoothstep(float(0.5), float(2.5), temperature);
  const diskColor = mix(uniforms.diskOuterColor, uniforms.diskInnerColor, colorMix);

  // Edge falloff - disk fades at boundaries
  const edgeFalloff = smoothstep(float(0.0), uniforms.diskEdgeSoftnessInner, normR)
    .mul(smoothstep(float(1.0), float(1.0).sub(uniforms.diskEdgeSoftnessOuter), normR));

  // === RING PATTERN ===
  const ringOpacity = float(1.0).toVar('ringOpacity');

  If(uniforms.ringEnabled.greaterThan(0.5), () => {
    // Uniform rotation
    const rotation = time.mul(uniforms.diskRotationSpeed);

    // Static twist based on radius
    const staticTwist = hitR.add(1.0).log().mul(uniforms.ringTwist);

    // Combined angle
    const sampleAngle = hitAngle.add(rotation).add(staticTwist);

    // Cartesian coordinates for noise
    const noiseX = hitR.mul(cos(sampleAngle));
    const noiseY = hitR.mul(sin(sampleAngle));

    // Domain warping: sample a second noise field to generate Z coordinate
    // As the disk rotates, noiseX/noiseY change, causing zWarp to evolve smoothly
    // This creates organic animation tied to rotation without decorrelation
    const warpCoord = vec3(noiseX, noiseY, hitR).mul(uniforms.noiseAnimFrequency);
    const zWarp = fbm(warpCoord).mul(uniforms.noiseAnimAmplitude);

    // Sample 3D FBM noise with warped Z
    const noiseCoord = vec3(noiseX, noiseY, zWarp).mul(uniforms.ringScale);
    const ringNoise = fbm(noiseCoord);

    // Apply contrast, brightness, and sharpness
    const rawRing = ringNoise.mul(uniforms.ringContrast).add(uniforms.ringBrightness);
    ringOpacity.assign(pow(clamp(rawRing, float(0.0), float(1.0)), uniforms.ringSharpness));
  });

  const finalOpacity = ringOpacity;

  // Return vec4: rgb = disk color (with edge falloff and brightness), a = opacity
  const finalColor = diskColor.mul(edgeFalloff).mul(uniforms.diskBrightness);
  return vec4(finalColor, finalOpacity);
});

// ============================================================================
// SECTION 4: MAIN RAYMARCHING SHADER
// ============================================================================

/**
 * Create the complete black hole raymarching shader.
 * @param {Object} uniforms - Shader uniforms object
 * @returns {Function} TSL shader function
 */
export function createBlackHoleShader(uniforms) {
  // Create shader functions that depend on uniforms
  const starField = createStarField(uniforms);
  const nebulaField = createNebulaField(uniforms);
  const accretionDiskColor = createAccretionDiskColor(uniforms);

  return Fn(() => {
    // === SCHWARZSCHILD PARAMETERS ===
    const rs = uniforms.blackHoleMass.mul(2.0);

    // === CAMERA SETUP ===
    const uv = screenUV.sub(0.5).mul(2.0);
    const aspect = uniforms.resolution.x.div(uniforms.resolution.y);
    const screenPos = vec2(uv.x.mul(aspect), uv.y);

    const camPos = uniforms.cameraPosition;
    const camTarget = uniforms.cameraTarget;

    // Build camera coordinate system
    const camForward = normalize(camTarget.sub(camPos));
    const worldUp = vec3(0.0, 1.0, 0.0);
    const camRight = normalize(cross(worldUp, camForward));
    const camUp = cross(camForward, camRight);

    // Generate ray direction through pixel
    const fov = float(1.0);
    const rayDir = normalize(
      camForward.mul(fov)
        .add(camRight.mul(screenPos.x))
        .add(camUp.mul(screenPos.y))
    ).toVar('rayDir');

    // === INITIALIZE RAY STATE ===
    const rayPos = camPos.toVar('rayPos');

    // Accumulated color with alpha for blending
    const color = vec3(0.0, 0.0, 0.0).toVar('color');
    const alpha = float(0.0).toVar('alpha');

    // Ray status
    const escaped = float(0.0).toVar('escaped');
    const captured = float(0.0).toVar('captured');

    // Disk parameters
    const innerR = uniforms.diskInnerRadius;
    const outerR = uniforms.diskOuterRadius;

    // === RAYMARCHING LOOP ===
    Loop(128, () => {
      // Check if we've already terminated
      If(escaped.greaterThan(0.5).or(captured.greaterThan(0.5)).or(alpha.greaterThan(0.99)), () => {
        Break();
      });

      const r = length(rayPos);

      // === TERMINATION: CAPTURED BY BLACK HOLE ===
      If(r.lessThan(rs.mul(1.01)), () => {
        captured.assign(1.0);
        Break();
      });

      // === TERMINATION: ESCAPED TO INFINITY ===
      If(r.greaterThan(100.0), () => {
        escaped.assign(1.0);
        Break();
      });

      // === ADAPTIVE STEP SIZE ===
      // Factor 1: Distance from event horizon
      const distFromHorizon = r.sub(rs);
      const horizonFactor = smoothstep(float(0.0), rs.mul(5.0), distFromHorizon)
        .mul(0.8).add(0.2);

      // Factor 2 & 3: Disk proximity with thickness awareness
      const rHoriz = sqrt(rayPos.x.mul(rayPos.x).add(rayPos.z.mul(rayPos.z)));

      const diskMargin = float(2.0);
      const inDiskRegion = smoothstep(innerR.sub(diskMargin.mul(2.0)), innerR.sub(diskMargin), rHoriz)
        .mul(smoothstep(outerR.add(diskMargin.mul(2.0)), outerR.add(diskMargin), rHoriz));

      // Calculate local disk thickness at this radius
      const normRForThickness = clamp(
        rHoriz.sub(innerR).div(outerR.sub(innerR)),
        float(0.0), float(1.0)
      );
      const localThickness = mix(
        uniforms.diskInnerThickness,
        uniforms.diskOuterThickness,
        normRForThickness
      );

      // Reduce step size near disk plane
      const approachDistance = float(3.0);
      const distToPlane = abs(rayPos.y);
      const thicknessScale = localThickness.max(0.05);
      const diskProximity = distToPlane.div(thicknessScale.mul(approachDistance));
      const minStep = uniforms.adaptiveMinStep;
      const diskFactor = smoothstep(float(0.0), float(1.0), diskProximity)
        .mul(float(1.0).sub(minStep)).add(minStep);

      // Combine factors
      const combinedDiskFactor = mix(float(1.0), diskFactor, inDiskRegion);
      const adaptiveStep = uniforms.stepSize.mul(horizonFactor).mul(combinedDiskFactor);

      // === GRAVITATIONAL LIGHT BENDING ===
      const toCenter = rayPos.negate().normalize();
      const bendStrength = rs.div(r.mul(r)).mul(adaptiveStep).mul(uniforms.gravitationalLensing);

      // Apply bending to ray direction
      rayDir.addAssign(toCenter.mul(bendStrength));
      rayDir.assign(normalize(rayDir));

      // Step ray forward
      rayPos.addAssign(rayDir.mul(adaptiveStep));

      // === VOLUMETRIC DISK SAMPLING ===
      const sampleNoise = hash33(rayPos.add(vec3(uniforms.frameIndex.mul(0.1))));
      const jitterOffset = sampleNoise.sub(0.5).mul(adaptiveStep).mul(uniforms.stepJitter);
      const samplePos = rayPos.add(jitterOffset);

      // Check if ray is inside the disk volume
      const hitR = sqrt(samplePos.x.mul(samplePos.x).add(samplePos.z.mul(samplePos.z)));

      // Normalized radius for tapering
      const normR = clamp(hitR.sub(innerR).div(outerR.sub(innerR)), float(0.0), float(1.0));

      // === SOFT RADIAL FALLOFF ===
      const radialFalloffWidth = uniforms.diskRadialFalloff;
      const innerFalloff = smoothstep(
        innerR.sub(radialFalloffWidth),
        innerR.add(radialFalloffWidth),
        hitR
      );
      const outerFalloff = smoothstep(
        outerR.add(radialFalloffWidth),
        outerR.sub(radialFalloffWidth),
        hitR
      );
      const radialDensity = innerFalloff.mul(outerFalloff);

      // === DISK THICKNESS PROFILE ===
      const innerHalf = uniforms.diskInnerThickness.mul(0.5);
      const outerHalf = uniforms.diskOuterThickness.mul(0.5);
      const localHalfThickness = mix(innerHalf, outerHalf, normR);

      // === SOFT HEIGHT FALLOFF ===
      const heightRatio = abs(samplePos.y).div(localHalfThickness.max(0.01));
      const heightDensity = smoothstep(float(1.0), float(0.0), heightRatio);

      // Combined density
      const totalDensity = radialDensity.mul(heightDensity);

      // Only process if density is significant
      If(totalDensity.greaterThan(0.001).and(alpha.lessThan(0.99)), () => {
        const hitAngle = atan(samplePos.z, samplePos.x);

        // Get disk color and turbulence opacity
        const diskResult = accretionDiskColor(hitR, hitAngle, uniforms.time);
        const diskCol = diskResult.xyz;
        const turbOpacity = diskResult.w;

        // === GRAVITATIONAL REDSHIFT ===
        const redshift = sqrt(clamp(float(1.0).sub(rs.div(hitR)), float(0.1), float(1.0)));

        // Volumetric accumulation
        const sampleDensity = totalDensity.mul(uniforms.diskDensity).mul(turbOpacity);
        const contribution = diskCol.mul(redshift).mul(sampleDensity);
        const remainingAlpha = float(1.0).sub(alpha);
        color.addAssign(contribution.mul(remainingAlpha));
        alpha.addAssign(remainingAlpha.mul(sampleDensity.mul(uniforms.diskOpacityFalloff)));
      });
    });

    // After loop: if ray wasn't captured, it escaped
    If(captured.lessThan(0.5), () => {
      escaped.assign(1.0);
    });

    // === BACKGROUND (for escaped rays) ===
    If(escaped.greaterThan(0.5).and(alpha.lessThan(0.99)), () => {
      const bgColor = uniforms.starBackgroundColor.toVar('bgColor');

      // Add stars if enabled
      If(uniforms.starsEnabled.greaterThan(0.5), () => {
        const stars = starField(rayDir);
        bgColor.addAssign(stars);
      });

      // Add nebula if enabled
      If(uniforms.nebulaEnabled.greaterThan(0.5), () => {
        const nebula = nebulaField(rayDir, uniforms.time);
        bgColor.addAssign(nebula);
      });

      // Blend background with accumulated disk color
      color.addAssign(bgColor.mul(float(1.0).sub(alpha)));
    });

    // === TONE MAPPING (ACES Filmic) ===
    const a = float(2.51);
    const b = float(0.03);
    const c = float(2.43);
    const d = float(0.59);
    const e = float(0.14);

    const toneMapped = clamp(
      color.mul(color.mul(a).add(b))
        .div(color.mul(color.mul(c).add(d)).add(e)),
      vec3(0.0),
      vec3(1.0)
    );

    // === GAMMA CORRECTION ===
    const finalColor = pow(toneMapped, vec3(1.0 / 2.2));

    return vec4(finalColor, 1.0);
  })();
}
