/**
 * TSL Shader Helper Functions for Galaxy Simulation
 *
 * This module contains reusable shader functions written in Three.js Shading Language (TSL).
 * These functions run on the GPU and handle particle physics, rotation, and positioning.
 */

import {
  vec3,
  float,
  Fn,
  length,
  normalize,
  sin,
  cos,
  fract
} from 'three/tsl';

// ==============================================================================
// RANDOM NUMBER GENERATION
// ==============================================================================

/**
 * Improved hash function for pseudo-random number generation
 * Avoids precision loss with large seed values by normalizing first
 *
 * @param {float} seed - Random seed value
 * @returns {float} - Random value between 0 and 1
 */
export const hash = Fn(([seed]) => {
  const p = fract(seed.mul(0.1031));
  const h = p.add(19.19);
  const x = fract(h.mul(h.add(47.43)).mul(p));
  return x;
});

// ==============================================================================
// ROTATION & PHYSICS
// ==============================================================================

/**
 * Rotates a 2D position (x, z) around the Y-axis using rotation matrix:
 *
 * | cos(θ)  -sin(θ) |   | x |
 * | sin(θ)   cos(θ) | * | z |
 *
 * This is a pedagogical implementation showing the rotation matrix clearly.
 *
 * @param {vec3} position - 3D position to rotate
 * @param {float} angle - Rotation angle in radians (counter-clockwise)
 * @returns {vec3} - Rotated position
 */
export const rotateXZ = Fn(([position, angle]) => {
  const cosTheta = cos(angle);
  const sinTheta = sin(angle);

  const newX = position.x.mul(cosTheta).sub(position.z.mul(sinTheta));
  const newZ = position.x.mul(sinTheta).add(position.z.mul(cosTheta));

  return vec3(newX, position.y, newZ);
});

/**
 * Applies differential rotation based on distance from center
 * Inner regions rotate faster than outer regions (like a real galaxy)
 *
 * The rotation factor uses: 1 / (distance * 0.1 + 1)
 * This creates faster rotation near the center, slower at the edges.
 *
 * @param {vec3} position - Current position
 * @param {float} rotationSpeed - Base rotation speed
 * @param {float} deltaTime - Time step
 * @returns {vec3} - Rotated position
 */
export const applyDifferentialRotation = Fn(([position, rotationSpeed, deltaTime]) => {
  // Calculate rotation factor: inner regions rotate faster
  const distFromCenter = length(vec3(position.x, 0, position.z));
  const rotationFactor = float(1.0).div(distFromCenter.mul(0.1).add(1.0));

  // Calculate angular speed and apply rotation
  const angularSpeed = rotationSpeed.mul(rotationFactor).mul(deltaTime).negate();

  return rotateXZ(position, angularSpeed);
});

/**
 * Calculates mouse interaction force
 * Repels particles away from mouse position with falloff based on distance
 *
 * Force = direction * strength * influence * deltaTime
 * Influence = 1 - (distance / radius), clamped to [0, 1]
 *
 * @param {vec3} position - Particle position
 * @param {vec3} mouse - Mouse world position
 * @param {float} mouseActive - Whether mouse is pressed (0 or 1)
 * @param {float} mouseForce - Strength of mouse force
 * @param {float} mouseRadius - Radius of mouse influence
 * @param {float} deltaTime - Time step
 * @returns {vec3} - Force vector to apply
 */
export const applyMouseForce = Fn(([position, mouse, mouseActive, mouseForce, mouseRadius, deltaTime]) => {
  const toMouse = mouse.sub(position);
  const distToMouse = length(toMouse);

  // Calculate influence with distance falloff
  const mouseInfluence = mouseActive.mul(
    float(1.0).sub(distToMouse.div(mouseRadius)).max(0.0)
  );

  // Push particles away from mouse (negate direction)
  const mouseDir = normalize(toMouse);
  return mouseDir.mul(mouseForce).mul(mouseInfluence).mul(deltaTime).negate();
});

/**
 * Applies spring force to restore particle to original position
 * Uses Hooke's law: F = k * (target - current)
 *
 * @param {vec3} currentPos - Current particle position
 * @param {vec3} targetPos - Target (original) position
 * @param {float} strength - Spring strength constant
 * @param {float} deltaTime - Time step
 * @returns {vec3} - Force vector to apply
 */
export const applySpringForce = Fn(([currentPos, targetPos, strength, deltaTime]) => {
  const toTarget = targetPos.sub(currentPos);
  return toTarget.mul(strength).mul(deltaTime);
});
