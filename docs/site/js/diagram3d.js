// Copyright (c) 2026 Tim Urista. All rights reserved; no license is granted (LICENSE, Part 3).
//
// Structural 3D view of the upstream LittleBitLinear forward pass:
//   y = ((((x * v2) @ Vq.T) * (v1 * u2)) @ Uq.T) * u1
//       + the same with U_R, V_R, u1_R, u2_R, v1_R, v2_R (residual path) + bias
// STRUCTURE ONLY. The sizes below and the sign tiles are placeholders chosen for legibility; they
// are not measured values or model weights. Three.js is loaded from the local vendor copy; no
// external network requests are made. On any failure the static SVG diagram stays in place.

import * as THREE from '../vendor/three.module.js';

const C = {
  paper: 0xFBF8F1,
  ink: 0x14213D,
  cobalt: 0x2747B8,
  teal: 0x0F7C7A,
  tealLight: 0xBFDCD9,
  slate: 0x7C88A8,
  plus: 0xE6E1D3,
};
// Illustrative sizes, not measured: see the figure caption.
const DIMS = { dIn: 12, s: 4, dOut: 10 };
const CELL = 0.36;
const STEP_COUNT = 8;
const DIM_OPACITY = 0.22;
const BASE_ROT = { x: 0.22, y: -0.42 };

function label(text, { size = 0.62, color = '#14213D', weight = 600, mono = false } = {}) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const px = 48;
  const font = weight + ' ' + px + 'px ' + (mono
    ? 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
    : 'system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif');
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 24;
  const h = px + 24;
  canvas.width = w;
  canvas.height = h;
  ctx.font = font;
  ctx.fillStyle = color;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 12, h / 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(size * (w / h), size, 1);
  sprite.renderOrder = 10;
  return sprite;
}

function block(w, h, d, color, edgeColor = C.ink) {
  const geometry = new THREE.BoxGeometry(w, h, d);
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 1 }));
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry),
    new THREE.LineBasicMaterial({ color: edgeColor, transparent: true, opacity: 0.55 }));
  mesh.add(edges);
  return mesh;
}

// A column vector of n cells, centred on the origin.
function vectorNode(n, color) {
  const g = new THREE.Group();
  for (let i = 0; i < n; i += 1) {
    const cell = block(CELL * 0.9, CELL * 0.9, CELL * 0.9, color);
    cell.position.y = (n / 2 - i - 0.5) * CELL;
    g.add(cell);
  }
  return g;
}

// A rows x cols grid of +1/-1 tiles. The pattern is a fixed placeholder, not data.
function signMatrix(rows, cols, seed) {
  const g = new THREE.Group();
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const minus = ((r * 7 + c * 3 + seed) % 5) < 2;
      const tile = block(CELL * 0.9, CELL * 0.9, CELL * 0.35, minus ? C.cobalt : C.plus);
      tile.position.set((c - cols / 2 + 0.5) * CELL, (rows / 2 - r - 0.5) * CELL, 0);
      g.add(tile);
    }
  }
  return g;
}

function connector(a, b) {
  const geometry = new THREE.BufferGeometry().setFromPoints([a, b]);
  return new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: C.ink, transparent: true, opacity: 0.6 }));
}

function tag(object, steps) {
  object.userData.steps = steps;
  return object;
}

function buildScene() {
  const root = new THREE.Group();
  const tagged = [];
  const add = (object, steps) => {
    root.add(tag(object, steps));
    tagged.push(object);
    return object;
  };

  const xIn = add(vectorNode(DIMS.dIn, C.paper), [0, 1, 6]);
  xIn.position.set(-15, 0, 0);
  const xLabel = add(label('x', { mono: true, size: 0.8 }), [0]);
  xLabel.position.set(-15, DIMS.dIn * CELL / 2 + 0.6, 0);

  const paths = [
    { y: 3.3, z: 1.4, suffix: '', steps: [1, 2, 3, 4, 5], pathStep: null, title: 'main path', seed: 0 },
    { y: -3.3, z: -1.4, suffix: '_R', steps: [6, 6, 6, 6, 6], pathStep: 6, title: 'residual path', seed: 2 },
  ];
  const xs = [-11, -6.6, -2.6, 1.4, 5.4];
  const sumPos = new THREE.Vector3(9.6, 0, 0);

  for (const p of paths) {
    const nodes = [
      { obj: vectorNode(DIMS.dIn, C.tealLight), text: '* v2' + p.suffix, h: DIMS.dIn },
      { obj: signMatrix(DIMS.dIn, DIMS.s, p.seed), text: '@ Vq' + p.suffix + '.T', h: DIMS.dIn },
      { obj: vectorNode(DIMS.s, C.tealLight), text: '* (v1' + p.suffix + ' * u2' + p.suffix + ')', h: DIMS.s },
      { obj: signMatrix(DIMS.dOut, DIMS.s, p.seed + 1), text: '@ Uq' + p.suffix + '.T', h: DIMS.dOut },
      { obj: vectorNode(DIMS.dOut, C.tealLight), text: '* u1' + p.suffix, h: DIMS.dOut },
    ];
    const pathTitle = add(label(p.title, { size: 0.5, color: '#555B69' }), p.pathStep === null ? [1, 2, 3, 4, 5] : [6]);
    pathTitle.position.set(xs[0] - 1.2, p.y + DIMS.dIn * CELL / 2 + 1.35, p.z);
    let prev = new THREE.Vector3(-15 + CELL, 0, 0);
    nodes.forEach((n, i) => {
      const steps = [p.steps[i]];
      n.obj.position.set(xs[i], p.y, p.z);
      add(n.obj, steps);
      const l = add(label(n.text, { mono: true, size: 0.5 }), steps);
      l.position.set(xs[i], p.y + n.h * CELL / 2 + 0.5, p.z);
      const here = new THREE.Vector3(xs[i] - CELL, p.y, p.z);
      add(connector(prev, here), steps);
      prev = new THREE.Vector3(xs[i] + CELL, p.y, p.z);
    });
    add(connector(prev, sumPos), [7]);
  }

  const sum = add(block(0.7, 0.7, 0.7, C.paper), [7]);
  sum.position.copy(sumPos);
  const sumLabel = add(label('+ bias', { mono: true, size: 0.5 }), [7]);
  sumLabel.position.set(sumPos.x, sumPos.y - 0.95, 0);
  const yOut = add(vectorNode(DIMS.dOut, C.slate), [7]);
  yOut.position.set(13, 0, 0);
  add(connector(new THREE.Vector3(sumPos.x + 0.4, 0, 0), new THREE.Vector3(13 - CELL, 0, 0)), [7]);
  const yLabel = add(label('y', { mono: true, size: 0.8 }), [7]);
  yLabel.position.set(13, DIMS.dOut * CELL / 2 + 0.6, 0);

  const note = label('structure only: sizes and signs are placeholders', { size: 0.42, color: '#555B69', weight: 400 });
  note.position.set(0, -7.4, 0);
  root.add(note);
  return { root, tagged };
}

function setOpacity(object, factor) {
  object.traverse((o) => {
    if (!o.material) return;
    if (o.userData.baseOpacity === undefined) o.userData.baseOpacity = o.material.opacity;
    o.material.opacity = o.userData.baseOpacity * factor;
  });
}

export function initDiagram({ stage, status, controls, steps }) {
  const fallback = stage.querySelector('#diagram-fallback');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true });
  } catch (err) {
    status.textContent = '3D view unavailable: WebGL could not start in this browser ('
      + (err && err.message ? err.message : 'unknown error') + '). The static diagram above shows the same structure.';
    return;
  }

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(C.paper);
  const camera = new THREE.PerspectiveCamera(30, 2, 0.1, 400);
  const { root, tagged } = buildScene();
  scene.add(root);
  root.rotation.set(BASE_ROT.x, BASE_ROT.y, 0);

  const canvas = renderer.domElement;
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', 'Three-dimensional structural diagram of the main and residual LittleBit paths. '
    + 'The numbered step list below describes the same structure. Sizes are illustrative, not measured.');
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const items = steps ? Array.from(steps.querySelectorAll('li')) : [];
  let current = -1;
  let target = { x: BASE_ROT.x, y: BASE_ROT.y };
  let frame = 0;

  function render() {
    renderer.render(scene, camera);
  }

  function fit() {
    const width = Math.max(240, stage.clientWidth);
    const height = Math.max(260, Math.round(width * 0.5));
    renderer.setSize(width, height);
    camera.aspect = width / height;
    const halfWidth = 17.5;
    const halfHeight = 8.5;
    const vFov = THREE.MathUtils.degToRad(camera.fov);
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
    const distance = Math.max(halfWidth / Math.tan(hFov / 2), halfHeight / Math.tan(vFov / 2)) + 4;
    camera.position.set(0, 0, distance);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
    render();
  }

  function applyRotation() {
    cancelAnimationFrame(frame);
    if (reduceMotion.matches) {
      root.rotation.x = target.x;
      root.rotation.y = target.y;
      render();
      return;
    }
    const from = { x: root.rotation.x, y: root.rotation.y };
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / 280);
      const e = t * (2 - t);
      root.rotation.x = from.x + (target.x - from.x) * e;
      root.rotation.y = from.y + (target.y - from.y) * e;
      render();
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
  }

  function setStep(n) {
    current = Math.max(-1, Math.min(STEP_COUNT - 1, n));
    for (const object of tagged) {
      const on = current === -1 || object.userData.steps.includes(current);
      setOpacity(object, on ? 1 : DIM_OPACITY);
    }
    items.forEach((li, i) => {
      li.classList.toggle('is-current', i === current);
      if (i === current) li.setAttribute('aria-current', 'step');
      else li.removeAttribute('aria-current');
    });
    status.textContent = current === -1
      ? 'Overview: all operations shown. Use Next step to walk through them.'
      : 'Step ' + (current + 1) + ' of ' + STEP_COUNT + ': ' + items[current].textContent;
    render();
  }

  function rotate(dy, dx) {
    target = {
      x: Math.max(-0.7, Math.min(0.7, target.x + dx)),
      y: target.y + dy,
    };
    applyRotation();
  }

  let showingStatic = false;
  const toggle = controls.querySelector('[data-action="toggle-view"]');
  function setStatic(on) {
    showingStatic = on;
    canvas.hidden = on;
    fallback.hidden = !on;
    toggle.setAttribute('aria-pressed', on ? 'true' : 'false');
    toggle.textContent = on ? 'Show 3D view' : 'Show static diagram';
    for (const b of controls.querySelectorAll('button')) {
      if (b !== toggle) b.disabled = on;
    }
    if (!on) fit();
  }

  controls.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-action]');
    if (!button) return;
    const action = button.dataset.action;
    if (action === 'next') setStep(current + 1);
    else if (action === 'prev') setStep(current - 1);
    else if (action === 'rotate-left') rotate(-0.3, 0);
    else if (action === 'rotate-right') rotate(0.3, 0);
    else if (action === 'reset') {
      target = { x: BASE_ROT.x, y: BASE_ROT.y };
      applyRotation();
      setStep(-1);
    } else if (action === 'toggle-view') setStatic(!showingStatic);
  });

  // Pointer drag rotates; the buttons above give the same control from the keyboard.
  let drag = null;
  canvas.addEventListener('pointerdown', (event) => {
    drag = { x: event.clientX, y: event.clientY, rx: root.rotation.x, ry: root.rotation.y };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!drag) return;
    root.rotation.y = drag.ry + (event.clientX - drag.x) * 0.008;
    root.rotation.x = Math.max(-0.7, Math.min(0.7, drag.rx + (event.clientY - drag.y) * 0.006));
    target = { x: root.rotation.x, y: root.rotation.y };
    render();
  });
  const endDrag = () => { drag = null; };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);

  canvas.addEventListener('webglcontextlost', (event) => {
    event.preventDefault();
    canvas.hidden = true;
    fallback.hidden = false;
    controls.hidden = true;
    status.textContent = '3D view stopped: the browser lost the WebGL context. The static diagram above shows the same structure.';
  });

  fallback.hidden = true;
  stage.append(canvas);
  controls.hidden = false;
  if (typeof ResizeObserver === 'function') {
    new ResizeObserver(() => { if (!showingStatic) fit(); }).observe(stage);
  } else {
    window.addEventListener('resize', () => { if (!showingStatic) fit(); });
  }
  fit();
  setStep(-1);
  status.textContent = '3D view active (structure only; sizes are illustrative). Use Next step and Previous step to walk '
    + 'through the operations, and the rotate buttons or a drag to turn the view.'
    + (reduceMotion.matches ? ' Reduced motion is on, so the view changes without animation.' : '');
}
