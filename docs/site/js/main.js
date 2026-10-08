// Copyright (c) 2026 Tim Urista. All rights reserved; no license is granted (LICENSE, Part 3).
//
// Entry point. Everything the page states is already in the HTML and the static SVG charts; this
// script only adds the decoder explorer and the optional 3D structure view. The 3D module is
// imported dynamically so that a failure to load Three.js or to start WebGL leaves the static
// diagram in place with an explanation.

import { initDecoder } from './decoder.js';

initDecoder(document.getElementById('decoder-explorer'));

const stage = document.getElementById('diagram-stage');
const status = document.getElementById('diagram-status');
if (stage && status) {
  import('./diagram3d.js')
    .then((mod) => mod.initDiagram({
      stage,
      status,
      controls: document.getElementById('diagram-controls'),
      steps: document.getElementById('diagram-steps'),
    }))
    .catch((err) => {
      status.textContent = '3D view unavailable: the local Three.js module could not be loaded ('
        + (err && err.message ? err.message : 'unknown error')
        + '). The static diagram above shows the same structure.';
    });
}
