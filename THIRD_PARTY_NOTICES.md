# Third-party notices

This file lists third-party material that is redistributed in this repository, with its full
license notice. Material that is only used, and not redistributed (upstream LittleBit code,
Qwen2.5-0.5B, WikiText-2, runtime Python packages), is described in LICENSE and LICENSE_NOTES.md.

Not legal advice.

## Three.js 0.186.1 (MIT License)

* Files: `docs/site/vendor/three.module.js`, `docs/site/vendor/three.core.js` (the second is
  imported by the first), copied unmodified from the official `three` package, version 0.186.1
  (npm package `three`, `build/` directory; source https://github.com/mrdoob/three.js).
* License file: `docs/site/vendor/THREE-LICENSE.txt`, the package's LICENSE file extracted
  verbatim from the same package. Provenance (npm tarball URL, integrity, and the byte count and
  SHA-256 of each vendored file) is recorded in `docs/site/vendor/VENDOR.json`. The copyright line
  below must match `THREE-LICENSE.txt` exactly; `tests/test_companion.py` checks this.
* Used by: the optional 3D structure view of the companion page (`docs/site/js/diagram3d.js`).
  The page works without it; the static SVG diagram is always present.
* These files are not covered by the scoped terms in LICENSE Parts 1 to 3. They stay under the
  MIT License of their authors and are not relicensed.

```
The MIT License

Copyright © 2010-2026 three.js authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Upstream license text included for adapted files

* `LICENSES/CC-BY-NC-4.0.txt`: the Creative Commons Attribution-NonCommercial 4.0 International
  Public License, copied unchanged from SamsungLabs/LittleBit at commit
  `933857ed1443b53fc43a875c2cf64249e3c56f0c`. It covers the five adapted files listed in
  LICENSE, Part 2. No upstream source file is redistributed.
