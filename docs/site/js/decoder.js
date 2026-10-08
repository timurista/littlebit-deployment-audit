// Copyright (c) 2026 Tim Urista. All rights reserved; no license is granted (LICENSE, Part 3).
//
// Decoder explorer. Loads the recorded unpack cases (site/data/unpack_cases.json, generated from
// results/unpack_cases_upstream_import.csv) and decodes one case's recorded packed words in the
// browser, twice:
//   shipped expression  bit_k = (word << k) & 1    (upstream binary_unpacker, line 88)
//   fixed expression    bit_k = (word >>> k) & 1   (the proposed right shift, read as unsigned)
// A set bit decodes to -1. The browser's error count is compared with the count recorded in the
// CSV. No case is invented: only recorded words are decoded.

const WORD_BITS = 32;
const DEFAULT_CASE = 'byte_w32_b129';
const FAMILY_LABELS = {
  byte_tiled: 'Byte tiled across the row',
  one_hot: 'A single -1 at one column',
  sign_example: 'Fixed sign patterns',
};

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [key, value] of Object.entries(attrs)) {
      if (value === undefined || value === null || value === false) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else node.setAttribute(key, value === true ? '' : String(value));
    }
  }
  for (const child of children || []) {
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// Recorded words are hex strings; >>> 0 keeps them as unsigned 32-bit numbers.
export function parseWord(hex) {
  return parseInt(hex, 16) >>> 0;
}

// As shipped: JavaScript, like torch on int32 words, converts the word to a signed 32-bit integer
// before shifting. Bit 0 of (word << k) is always 0 for k >= 1.
export function shippedBit(word, k) {
  return (word << k) & 1;
}

// Fixed: >>> shifts in zeros, so negative int32 words (word >= 0x80000000) read correctly.
export function fixedBit(word, k) {
  return (word >>> k) & 1;
}

export function decodeCase(words, width) {
  const recorded = [];
  const shipped = [];
  for (let j = 0; j < width; j += 1) {
    const word = words[Math.floor(j / WORD_BITS)];
    const k = j % WORD_BITS;
    recorded.push(fixedBit(word, k));
    shipped.push(shippedBit(word, k));
  }
  const errors = [];
  for (let j = 0; j < width; j += 1) {
    if (recorded[j] !== shipped[j]) errors.push(j);
  }
  return { recorded, shipped, errors };
}

function hex32(value) {
  return '0x' + (value >>> 0).toString(16).padStart(8, '0');
}

function caseLabel(c) {
  const errors = c.signErrors === 1 ? '1 sign error' : c.signErrors + ' sign errors';
  if (c.family === 'byte_tiled') {
    const b = Number(c.param);
    return 'byte ' + b + ' (0x' + b.toString(16).padStart(2, '0') + '): ' + errors;
  }
  if (c.family === 'one_hot') return '-1 at column ' + c.param + ': ' + errors;
  return c.param.replace(/_/g, ' ') + ': ' + errors;
}

function listColumns(cols) {
  if (cols.length === 0) return 'none';
  if (cols.length <= 12) return cols.join(', ');
  return cols.slice(0, 12).join(', ') + ' and ' + (cols.length - 12) + ' more';
}

function bitGrid(word, wordIndex, width, decoded, k) {
  const grid = el('div', { class: 'bitgrid', 'aria-hidden': 'true' });
  grid.append(el('span', { class: 'rowlabel', text: 'in-word bit' }));
  for (let b = 0; b < WORD_BITS; b += 1) {
    grid.append(el('span', { class: 'colhead' + (b === k ? ' k' : ''), text: String(b) }));
  }
  const rows = [
    ['recorded sign', 'recorded'],
    ['shipped  << ', 'shipped'],
    ['fixed  >>> ', 'fixed'],
  ];
  for (const [label, kind] of rows) {
    grid.append(el('span', { class: 'rowlabel', text: label }));
    for (let b = 0; b < WORD_BITS; b += 1) {
      const j = wordIndex * WORD_BITS + b;
      if (j >= width) {
        grid.append(el('span', { class: 'cell pad', text: '.' }));
        continue;
      }
      const bit = kind === 'shipped' ? decoded.shipped[j] : decoded.recorded[j];
      const isError = kind === 'shipped' && decoded.shipped[j] !== decoded.recorded[j];
      const cls = 'cell ' + (bit ? 'minus' : 'plus') + (isError ? ' error' : '') + (b === k ? ' k' : '');
      grid.append(el('span', { class: cls, text: bit ? '-' : '+' }));
    }
  }
  return grid;
}

function shiftTable(word, k) {
  const signed = word | 0;
  const left = word << k;
  const arith = word >> k;
  const logical = word >>> k;
  const rows = [
    ['w', String(signed), hex32(word), '', 'the recorded word as a signed 32-bit integer'],
    ['w << ' + k, String(left), hex32(left), String(left & 1), 'shipped: bit 0 of the left-shifted word'],
    ['w >> ' + k, String(arith), hex32(arith), String(arith & 1), 'arithmetic right shift: sign-extends, low bit still right'],
    ['w >>> ' + k, String(logical), hex32(logical), String(logical & 1), 'fixed: logical right shift, unsigned'],
  ];
  const table = el('table', { class: 'shift-table' });
  table.append(el('caption', { text: 'Shift arithmetic for word 0, bit k = ' + k + ' (value of the true bit: ' + fixedBit(word, k) + ')' }));
  const head = el('tr');
  for (const h of ['expression', 'value (int32)', 'hex', '& 1', 'meaning']) head.append(el('th', { scope: 'col', text: h }));
  table.append(el('thead', null, [head]));
  const body = el('tbody');
  for (const r of rows) {
    const tr = el('tr');
    tr.append(el('th', { scope: 'row', text: r[0] }));
    for (const cell of r.slice(1)) tr.append(el('td', { text: cell }));
    body.append(tr);
  }
  table.append(body);
  return el('div', { class: 'table-wrap', role: 'region', tabindex: '0', 'aria-label': 'Shift arithmetic for the selected bit' }, [table]);
}

export async function initDecoder(root) {
  if (!root) return;
  const status = root.querySelector('#decoder-status');
  const form = root.querySelector('#decoder-form');
  const out = root.querySelector('#decoder-output');
  const widthSel = root.querySelector('#dec-width');
  const familySel = root.querySelector('#dec-family');
  const caseSel = root.querySelector('#dec-case');
  const bitInput = root.querySelector('#dec-bit');
  const bitOut = root.querySelector('#dec-bit-out');

  let doc;
  try {
    const response = await fetch(root.dataset.src, { credentials: 'omit' });
    if (!response.ok) throw new Error('HTTP ' + response.status);
    doc = await response.json();
  } catch (err) {
    status.textContent = 'The recorded cases could not be loaded (' + err.message + '). The histogram and table below carry the same recorded counts.';
    status.classList.add('is-error');
    return;
  }

  const idx = {};
  doc.fields.forEach((name, i) => { idx[name] = i; });
  const cases = doc.cases.map((row) => ({
    id: row[idx.case_id],
    family: row[idx.family],
    width: row[idx.width],
    param: row[idx.param],
    nMinus: row[idx.n_minus],
    wordsHex: row[idx.packed_words_hex],
    words: row[idx.packed_words_hex].map(parseWord),
    exact: row[idx.upstream_exact],
    signErrors: row[idx.sign_errors],
    firstError: row[idx.first_error_col],
  }));
  const byId = new Map(cases.map((c) => [c.id, c]));
  const widths = [...new Set(cases.map((c) => c.width))].sort((a, b) => a - b);

  function fillSelect(select, options, selected) {
    select.replaceChildren(...options.map(([value, label]) => {
      const o = el('option', { value, text: label });
      if (String(value) === String(selected)) o.selected = true;
      return o;
    }));
  }

  function familiesFor(width) {
    return Object.keys(FAMILY_LABELS).filter((f) => cases.some((c) => c.width === width && c.family === f));
  }

  function casesFor(width, family) {
    return cases.filter((c) => c.width === width && c.family === family);
  }

  function selectCase(c) {
    fillSelect(widthSel, widths.map((w) => [w, String(w)]), c.width);
    fillSelect(familySel, familiesFor(c.width).map((f) => [f, FAMILY_LABELS[f]]), c.family);
    fillSelect(caseSel, casesFor(c.width, c.family).map((x) => [x.id, caseLabel(x)]), c.id);
    render(c);
  }

  function render(c) {
    const k = Number(bitInput.value);
    bitOut.value = String(k);
    bitOut.textContent = String(k);
    const decoded = decodeCase(c.words, c.width);
    const recordedMinus = decoded.recorded.reduce((a, b) => a + b, 0);
    const matches = decoded.errors.length === c.signErrors && (decoded.errors.length === 0) === c.exact
      && recordedMinus === c.nMinus && (decoded.errors.length ? decoded.errors[0] : -1) === c.firstError;

    const verdict = el('dl', { class: 'verdict' }, [
      el('div', null, [el('dt', { text: 'Recorded in the CSV (' + c.id + ')' }),
        el('dd', { text: c.signErrors + ' sign errors, ' + (c.exact ? 'exact' : 'not exact') })]),
      el('div', { class: decoded.errors.length ? 'bad' : '' }, [el('dt', { text: 'Recomputed here, shipped << expression' }),
        el('dd', { text: decoded.errors.length + ' sign errors' })]),
      el('div', null, [el('dt', { text: 'Recomputed here, fixed >>> expression' }), el('dd', { text: '0 sign errors' })]),
      el('div', { class: matches ? '' : 'bad' }, [el('dt', { text: 'Browser result versus recorded row' }),
        el('dd', { text: matches ? 'matches' : 'does not match' })]),
    ]);

    const words = c.words.map((word, i) => {
      const minusCols = [];
      const errorCols = [];
      for (let b = 0; b < WORD_BITS; b += 1) {
        const j = i * WORD_BITS + b;
        if (j >= c.width) break;
        if (decoded.recorded[j]) minusCols.push(j);
        if (decoded.recorded[j] !== decoded.shipped[j]) errorCols.push(j);
      }
      const summary = 'Word ' + i + ' = ' + hex32(word) + ' (int32 ' + (word | 0) + '). Recorded -1 at columns '
        + listColumns(minusCols) + '. The shipped expression misreads ' + errorCols.length
        + (errorCols.length === 1 ? ' sign' : ' signs') + (errorCols.length ? ', at columns ' + listColumns(errorCols) : '') + '.';
      return el('div', { class: 'word' }, [
        el('h4', { text: 'Word ' + i + ': ' + hex32(word) }),
        el('p', { class: 'word-summary', text: summary }),
        bitGrid(word, i, c.width, decoded, k),
      ]);
    });

    out.replaceChildren(verdict, ...words, shiftTable(c.words[0], k));
    status.classList.toggle('is-error', !matches);
    status.textContent = 'Case ' + c.id + ', width ' + c.width + ': recorded ' + c.signErrors + ' sign errors; the browser recomputed '
      + decoded.errors.length + (matches ? ', matching the recorded row.' : '. This does not match the recorded row.');
  }

  widthSel.addEventListener('change', () => {
    const width = Number(widthSel.value);
    const fams = familiesFor(width);
    const family = fams.includes(familySel.value) ? familySel.value : fams[0];
    selectCase(casesFor(width, family)[0]);
  });
  familySel.addEventListener('change', () => {
    selectCase(casesFor(Number(widthSel.value), familySel.value)[0]);
  });
  caseSel.addEventListener('change', () => selectCase(byId.get(caseSel.value)));
  bitInput.addEventListener('input', () => render(byId.get(caseSel.value)));
  form.addEventListener('submit', (event) => event.preventDefault());

  form.hidden = false;
  selectCase(byId.get(DEFAULT_CASE) || cases[0]);
}
