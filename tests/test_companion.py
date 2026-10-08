# Copyright (c) 2026 Tim Urista. All rights reserved.
# No license is granted for this file (LICENSE, Part 3). It is not one of the seven MIT files.
"""Checks for the GitHub Pages companion in docs/ and its build script src/build_companion.py.

Run from the project root:  python -m unittest tests.test_companion -v

Standard library only. The checks compare the generated page, data files and SVG charts with the
recorded result files directly (not only with the build script's own output), and check that every
local link works under the GitHub project base path /littlebit-deployment-audit/.
"""

import csv
import hashlib
import json
import math
import os
import re
import sys
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import build_companion as bc  # noqa: E402

DOCS = os.path.join(PROJECT_ROOT, "docs")
SITE = os.path.join(DOCS, "site")
ASSETS = os.path.join(SITE, "assets")
VENDOR = os.path.join(SITE, "vendor")
PAGES_BASE = "https://timurista.github.io/littlebit-deployment-audit/"
ALLOWED_LINK_HOSTS = {"github.com", "timurista.github.io"}
BUILD_HINT = "run: python src/build_companion.py"

# Recorded at v0.1.0 (README section 1 and results/summary.json). Written out here so that the
# checks do not depend only on the build script reading the same files.
RECORDED_CASES = 2010
RECORDED_EXACT = 296
RECORDED_SIGN_ERRORS = 21881
RECORDED_FAMILIES = {"byte_tiled": 1792, "one_hot": 176, "sign_example": 42}
RECORDED_BYTE_ROWS_EXACT = {1: 256, 7: 4, 8: 2, 31: 1, 32: 1, 33: 1, 64: 1}
RECORDED_WIDTH_ERRORS = {1: 0, 7: 786, 8: 917, 31: 3930, 32: 4062, 33: 4062, 64: 8124}
RECORDED_DISK = {"disk.original": 988097824, "disk.student": 299683600}
RECORDED_RESIDENT = {"resident.original": 1976131200, "resident.student": 1234703616}
RECORDED_KV_BYTES = 3145728
RECORDED_TEXT = ("28.8816", "82,426.3372", "9,735.6350", "13,558,558.4902", "0.52955", "0.6094", "0.4967",
                 "988,097,824", "299,683,600", "1,976,131,200", "1,234,703,616", "3,145,728", "6.53",
                 "2,048", "1,016", "168", "seed 42", "ITQ false", "2,010", "21,881")


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_json(rel):
    with open(os.path.join(PROJECT_ROOT, rel), "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(rel):
    with open(os.path.join(PROJECT_ROOT, rel), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def csv_rows():
    with open(os.path.join(PROJECT_ROOT, bc.UNPACK_CSV), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def svg_marks(name):
    path = os.path.join(ASSETS, name)
    if not os.path.exists(path):
        raise AssertionError("%s missing; %s" % (path, BUILD_HINT))
    root = ET.parse(path).getroot()
    return [el.attrib for el in root.iter() if "data-value" in el.attrib or "data-count" in el.attrib]


def require(path):
    if not os.path.exists(path):
        raise AssertionError("%s missing; %s" % (os.path.relpath(path, PROJECT_ROOT), BUILD_HINT))
    return path


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.refs = []
        self.ids = set()
        self.headings = []
        self.imgs = []
        self.labels_for = set()
        self.controls = []
        self.metas = []
        self.buttons = []
        self.scripts = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id"):
            self.ids.add(a["id"])
        for attr in ("href", "src", "data-src", "srcset", "poster", "action"):
            if a.get(attr) is not None:
                self.refs.append((tag, attr, a[attr]))
        if re.fullmatch(r"h[1-6]", tag):
            self.headings.append(int(tag[1]))
        if tag == "img":
            self.imgs.append(a)
        if tag == "label" and a.get("for"):
            self.labels_for.add(a["for"])
        if tag in ("select", "input", "textarea"):
            self.controls.append(a)
        if tag == "meta":
            self.metas.append(a)
        if tag == "button":
            self.buttons.append(a)
        if tag == "script":
            self.scripts.append(a)
        if tag == "link":
            self.links.append(a)


def parse_page():
    parser = PageParser()
    parser.feed(read_text(require(os.path.join(DOCS, "index.html"))))
    return parser


class TestBuildIsCurrent(unittest.TestCase):
    def test_outputs_match_a_fresh_build(self):
        outputs = bc.build_outputs()
        self.assertGreaterEqual(len(outputs), 12)
        for rel, data in sorted(outputs.items()):
            path = require(os.path.join(PROJECT_ROOT, rel))
            with open(path, "rb") as f:
                self.assertEqual(f.read(), data, "%s is out of date; %s" % (rel, BUILD_HINT))

    def test_build_is_deterministic(self):
        self.assertEqual(bc.build_outputs(), bc.build_outputs())

    def test_template_has_no_unresolved_placeholders(self):
        page = read_text(require(os.path.join(DOCS, "index.html")))
        self.assertIsNone(re.search(r"\{\{[^}]*\}\}", page))

    def test_outputs_stay_inside_docs(self):
        for rel in bc.build_outputs():
            self.assertTrue(rel.startswith("docs/"), rel)


class TestUnpackCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = csv_rows()
        cls.data = load_json(bc.OUT_DATA)
        cls.cases_doc = load_json(bc.OUT_CASES)

    def test_recorded_totals(self):
        self.assertEqual(len(self.rows), RECORDED_CASES)
        self.assertEqual(sum(r["upstream_exact"] == "True" for r in self.rows), RECORDED_EXACT)
        self.assertEqual(sum(int(r["sign_errors"]) for r in self.rows), RECORDED_SIGN_ERRORS)
        fams = {}
        for r in self.rows:
            fams[r["family"]] = fams.get(r["family"], 0) + 1
        self.assertEqual(fams, RECORDED_FAMILIES)
        u = self.data["unpack"]
        self.assertEqual((u["cases"], u["exact_cases"], u["sign_errors"]), (RECORDED_CASES, RECORDED_EXACT, RECORDED_SIGN_ERRORS))
        self.assertEqual(u["plus_to_minus_flips"], 0)
        self.assertEqual(u["minus_to_plus_flips"], RECORDED_SIGN_ERRORS)

    def test_width_table_matches_recorded_values(self):
        for row in self.data["unpack"]["widths"]:
            self.assertEqual(row["byte_rows_exact"], RECORDED_BYTE_ROWS_EXACT[row["width"]], row)
            self.assertEqual(row["byte_rows"], 256, row)
            self.assertEqual(row["sign_errors"], RECORDED_WIDTH_ERRORS[row["width"]], row)
        self.assertEqual(sum(r["cases"] for r in self.data["unpack"]["widths"]), RECORDED_CASES)

    def test_histogram_bins_cover_every_case_exactly_once(self):
        hist = self.data["unpack"]["histogram"]
        self.assertEqual(hist[0]["lo"], 0)
        for a, b in zip(hist, hist[1:]):
            self.assertEqual(b["lo"], a["hi"] + 1, "bins must be contiguous and non-overlapping")
        errors = [int(r["sign_errors"]) for r in self.rows]
        self.assertLessEqual(max(errors), hist[-1]["hi"])
        for b in hist:
            self.assertEqual(b["count"], sum(1 for e in errors if b["lo"] <= e <= b["hi"]), b)
        self.assertEqual(sum(b["count"] for b in hist), RECORDED_CASES)

    def test_histogram_svg_counts(self):
        marks = svg_marks("unpack_error_histogram.svg")
        counts = [(int(m["data-bin-lo"]), int(m["data-bin-hi"]), int(m["data-count"])) for m in marks]
        self.assertEqual(counts, [(b["lo"], b["hi"], b["count"]) for b in self.data["unpack"]["histogram"]])
        self.assertEqual(sum(c for _, _, c in counts), RECORDED_CASES)

    def test_case_file_is_the_csv(self):
        doc = self.cases_doc
        f = {name: i for i, name in enumerate(doc["fields"])}
        self.assertEqual(len(doc["cases"]), RECORDED_CASES)
        self.assertEqual(doc["source"]["sha256"], sha256_file(bc.UNPACK_CSV))
        for row, rec in zip(self.rows, doc["cases"]):
            self.assertEqual(rec[f["case_id"]], row["case_id"])
            self.assertEqual(" ".join(rec[f["packed_words_hex"]]), row["packed_words_hex"])
            self.assertEqual(rec[f["sign_errors"]], int(row["sign_errors"]))
            self.assertEqual(rec[f["upstream_exact"]], row["upstream_exact"] == "True")

    def test_shipped_expression_reproduces_every_recorded_row(self):
        # Independent of the build: decode each recorded word with the shipped left shift.
        for r in self.rows:
            words = [int(w, 16) for w in r["packed_words_hex"].split()]
            width = int(r["width"])
            truth = [(words[j // 32] >> (j % 32)) & 1 for j in range(width)]
            shipped = [(words[j // 32] << (j % 32)) & 1 for j in range(width)]
            errors = sum(1 for t, s in zip(truth, shipped) if t != s)
            self.assertEqual(sum(truth), int(r["n_minus"]), r["case_id"])
            self.assertEqual(errors, int(r["sign_errors"]), r["case_id"])
            self.assertEqual(errors == 0, r["upstream_exact"] == "True", r["case_id"])

    def test_explorer_default_case_is_recorded(self):
        js = read_text(os.path.join(SITE, "js", "decoder.js"))
        default = re.search(r"DEFAULT_CASE = '([^']+)'", js).group(1)
        self.assertIn(default, {r["case_id"] for r in self.rows})
        self.assertIn("(word << k) & 1", js)
        self.assertIn("(word >>> k) & 1", js)


class TestQualityCharts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_json(bc.SUMMARY)
        cls.cond = cls.summary["heldout_test"]["conditions"]

    def test_eight_paired_deltas_exact(self):
        expected = self.cond["qat_trained"]["paired_diff_vs_original"]
        self.assertEqual(len(expected), 8)
        self.assertEqual(load_json(bc.OUT_DATA)["quality"]["qat_minus_original"], expected)
        marks = svg_marks("nll_delta_strip.svg")
        self.assertEqual(len(marks), 8)
        got = sorted((int(m["data-window"]), float(m["data-value"])) for m in marks)
        self.assertEqual(got, list(enumerate(expected)))

    def test_log10_perplexity_bars(self):
        marks = {m["data-key"]: m for m in svg_marks("quality_log10_ppl.svg")}
        self.assertEqual(len(marks), 4)
        for key in ("original", "initialized_no_qat", "qat_trained", "qat_trained_upstream_decoder"):
            m = marks["ppl." + key]
            self.assertEqual(float(m["data-value"]), self.cond[key]["ppl"], key)
            self.assertAlmostEqual(float(m["data-log10"]), math.log10(self.cond[key]["ppl"]), places=5)
        self.assertIn("url(#defect-hatch)", marks["ppl.qat_trained_upstream_decoder"]["fill"])
        svg = read_text(os.path.join(ASSETS, "quality_log10_ppl.svg"))
        self.assertIn("decoder defect condition, not a quality result", svg)

    def test_paired_window_markers(self):
        marks = svg_marks("paired_window_nll.svg")
        self.assertEqual(len(marks), 32)
        for m in marks:
            key = m["data-key"].split(".", 1)[1]
            self.assertEqual(float(m["data-value"]), self.cond[key]["per_window_mean_nll"][int(m["data-window"])])


class TestStorageCharts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_json(bc.SUMMARY)

    def _totals(self, name):
        totals = {}
        for m in svg_marks(name):
            row = m["data-key"].rsplit(".", 1)[0]
            totals[row] = totals.get(row, 0) + int(m["data-value"])
        return totals

    def test_disk_and_resident_panels_are_separate_and_sum_to_recorded_totals(self):
        self.assertEqual(self._totals("storage_disk.svg"), RECORDED_DISK)
        self.assertEqual(self._totals("storage_resident.svg"), RECORDED_RESIDENT)
        st = self.summary["storage"]
        self.assertEqual(RECORDED_DISK["disk.original"], st["disk"]["original"]["bytes"])
        self.assertEqual(RECORDED_RESIDENT["resident.student"], st["resident_fp32_unique"]["student_bytes"])
        for name in ("storage_disk.svg", "storage_resident.svg"):
            self.assertNotIn("rss", read_text(os.path.join(ASSETS, name)).lower())

    def test_kv_cache_unchanged(self):
        kv = self.summary["storage"]["kv_cache"]
        self.assertEqual((kv["original_bytes"], kv["student_bytes"]), (RECORDED_KV_BYTES, RECORDED_KV_BYTES))
        self.assertIn("3,145,728", read_text(os.path.join(DOCS, "index.html")))

    def test_bpw_bars(self):
        b = self.summary["storage"]["bpw_converted_block_linears"]
        marks = {m["data-key"]: float(m["data-value"]) for m in svg_marks("bpw.svg")}
        self.assertEqual(marks, {"bpw.packed_signs_only": b["packed_signs_only"],
                                 "bpw.advertised_upstream_formula": b["advertised_upstream_formula"],
                                 "bpw.actual_checkpoint_file_fp32_scales": b["actual_checkpoint_file_fp32_scales"]})

    def test_rss_is_its_own_panel(self):
        marks = {m["data-key"]: int(m["data-value"]) for m in svg_marks("os_peak_rss.svg")}
        for stage in ("baseline", "feasibility", "train", "evaluate"):
            self.assertEqual(marks["rss." + stage], self.summary["resources"][stage]["peak_rss_os_bytes"])
        self.assertEqual(marks["rss_sampled.train"], self.summary["resources"]["train"]["guard_peak_rss_seen_at_checks_bytes"])
        svg = read_text(os.path.join(ASSETS, "os_peak_rss.svg"))
        self.assertIn("Not comparable with the storage panels", svg)


class TestProvenance(unittest.TestCase):
    def test_source_hashes(self):
        data = load_json(bc.OUT_DATA)
        paths = [s["path"] for s in data["sources"]]
        self.assertEqual(paths, [bc.SUMMARY, bc.UNPACK_CSV, bc.PATCHED, bc.TRAIN])
        for s in data["sources"]:
            self.assertEqual(s["sha256"], sha256_file(s["path"]), s["path"])
            self.assertEqual(s["bytes"], os.path.getsize(os.path.join(PROJECT_ROOT, s["path"])), s["path"])
        self.assertEqual(data["measured_data"], {"release": "v0.1.0", "commit": "97f075cd1f995e66b693b3c1657fcc2f1a43ed0c",
                                                 "note": data["measured_data"]["note"]})
        self.assertEqual(data["upstream"]["commit"], "933857ed1443b53fc43a875c2cf64249e3c56f0c")
        self.assertEqual(data["companion_version"], "0.2.0")
        self.assertTrue(data["companion_release_status"].startswith("v0.2.0:"))
        self.assertNotIn("prospective", data["companion_release_status"])

    def test_recorded_pilot_inputs_unchanged(self):
        # The summary's own input hashes must still match the recorded stage files.
        for rec in load_json(bc.SUMMARY)["inputs"].values():
            self.assertEqual(sha256_file(rec["path"]), rec["sha256"], rec["path"])

    def test_page_states_recorded_values_and_versions(self):
        page = read_text(os.path.join(DOCS, "index.html"))
        for s in RECORDED_TEXT:
            self.assertIn(s, page)
        for s in (bc.DATA_COMMIT, bc.UPSTREAM_COMMIT, bc.UPSTREAM_FORMULA, "Companion v0.2.0", "illustrative", "not measured",
                  "decoder defect condition", "not independent draws"):
            self.assertIn(s, page)


class TestLinksAndPagesBasePath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = parse_page()

    def _local_target(self, ref):
        path = ref.split("#", 1)[0].split("?", 1)[0]
        target = os.path.normpath(os.path.join(DOCS, path))
        if os.path.isdir(target):
            target = os.path.join(target, "index.html")
        return target

    def test_every_reference_is_relative_or_allowlisted(self):
        self.assertGreater(len(self.page.refs), 20)
        for tag, attr, ref in self.page.refs:
            parts = urlsplit(ref)
            if parts.scheme or ref.startswith("//"):
                self.assertEqual(parts.scheme, "https", ref)
                self.assertIn(parts.hostname, ALLOWED_LINK_HOSTS, ref)
                self.assertEqual((tag, attr), ("a", "href"), "only plain links may leave the site: %s" % ref)
                continue
            if ref.startswith("#"):
                self.assertIn(ref[1:], self.page.ids, ref)
                continue
            self.assertFalse(ref.startswith("/"), "root-relative links break under the project base path: %s" % ref)
            self.assertTrue(urljoin(PAGES_BASE, ref).startswith(PAGES_BASE), ref)
            target = self._local_target(ref)
            self.assertTrue(target.startswith(DOCS + os.sep), ref)
            self.assertTrue(os.path.isfile(target), "%s -> %s" % (ref, os.path.relpath(target, PROJECT_ROOT)))

    def test_no_external_scripts_styles_fonts_or_trackers(self):
        for s in self.page.scripts:
            self.assertFalse(urlsplit(s.get("src", "")).scheme, s)
        for link in self.page.links:
            self.assertFalse(urlsplit(link.get("href", "")).scheme, link)
        page = read_text(os.path.join(DOCS, "index.html"))
        self.assertNotIn("http://", page)
        for word in ("googleapis", "gstatic", "google-analytics", "googletagmanager", "gtag(", "unpkg", "jsdelivr",
                     "cdnjs", "plausible.io"):
            self.assertNotIn(word, page.lower(), word)
        csp = [m for m in self.page.metas if m.get("http-equiv") == "Content-Security-Policy"]
        self.assertEqual(len(csp), 1)
        self.assertIn("default-src 'self'", csp[0]["content"])
        self.assertIn("script-src 'self'", csp[0]["content"])
        css = read_text(os.path.join(SITE, "css", "site.css"))
        self.assertNotIn("@import", css)
        self.assertNotIn("@font-face", css)
        self.assertNotRegex(css, r"url\(\s*['\"]?https?:")

    def test_js_imports_are_local_modules(self):
        spec = re.compile(r"""(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)['"]([^'"]+)['"]""")
        for name in ("main.js", "decoder.js", "diagram3d.js"):
            path = os.path.join(SITE, "js", name)
            text = read_text(path)
            self.assertNotRegex(text, r"https?://", name)
            for ref in spec.findall(text):
                self.assertTrue(ref.startswith(("./", "../")), "%s imports %s" % (name, ref))
                target = os.path.normpath(os.path.join(os.path.dirname(path), ref))
                self.assertTrue(target.startswith(SITE + os.sep), ref)
                self.assertTrue(os.path.isfile(target), "%s imports missing %s" % (name, ref))

    def test_readme_points_at_pages_and_existing_svgs(self):
        readme = read_text(os.path.join(PROJECT_ROOT, "README.md"))
        self.assertIn(PAGES_BASE, readme)
        refs = re.findall(r"\]\((docs/site/assets/[^)\s]+\.svg)\)", readme)
        self.assertGreaterEqual(len(refs), 4)
        for ref in refs:
            self.assertTrue(os.path.isfile(os.path.join(PROJECT_ROOT, ref)), ref)


class TestAccessibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = parse_page()

    def test_skip_link_and_main_target(self):
        self.assertIn(("a", "href", "#main"), self.page.refs[:3])
        self.assertIn("main", self.page.ids)

    def test_headings_do_not_skip_levels(self):
        levels = self.page.headings
        self.assertEqual(levels.count(1), 1)
        self.assertEqual(levels[0], 1)
        for a, b in zip(levels, levels[1:]):
            self.assertLessEqual(b, a + 1, levels)

    def test_images_have_alt_text_and_size(self):
        self.assertGreaterEqual(len(self.page.imgs), 9)
        for img in self.page.imgs:
            self.assertGreater(len(img.get("alt", "").strip()), 20, img)
            self.assertTrue(img.get("width") and img.get("height"), img)

    def test_controls_are_labelled_and_buttons_typed(self):
        for c in self.page.controls:
            self.assertIn(c.get("id"), self.page.labels_for, c)
        self.assertGreaterEqual(len(self.page.buttons), 6)
        for b in self.page.buttons:
            self.assertEqual(b.get("type"), "button", b)

    def test_full_hashes_can_wrap_on_narrow_screens(self):
        page = read_text(os.path.join(DOCS, "index.html"))
        css = read_text(os.path.join(SITE, "css", "site.css"))
        self.assertIn("code.hash, code.commit { overflow-wrap: anywhere; word-break: break-all; }", css)
        # Every full commit id or SHA-256 shown as text sits in a code element with a wrapping class.
        shown = re.findall(r"<code([^>]*)>([0-9a-f]{40,64})</code>", page)
        self.assertGreaterEqual(len(shown), 6)
        for attrs, value in shown:
            self.assertRegex(attrs, r'class="(hash|commit)"', value)
        text_only = re.sub(r'<[^>]+>', ' ', page)
        for token in re.findall(r"\b[0-9a-f]{40,64}\b", text_only):
            self.assertIn(token, {v for _, v in shown}, token)

    def test_inline_code_wraps_but_preformatted_code_scrolls(self):
        css = read_text(os.path.join(SITE, "css", "site.css"))
        self.assertIn(":not(pre) > code { overflow-wrap: anywhere; }", css)
        self.assertRegex(css, r"pre\.code \{[^}]*overflow-x: auto;")
        page = read_text(os.path.join(DOCS, "index.html"))
        self.assertIn("<code>results/unpack_cases_upstream_import.csv</code>", page)

    def test_manuscript_pdf_link(self):
        url = "https://github.com/timurista/littlebit-deployment-audit/releases/download/v0.2.0/littlebit-v0.2.0-manuscript.pdf"
        self.assertEqual(bc.MANUSCRIPT_PDF_URL, url)
        self.assertGreaterEqual(read_text(os.path.join(DOCS, "index.html")).count('href="%s"' % url), 2)
        self.assertIn(url, read_text(os.path.join(PROJECT_ROOT, "README.md")))

    def test_resident_footer_is_split_and_canvas_grows(self):
        root = ET.parse(os.path.join(ASSETS, "storage_resident.svg")).getroot()
        self.assertEqual((root.get("width"), root.get("height"), root.get("viewBox")), ("720", "315", "0 0 720 315"))
        lines = [t.text for t in root.iter("{http://www.w3.org/2000/svg}text")]
        self.assertIn("Source: results/summary.json, storage.resident_fp32_unique.", lines)
        self.assertIn("KV cache (128 tokens, FP32): 3,145,728 bytes in both, unchanged, not shown.", lines)
        self.assertIn('src="site/assets/storage_resident.svg" width="720" height="315"',
                      read_text(os.path.join(DOCS, "index.html")))

    def test_reduced_motion_is_respected(self):
        css = read_text(os.path.join(SITE, "css", "site.css"))
        self.assertIn("prefers-reduced-motion: reduce", css)
        self.assertIn("[hidden] { display: none !important; }", css)
        js = read_text(os.path.join(SITE, "js", "diagram3d.js"))
        self.assertIn("prefers-reduced-motion: reduce", js)


class TestVendoredThree(unittest.TestCase):
    """The executor vendors the official three 0.186.1 build; these checks fail until it does."""

    def test_files_present_with_license(self):
        for name in ("three.module.js", "three.core.js", "THREE-LICENSE.txt", "VENDOR.json"):
            require(os.path.join(VENDOR, name))
        core = read_text(os.path.join(VENDOR, "three.core.js"))
        self.assertRegex(core, r"REVISION\s*=\s*['\"]186['\"]")
        module = read_text(os.path.join(VENDOR, "three.module.js"))
        self.assertRegex(module, r"['\"]\./three\.core\.js['\"]")
        lic = read_text(os.path.join(VENDOR, "THREE-LICENSE.txt"))
        self.assertIn("MIT License", lic)
        self.assertIn("three.js authors", lic.lower())

    def test_third_party_notice(self):
        notices = read_text(os.path.join(PROJECT_ROOT, "THIRD_PARTY_NOTICES.md"))
        self.assertIn("0.186.1", notices)
        self.assertIn("docs/site/vendor/", notices)
        self.assertIn("MIT License", notices)
        # The copyright line quoted in the notice must be the one in the vendored LICENSE file.
        line = re.compile(r"^Copyright .*three\.js authors\s*$", re.IGNORECASE | re.MULTILINE)
        vendored = line.findall(read_text(require(os.path.join(VENDOR, "THREE-LICENSE.txt"))))
        self.assertEqual(len(vendored), 1)
        self.assertIn(vendored[0].strip(), [m.strip() for m in line.findall(notices)])


class TestAuthoredStyle(unittest.TestCase):
    AUTHORED = ("src/build_companion.py", "src/companion/index.template.html", "tests/test_companion.py",
                "docs/index.html", "docs/site/css/site.css", "docs/site/js/main.js", "docs/site/js/decoder.js",
                "docs/site/js/diagram3d.js", "THIRD_PARTY_NOTICES.md", "README.md", "RELEASE_NOTES.md",
                "LICENSE", "docs/research-draft.tex")

    def test_no_em_dashes(self):
        paths = list(self.AUTHORED)
        for top in ("docs/site/assets", "docs/site/data"):
            d = os.path.join(PROJECT_ROOT, top)
            if os.path.isdir(d):
                paths.extend(os.path.join(top, n) for n in sorted(os.listdir(d)))
        for rel in paths:
            path = os.path.join(PROJECT_ROOT, rel)
            if os.path.isfile(path):
                self.assertNotIn(chr(0x2014), read_text(path), rel)

    def test_latex_has_no_em_dash_ligature(self):
        tex = read_text(os.path.join(DOCS, "research-draft.tex"))
        self.assertNotIn("---", tex)

    def test_new_files_carry_rights_notice(self):
        for rel in ("src/build_companion.py", "src/companion/index.template.html", "tests/test_companion.py",
                    "docs/site/css/site.css", "docs/site/js/main.js", "docs/site/js/decoder.js", "docs/site/js/diagram3d.js"):
            head = read_text(os.path.join(PROJECT_ROOT, rel))[:400]
            self.assertIn("All rights reserved", head, rel)
            self.assertNotIn("SPDX-License-Identifier: MIT", head, rel)


if __name__ == "__main__":
    unittest.main()
