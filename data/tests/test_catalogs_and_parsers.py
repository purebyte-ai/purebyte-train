"""Offline tests of the small parsers: Go string literals (catalogs), the TOML subset, the legacy .xls reader."""
import struct

import pytest

import build_secret_catalogs as C
from pbdata import tomlsub, xls

GO_RULE = r'''
package rules

func Example() *config.Rule {
	r := config.Rule{RuleID: "example-token", Regex: utils.GenerateUniqueTokenRegex(`ex_[a-z0-9]{32}`, false)}
	tps := utils.GenerateSampleSecrets("example", "ex_0123456789abcdef0123456789abcdef")
	tps = append(tps,
		// a comment with "a string" inside
		"token = ex_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", // trailing comment
		`raw = "ex_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"`,
		"built " + secrets.NewSecret(utils.Hex("32")),
	)
	fps := []string{
		"ex_short",
		"escaped \"quote\" and \\ backslash\n",
		fmt.Sprintf("ex_%s", "x"),
	}
	return utils.Validate(r, tps, fps)
}
'''

DETECTOR = r'''
package example
var keyPat = regexp.MustCompile(`\b(ex_[a-z0-9]{32})\b`)
var other = regexp.MustCompile(detectors.PrefixRegex([]string{"example"}) + `\b([0-9a-f]{40})\b`)
'''

DETECTOR_TEST = r'''
package example
var validPattern = "ex_0123456789abcdef0123456789abcdef"
var notAnExample = "hello world"
var url = "https://example.invalid/path/0123456789"
'''


def test_slice_literals():
    tps = C.slice_literals(GO_RULE, "tps")
    assert tps == ["ex_0123456789abcdef0123456789abcdef", "token = ex_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                   'raw = "ex_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"']
    fps = C.slice_literals(GO_RULE, "fps")
    assert fps == ["ex_short", 'escaped "quote" and \\ backslash\n']


def test_trufflehog_catalog_shape():
    files = {"pkg/detectors/example/example.go": DETECTOR, "pkg/detectors/example/example_test.go": DETECTOR_TEST,
             "pkg/detectors/other/v1/other.go": DETECTOR}
    t = C.trufflehog_catalog(files, 16, 200)
    assert t["providers"] == ["example", "v1"]
    assert t["patterns"] == [r"\b(ex_[a-z0-9]{32})\b"]
    assert t["examples"] == ["ex_0123456789abcdef0123456789abcdef"]


GITLEAKS_TOML = "\n".join([
    'title = "t"',
    "[[rules]]",
    'id = "one"',
    "regex = '''(?i)\\bone_[a-z]{8}'''",
    "keywords = [",
    '    "one",',
    '    "uno",',
    "]",
    "[[rules]]",
    'id = "path-only"',
    "path = '''\\.p12$'''",
    "",
])


def test_gitleaks_catalog_shape():
    g = C.gitleaks_catalog({"config/gitleaks.toml": GITLEAKS_TOML, "cmd/generate/config/rules/one.go": GO_RULE})
    assert g["rules"] == [{"id": "one", "regex": "(?i)\\bone_[a-z]{8}", "kw": ["one", "uno"]},
                          {"id": "path-only", "kw": []}]
    assert len(g["tps"]) == 3 and len(g["fps"]) == 2


TOML = "\n".join([
    "# comment",
    'title = "gitleaks config"',
    "[allowlist]",
    'description = "global"',
    "paths = ['''a\\b''', \"c\\\\d\"]",
    "[[rules]]",
    'id = "x"',
    "entropy = 3.5",
    "secretGroup = 1",
    'regex = """two\\',
    '   lines"""',
    'tags = ["a", "b",]',
    "[[rules]]",
    'id = "y"',
    "[rules.allowlist]",
    'regexTarget = "line"',
    "",
])


def test_toml_subset_matches_tomllib_when_available():
    mine = tomlsub._Parser(TOML).parse()
    assert mine["title"] == "gitleaks config"
    assert mine["allowlist"]["paths"] == ["a\\b", "c\\d"]
    assert mine["rules"][0] == {"id": "x", "entropy": 3.5, "secretGroup": 1, "regex": "twolines", "tags": ["a", "b"]}
    assert mine["rules"][1]["allowlist"] == {"regexTarget": "line"}
    tomllib = pytest.importorskip("tomllib")
    assert tomllib.loads(TOML) == mine


def _rec(rtype, data):
    return struct.pack("<HH", rtype, len(data)) + data


def _rk_int(n):
    return (n << 2) | 2


def _tiny_xls(tmp_path):
    """A minimal BIFF8 workbook in an OLE2 container: one sheet; the shared string table is split by a CONTINUE
    record in the middle of a string's characters (the CONTINUE starts with the option byte of the rest)."""
    strings = ["Orden", "GARCÍA", "LÓPEZ"]
    body = struct.pack("<II", 3, 3) + b"".join(struct.pack("<HB", len(x), 1) + x.encode("utf-16-le") for x in strings)
    cut = 8 + (3 + 10) + 3 + 4            # after "Orden", the header of "GARCÍA" and two of its characters
    sst = _rec(0x00FC, body[:cut]) + _rec(0x003C, bytes([1]) + body[cut:])
    zeros16 = bytes(16)
    sheet = b"".join([
        _rec(0x0809, zeros16),
        _rec(0x00FD, struct.pack("<HHHI", 0, 1, 0, 0)),
        _rec(0x0203, struct.pack("<HHHd", 1, 0, 0, 1.0)),
        _rec(0x00FD, struct.pack("<HHHI", 1, 1, 0, 1)),
        _rec(0x027E, struct.pack("<HHHI", 1, 2, 0, _rk_int(1234))),
        _rec(0x00BD, struct.pack("<HH", 2, 0) + struct.pack("<HI", 0, _rk_int(2)) + struct.pack("<HI", 0, _rk_int(7))
             + struct.pack("<H", 1)),
        _rec(0x00FD, struct.pack("<HHHI", 2, 2, 0, 2)),
        _rec(0x000A, b""),
    ])
    name = b"Hoja1"

    def globals_(offset):
        return (_rec(0x0809, zeros16) + sst
                + _rec(0x0085, struct.pack("<IBBBB", offset, 0, 0, len(name), 0) + name) + _rec(0x000A, b""))

    wb = globals_(len(globals_(0))) + sheet
    wb += bytes(max(4096, len(wb) + 511) // 512 * 512 - len(wb))
    n = len(wb) // 512
    fat = [0xFFFFFFFD, 0xFFFFFFFE] + [i + 3 for i in range(n - 1)] + [0xFFFFFFFE]
    fat += [0xFFFFFFFF] * (128 - len(fat))
    header = bytearray(512)
    header[:8] = bytes([0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1])
    struct.pack_into("<HH", header, 30, 9, 6)
    struct.pack_into("<II", header, 44, 1, 1)
    struct.pack_into("<IIIII", header, 56, 4096, 0xFFFFFFFE, 0, 0xFFFFFFFE, 0)
    struct.pack_into("<109I", header, 76, *([0] + [0xFFFFFFFF] * 108))

    def dirent(nm, typ, start, size):
        e = bytearray(128)
        raw = nm.encode("utf-16-le") + bytes(2)
        e[:len(raw)] = raw
        struct.pack_into("<H", e, 64, len(raw))
        e[66] = typ
        struct.pack_into("<IQ", e, 116, start, size)
        return bytes(e)

    directory = dirent("Root Entry", 5, 0xFFFFFFFE, 0) + dirent("Workbook", 2, 2, len(wb)) + bytes(256)
    p = tmp_path / "tiny.xls"
    p.write_bytes(bytes(header) + struct.pack("<128I", *fat) + directory + wb)
    return str(p)


def test_xls_reader_on_a_synthetic_workbook(tmp_path):
    sheets = xls.read_xls(_tiny_xls(tmp_path))
    assert [n for n, _ in sheets] == ["Hoja1"]
    cells = sheets[0][1]
    assert xls.row_values(cells, 0, 3) == ["", "Orden", ""]
    assert xls.row_values(cells, 1, 3) == [1.0, "GARCÍA", 1234.0]
    assert xls.row_values(cells, 2, 3) == [2.0, 7.0, "LÓPEZ"]
    assert xls._rk((3 << 2) | 2 | 1) == 0.03
