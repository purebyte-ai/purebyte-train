"""PII resources: the public lists, the public TRAIN documents, and the sampled public benchmark (evaluation).

    lists          names / surnames / streets / municipalities (Spain, INE), names / surnames / places (US Census),
                   sentences (Tatoeba, filtered; person names turned into {P} slots), street suffixes (Faker)
    public_train   documents with their SOURCE labels (character offsets) from the TRAIN splits of synthetic PII datasets
    bench          documents from the TEST / VALIDATION splits of the same datasets (evaluation sets pii-bench-*)

Every choice is seeded; the same raw files and dataset revisions give the same bytes. Hugging Face datasets stored as
Parquet are read through the datasets-server rows API, which serves the current revision: the pinned revision is
checked first and a different one stops the fetch (unless --allow-revision-drift, recorded in the manifest).
"""
import bz2
import csv
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from . import fetch, paths
from .util import DataError, ensure_dir, log, sha256_bytes, sha256_file, warn, write_json

USER_AGENT = fetch.USER_AGENT


# ================================================================================================= raw files
class RawFiles:
    """The raw list files of data/sources/pii.yaml, verified by sha256: from the download cache, or from a local copy
    (--raw-dir, matched by name and sha256). With `accept_change`, a file whose upstream copy changed is used anyway
    and the new digest is recorded as a deviation."""

    def __init__(self, man, raw_dir=None, accept_change=False):
        self.by_name = {}
        for e in man.entries:
            for f in e.get("files") or []:
                if e["type"] == "url" and f.get("name"):
                    self.by_name[f["name"]] = (e, f)
        self.raw_dir, self.accept_change = raw_dir, accept_change
        self.used, self.deviations = {}, []

    def path(self, name):
        e, f = self.by_name[name]
        if self.raw_dir:
            p = os.path.join(self.raw_dir, name)
            if os.path.exists(p) and sha256_file(p) == f["sha256"]:
                self.used[name] = dict(entry=e["id"], sha256=f["sha256"], source="local copy")
                return p
        try:
            p = fetch.fetch_file(e)
        except DataError as err:
            if not self.accept_change or "does not match the pin" not in str(err):
                raise
            dest = fetch.unpinned_path(e["url"])
            fetch.http_get(e["url"], dest)
            got = sha256_file(dest)
            self.deviations.append(dict(entry=e["id"], file=name, pinned=f["sha256"], used=got))
            warn(f"{name}: upstream file changed (pinned {f['sha256'][:12]}, using {got[:12]}): recorded as a deviation")
            p = dest
        self.used[name] = dict(entry=e["id"], sha256=sha256_file(p), url=e["url"])
        return p


# ================================================================================================= xlsx (zip + XML)
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def read_xlsx(path):
    """{sheet name: [ {column letter: value} per row ]} (shared / inline strings, numbers as text)."""
    z = zipfile.ZipFile(path)
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
            ss.append("".join(t.text or "" for t in si.iter("{%s}t" % _NS["m"])))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rid2t = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    out = {}
    for sh in wb.find("m:sheets", _NS):
        t = rid2t[sh.get("{%s}id" % _NS["r"])].lstrip("/")
        t = t if t.startswith("xl/") else "xl/" + t
        rows = []
        for row in ET.fromstring(z.read(t)).iter("{%s}row" % _NS["m"]):
            vals = {}
            for c in row.findall("m:c", _NS):
                col = re.match(r"[A-Z]+", c.get("r")).group(0)
                v, typ = c.find("m:v", _NS), c.get("t")
                if typ == "s" and v is not None:
                    vals[col] = ss[int(v.text)]
                elif typ == "inlineStr":
                    vals[col] = "".join(x.text or "" for x in c.iter("{%s}t" % _NS["m"]))
                else:
                    vals[col] = v.text if v is not None else None
            rows.append(vals)
        out[sh.get("name")] = rows
    return out


# ================================================================================================= Spanish casing
# INE writes names, surnames and streets in CAPITALS WITHOUT ACCENTS. Real text has them (and at the byte level an
# accented letter is two bytes), so they are restored by a curated map of frequent words, the -CIÓN / -SIÓN endings and
# the rule of the surnames in -EZ. Cosmetic realism, not linguistics: an unrecognised word stays plain.
_ACC = {w.upper().translate(str.maketrans("ÁÉÍÓÚ", "AEIOU")): w for w in """
José María Jesús Ángel Ramón Martín Andrés Tomás Nicolás Sebastián Julián Agustín Joaquín Fermín Simón Adrián Álvaro Óscar
Víctor Héctor César Iván Rubén Raúl Ángela Mónica Verónica Sofía Lucía Rocío Inés Belén Ascensión Encarnación Concepción
Asunción Purificación Visitación Natividad Remedios Begoña Iñaki Iñigo Aitor Eneko Íker Ainhoa Maite Nerea Garazi Jon Josué
Benjamín Efraín Germán Damián Fabián Cristián Valentín Hernán Sergio Álex Alejandro Cristóbal Gaspar Baltasar Salomón Ismael
Rafael Miguel Gabriel Emilio Mario Darío Lázaro Ignacio Ángeles Mª Díaz Martín Pérez Gómez López Sánchez Álvarez García
Muñoz Rodríguez González Fernández Martínez Hernández Jiménez Ramírez Suárez Vázquez Domínguez Gutiérrez Méndez Núñez
Benítez Márquez Velázquez Sáez Sainz Durán Iglesias Ortín Calderón Castaño León Aragón Andalucía Cádiz Málaga Córdoba Jaén
Almería Ávila Cáceres Álava Castellón Gijón Logroño Alcalá Móstoles Cartagena Mérida Valdés Galdós Góngora Colón Jardín
Estación Travesía Polígono Vía Río Ríos Olímpico Olímpica Músico Música Pública Público Fábrica Mártires Mártir Príncipe
Constitución Médico Pío Tomé Nuñez Marañón Echevarría Etxeberría Menéndez Pidal Ibáñez Muñiz Ordóñez Peñalver Castañeda
""".split()}
_PARTICLES = {"de", "del", "la", "las", "los", "el", "y", "e", "da", "do", "dos", "das", "i", "d", "l", "en", "a"}


def _accent_ez(w):
    if len(w) < 5 or not w.endswith("EZ") or w in ("JEREZ", "PAREZ"):
        return None
    stem = w[:-2]
    i = len(stem) - 1
    while i >= 0:
        c = stem[i]
        if c in "AEIOU" and not (c == "U" and i > 0 and stem[i - 1] in "GQ"):
            return w[:i] + "ÁÉÍÓÚ"["AEIOU".index(c)] + w[i + 1:]
        i -= 1
    return None


def es_word(w):
    if not w:
        return w
    if w in _ACC:
        return _ACC[w]
    if w.endswith("CION") and len(w) > 5:
        base = w[:-4] + "CIÓN"
    elif w.endswith("SION") and len(w) > 5:
        base = w[:-4] + "SIÓN"
    else:
        base = _accent_ez(w) or w
    low = base.lower()
    if low in _PARTICLES:
        return low
    return low[:1].upper() + low[1:]


def es_title(s, first_particle_lower=False):
    out = []
    for k, tok in enumerate(re.split(r"(\s+|-|')", s.strip())):
        if not tok or re.fullmatch(r"\s+|-|'", tok):
            out.append(tok)
            continue
        wd = es_word(tok.upper())
        if k == 0 and not first_particle_lower:
            wd = wd[:1].upper() + wd[1:]
        out.append(wd)
    return "".join(out)


def en_title(s):
    low = s.strip().lower()
    t = low[:1].upper() + low[1:]
    if low.startswith("mc") and len(low) > 3:
        t = "Mc" + low[2].upper() + low[3:]
    return "-".join(p[:1].upper() + p[1:] for p in t.split("-"))


# ================================================================================================= list builders
class Lists:
    def __init__(self, raw, out_dir, build):
        self.raw, self.out, self.build = raw, out_dir, build
        self.counts = {}

    def write_tsv(self, name, rows, header):
        ensure_dir(self.out)
        with open(os.path.join(self.out, name), "w", encoding="utf-8", newline="") as f:
            f.write("\t".join(header) + "\n")
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")
        self.counts[name] = len(rows)
        log(f"  {name}: {len(rows)} rows")

    def read_col(self, name, col=0):
        with open(os.path.join(self.out, name), encoding="utf-8") as f:
            next(f)
            return [ln.rstrip("\n").split("\t")[col] for ln in f if ln.strip()]

    def names_es(self):
        d = read_xlsx(self.raw.path("nombres_por_edad_media.xlsx"))
        for sheet, out in (("Hombres", "names_es_male.tsv"), ("Mujeres", "names_es_female.tsv")):
            rows = [(es_title(r["B"]), int(float(r["C"]))) for r in d[sheet]
                    if (r.get("A") or "").strip().isdigit() and r.get("B") and r.get("C")]
            self.write_tsv(out, rows, ("name", "frequency"))

    def surnames_es(self):
        from .xls import ncols, read_xls, row_values
        rows, seen = [], set()
        for _, cells in read_xls(self.raw.path("apellidos_frecuencia.xls")):
            n = max(3, ncols(cells))
            for i in range(max(cells, default=-1) + 1):
                v = row_values(cells, i, n)
                if isinstance(v[0], float) and isinstance(v[1], str) and v[1].strip() and isinstance(v[2], float):
                    name = es_title(v[1])
                    if name not in seen:
                        seen.add(name)
                        rows.append((name, int(v[2])))
        self.write_tsv("surnames_es.tsv", rows, ("surname", "frequency"))

    PROVINCES = {"01": "Álava", "02": "Albacete", "03": "Alicante", "04": "Almería", "05": "Ávila", "06": "Badajoz",
                 "07": "Illes Balears", "08": "Barcelona", "09": "Burgos", "10": "Cáceres", "11": "Cádiz",
                 "12": "Castellón", "13": "Ciudad Real", "14": "Córdoba", "15": "A Coruña", "16": "Cuenca",
                 "17": "Girona", "18": "Granada", "19": "Guadalajara", "20": "Gipuzkoa", "21": "Huelva", "22": "Huesca",
                 "23": "Jaén", "24": "León", "25": "Lleida", "26": "La Rioja", "27": "Lugo", "28": "Madrid",
                 "29": "Málaga", "30": "Murcia", "31": "Navarra", "32": "Ourense", "33": "Asturias", "34": "Palencia",
                 "35": "Las Palmas", "36": "Pontevedra", "37": "Salamanca", "38": "Santa Cruz de Tenerife",
                 "39": "Cantabria", "40": "Segovia", "41": "Sevilla", "42": "Soria", "43": "Tarragona", "44": "Teruel",
                 "45": "Toledo", "46": "Valencia", "47": "Valladolid", "48": "Bizkaia", "49": "Zamora",
                 "50": "Zaragoza", "51": "Ceuta", "52": "Melilla"}

    def municipalities_es(self):
        d = read_xlsx(self.raw.path("diccionario26.xlsx"))
        rows = []
        for r in next(iter(d.values())):
            if (r.get("B") or "").isdigit() and (r.get("C") or "").isdigit() and r.get("E"):
                name = r["E"].strip()
                m = re.fullmatch(r"(.+), (El|La|Los|Las|L'|A|O|Os|As|Es|Sa|Ses|Els)", name)
                if m:
                    name = m.group(2) + ("" if m.group(2).endswith("'") else " ") + m.group(1)
                rows.append((r["B"] + r["C"], name, self.PROVINCES.get(r["B"], "")))
        self.write_tsv("municipalities_es.tsv", rows, ("code", "name", "province"))
        return {c: (n, p) for c, n, p in rows}

    URBAN_TYPES = {"CALLE": "Calle", "CARRER": "Carrer", "KALE": "Kalea", "KALEA": "Kalea", "RUA": "Rúa",
                   "AVENIDA": "Avenida", "AVINGUDA": "Avinguda", "PLAZA": "Plaza", "PLAÇA": "Plaça", "PRAZA": "Praza",
                   "PASEO": "Paseo", "PASSEIG": "Passeig", "CAMINO": "Camino", "CAMI": "Camí", "CAMIÑO": "Camiño",
                   "CAMIN": "Camín", "CARRETERA": "Carretera", "TRAVESIA": "Travesía", "RONDA": "Ronda",
                   "PASAJE": "Pasaje", "PASSATGE": "Passatge", "GLORIETA": "Glorieta", "CALLEJON": "Callejón",
                   "URBANIZACION": "Urbanización", "BULEVAR": "Bulevar", "ALAMEDA": "Alameda", "CUESTA": "Cuesta"}
    _TAIL_ART = re.compile(r"^(.*?)(?:, | \()(EL|LA|LOS|LAS|DE|DEL|DE LA|DE LOS|DE LAS|L'|D'|DELS|DES|SA|SES|ES|A|O|OS|"
                           r"AS|DA|DO|DAS|DOS|NA|NAS|EN|DE L'|DE SA|DE SES)\)?$")

    def _street_name(self, nv):
        m = self._TAIL_ART.match(nv)
        if m:
            art = m.group(2)
            nv = art + ("" if art.endswith("'") else " ") + m.group(1)
            return es_title(nv, first_particle_lower=True)
        return es_title(nv)

    def streets_es(self, munis):
        """A seeded sample of real streets with the postcode of one of their sections and their municipality. The
        house number the generator adds is random and the person invented: no real address of anyone."""
        n_keep, seed = self.build.get("streets_kept", 80000), self.build.get("seed", 0)
        tipos = {}
        with open(self.raw.path("TiposVia.csv"), encoding="latin-1") as f:
            for r in csv.reader(f, delimiter=";"):
                if len(r) >= 3 and r[0] and r[0] != "TIPO_VIA":
                    tipos.setdefault(r[0].strip(), r[2].strip())
        z = zipfile.ZipFile(self.raw.path("caj_esp_012026.zip"))
        names = {n.split("/")[-1].split(".")[0]: n for n in z.namelist() if not n.endswith("/")}
        vias = {}
        with z.open(names["VIAS"]) as f:
            for raw in f:
                ln = raw.decode("latin-1")
                if len(ln) < 108:
                    continue
                key = ln[0:5] + ln[47:52]
                tv, nv = ln[52:57].strip(), ln[57:107].strip()
                desc = tipos.get(tv, "")
                if nv and desc in self.URBAN_TYPES and not re.search(
                        r"\d{3,}|^S/N|DISEMINADO|DISEMINADOS|[^A-ZÑÇÜÀÈÉÍÏÒÓÚ'·.,()\- ]", nv):
                    vias[key] = (self.URBAN_TYPES[desc], nv)
        cps = {}
        with z.open(names["TRAM"]) as f:
            for raw in f:
                ln = raw.decode("latin-1")
                if len(ln) < 262:
                    continue
                key = ln[0:5] + ln[160:165]
                cp = ln[257:262]
                if key in vias and cp.isdigit() and cp != "00000":
                    cps.setdefault(key, cp)
        keys = sorted(k for k in vias if k in cps and k[:5] in munis)
        rng = np.random.default_rng(seed)
        pick = sorted(rng.choice(len(keys), size=min(n_keep, len(keys)), replace=False))
        rows = []
        for i in pick:
            k = keys[i]
            tv, nv = vias[k]
            muni, prov = munis[k[:5]]
            rows.append((tv, self._street_name(nv), cps[k], muni, prov))
        self.write_tsv("streets_es.tsv", rows, ("type", "name", "cp", "municipality", "province"))

    def names_en(self):
        for src, out in (("dist.male.first", "names_en_male.tsv"), ("dist.female.first", "names_en_female.tsv")):
            rows = []
            with open(self.raw.path(src), encoding="latin-1") as f:
                for ln in f:
                    p = ln.split()
                    if len(p) >= 2 and p[0].isalpha():
                        rows.append((en_title(p[0]), float(p[1])))
            self.write_tsv(out, rows, ("name", "frequency_pct"))
        z = zipfile.ZipFile(self.raw.path("names.zip"))
        rows = []
        for r in csv.DictReader(io.TextIOWrapper(z.open("Names_2010Census.csv"), encoding="latin-1")):
            n = r["name"].strip()
            if n.isalpha() and n != "ALL" and r["count"].isdigit():
                rows.append((en_title(n), int(r["count"])))
        self.write_tsv("surnames_en.tsv", rows, ("surname", "count"))

    def places_us(self):
        z = zipfile.ZipFile(self.raw.path("2020_Gaz_place_national.zip"))
        rows = []
        for r in csv.DictReader(io.TextIOWrapper(z.open("2020_Gaz_place_national.txt"), encoding="latin-1"),
                                delimiter="\t"):
            name = re.sub(r"\s+(city|town|village|CDP|borough|municipality|comunidad|zona urbana|city and borough|"
                          r"unified government \(balance\)|consolidated government \(balance\)|metro government "
                          r"\(balance\)|metropolitan government \(balance\)|urban county)$", "", r["NAME"].strip())
            if name and "(" not in name:
                rows.append((name, r["USPS"].strip()))
        self.write_tsv("places_us.tsv", rows, ("name", "state"))

    def street_affixes_en(self):
        """Only the USPS-style suffix WORDS of Faker's en_US provider, read with `ast` (nothing is executed)."""
        import ast
        with open(self.raw.path("faker_address_en_US.py"), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        words = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "street_suffixes" for t in node.targets):
                words = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
        self.raw.path("faker_LICENSE.txt")
        self.write_tsv("street_affixes_en.tsv", [(w,) for w in words], ("suffix",))

    _WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿ]+)?")
    NEUTRAL = set("""
january february march april may june july august september october november december monday tuesday wednesday thursday
friday saturday sunday enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre lunes martes
miércoles jueves viernes sábado domingo spain españa france francia japan japón germany alemania england inglaterra london
londres paris parís rome roma berlin berlín tokyo tokio america américa europe europa africa áfrica asia china italy italia
portugal brazil brasil mexico méxico argentina canada canadá russia rusia egypt egipto greece grecia australia india ireland
irlanda scotland escocia switzerland suiza sweden suecia holland holanda belgium bélgica austria poland polonia turkey turquía
israel peru perú chile cuba colombia venezuela morocco marruecos korea corea vietnam kenya nigeria norway noruega denmark
dinamarca finland finlandia boston chicago kyoto osaka hawaii madrid barcelona sevilla valencia toledo granada bilbao lisbon
lisboa venice venecia florence florencia athens atenas moscow moscú washington texas california florida york jersey
english spanish french german japanese chinese american italian russian portuguese arabic latin greek dutch irish scottish
british european african asian canadian mexican christian muslim jewish catholic buddhist inglés español francés alemán
japonés chino americano italiano ruso portugués árabe latín griego holandés europeo africano asiático católico cristiano
musulmán judío god dios christmas navidad easter pascua internet tv ok jesus jesús christ cristo allah buddha bible biblia
""".split())

    def sentences(self, lang, first_ranked, surnames, places):
        """Real sentences of two kinds: `clean` (no digit, no '@' / URL, no capitalised word that may be a person) and
        `holes` (the same, but frequent given names become {P} slots the generator fills with SYNTHETIC names). An
        ambiguous sentence is dropped, never kept with an unlabelled name."""
        keep = self.build.get("sentences_kept", 150000)
        seed = self.build.get("seed", 0)
        safe_first = self.build.get("sentences_safe_first_names", 4000)
        src = {"es": "spa_sentences.tsv.bz2", "en": "eng_sentences.tsv.bz2"}[lang]
        texts = []
        with bz2.open(self.raw.path(src), "rt", encoding="utf-8", errors="replace") as f:
            for ln in f:
                p = ln.rstrip("\n").split("\t")
                if len(p) >= 3:
                    texts.append(p[2])
        lowfreq = {}
        for t in texts:
            for w in self._WORD.findall(t):
                if w.islower():
                    lowfreq[w] = lowfreq.get(w, 0) + 1
        first = {n.lower() for n in first_ranked}
        safe = {n.lower() for n in first_ranked[:safe_first]}
        sur = {n.lower() for n in surnames}
        place = {p.lower() for p in places} | self.NEUTRAL
        out = []
        for t in texts:
            if not (20 <= len(t) <= 220) or re.search(r"\d|@|https?:|www\.|\.com|\t|[{}]", t):
                continue
            toks = list(self._WORD.finditer(t))
            slots, bad = [], False
            for k, m in enumerate(toks):
                w = m.group(0)
                if not w[:1].isupper() or len(w) < 2:
                    continue
                lw = w.lower()
                if lw not in first and lw not in sur:
                    continue
                initial = k == 0 or t[:m.start()].rstrip().endswith((".", "?", "!", "¿", "¡", "\"", "«", "—", ":"))
                common = lowfreq.get(lw, 0) >= 20
                if lw in place or common:
                    if initial and common and lw not in place:
                        continue
                    bad = True
                    break
                if lw in safe:
                    slots.append((m.start(), m.end()))
                elif lw in sur and slots and slots[-1][1] + 1 == m.start() and t[slots[-1][1]] == " ":
                    slots[-1] = (slots[-1][0], m.end())
                else:
                    bad = True
                    break
            if bad:
                continue
            if not slots:
                out.append(("clean", t))
                continue
            merged = []
            for a, b in slots:
                if merged and t[merged[-1][1]:a] == " ":
                    merged[-1] = (merged[-1][0], b)
                else:
                    merged.append((a, b))
            s, last = [], 0
            for a, b in merged:
                s.append(t[last:a].replace("{", "(").replace("}", ")"))
                s.append("{P}")
                last = b
            s.append(t[last:].replace("{", "(").replace("}", ")"))
            out.append(("holes", "".join(s)))
        rng = np.random.default_rng(seed)
        clean = [x for x in out if x[0] == "clean"]
        holes = [x for x in out if x[0] == "holes"]
        sel = [clean[i] for i in sorted(rng.choice(len(clean), size=min(keep, len(clean)), replace=False))] + \
              [holes[i] for i in sorted(rng.choice(len(holes), size=min(keep // 2, len(holes)), replace=False))]
        self.write_tsv(f"sentences_{lang}.tsv", sel, ("kind", "text"))

    def build_all(self, sentences=True):
        """Every list. `sentences=False` leaves out the two sentence lists (from the Tatoeba exports, which Tatoeba
        replaces every week): the other ten do not depend on them, and a recipe without prose or e-mail scenes never
        reads them."""
        self.names_es()
        self.surnames_es()
        munis = self.municipalities_es()
        self.streets_es(munis)
        self.names_en()
        self.places_us()
        self.street_affixes_en()
        heads = [self.read_col(n) for n in ("names_en_male.tsv", "names_en_female.tsv", "names_es_male.tsv",
                                            "names_es_female.tsv")]
        ranked, seen = [], set()
        for i in range(max(len(h) for h in heads)):
            for h in heads:
                if i < len(h):
                    for w in h[i].split():
                        if w.lower() not in seen:
                            seen.add(w.lower())
                            ranked.append(w)
        surn = self.read_col("surnames_es.tsv")[:30000] + self.read_col("surnames_en.tsv")[:30000]
        places = [p for p in self.read_col("places_us.tsv") if " " not in p] + \
                 [m for m in self.read_col("municipalities_es.tsv", 1) if " " not in m]
        if not sentences:
            return
        self.sentences("es", ranked, surn, places)
        self.sentences("en", ranked, surn, places)


LICENSES_MD = """# PII lists: sources and licences

Built by `data/fetch_pii.py lists` from the files pinned in `data/sources/pii.yaml`. No list holds personal data: they
are FREQUENCY lists of given names and surnames, a street map, a gazetteer and volunteer-written sentences. The generator
composes SYNTHETIC people from them (a frequent name with frequent surnames, a real street with a random number), linked
to nobody.

| list | source | licence |
|---|---|---|
| names_es_{male,female}.tsv, surnames_es.tsv, streets_es.tsv, municipalities_es.tsv | Instituto Nacional de Estadística (INE) | CC BY 4.0 |
| names_en_{male,female}.tsv, surnames_en.tsv, places_us.tsv | US Census Bureau | public domain (17 U.S.C. 105) |
| sentences_{es,en}.tsv | Tatoeba (tatoeba.org), filtered; in the `holes` kind, name words replaced by `{P}` slots | CC BY 2.0 FR |
| street_affixes_en.tsv | Faker (joke2k/faker), en_US street suffix words | MIT |

Attribution required by CC BY: "Contains data from the Instituto Nacional de Estadística (INE), www.ine.es, CC BY 4.0"
and "Sentences from Tatoeba (tatoeba.org), CC BY 2.0 FR". The raw files and their sha256 are listed in
`SHA256.json` (field `raw`).
"""


# ================================================================================================= Hugging Face
def hf_get_json(url, tries=8):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception as e:                    # the datasets-server API is flaky: retry with back-off
            last = e
            time.sleep(3 + 4 * i)
    raise DataError(f"{url}: {last}")


def hf_revision(dataset):
    return hf_get_json(f"https://huggingface.co/api/datasets/{dataset}").get("sha")


def check_revision(entry, allow_drift, report):
    cur = hf_revision(entry["dataset"])
    pinned = entry["pin"]["revision"]
    rec = dict(dataset=entry["dataset"], pinned=pinned, current=cur)
    if cur != pinned:
        msg = (f"{entry['dataset']}: the current revision is {cur}, the manifest pins {pinned}. The rows API only serves "
               f"the current revision, so the sample would differ")
        if not allow_drift:
            raise DataError(msg + " (pass --allow-revision-drift to sample it anyway; the manifest will say so).")
        warn(msg)
        rec["drift"] = True
    report.append(rec)
    return cur


def hf_rows(dataset, split, offset, length):
    q = urllib.parse.urlencode(dict(dataset=dataset, config="default", split=split, offset=offset, length=length))
    d = hf_get_json("https://datasets-server.huggingface.co/rows?" + q)
    return [r["row"] for r in d["rows"]], d.get("num_rows_total")


class ByteRanges:
    """Byte-range reads of one pinned JSONL file (the file is sorted by language: a binary search finds a segment)."""

    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def get(self, a, b):
        last = None
        for i in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={a}-{b}"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    return r.read()
            except Exception as e:
                last = e
                time.sleep(3 + 4 * i)
        raise DataError(f"{self.url} bytes {a}-{b}: {last}")

    def lang_at(self, off):
        for ln in self.get(off, off + 200000).split(b"\n")[1:-1]:
            try:
                return json.loads(ln)["language"]
            except (ValueError, KeyError):
                continue
        return None

    def segment(self, lang, lo=0):
        def first_at_least(cmp):
            a, b = lo, self.size
            while b - a > 400000:
                mid = (a + b) // 2
                cur = self.lang_at(mid)
                if cur is None or cmp(cur):
                    b = mid
                else:
                    a = mid
            return a
        return first_at_least(lambda x: x >= lang), first_at_least(lambda x: x > lang)


class LocalRanges(ByteRanges):
    """The same reads on a local copy of the file (a pinned download already on disk): the same bytes, no network.
    `get(a, b)` is inclusive, as an HTTP Range request."""

    def __init__(self, path):
        self.url, self.path = path, path
        self.size = os.path.getsize(path)

    def get(self, a, b):
        with open(self.path, "rb") as f:
            f.seek(a)
            return f.read(max(0, min(b, self.size - 1) - a + 1))


def openpii_url(entry, path):
    return f"https://huggingface.co/datasets/{entry['dataset']}/resolve/{entry['pin']['revision']}/{path}"


def _openpii_doc(r, lang, prefix):
    text = r["source_text"]
    labs = []
    for m in r["privacy_mask"]:
        a, b = int(m["start"]), int(m["end"])
        if text[a:b] != m["value"]:
            j = text.find(m["value"])
            if j < 0:
                continue
            a, b = j, j + len(m["value"])
        labs.append((a, b, m["label"]))
    return make_doc(f"{prefix}-{r['uid']}", "ai4privacy/pii-masking-openpii-1m", lang, text, labs)


def make_doc(uid, source, lang, text, labs, **extra):
    """A document with SOURCE labels as character offsets of `text` (spans outside the text or blank are dropped)."""
    out = [[a, b, str(lab)] for a, b, lab in labs if 0 <= a < b <= len(text) and text[a:b].strip()]
    d = dict(id=uid, source=source, lang=lang, text=text, labels=sorted(out))
    d.update({k: v for k, v in extra.items() if v is not None})
    return d


def _nemotron_labels(r, t):
    import ast
    sp = r["spans"]
    if isinstance(sp, str):
        sp = json.loads(sp) if sp.startswith('[{"') else ast.literal_eval(sp)
    out = []
    for s in sp:
        a, b = int(s["start"]), int(s["end"])
        st = None if s.get("text") is None else str(s["text"])
        if st is not None and t[a:b] != st:
            j = t.find(st, max(0, a - 20))
            if j < 0:
                continue
            a, b = j, j + len(st)
        out.append((a, b, s["label"]))
    return out


def _gretel_labels(r, t):
    sp = r.get("pii_spans")
    sp = json.loads(sp) if isinstance(sp, str) else sp
    return [(int(s["start"]), int(s["end"]), s["label"]) for s in sp]


def _gretel_en_labels(r, t):
    """gretel-pii-masking-en-v1 lists entities without offsets: every occurrence of the entity text is labelled."""
    import ast
    ents = r.get("entities")
    if isinstance(ents, str):
        try:
            ents = json.loads(ents)
        except ValueError:
            ents = ast.literal_eval(ents)
    out = []
    for e in ents or []:
        if "start" in e and "end" in e:
            out.append((int(e["start"]), int(e["end"]), e.get("label") or (e.get("types") or ["pii"])[0]))
            continue
        s = str(e.get("entity", "")).strip()
        typ = (e.get("types") or [e.get("type", "pii")])[0]
        if len(s) < 2:
            continue
        for m in re.finditer(re.escape(s), t):
            out.append((m.start(), m.end(), typ))
    return out


# ================================================================================================= public TRAIN
def train_openpii(entry, n_per_lang, slices=30, width=900000, ranges=None):
    """`ranges`: a LocalRanges over a local copy of the pinned file (default: byte ranges over HTTPS)."""
    br = ranges or ByteRanges(openpii_url(entry, "data/train.jsonl"))
    docs = []
    for lang in ("en", "es"):
        s0, s1 = br.segment(lang)
        per = n_per_lang // slices + 1
        for k, off in enumerate(np.linspace(s0, s1 - width, slices).astype(np.int64)):
            n = 0
            for ln in br.get(int(off), int(off) + width).split(b"\n")[1:-1]:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if r.get("language") != lang:
                    continue
                docs.append(_openpii_doc(r, lang, "ai4p-train"))
                n += 1
                if n >= per:
                    break
            log(f"    {lang} slice {k + 1}/{slices}: {n}")
    return docs


def train_paged(dataset, split, n, page=100, seed=0, lang_of=lambda r: "en", keep=lambda r: True, text_of=None,
                labels_of=None, rows=None):
    """`rows`: rows(dataset, split, offset, length) -> (rows, total), the rows of a local copy of the pinned file in
    the order the rows API serves them (default: the rows API)."""
    rows = rows or hf_rows
    _, total = rows(dataset, split, 0, 1)
    rng = np.random.default_rng(seed)
    pages = rng.permutation(max(1, total // page))
    docs = []
    for k, pg in enumerate(pages):
        rs, _ = rows(dataset, split, int(pg) * page, page)
        for r in rs:
            if keep(r):
                t = text_of(r)
                labs = labels_of(r, t)
                out = [[a, b, str(lab)] for a, b, lab in labs if 0 <= a < b <= len(t) and t[a:b].strip()]
                docs.append(dict(id=f"{dataset.split('/')[1]}-{split}-{int(pg) * page + len(docs)}", source=dataset,
                                 lang=lang_of(r), text=t, labels=sorted(out)))
        log(f"    {dataset} {split} page {k + 1}: {len(docs)}/{n}")
        if len(docs) >= n:
            break
    return docs[:n]


def sample_train(entry, per_source, local=None):
    """The TRAIN sample of one source. `local` (optional): {dataset: LocalRanges or rows function} to draw the same
    sample from local copies of the pinned files instead of the network (build_pii_pool.py does)."""
    s = entry.get("sampling") or {}
    frac = s.get("fraction", 1.0)
    src = (local or {}).get(entry["dataset"])
    if entry["dataset"] == "ai4privacy/pii-masking-openpii-1m":
        return train_openpii(entry, per_source // 2, slices=s.get("slices", 30), ranges=src)
    n = int(per_source * frac) if frac != 1.0 else per_source
    if entry["dataset"] == "nvidia/Nemotron-PII":
        return train_paged(entry["dataset"], "train", n, seed=s.get("seed", 1), text_of=lambda r: r["text"],
                           labels_of=_nemotron_labels, rows=src)
    if entry["dataset"] == "gretelai/gretel-pii-masking-en-v1":
        return train_paged(entry["dataset"], "train", n, seed=s.get("seed", 2), text_of=lambda r: r["text"],
                           labels_of=_gretel_en_labels, rows=src)
    if entry["dataset"] == "gretelai/synthetic_pii_finance_multilingual":
        langs = set(s.get("languages") or ("Spanish", "English"))
        return train_paged(entry["dataset"], "train", n, seed=s.get("seed", 3),
                           keep=lambda r: r["language"] in langs,
                           lang_of=lambda r: "es" if r["language"] == "Spanish" else "en",
                           text_of=lambda r: r["generated_text"], labels_of=_gretel_labels, rows=src)
    raise DataError(f"{entry['id']}: no sampler for {entry['dataset']}")


def jsonl_bytes(docs):
    return "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs).encode("utf-8")


# ================================================================================================= benchmark (exam)
BENCH_SEED = 20260923


def bench_nemotron(entry):
    s = entry.get("sampling") or {}
    n_test, n_dev, page = s.get("test_docs", 1200), s.get("dev_docs", 600), s.get("page", 50)
    ds = entry["dataset"]
    _, total = hf_rows(ds, "test", 0, 1)
    rng = np.random.default_rng(s.get("seed", BENCH_SEED))
    pages = rng.choice(total // page, size=(n_test + n_dev) // page, replace=False)
    out = {"test": [], "dev": []}
    for k, pg in enumerate(pages):
        rs, _ = hf_rows(ds, "test", int(pg) * page, page)
        part = "test" if k < n_test // page else "dev"
        for r in rs:
            text = r["text"]
            out[part].append(make_doc(f"nemotron-{r['uid']}", ds, "en", text, _nemotron_labels(r, text),
                                      format=r.get("document_format")))
        log(f"    nemotron page {k + 1}/{len(pages)}")
    return out


def bench_openpii(entry):
    s = entry.get("sampling") or {}
    per_test, per_dev = s.get("test_docs_per_language", 1000), s.get("dev_docs_per_language", 500)
    slices, width = s.get("slices", 20), s.get("slice_bytes", 600000)
    br = ByteRanges(openpii_url(entry, s.get("path", "data/validation.jsonl")))
    out = {"test": [], "dev": []}
    for lang in s.get("languages", ["en", "es"]):
        s0, s1 = br.segment(lang)
        per_slice = (per_test + per_dev) // slices
        got = []
        for k, off in enumerate(np.linspace(s0, s1 - width, slices).astype(np.int64)):
            n = 0
            for ln in br.get(int(off), int(off) + width).split(b"\n")[1:-1]:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if r.get("language") != lang:
                    continue
                got.append((k, r))
                n += 1
                if n >= per_slice:
                    break
            log(f"    openpii {lang} slice {k + 1}/{slices}: {n}")
        for k, r in got:
            out["test" if k % 3 != 2 else "dev"].append(_openpii_doc(r, lang, "ai4p"))
    return out


def bench_gretel(entry):
    s = entry.get("sampling") or {}
    ds = entry["dataset"]
    allr, off = [], 0
    while True:
        rs, total = hf_rows(ds, "test", off, 100)
        if not rs:
            break
        allr += rs
        off += 100
        if off >= total:
            break
    es = [r for r in allr if r["language"] == "Spanish"]
    en = [r for r in allr if r["language"] == "English"]
    rng = np.random.default_rng(s.get("seed", BENCH_SEED + 1))
    idx = rng.permutation(len(en))
    n_test, n_dev = s.get("english_test_docs", 600), s.get("english_dev_docs", 300)
    picks = [(r, "test" if i % 2 == 0 else "dev", "es") for i, r in enumerate(es)] + \
            [(en[j], "test", "en") for j in idx[:n_test]] + [(en[j], "dev", "en") for j in idx[n_test:n_test + n_dev]]
    out = {"test": [], "dev": []}
    for r, part, lang in picks:
        text = r["generated_text"]
        out[part].append(make_doc(f"gretel-{r['index']}", ds, lang, text, _gretel_labels(r, text),
                                  format=r.get("document_type")))
    return out


BENCH = {"nvidia/Nemotron-PII": bench_nemotron, "ai4privacy/pii-masking-openpii-1m": bench_openpii,
         "gretelai/synthetic_pii_finance_multilingual": bench_gretel}

# The samples the pii release pool was checked against (build_pii_pool.py's hard check), fetched on 2026-09-24 by this
# code from the pinned revisions: the sha256 of every file as it is written. A fetch compares with them and records
# the result; the Hugging Face rows API serves the current revision only, so a newer revision gives other samples.
BENCH_REFERENCE_SHA256 = {
    ("pii-bench-test", "nemotron-pii.jsonl"): "b706c996cde5d9bf5122d202f28d849e9fb3d306f6e033eecc7d465a47d2b1c9",
    ("pii-bench-test", "openpii-1m.jsonl"): "ff79017da576a6a1c46d0c3dd455c22ac90151ba94ffb56fce254598ff9829dd",
    ("pii-bench-test", "gretel-finance.jsonl"): "d3e0258a9388caec379339e4ccb4583697fe498c8236858ad755aa87ee1a1c5a",
    ("pii-bench-dev", "nemotron-pii.jsonl"): "9a9b8d668ad1a77690e73e1cbc2de6684977221812b21dce7b96774b0eb0ab3e",
    ("pii-bench-dev", "openpii-1m.jsonl"): "5fbacb93771627cc85a6f4e763ea7811b6075d947a0203f162e1b10f42496047",
    ("pii-bench-dev", "gretel-finance.jsonl"): "c71bb16218672a353a07f3a19cb9965da8ba34a3a65627db511f15d4516dccc4",
}


def _declared_sampling(e):
    """The entry whose `sampling` block governs this dataset split: pii-bench-dev lists the same datasets as
    pii-bench-test without repeating it, so both sets are always cut by the parameters written once."""
    if e.get("sampling"):
        return e
    from .manifest import exam_sets
    revision = (e.get("pin") or {}).get("revision")
    for s, _ in exam_sets():
        for x in s["entries"]:
            if x.get("type") == "hf" and x.get("dataset") == e["dataset"] and x.get("sampling") \
                    and (x.get("pin") or {}).get("revision") == revision:
                return x
    return e


def fetch_exam_set(s, dry_run=False, allow_drift=False, sampled=None):
    """Fetch one document evaluation set. The two sampled sets (pii-bench-test / -dev) are cut by one sampling run
    per dataset and written together; `sampled` ({(dataset, revision): record}, shared by one run of fetch_exams.py)
    keeps the second set from sampling them again."""
    from .exams import file_name
    records = []
    root = paths.exam_dir(s["name"])
    for e in s["entries"]:
        if e["type"] == "url":
            if dry_run:
                fetch.fetch_file(e, dry_run=True)
                continue
            p = fetch.fetch_file(e)
            for f in e.get("files") or []:
                dest = os.path.join(root, file_name(f))
                ensure_dir(root)
                with open(p, "rb") as src, open(dest, "wb") as out:
                    out.write(src.read())
                if sha256_file(dest) != f["sha256"]:
                    raise DataError(f"{s['name']}:{e['id']}: {file_name(f)} does not match its sha256")
            records.append(dict(id=e["id"], type="url", files=[file_name(f) for f in e.get("files") or []]))
            continue
        sampler = BENCH.get(e["dataset"])
        if sampler is None:
            raise DataError(f"{s['name']}:{e['id']}: no sampler for {e['dataset']}")
        if dry_run:
            log(f"  would sample {e['dataset']} ({', '.join(e['splits'])}) at revision {e['pin']['revision'][:12]}")
            continue
        key = (e["dataset"], e["pin"]["revision"])
        if sampled is not None and key in sampled:
            log(f"  {e['dataset']}: sampled earlier in this run (pii-bench-test and -dev are cut together)")
            records.append(sampled[key])
            continue
        revs = []
        check_revision(e, allow_drift, revs)
        parts = sampler(_declared_sampling(e))
        name = file_name((e.get("files") or [{"name": e["id"] + ".jsonl"}])[0])
        matches = {}
        for part, docs in parts.items():
            dest_dir = paths.exam_dir(f"pii-bench-{part}")
            ensure_dir(dest_dir)
            data = jsonl_bytes(docs)
            with open(os.path.join(dest_dir, name), "wb") as f:
                f.write(data)
            ref = BENCH_REFERENCE_SHA256.get((f"pii-bench-{part}", name))
            matches[part] = None if ref is None else sha256_bytes(data) == ref
            log(f"  pii-bench-{part}/{name}: {len(docs)} documents"
                + ("" if ref is None else ("  (the reference sample)" if matches[part] else
                                           "  (NOT the reference sample: a newer revision or rows API; recorded)")))
            if matches[part] is False:
                warn(f"pii-bench-{part}/{name} differs from the sample the pii release pool was checked against")
        rec = dict(id=e["id"], type="hf", dataset=e["dataset"], revision=revs, file=name,
                   docs={k: len(v) for k, v in parts.items()},
                   sha256={k: sha256_bytes(jsonl_bytes(v)) for k, v in parts.items()}, matches_reference=matches)
        if sampled is not None:
            sampled[key] = rec
        records.append(rec)
    return records
