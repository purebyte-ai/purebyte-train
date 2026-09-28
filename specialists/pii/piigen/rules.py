"""The RULES baseline: strong regular expressions per type, in the style of pattern-based recognisers (pattern +
validation + context words that raise the score), implemented without dependencies. It is the classic rival of every
type in the `pii_spans` vertical.

A rule without its checksum would be a strawman, so every structured rule validates: Luhn with the issuer prefixes
(cards), mod-97 (IBAN), the control letter (DNI/NIE), `ipaddress` + the non-identifying set (IP), an existing top-level
domain (e-mail). Context rules (passport, date of birth, bare national phone numbers) need a trigger word in the 40-50
characters BEFORE the value. Names use the public name lists (ALL of them, held-out half included: a rules engineer
would use them, so on an evaluation built with held-out names the comparison favours the rules), addresses use
street-type words and number patterns. SECRET uses the credential-format catalogue of the secrets-code specialist
plus a keyword rule (`password = literal`) and private-key headers.

    detect(data: bytes | str, types=None) -> [(start, end, type index, score)]   byte offsets, sorted, non-overlapping

Text is decoded with `surrogateescape` (every byte maps back); offsets are converted to bytes at the end. Overlaps are
resolved by `PRIORITY` (the more specific type wins), then by length.
"""
import re
import numpy as np
from . import validators as V
from .types import IDX

def _byte_offsets(text):
    if text.isascii():
        return np.arange(len(text) + 1, dtype=np.int64)
    lens = np.fromiter((1 if 56448 <= ord(c) <= 56575 else len(c.encode('utf-8')) for c in text), dtype=np.int64, count=len(text))
    cum = np.zeros(len(text) + 1, dtype=np.int64)
    np.cumsum(lens, out=cum[1:])
    return cum

def _ctx(text, a, words_rx, width=45):
    return bool(words_rx.search(text[max(0, a - width):a]))
_PLACEHOLDER = re.compile('^(?:user(?:name)?|usuario|pass(?:word)?|contraseña|x+|\\*+|<[^>]*>|\\$\\{[^}]*\\}|\\{\\{[^}]*\\}\\}|%s|\\$[A-Z_]+|changeme|change_me|secret|example|dummy|test|null|none|todo|redacted|your[_-]?\\w*)$', re.I)
_EMAIL = re.compile('(?<![\\w.%+\\-@])([A-Za-z0-9._%+\\-]{1,64}@(?:[A-Za-z0-9](?:[A-Za-z0-9\\-]{0,61}[A-Za-z0-9])?\\.)+([A-Za-z]{2,24}))(?![\\w\\-@])')

def r_email(text):
    for m in _EMAIL.finditer(text):
        s = m.group(1)
        if m.group(2).lower() not in V.TLDS:
            continue
        if s.lower().startswith('git@') and text[m.end():m.end() + 1] == ':':
            continue
        yield (m.start(1), m.end(1), 0.9)
_PHONE_CTX = re.compile('(?i)(\\btel\\b|\\btel\\.|\\btlf|t[eé]l[eé]fono|telephone|phone|m[oó]vil|mobile|\\bcell\\b|\\bfax\\b|\\bllam[aeoá]|\\bcall\\b|whatsapp|contact|\\bsms\\b|\\btfno|\\bm[oó]v\\.)')
_PHONE_INTL = re.compile('(?<![\\w+])((?:\\+|00)\\d{1,3}[\\s.\\-]?(?:\\(\\d{1,4}\\)[\\s.\\-]?)?\\d{1,4}(?:[\\s.\\-]?\\d{2,4}){1,5})(?![\\w])')
_PHONE_US = re.compile('(?<![\\w(])(\\(?\\d{3}\\)?[\\s.\\-]\\d{3}[\\s.\\-]\\d{4})(?![\\w])')
_PHONE_ES = re.compile('(?<![\\w.,/\\-])([6789]\\d{2}(?:[\\s.\\-]?\\d{2,3}){2,3})(?![\\w.,/\\-])')
_PHONE_UK = re.compile('(?<![\\w])(0\\d{2,4}[\\s]?\\d{3,4}[\\s]?\\d{3,4})(?![\\w])')

def r_phone(text):
    seen = []
    for (rx, base) in ((_PHONE_INTL, 0.75), (_PHONE_US, 0.6), (_PHONE_ES, 0.35), (_PHONE_UK, 0.3)):
        for m in rx.finditer(text):
            s = m.group(1)
            d = V.digits(s)
            if not 7 <= len(d) <= 15:
                continue
            if rx is _PHONE_ES and len(d) != 9:
                continue
            grouped = bool(re.search('[\\s.\\-()]', s))
            score = base + (0.15 if grouped and rx is not _PHONE_INTL else 0) + (0.3 if _ctx(text, m.start(1), _PHONE_CTX) else 0)
            if score >= 0.5 and (not any((a < m.end(1) and m.start(1) < b for (a, b) in seen))):
                seen.append((m.start(1), m.end(1)))
                yield (m.start(1), m.end(1), min(score, 1.0))
_IBAN = re.compile('(?<![A-Za-z0-9])([A-Z]{2}\\d{2}(?:[ \\-]?[A-Z0-9]{4}){2,7}(?:[ \\-]?[A-Z0-9]{1,4})?)(?![A-Za-z0-9])')

def r_iban(text):
    for m in _IBAN.finditer(text):
        s = m.group(1)
        parts = re.split('([ \\-])', s)
        while parts:
            cand = ''.join(parts)
            if V.iban_ok(cand):
                yield (m.start(1), m.start(1) + len(cand), 1.0)
                break
            parts = parts[:-2]
_CARD = re.compile('(?<![\\d\\-])((?:\\d[ \\-]?){12,18}\\d)(?![\\d\\-])')
_IIN = re.compile('^(?:4|5[1-5]|2(?:2[2-9][1-9]|2[3-9]\\d|[3-6]\\d\\d|7[01]\\d|720)|3[47]|6(?:011|5|4[4-9])|3(?:0[0-5]|[68])|35(?:2[89]|[3-8]))')

def r_card(text):
    for m in _CARD.finditer(text):
        s = m.group(1)
        d = V.digits(s)
        if 13 <= len(d) <= 19 and V.luhn_ok(d) and _IIN.match(d) and (len(set(d)) >= 2):
            yield (m.start(1), m.end(1), 0.95)
_IPV4 = re.compile('(?<![\\d.])((?:25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]?\\d)(?:\\.(?:25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]?\\d)){3})(?![\\d.])')
_IPV6 = re.compile('(?<![0-9A-Fa-f:])((?:[0-9A-Fa-f]{1,4}:){1,7}:?(?:[0-9A-Fa-f]{1,4})?(?::[0-9A-Fa-f]{1,4}){0,6})(?![0-9A-Fa-f:])')
_VERSION_CTX = re.compile('(?i)(version|versi[oó]n|\\bver\\b|\\bv\\s*$|build|firmware|release|oid)')

def r_ip(text):
    for m in _IPV4.finditer(text):
        s = m.group(1)
        if not V.ip_identifying(s) or _ctx(text, m.start(1), _VERSION_CTX, 16):
            continue
        yield (m.start(1), m.end(1), 0.85)
    for m in _IPV6.finditer(text):
        s = m.group(1)
        if s.count(':') >= 2 and V.ipv6_ok(s) and V.ip_identifying(s) and re.search('[0-9A-Fa-f]{3}', s):
            yield (m.start(1), m.end(1), 0.85)
_DNI = re.compile('(?<![A-Za-z0-9])((?:\\d{8}|\\d{2}\\.\\d{3}\\.\\d{3}|[XYZxyz]-?\\d{7})[\\s\\-]?[A-Za-z])(?![A-Za-z0-9])')

def r_dni(text):
    for m in _DNI.finditer(text):
        if V.dni_nie_ok(m.group(1)):
            yield (m.start(1), m.end(1), 0.95)
_PASS_CTX = re.compile('(?i)(pasaporte|passport|pass\\.?\\s*n|n[º°o]\\.?\\s*de\\s*pasaporte)')
_PASSPORT = re.compile('(?<![A-Za-z0-9])([A-Z]{3}\\d{6}|[A-Z]\\d{8}|\\d{9}|\\d{2}[A-Z]{2}\\d{5}|[CFGHJKLMNPRTVWXYZ][CFGHJKLMNPRTVWXYZ0-9]{8}|[A-Z]{2}\\d{7})(?![A-Za-z0-9])')

def r_passport(text):
    for m in _PASSPORT.finditer(text):
        if _ctx(text, m.start(1), _PASS_CTX, 40):
            yield (m.start(1), m.end(1), 0.7)
_MES = '(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre|ene|feb|mar|abr|may|jun|jul|ago|sep|sept|oct|nov|dic|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)'
_DATE = re.compile('(?<![\\w/.\\-])((?:19|20)\\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\\d|3[01])|(?:0?[1-9]|[12]\\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)?\\d{2}|(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\\d|3[01])/(?:19|20)?\\d{2}|(?:0?[1-9]|[12]\\d|3[01])(?:\\s+de)?\\s+' + _MES + '\\.?(?:\\s+de)?,?\\s+(?:19|20)\\d{2}|' + _MES + '\\.?\\s+(?:0?[1-9]|[12]\\d|3[01]),?\\s+(?:19|20)\\d{2}|(?:0?[1-9]|[12]\\d|3[01])-' + _MES + '-(?:19|20)\\d{2}|(?:19|20)\\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\\d|3[01]))(?![\\w/\\-])', re.I)
_DOB_CTX = re.compile('(?i)(nacimiento|nacid[oa]|nac[ií][oó]?\\b|\\bnac\\b|f\\.?\\s*nac|fecha_nac|fnac|birth|born|\\bdob\\b|d\\.o\\.b|cumplea[ñn]os|birthday|bday)')

def r_dob(text):
    for m in _DATE.finditer(text):
        if _ctx(text, m.start(1), _DOB_CTX, 50):
            yield (m.start(1), m.end(1), 0.75)
_URLC = re.compile('[A-Za-z][A-Za-z0-9+.\\-]*://([^\\s:/@\'\\"`<>]+):([^\\s/@\'\\"`<>]+)@')

def r_urlc(text):
    for m in _URLC.finditer(text):
        (u, p) = (m.group(1), m.group(2))
        if _PLACEHOLDER.match(u) and _PLACEHOLDER.match(p) or _PLACEHOLDER.match(p) or re.search('[${}<>]|%s', u + p):
            continue
        yield (m.start(1), m.end(2), 0.9)
_SECRET_KW = re.compile('(?i)(?:password|passwd|pwd|secret|token|api[_\\-]?key|apikey|access[_\\-]?key|client[_\\-]?secret|contrase[ñn]a|clave|auth)[\\w\\-]{0,20}[\\"\']?\\s*(?:[:=]|=>|:=)\\s*[\\"\']?([^\\s\\"\',;`)}\\]]{6,200})')
_SECRET_RULES = None

def _secret_rules():
    global _SECRET_RULES
    if _SECRET_RULES is None:
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        _SECRET_RULES = []
        try:
            from secretsgen.catalog import RULES
            for r in RULES:
                rx = r.get('regex', '')
                try:
                    _SECRET_RULES.append(re.compile(rx.replace('(?i)', '') if rx.count('(?i)') and (not rx.startswith('(?i)')) else rx))
                except re.error:
                    pass
        except Exception:
            pass
        _SECRET_RULES.append(re.compile('-----BEGIN [A-Z ]*PRIVATE KEY-----'))
    return _SECRET_RULES

def r_secret(text):
    for m in _SECRET_KW.finditer(text):
        v = m.group(1)
        if _PLACEHOLDER.match(v) or re.match('(?i)(os\\.|process\\.env|\\$|\\{|<|%|env\\(|getenv|config|settings|self\\.|this\\.|vault)', v):
            continue
        yield (m.start(1), m.end(1), 0.6)
    for rx in _secret_rules():
        for m in rx.finditer(text):
            g = m.lastindex and m.lastindex >= 1 and m.group(1)
            (a, b) = (m.start(1), m.end(1)) if g and len(m.group(1)) >= 8 else (m.start(), m.end())
            yield (a, b, 0.8)
_NAME_TOKEN = "[A-ZÁÉÍÓÚÑÜÀÈÒÇ][a-záéíóúñüàèòçï'’]+(?:-[A-ZÁÉÍÓÚÑÜ][a-záéíóúñüàèòçï]+)?"
_NAME_SEQ = re.compile('(?<![\\w@.])(' + _NAME_TOKEN + '(?:\\s+(?:(?:de|del|de la|de los|van|von|da|di|le|mac|y)\\s+)?' + _NAME_TOKEN + '){0,3})(?![\\w@])')
_TITLE = re.compile('(?:Sr|Sra|Srta|Dña|Dr|Dra|Mr|Mrs|Ms|Miss|Prof|Don|Doña|Dear|Estimad[oa]|Querid[oa]|Hola|Hi|Hello)\\.?,?\\s+$')
_NAME_CTX = re.compile("(?i)(name|nombre|apellidos|author|autor|firmado|signed|contact|contacto|titular|holder|cliente|customer|patient|paciente|employee|empleado|from|de parte de|atentamente|regards|saludos|attn|a/a|soy|me llamo|my name is|i am|i'm|firma|atte|con|with|to|para|by|por)\\W{0,6}$")
_NAMES = None

def _name_sets():
    global _NAMES
    if _NAMES is None:
        from .names import _read
        first = set()
        for f in ('names_es_male.tsv', 'names_es_female.tsv', 'names_en_male.tsv', 'names_en_female.tsv'):
            for r in _read(f):
                for w in r[0].split():
                    first.add(w.lower())
        sur = {r[0].lower() for f in ('surnames_es.tsv', 'surnames_en.tsv') for r in _read(f)[:60000]}
        _NAMES = (first, sur)
    return _NAMES
_NOT_NAMES = {'el', 'la', 'los', 'las', 'the', 'this', 'that', 'and', 'for', 'with', 'para', 'con', 'por', 'una', 'uno', 'un', 'de', 'del', 'hola', 'gracias', 'thanks', 'dear', 'hello', 'hi', 'estimado', 'estimada', 'querido', 'querida', 'calle', 'avenida', 'plaza', 'paseo', 'street', 'avenue', 'road', 'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo'}

def r_name(text):
    (first, sur) = _name_sets()
    for m in _NAME_SEQ.finditer(text):
        tk = [(x.start() + m.start(1), x.end() + m.start(1), x.group(0)) for x in re.finditer('\\S+', m.group(1))]
        while tk and (tk[0][2].lower() in _NOT_NAMES or (tk[0][2].lower() not in first and tk[0][2].lower() not in sur)):
            tk.pop(0)
        while tk and tk[-1][2].lower() in ('de', 'del', 'la', 'los', 'las', 'van', 'von', 'da', 'di', 'le', 'mac', 'y'):
            tk.pop()
        if not tk:
            continue
        toks = [x[2] for x in tk]
        low = [t.lower() for t in toks]
        (a0, b0) = (tk[0][0], tk[-1][1])
        before = text[max(0, a0 - 30):a0]
        titled = bool(_TITLE.search(before))
        ctx = bool(_NAME_CTX.search(before))
        f0 = low[0] in first
        sur_hits = sum((1 for t in low[1:] if t in sur))
        if f0 and len(toks) >= 2 and (sur_hits >= 1):
            score = 0.8
        elif f0 and (titled or ctx):
            score = 0.7
        elif titled and (low[0] in sur or f0):
            score = 0.7
        elif ctx and len(toks) >= 2 and (f0 or sur_hits >= 1):
            score = 0.6
        else:
            continue
        yield (a0, b0, score)
_ES_TYPES = '(?:Calle|CALLE|C/\\.?|c/|Cl\\.|Avenida|Avda\\.?|Av\\.|Plaza|Pza\\.|Pl\\.|Paseo|Pº|P\\.º|Carretera|Ctra\\.?|Camino|Cno\\.|Travesía|Trav\\.|Ronda|Rda\\.|Carrer|Avinguda|Avgda\\.|Rúa|Glorieta|Gta\\.|Urbanización|Urb\\.|Pasaje|Pje\\.|Camí|Passeig|Plaça|Kalea|Callejón|Cuesta|Camiño|Camín|Praza|Passatge|Alameda|Bulevar)'
_ADDR_ES = re.compile('(?<![\\w])(' + _ES_TYPES + '\\s+[\\w\'ªº·.\\- ]{2,60}?,?\\s*(?:nº\\s*|n\\.º\\s*|núm\\.\\s*)?(?:\\d{1,4}(?:\\s*bis)?|s/n)(?:,?\\s*(?:\\d{1,2}º\\s*[A-Z0-9]ª?|\\d{1,2}º\\s*\\d{1,2}ª|bajo|\\d{1,2}º\\s*(?:izq|dcha)\\.?|Esc\\.\\s*\\w,\\s*\\d{1,2}º\\s*\\w|piso\\s*\\d+,\\s*puerta\\s*\\d+))?(?:(?:,\\s*|\\s+-\\s+|\\s*\\n\\s*|\\s+)\\d{5}\\s+[A-ZÁÉÍÓÚÑ][\\w\'·\\- ]{1,40}?(?:\\s*\\([\\w ]+\\))?(?=$|[\\s.,;:\\"\'<)\\]]))?)', re.M)
_EN_SFX = '(?:Street|St\\.?|Avenue|Ave\\.?|Road|Rd\\.?|Lane|Ln\\.?|Drive|Dr\\.?|Boulevard|Blvd\\.?|Court|Ct\\.?|Place|Pl\\.?|Way|Terrace|Circle|Close|Crescent|Gardens)'
_ADDR_EN = re.compile("(?<![\\w])(\\d{1,5}\\s+(?:[NSEW]\\.?\\s+|North\\s+|South\\s+)?(?:[A-Z][A-Za-z'\\-]*|\\d{1,3}(?:st|nd|rd|th))(?:\\s+[A-Z][A-Za-z'\\-]*){0,2}\\s+" + _EN_SFX + "(?![\\w])(?:,?\\s*(?:Apt\\.?|Suite|Unit|#)\\s*\\w+)?(?:(?:,\\s*|\\s*\\n\\s*)[A-Z][A-Za-z .'\\-]{1,30},\\s*[A-Z]{2}\\s+\\d{5}(?:-\\d{4})?|(?:,\\s*|\\s*\\n\\s*)[A-Z][a-z]+\\s+[A-Z]{1,2}\\d{1,2}[A-Z]?\\s+\\d[A-Z]{2})?)")

def r_address(text):
    for rx in (_ADDR_ES, _ADDR_EN):
        for m in rx.finditer(text):
            (a, b) = (m.start(1), m.end(1))
            while b > a and text[b - 1] in ' ,.;:\n\t':
                b -= 1
            yield (a, b, 0.7)
RULES = {'EMAIL': r_email, 'PHONE': r_phone, 'IBAN': r_iban, 'CREDIT_CARD': r_card, 'IP': r_ip, 'DNI_NIE': r_dni, 'PASSPORT': r_passport, 'DATE_OF_BIRTH': r_dob, 'URL_CREDENTIALS': r_urlc, 'SECRET': r_secret, 'PERSON_NAME': r_name, 'ADDRESS': r_address}
PRIORITY = {'URL_CREDENTIALS': 0, 'IBAN': 1, 'CREDIT_CARD': 1, 'DNI_NIE': 1, 'EMAIL': 2, 'IP': 2, 'DATE_OF_BIRTH': 3, 'PASSPORT': 3, 'SECRET': 4, 'PHONE': 5, 'ADDRESS': 6, 'PERSON_NAME': 7}

def detect(data, types=None):
    if isinstance(data, str):
        text = data
    else:
        text = bytes(data).decode('utf-8', 'surrogateescape')
    cands = []
    for t in types or RULES:
        for (a, b, sc) in RULES[t](text):
            if b > a:
                cands.append((a, b, t, sc))
    cands.sort(key=lambda c: (PRIORITY[c[2]], -(c[1] - c[0]), c[0]))
    (taken, out) = ([], [])
    for (a, b, t, sc) in cands:
        if any((a < y and x < b for (x, y) in taken)):
            continue
        taken.append((a, b))
        out.append((a, b, t, sc))
    if not out:
        return []
    cum = _byte_offsets(text)
    return sorted(((int(cum[a]), int(cum[b]), IDX[t], float(sc)) for (a, b, t, sc) in out))
