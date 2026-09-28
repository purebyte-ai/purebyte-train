"""Synthetic VALUES of every type (positives, all with valid checksums) and their LOOK-ALIKES (negatives: the same
shape in a role that is not personal data).

No value is taken from anyone: names are frequent list names combined at random, numbers are random with their checksum
computed, streets are real streets with a random number. A random value can still coincide with a real one (a valid
card or DNI number, a phone number of a numbering plan in use, a frequent name with a frequent surname). Everything that is not the signal is randomised -- format,
grouping, case, separators -- so what decides the label is WHAT the value is and WHERE it sits, never how it is spelled.
SECRET values come from the credential generator of the secrets-code specialist (`secretsgen`, imported lazily).

Every function takes a `numpy.random.Generator` first; `lang` is "es" or "en".
"""
import string
import unicodedata
from . import validators as V
DIG = string.digits
UP = string.ascii_uppercase
LOW = string.ascii_lowercase
ALNUM = string.ascii_letters + string.digits

def pick(rng, seq):
    return seq[int(rng.integers(0, len(seq)))]

def chance(rng, p):
    return rng.random() < p

def rstr(rng, alphabet, n):
    return ''.join((alphabet[i] for i in rng.integers(0, len(alphabet), int(n))))

def ascii_fold(s):
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
TITLES = {'es': ['Sr.', 'Sra.', 'Srta.', 'D.', 'Dña.', 'Don', 'Doña', 'Dr.', 'Dra.', 'Prof.', 'Lic.', 'Ing.'], 'en': ['Mr.', 'Mrs.', 'Ms.', 'Miss', 'Dr.', 'Prof.', 'Mr', 'Mrs', 'Ms', 'Sir']}

def person(rng, L, lang, style=None):
    g = 'm' if chance(rng, 0.5) else 'f'
    f = L.first(rng, lang, g)
    (s1, s2) = (L.surname(rng, lang), L.surname(rng, lang))
    style = style or ('es' if lang == 'es' else 'en')
    p = rng.random()
    if style == 'es':
        if p < 0.42:
            full = f'{f} {s1} {s2}'
        elif p < 0.7:
            full = f'{f} {s1}'
        elif p < 0.76:
            full = f'{s1} {s2}, {f}'
        elif p < 0.82:
            full = f'{f[0]}. {s1}'
        elif p < 0.86:
            full = f'{f} {s1}-{s2}'
        elif p < 0.91:
            full = f'{f} {s1} {s2}'.upper()
        elif p < 0.95:
            full = f'{s1.upper()} {s2.upper()}, {f}'
        elif p < 0.98:
            f2 = L.first(rng, lang, g)
            full = f'{f} {f2} {s1}'
        else:
            full = f'{f} {s1}'.lower()
    elif p < 0.55:
        full = f'{f} {s1}'
    elif p < 0.68:
        full = f'{f} {pick(rng, UP)}. {s1}'
    elif p < 0.75:
        full = f'{s1}, {f}'
    elif p < 0.8:
        full = f'{f[0]}. {s1}'
    elif p < 0.84:
        full = f'{f} {s1}-{s2}'
    elif p < 0.89:
        full = f'{f} {s1}'.upper()
    elif p < 0.93:
        full = f"{f} {s1} {pick(rng, ['Jr.', 'Sr.', 'II', 'III'])}"
    elif p < 0.97:
        f2 = L.first(rng, lang, g)
        full = f'{f} {f2} {s1}'
    else:
        full = f'{f} {s1}'.lower()
    return (full, f, s1, g)
PROVIDERS = ['gmail.com', 'hotmail.com', 'hotmail.es', 'yahoo.com', 'yahoo.es', 'outlook.com', 'outlook.es', 'live.com', 'icloud.com', 'protonmail.com', 'proton.me', 'gmx.com', 'gmx.es', 'telefonica.net', 'movistar.es', 'orange.es', 'aol.com', 'mail.com', 'zoho.com', 'yandex.com', 'msn.com', 'me.com', 'ya.ru', 'web.de', 'libero.it', 'free.fr']
_WORDS = ['acme', 'iberia', 'nova', 'delta', 'atlas', 'sigma', 'orion', 'vertex', 'boreal', 'cierzo', 'sol', 'mar', 'norte', 'sur', 'levante', 'ponent', 'tramontana', 'aurora', 'lumen', 'nexo', 'pixel', 'quanta', 'vector', 'tecno', 'grupo', 'consulting', 'logistica', 'seguros', 'clinica', 'dental', 'abogados', 'gestoria', 'inmobiliaria', 'hotel', 'viajes', 'digital', 'labs', 'studio', 'systems', 'cloud', 'data', 'health', 'legal', 'finance', 'retail', 'shop', 'market', 'foods', 'energy', 'solar', 'motors', 'design', 'media', 'press', 'school', 'academy', 'uni', 'hospital', 'farma', 'bio']
_TLDS = ['com', 'es', 'net', 'org', 'io', 'dev', 'eu', 'co.uk', 'de', 'fr', 'it', 'com.mx', 'com.ar', 'cat', 'gal', 'eus', 'info', 'biz', 'pt', 'nl', 'ch', 'us', 'ca', 'cl', 'co', 'pe', 'edu', 'gob.es', 'ai', 'app', 'tech']
ROLE_LOCAL = ['info', 'contacto', 'soporte', 'admin', 'ventas', 'rrhh', 'facturacion', 'noreply', 'no-reply', 'support', 'sales', 'hello', 'hola', 'team', 'billing', 'jobs', 'press', 'prensa', 'office', 'oficina', 'reservas', 'bookings', 'help', 'security', 'privacy', 'dpo', 'webmaster', 'postmaster', 'careers', 'hr', 'legal', 'compras', 'pedidos']

def company_domain(rng):
    d = pick(rng, _WORDS) + (pick(rng, ['', '-', '']) + pick(rng, _WORDS) if chance(rng, 0.5) else '')
    if chance(rng, 0.15):
        d = pick(rng, ['mail', 'correo', 'corp', 'eu', 'es', 'dept', 'int']) + '.' + d
    return d + '.' + pick(rng, _TLDS)

def email(rng, L, lang, first=None, sur=None):
    f = ascii_fold(first or L.first(rng, lang)).lower().replace(' ', '')
    s = ascii_fold(sur or L.surname(rng, lang)).lower().replace(' ', '').replace("'", '')
    p = rng.random()
    if p < 0.22:
        local = f'{f}.{s}'
    elif p < 0.34:
        local = f'{f[0]}{s}'
    elif p < 0.42:
        local = f'{f}_{s}'
    elif p < 0.52:
        local = f'{f}{s}{int(rng.integers(1, 100))}'
    elif p < 0.6:
        local = f'{f}{int(rng.integers(1950, 2010))}'
    elif p < 0.66:
        local = f"{f}.{s}+{pick(rng, ['news', 'shop', 'test', 'work', 'github', 'bank'])}"
    elif p < 0.74:
        local = f'{s}.{f[0]}'
    elif p < 0.82:
        local = pick(rng, ROLE_LOCAL)
    elif p < 0.9:
        local = f + rstr(rng, DIG, int(rng.integers(2, 5)))
    else:
        local = rstr(rng, LOW, int(rng.integers(4, 9))) + (pick(rng, ['.', '_', '']) + rstr(rng, LOW + DIG, 4) if chance(rng, 0.5) else '')
    dom = pick(rng, PROVIDERS) if chance(rng, 0.55) else company_domain(rng)
    e = f'{local}@{dom}'
    if chance(rng, 0.04):
        e = e[0].upper() + e[1:]
    if chance(rng, 0.02):
        e = e.upper()
    return e

def email_negative(rng):
    p = rng.random()
    (org, repo) = (rstr(rng, LOW, int(rng.integers(3, 9))), rstr(rng, LOW + '-', int(rng.integers(3, 12))).strip('-') or 'repo')
    if p < 0.18:
        return f"git@{pick(rng, ['github.com', 'gitlab.com', 'bitbucket.org', 'ssh.dev.azure.com'])}:{org}/{repo}.git"
    if p < 0.3:
        return f"{pick(rng, ['icon', 'logo', 'avatar', 'banner', 'button', 'hero', 'bg'])}@{pick(rng, ['2x', '3x', '1.5x'])}.{pick(rng, ['png', 'jpg', 'webp'])}"
    if p < 0.44:
        return f"{pick(rng, ['lodash', 'react', 'vue', 'express', 'typescript', 'eslint', '@types/node', 'axios', 'webpack'])}@{int(rng.integers(0, 20))}.{int(rng.integers(0, 30))}.{int(rng.integers(0, 40))}"
    if p < 0.54:
        return '@' + pick(rng, ['Override', 'Test', 'Autowired', 'Component', 'Injectable', 'dataclass', 'property', 'staticmethod', 'pytest.fixture', 'media', 'import', 'keyframes', 'font-face', 'Deprecated', 'JsonProperty'])
    if p < 0.64:
        return f"{pick(rng, ['nginx', 'postgres', 'redis', 'python', 'node', 'alpine'])}@sha256:{rstr(rng, '0123456789abcdef', 64)}"
    if p < 0.74:
        return f"{rstr(rng, LOW, int(rng.integers(3, 8)))}@{pick(rng, ['localhost', 'example', 'host', 'server', 'db'])}"
    if p < 0.84:
        return '@' + rstr(rng, LOW + '_', int(rng.integers(4, 12)))
    if p < 0.92:
        return f"{pick(rng, ['user', 'usuario', 'name', 'nombre', 'email'])}@{pick(rng, ['domain.tld', 'dominio', 'host.local', 'mail.invalid'])}"
    return f"{{{{ {pick(rng, ['user.email', 'contact.email', 'customer.email'])} }}}}"

def phone(rng, lang):
    p = rng.random()
    if lang == 'es' or p < 0.2:
        first = pick(rng, '6666677789')
        nine = first + rstr(rng, DIG, 8)
        if first == '7':
            nine = '7' + pick(rng, '123456789') + rstr(rng, DIG, 7)
        if first in '89':
            nine = first + pick(rng, '1123456789') + rstr(rng, DIG, 7)
        q = rng.random()
        if q < 0.28:
            s = nine
        elif q < 0.52:
            s = f'{nine[:3]} {nine[3:6]} {nine[6:]}'
        elif q < 0.66:
            s = f'{nine[:3]} {nine[3:5]} {nine[5:7]} {nine[7:]}'
        elif q < 0.72:
            s = f'{nine[:2]} {nine[2:5]} {nine[5:7]} {nine[7:]}' if nine[0] in '89' else f'{nine[:3]}-{nine[3:6]}-{nine[6:]}'
        elif q < 0.78:
            s = f'{nine[:3]}.{nine[3:6]}.{nine[6:]}'
        else:
            s = nine if chance(rng, 0.4) else f'{nine[:3]} {nine[3:6]} {nine[6:]}'
            s = pick(rng, ['+34 ', '+34', '0034 ', '(+34) ', '+34-', '+34 ']) + s
        return s
    q = rng.random()
    if q < 0.35:
        (a, b, c) = (str(int(rng.integers(201, 990))), str(int(rng.integers(200, 999))), rstr(rng, DIG, 4))
        return pick(rng, [f'({a}) {b}-{c}', f'{a}-{b}-{c}', f'{a}.{b}.{c}', f'+1 {a} {b} {c}', f'+1 ({a}) {b}-{c}', f'1-{a}-{b}-{c}', f'+1-{a}-{b}-{c}', f'{a}{b}{c}'])
    if q < 0.55:
        n = rstr(rng, DIG, 4) + ' ' + rstr(rng, DIG, 6)
        return pick(rng, [f'+44 7{rstr(rng, DIG, 3)} {rstr(rng, DIG, 6)}', f'07{rstr(rng, DIG, 3)} {rstr(rng, DIG, 6)}', f'+44 20 {rstr(rng, DIG, 4)} {rstr(rng, DIG, 4)}', f'020 {rstr(rng, DIG, 4)} {rstr(rng, DIG, 4)}', f'+44 1{n[1:]}'])
    if q < 0.68:
        return f"+33 {pick(rng, '67')} {rstr(rng, DIG, 2)} {rstr(rng, DIG, 2)} {rstr(rng, DIG, 2)} {rstr(rng, DIG, 2)}"
    if q < 0.78:
        return f"+49 {pick(rng, ['30', '40', '89', '151', '160', '171'])} {rstr(rng, DIG, int(rng.integers(6, 9)))}"
    if q < 0.86:
        return f"+52 {pick(rng, ['55', '33', '81'])} {rstr(rng, DIG, 4)} {rstr(rng, DIG, 4)}"
    if q < 0.92:
        return f'+54 9 11 {rstr(rng, DIG, 4)}-{rstr(rng, DIG, 4)}'
    return f"+{pick(rng, ['39', '351', '31', '32', '41', '43', '45', '46', '47', '48', '353', '57', '56'])} {rstr(rng, DIG, 3)} {rstr(rng, DIG, 3)} {rstr(rng, DIG, int(rng.integers(3, 5)))}"

def phone_negative(rng):
    p = rng.random()
    if p < 0.14:
        return str(int(rng.integers(600000000, 999999999)))
    if p < 0.26:
        return str(int(rng.integers(1500000000, 1800000000)))
    if p < 0.33:
        return str(int(rng.integers(1500000000000, 1800000000000)))
    if p < 0.42:
        a = int(rng.integers(1, 999))
        b = int(rng.integers(0, 999))
        c = int(rng.integers(0, 999))
        return pick(rng, [f'{a}.{b:03d}.{c:03d}', f'{a}.{b:03d}.{c:03d},{int(rng.integers(0, 99)):02d}', f'{a},{b:03d},{c:03d}'])
    if p < 0.5:
        return f"978-{pick(rng, ['84', '0', '1', '3'])}-{rstr(rng, DIG, 3)}-{rstr(rng, DIG, 4)}-{pick(rng, DIG)}"
    if p < 0.58:
        return f'{int(rng.integers(1, 30))}.{int(rng.integers(0, 20))}.{int(rng.integers(10000, 30000))}'
    if p < 0.66:
        return f'{int(rng.integers(2000, 2030))}{int(rng.integers(1, 13)):02d}{int(rng.integers(1, 29)):02d}{rstr(rng, DIG, int(rng.integers(0, 7)))}'
    if p < 0.74:
        return f'{rstr(rng, DIG, 3)}-{rstr(rng, DIG, 3)}-{rstr(rng, DIG, 3)}'
    if p < 0.82:
        return f"{pick(rng, ['PED', 'INV', 'FAC', 'ORD', 'REF', 'TKT'])}-{rstr(rng, DIG, int(rng.integers(6, 10)))}"
    if p < 0.9:
        return rstr(rng, DIG, int(rng.integers(10, 13)))
    return f'{int(rng.integers(0, 256))}{rstr(rng, DIG, int(rng.integers(5, 8)))}'
_BBAN = {'ES': '20n', 'DE': '18n', 'FR': '10n11c2n', 'IT': '1a10n12c', 'PT': '21n', 'GB': '4a14n', 'NL': '4a10n', 'BE': '12n', 'IE': '4a14n', 'CH': '5n12c', 'AT': '16n', 'PL': '24n', 'SE': '20n', 'DK': '14n', 'NO': '11n', 'FI': '14n', 'LU': '3n13c', 'GR': '7n16c', 'RO': '4a16c', 'CZ': '20n'}

def iban(rng):
    cc = 'ES' if chance(rng, 0.55) else pick(rng, list(_BBAN))
    (spec, bban, i) = (_BBAN[cc], '', 0)
    while i < len(spec):
        j = i
        while spec[j].isdigit():
            j += 1
        (n, kind) = (int(spec[i:j]), spec[j])
        bban += rstr(rng, {'n': DIG, 'a': UP, 'c': UP + DIG + DIG + DIG}[kind], n)
        i = j + 1
    full = cc + V.iban_check(cc, bban) + bban
    assert V.iban_ok(full), full
    q = rng.random()
    if q < 0.45:
        return full
    if q < 0.92:
        return ' '.join((full[k:k + 4] for k in range(0, len(full), 4)))
    if q < 0.96:
        return '-'.join((full[k:k + 4] for k in range(0, len(full), 4)))
    return full

def iban_negative(rng):
    p = rng.random()
    if p < 0.3:
        return rstr(rng, UP, 4) + pick(rng, ['ES', 'DE', 'FR', 'GB', 'IT', 'PT', 'NL']) + rstr(rng, UP + DIG, 2) + (rstr(rng, UP + DIG, 3) if chance(rng, 0.6) else '')
    if p < 0.52:
        letter = pick(rng, 'ABCDEFGHJ')
        seven = rstr(rng, DIG, 7)
        return pick(rng, ['ES', 'ES', 'PT', 'FR']) + letter + seven + V.cif_control(letter, seven)
    if p < 0.7:
        return f"{pick(rng, ['ES', 'FR', 'DE', 'PT'])}-{int(rng.integers(2015, 2031))}-{rstr(rng, DIG, 6)}"
    if p < 0.85:
        return rstr(rng, UP, 2) + rstr(rng, DIG, 2) + rstr(rng, UP + DIG, int(rng.integers(8, 14)))
    return f'{rstr(rng, UP, 2)}{rstr(rng, DIG, 2)} XXXX XXXX XXXX XXXX {rstr(rng, DIG, 4)}'

def credit_card(rng):
    p = rng.random()
    if p < 0.42:
        (pre, n) = ('4', 16)
    elif p < 0.7:
        (pre, n) = (str(int(rng.integers(51, 56))) if chance(rng, 0.7) else str(int(rng.integers(2221, 2721))), 16)
    elif p < 0.84:
        (pre, n) = (pick(rng, ['34', '37']), 15)
    elif p < 0.92:
        (pre, n) = (pick(rng, ['6011', '65', '644', '649']), 16)
    elif p < 0.96:
        (pre, n) = (pick(rng, ['36', '38', '300', '305']), 14)
    else:
        (pre, n) = (str(int(rng.integers(3528, 3590))), 16)
    body = pre + rstr(rng, DIG, n - len(pre) - 1)
    d = body + V.luhn_complete(body)
    q = rng.random()
    if n == 15 and q < 0.6:
        sep = pick(rng, [' ', '-', ' '])
        return f'{d[:4]}{sep}{d[4:10]}{sep}{d[10:]}'
    if q < 0.4:
        return d
    sep = ' ' if q < 0.85 else '-'
    return sep.join((d[k:k + 4] for k in range(0, len(d), 4)))

def card_negative(rng):
    p = rng.random()
    if p < 0.18:
        body = rstr(rng, DIG, 14)
        return body + V.luhn_complete(body)
    if p < 0.36:
        return rstr(rng, '123456789', 1) + rstr(rng, DIG, 15)
    if p < 0.48:
        return pick(rng, ['**** **** **** ', 'XXXX-XXXX-XXXX-', '************', '•••• •••• •••• ']) + rstr(rng, DIG, 4)
    if p < 0.58:
        return f'1Z{rstr(rng, UP + DIG, 6)}{rstr(rng, DIG, 10)}'
    if p < 0.7:
        return f'{int(rng.integers(1, 13)):02d}/{int(rng.integers(24, 34))}'
    if p < 0.8:
        return rstr(rng, DIG, 3)
    if p < 0.9:
        return ' '.join((rstr(rng, DIG, 4) for _ in range(4)))
    return str(int(rng.integers(10 ** 12, 10 ** 13)))

def ip(rng):
    if chance(rng, 0.8):
        p = rng.random()
        if p < 0.2:
            o = [10, int(rng.integers(0, 256)), int(rng.integers(0, 256)), int(rng.integers(1, 255))]
        elif p < 0.4:
            o = [192, 168, int(rng.integers(0, 256)), int(rng.integers(1, 255))]
        elif p < 0.5:
            o = [172, int(rng.integers(16, 32)), int(rng.integers(0, 256)), int(rng.integers(1, 255))]
        else:
            a = int(rng.integers(1, 224))
            while a in (10, 127, 0, 172, 192):
                a = int(rng.integers(1, 224))
            o = [a, int(rng.integers(0, 256)), int(rng.integers(0, 256)), int(rng.integers(1, 255))]
        return '.'.join(map(str, o))
    groups = ['%x' % int(rng.integers(0, 65536)) for _ in range(8)]
    p = rng.random()
    if p < 0.3:
        groups[0] = pick(rng, ['2001', '2a02', '2600', '2a01', '2804', '2400'])
        return ':'.join(groups)
    if p < 0.6:
        k = int(rng.integers(2, 6))
        return ':'.join([pick(rng, ['2001', '2a02', '2600', '2a01'])] + groups[1:k]) + '::' + ':'.join(groups[-int(rng.integers(1, 3)):])
    if p < 0.8:
        return 'fe80::' + ':'.join(groups[-4:])
    return 'fd' + '%02x' % int(rng.integers(0, 256)) + ':' + ':'.join(groups[1:3]) + '::' + groups[-1]

def ip_negative(rng):
    p = rng.random()
    if p < 0.2:
        return pick(rng, ['127.0.0.1', '0.0.0.0', '255.255.255.255', '::1', '::', '255.255.255.0', '255.255.0.0', 'localhost'])
    if p < 0.4:
        return f'{int(rng.integers(0, 20))}.{int(rng.integers(0, 30))}.{int(rng.integers(0, 60))}.{int(rng.integers(0, 3000))}'
    if p < 0.52:
        return '1.3.6.1.' + '.'.join((str(int(rng.integers(1, 400))) for _ in range(int(rng.integers(3, 8)))))
    if p < 0.64:
        return ':'.join(('%02X' % int(rng.integers(0, 256)) for _ in range(6))) if chance(rng, 0.5) else '-'.join(('%02x' % int(rng.integers(0, 256)) for _ in range(6)))
    if p < 0.74:
        return f'{int(rng.integers(0, 24)):02d}:{int(rng.integers(0, 60)):02d}:{int(rng.integers(0, 60)):02d}.{rstr(rng, DIG, 3)}'
    if p < 0.84:
        return '.'.join((str(int(rng.integers(1, 12))) for _ in range(int(rng.integers(3, 5)))))
    if p < 0.92:
        return ':'.join(('%02x' % int(rng.integers(0, 256)) for _ in range(16)))
    return f'10.0.{int(rng.integers(10000, 30000))}.{int(rng.integers(100, 5000))}'

def dni_nie(rng):
    if chance(rng, 0.8):
        n = int(rng.integers(0, 100000000))
        (num, letter) = ('%08d' % n, V.dni_letter(n))
    else:
        pre = pick(rng, 'XYZ')
        seven = '%07d' % int(rng.integers(0, 10000000))
        (num, letter) = (pre + seven, V.dni_letter('XYZ'.index(pre) * 10 ** 7 + int(seven)))
    q = rng.random()
    if q < 0.62:
        s = num + letter
    elif q < 0.82:
        s = f'{num}-{letter}'
    elif q < 0.88 and num[0].isdigit():
        s = f'{num[:2]}.{num[2:5]}.{num[5:]}-{letter}'
    elif q < 0.94:
        s = f'{num} {letter}'
    else:
        s = num + letter.lower()
    assert V.dni_nie_ok(s), s
    return s

def dni_negative(rng):
    p = rng.random()
    if p < 0.35:
        letter = pick(rng, 'ABCDEFGHJNPQRSUVW')
        seven = rstr(rng, DIG, 7)
        return letter + seven + V.cif_control(letter, seven)
    if p < 0.55:
        return rstr(rng, DIG, 8)
    if p < 0.7:
        return f"{int(rng.integers(1000, 9999))} {rstr(rng, 'BCDFGHJKLMNPRSTVWXYZ', 3)}"
    if p < 0.85:
        return rstr(rng, DIG, 5)
    return f"{pick(rng, ['EXP', 'REG', 'SOL', 'EXPTE'])}-{rstr(rng, DIG, 8)}{pick(rng, UP)}"

def passport(rng):
    p = rng.random()
    if p < 0.4:
        return rstr(rng, UP, 3) + rstr(rng, DIG, 6)
    if p < 0.55:
        return rstr(rng, DIG, 9) if chance(rng, 0.5) else pick(rng, UP) + rstr(rng, DIG, 8)
    if p < 0.65:
        return rstr(rng, DIG, 9)
    if p < 0.75:
        return rstr(rng, DIG, 2) + rstr(rng, UP, 2) + rstr(rng, DIG, 5)
    if p < 0.85:
        return rstr(rng, 'CFGHJKLMNPRTVWXYZ', 1) + rstr(rng, 'CFGHJKLMNPRTVWXYZ0123456789', 8)
    if p < 0.93:
        return pick(rng, UP) + rstr(rng, DIG, 8)
    return rstr(rng, UP, 2) + rstr(rng, DIG, 7)

def passport_negative(rng):
    p = rng.random()
    if p < 0.3:
        return rstr(rng, UP, 3) + rstr(rng, DIG, 6)
    if p < 0.5:
        return pick(rng, ['IB', 'VY', 'UX', 'FR', 'BA', 'AF', 'LH', 'U2']) + rstr(rng, DIG, int(rng.integers(3, 5)))
    if p < 0.7:
        return rstr(rng, UP + DIG, 6)
    if p < 0.85:
        return rstr(rng, DIG, 9)
    return f"{pick(rng, ['FAC', 'ALB', 'PED'])}{rstr(rng, DIG, 6)}"
MONTHS = {'es': ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre'], 'en': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']}

def date_str(rng, lang, y, m, d, kind='any'):
    mon = MONTHS[lang][m - 1]
    fm = ['{Y}-{M:02d}-{D:02d}'] * (6 if kind == 'data' else 2)
    if lang == 'es':
        fm += ['{D:02d}/{M:02d}/{Y}', '{D:02d}/{M:02d}/{Y}', '{D}/{M}/{Y}', '{D:02d}-{M:02d}-{Y}', '{D:02d}.{M:02d}.{Y}', '{D} de {mon} de {Y}', '{D} {mon3} {Y}', '{D:02d}/{M:02d}/{y2:02d}']
    else:
        fm += ['{M:02d}/{D:02d}/{Y}', '{M}/{D}/{Y}', '{mon} {D}, {Y}', '{D} {mon} {Y}', '{mon3} {D}, {Y}', '{D}-{mon3}-{Y}', '{M}/{D}/{y2:02d}', '{D:02d}/{M:02d}/{Y}']
    if kind == 'data':
        fm += ['{Y}{M:02d}{D:02d}', '{Y}/{M:02d}/{D:02d}']
    f = pick(rng, fm)
    return f.format(Y=y, M=m, D=d, y2=y % 100, mon=mon, mon3=mon[:3].capitalize() if lang == 'en' else mon[:3])

def date_of_birth(rng, lang, kind='any'):
    (y, m, d) = (int(rng.integers(1935, 2009)), int(rng.integers(1, 13)), int(rng.integers(1, 29)))
    return date_str(rng, lang, y, m, d, kind)

def date_other(rng, lang, kind='any'):
    (y, m, d) = (int(rng.integers(1990, 2031)), int(rng.integers(1, 13)), int(rng.integers(1, 29)))
    if chance(rng, 0.25):
        y = int(rng.integers(1935, 2009))
    return date_str(rng, lang, y, m, d, kind)

def timestamp(rng):
    (y, m, d) = (int(rng.integers(2015, 2031)), int(rng.integers(1, 13)), int(rng.integers(1, 29)))
    (hh, mm, ss) = (int(rng.integers(0, 24)), int(rng.integers(0, 60)), int(rng.integers(0, 60)))
    return pick(rng, [f'{y}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}:{ss:02d}Z', f'{y}-{m:02d}-{d:02d} {hh:02d}:{mm:02d}:{ss:02d}', f'{y}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}:{ss:02d}.{rstr(rng, DIG, 3)}+0{int(rng.integers(0, 3))}:00'])
SCHEMES = ['postgres', 'postgresql', 'mysql', 'mongodb', 'mongodb+srv', 'redis', 'rediss', 'amqp', 'amqps', 'ftp', 'sftp', 'https', 'http', 'smtp', 'smtps', 'ldap', 'mssql', 'mariadb', 'nats', 'kafka']
_USERS = ['admin', 'root', 'postgres', 'app', 'app_user', 'appuser', 'svc', 'service', 'deploy', 'backup', 'reader', 'writer', 'etl', 'api', 'web', 'jenkins', 'ci', 'dev', 'prod', 'readonly', 'replica', 'guest', 'ftpuser', 'mailer', 'grafana']
_PASS_CH = ALNUM + '!$%*-_.~'

def secret_password(rng):
    p = rng.random()
    if p < 0.5:
        return rstr(rng, _PASS_CH, int(rng.integers(10, 25)))
    if p < 0.8:
        return pick(rng, ['Verano', 'Invierno', 'Primavera', 'Dragon', 'Shadow', 'Monkey', 'Madrid', 'Sevilla', 'Barcelona', 'Password', 'Welcome', 'Admin', 'Qwerty', 'Sunshine', 'Galaxy']) + rstr(rng, DIG, int(rng.integers(2, 5))) + pick(rng, ['', '!', '$', '*', '#', '.'])
    return rstr(rng, ALNUM, int(rng.integers(16, 33)))

def url_host(rng):
    p = rng.random()
    if p < 0.3:
        return pick(rng, ['localhost', 'db', 'postgres', 'redis', 'mongo', 'rabbitmq', 'mysql', 'cache', 'broker'])
    if p < 0.6:
        return company_domain(rng)
    if p < 0.8:
        return f"{rstr(rng, LOW, int(rng.integers(3, 8)))}.{pick(rng, ['internal', 'svc.cluster.local', 'local', 'lan'])}"
    return ip(rng)

def url_with_credentials(rng, L=None, lang='en'):
    sch = pick(rng, SCHEMES)
    user = pick(rng, _USERS) if L is None or chance(rng, 0.7) else ascii_fold(L.first(rng, lang)).lower()
    pw = secret_password(rng)
    if chance(rng, 0.1):
        pw = pw.replace('!', '%21').replace('$', '%24').replace('*', '%2A')
    host = url_host(rng)
    port = {'postgres': 5432, 'postgresql': 5432, 'mysql': 3306, 'mongodb': 27017, 'redis': 6379, 'rediss': 6380, 'amqp': 5672, 'amqps': 5671, 'ftp': 21, 'sftp': 22, 'smtp': 587, 'smtps': 465, 'ldap': 389, 'mssql': 1433, 'mariadb': 3306, 'nats': 4222, 'kafka': 9092}.get(sch)
    tail = ''
    if port and chance(rng, 0.6) and (':' not in host):
        tail += f':{port}'
    if chance(rng, 0.6):
        tail += '/' + pick(rng, ['app', 'prod', 'db', 'main', 'orders', 'users', '0', '1', 'vhost', 'inbox', 'api/v1'])
    if chance(rng, 0.2):
        tail += '?' + pick(rng, ['sslmode=require', 'authSource=admin', 'retryWrites=true&w=majority', 'timeout=30'])
    pre = f'{sch}://'
    url = f'{pre}{user}:{pw}@{host}{tail}'
    return (url, (len(pre), len(pre) + len(user) + 1 + len(pw)))

def url_negative(rng):
    sch = pick(rng, SCHEMES)
    host = url_host(rng)
    p = rng.random()
    if p < 0.35:
        return f"{sch}://{host}/{pick(rng, ['app', 'api', 'db', 'health', 'metrics', ''])}"
    if p < 0.55:
        (u, pw) = pick(rng, [('user', 'password'), ('username', 'password'), ('USER', 'PASS'), ('<user>', '<password>'), ('${DB_USER}', '${DB_PASSWORD}'), ('{{user}}', '{{pass}}'), ('xxx', 'xxx'), ('user', 'pass'), ('${POSTGRES_USER}', '${POSTGRES_PASSWORD}'), ('<usuario>', '<contraseña>'), ('%s', '%s')])
        return f'{sch}://{u}:{pw}@{host}'
    if p < 0.75:
        return f'{sch}://{pick(rng, _USERS)}@{host}'
    if p < 0.9:
        return f"https://{company_domain(rng)}/{pick(rng, ['login', 'user', 'es/contacto', 'docs', 'blog'])}"
    return f"ssh://git@{pick(rng, ['github.com', 'gitlab.com'])}/{rstr(rng, LOW, 5)}/{rstr(rng, LOW, 6)}.git"
_TYPE_DISPLAY = {'Calle': ['Calle', 'Calle', 'C/', 'C/', 'c/', 'Cl.', 'Calle de', 'CALLE'], 'Avenida': ['Avenida', 'Avda.', 'Av.', 'Avda'], 'Plaza': ['Plaza', 'Pza.', 'Pl.', 'Plaza de'], 'Paseo': ['Paseo', 'Pº', 'P.º', 'Paseo de'], 'Carretera': ['Carretera', 'Ctra.', 'Ctra'], 'Camino': ['Camino', 'Cno.', 'Camino de'], 'Travesía': ['Travesía', 'Trav.', 'Travesía de'], 'Carrer': ['Carrer', 'C/', 'Carrer de'], 'Avinguda': ['Avinguda', 'Av.', 'Avgda.'], 'Ronda': ['Ronda', 'Rda.'], 'Urbanización': ['Urbanización', 'Urb.'], 'Pasaje': ['Pasaje', 'Pje.'], 'Glorieta': ['Glorieta', 'Gta.']}
UK_TOWNS = ['London', 'Manchester', 'Birmingham', 'Leeds', 'Glasgow', 'Liverpool', 'Bristol', 'Sheffield', 'Edinburgh', 'Cardiff', 'Leicester', 'Nottingham', 'Coventry', 'Brighton', 'Oxford', 'Cambridge', 'York', 'Bath', 'Norwich', 'Exeter']
_EN_STEMS = ['Main', 'Oak', 'Pine', 'Maple', 'Cedar', 'Elm', 'Park', 'Lake', 'Hill', 'Church', 'Mill', 'River', 'Spring', 'Ridge', 'Sunset', 'Highland', 'Forest', 'Meadow', 'Valley', 'Willow', 'Chestnut', 'Walnut', 'Cherry', 'Birch', 'Station', 'High', 'Market', 'Bridge', 'Victoria', 'King', 'Queen', 'Park View', 'Orchard', 'Green', 'Grove', 'North', 'South']
_EN_SUFFIX = ['Street', 'St', 'St.', 'Avenue', 'Ave', 'Ave.', 'Road', 'Rd', 'Rd.', 'Lane', 'Ln', 'Drive', 'Dr', 'Boulevard', 'Blvd', 'Court', 'Ct', 'Place', 'Pl', 'Way', 'Terrace', 'Circle', 'Close', 'Crescent', 'Gardens']
_STATES = ['AL', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'IL', 'IN', 'MA', 'MD', 'MI', 'MN', 'MO', 'NC', 'NJ', 'NY', 'OH', 'OR', 'PA', 'TN', 'TX', 'VA', 'WA', 'WI']

def address(rng, L, lang, multiline=None):
    ml = chance(rng, 0.12) if multiline is None else multiline
    if lang == 'es':
        (typ, name, cp, muni, prov) = L.street_es(rng)
        t = pick(rng, _TYPE_DISPLAY.get(typ, [typ]))
        if t.endswith(' de') and name.split(' ')[0] in ('de', 'del', 'la', 'las', 'los', 'el'):
            t = t[:-3]
        num = pick(rng, [str(int(rng.integers(1, 200)))] * 6 + [f'nº {int(rng.integers(1, 200))}', f'n.º {int(rng.integers(1, 120))}', f'núm. {int(rng.integers(1, 120))}', 's/n', f'{int(rng.integers(1, 90))} bis'])
        street = f'{t} {name}' + pick(rng, [', ', ' ', ', ']) + num
        if chance(rng, 0.45):
            street += pick(rng, [', ', ' ', ', ']) + pick(rng, [f"{int(rng.integers(1, 12))}º {pick(rng, 'ABCDEFGH')}", f"{int(rng.integers(1, 12))}º{pick(rng, 'ABCD')}", f'{int(rng.integers(1, 9))}º {int(rng.integers(1, 5))}ª', 'bajo', f'{int(rng.integers(1, 9))}º izq.', f'{int(rng.integers(1, 9))}º dcha.', f"Esc. {pick(rng, 'ABC')}, {int(rng.integers(1, 9))}º {pick(rng, 'ABCD')}", f'piso {int(rng.integers(1, 9))}, puerta {int(rng.integers(1, 6))}'])
        if chance(rng, 0.7):
            town = f'{cp} {muni}' + (f' ({prov})' if prov and prov != muni and chance(rng, 0.3) else '')
            return street + ('\n' if ml else pick(rng, [', ', ', ', ' - ', ' '])) + town
        return street
    us = chance(rng, 0.75)
    stem = pick(rng, _EN_STEMS) if chance(rng, 0.6) else L.surname(rng, 'en') if chance(rng, 0.6) else f"{int(rng.integers(1, 120))}{pick(rng, ['st', 'nd', 'rd', 'th'])}"
    sfx = pick(rng, _EN_SUFFIX)
    num = str(int(rng.integers(1, 9999)))
    pre = pick(rng, ['N ', 'S ', 'E ', 'W ', 'North ', ''] + [''] * 6)
    street = f'{num} {pre}{stem} {sfx}'
    if chance(rng, 0.3):
        street += pick(rng, [f", Apt {int(rng.integers(1, 40))}{pick(rng, 'ABCD')}", f' Apt. {int(rng.integers(1, 500))}', f', Suite {int(rng.integers(100, 900))}', f' #{int(rng.integers(1, 60))}', f', Unit {int(rng.integers(1, 30))}'])
    if chance(rng, 0.25):
        return street
    if us:
        (city, st) = L.place_us(rng)
        zp = rstr(rng, DIG, 5) + (f'-{rstr(rng, DIG, 4)}' if chance(rng, 0.15) else '')
        return street + ('\n' if ml else ', ') + f'{city}, {st} {zp}'
    pc = f"{rstr(rng, UP, pick(rng, [1, 2]))}{int(rng.integers(1, 30))} {int(rng.integers(1, 10))}{rstr(rng, 'ABDEFGHJLNPQRSTUWXYZ', 2)}"
    return street + ('\n' if ml else ', ') + f'{pick(rng, UK_TOWNS)} {pc}'

def address_negative(rng, lang):
    p = rng.random()
    if p < 0.25:
        return f"0x{rstr(rng, '0123456789abcdef', pick(rng, [8, 12, 16]))}"
    if p < 0.45:
        return pick(rng, ['la calle estaba vacía', 'en la plaza del pueblo', 'una avenida muy transitada', 'calle abajo', 'the street was empty', 'down the road', "on Main Street's corner"] if lang else ['-'])
    if p < 0.6:
        return f'{rng.uniform(-90, 90):.6f}, {rng.uniform(-180, 180):.6f}'
    if p < 0.75:
        return rstr(rng, DIG, 5)
    return pick(rng, ['Main Street', 'Wall Street', 'Silicon Valley', 'Gran Vía', 'Sesame Street', 'Abbey Road']) + pick(rng, [' Capital', ' Studios', ' Partners', ' Records', ' Labs', ''])

def secret_line(rng, positive, language=None):
    from secretsgen import credentials as C
    return C.line(rng, positive, language)

def secret_value(rng):
    from secretsgen import credentials as C
    from secretsgen.catalog import catalog_secret
    p = rng.random()
    if p < 0.35:
        v = catalog_secret(rng)
        if v:
            return v
    if p < 0.6:
        return C._sec_prefix(rng)
    if p < 0.85:
        return C._sec_generic(rng)
    return secret_password(rng)

def secret_negative(rng):
    p = rng.random()
    if p < 0.3:
        return pick(rng, ['changeme', 'CHANGE_ME', '<your-api-key>', 'YOUR_API_KEY_HERE', 'xxxxxxxx', '********', 'REDACTED', 'TODO', 'dummy', 'test', 'example', '${API_KEY}', '{{ secret }}', '%s', 'null', '', 'none', '<contraseña>', 'tu_clave_aqui'])
    if p < 0.55:
        return pick(rng, ["os.environ['API_KEY']", 'process.env.TOKEN', '${SECRET_KEY}', '$DB_PASSWORD', 'vault:secret/data/app', 'System.getenv("TOKEN")', 'settings.SECRET_KEY', '!secret db_password', '{{ .Values.password }}'])
    if p < 0.8:
        return rstr(rng, '0123456789abcdef', pick(rng, [32, 40, 64]))
    return '-'.join((rstr(rng, '0123456789abcdef', n) for n in (8, 4, 4, 4, 12)))
