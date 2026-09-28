"""The credential line generator of the secrets specialists (code and binaries).

`line(rng, positive)` returns one line of text and the (start, end) of the credential value in it (for a negative, a
line that looks like it could hold one and does not). The families of lines -- providers' real formats (sampled from the
gitleaks / trufflehog catalogues, validated against the original regexes), connection strings and URLs with passwords,
PEM blocks, cloud keys, generic high-entropy values under credential names, and the contrasting negatives (placeholders,
environment lookups, references, hashes, UUIDs, public keys, ordinary short values, documentation examples) -- are
weighted by module-level switches that the `secrets_synthetic` vertical sets from the task's `data:` block
(`plugin.py`, DATA_KEYS). With a switch at its default, not one random number is drawn for it: the data of earlier
recipes regenerate identically.

Held-out providers (PUREBYTE_HOLDOUT=comma,separated,prefixes): lines that would carry them are never generated, so an
evaluation can measure generalisation to formats the model never saw.

This module is generated from the maintainers' generator with comments removed; its behaviour is locked by
training/tests/test_secrets_data_golden.py. Change it only together with that golden file and a new recipe version.
"""
import base64
import string
import numpy as np
from .catalog import example_trufflehog, false_positive_curated, name_of_provider, word_key, catalog_secret
HEX = '0123456789abcdef'
HEXU = '0123456789ABCDEF'
B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
B64URL = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'
ALNUM = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
LOWER = 'abcdefghijklmnopqrstuvwxyz'

def _r(rng, alphabet, n):
    return ''.join((alphabet[i] for i in rng.integers(0, len(alphabet), n)))

def _uuid(rng):
    return '-'.join((_r(rng, HEX, n) for n in (8, 4, 4, 4, 12)))
import os as _os
_H = _os.environ.get('PUREBYTE_HOLDOUT', '').strip()
HOLDOUT = {x.strip() for x in _H.split(',') if x.strip()}
_LABELS = {4: 'glpat', 6: 'rk_live', 7: 'whsec', 13: 'sendgrid', 16: 'npm', 17: 'digitalocean', 18: 'shopify', 19: 'vault', 20: 'docker', 21: 'pypi'}
_HELD_OUT = {i for (i, t) in _LABELS.items() if t in HOLDOUT}
LITERAL_HOLDABLE = {'glpat': 'glpat-', 'rk_live': 'rk_live_', 'whsec': 'whsec_', 'sendgrid': 'SG.', 'npm': 'npm_', 'digitalocean': 'dop_v1_', 'shopify': 'shpat_', 'vault': 'hvs.', 'docker': 'dckr_pat_', 'pypi': 'pypi-AgEI'}
_unknown = HOLDOUT - set(LITERAL_HOLDABLE)
if _unknown:
    raise ValueError(f'PUREBYTE_HOLDOUT asks to hold out {sorted(_unknown)}, whose prefix is unknown. Known: {sorted(LITERAL_HOLDABLE)}')
_LITERALS = tuple((LITERAL_HOLDABLE[t] for t in sorted(HOLDOUT)))

def contains_held_out(text: str) -> bool:
    return any((l in text for l in _LITERALS))
INVENTED_PREFIXES = 0.0

def _prefix_invented(rng):
    while True:
        root = _r(rng, LOWER, int(rng.integers(2, 6)))
        if not root.startswith('pk'):
            break
    sep = str(rng.choice(['_', '-', '.', '_', '_']))
    t = int(rng.integers(0, 4))
    if t == 0:
        pre = root + sep
    elif t == 1:
        pre = root + sep + str(rng.choice(['live', 'prod', 'secret', 'api', 'key', 'tok', 'v1', 'pat'])) + sep
    elif t == 2:
        pre = root.upper()[:4]
    else:
        pre = root + str(rng.choice(['pat', 'sec', 'key'])) + sep
    alphabet = ALNUM if rng.integers(0, 3) else LOWER + '0123456789'
    return pre + _r(rng, alphabet, int(rng.integers(24, 56)))

def _sec_prefix(rng):
    if INVENTED_PREFIXES and rng.random() < INVENTED_PREFIXES:
        return _prefix_invented(rng)
    p = int(rng.integers(0, 22))
    if _HELD_OUT:
        for _ in range(40):
            if p not in _HELD_OUT:
                break
            p = int(rng.integers(0, 22))
        else:
            return _r(rng, ALNUM, 32)
    return prefix_by_index(rng, p)
N_PREFIXES = 22

def prefix_by_index(rng, p):
    if p == 0:
        return 'ghp_' + _r(rng, ALNUM, 36)
    if p == 1:
        return 'gho_' + _r(rng, ALNUM, 36)
    if p == 2:
        return 'ghs_' + _r(rng, ALNUM, 36)
    if p == 3:
        return 'github_pat_' + _r(rng, ALNUM, 22) + '_' + _r(rng, ALNUM, 59)
    if p == 4:
        return 'glpat-' + _r(rng, ALNUM + '-_', 20)
    if p == 5:
        return 'sk_live_' + _r(rng, ALNUM, int(rng.integers(24, 56)))
    if p == 6:
        return 'rk_live_' + _r(rng, ALNUM, 24)
    if p == 7:
        return 'whsec_' + _r(rng, ALNUM, 32)
    if p == 8:
        return 'AKIA' + _r(rng, HEXU + 'GHIJKLMNOPQRSTUVWXYZ', 16)
    if p == 9:
        return 'ASIA' + _r(rng, HEXU + 'GHIJKLMNOPQRSTUVWXYZ', 16)
    if p == 10:
        return 'AIza' + _r(rng, ALNUM + '-_', 35)
    if p == 11:
        return 'xoxb-' + '-'.join((_r(rng, '0123456789', n) for n in (10, 11))) + '-' + _r(rng, ALNUM, 24)
    if p == 12:
        return 'xoxp-' + '-'.join((_r(rng, '0123456789', n) for n in (11, 11))) + '-' + _r(rng, ALNUM, 32)
    if p == 13:
        return 'SG.' + _r(rng, ALNUM + '-_', 22) + '.' + _r(rng, ALNUM + '-_', 43)
    if p == 14:
        return 'sk-proj-' + _r(rng, ALNUM, int(rng.integers(40, 64)))
    if p == 15:
        return 'sk-ant-api03-' + _r(rng, ALNUM + '-_', 95)
    if p == 16:
        return 'npm_' + _r(rng, ALNUM, 36)
    if p == 17:
        return 'dop_v1_' + _r(rng, LOWER + '0123456789', 64)
    if p == 18:
        return 'shpat_' + _r(rng, LOWER + '0123456789', 32)
    if p == 19:
        return 'hvs.' + _r(rng, ALNUM + '-_', 48)
    if p == 20:
        return 'dckr_pat_' + _r(rng, ALNUM + '-_', 36)
    return 'pypi-AgEIcHlwaS5vcmc' + _r(rng, ALNUM + '-_', 140)

def _sec_prefix_rare(rng):
    p = int(rng.integers(0, 23))
    if p == 0:
        return 'hf_' + _r(rng, ALNUM, 34)
    if p == 1:
        return 'fm2_' + _r(rng, ALNUM, 54)
    if p == 2:
        return 'eu01xx' + _r(rng, LOWER + '0123456789', 34) + 'NRAL'
    if p == 3:
        return 'cmVmdGtu:01:2026:' + _r(rng, LOWER + '0123456789', 36)
    if p == 4:
        return 'sv=2023-11-03&ss=b&srt=co&sp=rwdlac&se=2027-01-01T00:00:00Z&sig=' + _r(rng, ALNUM, 40) + '%3D'
    if p == 5:
        return 'v1.0-' + _r(rng, ALNUM, 24) + '-' + _r(rng, ALNUM, 44)
    if p == 6:
        return 'u+' + _r(rng, ALNUM, 20)
    if p == 7:
        return '00' + _r(rng, ALNUM + '-_', 40)
    if p == 8:
        return 'pmk-' + _r(rng, HEX, 32)
    if p == 9:
        return 'sq0atp-' + _r(rng, ALNUM + '-_', 22)
    if p == 10:
        return 'EAACEdEose0cBA' + _r(rng, ALNUM, 40)
    if p == 11:
        return 'AC' + _r(rng, HEX, 32)
    if p == 12:
        return 'key-' + _r(rng, HEX, 32)
    if p == 13:
        return 'rk_prod_' + _r(rng, ALNUM, 32)
    if p == 14:
        return 'atlasv1.' + _r(rng, ALNUM, 60)
    if p == 15:
        return 'lin_api_' + _r(rng, ALNUM, 40)
    if p == 16:
        return 'secret_' + _r(rng, ALNUM, 43)
    if p == 17:
        return 'phc_' + _r(rng, ALNUM, 43)
    if p == 18:
        return 'sntrys_' + _r(rng, B64, 60)
    if p == 19:
        return 'figd_' + _r(rng, ALNUM + '-_', 40)
    if p == 20:
        return 'glsa_' + _r(rng, ALNUM, 32) + '_' + _r(rng, HEX, 8)
    if p == 21:
        return 'nfp_' + _r(rng, ALNUM, 36)
    if p == 22:
        return 'https://hooks.slack.com/services/T' + _r(rng, ALNUM.upper(), 8) + '/B' + _r(rng, ALNUM.upper(), 10) + '/' + _r(rng, ALNUM, 24)
    return 'doo_v1_' + _r(rng, HEX, 64)

def _sec_obfuscated(rng):
    real = _sec_prefix(rng) if rng.integers(0, 2) else _sec_generic(rng)
    p = int(rng.integers(0, 4))
    if p == 0:
        return ''.join(('\\x%02x' % ord(c) for c in real[:24]))
    if p == 1:
        return ''.join(('%02x' % ord(c) for c in real[:28]))
    if p == 2:
        import base64
        return base64.b64encode(real.encode()).decode()
    chunks = [real[i:i + 8] for i in range(0, min(len(real), 32), 8)]
    q = "'" if rng.integers(0, 2) else '"'
    return (q + (q + ' + ' + q).join(chunks) + q)[1:-1]

def _sec_generic(rng):
    p = int(rng.integers(0, 8))
    if p == 0:
        return _r(rng, HEX, int(rng.choice([32, 40, 48, 64])))
    if p == 1:
        return _r(rng, B64, int(rng.integers(24, 64)))
    if p == 2:
        return _r(rng, ALNUM, int(rng.integers(20, 48)))
    if p == 3:
        return _r(rng, B64URL, int(rng.integers(32, 56)))
    if p == 4:
        return 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.' + _r(rng, B64URL, 40) + '.' + _r(rng, B64URL, 43)
    if p == 5:
        return _r(rng, HEX, 32) + '-' + _r(rng, HEX, 16)
    if p == 6:
        return _r(rng, ALNUM + '!@#$%^&*()_+-=', int(rng.integers(16, 32)))
    return _r(rng, HEXU, int(rng.choice([32, 40, 64])))
SHORT_PASSWORDS = 0.0
_WORD_KEY = ['summer', 'winter', 'spring', 'dragon', 'monkey', 'shadow', 'master', 'welcome', 'orders', 'admin', 'super', 'secret', 'pass', 'madrid', 'london', 'tiger', 'falcon', 'coffee', 'prod', 'postgres', 'redis', 'mysql', 'backend', 'deploy', 'staging', 'acme', 'rocket', 'banana']
_LEET = {'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7'}
_WORD_ORDINARY = ['bearer', 'basic', 'digest', 'oauth2', 'password', 'client_credentials', 'authorization_code', 'refresh_token', 'access_token', 'id_token', 'api_key', 'x-api-key', 'Authorization', 'X-Auth-Token', 'secret', 'token', 'apikey', 'credentials', 'username', 'hidden', 'text', 'required', 'optional', 'string', 'utf8mb4', 'sha256', 'HS256', 'RS256', 'bcrypt', 'argon2id', 'pbkdf2_sha256', 'plain', 'none', 'default', 'header', 'cookie', 'query', 'session', 'keyring', 'env', 'file', 'vault', 'prompt', 'stdin', 'disabled', 'enabled', 'auto', 'local', 'ldap']

def _password_short(rng):
    t = int(rng.integers(0, 3))
    if t == 0:
        n = int(rng.integers(8, 17))
        while True:
            v = _r(rng, ALNUM, n)
            if any((c.isdigit() for c in v)) and any((c.isupper() for c in v)) and any((c.islower() for c in v)):
                return v
    if t == 1:
        n = int(rng.integers(8, 16))
        v = _r(rng, ALNUM, n)
        k = int(rng.integers(1, n))
        return v[:k] + str(rng.choice(list('!#%&*-_.$@+'))) + v[k:]
    w = _WORD_KEY[rng.integers(0, len(_WORD_KEY))]
    if rng.integers(0, 2):
        w = ''.join((_LEET.get(c, c) if rng.random() < 0.5 else c for c in w))
    w = w.capitalize() if rng.integers(0, 3) else w
    if rng.integers(0, 3) == 0:
        w += _WORD_KEY[rng.integers(0, len(_WORD_KEY))].capitalize()
    w += str(rng.choice(['', '_', '-', '.'])) + str(rng.integers(1, 10000) if rng.integers(0, 2) else rng.integers(1990, 2031))
    return w + str(rng.choice(['', '!', '!!', '#', '$', '*', '?']))
ORDINARY_SHORT_VALUES = False
_WORD_WEB = ['foo', 'bar', 'baz', 'quux', 'name', 'value', 'id', 'page', 'lang', 'sort', 'q', 'user', 'item', 'theme', 'mode', 'size', 'color', 'type', 'ref', 'tab', 'view', 'limit', 'offset', 'zoo', 'abc']
_NM_PROTOCOL = ['HTTP_COOKIE', 'Cookie', 'Set-Cookie', 'HTTP_FORWARDED', 'Forwarded', 'X-Forwarded-Host', 'X-Forwarded-For', 'Strict-Transport-Security', 'Cache-Control', 'Content-Type', 'Accept', 'Content-Disposition', 'query', 'query_string', 'input', 'body', 'form', 'email', 'login', 'username', 'Host', 'Origin', 'Referer', 'Accept-Language', 'If-None-Match', 'Range']

def _short_ordinary(rng):
    w = lambda : _WORD_WEB[rng.integers(0, len(_WORD_WEB))]
    n = lambda a, b: str(rng.integers(a, b))
    t = int(rng.integers(0, 10))
    if t == 0:
        v = '; '.join((f'{w()}={w()}' for _ in range(int(rng.integers(1, 4)))))
    elif t == 1:
        v = f"host={_r(rng, LOWER, 6)}.{rng.choice(['org', 'com', 'net', 'io'])}" + (f':{n(80, 9999)}' if rng.integers(0, 2) else '')
    elif t == 2:
        v = f'max-age={n(60, 99999999)}' + str(rng.choice(['', '; preload', '; includeSubDomains', ', public', ', must-revalidate']))
    elif t == 3:
        v = '?' + '&'.join((f'{w()}={(n(1, 999) if rng.integers(0, 2) else w())}' for _ in range(int(rng.integers(1, 4)))))
    elif t == 4:
        v = f"{_r(rng, LOWER, 5)}{rng.choice(['.', '_', ''])}{_r(rng, LOWER, 5)}{(n(1, 99) if rng.integers(0, 2) else '')}@{_r(rng, LOWER, 7)}.{rng.choice(['com', 'org', 'io'])}"
    elif t == 5:
        v = str(rng.choice(['application/json', 'text/html; charset=utf-8', 'multipart/form-data', 'image/png', 'application/x-www-form-urlencoded', 'text/plain', 'application/octet-stream']))
    elif t == 6:
        v = f'{n(2015, 2031)}-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 29)):02d}' + (f'T{int(rng.integers(0, 24)):02d}:{int(rng.integers(0, 60)):02d}:00Z' if rng.integers(0, 2) else '')
    elif t == 7:
        v = f"{rng.choice(['v', '', '^', '~', '>='])}{n(0, 30)}.{n(0, 60)}.{n(0, 99)}" + str(rng.choice(['', '-beta.1', '-rc2', '+build5']))
    elif t == 8:
        v = '/' + '/'.join((_r(rng, LOWER, int(rng.integers(3, 9))) for _ in range(int(rng.integers(1, 4))))) + str(rng.choice(['', '.html', '.json', '/']))
    else:
        v = f"{rng.choice(['bytes=0-', 'en-US,en;q=0.', 'attachment; filename=f', 'W/', 'form-data; name='])}{n(1, 9999)}"
    name = _NM_PROTOCOL[rng.integers(0, len(_NM_PROTOCOL))] if rng.integers(0, 3) else _credential_name_not_password(rng) if PASSWORD_NAMED_SHORT else credential_name(rng)
    return (name, v)

def _word_ordinary(rng):
    w = _WORD_ORDINARY[rng.integers(0, len(_WORD_ORDINARY))]
    return w.upper() if rng.integers(0, 6) == 0 else w
SHARED_HOSTS = False

def _host_any(rng):
    t = int(rng.integers(0, 8))
    if t == 0:
        return _r(rng, LOWER, int(rng.integers(4, 10))) + '.example.com'
    if t == 1:
        return str(rng.choice(['localhost', '127.0.0.1', '[::1]', 'host.docker.internal']))
    if t == 2:
        return str(rng.choice(['db', 'postgres', 'mysql', 'redis', 'mongo', 'rabbitmq', 'cache', 'broker', 'api', 'registry', 'minio', 'kafka']))
    if t == 3:
        return f'10.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}' if rng.integers(0, 2) else f'192.168.{rng.integers(0, 256)}.{rng.integers(1, 255)}'
    if t == 4:
        return str(rng.choice(['db', 'pg', 'sql', 'cache', 'mq', 'svc', 'api', 'cdn'])) + str(rng.choice(['', '-1', '-prod', '-staging', '01'])) + '.' + _r(rng, LOWER, int(rng.integers(4, 9))) + str(rng.choice(['.internal', '.corp', '.local', '.io', '.net', '.cloud', '.dev', '.com']))
    if t == 5:
        return f"{_r(rng, LOWER, 6)}.{_r(rng, LOWER + '0123456789', 12)}.{rng.choice(['us-east-1', 'eu-west-1', 'ap-south-1'])}.rds.amazonaws.com"
    if t == 6:
        return str(rng.choice([f'cluster0.{_r(rng, LOWER, 5)}.mongodb.net', f'{_r(rng, LOWER, 8)}.redis.cache.windows.net', f'{_r(rng, LOWER, 8)}.postgres.database.azure.com', f"s3.{rng.choice(['eu-west-1', 'us-east-2'])}.amazonaws.com", f'{_r(rng, LOWER, 6)}.supabase.co', f'{_r(rng, LOWER, 7)}.herokuapp.com']))
    return _r(rng, LOWER, int(rng.integers(3, 8))) + '.' + _r(rng, LOWER, int(rng.integers(4, 9))) + str(rng.choice(['.com', '.org', '.es', '.io', '.net']))

def _sec_connection(rng):
    u = _r(rng, LOWER, int(rng.integers(3, 9)))
    pw = _r(rng, ALNUM, int(rng.integers(12, 28)))
    h = _r(rng, LOWER, int(rng.integers(4, 10))) + '.example.com'
    if SHARED_HOSTS:
        h = _host_any(rng)
    port = int(rng.integers(1000, 9999))
    base = _r(rng, LOWER, 6)
    t = int(rng.integers(0, 8))
    if t <= 2:
        e = str(rng.choice(['postgres', 'postgresql', 'mysql', 'mongodb', 'mongodb+srv', 'redis', 'rediss', 'amqp', 'amqps', 'ftp', 'sftp', 'clickhouse', 'mssql', 'smtp', 'smtps', 'https', 'ldap', 'kafka', 'nats', 'elasticsearch', 'cassandra']))
        tail = str(rng.choice(['', '', f'?ssl=true', '?retryWrites=true&w=majority', '?sslmode=require', '?charset=utf8mb4', '?authSource=admin']))
        sinp = rng.integers(0, 4) == 0
        return f'{e}://{u}:{pw}@{h}' + ('' if sinp else f':{port}') + f'/{base}{tail}'
    if t == 3:
        return f'redis://:{pw}@{h}:{port}/0'
    if t == 4:
        e = str(rng.choice(['postgresql', 'mysql', 'sqlserver', 'oracle:thin', 'mariadb']))
        return f'jdbc:{e}://{h}:{port}/{base}?user={u}&password={pw}'
    if t == 5:
        cl = str(rng.choice(['Password', 'Pwd', 'User Password']))
        return f'Server={h},{port};Database={base};User Id={u};{cl}={pw};TrustServerCertificate=True'
    if t == 6:
        key = base64.b64encode(_r(rng, ALNUM, 48).encode()).decode()
        return f'DefaultEndpointsProtocol=https;AccountName={u};AccountKey={key};EndpointSuffix=core.windows.net'
    if rng.integers(0, 2):
        return f'host={h} port={port} dbname={base} user={u} password={pw}'
    return f'Data Source={h};Initial Catalog={base};UID={u};PWD={pw}'

def _sec_pem(rng):
    kind = ['RSA PRIVATE KEY', 'PRIVATE KEY', 'EC PRIVATE KEY', 'OPENSSH PRIVATE KEY', 'DSA PRIVATE KEY', 'ENCRYPTED PRIVATE KEY'][rng.integers(0, 6)]
    return f'-----BEGIN {kind}-----\\n' + _r(rng, B64, 64) + f'\\n-----END {kind}-----'

def secret_value(rng):
    p = rng.random()
    if p < 0.38:
        return _sec_prefix(rng)
    if p < 0.82:
        return _sec_generic(rng)
    if p < 0.94:
        return _sec_connection(rng)
    return _sec_pem(rng)

def _ben_hash(rng):
    p = int(rng.integers(0, 9))
    if p == 0:
        return _r(rng, HEX, 40)
    if p == 1:
        return _r(rng, HEX, 7)
    if p == 2:
        return _r(rng, HEX, 64)
    if p == 3:
        return _r(rng, HEX, 32)
    if p == 4:
        return _r(rng, HEX, 128)
    if p == 5:
        return 'sha256:' + _r(rng, HEX, 64)
    if p == 6:
        return 'sha384-' + _r(rng, B64, 64)
    if p == 7:
        return 'sha512-' + _r(rng, B64, 88)
    return _r(rng, HEX, 40) + '-' + _r(rng, HEX, 8)
BINARY_SYMBOLS = False

def _symbol_mangled(rng):

    def chunk():
        w = _r(rng, LOWER, int(rng.integers(3, 12)))
        return f'{len(w)}{w}'
    n = int(rng.integers(2, 7))
    t = int(rng.integers(0, 4))
    if t == 0:
        return '_ZN' + ''.join((chunk() for _ in range(n))) + 'E' + _r(rng, 'vijlcPKRS_', int(rng.integers(0, 6)))
    if t == 1:
        return '__pyx_' + str(rng.choice(['pf', 'pw', 'f', 'k', 'n_s', 'tp_new'])) + '_' + '_'.join((chunk() for _ in range(n)))
    if t == 2:
        return '_ZTVN' + ''.join((chunk() for _ in range(n))) + 'E'
    return ''.join((chunk() for _ in range(n))) + '_' + _r(rng, HEX, 8)

def _table_of_pairs(rng):
    c = str(rng.choice(list('KDpmrZ^d(')))
    ini = int(rng.integers(33, 80))
    n = int(rng.integers(12, 40))
    stride = int(rng.integers(1, 3))
    return ''.join((c + chr(ini + i * stride) if rng.integers(0, 2) else chr(ini + i * stride) + c for i in range(n) if ini + i * stride < 126))

def _ben_id(rng):
    if BINARY_SYMBOLS and rng.random() < 0.4:
        return _symbol_mangled(rng) if rng.integers(0, 3) else _table_of_pairs(rng)
    p = int(rng.integers(0, 8))
    if p == 0:
        return _uuid(rng)
    if p == 1:
        return _r(rng, '0123456789ABCDEFGHJKMNPQRSTVWXYZ', 26)
    if p == 2:
        return _r(rng, ALNUM + '-_', 21)
    if p == 3:
        return _r(rng, '0123456789', 19)
    if p == 4:
        return _r(rng, HEX, 32)
    if p == 5:
        return _r(rng, HEX, 16)
    if p == 6:
        return f'{_r(rng, LOWER, 3)}_{_r(rng, ALNUM, 24)}'
    return f"{int(rng.integers(2020, 2027))}{_r(rng, '0123456789', 10)}_{_r(rng, HEX, 16)}"

def _ben_public(rng):
    p = int(rng.integers(0, 13))
    if p == 0:
        return 'pk_live_' + _r(rng, ALNUM, int(rng.integers(24, 56)))
    if p == 1:
        return 'pk_test_' + _r(rng, ALNUM, 32)
    if p == 2:
        return 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI' + _r(rng, B64, 43) + ' ' + _r(rng, LOWER, 6) + '@host'
    if p == 3:
        return 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQ' + _r(rng, B64, 120) + ' deploy@ci'
    if p == 4:
        return 'SHA256:' + _r(rng, B64, 43)
    if p == 5:
        return ':'.join((_r(rng, HEXU, 2) for _ in range(16)))
    if p == 6:
        return '-----BEGIN PUBLIC KEY-----\\n' + _r(rng, B64, 64) + '\\n-----END PUBLIC KEY-----'
    if p == 7:
        return '-----BEGIN CERTIFICATE-----\\n' + _r(rng, B64, 64) + '\\n-----END CERTIFICATE-----'
    if p == 8:
        return 'did:key:z6Mk' + _r(rng, ALNUM, 44)
    if p == 9:
        return '0x' + _r(rng, HEX, 40)
    if p == 10:
        B58 = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
        return str(rng.choice(['1', '3', 'bc1q'])) + _r(rng, B58, int(rng.integers(25, 39)))
    if p == 11:
        return _r(rng, '0123456789ABCDEFGHJKMNPQRSTVWXYZabcdefghijkmnpqrstvwxyz', 27)
    # the header and payload of jwt.io's public example, as two literals so that secret scanners do not flag this file
    return 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9' '.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0.' + _r(rng, B64URL, 43)
PAIRS = [('FIREBASE_API_KEY', lambda r: 'AIza' + _r(r, ALNUM + '-_', 35), 'FIREBASE_PRIVATE_KEY', lambda r: _sec_pem(r)), ('SUPABASE_ANON_KEY', lambda r: _jwt_rol(r, 'anon'), 'SUPABASE_SERVICE_ROLE', lambda r: _jwt_rol(r, 'service_role')), ('ALGOLIA_APP_ID', lambda r: _r(r, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', 10), 'ALGOLIA_ADMIN_KEY', lambda r: _r(r, HEX, 32)), ('MAILGUN_PUBLIC_KEY', lambda r: 'pubkey-' + _r(r, HEX, 32), 'MAILGUN_API_KEY', lambda r: 'key-' + _r(r, HEX, 32)), ('STRIPE_PUBLISHABLE', lambda r: 'pk_live_' + _r(r, ALNUM, 48), 'STRIPE_SECRET', lambda r: 'sk_live_' + _r(r, ALNUM, 48)), ('SENTRY_DSN_PUBLIC', lambda r: 'https://' + _r(r, HEX, 32) + '@o1.ingest.sentry.io/1', 'SENTRY_AUTH_TOKEN', lambda r: 'sntrys_' + _r(r, B64, 60)), ('VAPID_PUBLIC_KEY', lambda r: 'B' + _r(r, B64URL, 86), 'VAPID_PRIVATE_KEY', lambda r: _r(r, B64URL, 43))]

def _jwt_rol(rng, rol):
    import base64 as _b
    cab = _b.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').decode().rstrip('=')
    pay = _b.urlsafe_b64encode(('{"iss":"supabase","role":"%s","iat":1737000000}' % rol).encode()).decode().rstrip('=')
    return cab + '.' + pay + '.' + _r(rng, B64URL, 43)

def _url_public(rng):
    host = str(rng.choice(['host.com', 'api.example.com', 'localhost', '127.0.0.1', 'svc.interno', 'cdn.example.org', '10.0.0.7', '[::1]', 'registry.example.com', 's3.eu-west-1.amazonaws.com']))
    if SHARED_HOSTS:
        host = _host_any(rng)
    corner = str(rng.choice(['http', 'https', 'http', 'https', 'ws', 'grpc']))
    if SHARED_HOSTS and rng.integers(0, 2):
        corner = str(rng.choice(['postgres', 'postgresql', 'mysql', 'mongodb', 'redis', 'amqp', 'sftp', 'smtp', 'ldap', 'kafka', 'nats']))
    port = '' if rng.integers(0, 2) else ':' + str(rng.choice(['80', '443', '8080', '3000', '5432', '65536', '0', '-1', 'port']))
    path = str(rng.choice(['/', '/path', '/v1/health', '/api/v2/items', '', '/.well-known/openid-configuration', '/static/app.js']))
    tail = str(rng.choice(['', '', '', '?page=2', '?limit=10&offset=0', '#seccion', '?format=json', '?region=eu-west-1']))
    return f'{corner}://{host}{port}{path}{tail}'
_DIG = '0123456789'
_LOWER = 'abcdefghijklmnopqrstuvwxyz'
_UPPER = _LOWER.upper()
_SYMB = ['+/', '+/=', '-_', '._-', '!#$%&', '', '', '']

def _table_of_chars(rng):
    t = int(rng.integers(0, 9))
    if t == 0:
        h = _DIG + (_LOWER if rng.integers(0, 2) else _UPPER)[:6]
        return h if rng.integers(0, 3) else '0x' + h
    if t == 1:
        if rng.integers(0, 2):
            return _UPPER + '234567'
        return _DIG + ''.join((c for c in _UPPER if c not in 'ILOU'))
    if t == 2:
        return '123456789' + ''.join((c for c in _UPPER if c not in 'OI')) + ''.join((c for c in _LOWER if c not in 'l'))
    if t == 3:
        rows = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
        if rng.integers(0, 2):
            rows = [f.upper() for f in rows]
        return ''.join(rows[:int(rng.integers(2, 4))])
    if t == 4:
        ini_ = int(rng.integers(32, 74))
        length = int(rng.integers(20, max(21, 127 - ini_)))
        return ''.join((chr(c) for c in range(ini_, min(127, ini_ + length))))
    if t == 5:
        remove = str(rng.choice(['0O1lI', '01lI', 'OIl', '0Oo']))
        parts = [_DIG, _UPPER, _LOWER][:int(rng.integers(2, 4))]
        return ''.join((c for c in ''.join(parts) if c not in remove))
    if t == 6:
        return str(rng.choice(['aeiouAEIOUaeiouAEIOU', 'bcdfghjklmnpqrstvwxyz', 'IVXLCDMivxlcdmIVXLCDM', '0123456789ABCDEF', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'EneFebMarAbrMayJunJulAgoSepOctNovDic', 'JanFebMarAprMayJunJulAugSepOctNovDec', 'LunMarMieJueVieSabDom']))
    if t == 7:
        sep = str(rng.choice(['', '', ' ', ',', '|', '-']))
        return sep.join(_DIG + _LOWER)
    parts = [_DIG, _LOWER, _UPPER]
    if rng.integers(0, 2):
        parts = [_DIG, _UPPER, _LOWER]
    body = str(rng.choice(_SYMB)).join(parts) + str(rng.choice(_SYMB))
    if rng.integers(0, 4) == 0:
        body = body[:int(rng.integers(16, len(body) + 1))]
    return body
LONG_HEX = False

def _hex_length(rng):
    n = int(rng.integers(96, 300))
    alf = '0123456789ABCDEF' if rng.integers(0, 3) else '0123456789abcdef'
    v = _r(rng, alf, n)
    t = int(rng.integers(0, 4))
    f = 'F' * 16 if alf[-1] == 'F' else 'f' * 16
    if t == 0:
        v = f + v[16:-16] + f
    elif t == 1:
        v = v[:-16] + f
    elif t == 2:
        v = ' '.join((v[i:i + 8] for i in range(0, len(v) - len(v) % 8, 8)))
    return v[:300]

def _ben_material_crypto(rng):
    if LONG_HEX and rng.random() < 0.4:
        return _hex_length(rng)
    p = int(rng.integers(0, 6))
    if p == 0:
        return _r(rng, HEX, 32)
    if p == 1:
        return _r(rng, HEX, 24)
    if p == 2:
        return _r(rng, B64, 16)
    if p == 3:
        return '000102030405060708090a0b0c0d0e0f'
    if p == 4:
        return '2b7e151628aed2a6abf7158809cf4f3c'
    return _r(rng, HEX, 16)
_MARK_ASK = ['YOUR', 'MY', 'INSERT', 'REPLACE', 'PUT', 'ADD', 'SET', 'TU', 'THE']
_MARK_THING = ['KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'API_KEY', 'CREDENTIAL', 'VALUE', 'PASS', 'CLIENT_SECRET', 'ACCESS_KEY', 'APIKEY', 'AUTH']
_MARK_HERE = ['HERE', 'GOES_HERE', '', '_VALUE', '_PLEASE', '_NOW']
_MARK_STICK = ['changeme', 'change-me', 'change_me', 'CHANGEME', 'replaceme', 'replace-me', 'todo', 'TODO', 'fixme', 'FIXME', 'tbd', 'TBD', 'n/a', 'N/A', 'none', 'unset', 'undefined', 'pending', 'wip', 'xxx', 'XXX', 'dummy', 'fake', 'sample', 'example', 'placeholder', 'test', 'testing', 'demo', 'foo', 'bar', 'baz', 'qux', 'lorem', 'asdf', 'qwerty', 'secret', 'supersecret', 'mysecret', 'password', 'passwd', 'hunter2', '123456', 'letmein', 'admin', 'root', 'guest', 'abc123', '1234']
_MARK_ADJ = ['not-a-real', 'not_a_real', 'no-es-real', 'invalid', 'bogus', 'fake', 'dummy', 'sample', 'example', 'temp', 'tmp', 'local', 'dev', 'test']
DOCUMENTED_EXAMPLES = False
_MAYDIG = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'

def _example_documented(rng):
    t = int(rng.integers(0, 4))
    if t == 0:
        while True:
            v = 'AKIA' + _r(rng, _MAYDIG, 9) + 'EXAMPLE'
            if v != 'AKIAIOSFODNN7EXAMPLE':
                return v
    if t == 1:
        return _r(rng, string.ascii_letters + string.digits + '/+', 30) + 'EXAMPLEKEY'
    v = _sec_prefix(rng)
    mark = str(rng.choice(['EXAMPLE', 'EXAMPLEKEY', '_EXAMPLE', '-example', 'Example']))
    return v[:max(8, len(v) - len(mark))] + mark

def _placeholder(rng):
    if DOCUMENTED_EXAMPLES and rng.random() < 0.3:
        return _example_documented(rng)
    t = int(rng.integers(0, 7))
    thing = str(rng.choice(_MARK_THING))
    if t == 0:
        here = str(rng.choice(_MARK_HERE))
        v = f'{rng.choice(_MARK_ASK)}_{thing}' + (f'_{here}' if here and (not here.startswith('_')) else here)
        return v if rng.integers(0, 2) else v.replace('_', '-')
    if t == 1:
        b = thing.lower().replace('_', '-')
        env = int(rng.integers(0, 4))
        return f'<{b}>' if env == 0 else f'<YOUR_{thing}>' if env == 1 else f'[{b}]' if env == 2 else f'{{{{{b}}}}}'
    if t == 2:
        return str(rng.choice(_MARK_STICK))
    if t == 3:
        sep = str(rng.choice(['-', '_']))
        return f"{rng.choice(_MARK_ADJ)}{sep}{thing.lower().replace('_', sep)}"
    if t == 4:
        c = str(rng.choice(['x', 'X', '0', 'a', 'A', '*', '.', '?', '1']))
        return c * int(rng.integers(6, 41))
    if t == 5:
        base = str(rng.choice(['deadbeef', 'cafebabe', '0123456789abcdef', 'baadf00d', 'feedface', '8badf00d']))
        return (base * 6)[:int(rng.integers(8, 65))]
    pref = str(rng.choice(['sk_test_', 'pk_test_', 'test_', 'sandbox_', 'ghp_test', 'AKIAEXAMPLE', 'dev_', 'fake_']))
    tail = str(rng.choice(['ExampleKeyDoNotUse', '0' * 24, 'x' * 24, 'REPLACE_ME', 'YOUR_KEY_HERE', 'notarealkey', '1234567890abcdef']))
    return pref + tail

def _ben_redacted(rng):
    pre = str(rng.choice(['sk_live_', 'sk_test_', 'ghp_', 'github_pat_', 'glpat-', 'AKIA', 'xoxb-', 'AIza', 'hf_', 'npm_', 'SG.', 'sk-proj-', 'shpat_', 'dop_v1_', 'pk_live_']))
    n = int(rng.integers(16, 40))
    cover = int(rng.integers(0, 7))
    if cover == 0:
        return pre + '*' * n + 'REDACTED'
    if cover == 1:
        return pre + 'X' * n
    if cover == 2:
        return pre + 'x' * n
    if cover == 3:
        return pre + '.' * n
    if cover == 4:
        return pre + '<' + _r(rng, LOWER + '_', 12) + '>'
    if cover == 5:
        visible = _r(rng, ALNUM, int(rng.integers(4, 12)))
        hidden = str(rng.choice(['*', 'x', 'X', '•'])) * int(rng.integers(4, 24))
        sep = str(rng.choice(['-', '', '_']))
        return pre + visible + sep + hidden + (sep + hidden if rng.integers(0, 2) else '')
    return pre + '0' * n
BODYLESS_PEM = False

def _pem_quoted(rng):
    kind = str(rng.choice(['RSA PRIVATE KEY', 'PRIVATE KEY', 'EC PRIVATE KEY', 'OPENSSH PRIVATE KEY', 'DSA PRIVATE KEY', 'ENCRYPTED PRIVATE KEY', 'PGP PRIVATE KEY BLOCK']))
    (cab, pie) = (f'-----BEGIN {kind}-----', f'-----END {kind}-----')
    t = int(rng.integers(0, 6))
    if t == 0:
        return f'expected {cab}'
    if t == 1:
        return f'^{cab}$'
    if t == 2:
        return f'^{cab}(.|\\n)*?{pie}$'
    if t == 3:
        return f'falta la cabecera `{cab}`'
    if t == 4:
        return cab[int(rng.integers(1, 6)):]
    return f'no empieza por {cab}'

def _ben_pattern(rng):
    if BODYLESS_PEM and rng.random() < 0.2:
        return _pem_quoted(rng)
    pre = str(rng.choice(['sk_live_', 'ghp_', 'AKIA', 'xoxb-', 'glpat-', 'AIza', 'hf_', 'npm_', 'sk-ant-api03-']))
    klass = str(rng.choice(['[A-Za-z0-9]', '[0-9a-f]', '\\w', '[A-Za-z0-9_-]', '[a-z0-9]']))
    quant = str(rng.choice(['{24}', '{32,40}', '{16,}', '+', '{36}']))
    body = pre + klass + quant
    form = int(rng.integers(0, 4))
    if form == 0:
        return body
    if form == 1:
        return '^' + body + '$'
    if form == 2:
        return '(?i)' + body
    return '\\b' + body + '\\b'

def _ben_hash_of_pass(rng):
    p = int(rng.integers(0, 10))
    if p == 0:
        return '$2b$12$' + _r(rng, ALNUM + './', 53)
    if p == 1:
        return '$2y$10$' + _r(rng, ALNUM + './', 53)
    if p == 2:
        return '$argon2id$v=19$m=65536,t=3,p=4$' + _r(rng, B64, 22) + '$' + _r(rng, B64, 43)
    if p == 3:
        return '$6$' + _r(rng, ALNUM + './', 16) + '$' + _r(rng, ALNUM + './', 86)
    if p == 4:
        return 'pbkdf2_sha256$600000$' + _r(rng, ALNUM, 12) + '$' + _r(rng, B64, 44)
    if p == 5:
        return '{SSHA}' + _r(rng, B64, int(rng.integers(28, 40)))
    if p == 6:
        return '{SSHA512}' + _r(rng, B64, int(rng.integers(60, 90)))
    if p == 7:
        return '$scrypt$ln=' + str(int(rng.integers(14, 18))) + ',r=8,p=1$' + _r(rng, B64, 16) + '$' + _r(rng, B64, 44)
    if p == 8:
        return '{CRYPT}$1$' + _r(rng, ALNUM + './', 8) + '$' + _r(rng, ALNUM + './', 22)
    return '{MD5}' + _r(rng, B64, 24)
ENVIRONMENT = {'sql': ["current_setting('app.{k}')", "(SELECT valor FROM secretos WHERE clave = '{k}')", "vault.read('secret/{k}')"], 'python': ['os.environ["{K}"]', 'os.getenv("{K}")', 'os.environ.get("{K}", "")', 'settings.{k}', 'config["{k}"]', 'boto3.client("ssm").get_parameter(Name="{k}")', 'vault.read("secret/{k}")["data"]', 'SecretStr(os.environ["{K}"])'], 'js': ['process.env.{K}', 'process.env["{K}"]', 'import.meta.env.{K}', 'config.get("{k}")', 'await secrets.get("{k}")', 'Deno.env.get("{K}")'], 'java': ['System.getenv("{K}")', 'System.getProperty("{k}")', '${K}', 'env.getProperty("{k}")', 'vault.read("{k}")'], 'go': ['os.Getenv("{K}")', 'os.LookupEnv("{K}")', 'viper.GetString("{k}")'], 'ruby': ["ENV['{K}']", 'ENV.fetch("{K}")', 'Rails.application.credentials.{k}'], 'php': ['getenv("{K}")', '$_ENV["{K}"]', 'env("{k}")', 'config("app.{k}")'], 'csharp': ['Environment.GetEnvironmentVariable("{K}")', '_config["{k}"]', 'builder.Configuration["{k}"]'], 'rust': ['std::env::var("{K}").unwrap()', 'env::var("{K}")?', 'dotenvy::var("{K}")?'], 'shell': ['"${K}"', '"$(cat /run/secrets/{k})"', '"$(vault kv get -field={k} secret/app)"'], 'yaml': ['${K}', '${K:-}', "{{ vault('secret/{k}') }}", '!vault |', 'valueFrom:'], 'generic': ['${K}', '{{ {k} }}', '%({k})s', '$({K})'], 'gha': ['${{ secrets.{K} }}', '${{ vars.{K} }}', '${{ secrets.GITHUB_TOKEN }}', '${{ env.{K} }}'], 'gitlabci': ['${{K}}', '$CI_JOB_TOKEN', '$CI_REGISTRY_PASSWORD'], 'jenkins': ["credentials('{k}')", 'env.{K}', 'params.{K}'], 'circleci': ['${{K}}', '$CIRCLE_TOKEN'], 'compose': ['${{K}}', '${{K}:?falta}', '${{K}:-}'], 'helm': ['{{ .Values.{k} }}', '{{ .Release.Name }}', '{{ required "falta" .Values.{k} }}'], 'ansible': ['{{ vault_{k} }}', '{{ lookup("env", "{K}") }}', '{{ {k} }}'], 'makefile': ['$({K})', '$(shell cat /run/secrets/{k})'], 'kotlin': ['System.getenv("{K}")', 'BuildConfig.{K}'], 'swift': ['ProcessInfo.processInfo.environment["{K}"]', 'Bundle.main.object(forInfoDictionaryKey: "{K}")'], 'scala': ['sys.env("{K}")', 'config.getString("{k}")'], 'perl': ['$ENV{{K}}', '$config->{{k}}'], 'lua': ['os.getenv("{K}")', 'config.{k}'], 'dart': ['Platform.environment["{K}"]', 'String.fromEnvironment("{K}")'], 'elixir': ['System.get_env("{K}")', 'Application.get_env(:app, :{k})'], 'powershell': ['$env:{K}', '(Get-Secret -Name {k})'], 'groovy': ['System.getenv("{K}")', 'project.{k}']}
PLACEHOLDERS = ['YOUR_{K}_HERE', 'YOUR-{K}-HERE', '<{k}>', '<REEMPLAZAR>', '<YOUR_{K}>', 'changeme', 'CHANGE_ME', 'change-me', 'TODO', 'FIXME', 'REDACTED', '***', 'xxxxxxxxxxxxxxxx', 'XXXXXXXXXXXXXXXXXXXXXXXX', 'not-a-real-key', 'fake-{k}', 'dummy', 'example', 'test', 'placeholder', 'INSERT_{K}_HERE', 'sample-{k}', '0000000000000000000000000000000', 'aaaaaaaaaaaaaaaaaaaaaaaa', '1234567890abcdef1234567890abcdef', 'deadbeefdeadbeefdeadbeefdeadbeef', 'sk_' 'test_51ExampleKeyDoNotUse', 'AKIAIOSFODNN7EXAMPLE']
EMPTIES = ['""', "''", 'None', 'null', 'nil', 'undefined', '{}', '[]', '0', 'false', '"" if DEBUG else None', 'NULL', 'Optional.empty()', 'std::ptr::null()']
REFERENCES = ['{K}', 'self.{k}', 'cfg.{k}', 'opts.{k}', 'this.{k}', 'DEFAULT_{K}', 'parent.{k}', 'props.{k}', 'args.{k}', "CONFIG['{k}']", 'settings.{K}', 'credentials.{k}', 'secrets.{k}']
LANGUAGES = ('python', 'js', 'ts', 'java', 'go', 'ruby', 'php', 'csharp', 'rust', 'shell', 'yaml', 'json', 'toml', 'properties', 'xml', 'hcl', 'docker', 'ini', 'gha', 'gitlabci', 'jenkins', 'circleci', 'compose', 'helm', 'ansible', 'makefile', 'sql', 'kotlin', 'swift', 'scala', 'perl', 'lua', 'dart', 'elixir', 'powershell', 'groovy')

def _name(rng, base, language):
    if language in ('shell', 'docker', 'properties', 'ini', 'makefile', 'powershell', 'gha', 'gitlabci', 'circleci'):
        return base.upper()
    if language in ('js', 'ts', 'java', 'csharp', 'kotlin', 'swift', 'scala', 'dart', 'groovy'):
        p = base.split('_')
        n = p[0] + ''.join((x.capitalize() for x in p[1:]))
        return n[0].upper() + n[1:] if language == 'csharp' and rng.integers(0, 2) else n
    if language in ('yaml', 'hcl') and rng.integers(0, 3) == 0:
        return base.replace('_', '-')
    return base.upper() if rng.integers(0, 5) == 0 else base

def _sp(rng, maximum=12):
    n = int(rng.choice([0, 1, 1, 1, 1, 2, 3, 4, 6, 9, maximum]))
    return ' ' * n

def _ind(rng):
    k = int(rng.choice([0, 0, 0, 0, 1, 1, 2, 3]))
    if k == 0:
        return ''
    return '\t' * k if rng.integers(0, 5) == 0 else '    ' * k

def _com(rng, language):
    if rng.integers(0, 4):
        return ''
    txt = str(rng.choice(['ok', 'TODO', 'FIXME', 'ver #4213', 'no tocar', 'rotar cada 90 dias', 'heredado', 'temporal']))
    m = {'python': '#', 'ruby': '#', 'shell': '#', 'yaml': '#', 'toml': '#', 'properties': '#', 'ini': ';', 'docker': '#', 'hcl': '#', 'perl': '#', 'makefile': '#', 'sql': '--', 'powershell': '#', 'elixir': '#', 'gha': '#', 'gitlabci': '#', 'circleci': '#', 'compose': '#', 'helm': '#', 'ansible': '#', 'lua': '--'}.get(language, '//')
    return f'  {m} {txt}'

def line_in(rng, language, name, value, quotes=True):
    n = _name(rng, name, language)
    q = '"' if rng.integers(0, 5) else "'"
    v = f'{q}{value}{q}' if quotes else value
    (a, b) = (_sp(rng), _sp(rng))
    (i, c) = (_ind(rng), _com(rng, language))
    L = language
    if L == 'python':
        f = int(rng.integers(0, 4))
        s = f'{n}{a}={b}{v}' if f == 0 else f'self.{n}{a}={b}{v}' if f == 1 else f'{q}{n}{q}:{b}{v},' if f == 2 else f'{n}: str = {v}'
    elif L in ('js', 'ts'):
        kw = str(rng.choice(['const', 'let', 'var', '']))
        kind = ': string' if L == 'ts' and rng.integers(0, 3) == 0 else ''
        f = int(rng.integers(0, 3))
        s = f'{kw} {n}{kind}{a}={b}{v};'.lstrip() if f == 0 else f'{n}:{b}{v},' if f == 1 else f'{q}{n}{q}:{b}{v},'
    elif L == 'java':
        mod = str(rng.choice(['private static final String', 'public static final String', 'static final String', 'String', 'private String']))
        s = f'{mod} {n}{a}={b}{v};'
    elif L == 'go':
        f = int(rng.integers(0, 3))
        s = f'{n}{a}:={b}{v}' if f == 0 else f'{n}{a}={b}{v}' if f == 1 else f'{n}:{b}{v},'
    elif L == 'ruby':
        f = int(rng.integers(0, 3))
        s = f'{n}{a}={b}{v}' if f == 0 else f'{n}:{b}{v},' if f == 1 else f':{n}{a}=>{b}{v},'
    elif L == 'php':
        f = int(rng.integers(0, 3))
        s = f'${n}{a}={b}{v};' if f == 0 else f'define({q}{n.upper()}{q},{b}{v});' if f == 1 else f'{q}{n}{q}{a}=>{b}{v},'
    elif L == 'csharp':
        mod = str(rng.choice(['private const string', 'public const string', 'const string', 'private readonly string', 'var']))
        s = f'{mod} {n}{a}={b}{v};'
    elif L == 'rust':
        f = int(rng.integers(0, 3))
        s = f'let {n}{a}={b}{v};' if f == 0 else f'const {n.upper()}: &str{a}={b}{v};' if f == 1 else f'{n}:{b}{v}.to_string(),'
    elif L == 'shell':
        f = int(rng.integers(0, 3))
        s = f'export {n}={v}' if f == 0 else f'{n}={v}' if f == 1 else f'{n}={v}; export {n}'
    elif L == 'yaml':
        s = f'{n}:{b}{v}'
    elif L == 'json':
        s = f'{q}{n}{q}:{b}{v},'
    elif L == 'toml':
        s = f'{n}{a}={b}{v}'
    elif L == 'properties':
        s = f'{n.lower()}{a}={b}{value}' if rng.integers(0, 2) else f'{n.lower()}={value}'
        v = value
    elif L == 'xml':
        s = f'<{n.lower()}>{value}</{n.lower()}>' if rng.integers(0, 2) else f'<add key={q}{n}{q} value={q}{value}{q} />'
        v = value
    elif L == 'hcl':
        s = f'{n}{a}={b}{v}'
    elif L == 'docker':
        f = int(rng.integers(0, 3))
        s = f'ENV {n}={v}' if f == 0 else f'ARG {n}={v}' if f == 1 else f'ENV {n} {value}'
        if f == 2:
            v = value
    elif L == 'gha':
        f = int(rng.integers(0, 3))
        s = f'          {n.upper()}: {v}' if f == 0 else f'        env:\n          {n.upper()}: {v}' if f == 1 else f'          --{n.lower()} {value}'
    elif L == 'gitlabci':
        s = f'  {n.upper()}: {v}' if rng.integers(0, 2) else f'    - export {n.upper()}={v}'
    elif L == 'jenkins':
        f = int(rng.integers(0, 3))
        s = f'        {n.upper()} = {v}' if f == 0 else f"        withCredentials([string(credentialsId: {v}, variable: '{n.upper()}')]) {{" if f == 1 else f"""        sh 'curl -H "Authorization: Bearer {value}"'"""
    elif L == 'circleci':
        s = f'      {n.upper()}: {v}'
    elif L == 'compose':
        f = int(rng.integers(0, 3))
        s = f'      - {n.upper()}={value}' if f == 0 else f'      {n.upper()}: {v}' if f == 1 else f'    command: --{n.lower()}={value}'
    elif L == 'helm':
        s = f'  {n}: {v}'
    elif L == 'ansible':
        f = int(rng.integers(0, 3))
        s = f'    {n}: {v}' if f == 0 else f'  - name: configura\n    shell: app --{n}={value}' if f == 1 else f'''    {n}: "{{{{ {n} | default('{value}') }}}}"'''
    elif L == 'sql':
        f = int(rng.integers(0, 4))
        s = f'INSERT INTO config (clave, valor) VALUES ({q}{n}{q}, {v});' if f == 0 else f'CREATE USER svc WITH PASSWORD {v};' if f == 1 else f'UPDATE ajustes SET valor{a}={b}{v} WHERE clave{a}={b}{q}{n}{q};' if f == 2 else f'-- {n}{a}={b}{v}'
    elif L == 'makefile':
        s = f'{n.upper()} := {value}' if rng.integers(0, 2) else f'\t@curl -H "Authorization: Bearer {value}" $(URL)'
    elif L == 'kotlin':
        s = f'val {n}{a}={b}{v}' if rng.integers(0, 2) else f'const val {n.upper()}{a}={b}{v}'
    elif L == 'swift':
        s = f'let {n}{a}={b}{v}' if rng.integers(0, 2) else f'static let {n}: String{a}={b}{v}'
    elif L == 'scala':
        s = f'val {n}{a}={b}{v}' if rng.integers(0, 2) else f'private val {n}: String{a}={b}{v}'
    elif L == 'perl':
        s = f'my ${n}{a}={b}{v};' if rng.integers(0, 2) else f'$ENV{{{n.upper()}}}{a}={b}{v};'
    elif L == 'lua':
        s = f'local {n}{a}={b}{v}' if rng.integers(0, 2) else f'config.{n}{a}={b}{v}'
    elif L == 'dart':
        s = f'const {n}{a}={b}{v};' if rng.integers(0, 2) else f'static const String {n}{a}={b}{v};'
    elif L == 'elixir':
        s = f'@{n}{a}{v}' if rng.integers(0, 2) else f'config :app, {n}: {v}'
    elif L == 'powershell':
        s = f'${n}{a}={b}{v}' if rng.integers(0, 2) else f'$env:{n.upper()}{a}={b}{v}'
    elif L == 'groovy':
        s = f'def {n}{a}={b}{v}' if rng.integers(0, 2) else f'{n}{a}={b}{v}'
    else:
        s = f'{n.lower()}{a}={b}{v}'
    line = i + s + c
    return (line, line.find(value))
_CRED_PRE = ['', 'app_', 'api_', 'client_', 'service_', 'admin_', 'root_', 'db_', 'smtp_', 'aws_', 'gcp_', 'azure_', 'stripe_', 'github_', 'slack_', 'master_', 'private_', 'internal_', 'prod_', 'legacy_']
_CRED_ROOT = ['key', 'secret', 'token', 'password', 'passwd', 'pass', 'credential', 'auth', 'bearer', 'apikey', 'access_key', 'secret_key', 'private_key', 'client_secret', 'refresh_token', 'session_key', 'signing_key', 'encryption_key', 'webhook_secret', 'connection_string', 'dsn']
_CRED_SUF = ['', '_1', '_v2', '_prod', '_live', '_id', '_value', '_env']
_BEN_PRE = ['', 'cached_', 'last_', 'prev_', 'current_', 'expected_', 'computed_', 'stored_', 'remote_', 'local_', 'parent_', 'source_', 'target_', 'base_', 'upstream_', 'vendor_', 'public_', 'shared_', 'default_']
_BEN_ROOT = ['commit', 'checksum', 'digest', 'etag', 'uuid', 'revision', 'build', 'trace', 'span', 'request', 'correlation', 'session', 'cache', 'bundle', 'asset', 'migration', 'fixture', 'sample', 'snapshot', 'manifest', 'content', 'artifact', 'job', 'run', 'batch', 'message', 'event', 'document', 'image', 'layer', 'blob', 'package', 'release', 'version', 'nonce', 'seed', 'salt_public', 'transaction', 'invoice', 'order']
_BEN_SUF = ['', '_id', '_hash', '_sha', '_sha256', '_md5', '_sum', '_digest', '_ref', '_fingerprint', '_checksum', '_uid', '_guid', '_tag', '_version', '_key', '_name', '_slug', '_etag', '_integrity']
_SUF_PROV = ['_api_key', '_token', '_secret', '_key', '_password', '_apikey', '_access_token', '_client_secret', '_auth', '_credential', '_pat', '_api_secret', '_private_key', '_webhook_secret', '_signing_key']

def credential_name(rng):
    d = rng.random()
    if d < 0.35:
        n = name_of_provider(rng)
        if n:
            return n
    elif d < 0.5:
        p_ = word_key(rng)
        if p_:
            return p_ + _SUF_PROV[rng.integers(0, len(_SUF_PROV))]
    return _CRED_PRE[rng.integers(0, len(_CRED_PRE))] + _CRED_ROOT[rng.integers(0, len(_CRED_ROOT))] + _CRED_SUF[rng.integers(0, len(_CRED_SUF))]

def benign_name(rng):
    return _BEN_PRE[rng.integers(0, len(_BEN_PRE))] + _BEN_ROOT[rng.integers(0, len(_BEN_ROOT))] + _BEN_SUF[rng.integers(0, len(_BEN_SUF))]
NEGATIVE_WEIGHTS = {'benign_entropy': 22, 'public': 14, 'password_hash': 11, 'encrypted': 8, 'redacted': 14, 'pattern': 13, 'public_pair': 10, 'crypto': 9, 'curated_fp': 9, 'environment': 20, 'placeholder': 17, 'empty': 9, 'reference': 8, 'url_without_credential': 9, 'table': 8}
_NM_PUB = ('public_key', 'pubkey', 'publishable_key', 'fingerprint', 'cert', 'ssh_public_key', 'verify_key', 'integrity', 'sri_hash', 'did', 'wallet_address', 'cert_fingerprint', 'host_key')
_NM_HPW = ('password_hash', 'hashed_password', 'pw_hash', 'digest', 'stored_hash', 'crypt', 'password_digest', 'pass_hash', 'credential_digest')
_NM_ENC = ('password', 'api_key', 'db_password', 'token', 'tls_key', 'secret', 'client_secret', 'private_key')

def _pick(rng, weights):
    keys = list(weights)
    total = sum(weights.values())
    x = rng.random() * total
    acc = 0.0
    for k in keys:
        acc += weights[k]
        if x < acc:
            return k
    return keys[-1]
WEB_NEGATIVES = False
_NM_WEB = ['expected', 'query', 'qs', 'href', 'path', 'pathname', 'raw_path', 'input', 'encoded', 'escaped', 'search', 'location', 'redirect_to', 'filename', 'body', 'integrity', 'want', 'got', 'uri', 'target']

def _pct(rng, text):
    return ''.join((c if c.isalnum() and rng.random() < 0.8 else f'%{ord(c):02X}' for c in text))

def _encoded_web(rng):
    t = int(rng.integers(0, 5))
    word = lambda : _r(rng, string.ascii_lowercase, int(rng.integers(1, 7)))
    if t == 0:
        n = int(rng.integers(1, 5))
        return '&'.join((f"{word()}%5B{(word() if rng.integers(0, 2) else '')}%5D{('%5B%5D' if rng.integers(0, 2) else '')}={rng.integers(0, 99)}" for _ in range(n)))
    if t == 1:
        cps = [int(rng.integers(12353, 12438)) for _ in range(int(rng.integers(2, 9)))]
        return ''.join((f'%{b:02X}' for ch in cps for b in chr(ch).encode('utf8')))
    if t == 2:
        base = str(rng.choice(['http://example.com/', 'https://www.example.com/', '/', 'sc://', 'ftp://example.org/', '/search?q=']))
        return base + _pct(rng, _r(rng, string.ascii_letters + ' /?=&#<>{}\'"', int(rng.integers(6, 40))))
    if t == 3:
        ents = ['&lt;', '&gt;', '&amp;', '&quot;', '&#039;', '&#13;', '&#10;', '&#9;', '&nbsp;']
        return ''.join((str(rng.choice(ents)) for _ in range(int(rng.integers(3, 12)))))
    (alg, nb) = [('sha512', 64), ('sha384', 48), ('sha256', 32), ('sha1', 20)][int(rng.integers(0, 4))]
    return f'{alg}-' + base64.b64encode(rng.bytes(nb)).decode()

def _negative(rng, family, L):
    if family == 'benign_entropy':
        return (benign_name(rng), _ben_hash(rng) if rng.integers(0, 2) else _ben_id(rng), True)
    if family == 'public':
        n = str(rng.choice(_NM_PUB))
        if rng.integers(0, 4) == 0:
            return (n, _pem_multiline(rng, False), False)
        return (n, _ben_public(rng), True)
    if family == 'password_hash':
        return (str(rng.choice(_NM_HPW)), _ben_hash_of_pass(rng), True)
    if family == 'redacted':
        return (credential_name(rng), _ben_redacted(rng), True)
    if family == 'public_pair':
        par = PAIRS[rng.integers(0, len(PAIRS))]
        return (par[0], par[1](rng), True)
    if family == 'curated_fp':
        v = false_positive_curated(rng)
        if v:
            return (credential_name(rng), v, '"' not in v and "'" not in v)
        return (benign_name(rng), _ben_hash(rng), True)
    if family == 'url_without_credential':
        if WEB_NEGATIVES and rng.random() < 0.6:
            return (str(rng.choice(_NM_WEB)), _encoded_web(rng), True)
        return (benign_name(rng), _url_public(rng), True)
    if family == 'table':
        return (str(rng.choice(['ALFABETO', 'CHARSET', 'BASE62', 'DIGITS', 'HEXCHARS', 'kAlphabet', 'SAFE_CHARS', 'b58_alphabet', 'TABLA', 'ASCII_MAP', 'printable', 'URL_SAFE'])), _table_of_chars(rng), True)
    if family == 'crypto':
        n = str(rng.choice(['salt', 'kdf_salt', 'iv', 'nonce', 'aes_iv', 'gcm_nonce', 'test_vector', 'NIST_VECTOR', 'rfc_key', 'counter']))
        return (n, _ben_material_crypto(rng), True)
    if family == 'pattern':
        n = str(rng.choice(['PATRON', 'regex', 'RE_SECRETO', 'pattern', 'rule', 'detector', 'SECRET_RE', 'matcher', 'signature']))
        return (n, _ben_pattern(rng), True)
    if family == 'encrypted':
        v = _encrypted_in_rest(rng)
        return (str(rng.choice(_NM_ENC)), v, '\n' not in v)
    if family == 'environment':
        tab = ENVIRONMENT.get(L, ENVIRONMENT['generic'])
        return (credential_name(rng), tab[rng.integers(0, len(tab))], False)
    name_short = _credential_name_not_password if PASSWORD_NAMED_SHORT else credential_name
    if family == 'ordinary_short_value':
        if rng.integers(0, 4) == 0:
            return (name_short(rng), _word_ordinary(rng), True)
        (nm, val) = _short_ordinary(rng)
        return (nm, val, True)
    if family == 'placeholder':
        if ORDINARY_SHORT_VALUES is True and rng.random() < 0.6:
            if rng.integers(0, 4) == 0:
                return (name_short(rng), _word_ordinary(rng), True)
            (nm, val) = _short_ordinary(rng)
            return (nm, val, True)
        if SHORT_PASSWORDS and ORDINARY_SHORT_VALUES != 'separate' and (rng.random() < 0.4):
            return (name_short(rng), _word_ordinary(rng), True)
        (n, v) = (credential_name(rng), _placeholder(rng))
        if PASSWORD_NAMED_SHORT and v.lower() in _WEAK_STICKS and is_password_name(n):
            v = str(rng.choice(_MARKERS_G1))
        return (n, v, True)
    if family == 'empty':
        return (credential_name(rng), EMPTIES[rng.integers(0, len(EMPTIES))], False)
    return (credential_name(rng), REFERENCES[rng.integers(0, len(REFERENCES))], False)
FORMS_WITH_NAME = ('assignment', 'kwargs', 'decoded', 'diff', 'comment')
FAMILIES_BY_NAME = ('public_pair', 'benign_entropy')
FORM_WEIGHTS = {'assignment': 60, 'non_assignment': 12, 'kwargs': 5, 'header': 6, 'comment': 5, 'k8s': 5, 'decoded': 3, 'diff': 4}
CONNECTION_CONFIG = 0.0
_NM_CONNECTION = ['DATABASE_URL', 'DB_URL', 'REDIS_URL', 'MONGO_URI', 'MONGODB_URI', 'BROKER_URL', 'CELERY_BROKER_URL', 'SQLALCHEMY_DATABASE_URI', 'AMQP_URL', 'CACHE_URL', 'PG_DSN', 'spring.datasource.url', 'database.url', 'jdbc.url', 'db.connection', 'quarkus.datasource.jdbc.url', 'connection_string', 'connectionString', 'dsn', 'database_url', 'redis_url', 'mongo_uri']

def _connection_config(rng, positive):
    e = str(rng.choice(['postgres', 'postgresql', 'mysql', 'mongodb', 'mongodb+srv', 'redis', 'rediss', 'amqp', 'amqps', 'mariadb', 'clickhouse', 'sqlserver']))
    h = _host_any(rng)
    port = '' if rng.integers(0, 4) == 0 else f':{rng.integers(1000, 9999)}'
    base = '/' + _r(rng, LOWER, int(rng.integers(3, 9)))
    tail = str(rng.choice(['', '', '', '?sslmode=require', '?ssl=true', '?retryWrites=true']))
    u = _r(rng, LOWER, int(rng.integers(3, 9)))
    if positive:
        v = f"{e}://{u}:{_r(rng, ALNUM + '-_', int(rng.integers(12, 30)))}@{h}{port}{base}{tail}"
    else:
        k = int(rng.integers(0, 4))
        if k == 0:
            v = f'{e}://{h}{port}{base}{tail}'
        elif k == 1:
            v = f'{e}://{u}@{h}{port}{base}{tail}'
        elif k == 2:
            v = f"{e}://{u}:${{{rng.choice(['DB_PASSWORD', 'PGPASSWORD', 'REDIS_PASS', 'DB_PASS'])}}}@{h}{port}{base}"
        else:
            v = f"{e}://{u}:{rng.choice(['<password>', 'PASSWORD', 'changeme', 'xxxxxxxx', '********'])}@{h}{port}{base}"
    if e in ('postgresql', 'mysql', 'mariadb', 'sqlserver') and rng.integers(0, 3) == 0:
        v = 'jdbc:' + v
    n = str(rng.choice(_NM_CONNECTION))
    t = int(rng.integers(0, 6))
    q = '"' if rng.integers(0, 2) else "'"
    if t == 0:
        pre = f'{n}='
    elif t == 1:
        pre = f'{_ind(rng)}{n}: '
    elif t == 2:
        pre = f'{n} = {q}'
        v_post = q
    elif t == 3:
        pre = f'{_ind(rng)}{n}: {q}'
        v_post = q
    elif t == 4:
        pre = f"export {n.upper().replace('.', '_')}="
    else:
        pre = f"{_ind(rng)}- name: {n.upper().replace('.', '_')}\n{_ind(rng)}  value: {q}"
        v_post = q
    post = v_post if t in (2, 3, 5) else ''
    return (pre + v + post, (len(pre), len(pre) + len(v)))
NO_EQUALS_SIGN = 0.0
_DIR_SECRET = ['password', 'passwd', 'secret', 'requirepass', 'masterauth', 'auth-token', 'api-key', 'apikey', 'token', 'bind-password', 'bindpw', 'rootpw', 'shared-secret', 'key', 'passphrase', 'auth_pass', 'enable secret', 'wpa-psk', 'snmp-community', 'rpc-secret']
_DIR_BENIGN = ['auth', 'password', 'secret', 'token', 'key', 'passwd', 'auth-method', 'password-policy', 'secret-file', 'key-file', 'token-endpoint', 'passphrase-file']
_VAL_BENIGN = ['required', 'sufficient', 'optional', 'none', 'ask', 'prompt', 'yes', 'no', 'plain', 'md5', 'scram-sha-256', 'external', 'pam_unix.so', 'include common-auth', '/etc/ssl/private/server.key', '/run/secrets/db_password', '/var/lib/app/token', '${DB_PASSWORD}', '$PASSWORD', '%{ENV:SECRET}', '<password>', '********', 'changeme', 'env:API_TOKEN', 'file:/etc/app/secret', 'min=12', 'expire 90']

def _no_equal(rng, positive):
    import base64 as _b64
    ind = _ind(rng) if rng.integers(0, 2) else ''
    sep = ' ' * int(rng.integers(1, 9)) if rng.integers(0, 4) else '\t'
    t = int(rng.integers(0, 5))
    if positive:
        v = _password_short(rng) if rng.integers(0, 2) else _sec_generic(rng)
        v = v.split()[0]
        if t == 0:
            pairs = [('machine', _host_any(rng)), ('login', _r(rng, LOWER, int(rng.integers(3, 9))))]
            pre = ind + ' '.join((f'{k} {x}' for (k, x) in pairs)) + ' password '
            return (pre + v + str(rng.choice(['', '', ' port 443', ' # ci'])), (len(pre), len(pre) + len(v)))
        if t == 1:
            b = _b64.b64encode(f'{_r(rng, LOWER, int(rng.integers(3, 10)))}:{v}'.encode()).decode()
            n = str(rng.choice(['auth', 'authorization', 'basic_auth', 'registry_auth', 'credentials', 'identitytoken']))
            pre = ind + str(rng.choice([f'"{n}": "', f'{n}: ', f'{n} = "', f'{n}{sep}']))
            post = '"' if pre.endswith('"') else ''
            return (pre + b + post, (len(pre), len(pre) + len(b)))
        d = _DIR_SECRET[rng.integers(0, len(_DIR_SECRET))]
        if rng.integers(0, 3) == 0:
            d = d.upper() if rng.integers(0, 2) else d.replace('-', '_')
        q = str(rng.choice(['', '', '', '"', "'"]))
        pre = ind + d + sep + q
        return (pre + v + q + str(rng.choice(['', '', '  # rotar', ' ;'])), (len(pre), len(pre) + len(v)))
    d = (_DIR_BENIGN if rng.integers(0, 2) else _DIR_SECRET)[rng.integers(0, 12)]
    if t == 0:
        x = f'machine {_host_any(rng)} login {_r(rng, LOWER, 6)}' + str(rng.choice(['', ' account ci', ' macdef init']))
        return (ind + x, (len(ind) + 8, len(ind) + len(x)))
    if t == 1:
        b = _b64.b64encode(str(rng.choice(['{"alg":"none"}', 'hello world, this is a test', 'application/json; charset=utf-8', 'Lorem ipsum dolor sit amet', '<svg xmlns="http://www.w3.org/2000/svg"/>'])).encode()).decode()
        n = str(rng.choice(['data', 'payload', 'content', 'body', 'icon', 'auth_header_example']))
        pre = f'{ind}"{n}": "'
        return (pre + b + '"', (len(pre), len(pre) + len(b)))
    vb = _VAL_BENIGN[rng.integers(0, len(_VAL_BENIGN))]
    pre = ind + d + sep
    return (pre + vb, (len(pre), len(pre) + len(vb)))
REINFORCED_FORMS = 0.0
REINFORCED_FORMS_TUPLE = True
_TAG_SECRET = ['password', 'passwd', 'secret', 'apiKey', 'api-key', 'token', 'accessToken', 'clientSecret', 'privateKey', 'dbPassword', 'db_password', 'auth-token', 'passphrase', 'credential', 'secretKey', 'masterKey']
_TAG_NEUTRAL = ['port', 'host', 'name', 'version', 'timeout', 'enabled', 'url', 'region', 'level', 'path', 'user', 'mode']

def _form_reinforced(rng, positive):
    ind = _ind(rng)
    sec = (_sec_generic(rng) if rng.integers(0, 3) else _sec_prefix(rng)) if rng.integers(0, 4) else _password_short(rng)
    sec = sec.split()[0]
    ben = str(rng.choice(['${env.DB_PASSWORD}', '${DB_PASS}', 'changeme', '********', '{{ vault_password }}', '%PASSWORD%', 'ENC(' + _r(rng, B64, 24) + ')', '', 'null', '@secret.ref@', '$(cat /run/secrets/pw)', '$PASSWORD', '<password>', 'xxxxxxxx', 'see-vault', 'REDACTED']))
    v = sec if positive else ben
    t = int(rng.integers(0, 3 if REINFORCED_FORMS_TUPLE else 2))
    if t == 0:
        e = _TAG_SECRET[rng.integers(0, len(_TAG_SECRET))]
        if rng.integers(0, 4) == 0:
            e = e.upper() if rng.integers(0, 2) else e.capitalize()
        before = ''.join((f'{ind}<{x}>{_r(rng, LOWER + _DIG, int(rng.integers(2, 9)))}</{x}>\n' for x in [_TAG_NEUTRAL[rng.integers(0, len(_TAG_NEUTRAL))] for _ in range(int(rng.integers(0, 3)))]))
        pre = before + f'{ind}<{e}>'
        return (pre + v + f'</{e}>', (len(pre), len(pre) + len(v)))
    if t == 1:
        u = str(rng.choice(['admin', 'deploy', 'svc', 'ci', 'root', 'user', _r(rng, LOWER, 6)]))
        order = str(rng.choice(['curl -u ', 'curl --user ', 'wget --user=' + u + ' --password=', 'http -a ', 'mysql -u ' + u + ' -p', 'sshpass -p ', 'redis-cli -a ', 'mongosh -u ' + u + ' -p ', 'smbclient -U ' + u + '%']))
        with_u = order.endswith(('-u ', '--user ', '-a ')) and 'redis' not in order
        q = str(rng.choice(['', '', "'", '"']))
        pre = ind + order + q + (u + ':' if with_u else '')
        tail = q + ' ' + str(rng.choice(['https://api.acme.io/v1', '-h db.interno', 'host', '//nas/share', '']))
        if not positive and rng.integers(0, 2):
            v = ''
            pre = pre.rstrip(':')
        return (pre + v + tail, (len(pre), len(pre) + len(v)))
    n = credential_name(rng).upper() if rng.integers(0, 2) else credential_name(rng)
    other = str(rng.choice(['admin', 'svc', 'deploy', 'api', 'eu-west-1', 'default']))
    (ab, ce) = (('(', ')'), ('[', ']'), ('{', '}'))[rng.integers(0, 3)]
    pre = f'{ind}{n} = {ab}"{other}", "'
    return (pre + v + f'"{ce}', (len(pre), len(pre) + len(v)))
WEAK_PASSWORDS = 0.0
_WORD_WEAK = ['dragon', 'monkey', 'shadow', 'sunshine', 'princess', 'football', 'baseball', 'letmein', 'welcome', 'master', 'qwerty', 'trustno', 'starwars', 'freedom', 'whatever', 'cheese', 'pepper', 'ginger', 'banana', 'summer', 'winter', 'orange', 'purple', 'silver', 'rocket', 'falcon', 'tiger', 'guitar', 'soccer', 'hockey', 'killer', 'jordan', 'harley', 'ranger', 'buster', 'thomas', 'robert', 'daniel', 'george', 'maggie', 'admin', 'root', 'test', 'demo', 'guest', 'changeit', 'secreto', 'clave', 'acceso', 'madrid', 'lisboa']
_NM_UNAMBIGUOUS = ['password', 'pass', 'passwd', 'passphrase', 'pwd', 'user_password', 'db_password', 'ftp_password', 'smtp_pass', 'admin_pass', 'root_password', 'secret', 'pin', 'clave', 'contrasena', 'master_password', 'login_password']
_NON_CRED_NAMES = ['username', 'user', 'login', 'host', 'name', 'theme', 'nickname', 'database', 'profile', 'label', 'alias', 'team', 'project', 'channel', 'mascot', 'codename', 'first_name', 'city', 'role', 'group']

def _weak(rng):
    w = _WORD_WEAK[rng.integers(0, len(_WORD_WEAK))]
    t = int(rng.integers(0, 7))
    if t == 0:
        return w + str(rng.integers(0, 100))
    if t == 1:
        return w + str(rng.integers(100, 10000))
    if t == 2:
        return w
    if t == 3:
        return w + str(rng.choice(['!', '!1', '_1', '123', '1234', '01', '2020', '#1']))
    if t == 4:
        return w + '_' + str(rng.choice(['pass', 'pwd', 'secret', 'pw', 'key']))
    if t == 5:
        return str(rng.choice(['test', 'my', 'the', 'new', 'old', 'dev'])) + w
    return w + _WORD_WEAK[rng.integers(0, len(_WORD_WEAK))][:4] + str(rng.integers(0, 10))

def _password_weak(rng, positive):
    ind = _ind(rng)
    v = _weak(rng)
    n = _NM_UNAMBIGUOUS[rng.integers(0, len(_NM_UNAMBIGUOUS))]
    if rng.integers(0, 4) == 0:
        n = n.upper()
    elif rng.integers(0, 5) == 0:
        n = ''.join((x.capitalize() if i else x for (i, x) in enumerate(n.split('_'))))
    if not positive:
        k = int(rng.integers(0, 3))
        if k == 0:
            n = _NON_CRED_NAMES[rng.integers(0, len(_NON_CRED_NAMES))]
        elif k == 1:
            v = str(rng.choice(['', 'false', 'true', 'null', 'None', '${' + n.upper() + '}', '%(' + n + ')s', '<' + n + '>', '********', 'changeme', 'env:' + n.upper(), 'prompt', 'ask', '{{ ' + n + ' }}', '$' + n.upper(), 'xxxxxx']))
        else:
            n = str(rng.choice(['promptForPass', 'password_hint', 'passwordPolicy', 'pass_through', 'passive', 'password_field', 'min_password_length', 'keyboardInteractiveForPass', 'password_reset_url', 'show_password', 'passenger']))
            v = str(rng.choice(['false', 'true', '8', v, 'optional', '/reset', 'none']))
    f = int(rng.integers(0, 7))
    if f == 0:
        pre = f'{ind}"{n}": "'
        post = '",'
    elif f == 1:
        pre = f"{ind}{n}={('' if rng.integers(0, 2) else chr(34))}"
        post = '"' if pre.endswith('"') else ''
    elif f == 2:
        pre = f'{ind}{n}: '
        post = ''
    elif f == 3:
        pre = f'{ind}<{n}>'
        post = f'</{n}>'
    elif f == 4:
        pre = f"{ind}{n} = '"
        post = "'"
    elif f == 5:
        pre = f'{ind}<add key="{n}" value="'
        post = '" />'
    elif not positive and n in _NON_CRED_NAMES:
        pre = f"{ind}{str(rng.choice(['ftp', 'mongodb', 'mysql', 'postgres', 'amqp']))}://"
        post = f'@{_host_any(rng)}/{_r(rng, LOWER, 5)}'
    else:
        pre = f"{ind}{str(rng.choice(['ftp', 'mongodb', 'mysql', 'postgres', 'amqp', 'sftp']))}://{_weak(rng)}:"
        post = f'@{_host_any(rng)}/{_r(rng, LOWER, 5)}'
        if not positive:
            v = str(rng.choice(['${DB_PASS}', '<password>', 'PASSWORD', '****']))
    return (pre + v + post, (len(pre), len(pre) + len(v)))
NON_ASSIGNMENT_WEIGHT = None
NESTED_WEIGHT = 0

def _nested(rng, name, value, language):
    q = '"' if rng.integers(0, 3) else "'"
    v = f'{q}{value}{q}'
    field = str(rng.choice([name, name, 'pass', 'password', 'secret', 'key', 'token', 'value']))
    a = str(rng.choice(['db', 'auth', 'api', 'service', 'backend', 'cache', 'smtp', 'storage']))
    b = str(rng.choice(['creds', 'credentials', 'auth', 'login', 'secrets', 'access', 'conn']))
    t = int(rng.integers(0, 7))
    if t == 0:
        return f'export const cfg = {{ {a}: {{ {b}: {{ {field}: {v} }} }} }};'
    if t == 1:
        return f'CFG = {{"{a}": {{"{b}": {{"{field}": {v}}}}}}}'
    if t == 2:
        return f'  "{a}": {{"type": "basic", "{b}": {{"user": "svc", "{field}": "{value}"}}}},'
    if t == 3:
        return f'var cfg = new {{ {a.capitalize()} = new {{ {b.capitalize()} = new {{ {field.capitalize()} = "{value}" }} }} }};'
    if t == 4:
        return f'local cfg = {{ {a} = {{ {b} = {{ {field} = {v} }} }} }}'
    if t == 5:
        return f'{a}:\n  {b}:\n    {field}: {v}'
    return f'cfg := Config{{{a.capitalize()}: {a.capitalize()}Config{{{b.capitalize()}: Creds{{{field.capitalize()}: "{value}"}}}}}}'

def _weights_form():
    if NON_ASSIGNMENT_WEIGHT is None and (not NESTED_WEIGHT) and (not AUTH_HEADER):
        return FORM_WEIGHTS
    p = dict(FORM_WEIGHTS)
    if NON_ASSIGNMENT_WEIGHT is not None:
        p['non_assignment'] = NON_ASSIGNMENT_WEIGHT
    if NESTED_WEIGHT:
        p['nested'] = NESTED_WEIGHT
    if AUTH_HEADER:
        p['header'] = HEADER_WEIGHT_G2
    return p

def line(rng, positive: bool, language=None):
    if CONNECTION_CONFIG and rng.random() < CONNECTION_CONFIG:
        return _connection_config(rng, positive)
    if NO_EQUALS_SIGN and rng.random() < NO_EQUALS_SIGN:
        return _no_equal(rng, positive)
    if REINFORCED_FORMS and rng.random() < REINFORCED_FORMS:
        return _form_reinforced(rng, positive)
    if WEAK_PASSWORDS and rng.random() < WEAK_PASSWORDS:
        return _password_weak(rng, positive)
    if PASSWORD_NAMED_SHORT or AUTH_HEADER or URL_SHORT_PASSWORD or UNNAMED_VALUES or DOTNET_RESOURCES:
        rec = _line_recipe(rng, positive)
        if rec is not None:
            return rec
    L = language or LANGUAGES[rng.integers(0, len(LANGUAGES))]
    family_neg = None
    short = False
    if positive:
        p = rng.random()
        if p < 0.14:
            n = credential_name(rng) if rng.integers(0, 4) else benign_name(rng)
            (v, q) = (_sec_prefix(rng), True)
        elif p < 0.21:
            n = credential_name(rng) if rng.integers(0, 4) else benign_name(rng)
            (v, q) = (_sec_prefix_rare(rng), True)
        elif p < 0.33:
            v = catalog_secret(rng) or example_trufflehog(rng) or _sec_prefix(rng)
            n = credential_name(rng) if rng.integers(0, 4) else benign_name(rng)
            q = True
        elif p < 0.36:
            par = PAIRS[rng.integers(0, len(PAIRS))]
            v = par[3](rng)
            (n, q) = (par[2], '\n' not in v)
        elif p < 0.47:
            (n, v, q) = (credential_name(rng), _sec_obfuscated(rng), True)
        elif p < 0.72:
            (n, v, q) = (credential_name(rng), _sec_generic(rng), True)
            if SHORT_PASSWORDS and rng.random() < SHORT_PASSWORDS:
                v = _password_short(rng)
                short = True
        elif p < 0.82:
            (n, v, q) = (credential_name(rng), _sec_connection(rng), True)
        else:
            n = credential_name(rng) if rng.integers(0, 2) else benign_name(rng)
            v = _pem_multiline(rng, True) if rng.integers(0, 3) else _sec_pem(rng)
            q = '\n' not in v
    else:
        family = family_neg = _pick(rng, dict(NEGATIVE_WEIGHTS, ordinary_short_value=14) if ORDINARY_SHORT_VALUES == 'separate' else NEGATIVE_WEIGHTS)
        (n, v, q) = _negative(rng, family, L)
        v = v.replace('{K}', n.upper()).replace('{k}', n.lower())
    v = v.strip() if isinstance(v, str) and v.strip() else v
    mark = v
    per_name = q and family_neg in FAMILIES_BY_NAME or short
    if per_name:
        weights = {k: v for (k, v) in _weights_form().items() if k in FORMS_WITH_NAME}
        form = _pick(rng, weights)
    elif q:
        form = _pick(rng, _weights_form())
    else:
        form = 'kwargs' if rng.integers(0, 4) == 0 else 'assignment'
    if form == 'assignment':
        (txt, i) = line_in(rng, L, n, v, quotes=q)
    elif form == 'kwargs':
        txt = _kwargs(rng, n, v, L, quotes=q)
        i = txt.find(v)
    elif form == 'non_assignment':
        (txt, mark) = _no_assignment(rng, n, v, L)
        i = txt.find(mark)
    elif form == 'nested':
        txt = _nested(rng, n, v, L)
        i = txt.find(v)
    elif form == 'header':
        txt = _header(rng, v)
        i = txt.find(v)
    elif form == 'comment':
        txt = _comment(rng, n, v, L)
        i = txt.find(v)
    elif form == 'k8s':
        (txt, mark) = _k8s(rng, v)
        i = txt.find(mark)
    elif form == 'decoded':
        (txt, mark) = _decoded(rng, n, v, L)
        i = txt.find(mark)
    else:
        (base, _) = line_in(rng, L, n, v, quotes=q)
        txt = _diff(rng, base)
        i = txt.find(v)
    if i < 0:
        (txt, i, mark) = (*line_in(rng, L, n, v, quotes=q), v)
    if per_name and n.lower() not in txt.lower():
        (txt, i) = line_in(rng, L, n, v, quotes=q)
        if n.lower() not in txt.lower():
            (txt, i) = (f'{n} = {v}', len(n) + 3)
        mark = v
    return (txt, (i, i + len(mark)))

def _kwargs(rng, name, value, language, quotes=True):
    fn = str(rng.choice(['connect', 'Client', 'login', 'authenticate', 'Session', 'create_engine', 'new Client', 'http.NewRequest', 'Database.connect', 'boto3.client', 'requests.get']))
    q = ('"' if rng.integers(0, 4) else "'") if quotes else ''
    if not quotes:
        if fn in ('boto3.client', 'requests.get'):
            first = "'s3', " if fn.startswith('boto3') else 'url, '
            return _ind(rng) + f'{fn}({first}{name}={value})'
        return _ind(rng) + f'{fn}({name}={value})'
    if language in ('js', 'ts', 'java', 'csharp', 'go'):
        s = f'{fn}({q}{value}{q})' if rng.integers(0, 2) else f'{fn}({{ {name}: {q}{value}{q} }})'
    elif fn in ('boto3.client', 'requests.get'):
        first = "'s3', " if fn.startswith('boto3') else 'url, '
        s = f'{fn}({first}{name}={q}{value}{q})'
    else:
        s = f'{fn}({name}={q}{value}{q})'
    return _ind(rng) + s

def _header(rng, value):
    p = int(rng.integers(0, 4))
    q = '"' if rng.integers(0, 3) else "'"
    if p == 0:
        return _ind(rng) + f'headers = {{{q}Authorization{q}: {q}Bearer {value}{q}}}'
    if p == 1:
        return _ind(rng) + f'-H {q}Authorization: Bearer {value}{q}'
    if p == 2:
        return _ind(rng) + f'Authorization: Basic {value}'
    return _ind(rng) + f'curl -u {q}admin:{value}{q} https://api.example.com'

def _comment(rng, name, value, language):
    m = {'python': '#', 'ruby': '#', 'shell': '#', 'yaml': '#', 'toml': '#', 'ini': ';', 'docker': '#', 'hcl': '#'}.get(language, '//')
    q = '"' if rng.integers(0, 3) else "'"
    pre = str(rng.choice(['', 'TODO: quitar ', 'antiguo: ', 'backup: ', 'prod: ']))
    return _ind(rng) + f'{m} {pre}{name} = {q}{value}{q}'

def _no_assignment(rng, name, value, language):
    q = '"' if rng.integers(0, 4) else "'"
    v = f'{q}{value}{q}'
    L = language
    op = int(rng.integers(0, 6))
    if L in ('python',):
        s_ = [f'parser.add_argument({q}--{name}{q}, default={v})', f'globals()[{q}{name.upper()}{q}] = {v}', f'os.environ.setdefault({q}{name.upper()}{q}, {v})', f'except KeyError:\n    {name} = {v}', f'conn = connect(host={q}db{q}, user={q}app{q}, {name}={v})', f'return Config(**{{{q}{name}{q}: {v}}})'][op]
    elif L in ('js', 'ts'):
        s_ = [f'fetch(url, {{ headers: {{ Authorization: `Bearer {value}` }} }});', f'const {{ {name} = {v} }} = opts;', f'export default {{ {name}: {v} }};', f"axios.create({{ auth: {{ username: 'svc', password: {v} }} }});", f'new Client({v});', f'app.use(session({{ secret: {v} }}));'][op]
    elif L == 'go':
        s_ = [f'req.Header.Set("Authorization", "Bearer {value}")', f'cfg := Config{{{name.capitalize()}: {v}}}', f'flag.String("{name}", {v}, "")', f'os.Setenv("{name.upper()}", {v})', f'client := New(WithToken({v}))', f'return &Auth{{Token: {v}}}'][op]
    elif L == 'shell':
        s_ = [f'curl -H "Authorization: Bearer {value}" https://api.example.com', f'echo "machine api.example.com login svc password {value}" > ~/.netrc', f'echo "db.example.com:5432:app:svc:{value}" >> ~/.pgpass', f'docker login -u svc -p {v} registry.example.com', f'psql "postgresql://app:{value}@db.example.com/prod"', f'AWS_SECRET_ACCESS_KEY={v} aws s3 ls'][op]
    elif L == 'java':
        s_ = [f'conn = DriverManager.getConnection(url, "app", {v});', f'builder.header("Authorization", "Bearer {value}");', f'props.setProperty("{name}", {v});', f'return new Credentials("svc", {v});', f'client = Client.builder().token({v}).build();', f'@Value("{value}") private String {name};'][op]
    elif L == 'yaml':
        half = len(value) // 2
        folded = f'{value[:half]}\n    {value[half:]}'
        s_ = [f'  - name: {name.upper()}\n    value: {v}', f'  args: ["--{name}", {v}]', f'  {name}: >-\n    {folded}', f'  env:\n    {name.upper()}: {v}', f'  _{name}: &{name} {value}\n  auth: *{name}', f'  {name}: {v}'][op]
        if op == 2:
            return (_ind(rng) + s_, folded)
    elif L == 'sql':
        s_ = [f"CREATE USER svc WITH PASSWORD '{value}';", f"INSERT INTO config (clave, valor) VALUES ('{name}', '{value}');", f"ALTER USER app WITH PASSWORD '{value}';", f"CREATE SERVER s FOREIGN DATA WRAPPER w OPTIONS (password '{value}');", f"""COPY t FROM PROGRAM 'curl -H "Authorization: Bearer {value}" u';""", f"CREATE ROLE svc LOGIN PASSWORD '{value}';"][op]
    else:
        s_ = [f'connect({name}={v})', f'Header("Authorization", "Bearer {value}")', f'setOption({q}{name}{q}, {v})', f'auth({v})', f'new Client({v})', f'{name}({v})'][op]
    if rng.integers(0, 5) == 0:
        m = int(rng.integers(0, 6))
        s_ = [f'logger.info("usando {name}={value}")', f'raise ValueError(f"no pude autenticar con {value}")', f'print(f"DEBUG {name} -> {value}")', f'console.error(`fallo de auth con {value}`);', f'log.Printf("token=%s", "{value}")', f'assert.Equal(t, "{value}", cfg.{name})'][m]
    return (_ind(rng) + s_, value.strip('"\''))

def _decoded(rng, name, value, language):
    raw = value.strip('"\'')
    hexa = ''.join(('%02x' % ord(c) for c in raw))
    b64 = base64.b64encode(raw.encode()).decode()
    if language in ('js', 'ts'):
        op = [f'const {name} = Buffer.from("{b64}", "base64").toString();', f'const {name} = atob("{b64}");']
    elif language == 'go':
        op = [f'{name}, _ := base64.StdEncoding.DecodeString("{b64}")', f'{name}, _ := hex.DecodeString("{hexa}")']
    elif language == 'java':
        op = [f'byte[] {name} = Base64.getDecoder().decode("{b64}");', f'String {name} = new String(Hex.decodeHex("{hexa}"));']
    elif language == 'shell':
        op = [f'{name.upper()}=$(echo {b64} | base64 -d)', f'{name.upper()}=$(xxd -r -p <<< {hexa})']
    else:
        op = [f'{name} = binascii.unhexlify("{hexa}")', f'{name} = base64.b64decode("{b64}")', f'{name} = codecs.decode("{hexa}", "hex")', f'{name} = bytes.fromhex("{hexa}")']
    line_ = _ind(rng) + op[int(rng.integers(0, len(op)))]
    return (line_, b64 if b64 in line_ else hexa)

def _diff(rng, line):
    return str(rng.choice(['+', '+', '+', '-', ' '])) + line

def _pem_multiline(rng, private=True):
    kind = str(rng.choice(['RSA PRIVATE KEY', 'PRIVATE KEY', 'EC PRIVATE KEY', 'OPENSSH PRIVATE KEY'])) if private else str(rng.choice(['PUBLIC KEY', 'CERTIFICATE', 'RSA PUBLIC KEY']))
    n = int(rng.integers(2, 5))
    body = '\n'.join((_r(rng, B64, 64) for _ in range(n)))
    return f'-----BEGIN {kind}-----\n{body}\n-----END {kind}-----'

def _k8s(rng, value, secret=True):
    b = base64.b64encode(value.encode()).decode()
    field = str(rng.choice(['password', 'token', 'api-key', 'tls.key', 'secret']))
    if rng.integers(0, 2):
        return (f'{_ind(rng)}data:\n  {field}: {b}', b)
    return (f'{_ind(rng)}stringData:\n  {field}: {value}', value)

def _encrypted_in_rest(rng):
    p = int(rng.integers(0, 9))
    if p == 0:
        return 'ENC[AES256_GCM,data:' + _r(rng, B64, 44) + ',iv:' + _r(rng, B64, 44) + ',tag:' + _r(rng, B64, 24) + ',type:str]'
    if p == 1:
        return 'AgB' + _r(rng, B64, 200)
    if p == 2:
        return '$ANSIBLE_VAULT;1.1;AES256\n' + _r(rng, HEX, 80)
    if p == 3:
        return '\\x00GITCRYPT\\x00' + _r(rng, B64, 60)
    if p == 4:
        return 'vault:v1:' + _r(rng, B64, 88)
    if p == 5:
        return 'ENC(' + _r(rng, B64, int(rng.integers(40, 90))) + ')'
    if p == 6:
        return 'sops:enc:AES256_GCM:' + _r(rng, B64, int(rng.integers(40, 80)))
    if p == 7:
        return 'gAAAAAB' + _r(rng, B64URL, int(rng.integers(50, 120)))
    return 'age-encryption.org/v1\n-> X25519 ' + _r(rng, B64, 43)
PASSWORD_NAMED_SHORT = 0.0
AUTH_HEADER = 0.0
URL_SHORT_PASSWORD = 0.0
UNNAMED_VALUES = 0.0
DOTNET_RESOURCES = 0.0
HEADER_WEIGHT_G2 = 12
import re as _re_g
_WORD_G1 = ['tortuga', 'naranja', 'pelota', 'ventana', 'caballo', 'montana', 'estrella', 'galleta', 'pajaro', 'bosque', 'river', 'forest', 'castle', 'pirate', 'wizard', 'rabbit', 'kitten', 'puppy', 'flower', 'cookie', 'mango', 'lemon', 'cherry', 'zebra', 'panda', 'koala', 'otter', 'salmon', 'cactus', 'comet', 'anchor', 'bridge', 'candle', 'button', 'pencil', 'marble', 'copper', 'bamboo', 'violet', 'indigo', 'crimson', 'snowman', 'sunset', 'harbor', 'meadow', 'canyon', 'island', 'lantern', 'compass', 'mustang', 'phoenix', 'raptor', 'viking', 'samurai', 'ninja', 'gopher', 'python', 'ubuntu', 'kubernetes', 'docker', 'jenkins', 'grafana', 'kafka', 'oracle', 'sparrow', 'walrus', 'badger', 'beaver', 'moose', 'bison', 'lynx']
_WEAK_CLASSIC_G1 = ['admin', 'admin123', 'root', 'secret', 'secret123', 'passw0rd', 'password1', 'qwerty', 'qwerty123', '123456', '12345678', 'abc123', 'test', 'test123', 'guest', 'demo', 'pass', 'pass123', 'letmein', 'welcome1', 'master', 'default', 's3cr3t', 'p4ssw0rd', 'toor', 'changeit']
_MARKERS_G1 = ['changeme', 'CHANGEME', 'change_me', 'change-me', '<password>', '<your-password>', '<PASSWORD>', 'your_password', 'YOUR_PASSWORD', 'your-password-here', 'YOUR_PASSWORD_HERE', 'REPLACE_ME', 'replace-me', 'xxxxxxxx', 'XXXXXXXXXX', '****', '********', '*****', '[password]', '{{password}}', 'TODO', 'tbd', '<secret>', '<db-password>', 'PASSWORD_HERE', 'hunter2']
_WEAK_STICKS = {'secret', 'supersecret', 'mysecret', 'password', 'passwd', '123456', 'letmein', 'admin', 'root', 'guest', 'abc123', '1234', 'qwerty', 'asdf', 'test', 'testing', 'demo', 'foo', 'bar', 'baz', 'qux', 'lorem', 'wip'}
_PW_NAME_RX = _re_g.compile('pass|pwd|(^|[^a-z])pw([^a-z]|$)|contrase|clave', _re_g.I)
_PW_PRE = ['', '', '', '', 'db_', 'admin_', 'root_', 'user_', 'mysql_', 'postgres_', 'smtp_', 'redis_', 'ftp_', 'ldap_', 'mail_', 'app_', 'default_', 'test_', 'new_', 'old_', 'master_', 'keystore_', 'bind_', 'rabbitmq_', 'mongo_', 'sa_', 'proxy_', 'vnc_', 'wifi_', 'service_', 'api_user_']
_PW_ROOT = ['password', 'password', 'password', 'password', 'passwd', 'pass', 'pwd', 'passphrase', 'pw']
_PW_SPECIAL = ['MYSQL_ROOT_PASSWORD', 'POSTGRES_PASSWORD', 'PGPASSWORD', 'MYSQL_PASSWORD', 'REDIS_PASSWORD', 'RABBITMQ_DEFAULT_PASS', 'MONGO_INITDB_ROOT_PASSWORD', 'mysqlrootpwd', 'ADMIN_PASSWORD', 'password_confirmation', 'passwordConfirm', 'current_password', 'new_password', 'userPassword', 'KEYSTORE_PASS', 'SA_PASSWORD', 'DB_PASS']

def is_password_name(n):
    return bool(_PW_NAME_RX.search(n or ''))

def _credential_name_not_password(rng):
    for _ in range(50):
        n = credential_name(rng)
        if not is_password_name(n):
            return n
    return 'api_key'

def _obfuscated_like(rng, w):
    return ''.join((_r(rng, LOWER, 1) if c.islower() else _r(rng, LOWER.upper(), 1) if c.isupper() else _r(rng, '0123456789', 1) if c.isdigit() else c for c in w))

def _password_g1(rng, lo=4, hi=16):
    w = _WORD_G1[int(rng.integers(0, len(_WORD_G1)))]
    t = int(rng.integers(0, 9))
    if t == 0:
        v = w
    elif t == 1:
        v = w + str(rng.integers(0, 10000))
    elif t == 2:
        v = ''.join((_LEET.get(c, c) if rng.random() < 0.6 else c for c in w)) + ('' if rng.integers(0, 2) else str(rng.integers(0, 100)))
    elif t == 3:
        v = w.capitalize() + str(rng.integers(1990, 2031)) + str(rng.choice(['!', '#', '$', '*', '.', '?', '@']))
    elif t == 4:
        v = w + _WORD_G1[int(rng.integers(0, len(_WORD_G1)))]
    elif t == 5:
        v = str(rng.choice(_WEAK_CLASSIC_G1))
    elif t == 6:
        v = _r(rng, '0123456789', int(rng.integers(4, 9)))
    elif t == 7:
        v = _r(rng, ALNUM, int(rng.integers(8, 17)))
    else:
        v = _obfuscated_like(rng, w + (str(rng.integers(0, 1000)) if rng.integers(0, 2) else ''))
    if t not in (6, 7, 8) and rng.integers(0, 5) == 0:
        v = _obfuscated_like(rng, v)
    v = v[:hi]
    while len(v) < lo:
        v += str(rng.integers(0, 10))
    return v

def _pw_name_g1(rng):
    if rng.integers(0, 6) == 0:
        n = str(rng.choice(_PW_SPECIAL))
        return (n, 'password')
    while True:
        (pre, root) = (str(rng.choice(_PW_PRE)), str(rng.choice(_PW_ROOT)))
        if root != 'pw' or pre:
            return (pre + root, root)

def _in_style(n, style):
    if n != n.lower() and n != n.upper():
        return n
    low = n.lower()
    p = low.split('_')
    if style == 'upper':
        return low.upper()
    if style == 'camel':
        return p[0] + ''.join((x.capitalize() for x in p[1:]))
    if style == 'pascal':
        return ''.join((x.capitalize() for x in p))
    if style == 'kebab':
        return low.replace('_', '-')
    return low if n == n.upper() and style == 'snake' else n

def _contrast_g1(rng, kind, name):
    k = int(rng.integers(0, 4))
    N = name.upper().replace('.', '_').replace('-', '_')
    if kind == 'sql':
        return (str(rng.choice(_MARKERS_G1)) if rng.integers(0, 4) else '', False)
    if k == 0:
        return (str(rng.choice(_MARKERS_G1)), False)
    if k == 1:
        if kind == 'code':
            return (str(rng.choice([f'os.getenv("{N}")', f'process.env.{N}', f'ENV["{N}"]', f'System.getenv("{N}")', f'os.Getenv("{N}")', f'env("{N}")', f'getenv("{N}")'])), True)
        return (str(rng.choice([f'${{{N}}}', f'${N}', f'${{{N}:-}}', '{{ .Values.password }}', f'%({name.lower()})s', f'{{{{ {name.lower()} }}}}', f'#{{{N}}}', f'%{N}%', f'${{env:{N}}}'])), False)
    if k == 2:
        return ('', False)
    if kind == 'code':
        return (str(rng.choice(['cfg.password', 'self.password', 'opts[:password]', 'params.password', 'DEFAULT_PASSWORD', 'password', 'config.Password', 'settings.DB_PASSWORD', 'creds.pass', 'this.password', 'user.password'])), True)
    return (str(rng.choice([f'${{{N}}}', f'${{secrets.{N}}}', f'${{vault:{name.lower()}}}'])), False)

def _password_named_short(rng, positive):
    ind = _ind(rng) if rng.integers(0, 2) else ''
    (base, root) = _pw_name_g1(rng)
    q = '"' if rng.integers(0, 3) else "'"
    f = int(rng.integers(0, 10))
    (kind, qc) = ('code', q)
    if f == 0:
        L = int(rng.integers(0, 7))
        n = _in_style(base, 'camel' if L in (1, 4, 6) else 'snake')
        (pre, post) = [(f'{n} = {q}', q), (f'const {n} = {q}', q + ';'), (f'{n} := {q}', q), (f'{n} = {q}', q), (f'''private static final String {_in_style(base, 'upper')} = "''', '";'), (f'${n} = {q}', q + ';'), (f'var {n} = "', '";')][L]
        if L in (4, 6):
            qc = '"'
    elif f == 1:
        obj = str(rng.choice(['o', 'cfg', 'user', 'opts', 'conn', 'c.Websocket', 'settings', 'config.Database', 'self', 'u', 'account', 's.Auth', 'options.Credentials', 'client.config']))
        t = int(rng.integers(0, 3))
        if t == 0:
            (pre, post, qc) = (f'''{obj}.{_in_style(base, 'pascal')} = "''', '"', '"')
        elif t == 1:
            (pre, post) = (f"{obj}.{_in_style(base, 'snake')} = {q}", q)
        else:
            (pre, post, qc) = (f'''{_in_style(base, 'pascal')}: "''', '",', '"')
    elif f == 2:
        who = str(rng.choice(['alice', 'bob', 'admin', 'tester', 'user1', 'jdoe']))
        n = _in_style(base, 'snake')
        (pre, post) = [(f'signup(email: {q}{who}@example.com{q}, {n}: {q}', q + ')'), (f'User.create!(name: {q}{who}{q}, {n}: {q}', q + ')'), (f'login(username={q}{who}{q}, {n}={q}', q + ')'), (f"{{ username: {q}{who}{q}, {_in_style(base, 'camel')}: {q}", q + ' }'), (f'client.auth(user={q}{who}{q}, {n}={q}', q + ')')][int(rng.integers(0, 5))]
    elif f == 3:
        (kind, qc) = ('config', '"')
        (pre, post) = (f'''"{_in_style(base, 'camel' if rng.integers(0, 2) else 'snake')}": "''', '",')
    elif f == 4:
        kind = 'config'
        n = _in_style(base, 'kebab' if rng.integers(0, 4) == 0 else 'snake')
        if rng.integers(0, 2):
            (pre, post, qc) = (f'{n}: ', '', '')
        else:
            (pre, post) = (f'{n}: {q}', q)
    elif f == 5:
        (kind, qc) = ('config', '')
        grp = str(rng.choice(['spring.datasource', 'smtp', 'mail.smtp', 'db', 'jdbc', 'ldap', 'app.mail', 'database', 'spring.rabbitmq', 'keystore', 'redis']))
        (pre, post) = (f"{grp}.{(root if root != 'pw' else 'password')}{str(rng.choice(['=', ' = ', '=']))}", '')
    elif f == 6:
        kind = 'config'
        N = _in_style(base, 'upper')
        t = int(rng.integers(0, 3))
        if t == 0:
            (pre, post, qc) = (f'{N}=', '', '')
        elif t == 1:
            (pre, post, qc) = (f'export {N}=', '', '')
        else:
            (pre, post, qc) = (f'{N}="', '"', '"')
    elif f == 7:
        (kind, qc) = ('config', '')
        N = _in_style(base, 'upper')
        (pre, post) = [(f'docker run -d -e {N}=', f" {rng.choice(['mysql:8', 'postgres:16', 'redis:7', 'mongo:7'])}"), (f'      - {N}=', ''), ('/credpassword:', ''), ('--password=', ' --host=db'), ('--password ', ''), (f"mysql -u{rng.choice(['root', 'admin', 'app'])} -p", ' -h 127.0.0.1'), (f"net use \\\\srv\\share /user:{rng.choice(['admin', 'svc'])} ", '')][int(rng.integers(0, 7))]
    elif f == 8:
        (kind, qc) = ('config', '"')
        if rng.integers(0, 2):
            (pre, post) = (f'''<add key="{_in_style(base, 'pascal')}" value="''', '" />')
        else:
            (pre, post, qc) = (f"<{_in_style(base, 'camel')}>", f"</{_in_style(base, 'camel')}>", '')
    else:
        (kind, qc) = ('sql', "'")
        (pre, post) = [("CREATE USER app WITH PASSWORD '", "';"), ("ALTER USER 'root'@'localhost' IDENTIFIED BY '", "';"), ("GRANT ALL ON app.* TO 'app'@'%' IDENTIFIED BY '", "';"), ("CREATE LOGIN svc WITH PASSWORD = '", "';")][int(rng.integers(0, 4))]
    if positive:
        v = _password_g1(rng)
    else:
        (v, drop) = _contrast_g1(rng, kind, base)
        if drop and qc and pre.endswith(qc) and post.startswith(qc):
            (pre, post) = (pre[:-len(qc)], post[len(qc):])
    pre = ind + pre
    return (pre + v + post, (len(pre), len(pre) + len(v)))

def _b64url_json(d):
    import json as _json
    return base64.urlsafe_b64encode(_json.dumps(d, separators=(',', ':')).encode()).decode().rstrip('=')

def _jwt_real(rng):
    alg = str(rng.choice(['HS256', 'RS256', 'ES256', 'HS512']))
    head = _b64url_json(dict(alg=alg, typ='JWT')) if rng.integers(0, 4) else _b64url_json(dict(typ='JWT', alg=alg))
    iat = int(rng.integers(1500000000, 1800000000))
    claims = dict(sub=str(rng.integers(1, 10 ** 9)) if rng.integers(0, 2) else _uuid(rng), iat=iat)
    if rng.integers(0, 2):
        claims['exp'] = iat + int(rng.choice([3600, 86400, 604800]))
    if rng.integers(0, 2):
        claims['name'] = _r(rng, LOWER, int(rng.integers(3, 9))).capitalize()
    if rng.integers(0, 3) == 0:
        claims['role'] = str(rng.choice(['admin', 'user', 'service']))
    sig = _r(rng, B64URL, 43 if alg in ('HS256', 'ES256') else 86)
    return f'{head}.{_b64url_json(claims)}.{sig}'

def _basic_b64(rng):
    user = str(rng.choice(['admin', 'user', 'api', 'svc', 'test', 'alice', 'bob', 'deploy', 'root', 'client'])) if rng.integers(0, 3) else _r(rng, LOWER, int(rng.integers(3, 9)))
    pw = _password_g1(rng) if rng.integers(0, 2) else _r(rng, ALNUM, int(rng.integers(8, 25)))
    return base64.b64encode(f'{user}:{pw}'.encode()).decode()

def _bearer_value(rng):
    t = int(rng.integers(0, 4))
    if t == 0:
        return _jwt_real(rng)
    if t == 1:
        return _r(rng, ALNUM + '-._~', int(rng.integers(20, 61)))
    if t == 2:
        return _r(rng, HEX, int(rng.choice([32, 40, 64])))
    return _sec_prefix(rng)
_BEARER_FILL = ['${TOKEN}', '<token>', 'xxxxx', '$ACCESS_TOKEN', '{{token}}', 'YOUR_TOKEN', '<access_token>', '${API_TOKEN}', '<YOUR_JWT>', 'TOKEN_HERE', '...', '$TOKEN', '{token}', '<jwt>', 'xxxxxxxxxxxxxxxx']
_BASIC_FILL = ['YOUR_BASE64', '<base64(user:pass)>', '<credentials>', '${BASIC_AUTH}', 'base64(user:password)', '<base64-encoded-credentials>', '$(echo -n user:pass | base64)', 'XXXXXXXX', '<b64>', '$CREDENTIALS']

def _auth_header(rng, positive):
    t = int(rng.integers(0, 10))
    scheme = 'Basic' if t % 3 == 0 or rng.integers(0, 3) == 0 else 'Bearer' if rng.integers(0, 6) else 'Token'
    v = (_basic_b64(rng) if scheme == 'Basic' else _bearer_value(rng)) if positive else str(rng.choice(_BASIC_FILL if scheme == 'Basic' else _BEARER_FILL))
    host = str(rng.choice(['api.example.com', 'localhost:8080', '127.0.0.1:3000', 'api.acme.io', '{{baseUrl}}', 'example.org', '$HOST', 'service.local']))
    path = str(rng.choice(['/v1/me', '/api/users', '/api/v2/items?limit=10', '/graphql', '/oauth/token', '/health']))
    q = '"' if rng.integers(0, 2) else "'"
    ind = _ind(rng) if rng.integers(0, 2) else ''
    ref = not positive and rng.integers(0, 3) == 0
    if t == 0:
        (pre, post) = (f'{ind}Authorization: {scheme} ', '')
    elif t == 1:
        acc = 'Accept: application/json\n' if rng.integers(0, 2) else ''
        (pre, post) = (f"{rng.choice(['GET', 'POST', 'DELETE'])} {path} HTTP/1.1\nHost: {host}\n{acc}Authorization: {scheme} ", '')
    elif t == 2:
        pr = str(rng.choice(['', '$ ', '    ']))
        (pre, post) = (f"{pr}curl {rng.choice(['-H', '--header'])} {q}Authorization: {scheme} ", f'{q} https://{host}{path}')
    elif t == 3:
        if ref:
            return (f"{ind}  .set('Authorization', '{scheme} ' + token)", (0, 0))
        (pre, post) = (f'{ind}  .set({q}Authorization{q}, {q}{scheme} ', f'{q})')
    elif t == 4:
        if ref:
            return (f'{ind}req.Header.Set("Authorization", "{scheme} "+token)', (0, 0))
        (pre, post) = (f'{ind}req.Header.Set("Authorization", "{scheme} ', '")')
    elif t == 5:
        if ref:
            return (f'{ind}headers = {{"Authorization": f"{scheme} {{token}}"}}', (0, 0))
        call = rng.integers(0, 2)
        pre = f'''{ind}{('r = requests.get(url, headers={' if call else 'headers = {')}"Authorization": "{scheme} '''
        post = '"}' + (')' if call else '')
    elif t == 6:
        if ref:
            return (f'{ind}headers: {{ Authorization: `{scheme} ${{token}}` }},', (0, 0))
        (pre, post) = (f'{ind}headers: {{ Authorization: {q}{scheme} ', f'{q} }},')
    elif t == 7:
        (pre, post) = (f'{ind}"Authorization": "{scheme} ', '",')
    elif t == 8:
        if rng.integers(0, 2):
            (pre, post) = (f'{ind}{q}HTTP_AUTHORIZATION{q} => {q}{scheme} ', f'{q},')
        else:
            (pre, post) = (f'{ind}request.headers[{q}Authorization{q}] = {q}{scheme} ', q)
    elif rng.integers(0, 2):
        (pre, post) = (f'| Authorization | `{scheme} ', '` |')
    else:
        (pre, post) = (f'Send the header `Authorization: {scheme} ', '` with every request.')
    return (pre + v + post, (len(pre), len(pre) + len(v)))
_URL_USERS = ['user', 'admin', 'root', 'guest', 'dlc', 'test', 'app', 'postgres', 'user3', 'svc', 'dev', 'sa', 'demo', 'ci', 'mysql', 'redis', 'rabbit', 'nats']

def _url_short_password(rng, positive):
    e = str(rng.choice(['postgres', 'postgresql', 'mysql', 'mongodb', 'redis', 'amqp', 'nats', 'ftp', 'sftp', 'smtp', 'ldap', 'http', 'https', 'mssql', 'rediss', 'clickhouse', 'mariadb']))
    u = str(rng.choice(_URL_USERS)) if rng.integers(0, 3) else _r(rng, LOWER, int(rng.integers(3, 8)))
    ctx = int(rng.integers(0, 6))
    if ctx == 0:
        (host, port) = ('%s', str(rng.choice([':%d', ':%s', ''])))
    elif ctx == 1:
        (host, port) = (str(rng.choice(['{host}', 'localhost', '{self.host}', '127.0.0.1'])), str(rng.choice([':{port}', ':5432', ''])))
    elif ctx == 2:
        (host, port) = (str(rng.choice(['${host}', 'localhost', '${HOST}'])), str(rng.choice([':${port}', ':27017', ''])))
    else:
        host = str(rng.choice(['localhost', '127.0.0.1', 'db', 'postgres', 'redis', 'rabbitmq', 'mongo', 'mysql', 'broker', 'cache', 'nats', '[::1]', 'host.docker.internal', 'db.local']))
        port = '' if rng.integers(0, 3) == 0 else f':{int(rng.choice([5432, 3306, 6379, 5672, 4222, 27017, 1433, 21, 5555, 9000]))}'
    path = str(rng.choice(['', '', '/', '/db', '/test', '/app?sslmode=disable', '/0', '/%2f', '/admin']))
    if positive:
        pw = _password_g1(rng, 3, 12) if rng.integers(0, 3) else _r(rng, LOWER + '0123456789', int(rng.integers(3, 13)))
        url = f'{e}://{u}:{pw}@{host}{port}{path}'
    else:
        k = int(rng.integers(0, 4))
        param = ['%s', '{password}', '${password}', '${DB_PASSWORD}', '${DB_PASSWORD}', '{{password}}'][ctx]
        if k == 0:
            url = f'{e}://{u}@{host}{port}{path}'
        elif k == 1:
            url = f'{e}://{u}:{param}@{host}{port}{path}'
        elif k == 2:
            url = f"{e}://{u}:{rng.choice(['<password>', '***', 'PASSWORD', '<pass>', 'xxxx', '****'])}@{host}{port}{path}"
        else:
            url = f'{e}://{host}{port}{path}'
    q = '"' if rng.integers(0, 3) else "'"
    if ctx == 0:
        pre = str(rng.choice(['fmt.Sprintf("', 'dsn := fmt.Sprintf("', 'url := "', 'addr := fmt.Sprintf("']))
        post = '", host, port)' if 'Sprintf' in pre else '"'
    elif ctx == 1:
        pre = str(rng.choice(['url = f"', 'DATABASE_URL = f"', 'return f"', "engine = create_engine(f'"]))
        post = "')" if pre.endswith("'") else '"'
    elif ctx == 2:
        (pre, post) = (str(rng.choice(['const uri = `', 'await connect(`', 'url: `'])), '`' + str(rng.choice(['', ';', ');', ','])))
    elif ctx == 3:
        (pre, post) = [(f"{rng.choice(['DATABASE_URL', 'REDIS_URL', 'BROKER_URL', 'AMQP_URL', 'NATS_URL', 'MONGO_URI'])}=", ''), (f"  {rng.choice(['url', 'uri', 'dsn', 'broker_url'])}: ", ''), (f"      - {rng.choice(['DATABASE_URL', 'CELERY_BROKER_URL'])}=", ''), (f"{rng.choice(['spring.datasource.url', 'db.url'])}=", '')][int(rng.integers(0, 4))]
    elif ctx == 4:
        pre = str(rng.choice([f'url = {q}', f'conn = connect({q}', f'assert parse({q}', f'DSN = {q}']))
        post = q + (')' if '(' in pre else '')
    else:
        (pre, post) = ('String url = "', '";')
    pre = (_ind(rng) if rng.integers(0, 2) else '') + pre
    return (pre + url + post, (len(pre), len(pre) + len(url)))
B32 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'

def _otp_secret(rng):
    return _r(rng, B32, int(rng.choice([16, 16, 20, 26, 32])))

def _nkey_seed(rng):
    return 'S' + str(rng.choice(list('ACNOPUX'))) + _r(rng, B32, 56)

def _nkey_public(rng):
    return str(rng.choice(list('UANO'))) + _r(rng, B32, 55)

def _unnamed_value(rng, positive):
    q = '"' if rng.integers(0, 2) else "'"
    ind = _ind(rng) if rng.integers(0, 2) else ''
    iss = str(rng.choice(['Example', 'ACME', 'MyApp', 'GitLab', 'Test']))
    who = _r(rng, LOWER, int(rng.integers(3, 8)))
    if positive:
        t = int(rng.integers(0, 6))
        if t == 0:
            v = _otp_secret(rng)
            (pre, post) = [(f'ROTP::TOTP.new({q}', f'{q})'), (f'totp = pyotp.TOTP({q}', f'{q})'), (f'pyotp.TOTP({q}', f'{q}).now()'), (f'otp = OTP.new({q}', f'{q})'), ('new Totp(Base32Encoding.ToBytes("', '"))'), (f'TOTP.generate({q}', f'{q})'), ('code, _ := totp.GenerateCode("', '", time.Now())'), (f'authenticator.generate({q}', f'{q})')][int(rng.integers(0, 8))]
        elif t == 1:
            v = _otp_secret(rng)
            lead = str(rng.choice(['', '# ', '// ', 'uri = ' + q]))
            pre = f'{lead}otpauth://totp/{iss}:{who}@example.com?secret='
            post = f'&issuer={iss}' + (q if lead.startswith('uri') else '')
        elif t == 2:
            v = _nkey_seed(rng)
            (pre, post) = (str(rng.choice(['# ', '// ', '# seed: ', '// user seed ', '-- '])), '')
        elif t == 3:
            v = _nkey_seed(rng)
            kk = str(rng.choice(['USER', 'ACCOUNT', 'OPERATOR']))
            (pre, post) = (f'-----BEGIN {kk} NKEY SEED-----\n', f'\n------END {kk} NKEY SEED------')
        elif t == 4:
            v = _nkey_seed(rng)
            pre = str(rng.choice(['kp, _ := nkeys.FromSeed([]byte("', 'nkeys.from_seed(b"', 'NKeys.FromSeed("']))
            post = '"))' if pre.endswith('byte("') else '")'
        else:
            v = (_sec_generic(rng) if rng.integers(0, 3) else _sec_prefix(rng)).split()[0].replace('"', '').replace("'", '')
            (pre, post) = [(f'client = Client({q}', f'{q})'), (f'login({q}{who}{q}, {q}', f'{q})'), (f'g = Github({q}', f'{q})'), (f'slack = WebClient({q}', f'{q})'), (f'bot = telebot.TeleBot({q}', f'{q})'), (f'client.run({q}', f'{q})'), ('api := NewClient("', '")'), (f'sdk.init({q}', f'{q});')][int(rng.integers(0, 8))]
    else:
        k = int(rng.integers(0, 7))
        if k == 0:
            v = _r(rng, B32, int(rng.choice([16, 26, 32, 52])))
            pre = str(rng.choice([f'get_item({q}', f'lookup({q}', f'cache.get({q}', f'record = find({q}', f'assert_equal({q}', f'Base32.decode({q}']))
            post = f'{q})'
        elif k == 1:
            v = _r(rng, HEX, int(rng.choice([7, 40, 40])))
            pre = str(rng.choice([f'repo.commit({q}', 'git checkout ', '# see ', f'revert({q}', 'cherry-pick ']))
            post = f'{q})' if pre.endswith(q) else ''
        elif k == 2:
            v = str(rng.choice([_uuid(rng), _r(rng, '0123456789ABCDEFGHJKMNPQRSTVWXYZ', 26), _r(rng, HEX, 32)]))
            pre = str(rng.choice([f'get_item({q}', f'fetch_order({q}', f'load_fixture({q}', f'Order.find({q}', f'delete_user({q}']))
            post = f'{q})'
        elif k == 3:
            v = _nkey_public(rng)
            pre = str(rng.choice(['# account: ', '// public key ', '# user: ', 'nkeys.FromPublicKey("', 'issuer: ']))
            post = '")' if pre.endswith('"') else ''
        elif k == 4:
            v = str(rng.choice(['user.otp_secret', "os.environ['OTP_SECRET']", 'pyotp.random_base32()', 'secret', "ENV['TOTP_SECRET']", 'current_user.otp_secret_key']))
            (pre, post) = (str(rng.choice(['ROTP::TOTP.new(', 'totp = pyotp.TOTP(', 'pyotp.TOTP(', 'otp = OTP.new('])), ')')
        elif k == 5:
            v = str(rng.choice(['{secret}', '<secret>', '${OTP_SECRET}', 'SECRET', '%s']))
            (pre, post) = (f'otpauth://totp/{iss}:{who}?secret=', f'&issuer={iss}')
        else:
            v = str(rng.choice(['token', "os.getenv('API_TOKEN')", 'password', 'settings.TOKEN', 'process.env.BOT_TOKEN']))
            pre = str(rng.choice(['client = Client(', f'login({q}{who}{q}, ', 'g = Github(', 'bot = telebot.TeleBot(', 'client.run(']))
            post = ')'
    pre = ind + pre
    return (pre + v + post, (len(pre), len(pre) + len(v)))
_PKT_KNOWN = ['b77a5c561934e089', '31bf3856ad364e35', 'b03f5f7f11d50a3a', '30ad4fe6b2a6aeed', '669e0ddf0bb1aa2a', 'cc7b13ffcd2ddd51', '89845dcd8080cc91', '7cec85d7bea7798e', 'adb9793829ddae60', '92742159e12e44c8']
_ASM = ['Newtonsoft.Json', 'System.Web.Mvc', 'log4net', 'EntityFramework', 'Microsoft.Owin', 'System.Net.Http', 'Castle.Core', 'NLog', 'AutoMapper', 'Microsoft.Extensions.Logging', 'System.Runtime', 'Moq']
_RES_KEYS = ['ENTER_API_KEY_TITLE', 'API_KEY_LABEL', 'PASSWORD_HINT', 'SECRET_KEY_DESCRIPTION', 'TOKEN_EXPIRED', 'ENTER_KEY_TITLE', 'INVALID_PASSWORD', 'AUTH_FAILED', 'PasswordLabel', 'ApiKeyPlaceholder', 'SecretKeyHelp', 'login.password.label', 'error.token.expired', 'settings.api_key', 'password_placeholder', 'key_missing']
_RES_TEXT = ['Enter your API key', 'API Key', 'At least 8 characters', 'The secret key of your account', 'Your session token has expired', 'Enter key', 'Invalid password', 'Authentication failed', 'Password:', 'Paste your key here', 'Keep this key secret', 'Forgot your password?', 'Clave API', 'Mot de passe', 'Passwort vergessen?', 'Token de acceso', 'Wrong username or password', 'Generate a new token']

def _pkt(rng):
    return str(rng.choice(_PKT_KNOWN)) if rng.integers(0, 2) else _r(rng, HEX, 16)

def _dotnet_negative(rng):
    t = int(rng.integers(0, 12))
    a = str(rng.choice(_ASM))
    ver = f'{rng.integers(1, 14)}.{rng.integers(0, 10)}.{rng.integers(0, 10)}.0'
    k = str(rng.choice(_RES_KEYS))
    tx = str(rng.choice(_RES_TEXT))
    if t == 0:
        return f'  <resheader name="reader">\n    <value>System.Resources.ResXResourceReader, System.Windows.Forms, Version=4.0.0.0, Culture=neutral, PublicKeyToken={_pkt(rng)}</value>\n  </resheader>'
    if t == 1:
        return f'  <assembly alias="System.Windows.Forms" name="System.Windows.Forms, Version=4.0.0.0, Culture=neutral, PublicKeyToken={_pkt(rng)}" />'
    if t == 2:
        return f'  <data name="$this.Icon" type="System.Drawing.Icon, System.Drawing, Version=4.0.0.0, Culture=neutral, PublicKeyToken={_pkt(rng)}" mimetype="application/x-microsoft.net.object.bytearray.base64">'
    if t == 3:
        return f'        <assemblyIdentity name="{a}" publicKeyToken="{_pkt(rng)}" culture="neutral" />'
    if t == 4:
        return f'    <section name="entityFramework" type="System.Data.Entity.Internal.ConfigFile.EntityFrameworkSection, EntityFramework, Version=6.0.0.0, Culture=neutral, PublicKeyToken={_pkt(rng)}" requirePermission="false" />'
    if t == 5:
        return f'    <Reference Include="{a}, Version={ver}, Culture=neutral, PublicKeyToken={_pkt(rng)}, processorArchitecture=MSIL">'
    if t == 6:
        return f'        <add assembly="{a}, Version={ver}, Culture=neutral, PublicKeyToken={_pkt(rng)}" />'
    if t == 7:
        return f'"{k}" = "{tx}";'
    if t == 8:
        return f'msgid "{tx}"\nmsgstr "{rng.choice(_RES_TEXT)}"'
    if t == 9:
        return f'{k}#0="{tx}"'
    if t == 10:
        return f'''  {k.lower().replace('.', '_')}: "{tx}"'''
    return f'''  <data name="{k.replace('.', '_')}" xml:space="preserve">\n    <value>{tx}</value>\n  </data>'''

def _dotnet_resources(rng, positive):
    neg = _dotnet_negative(rng)
    if not positive:
        return (neg, (0, 0))
    t = int(rng.integers(0, 3))
    if t == 0:
        v = (_sec_generic(rng) if rng.integers(0, 3) else _sec_prefix(rng)).split()[0].replace('"', '').replace('&', '')
        (pre, post) = (f'''    <add key="{rng.choice(['ApiKey', 'ClientSecret', 'StripeSecretKey', 'AuthToken', 'SendGridKey'])}" value="''', '" />')
    elif t == 1:
        v = _password_g1(rng, 6, 16) if rng.integers(0, 2) else _r(rng, ALNUM, int(rng.integers(12, 25)))
        pre = '    <add name="DefaultConnection" connectionString="Server=' + str(rng.choice(['db', 'localhost', 'sql01'])) + ';Database=app;User Id=sa;Password='
        post = ';" providerName="System.Data.SqlClient" />'
    else:
        v = (_sec_generic(rng) if rng.integers(0, 2) else _sec_prefix(rng)).split()[0].replace('<', '').replace('&', '')
        pre = f'''  <data name="{rng.choice(['ApiKey', 'ServiceToken', 'ClientSecret'])}" xml:space="preserve">\n    <value>'''
        post = '</value>\n  </data>'
    if not neg.isascii():
        return (pre + v + post, (len(pre), len(pre) + len(v)))
    if rng.integers(0, 2):
        head = neg + '\n' + pre
        return (head + v + post, (len(head), len(head) + len(v)))
    return (pre + v + post + '\n' + neg, (len(pre), len(pre) + len(v)))

def _line_recipe(rng, positive):
    if PASSWORD_NAMED_SHORT and rng.random() < PASSWORD_NAMED_SHORT:
        return _password_named_short(rng, positive)
    if AUTH_HEADER and rng.random() < AUTH_HEADER:
        return _auth_header(rng, positive)
    if URL_SHORT_PASSWORD and rng.random() < URL_SHORT_PASSWORD:
        return _url_short_password(rng, positive)
    if UNNAMED_VALUES and rng.random() < UNNAMED_VALUES:
        return _unnamed_value(rng, positive)
    if DOTNET_RESOURCES and rng.random() < DOTNET_RESOURCES:
        return _dotnet_resources(rng, positive)
    return None
