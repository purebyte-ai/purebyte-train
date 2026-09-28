"""FORMS of appearance: how personal data sits in bytes.

Records (fields with keys in Spanish or English and a value that is personal data or a look-alike) rendered as JSON,
YAML, CSV/TSV, SQL, Python/JS/Ruby/PHP/Java/Go literals, XML, .env/INI/TOML, `Label: value` forms, vCard, LDIF,
Markdown tables and key=value log lines; code snippets (author headers, package metadata, calls with literal arguments,
log calls, configuration lines, fixtures, credential lines, comments); prose (real sentences with their person slots
filled, and sentences that carry a value or a look-alike); e-mails; access, application, JSON and sshd logs.

A `Doc` accumulates text and its typed spans in BYTES of the UTF-8 encoding. `HELD_OUT_RENDERERS` are only used when
`held_out` is asked for (an evaluation set), so those formats stay new to a model trained without them.
"""
import json
from . import values as VL
from .types import IDX
from .values import chance, pick, rstr
HELD_OUT_RENDERERS = ('ldif', 'markdown', 'php', 'toml')

class Doc:

    def __init__(self):
        (self.parts, self.n, self.spans) = ([], 0, [])

    def add(self, s, typ=None):
        b = s.encode('utf-8') if isinstance(s, str) else bytes(s)
        if typ is not None and b:
            self.spans.append((self.n, self.n + len(b), IDX[typ]))
        self.parts.append(b)
        self.n += len(b)
        return self

    def value(self, v):
        if v.typ is None:
            return self.add(v.text)
        (a, b) = v.inner if v.inner else (0, len(v.text))
        self.add(v.text[:a]).add(v.text[a:b], v.typ).add(v.text[b:])
        return self

    def data(self):
        return b''.join(self.parts)

class Val:
    __slots__ = ('text', 'typ', 'inner')

    def __init__(self, text, typ=None, inner=None):
        (self.text, self.typ, self.inner) = (text, typ, inner)

    def quoted(self, q):
        return Val(q + self.text + q, self.typ, ((self.inner[0] + len(q), self.inner[1] + len(q)) if self.inner else (len(q), len(q) + len(self.text))) if self.typ else None)
EMPTY = ['', 'null', 'N/A', '-', 'n/a', 'none', '***', 'REDACTED', '[REDACTED]', 'desconocido', 'unknown', '0']

def _money(rng, lang):
    x = rng.uniform(1, 25000)
    return f'{x:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.') if lang == 'es' and chance(rng, 0.6) else f'{x:.2f}'

def _username(rng, L, lang):
    f = VL.ascii_fold(L.first(rng, lang)).lower()
    s = VL.ascii_fold(L.surname(rng, lang)).lower().replace(' ', '')
    return pick(rng, [f[0] + s, f + '_' + s[:4], f + str(int(rng.integers(1, 999))), s + f[0], f + '.' + s[:1], rstr(rng, VL.LOW, 6)])
_COMPANIES = ['Construcciones {S} S.L.', '{S} y Asociados', 'Grupo {S}', 'Transportes {S} S.A.', '{S} Abogados', '{S} & {S2} LLP', '{S} Holdings Inc.', '{S} Consulting Ltd.', 'Talleres {S}', 'Farmacia {S}', 'Acme Corp.', 'Iberdata S.L.', 'Nova Labs', 'Ferretería La Esquina', 'Blue Harbor Logistics', 'Delta Seguros S.A.', '{S} Software GmbH']

def company(rng, L, lang):
    return pick(rng, _COMPANIES).format(S=L.surname(rng, lang), S2=L.surname(rng, lang))

def _hostname(rng):
    return pick(rng, ['api', 'db', 'web', 'app', 'mail', 'cache', 'worker', 'proxy', 'vpn', 'git']) + pick(rng, ['', '-01', '-prod', '-02', '.internal', '1']) + ('.' + VL.company_domain(rng) if chance(rng, 0.5) else '')

def _version(rng):
    return pick(rng, [f'{int(rng.integers(0, 12))}.{int(rng.integers(0, 30))}.{int(rng.integers(0, 50))}', f'v{int(rng.integers(0, 12))}.{int(rng.integers(0, 30))}.{int(rng.integers(0, 50))}', f'{int(rng.integers(1, 12))}.{int(rng.integers(0, 12))}.{int(rng.integers(0, 12))}.{int(rng.integers(0, 9999))}'])

def _uuid(rng):
    return '-'.join((rstr(rng, '0123456789abcdef', n) for n in (8, 4, 4, 4, 12)))
FIELDS = {'name': (['nombre completo', 'nombre y apellidos', 'nombre', 'titular', 'cliente', 'contacto', 'paciente', 'empleado', 'destinatario', 'remitente', 'responsable', 'autor', 'alumno', 'solicitante'], ['full name', 'name', 'customer name', 'contact name', 'account holder', 'cardholder', 'patient', 'employee', 'recipient', 'sender', 'owner', 'author', 'applicant', 'student', 'guest'], lambda r, L, g: Val(VL.person(r, L, g)[0], 'PERSON_NAME')), 'first_name': (['nombre', 'nombre de pila'], ['first name', 'given name'], lambda r, L, g: Val(L.first(r, g), 'PERSON_NAME')), 'last_name': (['apellidos', 'primer apellido', 'apellido'], ['last name', 'surname', 'family name'], lambda r, L, g: Val(L.surname(r, g) + (' ' + L.surname(r, g) if g == 'es' and chance(r, 0.6) else ''), 'PERSON_NAME')), 'email': (['email', 'correo', 'correo electrónico', 'e-mail', 'mail', 'email de contacto'], ['email', 'e-mail', 'email address', 'mail', 'contact email', 'work email', 'personal email'], lambda r, L, g: Val(VL.email(r, L, g), 'EMAIL')), 'phone': (['teléfono', 'tel', 'tlf', 'móvil', 'teléfono móvil', 'teléfono de contacto', 'fax'], ['phone', 'phone number', 'tel', 'mobile', 'cell', 'fax', 'contact number', 'telephone'], lambda r, L, g: Val(VL.phone(r, g), 'PHONE')), 'iban': (['iban', 'cuenta', 'cuenta bancaria', 'número de cuenta', 'iban de cargo'], ['iban', 'bank account', 'account number', 'iban number'], lambda r, L, g: Val(VL.iban(r), 'IBAN')), 'card': (['tarjeta', 'número de tarjeta', 'tarjeta de crédito', 'pan'], ['card number', 'credit card', 'card', 'pan', 'cc number'], lambda r, L, g: Val(VL.credit_card(r), 'CREDIT_CARD')), 'ip': (['ip', 'dirección ip', 'ip del cliente', 'ip origen'], ['ip', 'ip address', 'client ip', 'remote addr', 'last login ip', 'source ip'], lambda r, L, g: Val(VL.ip(r), 'IP')), 'dni': (['dni', 'nif', 'nie', 'dni/nie', 'documento', 'documento de identidad', 'nº documento'], ['national id', 'id number', 'dni', 'nif', 'document number'], lambda r, L, g: Val(VL.dni_nie(r), 'DNI_NIE')), 'passport': (['pasaporte', 'número de pasaporte', 'nº pasaporte'], ['passport', 'passport number', 'passport no'], lambda r, L, g: Val(VL.passport(r), 'PASSPORT')), 'dob': (['fecha de nacimiento', 'f. nacimiento', 'nacimiento', 'fecha nac'], ['date of birth', 'dob', 'birth date', 'birthdate', 'birthday'], lambda r, L, g: Val(VL.date_of_birth(r, g, 'data' if chance(r, 0.5) else 'any'), 'DATE_OF_BIRTH')), 'address': (['dirección', 'domicilio', 'dirección postal', 'dirección de envío', 'dirección de facturación'], ['address', 'home address', 'street address', 'shipping address', 'billing address', 'mailing address'], lambda r, L, g: Val(VL.address(r, L, g, multiline=False), 'ADDRESS')), 'secret': (['contraseña', 'clave', 'clave api', 'token'], ['password', 'api key', 'token', 'secret', 'access token', 'client secret'], lambda r, L, g: Val(VL.secret_value(r) if chance(r, 0.6) else VL.secret_password(r), 'SECRET')), 'dsn': (['url de la base de datos', 'cadena de conexión', 'dsn'], ['database url', 'dsn', 'connection string', 'broker url', 'redis url', 'mongo uri'], lambda r, L, g: Val(*_urlc(r, L, g))), 'id': (['id', 'id cliente', 'id pedido', 'referencia', 'nº pedido', 'ticket', 'expediente'], ['id', 'user id', 'customer id', 'order id', 'ticket', 'reference', 'case id'], lambda r, L, g: Val(pick(r, [VL.phone_negative(r), VL.card_negative(r), _uuid(r), str(int(r.integers(1, 99999)))]))), 'created': (['fecha alta', 'fecha de alta', 'fecha pedido', 'fecha factura', 'último acceso', 'fecha de creación', 'caduca', 'fecha de vencimiento', 'fecha de contratación', 'fecha de emisión', 'fecha'], ['created at', 'updated at', 'last login', 'modified', 'timestamp', 'issued', 'expires', 'valid until', 'due date', 'hire date', 'shipped', 'date'], lambda r, L, g: Val(VL.date_other(r, g, 'data') if chance(r, 0.6) else VL.timestamp(r))), 'version': (['versión', 'build', 'firmware'], ['version', 'app version', 'build', 'firmware'], lambda r, L, g: Val(_version(r))), 'sku': (['ref', 'código', 'localizador', 'vuelo', 'código de producto'], ['sku', 'product code', 'locator', 'booking ref', 'flight'], lambda r, L, g: Val(VL.passport_negative(r))), 'imei': (['imei', 'nº de serie', 'id dispositivo'], ['imei', 'device id', 'serial'], lambda r, L, g: Val(VL.card_negative(r))), 'vat': (['cif', 'nif empresa', 'cif proveedor'], ['vat', 'tax id', 'vat number', 'company tax id'], lambda r, L, g: Val(pick(r, [VL.dni_negative(r), VL.iban_negative(r)]))), 'bic': (['bic', 'swift', 'bic/swift'], ['bic', 'swift', 'swift code'], lambda r, L, g: Val(VL.iban_negative(r))), 'amount': (['importe', 'precio', 'total', 'saldo'], ['amount', 'total', 'price', 'balance'], lambda r, L, g: Val(_money(r, g))), 'zip': (['cp', 'código postal'], ['zip', 'postal code', 'postcode'], lambda r, L, g: Val(rstr(r, VL.DIG, 5))), 'city': (['ciudad', 'localidad', 'municipio'], ['city', 'town'], lambda r, L, g: Val(L.municipality_es(r)[1] if g == 'es' else L.place_us(r)[0])), 'country': (['país', 'nacionalidad'], ['country', 'nationality'], lambda r, L, g: Val(pick(r, ['España', 'Spain', 'ES', 'México', 'Argentina', 'Francia', 'Portugal', 'USA', 'United Kingdom', 'Germany', 'ESP', 'Chile', 'Colombia']))), 'status': (['estado', 'rol', 'plan', 'tipo', 'sexo', 'género'], ['status', 'role', 'plan', 'type', 'gender', 'tier'], lambda r, L, g: Val(pick(r, ['active', 'activo', 'pending', 'pendiente', 'admin', 'user', 'premium', 'free', 'M', 'F', 'blocked', 'verified', 'cliente', 'proveedor', 'gold', 'trial']))), 'username': (['usuario', 'login', 'alias', 'nick'], ['username', 'login', 'user', 'handle', 'alias'], lambda r, L, g: Val(_username(r, L, g))), 'company': (['empresa', 'organización', 'razón social'], ['company', 'organization', 'employer'], lambda r, L, g: Val(company(r, L, g))), 'mac': (['mac', 'dirección mac'], ['mac', 'mac address', 'hwaddr'], lambda r, L, g: Val(VL.ip_negative(r))), 'hash': (['hash', 'checksum', 'hash contraseña'], ['hash', 'checksum', 'etag', 'sha256', 'digest', 'password hash'], lambda r, L, g: Val(rstr(r, '0123456789abcdef', pick(r, [32, 40, 64])))), 'expiry': (['caducidad', 'expira'], ['expiry', 'exp', 'valid thru'], lambda r, L, g: Val(VL.card_negative(r) if chance(r, 0.3) else f'{int(r.integers(1, 13)):02d}/{int(r.integers(24, 34))}')), 'host': (['servidor', 'host'], ['host', 'hostname', 'server', 'endpoint'], lambda r, L, g: Val(_hostname(r))), 'port': (['puerto'], ['port'], lambda r, L, g: Val(str(pick(r, [22, 80, 443, 3306, 5432, 6379, 8080, 8443, 9200, 27017])))), 'url': (['web', 'url', 'página'], ['url', 'website', 'homepage', 'callback url'], lambda r, L, g: Val(VL.url_negative(r))), 'age': (['edad', 'intentos', 'puntuación', 'cantidad'], ['age', 'attempts', 'score', 'quantity', 'count'], lambda r, L, g: Val(str(int(r.integers(0, 99)))))}
POSITIVE_KINDS = ('name', 'first_name', 'last_name', 'email', 'phone', 'iban', 'card', 'ip', 'dni', 'passport', 'dob', 'address', 'secret', 'dsn')

def _urlc(rng, L, lang):
    (url, (a, b)) = VL.url_with_credentials(rng, L, lang)
    return (url, 'URL_CREDENTIALS', (a, b))
SCHEMAS = {'customer': [('id', 0.9), ('name', 0.8), ('email', 0.8), ('phone', 0.6), ('dni', 0.4), ('dob', 0.35), ('address', 0.5), ('city', 0.3), ('zip', 0.2), ('iban', 0.25), ('created', 0.7), ('status', 0.5), ('country', 0.2)], 'employee': [('id', 0.8), ('first_name', 0.4), ('last_name', 0.4), ('name', 0.5), ('email', 0.8), ('phone', 0.5), ('dni', 0.5), ('passport', 0.15), ('dob', 0.5), ('iban', 0.4), ('created', 0.6), ('company', 0.3), ('age', 0.2)], 'order': [('id', 0.95), ('name', 0.6), ('email', 0.5), ('address', 0.6), ('phone', 0.4), ('amount', 0.9), ('created', 0.9), ('sku', 0.6), ('status', 0.6), ('zip', 0.2)], 'payment': [('id', 0.9), ('card', 0.7), ('expiry', 0.5), ('name', 0.5), ('amount', 0.9), ('ip', 0.4), ('created', 0.8), ('iban', 0.2), ('bic', 0.2), ('status', 0.5)], 'account': [('id', 0.8), ('username', 0.8), ('email', 0.7), ('hash', 0.4), ('secret', 0.25), ('ip', 0.5), ('created', 0.8), ('status', 0.5), ('url', 0.2)], 'server': [('host', 0.8), ('ip', 0.7), ('port', 0.5), ('email', 0.3), ('dsn', 0.4), ('secret', 0.3), ('version', 0.6), ('mac', 0.2), ('hash', 0.2)], 'traveler': [('sku', 0.6), ('name', 0.8), ('passport', 0.6), ('country', 0.5), ('dob', 0.5), ('email', 0.4), ('phone', 0.4), ('created', 0.5)], 'patient': [('id', 0.9), ('name', 0.8), ('dob', 0.7), ('phone', 0.5), ('address', 0.4), ('dni', 0.3), ('created', 0.7), ('age', 0.3)], 'device': [('imei', 0.8), ('mac', 0.6), ('version', 0.7), ('ip', 0.5), ('username', 0.3), ('created', 0.6)], 'supplier': [('company', 0.9), ('vat', 0.8), ('iban', 0.4), ('bic', 0.3), ('email', 0.6), ('phone', 0.5), ('address', 0.4), ('url', 0.3)]}

def key_of(rng, label, lang_key):
    words = VL.ascii_fold(label).lower().replace('/', ' ').replace('.', ' ').replace('º', '').split()
    if chance(rng, 0.7):
        words = [w for w in words if w not in ('de', 'del', 'la', 'el', 'of', 'the', 'n')] or words
    s = rng.random()
    if s < 0.42:
        return '_'.join(words)
    if s < 0.62:
        return words[0] + ''.join((w.capitalize() for w in words[1:]))
    if s < 0.68:
        return ''.join((w.capitalize() for w in words))
    if s < 0.74:
        return '-'.join(words)
    if s < 0.82:
        return '_'.join(words).upper()
    return label[:1].upper() + label[1:]

def record(rng, L, lang, schema=None, empty_frac=0.08, all_negative=False):
    schema = schema or pick(rng, list(SCHEMAS))
    klang = lang if chance(rng, 0.7) else 'en' if lang == 'es' else 'es'
    (out, used) = ([], set())
    for (kind, p) in SCHEMAS[schema]:
        if not chance(rng, p):
            continue
        (es_l, en_l, mk) = FIELDS[kind]
        label = pick(rng, es_l if klang == 'es' else en_l)
        key = key_of(rng, label, klang)
        if key in used:
            continue
        used.add(key)
        v = mk(rng, L, lang)
        if v.typ is not None and (all_negative or chance(rng, empty_frac)):
            v = Val(pick(rng, EMPTY))
        out.append((key, v))
    if not out:
        out.append(('id', Val(str(int(rng.integers(1, 99999))))))
    return out

def _q(rng, v, quote):
    return v.quoted(quote)

def render(rng, doc, recs, fmt):
    keys = []
    for r in recs:
        for (k, _) in r:
            if k not in keys:
                keys.append(k)
    if fmt == 'json':
        ind = pick(rng, [None, 2, 4])
        many = len(recs) > 1
        doc.add('[' if many else '')
        for (j, r) in enumerate(recs):
            sp = '\n' + ' ' * (ind or 0) if ind else ''
            doc.add(('\n  ' if many and ind else '') + '{')
            for (i, (k, v)) in enumerate(r):
                doc.add((sp if ind else '' if i == 0 else ' ') + json.dumps(k) + (': ' if ind or chance(rng, 0.7) else ':'))
                doc.value(_q(rng, v, '"') if not (v.typ is None and v.text in ('null', '0')) else v)
                doc.add(',' if i < len(r) - 1 else '')
            doc.add(('\n' if ind else '') + '}' + (',' if many and j < len(recs) - 1 else ''))
        doc.add('\n]\n' if many else '\n')
    elif fmt == 'yaml':
        for r in recs:
            first = True
            for (k, v) in r:
                pre = '- ' if first and len(recs) > 1 else '  ' if len(recs) > 1 else ''
                doc.add(f'{pre}{k}: ')
                doc.value(_q(rng, v, pick(rng, ['"', "'", ''])) if v.text else Val('""'))
                doc.add('\n')
                first = False
    elif fmt in ('csv', 'tsv'):
        sep = '\t' if fmt == 'tsv' else pick(rng, [',', ';', ','])
        qq = chance(rng, 0.4)
        doc.add(sep.join(keys) + '\n')
        for r in recs:
            d = dict(r)
            for (i, k) in enumerate(keys):
                v = d.get(k, Val(''))
                doc.value(_q(rng, v, '"') if qq or sep in v.text or '\n' in v.text else v)
                doc.add(sep if i < len(keys) - 1 else '\n')
    elif fmt == 'sql':
        tbl = pick(rng, ['clientes', 'customers', 'users', 'usuarios', 'empleados', 'employees', 'orders', 'pedidos', 'contacts', 'pacientes', 'accounts', 'suppliers', 'proveedores'])
        cols = [k.lower().replace('-', '_').replace(' ', '_') for k in keys]
        doc.add(f"INSERT INTO {tbl} ({', '.join(cols)}) VALUES\n" if len(recs) > 1 else f"INSERT INTO {tbl} ({', '.join(cols)}) VALUES ")
        for (j, r) in enumerate(recs):
            d = dict(r)
            doc.add('  (' if len(recs) > 1 else '(')
            for (i, k) in enumerate(keys):
                v = d.get(k, Val('NULL'))
                doc.value(_q(rng, v, "'") if v.text not in ('NULL', 'null') else Val('NULL'))
                doc.add(', ' if i < len(keys) - 1 else '')
            doc.add(')' + (',' if j < len(recs) - 1 else ';') + '\n')
    elif fmt in ('python', 'js', 'ruby', 'php'):
        name = pick(rng, ['customer', 'user', 'data', 'payload', 'record', 'contacto', 'cliente', 'row', 'fixture', 'sample', 'item'])
        kq = pick(rng, ['"', "'"])
        head = {'python': f'{name} = ', 'js': f"{pick(rng, ['const', 'let', 'var'])} {name} = ", 'ruby': f'{name} = ', 'php': f'${name} = '}[fmt]
        many = len(recs) > 1
        doc.add(head + ('[' if many else '') + ('\n' if many else ''))
        for (j, r) in enumerate(recs):
            doc.add(('    ' if many else '') + ('array(' if fmt == 'php' and chance(rng, 0.3) else '[' if fmt == 'php' else '{'))
            for (i, (k, v)) in enumerate(r):
                kk = k if fmt == 'js' and chance(rng, 0.5) and k.replace('_', '').isalnum() and (not k[0].isdigit()) else kq + k + kq
                sep = {'python': ': ', 'js': ': ', 'ruby': ' => ', 'php': ' => '}[fmt]
                doc.add(('\n        ' if not many else ' ') + kk + sep)
                doc.value(_q(rng, v, kq))
                doc.add(',' if i < len(r) - 1 else '')
            doc.add(('\n' if not many else ' ') + (']' if fmt == 'php' else '}') + (',' if many and j < len(recs) - 1 else '') + ('\n' if many else ''))
        doc.add((']' if many else '') + (';' if fmt in ('js', 'php') else '') + '\n')
    elif fmt == 'java':
        cls = pick(rng, ['Customer', 'User', 'Employee', 'Contact', 'Cliente', 'Person'])
        doc.add(f'{cls} {cls.lower()} = {cls}.builder()\n')
        for (k, v) in recs[0]:
            m = k[0].lower() + k[1:].replace('_', '').replace('-', '').replace(' ', '')
            doc.add(f'        .{m}(')
            doc.value(_q(rng, v, '"'))
            doc.add(')\n')
        doc.add('        .build();\n')
    elif fmt == 'go':
        cls = pick(rng, ['Customer', 'User', 'Contact', 'Account'])
        doc.add(f"{pick(rng, ['c', 'u', 'x'])} := {cls}{{\n")
        for (k, v) in recs[0]:
            f = ''.join((w.capitalize() for w in k.replace('-', '_').replace(' ', '_').split('_')))
            doc.add(f'\t{f}: ')
            doc.value(_q(rng, v, '"'))
            doc.add(',\n')
        doc.add('}\n')
    elif fmt == 'xml':
        tag = pick(rng, ['customer', 'cliente', 'user', 'contact', 'record', 'employee'])
        for r in recs:
            doc.add(f'<{tag}>\n')
            for (k, v) in r:
                t = k.replace(' ', '_')
                t = t if t[:1].isalpha() else 'f_' + t
                doc.add(f'  <{t}>')
                doc.value(v)
                doc.add(f'</{t}>\n')
            doc.add(f'</{tag}>\n')
    elif fmt in ('env', 'ini', 'toml'):
        if fmt != 'env' and chance(rng, 0.6):
            doc.add(f"[{pick(rng, ['contact', 'admin', 'owner', 'database', 'billing', 'app', 'soporte'])}]\n")
        for (k, v) in recs[0]:
            kk = k.upper().replace('-', '_').replace(' ', '_') if fmt == 'env' else k.replace(' ', '_')
            doc.add(kk + (pick(rng, ['=', ' = ']) if fmt != 'env' else '='))
            doc.value(_q(rng, v, pick(rng, ['"', '', "'"]) if fmt != 'toml' else '"') if v.text else Val('""'))
            doc.add('\n')
    elif fmt == 'form':
        for r in recs:
            for (k, v) in r:
                lab = k.replace('_', ' ').replace('-', ' ')
                lab = lab[:1].upper() + lab[1:]
                doc.add(lab + pick(rng, [': ', ': ', ':\t', ' - ', ' = ']))
                doc.value(v)
                doc.add('\n')
            doc.add('\n')
    elif fmt == 'markdown':
        doc.add('| ' + ' | '.join(keys) + ' |\n|' + '---|' * len(keys) + '\n')
        for r in recs:
            d = dict(r)
            doc.add('| ')
            for k in keys:
                doc.value(d.get(k, Val('')))
                doc.add(' | ')
            doc.add('\n')
    elif fmt == 'ldif':
        for r in recs:
            d = dict(r)
            doc.add(f"dn: uid={_uuid(rng)[:8]},ou=people,dc={pick(rng, ['example', 'empresa', 'corp'])},dc=com\n")
            for (k, v) in r:
                doc.add(f"{k.replace(' ', '')}: ")
                doc.value(v)
                doc.add('\n')
            doc.add('\n')
    elif fmt == 'vcard':
        for r in recs:
            doc.add('BEGIN:VCARD\nVERSION:3.0\n')
            for (k, v) in r:
                tagv = {'PERSON_NAME': 'FN', 'EMAIL': 'EMAIL;TYPE=INTERNET', 'PHONE': 'TEL;TYPE=CELL', 'ADDRESS': 'ADR;TYPE=HOME:;;', 'DATE_OF_BIRTH': 'BDAY'}.get(v.typ, 'NOTE')
                doc.add(tagv + ('' if tagv.endswith(';;') else ':'))
                doc.value(v)
                doc.add('\n')
            doc.add('END:VCARD\n')
    elif fmt == 'kv':
        for r in recs:
            doc.add(VL.timestamp(rng) + ' ' + pick(rng, ['INFO', 'DEBUG', 'WARN', 'ERROR']) + ' ')
            for (k, v) in r:
                doc.add(k.replace(' ', '_') + '=')
                doc.value(_q(rng, v, '"') if ' ' in v.text else v)
                doc.add(' ')
            doc.add('\n')
    else:
        raise ValueError(fmt)
    return doc
RENDERERS = ('json', 'json', 'json', 'yaml', 'yaml', 'csv', 'csv', 'tsv', 'sql', 'sql', 'python', 'python', 'js', 'js', 'ruby', 'java', 'go', 'xml', 'env', 'ini', 'form', 'form', 'vcard', 'kv', 'kv', 'php', 'markdown', 'ldif', 'toml')

def renderer(rng, held_out=False):
    if held_out and chance(rng, 0.5):
        return pick(rng, HELD_OUT_RENDERERS)
    return pick(rng, [r for r in RENDERERS if r not in HELD_OUT_RENDERERS])
_CRED_LANG = {'python': 'python', 'js': 'js', 'java': 'java', 'go': 'go', 'ruby': 'ruby', 'php': 'php', 'yaml': 'yaml', 'json': 'json', 'shell': 'shell', 'ini': 'ini', 'xml': 'xml', 'sql': 'sql', 'c': None}

def secret_block(rng, doc, positive, hint=None):
    import re as _re
    from . import values as _VL
    (line, (a, b)) = _VL.secret_line(rng, positive, _CRED_LANG.get(hint))
    if not positive:
        return doc.add(line + '\n')
    val = line[a:b]
    m = _re.match('[A-Za-z][A-Za-z0-9+.\\-]*://([^\\s:/@\'\\"`<>]+:[^\\s/@\'\\"`<>]+)@', val)
    if m:
        return doc.add(line[:a + m.start(1)]).add(m.group(1), 'URL_CREDENTIALS').add(line[a + m.end(1):] + '\n')
    return doc.add(line[:a]).add(val, 'SECRET').add(line[b:] + '\n')

def guess_hint(bg):
    s = bg[:2048]
    score = {'python': s.count(b'def ') * 2 + s.count(b'import ') + s.count(b'self.') * 2 + s.count(b'elif '), 'js': s.count(b'const ') * 2 + s.count(b'function') + s.count(b'=> ') * 2 + s.count(b'let '), 'java': s.count(b'public ') * 2 + s.count(b'private ') * 2 + s.count(b'new ') + s.count(b'@Override') * 2, 'go': s.count(b'func ') * 3 + s.count(b':= ') * 2 + s.count(b'package '), 'ruby': s.count(b'\nend') * 2 + s.count(b'do |') * 3 + s.count(b"require '"), 'php': s.count(b'<?php') * 5 + s.count(b'$this->') * 2 + s.count(b'function '), 'yaml': s.count(b':\n') + s.count(b'\n- ') * 2 + s.count(b'\n  - '), 'json': s.count(b'": ') * 2 + s.count(b'",\n'), 'sql': s.upper().count(b'SELECT ') * 2 + s.upper().count(b'INSERT ') * 2 + s.upper().count(b'CREATE TABLE') * 3, 'shell': s.count(b'echo ') * 2 + s.count(b'export ') * 2 + s.count(b'#!/bin') * 5 + s.count(b'fi\n') * 2, 'c': s.count(b'#include') * 3 + s.count(b'->') + s.count(b'static ') + s.count(b'int ')}
    best = max(score, key=score.get)
    return best if score[best] >= 3 else None
_COMMENT = {'python': '# ', 'ruby': '# ', 'yaml': '# ', 'shell': '# ', 'ini': '; ', 'js': '// ', 'java': '// ', 'go': '// ', 'php': '// ', 'c': '// ', 'sql': '-- ', 'json': '// '}

def code_snippet(rng, L, lang, doc, positive=True, hint=None):
    q = pick(rng, ['"', "'"])
    per = VL.person(rng, L, lang)
    name_v = Val(per[0], 'PERSON_NAME') if positive else Val(company(rng, L, lang))
    mail_v = Val(VL.email(rng, L, lang, per[1], per[2]), 'EMAIL') if positive else Val(VL.email_negative(rng))
    p = rng.random()
    if p < 0.16:
        c = _COMMENT[hint] if hint in _COMMENT and chance(rng, 0.75) else pick(rng, ['# ', '// ', ' * ', '-- ', ';; '])
        form = rng.random()
        if form < 0.4:
            doc.add(f'{c}Copyright (c) {int(rng.integers(2000, 2027))} ').value(name_v).add(' <').value(mail_v).add('>\n')
        elif form < 0.6:
            doc.add(f'{c}@author ').value(name_v).add('\n')
        elif form < 0.8:
            doc.add(f"{c}{pick(rng, ['Author', 'Maintainer', 'Autor', 'Contact', 'Contacto'])}: ").value(name_v).add(' <').value(mail_v).add('>\n')
        else:
            doc.add(f"{pick(rng, ['Signed-off-by', 'Co-authored-by', 'Reviewed-by'])}: ").value(name_v).add(' <').value(mail_v).add('>\n')
    elif p < 0.26:
        form = rng.random()
        if form < 0.35:
            doc.add('  "author": "').value(name_v).add(' <').value(mail_v).add('>",\n')
        elif form < 0.65:
            doc.add('    author=').add(q).value(name_v).add(q).add(',\n    author_email=').add(q).value(mail_v).add(q).add(',\n')
        else:
            doc.add('authors = ["').value(name_v).add(' <').value(mail_v).add('>"]\n')
    elif p < 0.44:
        kind = pick(rng, ['phone', 'email', 'iban', 'card', 'dni', 'ip', 'dob', 'address', 'passport'])
        v = FIELDS[kind][2](rng, L, lang) if positive else Val(pick(rng, [VL.phone_negative(rng), VL.card_negative(rng), VL.ip_negative(rng), VL.date_other(rng, lang), VL.dni_negative(rng), VL.passport_negative(rng)]))
        fn = pick(rng, {'phone': ['send_sms', 'sendSms', 'normalize_phone', 'validar_telefono', 'call'], 'email': ['send_mail', 'sendEmail', 'notify', 'enviar_correo', 'invite'], 'iban': ['validate_iban', 'validarIban', 'charge_account', 'sepa_debit'], 'card': ['charge', 'tokenize_card', 'luhn_check', 'processPayment'], 'dni': ['validar_dni', 'check_nif', 'buscar_cliente'], 'ip': ['ban_ip', 'allow', 'geoip.lookup', 'whitelist'], 'dob': ['parse_birthdate', 'calcular_edad', 'age_from_dob'], 'address': ['geocode', 'normalizar_direccion'], 'passport': ['verify_passport', 'checkin']}[kind])
        pre = pick(rng, ['', 'assert ', 'result = ', 'await ', 'return ', 'expect(', 'self.assertTrue('])
        post = {'expect(': ')).toBeTruthy();', 'self.assertTrue(': '))'}.get(pre, ')')
        doc.add(f'{pre}{fn}({q}').value(v).add(f'{q}{post}\n')
    elif p < 0.58:
        ip_val = Val(VL.ip(rng), 'IP') if positive else Val(VL.ip_negative(rng))
        lg = pick(rng, ['logger.info', 'log.debug', 'console.log', 'logging.warning', 'log.Printf', 'print'])
        doc.add(f"{lg}({q}{pick(rng, ['login ok for', 'usuario autenticado', 'sent invoice to', 'password reset for', 'contacting'])} ")
        doc.value(mail_v).add(pick(rng, [' from ', ' desde ', ' ip='])).value(ip_val).add(f'{q})\n')
    elif p < 0.72:
        kind = pick(rng, ['email', 'phone', 'dsn', 'dsn', 'ip', 'secret'])
        v = FIELDS[kind][2](rng, L, lang) if positive else Val(VL.url_negative(rng)) if kind == 'dsn' else Val(VL.secret_negative(rng))
        key = pick(rng, {'email': ['ADMIN_EMAIL', 'SUPPORT_EMAIL', 'DEFAULT_FROM_EMAIL', 'alert_email', 'notify.to'], 'phone': ['SUPPORT_PHONE', 'ALERT_SMS_TO', 'on_call_phone', 'telefono_guardia'], 'dsn': ['DATABASE_URL', 'REDIS_URL', 'CELERY_BROKER_URL', 'MONGO_URI', 'SQLALCHEMY_DATABASE_URI'], 'ip': ['ALLOWED_HOST', 'bind_address', 'SERVER_IP', 'upstream', 'nameserver'], 'secret': ['API_KEY', 'SECRET_KEY', 'password', 'token', 'STRIPE_SECRET']}[kind])
        sep = pick(rng, ['=', ': ', ' = ', '='])
        doc.add(key + sep + (q if chance(rng, 0.4) else '')).value(v).add('\n')
    elif p < 0.84:
        fmt = hint if hint in ('python', 'js', 'sql', 'yaml', 'json', 'java', 'go', 'ruby') and chance(rng, 0.7) else pick(rng, ['python', 'js', 'sql', 'yaml', 'json', 'java', 'go', 'ruby'])
        render(rng, doc, [record(rng, L, lang, all_negative=not positive) for _ in range(int(rng.integers(1, 3)))], fmt)
    elif p < 0.93:
        secret_block(rng, doc, positive, hint)
    else:
        c = _COMMENT[hint] if hint in _COMMENT and chance(rng, 0.75) else pick(rng, ['# ', '// ', '-- '])
        if positive:
            doc.add(c + pick(rng, ['TODO: ', 'FIXME: ', 'NOTE: ', 'Nota: ', '']) + pick(rng, ['ask ', 'llamar a ', 'contact ', 'hablar con '])).value(name_v)
            doc.add(pick(rng, [' (', ' al ', ' at ', ' - '])).value(Val(VL.phone(rng, lang), 'PHONE') if chance(rng, 0.5) else mail_v)
            doc.add(pick(rng, [')', '', ' about the migration', ' sobre la factura']) + '\n')
        else:
            doc.add(c + pick(rng, ['bump to ', 'actualizado a ', 'fixed in ', 'see ticket ']) + pick(rng, [_version(rng), VL.phone_negative(rng), VL.passport_negative(rng)]) + '\n')
    return doc
PROSE_POS = {'es': {'PHONE': ['Puedes llamarme al {} cualquier tarde.', 'Mi número es el {}.', 'Llámame al {} si hay algún problema.', 'El teléfono de contacto es {}.', 'Te he dejado un mensaje, mi móvil es {}.'], 'EMAIL': ['Para cualquier duda, escribe a {}.', 'Envíame los documentos a {}, por favor.', 'Mi correo es {}.', 'Responde a {} antes del viernes.'], 'ADDRESS': ['Te dejo mi dirección: {}.', 'Vivo en {} desde hace años.', 'Envíe el paquete a {}.', 'La nueva oficina está en {}.'], 'DATE_OF_BIRTH': ['Nací el {}.', 'Su fecha de nacimiento es {}.', 'Nacida el {}, tiene doble nacionalidad.', 'Fecha de nacimiento del titular: {}.'], 'DNI_NIE': ['Mi DNI es {}.', 'Adjunto copia del documento {}.', 'El titular, con NIE {}, solicita la baja.', 'Con DNI número {}, declaro que los datos son ciertos.'], 'PASSPORT': ['El número de pasaporte que figura en la reserva es {}.', 'Viajo con el pasaporte {}.', 'Pasaporte nº {}, válido hasta 2031.'], 'IBAN': ['Haz la transferencia a la cuenta {}.', 'El recibo se domiciliará en el IBAN {}.', 'Mi IBAN es {}.'], 'CREDIT_CARD': ['Pagué con la tarjeta {} y me cobraron dos veces.', 'La tarjeta terminada en ... no, la {}, es la buena.', 'Cargo en tarjeta {}.'], 'IP': ['El acceso se hizo desde la IP {}.', 'Bloquea la dirección {} en el cortafuegos.', 'La conexión venía de {}.'], 'SECRET': ['La contraseña del wifi es {}.', 'Usa la clave {} para entrar.', 'El token de acceso es {}.'], 'PERSON_NAME': ['{} me pidió que le reenviara el informe.', 'Hablé con {} ayer por la tarde.', 'La responsable del proyecto es {}.', 'Atentamente, {}.']}, 'en': {'PHONE': ['You can call me on {} any afternoon.', 'My number is {}.', 'Call me at {} if anything goes wrong.', 'The contact phone is {}.', 'Text me at {}.'], 'EMAIL': ['For any question, write to {}.', 'Please send the documents to {}.', 'My email is {}.', 'Reply to {} before Friday.'], 'ADDRESS': ['Here is my address: {}.', 'I have lived at {} for years.', 'Ship the parcel to {}.', 'The new office is at {}.'], 'DATE_OF_BIRTH': ['I was born on {}.', 'Her date of birth is {}.', 'Born {}, she holds dual citizenship.', "Account holder's date of birth: {}."], 'DNI_NIE': ['My Spanish ID (DNI) is {}.', 'Please find attached a copy of NIE {}.'], 'PASSPORT': ['The passport number on the booking is {}.', 'I travel with passport {}.', 'Passport no. {}, valid until 2031.'], 'IBAN': ['Please transfer the amount to {}.', 'The direct debit will be charged to IBAN {}.', 'My IBAN is {}.'], 'CREDIT_CARD': ['I paid with card {} and was charged twice.', 'Charge it to {}, please.', 'Card on file: {}.'], 'IP': ['The login came from IP {}.', 'Block {} in the firewall.', 'The connection originated at {}.'], 'SECRET': ['The wifi password is {}.', 'Use the key {} to log in.', 'The access token is {}.'], 'PERSON_NAME': ['{} asked me to forward the report.', 'I spoke with {} yesterday afternoon.', 'The project lead is {}.', 'Kind regards, {}.']}}
PROSE_NEG = {'es': ['El pedido {ORDER} se envió el {DATE}.', 'La versión {VERSION} corrige el fallo del login.', 'La factura {REF} asciende a {AMOUNT} euros.', 'La reunión es el {DATE} a las 10:00.', 'El código de producto es {SKU}.', 'Actualiza a la versión {VERSION} antes del {DATE}.', 'Su vuelo {FLIGHT} sale a las 7:45.', 'El expediente {REF} se cerró el {DATE}.', 'La calle estaba cortada por obras.', 'El servidor {HOST} responde en el puerto {PORT}.'], 'en': ['Order {ORDER} shipped on {DATE}.', 'Version {VERSION} fixes the login bug.', 'Invoice {REF} totals {AMOUNT} dollars.', 'The meeting is on {DATE} at 10:00.', 'The product code is {SKU}.', 'Upgrade to {VERSION} before {DATE}.', 'Your flight {FLIGHT} leaves at 7:45.', 'Case {REF} was closed on {DATE}.', 'The street was closed for works.', 'Server {HOST} answers on port {PORT}.']}

def _neg_fill(rng, L, lang, tpl):
    return tpl.format(ORDER=VL.phone_negative(rng), DATE=VL.date_other(rng, lang), VERSION=_version(rng), REF=pick(rng, [VL.iban_negative(rng), VL.passport_negative(rng), VL.phone_negative(rng)]), AMOUNT=_money(rng, lang), SKU=VL.passport_negative(rng), FLIGHT=VL.passport_negative(rng), HOST=_hostname(rng), PORT=str(int(rng.integers(20, 9000))))

def value_of(rng, L, lang, typ):
    kind = {'PHONE': 'phone', 'EMAIL': 'email', 'ADDRESS': 'address', 'DATE_OF_BIRTH': 'dob', 'DNI_NIE': 'dni', 'PASSPORT': 'passport', 'IBAN': 'iban', 'CREDIT_CARD': 'card', 'IP': 'ip', 'SECRET': 'secret', 'PERSON_NAME': 'name', 'URL_CREDENTIALS': 'dsn'}[typ]
    return FIELDS[kind][2](rng, L, lang)

def sentence(rng, L, lang, doc, want=None, positive=True):
    p = rng.random()
    if want is None and p < 0.55:
        kind = 'holes' if positive and chance(rng, 0.45) else 'clean'
        s = L.sentence(rng, lang, kind)
        parts = s.split('{P}')
        for (i, part) in enumerate(parts):
            doc.add(part)
            if i < len(parts) - 1:
                full = VL.person(rng, L, lang)
                txt = full[0] if chance(rng, 0.35) else full[1]
                doc.add(txt, 'PERSON_NAME')
        return doc.add(' ')
    if not positive or want == 'NEG':
        return doc.add(_neg_fill(rng, L, lang, pick(rng, PROSE_NEG[lang])) + ' ')
    typ = want or pick(rng, list(PROSE_POS[lang]))
    tpl = pick(rng, PROSE_POS[lang][typ])
    (a, b) = tpl.split('{}')
    v = value_of(rng, L, lang, typ)
    if typ == 'ADDRESS':
        v = Val(VL.address(rng, L, lang, multiline=False), 'ADDRESS')
    doc.add(a).value(v).add(b + ' ')
    return doc

def paragraph(rng, L, lang, doc, n=None, positive=True, types=None):
    n = n or int(rng.integers(2, 6))
    for i in range(n):
        want = pick(rng, types) if types and chance(rng, 0.6) else None
        sentence(rng, L, lang, doc, want, positive)
    return doc.add('\n\n' if chance(rng, 0.7) else '\n')

def email_message(rng, L, lang, doc, positive=True, types=None):
    (me, you) = (VL.person(rng, L, lang), VL.person(rng, L, lang))
    (my_mail, your_mail) = (VL.email(rng, L, lang, me[1], me[2]), VL.email(rng, L, lang, you[1], you[2]))
    if not positive:
        roles = {'es': ['Equipo de soporte', 'Departamento de facturación', 'Atención al cliente', 'Recursos Humanos', 'Soporte'], 'en': ['Support Team', 'Billing Department', 'Customer Service', 'HR Team', 'The Team']}[lang]
        me = (pick(rng, roles), pick(rng, roles), company(rng, L, lang), 'm')
        you = (company(rng, L, lang), pick(rng, roles), pick(rng, roles), 'm')
    P = (lambda t, typ: Val(t, typ)) if positive else lambda t, typ: Val(t)
    if chance(rng, 0.7):
        doc.add('From: ').value(P(me[0], 'PERSON_NAME')).add(' <').value(P(my_mail, 'EMAIL') if positive else Val(VL.email_negative(rng))).add('>\n')
        doc.add('To: ').value(P(you[0], 'PERSON_NAME')).add(' <').value(P(your_mail, 'EMAIL') if positive else Val(VL.email_negative(rng))).add('>\n')
        if chance(rng, 0.3):
            doc.add('Cc: ').value(P(VL.email(rng, L, lang), 'EMAIL') if positive else Val(VL.email_negative(rng))).add('\n')
        doc.add(f"Date: {pick(rng, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])}, {VL.date_other(rng, 'en')} {int(rng.integers(7, 20)):02d}:{int(rng.integers(0, 60)):02d} +0{int(rng.integers(0, 3))}00\n")
        doc.add('Subject: ' + pick(rng, {'es': ['Re: factura pendiente', 'Documentación', 'Reserva confirmada', 'Cambio de datos', 'Consulta', 'Fwd: contrato'], 'en': ['Re: pending invoice', 'Documents', 'Booking confirmed', 'Change of details', 'Question', 'Fwd: contract']}[lang]) + '\n\n')
    greet = pick(rng, {'es': ['Hola ', 'Buenos días, ', 'Estimado ', 'Estimada ', 'Querido ', 'Hola de nuevo, '], 'en': ['Hi ', 'Hello ', 'Dear ', 'Good morning ', 'Hey ']}[lang])
    if chance(rng, 0.3):
        doc.add(greet + pick(rng, VL.TITLES[lang]) + ' ').value(P(you[2], 'PERSON_NAME')).add(':\n\n' if lang == 'es' else ',\n\n')
    else:
        doc.add(greet).value(P(you[1], 'PERSON_NAME')).add(',\n\n')
    paragraph(rng, L, lang, doc, positive=positive, types=types)
    doc.add(pick(rng, {'es': ['Un saludo,', 'Saludos,', 'Atentamente,', 'Gracias,'], 'en': ['Best,', 'Regards,', 'Thanks,', 'Kind regards,']}[lang]) + '\n')
    doc.value(P(me[0], 'PERSON_NAME')).add('\n')
    if chance(rng, 0.6):
        doc.add(company(rng, L, lang) + '\n')
        if chance(rng, 0.7):
            doc.add(pick(rng, ['Tel: ', 'Tel. ', 'Móvil: ', 'Phone: ', 'T. ', ''])).value(P(VL.phone(rng, lang), 'PHONE') if positive else Val(VL.phone_negative(rng))).add('\n')
        if chance(rng, 0.4):
            doc.value(P(VL.address(rng, L, lang), 'ADDRESS') if positive else Val(VL.address_negative(rng, lang))).add('\n')
    return doc.add('\n')
_PATHS = ['/api/v1/users', '/login', '/checkout', '/static/app.js', '/api/orders/{n}', '/health', '/es/contacto', '/api/v2/payments', '/wp-login.php', '/favicon.ico', '/api/customers/{n}/profile', '/metrics']
_UA = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36', 'curl/8.5.0', 'python-requests/2.32.3', 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X)', 'Go-http-client/1.1', 'kube-probe/1.29']

def log_line(rng, L, lang, doc, positive=True):
    p = rng.random()
    ipv = Val(VL.ip(rng), 'IP') if positive else Val(VL.ip_negative(rng))
    if p < 0.3:
        path = pick(rng, _PATHS).replace('{n}', str(int(rng.integers(1, 99999))))
        doc.value(ipv if chance(rng, 0.9) else Val(VL.ip(rng), 'IP') if positive else ipv)
        doc.add(f''' - {pick(rng, ['-', '-', _username(rng, L, lang)])} [{int(rng.integers(1, 29)):02d}/{pick(rng, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])}/{int(rng.integers(2019, 2027))}:{int(rng.integers(0, 24)):02d}:{int(rng.integers(0, 60)):02d}:{int(rng.integers(0, 60)):02d} +0000] "{pick(rng, ['GET', 'POST', 'PUT', 'DELETE'])} {path} HTTP/1.1" {pick(rng, [200, 200, 201, 204, 301, 302, 400, 401, 403, 404, 500])} {int(rng.integers(0, 90000))} "-" "{pick(rng, _UA)}"\n''')
    elif p < 0.62:
        doc.add(VL.timestamp(rng) + ' ' + pick(rng, ['INFO', 'WARN', 'ERROR', 'DEBUG']) + f" [{pick(rng, ['auth', 'billing', 'mailer', 'api', 'sms', 'kyc'])}] ")
        kind = pick(rng, ['email', 'phone', 'card', 'iban', 'name', 'dni', 'dsn', 'dob'])
        v = FIELDS[kind][2](rng, L, lang) if positive else Val(pick(rng, [VL.phone_negative(rng), VL.card_negative(rng), VL.email_negative(rng), VL.date_other(rng, lang)]))
        msg = pick(rng, {'email': ['login ok user=', 'password reset requested for ', 'bounce from ', 'sent welcome mail to '], 'phone': ['sms delivered to ', 'otp sent to ', 'call from '], 'card': ['payment declined card=', 'charging '], 'iban': ['sepa mandate created iban=', 'invalid IBAN '], 'name': ['profile updated for ', 'new customer: '], 'dni': ['kyc ok document=', 'duplicate NIF '], 'dsn': ['connecting to ', 'db pool url='], 'dob': ['age check dob=', 'birthdate mismatch ']}[kind])
        doc.add(msg).value(v).add(f' ip=').value(ipv).add(f' request_id={_uuid(rng)} latency_ms={int(rng.integers(1, 3000))}\n')
    elif p < 0.8:
        rec = [('ts', Val(VL.timestamp(rng))), ('level', Val(pick(rng, ['info', 'warn', 'error']))), ('msg', Val(pick(rng, ['user login', 'payment', 'signup', 'update profile', 'export'])))] + record(rng, L, lang, pick(rng, ['account', 'payment', 'customer']), all_negative=not positive)[:4]
        render(rng, doc, [rec], 'json')
    else:
        doc.add(f"{pick(rng, ['Jan', 'Mar', 'Sep', 'Oct', 'Dec'])} {int(rng.integers(1, 29)):2d} {int(rng.integers(0, 24)):02d}:{int(rng.integers(0, 60)):02d}:{int(rng.integers(0, 60)):02d} {_hostname(rng).split('.')[0]} sshd[{int(rng.integers(100, 65000))}]: {pick(rng, ['Accepted publickey', 'Failed password', 'Invalid user', 'Accepted password'])} for {_username(rng, L, lang)} from ")
        doc.value(ipv).add(f' port {int(rng.integers(1024, 65535))} ssh2\n')
    return doc
