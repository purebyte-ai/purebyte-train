"""Checksums and syntax checks of the structured types.

Used by the generator (every planted positive is VALID), by the rules baseline (a rule without its checksum would be a
strawman rival) and by any hybrid redaction that confirms a structured type. Luhn (cards), ISO 13616 mod-97 (IBAN), the
DNI/NIE control letter, the CIF control character (company ids: a look-alike of DNI_NIE), IPv4/IPv6 through `ipaddress`
with the non-identifying addresses excluded, and e-mail syntax with a top-level domain that exists.
"""
import ipaddress
import re
DNI_LETTERS = 'TRWAGMYFPDXBNJZSQVHLCKE'
IBAN_LENGTHS = dict(AD=24, AE=23, AL=28, AT=20, AZ=28, BA=20, BE=16, BG=22, BH=22, BR=29, BY=28, CH=21, CR=22, CY=28, CZ=24, DE=22, DK=18, DO=28, EE=20, EG=29, ES=24, FI=18, FO=18, FR=27, GB=22, GE=22, GI=23, GL=18, GR=27, GT=28, HR=21, HU=28, IE=22, IL=23, IQ=23, IS=26, IT=27, JO=30, KW=30, KZ=20, LB=28, LC=32, LI=21, LT=20, LU=20, LV=21, LY=25, MC=27, MD=24, ME=22, MK=19, MR=27, MT=31, MU=30, NL=18, NO=15, PK=24, PL=28, PS=29, PT=25, QA=29, RO=24, RS=22, SA=24, SC=31, SD=18, SE=24, SI=19, SK=24, SM=27, ST=25, SV=28, TL=23, TN=24, TR=26, UA=29, VA=22, VG=24, XK=20)
_CC = 'ac ad ae af ag ai al am ao aq ar as at au aw ax az ba bb bd be bf bg bh bi bj bm bn bo bq br bs bt bw by bz ca cc cd cf cg ch\nci ck cl cm cn co cr cu cv cw cx cy cz de dj dk dm do dz ec ee eg er es et eu fi fj fk fm fo fr ga gb gd ge gf gg gh gi gl gm gn gp\ngq gr gs gt gu gw gy hk hm hn hr ht hu id ie il im in io iq ir is it je jm jo jp ke kg kh ki km kn kp kr kw ky kz la lb lc li lk lr\nls lt lu lv ly ma mc md me mg mh mk ml mm mn mo mp mq mr ms mt mu mv mw mx my mz na nc ne nf ng ni nl no np nr nu nz om pa pe pf pg\nph pk pl pm pn pr ps pt pw py qa re ro rs ru rw sa sb sc sd se sg sh si sk sl sm sn so sr ss st su sv sx sy sz tc td tf tg th tj tk\ntl tm tn to tr tt tv tw tz ua ug uk us uy uz va vc ve vg vi vn vu wf ws ye yt za zm zw'
_G = 'com net org edu gov mil int info biz name pro aero coop museum mobi asia tel travel jobs cat post xxx app dev ai io co me\nxyz online site tech store shop blog cloud email live news space website club top vip life world today digital agency company\nsolutions services systems network group global media studio design art photo pics link click page one plus team zone center\ncity london berlin paris madrid barcelona nyc tokyo amsterdam eus gal bcn africa lat moe ninja rocks run fun games software\ntools works expert guru academy school university health care clinic legal law bank finance money capital insure ventures\npartners consulting support help social chat mail inc llc ltd gmbh srl sas corp limited foundation org ngo'
TLDS = frozenset(_CC.split() + _G.split())

def digits(s):
    return re.sub('\\D', '', s)

def luhn_ok(num):
    d = digits(num)
    if not 12 <= len(d) <= 19:
        return False
    tot = 0
    for (i, c) in enumerate(reversed(d)):
        x = int(c)
        if i % 2 == 1:
            x *= 2
            if x > 9:
                x -= 9
        tot += x
    return tot % 10 == 0

def luhn_complete(prefix_digits):
    for d in '0123456789':
        if luhn_ok(prefix_digits + d):
            return d
    raise AssertionError('unreachable')

def _iban_num(s):
    return int(''.join((str(int(c, 36)) for c in s)))

def iban_ok(s):
    c = re.sub('[\\s\\-]', '', s).upper()
    if not re.fullmatch('[A-Z]{2}\\d{2}[A-Z0-9]{10,30}', c):
        return False
    if IBAN_LENGTHS.get(c[:2]) != len(c):
        return False
    return _iban_num(c[4:] + c[:4]) % 97 == 1

def iban_check(country, bban):
    return '%02d' % (98 - _iban_num(bban.upper() + country.upper() + '00') % 97)

def dni_letter(n):
    return DNI_LETTERS[int(n) % 23]

def dni_ok(s):
    c = re.sub('[\\s.\\-]', '', s).upper()
    m = re.fullmatch('(\\d{8})([A-Z])', c)
    return bool(m) and dni_letter(m.group(1)) == m.group(2)

def nie_ok(s):
    c = re.sub('[\\s.\\-]', '', s).upper()
    m = re.fullmatch('([XYZ])(\\d{7})([A-Z])', c)
    return bool(m) and dni_letter('XYZ'.index(m.group(1)) * 10 ** 7 + int(m.group(2))) == m.group(3)

def dni_nie_ok(s):
    return dni_ok(s) or nie_ok(s)

def cif_control(letter, seven):
    a = sum((int(seven[i]) for i in (1, 3, 5)))
    b = 0
    for i in (0, 2, 4, 6):
        x = 2 * int(seven[i])
        b += x // 10 + x % 10
    c = (10 - (a + b) % 10) % 10
    if letter in 'PQRSNW':
        return 'JABCDEFGHI'[c]
    if letter in 'ABEH':
        return str(c)
    return str(c)

def ipv4_ok(s):
    try:
        return isinstance(ipaddress.ip_address(s), ipaddress.IPv4Address) and (not re.search('(^|\\.)0\\d', s))
    except ValueError:
        return False

def ipv6_ok(s):
    try:
        return isinstance(ipaddress.ip_address(s.strip('[]')), ipaddress.IPv6Address)
    except ValueError:
        return False
NON_IDENTIFYING_IPS = frozenset({'0.0.0.0', '127.0.0.1', '255.255.255.255', '::', '::1', '0:0:0:0:0:0:0:1'})

def ip_identifying(s):
    s = s.strip('[]')
    if s in NON_IDENTIFYING_IPS:
        return False
    if ipv4_ok(s):
        o = [int(x) for x in s.split('.')]
        if o[0] == 127 or o[0] == 0:
            return False
        if o[0] == 255 and o[1] == 255:
            return False
        return True
    return ipv6_ok(s)
_EMAIL = re.compile("[A-Za-z0-9!#$%&'*+/=?^_`{|}~.\\-]{1,64}@((?:[A-Za-z0-9](?:[A-Za-z0-9\\-]{0,61}[A-Za-z0-9])?\\.)+([A-Za-z]{2,24}))")

def email_ok(s):
    m = _EMAIL.fullmatch(s)
    return bool(m) and m.group(2).lower() in TLDS and (not s.startswith('.')) and ('..' not in s.split('@')[0])
