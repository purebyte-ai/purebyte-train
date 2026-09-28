"""How the labels of public PII datasets map to the 12 types. One table, for the public TRAIN documents of the
generator (`generator.PublicDocs`) and for evaluation sets built from public test splits.

  mapped    a source label that IS one of the types (first_name -> PERSON_NAME, iban -> IBAN, ...). Adjacent pieces of
            one person (given name + surname) or one address (street + number, + a postcode / town right after) are
            merged into one span, which is the labelling policy (one span per person, per postal address).
  ignore    personal data OUTSIDE the taxonomy (social security numbers, account numbers, a lone city, ...) or ambiguous
            for it (a generic date may or may not be a birth date): in an evaluation a prediction there is neither
            right nor wrong; in TRAIN those bytes are IGNORE (label -100), neither a type nor O.
  text      a source label that is not personal data under the policy (company names, occupations, generic dates):
            plain text in an evaluation (a prediction there is a false positive); in TRAIN it is ignored too.
`collapse=True` (a label-agnostic single-type variant): EVERY source label is personal data, one type, nothing ignored.
"""
import re
from .types import IDX
NEMOTRON = {'first_name': 'PERSON_NAME', 'last_name': 'PERSON_NAME', 'email': 'EMAIL', 'phone_number': 'PHONE', 'fax_number': 'PHONE', 'credit_debit_card': 'CREDIT_CARD', 'ipv4': 'IP', 'ipv6': 'IP', 'date_of_birth': 'DATE_OF_BIRTH', 'street_address': 'ADDRESS', 'password': 'SECRET', 'api_key': 'SECRET'}
NEMOTRON_IGNORE = {'ssn', 'account_number', 'customer_id', 'employee_id', 'medical_record_number', 'health_plan_beneficiary_number', 'certificate_license_number', 'license_plate', 'vehicle_identifier', 'device_identifier', 'unique_id', 'tax_id', 'bank_routing_number', 'swift_bic', 'cvv', 'pin', 'mac_address', 'http_cookie', 'biometric_identifier', 'user_name', 'coordinate', 'postcode', 'city', 'state', 'county', 'url', 'national_id', 'passport_number', 'driver_license_number', 'iban'}
AI4P = {'GIVENNAME': 'PERSON_NAME', 'SURNAME': 'PERSON_NAME', 'EMAIL': 'EMAIL', 'TELEPHONENUM': 'PHONE', 'CREDITCARDNUMBER': 'CREDIT_CARD', 'STREET': 'ADDRESS', 'BUILDINGNUM': 'ADDRESS', 'PASSPORTNUM': 'PASSPORT', 'IDCARDNUM': 'DNI_NIE?', 'TAXNUM': 'DNI_NIE?'}
AI4P_IGNORE = {'TITLE', 'DATE', 'AGE', 'GENDER', 'SEX', 'DRIVERLICENSENUM', 'SOCIALNUM', 'CITY', 'ZIPCODE', 'IDCARDNUM', 'TAXNUM'}
GRETEL = {'name': 'PERSON_NAME', 'first_name': 'PERSON_NAME', 'last_name': 'PERSON_NAME', 'email': 'EMAIL', 'phone_number': 'PHONE', 'iban': 'IBAN', 'credit_card_number': 'CREDIT_CARD', 'ipv4': 'IP', 'ipv6': 'IP', 'date_of_birth': 'DATE_OF_BIRTH', 'street_address': 'ADDRESS', 'passport_number': 'PASSPORT', 'password': 'SECRET', 'api_key': 'SECRET'}
GRETEL_IGNORE = {'swift_bic_code', 'bban', 'employee_id', 'credit_card_security_code', 'ssn', 'account_pin', 'customer_id', 'driver_license_number', 'bank_routing_number', 'user_name', 'local_latlng', 'account_number', 'tax_id', 'national_id', 'license_plate', 'medical_record_number', 'date', 'address', 'unique_identifier', 'device_identifier', 'biometric_identifier', 'certificate_license_number', 'vehicle_identifier', 'coordinate', 'postcode', 'city', 'state', 'health_plan_beneficiary_number', 'swift_bic', 'url', 'date_time', 'time'}
DNI_SHAPE = re.compile('^(?:\\d{8}|[XYZ]\\d{7})[\\-\\s]?[A-Z]$', re.I)
_GAP = re.compile('[\\s,.\\-]{0,3}|\\s*\\n\\s*')

def tables(source):
    s = source.lower()
    if 'nemotron' in s:
        return (NEMOTRON, NEMOTRON_IGNORE)
    if 'ai4privacy' in s or 'openpii' in s:
        return (AI4P, AI4P_IGNORE)
    return (GRETEL, GRETEL_IGNORE)

def map_doc(text, labelled, source, collapse=False, train=False):
    (table, ign_set) = tables(source)
    (spans, ign) = ([], [])
    for (a, b, lab) in sorted(labelled):
        if not 0 <= a < b <= len(text) or not text[a:b].strip():
            continue
        if collapse:
            spans.append([a, b, 'PII'])
            continue
        typ = table.get(lab)
        if typ == 'DNI_NIE?':
            typ = 'DNI_NIE' if DNI_SHAPE.match(text[a:b].strip()) else None
        if typ is None:
            if train or lab in ign_set:
                ign.append([a, b, lab])
            continue
        spans.append([a, b, typ])
    merged = []
    for (a, b, t) in spans:
        if merged and merged[-1][2] == t and (t in ('PERSON_NAME', 'ADDRESS', 'PII')) and _GAP.fullmatch(text[merged[-1][1]:a] or '') and (a >= merged[-1][1] - 0):
            merged[-1][1] = max(merged[-1][1], b)
        elif merged and a < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b, t])
    ign2 = []
    for (a, b, lab) in ign:
        hit = next((m for m in merged if m[2] == 'ADDRESS' and _GAP.fullmatch(text[m[1]:a] or '')), None)
        if hit is not None and lab in ('ZIPCODE', 'CITY', 'postcode', 'city', 'state'):
            hit[1] = max(hit[1], b)
        else:
            ign2.append([a, b, lab])
    return (merged, ign2)

def to_bytes(text, items):
    (out, cache) = ([], {})

    def bo(i):
        if i not in cache:
            cache[i] = len(text[:i].encode('utf-8'))
        return cache[i]
    for (a, b, x) in items:
        out.append([bo(a), bo(b), x])
    return out

def type_index(t, collapse=False):
    return 0 if collapse else IDX[t]
