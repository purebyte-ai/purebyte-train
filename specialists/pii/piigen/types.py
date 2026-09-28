"""The taxonomy: 12 span types, and how a typed span mask describes them.

A window's span mask `S` (uint8 [T]) holds 0 on bytes outside every span, `1 + e` on the bytes of a span of type `e`,
and IGNORE (255) on personal data outside this taxonomy or ambiguous for it (for example a generic date or a social
security number in a public dataset): those bytes are trained with label -100, neither as a type nor as outside, and
they never make a window positive. BIOES labels: 0 = O and, for type e, B = 1 + 4e, I = 2 + 4e, E = 3 + 4e,
S = 4 + 4e, so 12 types give 49 labels.

Two spans of the SAME type are separated by at least one byte outside every span (a mask cannot tell two adjacent runs of
one type apart; the generator guarantees the gap). Two adjacent spans of DIFFERENT types are two spans.

Labelling policy, one line per type:
  EMAIL            an e-mail address. NOT: SSH remotes (`git@host:org/repo`), `icon@2x.png`, `pkg@1.2.3`, decorators,
                   social handles, template placeholders.
  PHONE            a telephone or fax number, national or international, with its prefix; not the words around it.
  IBAN             an IBAN, compact or in groups. NOT: a BIC/SWIFT code, a company VAT id, an invoice number.
  CREDIT_CARD      a payment card number (13-19 digits, Luhn-valid, a known issuer prefix). NOT: IMEIs, CVV, expiry.
  IP               an IPv4 / IPv6 host address (a port after it is outside). NOT: loopback, unspecified, broadcast,
                   netmasks, version strings, OIDs, MAC addresses.
  DNI_NIE          a Spanish DNI (8 digits + letter) or NIE (X/Y/Z + 7 digits + letter) with a valid control letter.
                   NOT: a company CIF, plates, postcodes.
  PASSPORT         a passport number, only where the context says it is one.
  DATE_OF_BIRTH    a date that is someone's birth date (the context says so). NOT: creation, expiry or log dates.
  URL_CREDENTIALS  the `user:password` inside `scheme://user:password@host`. NOT: placeholders (`${USER}:${PASS}`).
  SECRET           a credential value (API key, token, password literal, private key), as the secrets-code specialist
                   labels it.
  PERSON_NAME      a person's name (given names and surnames, or a surname after a title; the title is outside).
                   NOT: usernames, company names, streets named after someone (those are inside an ADDRESS).
  ADDRESS          a postal address from the street through the postcode and town (one span, line breaks included);
                   the country is outside.

STRUCTURED types are decided mostly by the value's own shape (often a checksum); CONTEXTUAL types share their shape
with innocent values, or need knowledge (names, streets).
"""
import numpy as np
TYPES = ('EMAIL', 'PHONE', 'IBAN', 'CREDIT_CARD', 'IP', 'DNI_NIE', 'PASSPORT', 'DATE_OF_BIRTH', 'URL_CREDENTIALS', 'SECRET', 'PERSON_NAME', 'ADDRESS')
K = len(TYPES)
IDX = {t: i for (i, t) in enumerate(TYPES)}
IGNORE = 255
IGNORE_LABEL = -100
STRUCTURED = ('EMAIL', 'PHONE', 'IBAN', 'CREDIT_CARD', 'IP', 'DNI_NIE', 'URL_CREDENTIALS', 'SECRET')
CONTEXTUAL = ('PASSPORT', 'DATE_OF_BIRTH', 'PERSON_NAME', 'ADDRESS')

def n_labels(k=K):
    return 1 + 4 * k

def bioes_typed(row):
    row = np.asarray(row)
    T = len(row)
    out = np.zeros(T, dtype=np.int64)
    t = 0
    while t < T:
        v = int(row[t])
        if v == 0:
            t += 1
            continue
        if v == IGNORE:
            out[t] = IGNORE_LABEL
            t += 1
            continue
        j = t
        while j + 1 < T and int(row[j + 1]) == v:
            j += 1
        e = v - 1
        if j == t:
            out[t] = 4 + 4 * e
        else:
            (out[t], out[j]) = (1 + 4 * e, 3 + 4 * e)
            out[t + 1:j] = 2 + 4 * e
        t = j + 1
    return out

def spans_of(row):
    row = np.asarray(row)
    (out, t, T) = ([], 0, len(row))
    while t < T:
        v = int(row[t])
        if v == 0 or v == IGNORE:
            t += 1
            continue
        j = t
        while j + 1 < T and int(row[j + 1]) == v:
            j += 1
        out.append((t, j + 1, v - 1))
        t = j + 1
    return out

def ignored_of(row):
    row = np.asarray(row)
    idx = np.flatnonzero(row == IGNORE)
    if not len(idx):
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[cuts + 1]])
    ends = np.concatenate([idx[cuts] + 1, [idx[-1] + 1]])
    return [(int(a), int(b)) for (a, b) in zip(starts, ends)]

def window_class(S):
    S = np.asarray(S)
    return ((S >= 1) & (S != IGNORE)).any(axis=-1).astype(np.uint8)

def paint(mask, spans):
    T = len(mask)
    for (a, b, e) in spans:
        (a, b) = (max(0, int(a)), min(T, int(b)))
        if a < b:
            mask[a:b] = e + 1
    return mask
