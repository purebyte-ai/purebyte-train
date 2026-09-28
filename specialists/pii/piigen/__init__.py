"""piigen: the data generator of the `pii` specialist -- windows of bytes with TYPED spans of personal data.

    types        the 12 entity types and the typed span mask (0 outside, 1 + type index, 255 = ignore)
    validators   checksums and syntax checks of the structured types (Luhn + IIN, IBAN mod-97, DNI/NIE control letter,
                 company CIF, IPv4/IPv6, e-mail with an existing top-level domain)
    values       synthetic values of every type (all with valid checksums) and their look-alikes (the same shapes in roles
                 that are not personal data)
    names        the public lists (given names, surnames, streets, places, sentences) as samplers, with a held-out split
    forms        how personal data sits in bytes: records rendered in many formats, code snippets, prose, e-mails, logs
    generator    the scenes (real code, records, prose, e-mail, logs, public documents) and the windows
    public_map   how the labels of public PII datasets map to the 12 types (or to the ignore mask)
    rules        the rules baseline: patterns + validation + context words + name lists + a credential-format catalogue
    metrics      exact, overlap and redaction scores of span predictions

No value is taken from anyone: names are frequent list entries combined at random, numbers are random with their
checksum computed, streets are real streets with a random number; a random value can still coincide with a real one. The SECRET type reuses the credential generator of the
secrets-code specialist (`secretsgen`), imported lazily. Used by `specialists/pii/plugin.py`; see its README.md.
"""
