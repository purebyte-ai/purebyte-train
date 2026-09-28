"""A sanity check of the licence declared in a manifest against the licence files at the root of a repository.

It is deliberately simple: it recognises the families of the common licence texts. It cannot prove a licence; it
catches the mistakes that matter (a repository declared permissive whose root licence is copyleft or source-available)
and records what it saw in the build manifest.
"""
import re

MARKERS = [
    ("AGPL", r"GNU AFFERO GENERAL PUBLIC LICENSE"),
    ("LGPL", r"GNU (?:LESSER|LIBRARY) GENERAL PUBLIC LICENSE"),
    ("GPL", r"GNU GENERAL PUBLIC LICENSE"),
    ("SSPL", r"Server Side Public License"),
    ("BUSL", r"Business Source License"),
    ("Elastic", r"Elastic License"),
    ("MPL", r"Mozilla Public License"),
    ("Apache", r"Apache License,?\s+Version 2\.0"),
    ("MIT", r"Permission is hereby granted,? free of charge"),
    ("BSD", r"Redistribution and use in source and binary forms"),
    ("ISC", r"Permission to use, copy, modify,? and(?:/or)? distribute this software for any"),
    ("Unlicense", r"This is free and unencumbered software released into the public domain"),
    ("CC0", r"CC0 1\.0 Universal|Creative Commons Zero"),
    ("Zlib", r"This software is provided 'as-is', without any express or implied\s+warranty"),
    ("PostgreSQL", r"Permission to use, copy, modify, and distribute this software and its\s+documentation for any purpose, without fee"),
    ("Artistic", r"The Artistic License"),
    ("curl", r"COPYRIGHT AND PERMISSION NOTICE"),
    ("blessing", r"The author disclaims copyright to this source code|public domain"),
    ("PSF", r"PYTHON SOFTWARE FOUNDATION LICENSE"),
]
RESTRICTIVE = {"AGPL", "GPL", "LGPL", "SSPL", "BUSL", "Elastic"}
SPDX_FAMILY = [
    ("AGPL", "AGPL"), ("LGPL", "LGPL"), ("GPL", "GPL"), ("MPL", "MPL"), ("Apache", "Apache"), ("MIT", "MIT"),
    ("BSD", "BSD"), ("ISC", "ISC"), ("Unlicense", "Unlicense"), ("CC0", "CC0"), ("Zlib", "Zlib"),
    ("PostgreSQL", "PostgreSQL"), ("Artistic", "Artistic"), ("curl", "curl"), ("blessing", "blessing"),
    ("PSF", "PSF"), ("Python", "PSF"),
]
LICENCE_FILE = re.compile(r"^(?:[A-Z0-9]+[-_.])?(?:LICEN[CS]E|COPYING|COPYRIGHT|UNLICENSE|NOTICE)(?:[._-].*)?$", re.I)


def families_in_text(text):
    return {fam for fam, rx in MARKERS if re.search(rx, text, re.I)}


def declared_families(spdx):
    out = set()
    for token in re.split(r"[\s()]+", spdx):
        for prefix, fam in SPDX_FAMILY:
            if token.startswith(prefix):
                out.add(fam)
                break
    return out


def check(spdx, licence_texts):
    """{declared, found, files, status, note}. status: ok | unverified | mismatch | restrictive."""
    found = set()
    for text in licence_texts.values():
        found |= families_in_text(text)
    declared = declared_families(spdx)
    res = dict(declared=spdx, found=sorted(found), files=sorted(licence_texts), status="ok", note="")
    if not licence_texts or not found:
        res.update(status="unverified", note="no recognisable licence text at the repository root")
    elif found & RESTRICTIVE and not (declared & RESTRICTIVE) and not (found & declared):
        res.update(status="restrictive", note=f"declared {spdx} but the root licence looks like {sorted(found)}")
    elif not (found & declared):
        res.update(status="mismatch", note=f"declared {spdx}, root licence text looks like {sorted(found)}")
    return res
