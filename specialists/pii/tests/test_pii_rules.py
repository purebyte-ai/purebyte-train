"""Validators, generated values, the rules on clear cases and on look-alikes, and the classic rival.

Every value is assembled at test time from random digits and made-up words: no literal in this file could pass for a
real card, account, identity document, address or credential.
"""
import numpy as np
import pytest
from pbtrain.model.decode import IGNORE

from pii_lookalike import digits, word
from pii_testkit import LABELS, build, data_block


def _iban_es(g):
    from piigen import validators as V
    bban = digits(g, 20)
    return "ES" + V.iban_check("ES", bban) + bban


def _card(g):
    from piigen import validators as V
    body = "4" + digits(g, 14)
    return body + V.luhn_complete(body)


def _dni(g):
    from piigen import validators as V
    n = int(g.integers(0, 10 ** 8))
    return f"{n:08d}{V.dni_letter(n)}"


def _grouped(s):
    return " ".join(s[i:i + 4] for i in range(0, len(s), 4))


def _other_digit(c):
    return str((int(c) + 1) % 10)


def test_checksums_accept_valid_values_and_reject_one_changed_digit():
    from piigen import validators as V
    g = np.random.default_rng(0)
    for _ in range(50):
        iban, card, dni = _iban_es(g), _card(g), _dni(g)
        assert V.iban_ok(iban) and V.iban_ok(_grouped(iban)) and not V.iban_ok(iban[:-1] + _other_digit(iban[-1]))
        assert V.luhn_ok(card) and not V.luhn_ok(card[:-1] + _other_digit(card[-1]))
        wrong = V.DNI_LETTERS[(V.DNI_LETTERS.index(dni[-1]) + 1) % 23]
        assert V.dni_ok(dni) and V.dni_nie_ok(dni[:-1] + "-" + dni[-1]) and not V.dni_ok(dni[:-1] + wrong)
    ip = f"10.{int(g.integers(0, 256))}.{int(g.integers(0, 256))}.{int(g.integers(1, 255))}"
    assert V.ip_identifying(ip) and not V.ip_identifying("127.0.0.1") and not V.ip_identifying("255.255.255.0")
    assert not V.ip_identifying("::1")
    assert V.email_ok(f"{word(g).lower()}.{word(g).lower()}@{word(g).lower()}.com")
    assert not V.email_ok("icon@2x.png") and not V.email_ok("user@localhost")


def test_generated_values_are_valid(pii_env):
    from piigen import validators as V
    from piigen import values as VL
    from piigen.names import get_lists
    pii_env.plugin.sync_resources()
    L = get_lists("train")
    rng = np.random.default_rng(0)
    for _ in range(200):
        assert V.iban_ok(VL.iban(rng))
        assert V.luhn_ok(VL.credit_card(rng))
        assert V.dni_nie_ok(VL.dni_nie(rng))
        assert V.ip_identifying(VL.ip(rng))
        assert V.email_ok(VL.email(rng, L, "es"))
        url, (a, b) = VL.url_with_credentials(rng, L, "en")
        assert ":" in url[a:b] and url[b] == "@"


def test_held_out_lists_are_disjoint(pii_env):
    from piigen.names import get_lists
    pii_env.plugin.sync_resources()
    tr, te = get_lists("train"), get_lists("test")
    assert not {r[0] for r in tr.sur_es.items} & {r[0] for r in te.sur_es.items}
    assert not {r[1] for r in tr.streets_es.items} & {r[1] for r in te.streets_es.items}
    assert len(te.sur_es) > 0 and len(tr.sur_es) > len(te.sur_es)


CLEAR = {
    "IBAN": lambda g, e: f"Mi IBAN es {_grouped(_iban_es(g))}.",
    "CREDIT_CARD": lambda g, e: f"tarjeta {_grouped(_card(g))}",
    "DNI_NIE": lambda g, e: f"DNI: {_dni(g)}",
    "EMAIL": lambda g, e: f"escribe a {word(g).lower()}.{word(g).lower()}@{word(g).lower()}.com hoy",
    "PHONE": lambda g, e: f"Llamame al 6{digits(g, 2)} {digits(g, 3)} {digits(g, 3)}",
    "IP": lambda g, e: f"client ip=10.{int(g.integers(1, 255))}.{int(g.integers(1, 255))}.{int(g.integers(1, 255))} port 22",
    "URL_CREDENTIALS": lambda g, e: f"DATABASE_URL=postgres://app:{word(g)}{digits(g, 4)}Qx@db:5432/app",
    "DATE_OF_BIRTH": lambda g, e: f"Fecha de nacimiento: {int(g.integers(10, 28))}/{int(g.integers(10, 12))}/19{int(g.integers(50, 99))}",
    "PASSPORT": lambda g, e: "Pasaporte nº " + "".join(chr(65 + int(x)) for x in g.integers(0, 26, 3)) + digits(g, 6),
    "PERSON_NAME": lambda g, e: f"Dear Mr. {e.firsts[3]} {e.surnames[5]},",
    "ADDRESS": lambda g, e: f"Vivo en Calle {word(g)} {int(g.integers(1, 99))}, 3º B, 28{digits(g, 3)} {word(g)}",
}


@pytest.mark.parametrize("typ", sorted(CLEAR))
def test_the_rules_find_clear_cases(pii_env, typ):
    from piigen import rules as RU
    from piigen.types import IDX
    pii_env.plugin.sync_resources()
    g = np.random.default_rng(7)
    for _ in range(5):
        text = CLEAR[typ](g, pii_env)
        assert IDX[typ] in {t for _, _, t, _ in RU.detect(text.encode("utf-8"))}, typ


def _cif(g):
    from piigen import validators as V
    seven = digits(g, 7)
    return "B" + seven + V.cif_control("B", seven)


@pytest.mark.parametrize("make", [
    lambda g: "git@github.com:org/repo.git", lambda g: "icon@2x.png", lambda g: "version 1.2.3.4",
    lambda g: "127.0.0.1:8080", lambda g: "postgres://user:password@localhost/db", lambda g: f"CIF {_cif(g)}",
])
def test_the_rules_skip_policy_look_alikes(pii_env, make):
    from piigen import rules as RU
    from piigen.types import IDX
    pii_env.plugin.sync_resources()
    text = make(np.random.default_rng(3))
    kinds = {t for _, _, t, _ in RU.detect(text.encode("utf-8"))}
    assert not kinds & {IDX["EMAIL"], IDX["IP"], IDX["URL_CREDENTIALS"], IDX["DNI_NIE"]}, kinds


def test_the_rivals_resolution_is_rules_detect(pii_env):
    """The rival re-resolves the rules' candidates for every score floor; at floor 0 it IS `rules.detect`."""
    from piigen import rules as RU
    P = pii_env.plugin
    _, t, _ = build(pii_env, data_block(n_train=32, n_eval=48))
    for x in np.concatenate([t.select[0], t.test[0]]):
        assert P.rule_spans(*P.rule_candidates(x), 0.0) == {(a, b, k) for a, b, k, _ in RU.detect(x.tobytes())}


def test_gold_spans_leave_ignored_bytes_out(pii_env):
    from piigen.types import spans_of
    P = pii_env.plugin
    rng = np.random.default_rng(1)
    for _ in range(200):
        row = np.zeros(64, np.uint8)
        pos = 0
        while pos < 58:
            pos += int(rng.integers(1, 6))
            n = int(rng.integers(1, 8))
            row[pos:pos + n] = int(rng.integers(1, 13)) if rng.random() < 0.8 else IGNORE
            pos += n + 1
        assert P.gold_spans(row) == set(spans_of(row))


def test_the_rival_is_tuned_on_select_and_scored_on_test(pii_env):
    v, t, _ = build(pii_env, data_block(n_train=32, n_eval=64))
    r = v.span_rival(t.select, t.test)
    assert {"span_f1", "precision", "recall", "name", "min_score", "select_f1"} <= set(r)
    assert 0.0 <= r["span_f1"] <= 1.0 and r["min_score"] in pii_env.plugin.RULE_SCORE_FLOORS
    assert set(r["per_type_test"]) == set(LABELS) and r["test_f1_all_detections"] >= 0.0
