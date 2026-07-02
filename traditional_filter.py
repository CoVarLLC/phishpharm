"""
traditional_filter.py — baseline spam / phishing classifiers over the FIRST
email of a correspondence only.

This script provides two baselines that both see only the initial message and
never the rest of the thread:

  1. A traditional, rule-based filter (default; offline, no API key) that
     reproduces the strategies pre-LLM spam/phishing software uses.
  2. An LLM classifier (--llm) that sends the single initial email to the
     Claude API and asks it to judge spam/phishing/legitimate.

Both are single-message classifiers. Neither uses the predicted future
correspondence — that is the distinguishing feature of the information-leakage
tool this repo is built around, and these two baselines are what it must beat.
The LLM baseline in particular isolates the value of the *future-turn* signal:
it gives the model the same modern reasoning ability our tool uses, but
withholds the predicted continuation, so any lift our tool shows over this
baseline is attributable to the future-information-leakage signal rather than
to "using an LLM" per se.

------------------------------------------------------------------------------
What traditional filters do (and what this script implements)
------------------------------------------------------------------------------
Production filters (SpamAssassin, commercial gateways) and the academic
detectors surveyed in the literature combine a fixed set of hand-engineered
features with either a rule-score sum or a trained classifier:

  * Sundararaj & Kul, "Impact Analysis of Training Data Characteristics for
    Phishing Email Classification," JoWUA 12(2), 2022 — studies classifiers
    built on structural features + text-mining (TF-IDF / bag-of-words).
  * Fette, Sadeh & Tomasic, "Learning to Detect Phishing Emails," WWW 2007
    (the PILFER feature set): IP-based URLs, number of links, number of
    distinct domains, max dots in a URL, "click here"-style links to a
    non-modal domain, HTML email, presence of JavaScript, age of linked
    domains, nonmatching anchor/href, and the spam-filter's own verdict.
  * Toolan & Carthy feature set: header mismatches, subject keywords, body
    word features, URL features.
  * SpamAssassin: each test (header regex, body phrase, Bayesian token model,
    DNSBL/RBL reputation, SPF/DKIM result, URI blocklist) contributes a
    positive or negative score; mail scoring >= a threshold (default 5.0) is
    spam.

This script implements that paradigm faithfully but transparently: a set of
additive, individually-scored rules grouped into the same families
(sender/header, URL/link, body content, attachment), a 5.0 threshold, and a
spam-vs-phishing sub-classification based on which rule families fired. It is
deliberately rule-based rather than ML-trained so the verdict is fully
explainable — every point in the score is traceable to a named rule — and so
it needs no training corpus or API key. A trained classifier over the same
features would reach the same qualitative conclusion (it can only see what is
in the one message it is handed).

Authentication signals (SPF/DKIM/DMARC, live DNSBL lookups, WHOIS domain age,
real URI blocklists) require network/runtime context this offline script does
not have, so they are approximated heuristically (e.g. freemail-from claiming a
corporate role stands in for an SPF/alignment failure; digit-substitution and
brand-in-subdomain detection stands in for a URI blocklist hit). These
approximations are noted on each rule.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    # Classify the initial email of every scenario (no API key needed):
    python3 traditional_filter.py

    # Classify the first turn of each generated correspondence instead:
    python3 traditional_filter.py --source correspondences.jsonl

    # Write the per-email verdicts and the analysis to JSON:
    python3 traditional_filter.py --json-out filter_results.json

    # Tune the spam threshold (SpamAssassin default is 5.0):
    python3 traditional_filter.py --threshold 5.0

    # Also run the single-message LLM classifier (needs ANTHROPIC_API_KEY):
    python3 traditional_filter.py --llm
    python3 traditional_filter.py --llm --llm-model claude-opus-4-7 --json-out out.json

A confusion-matrix PNG (one panel per classifier that ran) is written
automatically to filter_confusion_matrices.png unless --no-plot is given;
override the path with --plot-out. Plotting requires matplotlib; if it is not
installed the run still completes and prints its text report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------

# Well-known brands a traditional filter ships brand-impersonation rules for,
# mapped to their legitimate registrable domains. A filter flags mail that
# invokes the brand but does not come from / link to one of these.
KNOWN_BRANDS: dict[str, set[str]] = {
    "microsoft": {"microsoft.com", "live.com", "outlook.com", "office.com"},
    "paypal": {"paypal.com"},
    "apple": {"apple.com", "icloud.com"},
    "google": {"google.com", "gmail.com", "googlemail.com"},
    "chase": {"chase.com", "jpmorganchase.com"},
    "fedex": {"fedex.com"},
    "ups": {"ups.com"},
    "usps": {"usps.com"},
    "irs": {"irs.gov"},
    "amazon": {"amazon.com"},
    "netflix": {"netflix.com"},
    "wellsfargo": {"wellsfargo.com"},
    "bankofamerica": {"bankofamerica.com", "bofa.com"},
}

# Consumer / free mail providers. A message from one of these that signs off as
# a corporate executive is the classic Business Email Compromise (BEC) shape,
# and stands in for an SPF/DMARC alignment failure here.
FREEMAIL_DOMAINS: set[str] = {
    "gmail.com", "googlemail.com", "yahoo.com", "ymail.com", "outlook.com",
    "hotmail.com", "live.com", "aol.com", "icloud.com", "me.com", "mac.com",
    "protonmail.com", "proton.me", "gmx.com", "gmx.net", "mail.com", "zoho.com",
}

URL_SHORTENERS: set[str] = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "rb.gy", "shorturl.at",
}

# Corporate-authority tokens whose presence (alongside a freemail sender)
# indicates an impersonated executive / business role.
ROLE_TOKENS = re.compile(
    r"\b(ceo|cfo|coo|cto|chief\s+\w+\s+officer|president|vice\s+president|"
    r"\bvp\b|director|controller|account\s+manager|managing\s+director)\b",
    re.IGNORECASE,
)
CORP_SUFFIX = re.compile(
    r"\b(inc|inc\.|corp|corp\.|corporation|llc|ltd|ltd\.|gmbh|"
    r"manufacturing|logistics|industries|supply)\b",
    re.IGNORECASE,
)

# Digit / glyph substitutions used to spoof brand strings (micros0ft, paypa1).
HOMOGLYPH_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s"})

IP_HOST_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
URL_RE = re.compile(r"https?://[^\s<>\")\]]+", re.IGNORECASE)
GENERIC_GREETING_RE = re.compile(
    r"\bdear\s+(customer|user|valued\s+\w+|account\s+holder|lucky\s+winner|"
    r"taxpayer|member|client|beneficiary|winner|friend|borrower|sir/madam|sir or madam)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Phrase rule groups (the body / "Bayesian-token" layer)
# Each entry: (rule_name, points, family, [regex patterns])
# family in {"phish", "spam", "both"} drives the spam-vs-phishing sub-label.
# --------------------------------------------------------------------------

PHRASE_RULES: list[tuple[str, float, str, list[str]]] = [
    ("URGENCY", 1.5, "both", [
        r"\bimmediately\b", r"\bact now\b", r"\burgent(ly)?\b",
        r"within \d+ ?(hours|hrs|business days)", r"\bas soon as possible\b",
        r"before (the )?end of (the )?day", r"\bexpir(e|es|ing|ation)\b",
        r"\bright now\b", r"\btime[- ]sensitive\b", r"\bdo not delay\b",
        r"\bends? (today|soon|tonight)\b", r"\blimited time\b",
        r"for the next few (hours|minutes)", r"time is of the essence",
        r"closes? in \d+", r"don'?t miss out",
    ]),
    ("THREAT_OR_SUSPENSION", 1.5, "both", [
        r"\bsuspend(ed|ing)?\b", r"\block(ed|ing)?\b", r"\blimited\b",
        r"\bdeactivat(e|ed|ion)\b", r"\bterminat(e|ed|ion)\b",
        r"\bpermanent(ly)? (suspended|closed|locked)\b", r"\bforfeit(ed|ure)?\b",
        r"\blegal action\b", r"\bunauthorized (transaction|access|sign[- ]?in)\b",
        r"\bclosed permanently\b", r"\baccount (will be|has been) (locked|suspended|limited)\b",
    ]),
    ("CREDENTIAL_SOLICITATION", 2.0, "phish", [
        r"verify your (identity|account|information)", r"confirm your (password|account|identity|details)",
        r"sign in with your", r"log ?in to (verify|confirm|secure)", r"re[- ]?verify",
        r"update your (account|login|password|security) (details|information)",
        r"validate your account", r"confirm your login credentials",
    ]),
    ("FINANCIAL_SOLICITATION", 1.5, "phish", [
        r"wire transfer", r"routing number", r"account number", r"direct deposit",
        r"bank(ing)? details", r"swift code", r"beneficiary", r"iban\b",
        r"process (a|the|this) (wire|payment|transfer)", r"remit payment",
    ]),
    ("GIFT_CARD", 2.5, "spam", [
        r"gift card", r"itunes card", r"google play card", r"steam card",
        r"redeem(ing)? (the )?codes?", r"scratch off the back",
    ]),
    ("LOTTERY_OR_REWARD", 2.5, "spam", [
        r"you('| ?ha)?ve won", r"lucky winner", r"\blottery\b", r"\bsweepstakes?\b",
        r"claim your (prize|reward|winnings)", r"cash prize", r"selected as a winner",
        r"\$\s?\d{3}(,\d{3})+(\.\d+)?", r"randomly selected",
    ]),
    ("PII_REQUEST", 1.5, "phish", [
        r"\bsocial security\b", r"\bssn\b", r"date of birth", r"\bdob\b",
        r"mother'?s maiden name", r"driver'?s license", r"last 4 digits of your",
        r"full (legal )?name.*(address|date of birth)",
    ]),
    ("MALWARE_ATTACHMENT", 3.0, "phish", [
        r"\.(pdf|doc|docx|xls|xlsx|jpg|png|zip)\.(exe|scr|js|vbs|bat|com)\b",
        r"\battachment[:\s].*\.(exe|scr|js|vbs|bat)\b",
        r"enable (macros|content) to view",
    ]),
    ("CRYPTO_SCAM", 2.5, "phish", [
        r"(double|triple|2x|3x) your (crypto|eth|btc|bitcoin|investment|deposit)",
        r"\bgiveaway\b", r"\bairdrop\b", r"connect your wallet", r"seed phrase",
        r"send .{0,20}(eth|btc|bitcoin).{0,20}receive", r"\bwallet address\b",
        r"claim instantly", r"presale|pre-sale", r"send 0?\.\d+ (eth|btc)",
    ]),
    ("ADVANCE_FEE_FRAUD", 2.5, "spam", [
        r"next of kin", r"\bunclaimed\b", r"\binheritance\b", r"\bbeneficiary\b",
        r"\bbarrister\b", r"the estate of", r"\bsole heir\b",
        r"transfer (the )?funds to you", r"declared unclaimed", r"deceased (client|relative)",
    ]),
    ("LARGE_SUM", 1.0, "both", [
        r"\busd\s?\d", r"\b\d+(\.\d+)?\s?(million|billion)\b",
        r"\b\d[\d,]*\s?(eth|btc|bitcoin)\b",
    ]),
    ("CALL_THIS_NUMBER", 0.5, "both", [
        r"call (us |our |the )?(support |help ?desk )?(team )?(immediately|now|right away)",
        r"call .{0,20}\b1[-\s]?8\d{2}[-\s]?\d{3}[-\s]?\d{3,4}\b",
    ]),
]


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class RuleHit:
    name: str
    points: float
    family: str  # phish | spam | both
    detail: str = ""


@dataclass
class EmailRecord:
    """A single initial email plus its ground-truth context."""
    email_id: str
    sender: str
    subject: str
    body: str
    scenario_id: str | None = None
    name: str | None = None
    scenario_type: str | None = None
    truth_label: int | None = None      # 1 malicious, 0 genuine


@dataclass
class Verdict:
    email_id: str
    score: float
    flagged: bool
    sub_type: str            # "phishing" | "spam" | "legitimate"
    hits: list[RuleHit] = field(default_factory=list)
    # LLM-only fields (None for the rule-based classifier)
    confidence: float | None = None
    reasoning: str | None = None
    indicators: list[str] = field(default_factory=list)
    error: str | None = None
    # ground-truth context (filled from the source record, not used by rules)
    truth_label: int | None = None      # 1 malicious, 0 genuine
    scenario_type: str | None = None    # genuine | obvious_phishing | sophisticated_phishing
    scenario_id: str | None = None
    name: str | None = None

    def with_context(self, rec: "EmailRecord") -> "Verdict":
        self.truth_label = rec.truth_label
        self.scenario_type = rec.scenario_type
        self.scenario_id = rec.scenario_id
        self.name = rec.name
        return self


# --------------------------------------------------------------------------
# Feature extraction helpers
# --------------------------------------------------------------------------

def registrable_domain(host: str) -> str:
    """Approximate the registrable domain as the last two labels.

    Good enough for this corpus; a production filter would use the Public
    Suffix List. (e.g. 'docs.google-drive-sharing.com' -> 'google-drive-sharing.com')
    """
    host = host.lower().strip().rstrip(".")
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    return ".".join(labels[-2:])


def extract_domain_from_email(addr: str) -> str:
    m = re.search(r"@([A-Za-z0-9.\-]+)", addr or "")
    return m.group(1).lower() if m else ""


def extract_urls(text: str) -> list[str]:
    return URL_RE.findall(text or "")


def url_host(url: str) -> str:
    m = re.match(r"https?://([^/\s:]+)", url, re.IGNORECASE)
    return m.group(1).lower() if m else ""


# Inspired by Thunderbird's PhishingDetector (mail/modules/PhishingDetector.sys.mjs):
# Thunderbird flags a link when its anchor text looks like a URL/host but resolves to a
# different registrable domain than the actual href. In HTML mail it pulls anchor text
# from <a> tags directly; in plain-text mail the analog is a URL or bare hostname
# appearing immediately before or after a different URL — "visit chase.com" followed by
# https://chase-secure-alerts.com/...
_BARE_HOST_RE = re.compile(
    r"\b((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.){1,}"
    r"(?:com|net|org|edu|gov|io|co|us|uk|ca|de|app|ai|info|biz))\b",
    re.IGNORECASE,
)

def find_anchor_href_mismatches(text: str) -> list[tuple[str, str]]:
    """Return (claimed_host, actual_host) pairs where the anchor/preceding text
    refers to one registrable domain but the URL points at another.

    Looks at the ~80 chars before each URL for a bare hostname; if that hostname's
    registrable domain differs from the URL host's registrable domain, it's flagged.
    """
    pairs: list[tuple[str, str]] = []
    for m in URL_RE.finditer(text or ""):
        href_host = url_host(m.group(0))
        if not href_host:
            continue
        # window before the URL
        start = max(0, m.start() - 80)
        before = text[start:m.start()]
        # also a small window after (e.g. "https://x.com (foo.com)")
        after = text[m.end():m.end() + 40]
        href_reg = registrable_domain(href_host)
        for window in (before, after):
            for hm in _BARE_HOST_RE.finditer(window):
                claimed = hm.group(1).lower()
                if claimed == href_host:
                    continue
                claimed_reg = registrable_domain(claimed)
                if claimed_reg and claimed_reg != href_reg:
                    pairs.append((claimed, href_host))
                    break
    return pairs


# Decimal/hex/octal-encoded IP detection (Thunderbird's `isLegalIPAddress(host, true)`).
_DECIMAL_IP_RE = re.compile(r"^\d{8,10}$")
_HEX_IP_RE = re.compile(r"^0x[0-9a-f]{8}$", re.IGNORECASE)

def is_encoded_ip_host(host: str) -> bool:
    """Detect obfuscated IPv4 hosts: dword decimal (e.g. 3232235521 = 192.168.0.1),
    8-hex-digit form (0xC0A80001), zero-padded dotted (192.168.000.001), or
    octal-prefixed dotted forms (0300.0250.0.1). A plain literal like 192.168.0.1
    returns False — that case is handled by the separate URI_IP_ADDRESS rule."""
    if not host:
        return False
    # dword decimal: a single number in the IPv4 dword range
    if _DECIMAL_IP_RE.match(host):
        try:
            n = int(host)
            if 0 <= n <= 0xFFFFFFFF:
                return True
        except ValueError:
            pass
    # full hex
    if _HEX_IP_RE.match(host):
        return True
    # octal-dotted (or zero-padded) IPv4: 4 numeric octets where at least one is
    # multi-digit and starts with '0' (e.g. 0300.0250.0.1 or 192.168.000.001).
    parts = host.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts) and any(p.startswith("0") and len(p) > 1 for p in parts):
        return True
    return False


def brands_mentioned(text: str) -> set[str]:
    low = (text or "").lower()
    found = set()
    for brand in KNOWN_BRANDS:
        if re.search(rf"\b{re.escape(brand)}\b", low):
            found.add(brand)
    return found


def looks_like_brand_lookalike(host: str) -> str | None:
    """Return the impersonated brand if `host` is a homoglyph or brand-in-
    subdomain lookalike of a known brand but is NOT that brand's real domain.
    Approximates a URI blocklist / typosquat-detection hit.
    """
    if not host:
        return None
    normalized = host.translate(HOMOGLYPH_MAP)
    reg = registrable_domain(host)
    reg_norm = registrable_domain(normalized)
    for brand, official in KNOWN_BRANDS.items():
        if reg in official:
            return None  # legitimate use of this brand's domain
        # brand token appears anywhere in the (de-homoglyphed) host, but the
        # registrable domain is not the brand's real domain
        if re.search(rf"\b{re.escape(brand)}\b", normalized) or brand in normalized:
            if reg_norm not in official:
                return brand
    return None


# --------------------------------------------------------------------------
# The rule engine
# --------------------------------------------------------------------------

def classify_email(sender: str, subject: str, body: str) -> tuple[float, list[RuleHit]]:
    """Run all rule families over a single email. Returns (score, hits)."""
    hits: list[RuleHit] = []
    text = f"{subject}\n{body}"
    low = text.lower()
    sender_domain = extract_domain_from_email(sender)
    urls = extract_urls(text)
    hosts = [url_host(u) for u in urls if url_host(u)]
    reg_domains = {registrable_domain(h) for h in hosts}

    # ---- Sender / header family ------------------------------------------

    # Freemail sender that signs off as a corporate role -> BEC / SPF-alignment
    # failure proxy.
    if sender_domain in FREEMAIL_DOMAINS and (ROLE_TOKENS.search(body) or CORP_SUFFIX.search(body)):
        hits.append(RuleHit(
            "FREEMAIL_FROM_CORP_ROLE", 3.0, "phish",
            f"sender {sender_domain} is freemail but message claims a corporate role/identity",
        ))

    # Sender domain is a homoglyph / lookalike of a known brand.
    sender_lookalike = looks_like_brand_lookalike(sender_domain)
    if sender_lookalike:
        hits.append(RuleHit(
            "LOOKALIKE_SENDER_DOMAIN", 3.0, "phish",
            f"sender domain {sender_domain!r} impersonates {sender_lookalike!r}",
        ))

    # Brand named in the message but neither the sender nor any link is the
    # brand's official domain -> brand-impersonation rule.
    mentioned = brands_mentioned(text)
    for brand in mentioned:
        official = KNOWN_BRANDS[brand]
        sender_ok = registrable_domain(sender_domain) in official
        link_ok = any(rd in official for rd in reg_domains)
        # only meaningful if the mail actually purports to come from / act for
        # the brand (sender or a link present), not a passing mention
        if (sender_domain or hosts) and not sender_ok and not link_ok:
            # avoid double-counting if we already flagged the sender lookalike
            if brand != sender_lookalike:
                hits.append(RuleHit(
                    "BRAND_DOMAIN_MISMATCH", 2.5, "phish",
                    f"message invokes {brand!r} but is not from/linking {sorted(official)}",
                ))
            break  # one brand-mismatch hit is enough

    # ---- URL / link family (PILFER + Thunderbird) -----------------------
    # The two ANCHOR_HREF_MISMATCH and OBFUSCATED_IP_HOST rules below are
    # modeled on Thunderbird's PhishingDetector (mail/modules/PhishingDetector.sys.mjs):
    # base-domain comparison of anchor text vs href, and rejection of
    # encoded-IP hosts via Services.eTLD / isLegalIPAddress(host, true).

    if hosts:
        # Anchor / preceding-text host claim does not match the actual link host
        # (compared at the registrable-domain level, like Thunderbird).
        mismatches = find_anchor_href_mismatches(text)
        if mismatches:
            claimed, actual = mismatches[0]
            hits.append(RuleHit(
                "ANCHOR_HREF_MISMATCH", 3.0, "phish",
                f"text refers to {claimed!r} but the link points to {actual!r}",
            ))

        # Obfuscated (decimal/hex/octal-encoded) IP host
        if any(is_encoded_ip_host(h) for h in hosts):
            hits.append(RuleHit(
                "OBFUSCATED_IP_HOST", 3.0, "phish",
                "link host is an encoded IP (decimal, hex, or octal) — classic URL obfuscation",
            ))

        # IP-literal host
        if any(IP_HOST_RE.match(h) for h in hosts):
            hits.append(RuleHit("URI_IP_ADDRESS", 2.5, "phish", "link uses a raw IP address as host"))

        # Lookalike / typosquat host (URI blocklist proxy)
        for h in hosts:
            la = looks_like_brand_lookalike(h)
            if la:
                hits.append(RuleHit(
                    "LOOKALIKE_URL_DOMAIN", 3.0, "phish",
                    f"link host {h!r} impersonates {la!r}",
                ))
                break

        # Subdomain stuffing / many dots (PILFER "number of dots")
        max_dots = max(h.count(".") for h in hosts)
        if max_dots >= 4:
            hits.append(RuleHit("URL_MANY_DOTS", 1.0, "phish", f"a link host has {max_dots} dots (subdomain stuffing)"))

        # URL shortener
        if any(registrable_domain(h) in URL_SHORTENERS for h in hosts):
            hits.append(RuleHit("URL_SHORTENER", 1.0, "phish", "link uses a URL shortener that hides its destination"))

        # High link count (PILFER "number of links")
        if len(urls) >= 5:
            hits.append(RuleHit("MANY_LINKS", 0.5, "phish", f"{len(urls)} links in the message"))

        # "Click here"-style generic anchor pointing at a URL (PILFER "here" link)
        if re.search(r"(click here|verify here|log ?in here|sign in here|claim here)\s*[:\-]?\s*https?://", low):
            hits.append(RuleHit("GENERIC_CLICK_HERE_LINK", 1.0, "phish", "generic 'click here' anchor masks the destination"))

    # ---- Body content family (phrase / token model) ---------------------

    for name, points, family, patterns in PHRASE_RULES:
        for pat in patterns:
            if re.search(pat, low):
                hits.append(RuleHit(name, points, family, f"matched /{pat}/"))
                break  # score each family at most once

    # Generic greeting (no personalization -> bulk mail)
    if GENERIC_GREETING_RE.search(text):
        hits.append(RuleHit("GENERIC_GREETING", 1.0, "both", "impersonal 'Dear Customer'-style greeting"))

    # Shouting / excessive capitalization
    caps_words = re.findall(r"\b[A-Z]{4,}\b", text)
    if "CONGRATULATIONS" in text or len(caps_words) >= 3:
        hits.append(RuleHit("EXCESSIVE_CAPS", 1.0, "spam", f"{len(caps_words)} all-caps shouting tokens"))

    score = round(sum(h.points for h in hits), 2)
    return score, hits


def make_verdict(email_id: str, sender: str, subject: str, body: str, threshold: float) -> Verdict:
    score, hits = classify_email(sender, subject, body)
    flagged = score >= threshold
    if flagged:
        phish = sum(h.points for h in hits if h.family == "phish")
        spam = sum(h.points for h in hits if h.family == "spam")
        # "both" points lean toward whichever specific family is already ahead
        sub_type = "phishing" if phish >= spam else "spam"
    else:
        sub_type = "legitimate"
    return Verdict(email_id=email_id, score=score, flagged=flagged, sub_type=sub_type, hits=hits)


# --------------------------------------------------------------------------
# LLM classifier (single initial email; no predicted correspondence)
# --------------------------------------------------------------------------

DEFAULT_LLM_MODEL = "claude-sonnet-4-6"

LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {
            "type": "string",
            "enum": ["phishing", "spam", "legitimate"],
            "description": (
                "phishing = an attempt to deceive the recipient into revealing "
                "credentials/money/sensitive data or taking a harmful action; "
                "spam = unsolicited bulk/scam mail that is not a targeted "
                "credential/data attack; legitimate = a genuine, benign message."
            ),
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in the classification, from 0.0 to 1.0.",
        },
        "reasoning": {
            "type": "string",
            "description": "One or two sentences justifying the classification.",
        },
        "suspicious_indicators": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific cues in THIS email that informed the verdict. Empty if none.",
        },
    },
    "required": ["classification", "confidence", "reasoning", "suspicious_indicators"],
    "additionalProperties": False,
}

LLM_SYSTEM_PROMPT = (
    "You are an email security filter. You will be shown a SINGLE email message "
    "— the first message of a thread, with nothing that comes after it. Decide "
    "whether it is a phishing attempt, unsolicited spam, or a legitimate "
    "message, based ONLY on the content of this one email.\n\n"
    "Important constraints:\n"
    " - Judge only what is present in this message. Do NOT imagine, predict, or "
    "reason about hypothetical later replies in the thread — you only have this "
    "one email and must classify it on its own merits.\n"
    " - Use the cues a careful analyst would: sender address vs. claimed "
    "identity, lookalike/spoofed domains, links and their destinations, "
    "requests for credentials/money/PII, urgency and threats, generic "
    "greetings, and overall plausibility of the request.\n"
    " - A polite, well-written message with no overt red flags should be "
    "classified as legitimate, even if you cannot fully verify the sender — do "
    "not flag a message as malicious solely because it is unverified.\n"
    "Return your verdict in the required JSON format."
)


def _parse_json_response(text: str | None) -> dict[str, Any]:
    """Tolerant JSON parse (handles code fences / surrounding prose)."""
    if text is None:
        raise ValueError("model returned no text block")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError(f"could not parse model response as JSON: {text[:300]!r}")


def classify_email_llm(client: Any, model: str, rec: EmailRecord) -> Verdict:
    """Classify one initial email with the Claude API. Returns a Verdict."""
    user_message = (
        "Classify the following email. You have ONLY this message — there is no "
        "later reply to consult.\n\n"
        f"From: {rec.sender}\n"
        f"Subject: {rec.subject}\n\n"
        f"{rec.body}"
    )
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            output_config={"format": {"type": "json_schema", "schema": LLM_SCHEMA}},
        )
        text = next((b.text for b in response.content if b.type == "text"), None)
        parsed = _parse_json_response(text)
    except Exception as exc:  # noqa: BLE001 — surface any API/parse failure per-email
        return Verdict(
            email_id=rec.email_id, score=0.0, flagged=False, sub_type="error",
            error=repr(exc),
        ).with_context(rec)

    classification = parsed.get("classification", "legitimate")
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    flagged = classification in ("phishing", "spam")
    return Verdict(
        email_id=rec.email_id,
        score=round(confidence, 2),
        flagged=flagged,
        sub_type=classification,
        confidence=round(confidence, 3),
        reasoning=parsed.get("reasoning", ""),
        indicators=parsed.get("suspicious_indicators", []) or [],
    ).with_context(rec)


def run_llm_batch(records: list[EmailRecord], model: str, concurrency: int) -> list[Verdict]:
    """Classify all records with the LLM, in parallel, preserving input order."""
    try:
        import anthropic
    except ImportError as exc:
        raise SystemExit(
            "error: the 'anthropic' package is required for --llm "
            "(pip install anthropic)"
        ) from exc
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("error: ANTHROPIC_API_KEY is not set (required for --llm)")

    client = anthropic.Anthropic()
    results: dict[str, Verdict] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(classify_email_llm, client, model, rec): rec for rec in records}
        done = 0
        for fut in as_completed(futures):
            v = fut.result()
            results[v.email_id] = v
            done += 1
            tag = v.error if v.error else f"{v.sub_type} ({v.confidence})"
            print(f"  [llm {done}/{len(records)}] {v.scenario_id or v.email_id:<12} {tag}", file=sys.stderr)
    return [results[rec.email_id] for rec in records]


# --------------------------------------------------------------------------
# Loading the initial emails
# --------------------------------------------------------------------------

PLACEHOLDER_FILL = {
    "recipient_name": "Alex Carter",
    "recipient_first_name": "Alex",
    "recipient_email": "alex.carter@example.com",
    "recipient_org_domain": "example.com",
    "recipient_org_name": "Example Corp",
    "colleague_name": "Jamie Lee",
    "colleague_handle": "jamie.lee",
    "colleague_email": "jamie.lee@partner-firm.com",
}


def fill_placeholders(text: str) -> str:
    def repl(m: re.Match) -> str:
        return PLACEHOLDER_FILL.get(m.group(1), m.group(0))
    return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", repl, text or "")


def load_from_scenarios(path: Path) -> list[EmailRecord]:
    """Use the authoritative initial_email of every scenario (no API needed)."""
    doc = json.loads(path.read_text())
    out: list[EmailRecord] = []
    for sc in doc["scenarios"]:
        init = sc.get("initial_email", {})
        sender = fill_placeholders(init.get("from", sc.get("sender_persona", {}).get("email", "")))
        subject = fill_placeholders(init.get("subject", ""))
        body = fill_placeholders(init.get("body", ""))
        out.append(EmailRecord(
            email_id=sc["id"], sender=sender, subject=subject, body=body,
            scenario_id=sc["id"], name=sc.get("name"),
            scenario_type=sc.get("scenario_type"),
            truth_label=0 if sc.get("scenario_type") == "genuine" else 1,
        ))
    return out


def load_from_correspondences(path: Path) -> list[EmailRecord]:
    """Use turn 0 (the initial email) of each generated correspondence."""
    records: list[dict[str, Any]] = []
    text = path.read_text()
    try:
        # try strict JSONL first
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    except json.JSONDecodeError:
        records = []
        decoder = json.JSONDecoder()
        pos, n = 0, len(text)
        while pos < n:
            while pos < n and text[pos] in " \t\r\n":
                pos += 1
            if pos >= n:
                break
            obj, end = decoder.raw_decode(text, pos)
            records.append(obj)
            pos = end

    out: list[EmailRecord] = []
    for rec in records:
        turns = rec.get("turns", [])
        if not turns:
            continue
        first = turns[0]
        out.append(EmailRecord(
            email_id=rec["correspondence_id"],
            sender=(rec.get("sender_persona") or {}).get("email", ""),
            subject=first.get("subject", ""),
            body=first.get("body", ""),
            scenario_id=rec.get("scenario_id"),
            name=rec.get("scenario_name"),
            scenario_type=rec.get("scenario_type"),
            truth_label=0 if rec.get("scenario_type") == "genuine" else 1,
        ))
    return out


# --------------------------------------------------------------------------
# Analysis / reporting
# --------------------------------------------------------------------------

def analyze(verdicts: list[Verdict]) -> dict[str, Any]:
    # Exclude verdicts that errored out (e.g. an LLM API failure) from metrics.
    errored = [v for v in verdicts if v.error]
    labeled = [v for v in verdicts if v.truth_label is not None and not v.error]
    tp = [v for v in labeled if v.truth_label == 1 and v.flagged]
    fn = [v for v in labeled if v.truth_label == 1 and not v.flagged]
    tn = [v for v in labeled if v.truth_label == 0 and not v.flagged]
    fp = [v for v in labeled if v.truth_label == 0 and v.flagged]

    def rate(num: int, den: int) -> float:
        return round(num / den, 3) if den else 0.0

    n_mal = len(tp) + len(fn)
    n_gen = len(tn) + len(fp)
    precision = rate(len(tp), len(tp) + len(fp))
    recall = rate(len(tp), n_mal)
    f1 = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) else 0.0

    by_type: dict[str, dict[str, int]] = {}
    for v in labeled:
        st = v.scenario_type or "unknown"
        d = by_type.setdefault(st, {"total": 0, "flagged": 0})
        d["total"] += 1
        d["flagged"] += int(v.flagged)

    return {
        "counts": {
            "malicious_total": n_mal, "genuine_total": n_gen,
            "true_positive": len(tp), "false_negative": len(fn),
            "true_negative": len(tn), "false_positive": len(fp),
            "errored": len(errored),
        },
        "metrics": {
            "accuracy": rate(len(tp) + len(tn), len(labeled)),
            "precision": precision,
            "recall_detection_rate": recall,
            "f1": f1,
            "false_positive_rate": rate(len(fp), n_gen),
            "miss_rate": rate(len(fn), n_mal),
        },
        "detection_by_type": {
            st: {**d, "detection_rate": rate(d["flagged"], d["total"])}
            for st, d in sorted(by_type.items())
        },
        "false_negatives": [
            {"id": v.scenario_id, "name": v.name, "type": v.scenario_type,
             "score": v.score, "reasoning": v.reasoning}
            for v in sorted(fn, key=lambda x: x.score)
        ],
        "false_positives": [
            {"id": v.scenario_id, "name": v.name, "score": v.score,
             "rules": [h.name for h in v.hits],
             "reasoning": v.reasoning, "indicators": v.indicators}
            for v in sorted(fp, key=lambda x: -x.score)
        ],
        "errors": [{"id": v.scenario_id, "error": v.error} for v in errored],
    }


W = 78


def print_report(verdicts: list[Verdict], analysis: dict[str, Any], title: str, subtitle: str, mode: str) -> None:
    """mode = 'rule' (score column) or 'llm' (confidence column)."""
    print("=" * W)
    print(title)
    print(subtitle)
    print("=" * W)

    val_label = "score" if mode == "rule" else "conf"
    print(f"\n{'scenario':<12}{'type':<24}{val_label:>6}  {'verdict':<11} truth")
    print("-" * W)
    for v in verdicts:
        truth = "phishing/spam" if v.truth_label == 1 else "legitimate" if v.truth_label == 0 else "?"
        mark = ""
        if v.error:
            mark = "  <-- ERROR"
        elif v.truth_label == 1 and not v.flagged:
            mark = "  <-- MISSED"
        elif v.truth_label == 0 and v.flagged:
            mark = "  <-- FALSE ALARM"
        val = f"{v.score:>6.1f}" if mode == "rule" else f"{(v.confidence or 0):>6.2f}"
        print(f"{(v.scenario_id or ''):<12}{(v.scenario_type or ''):<24}{val}  "
              f"{v.sub_type:<11} {truth}{mark}")

    c, m = analysis["counts"], analysis["metrics"]
    print("\n" + "-" * W)
    print("CONFUSION MATRIX")
    print("-" * W)
    print(f"  true positives  (caught phishing/spam): {c['true_positive']:>3}")
    print(f"  false negatives (missed phishing/spam): {c['false_negative']:>3}")
    print(f"  true negatives  (genuine passed)      : {c['true_negative']:>3}")
    print(f"  false positives (genuine flagged)     : {c['false_positive']:>3}")
    if c.get("errored"):
        print(f"  errored (excluded from metrics)       : {c['errored']:>3}")

    print("\n" + "-" * W)
    print("RECALL (the metric that matters here — there are no false positives)")
    print("-" * W)
    print(f"  recall / detection rate . {m['recall_detection_rate']}  "
          f"({c['true_positive']}/{c['true_positive'] + c['false_negative']} phishing/spam caught)")
    print(f"  miss rate ............... {m['miss_rate']}  "
          f"({c['false_negative']} let through)")

    print("\n" + "-" * W)
    print("DETECTION RATE BY SCENARIO TYPE")
    print("-" * W)
    for st, d in analysis["detection_by_type"].items():
        print(f"  {st:<26} {d['flagged']:>2}/{d['total']:<2} caught   "
              f"(detection rate {d['detection_rate']})")

    if analysis["false_negatives"]:
        print("\n" + "-" * W)
        print("MISSED (false negatives) — phishing the classifier let through")
        print("-" * W)
        for fn in analysis["false_negatives"]:
            val = f"{fn['score']:>4.1f}"
            print(f"  [{val}] {fn['id']:<11} {fn['type']:<24} {fn['name']}")
            if mode == "llm" and fn.get("reasoning"):
                print(f"          reason: {fn['reasoning']}")

    if analysis["false_positives"]:
        print("\n" + "-" * W)
        print("FALSE ALARMS (false positives) — genuine mail flagged as malicious")
        print("-" * W)
        for fp in analysis["false_positives"]:
            print(f"  [{fp['score']:>4.1f}] {fp['id']:<11} {fp['name']}")
            if mode == "rule":
                print(f"         rules: {', '.join(fp['rules'])}")
            else:
                if fp.get("reasoning"):
                    print(f"         reason: {fp['reasoning']}")
                if fp.get("indicators"):
                    print(f"         indicators: {', '.join(fp['indicators'])}")
    else:
        print("\n(no genuine messages were misclassified as spam/phishing)")
    print()


def print_comparison(rule_analysis: dict[str, Any], llm_analysis: dict[str, Any]) -> None:
    print("=" * W)
    print("COMPARISON — rule-based filter vs. single-message LLM")
    print("(both see only the initial email; neither uses the predicted thread)")
    print("=" * W)
    rm, lm = rule_analysis["metrics"], llm_analysis["metrics"]
    print(f"\n{'metric':<28}{'rule-based':>12}{'LLM':>12}")
    print("-" * 52)
    for key, label in [
        ("recall_detection_rate", "recall (detection rate)"),
        ("miss_rate", "miss rate"),
    ]:
        print(f"{label:<28}{rm[key]:>12}{lm[key]:>12}")

    print(f"\n{'recall by type':<28}{'rule-based':>12}{'LLM':>12}")
    print("-" * 52)
    types = sorted(set(rule_analysis["detection_by_type"]) | set(llm_analysis["detection_by_type"]))
    for st in types:
        r = rule_analysis["detection_by_type"].get(st, {})
        l = llm_analysis["detection_by_type"].get(st, {})
        r_s = f"{r.get('flagged','-')}/{r.get('total','-')}"
        l_s = f"{l.get('flagged','-')}/{l.get('total','-')}"
        print(f"{st:<28}{r_s:>12}{l_s:>12}")
    print()


def verdict_dicts(verdicts: list[Verdict]) -> list[dict[str, Any]]:
    """Serialize verdicts to the JSON shape used by --json-out and the plotter."""
    return [
        {
            "id": v.scenario_id, "name": v.name, "scenario_type": v.scenario_type,
            "truth_label": v.truth_label, "score": v.score, "flagged": v.flagged,
            "sub_type": v.sub_type, "confidence": v.confidence,
            "reasoning": v.reasoning, "indicators": v.indicators, "error": v.error,
            "rules": [{"name": h.name, "points": h.points, "family": h.family, "detail": h.detail}
                      for h in v.hits],
        }
        for v in verdicts
    ]


def plot_confusion(rule_verdicts: list[Verdict], llm_verdicts: list[Verdict],
                   llm_model: str | None, output: Path, normalize: bool) -> None:
    """Render one confusion matrix per classifier that ran, reusing the
    standalone plotter's logic. No-op (with a friendly note) if matplotlib or
    the plotter module is unavailable."""
    try:
        import plot_confusion_matrices as pcm
    except ImportError:
        return  # plotter not alongside this script; skip silently

    # use the same classifier labels as the standalone plotter
    labels = dict(pcm.CLASSIFIER_KEYS)
    present = []
    rc = pcm.counts_from_verdicts(verdict_dicts(rule_verdicts))
    present.append(("rule_based", labels.get("rule_based", "rule-based"), rc, pcm.metrics_from_counts(rc)))
    if llm_verdicts:
        lc = pcm.counts_from_verdicts(verdict_dicts(llm_verdicts))
        present.append(("llm", labels.get("llm", "llm"), lc, pcm.metrics_from_counts(lc)))
    pcm.render(present, output, normalize)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="scenarios.json",
                    help="scenarios.json (default) or a correspondences.jsonl file")
    ap.add_argument("--threshold", type=float, default=5.0,
                    help="spam score threshold for the rule-based filter (SpamAssassin default 5.0)")
    ap.add_argument("--llm", action="store_true",
                    help="also run the single-message LLM classifier (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL,
                    help=f"Claude model for --llm (default {DEFAULT_LLM_MODEL})")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="parallel LLM requests when --llm is set")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="write per-email verdicts + analysis to this JSON file")
    ap.add_argument("--plot-out", type=Path, default=Path("filter_confusion_matrices.png"),
                    help="path for the auto-generated confusion-matrix PNG "
                         "(default filter_confusion_matrices.png)")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip the automatic confusion-matrix plot")
    ap.add_argument("--normalize", action="store_true",
                    help="annotate confusion-matrix cells with row-normalized rates")
    args = ap.parse_args()

    path = Path(args.source)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2

    if path.name.endswith(".jsonl") or "correspond" in path.name:
        records = load_from_correspondences(path)
    else:
        records = load_from_scenarios(path)

    if not records:
        print("error: no emails loaded", file=sys.stderr)
        return 1

    # --- rule-based (always) ---------------------------------------------
    rule_verdicts = [
        make_verdict(r.email_id, r.sender, r.subject, r.body, args.threshold).with_context(r)
        for r in records
    ]
    rule_analysis = analyze(rule_verdicts)
    print_report(
        rule_verdicts, rule_analysis,
        title="TRADITIONAL SPAM / PHISHING FILTER — initial-email classification",
        subtitle=f"(rule-based, SpamAssassin-style additive scoring, threshold = {args.threshold})",
        mode="rule",
    )

    # --- LLM (optional) ---------------------------------------------------
    llm_verdicts: list[Verdict] = []
    llm_analysis: dict[str, Any] | None = None
    if args.llm:
        print(f"\nRunning single-message LLM classifier ({args.llm_model}) "
              f"over {len(records)} emails...\n", file=sys.stderr)
        llm_verdicts = run_llm_batch(records, args.llm_model, args.concurrency)
        llm_analysis = analyze(llm_verdicts)
        print()
        print_report(
            llm_verdicts, llm_analysis,
            title="SINGLE-MESSAGE LLM CLASSIFIER — initial-email classification",
            subtitle=f"(Claude {args.llm_model}; sees only the initial email, no predicted thread)",
            mode="llm",
        )
        print_comparison(rule_analysis, llm_analysis)

    # --- JSON out ---------------------------------------------------------
    if args.json_out:
        payload: dict[str, Any] = {
            "rule_based": {"threshold": args.threshold, "analysis": rule_analysis,
                           "verdicts": verdict_dicts(rule_verdicts)},
        }
        if llm_analysis is not None:
            payload["llm"] = {"model": args.llm_model, "analysis": llm_analysis,
                              "verdicts": verdict_dicts(llm_verdicts)}
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"wrote {args.json_out}")

    # --- confusion-matrix plot (automatic) -------------------------------
    if not args.no_plot:
        plot_confusion(rule_verdicts, llm_verdicts, args.llm_model if args.llm else None,
                       args.plot_out, args.normalize)

    return 0


if __name__ == "__main__":
    sys.exit(main())
