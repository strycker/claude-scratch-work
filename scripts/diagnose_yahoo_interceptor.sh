#!/usr/bin/env bash
# Throwaway diagnostic — WHO is intercepting Yahoo?
#
# Run on your machine:  bash scripts/diagnose_yahoo_interceptor.sh
#
# Evidence so far: query1.finance.yahoo.com fails certificate verification
# against BOTH curl_cffi's bundled CA store AND certifi, while stooq.com
# verifies cleanly through the same client. With verification disabled the
# response is a 19-byte "Too Many Requests\r\n" — far too small to be Yahoo's
# own throttle page, and characteristic of a middlebox.
#
# So something is intercepting Yahoo specifically and rate-limiting it. This
# names it, and shows whether the macOS system trust store already carries its
# CA (which would explain why browsers are unaffected).
#
# Delete this file once the answer is known.

set -uo pipefail

YAHOO_HOST="query1.finance.yahoo.com"
STOOQ_HOST="stooq.com"

echo "=============================================================="
echo " 1. Who issued the certificate we are being served for Yahoo?"
echo "=============================================================="
echo | openssl s_client -connect "${YAHOO_HOST}:443" -servername "${YAHOO_HOST}" 2>/dev/null \
  | openssl x509 -noout -issuer -subject 2>/dev/null \
  || echo "  (openssl probe failed)"
echo
echo "  ^ If the issuer is NOT DigiCert/Akamai/Yahoo, that name is your interceptor."

echo
echo "=============================================================="
echo " 2. Same probe for stooq.com (the control — this one verified)"
echo "=============================================================="
echo | openssl s_client -connect "${STOOQ_HOST}:443" -servername "${STOOQ_HOST}" 2>/dev/null \
  | openssl x509 -noout -issuer 2>/dev/null \
  || echo "  (openssl probe failed)"

echo
echo "=============================================================="
echo " 3. Does SYSTEM curl reach Yahoo? (uses the macOS Keychain)"
echo "=============================================================="
printf "  system curl -> Yahoo : "
curl -sS -o /tmp/_yh.txt -w '%{http_code}\n' --max-time 25 \
  "https://${YAHOO_HOST}/v8/finance/chart/SPY?range=5d&interval=1d" 2>&1 | tail -1
echo "  body (first 120):"
head -c 120 /tmp/_yh.txt 2>/dev/null; echo
echo
echo "  ^ 200 + JSON here means the Keychain HAS the interceptor's CA and the"
echo "    connection is fine for anything that reads the system trust store —"
echo "    curl_cffi does not, which is the whole problem."
echo "  ^ 429 here too means the rate limit is real for this network, not a"
echo "    verification artefact."

echo
echo "=============================================================="
echo " 4. Python requests (certifi) -> Yahoo"
echo "=============================================================="
python - <<'PY' 2>&1 | tail -5
import requests
url = "https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=5d&interval=1d"
try:
    r = requests.get(url, timeout=25)
    print(f"  status {r.status_code}, {len(r.text)} bytes: {r.text[:100]!r}")
except Exception as exc:
    print(f"  ERROR {type(exc).__name__}: {str(exc)[:160]}")
PY

echo
echo "=============================================================="
echo " 5. What is our public egress IP?"
echo "=============================================================="
printf "  egress IP: "
curl -sS --max-time 15 https://api.ipify.org 2>/dev/null || echo "(lookup failed)"
echo
echo "  ^ If this is a VPN/corporate exit, Yahoo may be throttling the shared"
echo "    address. Re-running on a phone hotspot is the fastest way to know."

echo
echo "=============================================================="
echo " READ"
echo "=============================================================="
echo "  Step 1 names a non-Yahoo issuer  -> that product is intercepting; export"
echo "                                      its root CA and pass it as verify=<path>."
echo "  Step 3 gives 200 but curl_cffi 429 -> interceptor allows curl, throttles us;"
echo "                                      trusting its CA properly should clear it."
echo "  Step 3 gives 429 as well           -> the limit is network-wide; try a hotspot."
echo "  Steps 3 and 4 both fine            -> only curl_cffi is affected; stop using it"
echo "                                      for Yahoo and let yfinance use requests."
