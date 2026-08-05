#!/usr/bin/env bash
# Build a CA bundle that trusts BOTH the public roots and whatever your
# machine/network uses to intercept TLS, then verify it works.
#
#   bash scripts/fix_local_ca.sh
#
# WHY THIS EXISTS
# ---------------
# On this machine query1.finance.yahoo.com fails certificate verification
# ("self signed certificate in certificate chain") against both curl_cffi's
# bundled CA store and certifi, while stooq.com verifies cleanly through the
# same client. Something is intercepting Yahoo and signing with a CA that is
# in no public bundle — but which is very likely trusted in your macOS
# Keychain, since browsers reach Yahoo Finance fine.
#
# curl_cffi deliberately ignores the Keychain (see ingestion/assets.py). So the
# fix is to export the Keychain roots, concatenate them with certifi's public
# roots, and point our HTTP clients at the combined file. That restores REAL
# certificate verification instead of the current verify=False bypass (P22).
#
# Everything is written to $HOME, nothing in the repo, no sudo required.

set -uo pipefail

OUT="${HOME}/.trading-crab-ca.pem"
TMPDIR_CA="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_CA}"' EXIT

echo "=============================================================="
echo " 1. Locating certifi's public root bundle"
echo "=============================================================="
CERTIFI_PEM="$(python -c 'import certifi; print(certifi.where())' 2>/dev/null)"
if [ -z "${CERTIFI_PEM}" ] || [ ! -f "${CERTIFI_PEM}" ]; then
  echo "  ERROR: could not locate certifi. Activate your venv first."
  exit 2
fi
echo "  ${CERTIFI_PEM}"

echo
echo "=============================================================="
echo " 2. Exporting trusted roots from the macOS Keychain"
echo "=============================================================="
# Admin-installed roots (where an interceptor's CA normally lands):
security find-certificate -a -p /Library/Keychains/System.keychain \
  > "${TMPDIR_CA}/system.pem" 2>/dev/null
# Apple's shipped roots:
security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain \
  > "${TMPDIR_CA}/approots.pem" 2>/dev/null
# Per-user roots (some products install here instead):
security find-certificate -a -p "${HOME}/Library/Keychains/login.keychain-db" \
  > "${TMPDIR_CA}/login.pem" 2>/dev/null

for f in system approots login; do
  n=$(grep -c 'BEGIN CERTIFICATE' "${TMPDIR_CA}/${f}.pem" 2>/dev/null || echo 0)
  printf "  %-10s %s certificates\n" "${f}" "${n}"
done

echo
echo "=============================================================="
echo " 3. Capturing the chain actually served for Yahoo"
echo "=============================================================="
echo | openssl s_client -showcerts -connect query1.finance.yahoo.com:443 \
  -servername query1.finance.yahoo.com 2>/dev/null \
  | awk '/BEGIN CERT/,/END CERT/' > "${TMPDIR_CA}/yahoo_chain.pem"
CHAIN_N=$(grep -c 'BEGIN CERTIFICATE' "${TMPDIR_CA}/yahoo_chain.pem" 2>/dev/null || echo 0)
echo "  captured ${CHAIN_N} certificate(s) from the live handshake"
echo "  issuer of the leaf:"
echo | openssl s_client -connect query1.finance.yahoo.com:443 \
  -servername query1.finance.yahoo.com 2>/dev/null \
  | openssl x509 -noout -issuer 2>/dev/null | sed 's/^/    /'
echo
echo "  ^^ If that issuer is not DigiCert/Akamai/Yahoo, it names your interceptor."

echo
echo "=============================================================="
echo " 4. Building the combined bundle"
echo "=============================================================="
# certifi first (public roots win), then Keychain roots, then the observed
# chain as a last resort for a CA that lives nowhere else.
cat "${CERTIFI_PEM}" \
    "${TMPDIR_CA}/approots.pem" \
    "${TMPDIR_CA}/system.pem" \
    "${TMPDIR_CA}/login.pem" \
    "${TMPDIR_CA}/yahoo_chain.pem" 2>/dev/null \
  | awk '/BEGIN CERTIFICATE/,/END CERTIFICATE/' > "${OUT}"
TOTAL=$(grep -c 'BEGIN CERTIFICATE' "${OUT}" 2>/dev/null || echo 0)
echo "  wrote ${OUT}  (${TOTAL} certificates)"

echo
echo "=============================================================="
echo " 5. Verifying — does the bundle actually work?"
echo "=============================================================="
printf "  system curl + bundle -> Yahoo : "
curl -sS --cacert "${OUT}" -o "${TMPDIR_CA}/y.json" -w '%{http_code}\n' --max-time 25 \
  "https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=5d&interval=1d" 2>&1 | tail -1
echo "    body (first 100): $(head -c 100 "${TMPDIR_CA}/y.json" 2>/dev/null)"

python - "${OUT}" <<'PY'
import sys
bundle = sys.argv[1]
url = "https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=5d&interval=1d"
try:
    from curl_cffi import requests as cr
    s = cr.Session(impersonate="chrome", verify=bundle)
    r = s.get(url, timeout=25)
    body = r.text or ""
    print(f"  curl_cffi + bundle   -> Yahoo : {r.status_code}, {len(body)} bytes")
    print(f"    body (first 100): {body[:100]!r}")
    if r.status_code == 200 and '"chart"' in body:
        print("    >> TLS FIXED and Yahoo answering ✅")
    elif r.status_code == 429:
        print("    >> TLS fixed, but still throttled — the limit is real for this network.")
except Exception as exc:  # noqa: BLE001
    print(f"  curl_cffi + bundle   -> ERROR {type(exc).__name__}: {str(exc)[:120]}")
PY

echo
echo "=============================================================="
echo " 6. To make it stick"
echo "=============================================================="
cat <<EOF
  Add to your .env (the pipeline reads TC_CA_BUNDLE and passes it to
  curl_cffi, which ignores the Keychain and the SSL_CERT_FILE env var):

      TC_CA_BUNDLE=${OUT}

  And for everything else that reads the standard variables:

      export SSL_CERT_FILE=${OUT}
      export REQUESTS_CA_BUNDLE=${OUT}
      export CURL_CA_BUNDLE=${OUT}

  Re-run:  python scripts/build_platform_data.py
EOF
