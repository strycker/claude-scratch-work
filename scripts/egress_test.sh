#!/usr/bin/env bash
# Egress re-test for the Claude Code remote container.
#
# Checks whether outbound HTTPS is open from this session's container. The
# container's egress is governed by the environment's network policy (bound at
# session-creation time) and routed through the local agent proxy — NOT through
# your local machine, IP, or VPN. Run this after changing the network policy,
# in a FRESH session, to confirm the change reached the container.
#
# Usage:  bash scripts/egress_test.sh
#
# Interpreting results:
#   example.com = 200  -> general egress is OPEN (allow-all policy is live).
#   example.com = 403  -> still restricted; only the proxy's built-in bypass
#                         list works, so the policy change did NOT reach this
#                         container. (pypi.org shows 200 even when restricted --
#                         it's in the hardcoded bypass list, not a real test.)

set -u

# control hosts first, then the endpoints this project actually needs
hosts=(
  example.com                 # generic control: proves general egress
  pypi.org                    # bypass-list control: 200 even when restricted
  api.stlouisfed.org          # FRED API
  fred.stlouisfed.org         # FRED scrape / downloads
  www.multpl.com              # multpl.com scraper
  www.macrotrends.net         # macrotrends.net scraper
  query1.finance.yahoo.com    # yfinance (ETF prices)
)

printf "%-28s %s\n" "HOST" "RESULT"
printf "%-28s %s\n" "----" "------"
for h in "${hosts[@]}"; do
  out=$(curl -sS -o /dev/null -w "HTTP %{http_code}" --max-time 25 "https://$h/" 2>&1)
  printf "%-28s %s\n" "$h" "$out"
done

echo
echo "Recent proxy-side failures (reasons for any 403):"
curl -sS --max-time 15 "${HTTPS_PROXY:-http://127.0.0.1:36941}/__agentproxy/status" 2>/dev/null \
  | grep -E 'recentRelayFailures|host|detail|kind' || echo "  (status endpoint unreachable)"
