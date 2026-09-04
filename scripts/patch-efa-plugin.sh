#!/usr/bin/env bash
# Teach the EFA device plugin about newer GPU instance types.
#
# WHY THIS EXISTS
#   aws-efa-k8s-device-plugin gates itself with a nodeAffinity carrying a HARDCODED list of
#   EFA-capable instance types. That list lags new hardware. On a fresh g7e node group the
#   daemonset reported DESIRED=0, so `vpc.amazonaws.com/efa` was never advertised, and any
#   pod requesting the EFA resource stayed Pending forever with no obvious cause.
#
#   The nodes were fine: `aws ec2 describe-instance-types` reports EfaSupported=true for
#   g7e.12xlarge, and each instance had an EFA interface attached. Only the plugin's
#   allow-list was stale.
#
#   Run this after creating a node group whose instance type is newer than the installed
#   plugin. Idempotent: adds only missing entries.
#
# USAGE
#   scripts/patch-efa-plugin.sh <kube-context> [instance-type ...]
#   scripts/patch-efa-plugin.sh efa-eu g7e.12xlarge g7e.24xlarge g7e.48xlarge
set -euo pipefail

CTX="${1:?usage: patch-efa-plugin.sh <kube-context> [instance-type ...]}"
shift || true
TYPES=("$@")
if [ ${#TYPES[@]} -eq 0 ]; then
  TYPES=(g7e.12xlarge g7e.24xlarge g7e.48xlarge)
fi

echo "context        : $CTX"
echo "adding types   : ${TYPES[*]}"

CTX="$CTX" TYPES="${TYPES[*]}" python3 - <<'PY' > /tmp/efa-plugin-patch.json
import json, os, subprocess, sys

ctx = os.environ["CTX"]
want = os.environ["TYPES"].split()

raw = subprocess.run(
    ["kubectl", "--context", ctx, "get", "ds", "-n", "kube-system",
     "aws-efa-k8s-device-plugin", "-o", "json", "--request-timeout=60s"],
    capture_output=True, text=True)
if raw.returncode != 0:
    sys.exit(f"could not read daemonset: {raw.stderr.strip()}")

ds = json.loads(raw.stdout)
aff = ds["spec"]["template"]["spec"].get("affinity")
if not aff:
    sys.exit("daemonset has no affinity block; nothing to patch")

terms = aff["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
vals = terms[0]["matchExpressions"][0]["values"]
before = len(vals)
added = [t for t in want if t not in vals]
vals.extend(added)

print(json.dumps({"spec": {"template": {"spec": {"affinity": aff}}}}))
print(f"allow-list {before} -> {len(vals)}; added: {added or '(none, already present)'}",
      file=sys.stderr)
PY

kubectl --context "$CTX" patch ds -n kube-system aws-efa-k8s-device-plugin \
  --type=merge --patch-file /tmp/efa-plugin-patch.json --request-timeout=60s

rm -f /tmp/efa-plugin-patch.json
echo
echo "waiting for the plugin to roll out ..."
kubectl --context "$CTX" rollout status ds/aws-efa-k8s-device-plugin -n kube-system --timeout=180s || true
echo
echo "EFA allocatable per node (expect >=1 on the new nodes):"
kubectl --context "$CTX" get nodes --request-timeout=60s \
  -o custom-columns='NAME:.metadata.name,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,GPU:.status.allocatable.nvidia\.com/gpu,EFA:.status.allocatable.vpc\.amazonaws\.com/efa'
