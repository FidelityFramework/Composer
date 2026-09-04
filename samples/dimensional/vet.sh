#!/usr/bin/env bash
# Dimensional vetting harness (clef docs/fidelity/phg/Dimensional_Vetting_Plan.md §4).
#
# Compiles every <rule>/<reject|accept>/*.fidproj under this directory with Composer, reads the
# diagnostics Composer prints, and compares the observed verdict and CCS code with the leaf's
# expect.toml. Prints one table (rule | expected | actual | code | pending) and a count.
#
# expect.toml keys (one-line strings, `key = "value"`):
#   verdict  "reject" | "accept"
#   code     the CCS code a reject must print ("" for accept; "" on a reject means the code is not yet
#            allocated, and any `error CCSxxxx` line satisfies the row)
#   count    minimum number of times the code must appear (default 1; one per reject site)
#   flags    extra Composer flags for this leaf (e.g. "--warnaserror" for a coverage finding)
#   pending  "step N": the hardening step (plan §5) that gates the row
#   rule     the one-line statement
#
# Verdict rules (plan §4): a rejected program is green only if the expected code appears on an error
# line (a Warning promoted under --warnaserror is printed as an error); an accepted program is green
# only if the exit code is 0 and no error is printed. Warnings are reported in the code column, not
# judged. A witness failure printed on stdout as "[ERROR] ..." without a diagnostic code is a reject
# with code "-".
#
# Usage: vet.sh [--through N] [--intermediates]
#   Rows whose gate step is <= N are judged; the rest are printed as pending. Exit status is 0 only
#   if every judged row matches. Default N = 0 (every row is pending: the baseline measurement).
#   Intermediates are kept on request (--intermediates passes -k, under <leaf>/targets/intermediates);
#   that costs about 60 s per leaf (83 MB of PSG JSON, mostly the platform library) against 7 s
#   without, and the diagnostics and the table are the same. Plan §4: intermediates on request.
#
# This script never calls tests/regression/Runner.fsx and never builds Composer. Rebuild first:
#   dotnet build /home/hhh/repos/Composer/src/Composer.fsproj
#
# Composer's diagnostic line format (src/CLI/Output.fs, emitDiagnostic):
#   <path>:<line>: <error|warning|info> <CODE>: <message>[ [unreachable]]
# The failure summary is "Error: Compilation failed with N error(s)" on stdout, exit status 1.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSER="${COMPOSER:-/home/hhh/repos/Composer/src/bin/Debug/net10.0/Composer}"
THROUGH=0
KEEP=""

while [ $# -gt 0 ]; do
    case "$1" in
        --through) THROUGH="$2"; shift 2 ;;
        --through=*) THROUGH="${1#--through=}"; shift ;;
        --intermediates) KEEP="-k"; shift ;;
        --no-intermediates) KEEP=""; shift ;;   # accepted for older invocations; the default
        -h|--help) sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "vet.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

case "$THROUGH" in
    ''|*[!0-9]*) echo "vet.sh: --through takes a step number, got '$THROUGH'" >&2; exit 2 ;;
esac

if [ ! -x "$COMPOSER" ]; then
    echo "vet.sh: Composer binary not found at $COMPOSER" >&2
    echo "        build it first: dotnet build /home/hhh/repos/Composer/src/Composer.fsproj" >&2
    exit 2
fi

echo "Dimensional vet: $HERE"
echo "Composer: $COMPOSER"
echo "Reminder: this harness does not build Composer; rebuild before trusting the table:"
echo "  dotnet build /home/hhh/repos/Composer/src/Composer.fsproj"
echo "Judging rows gated at step <= $THROUGH (--through); the rest are pending."
if [ -n "$KEEP" ]; then
    echo "Intermediates kept (-k, plan §4): about 60 s per leaf; pass --no-intermediates for the 7 s form."
fi
echo

# Plan §3 order. Rules found on disk but not listed here are appended alphabetically.
ORDER="UoM-1 UoM-2 UoM-3 UoM-4 UoM-5 UoM-6 UoM-7 UoM-8 UoM-9 W-1 W-2 W-3 W-4 W-5 W-6 NS-1 NS-2 NS-3 NS-4 M-1 M-2 M-3 M-4 M-5"

rules=""
for r in $ORDER; do
    [ -d "$HERE/$r" ] && rules="$rules $r"
done
for d in "$HERE"/*/; do
    r="$(basename "$d")"
    case " $rules " in *" $r "*) ;; *) [ -f "$d/reject/expect.toml" ] || [ -f "$d/accept/expect.toml" ] && rules="$rules $r" ;; esac
done

# toml_get <file> <key> -> value (empty if absent)
toml_get() {
    grep -E "^[[:space:]]*$2[[:space:]]*=[[:space:]]*\"" "$1" 2>/dev/null | head -1 \
        | sed -E 's/^[^=]*=[[:space:]]*"([^"]*)".*/\1/'
}

# Table accumulation (tab-separated rows; rendered at the end).
ROWS=""
DETAILS=""
total=0; matched=0; judged=0; judged_matched=0

add_row() { ROWS="$ROWS$1	$2	$3	$4	$5	$6
"; }

for rule in $rules; do
    for verdict_dir in reject accept; do
        leaf="$HERE/$rule/$verdict_dir"
        [ -f "$leaf/expect.toml" ] || continue

        exp_verdict="$(toml_get "$leaf/expect.toml" verdict)"
        exp_code="$(toml_get "$leaf/expect.toml" code)"
        exp_count="$(toml_get "$leaf/expect.toml" count)"; exp_count="${exp_count:-1}"
        exp_flags="$(toml_get "$leaf/expect.toml" flags)"
        exp_pending="$(toml_get "$leaf/expect.toml" pending)"
        exp_step="$(printf '%s' "$exp_pending" | grep -oE '[0-9]+' | head -1)"; exp_step="${exp_step:-0}"

        if [ "$exp_verdict" != "$verdict_dir" ]; then
            echo "vet.sh: $rule/$verdict_dir: expect.toml verdict '$exp_verdict' does not match its directory" >&2
            exit 2
        fi

        fidprojs="$(ls "$leaf"/*.fidproj 2>/dev/null | sort)"
        if [ -z "$fidprojs" ]; then
            echo "vet.sh: $rule/$verdict_dir: no .fidproj" >&2
            exit 2
        fi
        nproj="$(printf '%s\n' "$fidprojs" | wc -l)"

        for fidproj in $fidprojs; do
            base="$(basename "$fidproj" .fidproj)"
            label="$rule/$verdict_dir"
            if [ "$nproj" -gt 1 ]; then
                label="$label (${base#$rule.})"
            fi

            mkdir -p "$leaf/targets"
            out="$leaf/targets/vet.$base.stdout"
            err="$leaf/targets/vet.$base.stderr"
            # shellcheck disable=SC2086
            ( cd "$leaf" && NO_COLOR=1 "$COMPOSER" compile "$(basename "$fidproj")" $KEEP --no-color $exp_flags ) >"$out" 2>"$err"
            status=$?

            # Diagnostics printed by Composer, source lines only (dependency infos are not judged).
            error_lines="$(grep -E ': error [A-Z]+[0-9]+:' "$err" || true)"
            warn_lines="$(grep -E ': warning [A-Z]+[0-9]+:' "$err" || true)"
            witness_lines="$(grep -E '^\[ERROR\]|^Error: ' "$out" || true)"

            first_error_code="$(printf '%s\n' "$error_lines" | grep -oE 'error [A-Z]+[0-9]+' | head -1 | cut -d' ' -f2)"
            first_ccs_error="$(printf '%s\n' "$error_lines" | grep -oE 'error CCS[0-9]+' | head -1 | cut -d' ' -f2)"
            first_warn_code="$(printf '%s\n' "$warn_lines" | grep -oE 'warning [A-Z]+[0-9]+' | head -1 | cut -d' ' -f2)"

            if [ "$status" -eq 0 ] && [ -z "$error_lines" ]; then
                actual="accept"
            else
                actual="reject"
            fi

            # Code column: the first CCS error code, else the first error code of any series, else "-".
            code_col="${first_ccs_error:-${first_error_code:--}}"
            if [ -n "$first_warn_code" ]; then
                code_col="$code_col; warn $first_warn_code"
            fi

            ok=0
            if [ "$exp_verdict" = "reject" ]; then
                if [ -n "$exp_code" ]; then
                    n="$(printf '%s\n' "$error_lines" | grep -c "error $exp_code:")"
                    [ "$n" -ge "$exp_count" ] && ok=1
                else
                    [ "$status" -ne 0 ] && [ -n "$first_ccs_error" ] && ok=1
                fi
            else
                [ "$status" -eq 0 ] && [ -z "$error_lines" ] && [ -z "$witness_lines" ] && ok=1
            fi

            expected_col="$exp_verdict"
            if [ "$exp_verdict" = "reject" ]; then
                expected_col="reject ${exp_code:-(unallocated)}"
                [ "$exp_count" != "1" ] && expected_col="$expected_col x$exp_count"
            fi

            if [ "$exp_step" -le "$THROUGH" ]; then
                pending_col="judged"
                judged=$((judged + 1))
                [ $ok -eq 1 ] && judged_matched=$((judged_matched + 1))
            else
                pending_col="$exp_pending"
            fi

            if [ $ok -eq 1 ]; then mark="ok"; else mark="MISMATCH"; fi
            total=$((total + 1))
            [ $ok -eq 1 ] && matched=$((matched + 1))
            add_row "$label" "$expected_col" "$actual" "$code_col" "$pending_col" "$mark"

            if [ $ok -eq 0 ]; then
                first_line="$(printf '%s\n' "$error_lines" | head -1)"
                [ -z "$first_line" ] && first_line="$(printf '%s\n' "$witness_lines" | head -1)"
                [ -z "$first_line" ] && first_line="(no error printed; exit $status)"
                DETAILS="$DETAILS  $label: $first_line
"
            fi
        done
    done
done

# Render the table.
printf '%s' "$ROWS" | awk -F'\t' '
BEGIN { w1=length("rule"); w2=length("expected"); w3=length("actual"); w4=length("code"); w5=length("pending") }
{ rows[NR]=$0
  if (length($1)>w1) w1=length($1); if (length($2)>w2) w2=length($2); if (length($3)>w3) w3=length($3)
  if (length($4)>w4) w4=length($4); if (length($5)>w5) w5=length($5) }
END {
  fmt="%-" w1 "s | %-" w2 "s | %-" w3 "s | %-" w4 "s | %-" w5 "s | %s\n"
  printf fmt, "rule", "expected", "actual", "code", "pending", "match"
  line=""; for (i=0;i<w1+w2+w3+w4+w5+22;i++) line=line "-"; print line
  for (i=1;i<=NR;i++) { split(rows[i], f, "\t"); printf fmt, f[1], f[2], f[3], f[4], f[5], f[6] }
}'

echo
echo "$matched of $total rows match today; $judged_matched of $judged judged rows (gate step <= $THROUGH) match."
if [ -n "$DETAILS" ]; then
    echo
    echo "First printed error of each mismatching row (full transcripts in <leaf>/targets/vet.*.stdout|stderr):"
    printf '%s' "$DETAILS"
fi

if [ "$judged_matched" -eq "$judged" ]; then
    exit 0
else
    exit 1
fi
