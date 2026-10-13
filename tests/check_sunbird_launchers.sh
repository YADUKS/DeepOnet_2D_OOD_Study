#!/bin/bash
# Check interpreter selection and the environment forwarded to CPU/GPU jobs.
# Runtime names point to a harmless host library to check environment forwarding.
# This does not replace an NGSolve/PyTorch import check on the cluster.
set -euo pipefail
trap 'printf "Launcher check failed at line %s: %s\nOutput: %s\n" "$LINENO" "$BASH_COMMAND" "${output:-}" >&2' ERR
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fixture="$(mktemp -d)"
case "$(uname -s)" in
    MINGW*|MSYS*) test_library="/usr/bin/msys-z.dll" ;;
    *) test_library="$(ldd "$(command -v bash)" | awk '/=> \// {print $3; exit}')" ;;
esac
[[ -f "$test_library" ]]
mkdir -p "$fixture/env/bin" "$fixture/.venv/bin" "$fixture/override/bin" "$fixture/venv_only/.venv/bin"
for name in env .venv override; do
    cat > "$fixture/$name/bin/python" <<'PYTHON'
#!/bin/bash
printf 'SELECTED=%s\n' "${0%/bin/python}"
printf 'PRELOAD=%s\n' "${LD_PRELOAD:-unset}"
printf 'ARGS=%s\n' "$*"
PYTHON
    chmod +x "$fixture/$name/bin/python"
done
cp "$fixture/.venv/bin/python" "$fixture/venv_only/.venv/bin/python"
chmod +x "$fixture/venv_only/.venv/bin/python"

run_launcher() {
    env -u STUDY_ENV -u STUDY_PYTHON -u STUDY_MODULE -u LD_PRELOAD \
        STUDY_PROJECT_ROOT="$fixture" SLURM_CPUS_PER_TASK=3 "$@" \
        bash "$repo_dir/$script" smoke 2> "$fixture/launcher_stderr.log"
}

for script in jobs/sunbird/cpu.sh jobs/sunbird/gpu.sh; do
    bash -n "$repo_dir/$script"
    output=$(run_launcher)
    [[ "$output" == *"SELECTED=$fixture/env"* ]]
    output=$(run_launcher STUDY_ENV="$fixture/override")
    [[ "$output" == *"SELECTED=$fixture/override"* ]]
    output=$(run_launcher STUDY_PYTHON="$fixture/.venv/bin/python" STUDY_ENV="$fixture/override")
    [[ "$output" == *"SELECTED=$fixture/.venv"* ]]
    [[ "$output" == *'PRELOAD=unset'* ]]
    output=$(run_launcher STUDY_PROJECT_ROOT="$fixture/venv_only")
    [[ "$output" == *"SELECTED=$fixture/venv_only/.venv"* ]]
    if run_launcher STUDY_ENV="$fixture/missing" > "$fixture/missing.log"; then
        printf 'ERROR: explicit invalid environment was ignored\n' >&2
        exit 1
    else
        [[ $? -eq 2 ]]
    fi

    mkdir -p "$fixture/env/lib" "$fixture/override/lib"
    ln -sf "$test_library" "$fixture/env/lib/libgcc_s.so.1"
    ln -sf "$test_library" "$fixture/env/lib/libstdc++.so.6"
    ln -sf "$test_library" "$fixture/override/lib/libstdc++.so.6"
    expected="$fixture/env/lib/libgcc_s.so.1:$fixture/env/lib/libstdc++.so.6"
    output=$(run_launcher LD_PRELOAD="$test_library")
    [[ "$output" == *"PRELOAD=$expected:$test_library"* ]]
    [[ "$output" == *"Native runtime: $expected"* ]]
    if [[ "$script" == */cpu.sh ]]; then
        [[ "$output" == *'ARGS=-u run.py smoke --device cpu --cpu-threads 3'* ]]
    else
        [[ "$output" == *'ARGS=-u run.py smoke --device cuda --cpu-threads 3'* ]]
    fi
    output=$(run_launcher STUDY_PYTHON="$fixture/override/bin/python" STUDY_ENV="$fixture/env")
    [[ "$output" == *"PRELOAD=$fixture/override/lib/libstdc++.so.6"* ]]
    output=$(run_launcher STUDY_PYTHON="$fixture/.venv/bin/python" STUDY_ENV="$fixture/env")
    [[ "$output" == *'PRELOAD=unset'* ]]
    printf '%s: interpreter selection, runtime forwarding and command arguments passed.\n' "$script"
done
bash -n "$repo_dir/docs/manual/commands_sunbird.txt"
printf 'Sunbird guide syntax passed. Fixtures: %s\n' "$fixture"
