"""Small Slurm adapter. All commands use argv lists, never shell interpolation."""
from __future__ import annotations

import os
import re
import subprocess


TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "OUT_OF_MEMORY", "DEADLINE"}


class SlurmError(RuntimeError):
    pass


def state_name(value):
    return value.strip().split()[0].rstrip("+*").upper() if value.strip() else "UNKNOWN"


def resource_options(options):
    """Normalize resource options; the dispatcher owns job count and dependencies."""
    allowed = {"--account", "--qos", "--partition", "--time", "--cpus-per-task",
               "--mem", "--mem-per-cpu", "--mem-per-gpu", "--gres", "--gpus",
               "--gpus-per-node", "--gpus-per-task", "--constraint", "--reservation",
               "--nodelist", "--exclude", "--licenses", "--hint", "--nice"}
    result = []
    index = 0
    while index < len(options):
        token = options[index]
        key, separator, value = token.partition("=")
        if key == "--exclusive" and not separator:
            result.append(key)
        else:
            if key not in allowed:
                raise ValueError(f"Unsupported sbatch option {key}; use long resource/account options")
            if not separator:
                index += 1
                if index >= len(options) or options[index].startswith("--"):
                    raise ValueError(f"Missing value for {key}")
                value = options[index]
            if not value:
                raise ValueError(f"Empty value for {key}")
            result.append(f"{key}={value}")
        index += 1
    return result


class Slurm:
    def command(self, args):
        # Inherited SBATCH_ARRAY_INX or SBATCH_DEPENDENCY must not bypass the cap
        # or order. The activated Python/CUDA environment is otherwise retained.
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("SBATCH_") and k not in {"SLURM_CLUSTERS", "SLURM_FEDERATION"}}
        env["LC_ALL"] = "C"
        try:
            result = subprocess.run(args, capture_output=True, text=True, env=env, timeout=45)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SlurmError(f"{args[0]}: {exc}") from exc
        if result.returncode:
            raise SlurmError(f"{args[0]} failed: {result.stderr.strip() or result.stdout.strip()}")
        return result.stdout

    def submit(self, args):
        output = self.command(["sbatch", "--parsable", *args]).strip()
        if not re.fullmatch(r"[0-9]+(?:;[^\s;]+)?", output):
            raise SlurmError(f"Unrecognized sbatch receipt: {output!r}")
        return output.split(";", 1)[0]

    def queue(self):
        output = self.command(["squeue", "--local", "--me", "--states=all", "--noheader",
                               "--format=%i|%T|%j"])
        result = {}
        for line in output.splitlines():
            parts = line.strip().split("|", 2)
            if len(parts) == 3 and parts[0].isdigit():
                result[parts[0]] = {"state": state_name(parts[1]), "name": parts[2].strip()}
        return result

    def accounting(self, ids):
        output = self.command(["sacct", "--local", "--allocations", "--noheader", "--parsable2",
                               "--jobs=" + ",".join(ids), "--format=JobIDRaw,State%32"])
        result = {}
        for line in output.splitlines():
            parts = line.strip().split("|")
            if len(parts) == 2 and parts[0] in ids:
                result[parts[0]] = state_name(parts[1])
        return result

    def recover(self, name, created_date):
        matches = {job_id for job_id, job in self.queue().items() if job["name"] == name}
        output = self.command(["sacct", "--local", "--allocations", "--noheader", "--parsable2",
                               f"--name={name}", f"--starttime={created_date}",
                               "--format=JobIDRaw,State%32,JobName%80"])
        for line in output.splitlines():
            parts = line.strip().split("|")
            if len(parts) == 3 and parts[0].isdigit() and parts[2].strip() == name:
                matches.add(parts[0])
        if len(matches) != 1:
            raise RuntimeError(f"Submission receipt missing for {name}: found {len(matches)} jobs. "
                               "The batch is paused to avoid duplicate submissions. Check the Slurm logs, "
                               "then resume this batch once the job is visible.")
        return matches.pop()
