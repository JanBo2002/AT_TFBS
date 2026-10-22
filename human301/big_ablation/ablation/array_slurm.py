"""Slurm array commands; individual array elements retain explicit identities."""
from __future__ import annotations

import re

from .slurm import Slurm, state_name

JOB_ID = re.compile(r"^[0-9]+(?:_[0-9]+)?$")


class ArraySlurm(Slurm):
    def queue(self):
        output = self.command(["squeue", "--local", "--me", "--states=all", "--array",
                               "--noheader", "--format=%i|%T|%r|%j"])
        result = {}
        for line in output.splitlines():
            parts = line.strip().split("|", 3)
            if len(parts) == 4 and JOB_ID.fullmatch(parts[0]):
                result[parts[0]] = dict(state=state_name(parts[1]), reason=parts[2].strip(), name=parts[3].strip())
        return result

    def accounting(self, ids):
        # Query the array base IDs and request expanded JobID (not numeric
        # allocation IDs), so e.g. 123_4 is preserved throughout bookkeeping.
        bases = sorted({job_id.split("_", 1)[0] for job_id in ids})
        output = self.command(["sacct", "--local", "--allocations", "--array", "--noheader", "--parsable2",
                               "--jobs=" + ",".join(bases), "--format=JobID%80,State%32"])
        result = {}
        for line in output.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 2 and parts[0] in ids:
                result[parts[0]] = state_name(parts[1])
        return result

    def recover_array(self, name, created_date):
        matches = {job_id.split("_", 1)[0] for job_id, job in self.queue().items() if job["name"] == name}
        output = self.command(["sacct", "--local", "--allocations", "--array", "--noheader", "--parsable2",
                               f"--name={name}", f"--starttime={created_date}", "--format=JobID%80,JobName%80"])
        for line in output.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 2 and JOB_ID.fullmatch(parts[0]) and parts[1] == name:
                matches.add(parts[0].split("_", 1)[0])
        if len(matches) != 1:
            raise RuntimeError(f"Array receipt missing for {name}: found {len(matches)} arrays. "
                               "Check Slurm, then resume this batch; no duplicate array is submitted.")
        return matches.pop()

    def release(self, task_id, preceding):
        dependency = "after:" + ":".join(preceding) if preceding else ""
        self.command(["scontrol", "update", f"JobId={task_id}", f"Dependency={dependency}"])
        self.command(["scontrol", "release", task_id])
