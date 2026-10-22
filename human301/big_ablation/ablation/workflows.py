"""Workflows can orchestrate arbitrary phases and artifact dependencies."""
from __future__ import annotations

import time
from .specs import invoke


def supervised(context):
    data = invoke(context.config["data"], context)
    if context.config["fitting"] is not None:
        started = time.perf_counter()
        checkpoint = invoke(context.config["fitting"], context, data)
        context.recorder.update(training_wall_s=time.perf_counter()-started)
    else:
        checkpoint = context.artifacts.get("checkpoint")
    invoke(context.config["inference"], context, data, checkpoint)
    for analysis in context.config["analyses"]:
        invoke(analysis, context, data)


# A pretrain/finetune, teacher/student, multi-TF, estimator, or analysis-only
# workflow is a separate function with the same context argument. It can create
# child run contexts, register artifacts, invoke the shared recorder, and reuse
# data/fit/inference components. The runner does not assume a neural-network
# architecture, one optimizer update, or one model per TF.
