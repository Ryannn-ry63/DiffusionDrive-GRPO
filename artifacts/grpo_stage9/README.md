# DiffusionDrive GRPO stage-9 artifacts

This directory is intentionally independent of `grpo_stage0` through
`grpo_stage8`. The registered runners write selector and generation checkpoints
here, including steps 128/512/1024/2048 and epoch-1. Generation is started only
after the selector gate fails.
