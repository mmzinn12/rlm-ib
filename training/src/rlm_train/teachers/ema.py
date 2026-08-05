"""Exponential-moving-average teacher lifecycle."""

from __future__ import annotations

from typing import Any

from rlm_train.teachers.current_policy import CurrentPolicyTeacher


class EMATeacher(CurrentPolicyTeacher):
    def __init__(self, policy: Any, student_policy: Any, *, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must lie strictly between zero and one")
        super().__init__(policy, configuration={"strategy": "ema", "decay": decay})
        self.student_policy = student_policy
        self.decay = decay

    def after_optimizer_step(self) -> None:
        torch = __import__("torch")
        teacher_parameters = tuple(self.policy.trainable_parameters())
        student_parameters = tuple(self.student_policy.trainable_parameters())
        if len(teacher_parameters) != len(student_parameters):
            raise ValueError("EMA teacher and student parameters do not align")
        with torch.no_grad():
            for teacher, student in zip(teacher_parameters, student_parameters, strict=True):
                teacher.mul_(self.decay).add_(student.detach(), alpha=1.0 - self.decay)


__all__ = ["EMATeacher"]
