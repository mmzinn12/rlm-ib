"""Lifecycle tests for canonical fixed and EMA teachers."""

import copy

import pytest

from rlm_train.teachers import EMATeacher, FixedTeacher


class TorchPolicy:
    def __init__(self, module):
        self.module = module

    def trainable_parameters(self):
        return self.module.parameters()


def test_fixed_teacher_has_no_optimizer_step_mutation():
    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(2, 1, bias=False)
    policy = TorchPolicy(copy.deepcopy(module))
    teacher = FixedTeacher(policy)
    before = tuple(parameter.detach().clone() for parameter in policy.trainable_parameters())

    teacher.after_optimizer_step()

    for expected, actual in zip(before, policy.trainable_parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_ema_teacher_updates_toward_student_after_optimizer_step():
    torch = pytest.importorskip("torch")
    student_module = torch.nn.Linear(1, 1, bias=False)
    teacher_module = copy.deepcopy(student_module)
    with torch.no_grad():
        student_module.weight.fill_(2.0)
        teacher_module.weight.fill_(0.0)
    teacher = EMATeacher(
        TorchPolicy(teacher_module),
        TorchPolicy(student_module),
        decay=0.75,
    )

    teacher.after_optimizer_step()

    torch.testing.assert_close(teacher_module.weight, torch.full((1, 1), 0.5))
