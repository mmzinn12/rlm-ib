"""Verify fixed, EMA, and no-teacher lifecycle behavior without a trainer."""

import pytest

from rlm_train.sdpo import (
    TeacherStrategy,
    TorchEMATeacherController,
    TorchFixedTeacherController,
    build_torch_teacher_controller,
)


def test_fixed_teacher_fingerprint_and_parameters_remain_unchanged_after_student_steps():
    torch = pytest.importorskip("torch")
    student = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        student.weight.fill_(2.0)
    controller = TorchFixedTeacherController.from_student(
        student,
        checkpoint_identity="checkpoint-before-training",
    )
    before = controller.identity()
    with torch.no_grad():
        student.weight.fill_(9.0)

    controller.update_after_optimizer_step(student)
    after = controller.identity()

    assert before == after
    torch.testing.assert_close(controller.teacher.weight, torch.full((1, 2), 2.0))
    assert all(not parameter.requires_grad for parameter in controller.teacher.parameters())


def test_fixed_teacher_detects_illegal_mutation():
    torch = pytest.importorskip("torch")
    controller = TorchFixedTeacherController.from_student(torch.nn.Linear(1, 1))
    with torch.no_grad():
        controller.teacher.weight.add_(1.0)

    with pytest.raises(RuntimeError, match="fixed teacher state changed"):
        controller.validate_unchanged()


def test_teacher_factory_supports_none_fixed_and_ema():
    torch = pytest.importorskip("torch")
    student = torch.nn.Linear(1, 1)

    assert build_torch_teacher_controller(TeacherStrategy.NONE, student) is None
    assert isinstance(
        build_torch_teacher_controller(TeacherStrategy.FIXED, student),
        TorchFixedTeacherController,
    )
    assert isinstance(
        build_torch_teacher_controller(TeacherStrategy.EMA, student),
        TorchEMATeacherController,
    )
