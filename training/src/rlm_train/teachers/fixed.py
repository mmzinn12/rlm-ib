"""Immutable checkpoint teacher strategy."""

from rlm_train.teachers.current_policy import CurrentPolicyTeacher


class FixedTeacher(CurrentPolicyTeacher):
    """A teacher whose policy parameters are never updated by this lifecycle."""


__all__ = ["FixedTeacher"]
