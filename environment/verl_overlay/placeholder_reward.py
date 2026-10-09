"""Cheap label-free placeholder used during rollout streaming.

The driver-side batch hook overwrites these zeros after the complete group is
available. Gold labels are never read.
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    extra_info = extra_info or {}
    return {
        "score": 0.0,
        "placeholder_reward": True,
        # Preserve label-free source identity through the async reward loop so
        # the driver-side group audit can map UUID groups back to frozen routes.
        "source_question_id": extra_info.get("question_id"),
        "source_row": extra_info.get("source_row", extra_info.get("index")),
        "route_arm": extra_info.get("route_arm"),
    }
