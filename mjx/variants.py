"""Ablation variants of the stock mjlab Go1 tasks, registered under new task IDs.

Run:  python mjx/variants.py <TASK-ID> [any normal mjlab train flags]
      e.g. python mjx/variants.py Lc-Go1-Flat-NoDR --env.scene.num-envs 4096
"""
# [permanent practice] One named task per ablation: the run name alone tells you what changed.
from mjlab.tasks.registry import register_mjlab_task, load_env_cfg, load_rl_cfg, load_runner_cls
import mjlab.tasks  # noqa: F401  (registers the stock tasks first)

FLAT, ROUGH = "Mjlab-Velocity-Flat-Unitree-Go1", "Mjlab-Velocity-Rough-Unitree-Go1"
DR_EVENTS = ("push_robot", "encoder_bias", "base_com",
             "foot_friction_slide", "foot_friction_spin", "foot_friction_roll")


def no_dr(cfg):
    for k in DR_EVENTS:
        cfg.events.pop(k, None)
    return cfg


def drop(*keys):
    """Remove-one ablation: return an edit that pops the given event terms."""
    def edit(cfg):
        for k in keys:
            cfg.events.pop(k, None)
        return cfg
    return edit


def no_terrain_curriculum(cfg):
    cfg.curriculum.pop("terrain_levels", None)
    return cfg


VARIANTS = {
    "Lc-Go1-Flat-NoDR": (FLAT, no_dr),
    "Lc-Go1-Flat-NoPush": (FLAT, drop("push_robot")),
    "Lc-Go1-Flat-NoEncoderBias": (FLAT, drop("encoder_bias")),
    "Lc-Go1-Flat-NoBaseCom": (FLAT, drop("base_com")),
    "Lc-Go1-Flat-NoFootFriction": (FLAT, drop("foot_friction_slide", "foot_friction_spin", "foot_friction_roll")),
    "Lc-Go1-Rough-NoCurriculum": (ROUGH, no_terrain_curriculum),
}

for new_id, (base, edit) in VARIANTS.items():
    register_mjlab_task(
        task_id=new_id,
        env_cfg=edit(load_env_cfg(base)),
        play_env_cfg=edit(load_env_cfg(base, play=True)),
        rl_cfg=load_rl_cfg(base),
        runner_cls=load_runner_cls(base),
    )

if __name__ == "__main__":
    from mjlab.scripts.train import main
    main()
