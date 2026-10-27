"""Generate a MoveIt-style self-collision SRDF from a spherized URDF.

Samples random configurations and disables link pairs that are Adjacent (share a joint),
collide at the Default (home) configuration, Always collide, or Never collide.
Pairs that only sometimes collide stay enabled -- that is the mask a planner needs.

    python scripts/generate_srdf.py spherized.urdf [--output spherized.srdf]
"""
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np
from fire import Fire

from foam import *
from foam.viz import load_robot


def main(
        urdf: str,
        output: str | None = None,
        n_samples: int = 10000,
        contact_margin: float = 0.,
        seed: int = 0,
    ):
    """`contact_margin`: spheres closer than this signed distance count as colliding."""
    link_spheres = get_link_spheres(load_urdf(Path(urdf)))
    if not link_spheres:
        raise SystemExit(f"No sphere collision geometry in {urdf}; is it spherized?")

    robot = load_robot(Path(urdf))
    links = list(link_spheres)
    adjacent = {tuple(sorted((j.parent, j.child))) for j in robot.robot.joints}
    disabled = {pair: "Adjacent" for pair in adjacent if set(pair) <= set(links)}
    pairs = [(a, b) for i, a in enumerate(links) for b in links[i + 1:] if tuple(sorted((a, b))) not in adjacent]

    # Flatten spheres; enumerate sphere pairs grouped contiguously by link pair.
    centers = np.array([s.origin for l in links for s in link_spheres[l]])
    radii = np.array([s.radius for l in links for s in link_spheres[l]])
    owner = np.repeat(np.arange(len(links)), [len(link_spheres[l]) for l in links])
    start = np.concatenate([[0], np.cumsum([len(link_spheres[l]) for l in links])])
    sphere_i, sphere_j, group_start = [], [], []
    for a, b in pairs:
        ia, ib = links.index(a), links.index(b)
        group_start.append(len(sphere_i))
        ii, jj = np.meshgrid(np.arange(start[ia], start[ia + 1]), np.arange(start[ib], start[ib + 1]))
        sphere_i.extend(ii.ravel())
        sphere_j.extend(jj.ravel())

    sphere_i, sphere_j = np.array(sphere_i), np.array(sphere_j)
    contact = radii[sphere_i] + radii[sphere_j] + contact_margin

    def collides(cfg: np.ndarray) -> np.ndarray:
        """(len(pairs),) bool: does each link pair overlap at cfg."""
        robot.update_cfg(cfg)
        T = np.stack([robot.get_transform(l) for l in links])[owner]
        world = np.einsum('nij,nj->ni', T[:, :3, :3], centers) + T[:, :3, 3]
        hit = np.linalg.norm(world[sphere_i] - world[sphere_j], axis = 1) < contact
        return np.logical_or.reduceat(hit, group_start)

    limits = [robot.joint_map[n].limit for n in robot.actuated_joint_names]
    lower = np.array([l.lower if l is not None and l.lower is not None else -np.pi for l in limits])
    upper = np.array([l.upper if l is not None and l.upper is not None else np.pi for l in limits])

    if pairs:
        default = collides(np.clip(np.zeros_like(lower), lower, upper))
        ever = np.zeros(len(pairs), dtype = bool)
        always = np.ones(len(pairs), dtype = bool)
            # ponytail: yourdfpy FK in a Python loop (~1k cfgs/s); batch the FK if this gets slow
        for cfg in np.random.default_rng(seed).uniform(lower, upper, (n_samples, len(lower))):
            hit = collides(cfg)
            ever |= hit
            always &= hit

        for k, pair in enumerate(pairs):
            reason = "Default" if default[k] else "Always" if always[k] else "Never" if not ever[k] else None
            if reason:
                disabled[tuple(sorted(pair))] = reason

    output = Path(output or Path(urdf).with_suffix('.srdf'))
    lines = [f'<?xml version="1.0" encoding="utf-8"?>', f'<robot name={quoteattr(robot.robot.name)}>']
    lines += [
        f'  <disable_collisions link1={quoteattr(a)} link2={quoteattr(b)} reason={quoteattr(r)}/>'
        for (a, b), r in sorted(disabled.items())
        ]
    output.write_text('\n'.join(lines + ['</robot>']) + '\n')

    counts = {r: list(disabled.values()).count(r) for r in set(disabled.values())}
    print(f"Wrote {output}: {len(disabled)} disabled pairs {counts}")


if __name__ == "__main__":
    Fire(main)
