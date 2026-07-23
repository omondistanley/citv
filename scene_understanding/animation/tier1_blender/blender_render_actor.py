"""Blender-side (``bpy``-only) render script for Tier 1.

Invoked exactly once per animation as:
    blender --background --factory-startup --python blender_render_actor.py -- --job job.json --out frames/

For each frame in the job spec, renders two passes with a camera whose FOV
is matched to the real photo's pinhole intrinsics (``fx``), so the actor
lands in the correct place/scale when composited back onto that photo in
``render_job.py``:
  - ``actor_NNNN.png``: the actor only, transparent background (RGBA).
  - ``shadow_NNNN.png``: only the actor's cast shadow, via Cycles' shadow-
    catcher ground plane, transparent everywhere else (grayscale alpha).

Nothing here builds or renders the real scene -- only the actor + its shadow,
per the plan's cross-cutting requirement; the real photo stays the
background and is composited in afterward by ``render_job.py``.

**This file cannot be imported or unit-tested outside a real Blender
install** (``bpy`` only exists inside Blender's bundled Python) -- there is
no Blender binary in the development environment this was written in.
Written carefully against the documented Cycles/bpy API, but the camera-FOV
match, shadow-catcher pass, and procedural rig are UNVERIFIED pending an
actual render on a machine with Blender installed. The most likely thing to
need empirical sign-correction is the ground-plane rotation (marked below).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import bpy  # only importable inside Blender's bundled Python


def _parse_args() -> Dict[str, str]:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("usage: blender --background --python blender_render_actor.py -- --job job.json --out frames/")
    argv = argv[argv.index("--") + 1 :]
    args: Dict[str, str] = {}
    it = iter(argv)
    for tok in it:
        if tok.startswith("--"):
            args[tok[2:]] = next(it)
    return args


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block_collection in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights):
        for block in list(block_collection):
            if block.users == 0:
                block_collection.remove(block)


def _setup_render_settings(image_size: List[int]) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"  # always-on CPU deployment target (Phase 4) -- no GPU dependency
    scene.cycles.samples = 32  # kept low: procedural geometry doesn't need many samples
    scene.render.resolution_x, scene.render.resolution_y = int(image_size[0]), int(image_size[1])
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"


def _setup_camera(intrinsics: Dict[str, float], image_size: List[int]) -> "bpy.types.Object":
    cam_data = bpy.data.cameras.new("citv_cam")
    cam_obj = bpy.data.objects.new("citv_cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Camera at the world origin, rotated 180 deg about local X from Blender's
    # default (-Z forward, +Y up) so it instead looks down +Z with -Y "up" --
    # i.e. directly matches this project's pinhole convention (Z=depth,
    # Y increases downward like image v) with NO per-point coordinate remap
    # needed: job_spec world_xyz can be used as object.location verbatim.
    cam_obj.location = (0.0, 0.0, 0.0)
    cam_obj.rotation_euler = (math.pi, 0.0, 0.0)

    cam_data.sensor_fit = "HORIZONTAL"
    fx = float(intrinsics.get("fx", image_size[0]))
    cam_data.angle = 2.0 * math.atan((image_size[0] / 2.0) / fx)
    cam_data.clip_start, cam_data.clip_end = 0.01, 1000.0
    return cam_obj


def _setup_lighting_and_ground(ground_y: float) -> Dict[str, "bpy.types.Object"]:
    sun_data = bpy.data.lights.new("citv_sun", type="SUN")
    sun_data.energy = 3.0
    sun_obj = bpy.data.objects.new("citv_sun", sun_data)
    bpy.context.collection.objects.link(sun_obj)
    # Shines toward +Y (this convention's "down"), i.e. from above per the
    # camera's -Y-is-up orientation -- a plausible default overhead light.
    sun_obj.rotation_euler = (math.radians(35.0), 0.0, math.radians(20.0))

    bpy.ops.mesh.primitive_plane_add(size=200.0, location=(0.0, ground_y, 0.0))
    ground = bpy.context.active_object
    ground.name = "citv_ground_shadow_catcher"
    # NOTE (unverified sign): rotates the plane's default +Z-facing normal to
    # face -Y ("up" in this scene's convention) so it lies horizontal under
    # the actor instead of standing upright. If shadows render on the wrong
    # side of the actor once tested on real Blender, flip this to +pi/2.
    ground.rotation_euler = (-math.pi / 2.0, 0.0, 0.0)
    ground.is_shadow_catcher = True
    return {"sun": sun_obj, "ground": ground}


def _apply_actor_material(parts: Dict[str, Any], base_color: List[float]) -> None:
    """Principled BSDF + shade_smooth for every mesh part, shared by every
    creature build -- previously no material was ever set (bare Blender
    default, flat/faceted shading), which reinforced the artificial look
    independent of which shape was used. ``base_color`` (0-1 RGB, from
    open-vocab color-text extraction, see motion_contract.py) tints every
    part uniformly."""
    mat = bpy.data.materials.new(name=f"{parts['root'].name}_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (*base_color[:3], 1.0)
        bsdf.inputs["Roughness"].default_value = 0.55
    mesh_parts = [parts["body"], parts["head"], *parts["limbs"], *parts.get("extras", [])]
    for part in mesh_parts:
        part.data.materials.append(mat)
        bpy.context.view_layer.objects.active = part
        bpy.ops.object.shade_smooth()


def _build_creature(attrs: Dict[str, Any], name: str = "citv_actor") -> Dict[str, Any]:
    """Single parametric builder composing a body + head + legs (0/2/4) +
    optional wings + optional tail from ``attrs`` (see
    ``ground_plane.classify_actor_attributes_openvocab``) -- replaces the
    old fixed humanoid/quadruped-only functions so any combination of body
    attributes the open-vocab classifier infers gets a plausible silhouette
    (a bird gets wings + 2 legs + a horizontal body; a snake gets an
    elongated body with no legs at all) instead of being forced into one of
    a handful of predefined named shapes. No armature/bones -- direct
    per-part keyframing (see ``_pose_actor``) is far more robust to get
    right without a way to test the result.

    A plain cylinder/cone/cube, not the Extra Objects add-on's shapes: this
    script runs under --factory-startup, which disables non-default
    add-ons, so only always-available default primitives are usable here.
    """
    leg_count = attrs.get("leg_count", 2)
    has_wings = bool(attrs.get("has_wings", False))
    has_tail = bool(attrs.get("has_tail", False))
    elongated = bool(attrs.get("elongated", False))
    # A winged biped (bird) or any 4-legged/legless creature carries its
    # body horizontally; a wingless biped (humanoid) stands upright.
    horizontal_body = has_wings or leg_count != 2

    bpy.ops.object.empty_add(type="PLAIN_AXES")
    root = bpy.context.active_object
    root.name = name

    if horizontal_body:
        body_depth = 1.1 if elongated else 0.7
        body_radius = 0.11 if elongated else (0.2 if leg_count == 4 else 0.15)
        body_y = -0.5 if leg_count == 0 else -0.3
        bpy.ops.mesh.primitive_cylinder_add(radius=body_radius, depth=body_depth, rotation=(0.0, math.radians(90.0), 0.0))
        body = bpy.context.active_object
        body.location = (0.0, body_y, 0.0)
        head_loc = (0.0, body_y - 0.05, body_depth / 2.0 + 0.1)
    else:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.2, depth=0.9)
        body = bpy.context.active_object
        body.location = (0.0, -0.6, 0.0)  # -Y is "up": lift the body above the root/foot point
        head_loc = (0.0, -1.05, 0.0)
    body.name = f"{name}_body"
    body.parent = root

    limbs: List["bpy.types.Object"] = []
    if leg_count == 4:
        for i, (x_off, z_off) in enumerate([(-0.15, -0.3), (0.15, -0.3), (-0.15, 0.3), (0.15, 0.3)]):
            bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=0.35)
            limb = bpy.context.active_object
            limb.name = f"{name}_limb_{i}"
            limb.location = (x_off, -0.12, z_off)
            limb.parent = root
            limbs.append(limb)
    elif leg_count == 2 and horizontal_body:
        # Bipedal but horizontal-bodied (bird): 2 thin legs under the body, no arms.
        for i, x_off in enumerate([-0.1, 0.1]):
            bpy.ops.mesh.primitive_cylinder_add(radius=0.035, depth=0.3)
            limb = bpy.context.active_object
            limb.name = f"{name}_limb_{i}"
            limb.location = (x_off, -0.15, 0.0)
            limb.parent = root
            limbs.append(limb)
    elif leg_count == 2:
        # Upright bipedal (humanoid): 2 arms + 2 legs.
        for i, (x_off, is_arm) in enumerate([(-0.18, True), (0.18, True), (-0.12, False), (0.12, False)]):
            bpy.ops.mesh.primitive_cylinder_add(radius=0.06, depth=0.55)
            limb = bpy.context.active_object
            limb.name = f"{name}_limb_{i}"
            y_off = -0.85 if is_arm else -0.3  # arms hang near the top, legs near the bottom/foot
            limb.location = (x_off, y_off, 0.0)
            limb.parent = root
            limbs.append(limb)
    # leg_count == 0 (snake/fish/legless): no limbs at all.

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.14 if not horizontal_body else 0.12)
    head = bpy.context.active_object
    head.name = f"{name}_head"
    head.location = head_loc
    head.parent = root

    extras: List["bpy.types.Object"] = []
    if has_wings:
        for i, x_sign in enumerate([-1.0, 1.0]):
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            wing = bpy.context.active_object
            wing.name = f"{name}_wing_{i}"
            wing.scale = (0.35, 0.02, 0.16)
            wing.location = (x_sign * 0.32, body.location.y + 0.03, 0.0)
            wing.rotation_euler = (0.0, 0.0, math.radians(12.0 * x_sign))
            wing.parent = root
            extras.append(wing)
    if has_tail:
        bpy.ops.mesh.primitive_cone_add(radius1=0.09, radius2=0.01, depth=0.4)
        tail_z = (body.dimensions.z / 2.0 + 0.2) if horizontal_body else 0.0
        tail_y = body.location.y if horizontal_body else -0.25
        tail = bpy.context.active_object
        tail.name = f"{name}_tail"
        tail.rotation_euler = (math.radians(90.0), 0.0, 0.0) if horizontal_body else (0.0, 0.0, 0.0)
        tail.location = (0.0, tail_y, -tail_z if horizontal_body else 0.0)
        tail.parent = root
        extras.append(tail)

    return {"root": root, "body": body, "limbs": limbs, "head": head, "extras": extras}

# (bob_amplitude_m, bob_frequency_hz, limb_swing_deg, forward_lean_deg) per motion label.
_MOTION_STYLE = {
    "walk": (0.03, 1.8, 22.0, 0.0),
    "run": (0.06, 3.2, 38.0, 12.0),
    "crawl": (0.015, 1.0, 10.0, 45.0),
    "climb": (0.02, 1.2, 25.0, 20.0),
    "jump": (0.0, 0.0, 5.0, -5.0),
    "descend": (0.02, 1.4, 20.0, -8.0),
    # swim/fly (Phase 2.5, paths_export.py's terrain-aware motion override):
    # the procedural blob actor has no wing/fin geometry, so these are a
    # rough style approximation (reduced bob/limb-swing, slight tilt), not a
    # real flight/swim rig -- an accepted scope limit, not a bug.
    "swim": (0.02, 1.2, 15.0, 8.0),
    "fly": (0.015, 0.8, 8.0, -10.0),
}


_ENERGY_SCALE = {"high": 1.6, "low": 0.5, "normal": 1.0}


def _pose_actor(parts: Dict[str, Any], frame_index: int, frame_state: Dict[str, Any], action_style: Dict[str, Any]) -> None:
    x, y, z = frame_state["world_xyz"]
    heading_rad = math.radians(frame_state["heading_deg"])
    scale = float(frame_state.get("scale", 1.0))
    bob_amp, bob_freq, limb_swing_deg, lean_deg = _MOTION_STYLE.get(frame_state.get("motion", "walk"), _MOTION_STYLE["walk"])
    # action_text-driven style modulation (Phase 2.6) -- on top of the path
    # geometry's own motion label, not a replacement for it: "static" freezes
    # bob/limb-swing regardless of what the path says; energy scales them.
    if action_style.get("is_static"):
        bob_amp, limb_swing_deg = 0.0, 0.0
    else:
        energy_scale = _ENERGY_SCALE.get(action_style.get("energy", "normal"), 1.0)
        bob_amp *= energy_scale
        limb_swing_deg *= energy_scale

    root = parts["root"]
    root.location = (x, y, z)
    root.rotation_euler = (0.0, heading_rad, 0.0)
    root.scale = (scale, scale, scale)

    # Retargeted motion (from an uploaded clip via MediaPipe Pose, see
    # motion_retargeting.py) overrides the canned sine-wave style per
    # channel when present -- real motion *shape/timing*, our own puppet's
    # amplitude bounds (retargeted_limb_swing is normalized to roughly
    # [-1, 1], so it's scaled by this motion's own swing amplitude rather
    # than trusting an arbitrary real-world-to-puppet unit correspondence).
    if "retargeted_bob" in frame_state:
        bob = float(frame_state["retargeted_bob"]) * 1.5  # empirical gain: normalized hip-position units -> puppet meters
    else:
        phase = frame_index * bob_freq * 0.35
        bob = bob_amp * math.sin(phase)
    parts["body"].location.y = -0.6 - bob  # -Y is up: subtract to bob upward

    lean = float(frame_state["retargeted_lean_deg"]) if "retargeted_lean_deg" in frame_state else lean_deg
    parts["body"].rotation_euler = (math.radians(max(-60.0, min(60.0, lean))), 0.0, 0.0)

    if "retargeted_limb_swing" in frame_state:
        swing = math.radians(limb_swing_deg) * float(frame_state["retargeted_limb_swing"])
    else:
        phase = frame_index * bob_freq * 0.35
        swing = math.radians(limb_swing_deg) * math.sin(phase)
    for i, limb in enumerate(parts["limbs"]):
        limb.rotation_euler = (swing if i % 2 == 0 else -swing, 0.0, 0.0)

    for part in [root, parts["body"], parts["head"], *parts["limbs"]]:
        part.keyframe_insert(data_path="location", frame=frame_index)
        part.keyframe_insert(data_path="rotation_euler", frame=frame_index)
    root.keyframe_insert(data_path="scale", frame=frame_index)


def _render_frame(out_path: Path) -> None:
    bpy.context.scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = _parse_args()
    job = json.loads(Path(args["job"]).read_text())
    out_dir = Path(args["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    _clear_scene()
    _setup_render_settings(job["image_size"])
    _setup_camera(job["intrinsics"], job["image_size"])

    frames: List[Dict[str, Any]] = job["frames"]
    ground_y = max(f["world_xyz"][1] for f in frames) if frames else 0.0  # lowest-on-screen = largest Y in this convention
    rig = _setup_lighting_and_ground(ground_y)

    actor_attrs = job.get("actor_attrs") or {"leg_count": 2, "has_wings": False, "has_tail": False, "elongated": False}
    action_style = job.get("action_style") or {"is_static": False, "energy": "normal"}
    parts = _build_creature(actor_attrs)
    _apply_actor_material(parts, job.get("color_rgb") or [0.55, 0.55, 0.55])

    for i, frame_state in enumerate(frames):
        _pose_actor(parts, i, frame_state, action_style)
        bpy.context.scene.frame_set(i)

        # Pass 1: actor only (ground hidden) -> actor_NNNN.png (RGBA, transparent bg).
        rig["ground"].hide_render = True
        parts["root"].hide_render = False
        _render_frame(out_dir / f"actor_{i:04d}.png")

        # Pass 2: shadow only (actor hidden, ground as Cycles shadow-catcher)
        # -> shadow_NNNN.png (alpha channel used as a grayscale shadow mask
        # by render_job.py; RGB is irrelevant for a shadow-catcher pass).
        rig["ground"].hide_render = False
        parts["root"].hide_render = True
        _render_frame(out_dir / f"shadow_{i:04d}.png")

    print(f"[blender_render_actor] wrote {len(frames)} actor+shadow frame pairs -> {out_dir}")


if __name__ == "__main__":
    main()
