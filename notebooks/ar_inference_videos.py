"""
Run Alpamayo-R1 on a clip, then write a video with trajectory and chain-of-thought overlays.
"""

from __future__ import annotations

import argparse
import copy
import io
import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mediapy as mp
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import (
    load_physical_aiavdataset,
    physical_ai_av_interface,
)
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


# Reference-style colors (dark HUD / teal accent)
_COLOR_BG = (13, 27, 42, 230)
_COLOR_BORDER = (64, 224, 208, 255)
_COLOR_TEXT = (255, 255, 255, 255)
_COLOR_HEADER = (64, 224, 208, 255)
_COLOR_TS_BG = (0, 0, 0, 220)


def _fig_to_rgb(fig: Figure, dpi: int = 120) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="#0d1b2a",
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def render_trajectory_panel(
    hist_xy: np.ndarray,
    pred_xy: np.ndarray,
    inset_px: tuple[int, int] = (420, 340),
    lon_range: tuple[float, float] = (-20.0, 80.0),
    lat_range: tuple[float, float] = (-20.0, 20.0),
) -> np.ndarray:
    """Bird's-eye panel: lateral (x-axis) vs longitudinal (y-axis).

    Model frame: unicycle x = forward, y = lateral (see unicycle_accel_curvature.action_to_traj).
    """
    lon_h, lat_h = hist_xy[:, 0], hist_xy[:, 1]
    lon_p, lat_p = pred_xy[:, 0], pred_xy[:, 1]

    # Ensure prediction is drawn from the ego position at the origin
    if lon_p.size == 0 or not (abs(lon_p[0]) < 1e-3 and abs(lat_p[0]) < 1e-3):
        lon_p = np.concatenate([[0.0], lon_p])
        lat_p = np.concatenate([[0.0], lat_p])

    w, h = inset_px
    fig, ax = plt.subplots(figsize=(w / 120, h / 120), dpi=120)
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")
    ax.set_xlim(lat_range)
    ax.set_ylim(lon_range)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#4a5568", linestyle="-", linewidth=0.5, alpha=0.6)
    ax.tick_params(colors="#e2e8f0", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#718096")

    ax.set_xlabel("Lateral (m)", color="#e2e8f0", fontsize=8)
    ax.set_ylabel("Longitudinal (m)", color="#e2e8f0", fontsize=8)
    ax.set_title("Trajectory Prediction", color="#ffffff", fontsize=9, fontweight="bold")

    ax.plot(lat_h, lon_h, "o-", color="#9ca3af", markersize=3, linewidth=1.5, label="History")
    ax.plot(lat_p, lon_p, "-", color="#2dd4bf", linewidth=2.2, label="Prediction")
    ego = mpatches.Rectangle(
        (-0.9, -1.0),
        1.8,
        2.0,
        linewidth=1.0,
        edgecolor="#fbbf24",
        facecolor="#fbbf24",
        zorder=5,
    )
    ax.add_patch(ego)

    leg = ax.legend(
        loc="upper right",
        fontsize=7,
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#e2e8f0",
    )
    for text in leg.get_texts():
        text.set_color("#e2e8f0")

    return _fig_to_rgb(fig, dpi=120)


def _try_load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def decode_full_clip_camera_frames(
    avdi: object,
    clip_id: str,
    camera_feature: str,
    *,
    maybe_stream: bool = True,
    batch_size: int = 48,
    max_frames: int | None = None,
):
    """Yield (rgb_hwc_uint8, t_seconds_from_clip_start) for every frame of one camera."""
    reader = avdi.get_clip_feature(clip_id, camera_feature, maybe_stream=maybe_stream)
    try:
        timestamps = reader.timestamps
        n = len(timestamps)
        if max_frames is not None:
            n = min(n, max_frames)
        t0 = int(timestamps[0])
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = np.arange(start, end, dtype=np.int64)
            imgs = reader.decode_images_from_frame_indices(idx)
            for k in range(imgs.shape[0]):
                t_sec = (int(timestamps[start + k]) - t0) * 1e-6
                yield imgs[k], t_sec
    finally:
        reader.close()

def composite_overlay(
    frame_chw: np.ndarray,
    reasoning: str,
    t_seconds: float,
    traj_rgb: np.ndarray,
    margin: int = 12,
) -> np.ndarray:
    """frame_chw: uint8 (3, H, W). Returns uint8 (H, W, 3)."""
    rgb = np.transpose(frame_chw, (1, 2, 0)).copy()
    h, w, _ = rgb.shape
    pil = Image.fromarray(rgb).convert("RGBA")

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_bold = _try_load_font(max(14, h // 64))
    font_body = _try_load_font(max(13, h // 72))
    font_ts = _try_load_font(max(12, h // 80))

    def _text_h(fnt: ImageFont.ImageFont, text: str) -> int:
        b = draw.textbbox((0, 0), text, font=fnt)
        return b[3] - b[1]

    # Timestamp (top-left)
    ts = f"t = {t_seconds:.1f}s"
    tb = draw.textbbox((0, 0), ts, font=font_ts)
    tw_ts, th_ts = tb[2] - tb[0], tb[3] - tb[1]
    pad_ts = 6
    ts_bottom = margin + th_ts + 2 * pad_ts
    draw.rounded_rectangle(
        [margin, margin, margin + tw_ts + 2 * pad_ts, ts_bottom],
        radius=4,
        fill=_COLOR_TS_BG,
    )
    draw.text((margin + pad_ts, margin + pad_ts), ts, fill=_COLOR_TEXT, font=font_ts)

    # Reasoning bar (full width under timestamp)
    bar_x0 = margin
    bar_x1 = w - margin
    bar_y0 = ts_bottom + 8
    header = "Reasoning:"
    max_text_w = bar_x1 - bar_x0 - 24
    chars = max(40, max_text_w // 7)
    wrapped = textwrap.fill(reasoning.strip() or "(no reasoning text)", width=chars)
    line_h = _text_h(font_body, "Ay")
    header_h = _text_h(font_bold, header)
    body_lines = max(1, len(wrapped.split("\n")))
    bar_h = 16 + header_h + 6 + body_lines * (line_h + 4)

    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y0 + 4], fill=_COLOR_BORDER)
    draw.rectangle([bar_x0, bar_y0 + 4, bar_x1, bar_y0 + bar_h], fill=_COLOR_BG)

    cx = bar_x0 + 12
    cy = bar_y0 + 14
    draw.text((cx, cy), header, fill=_COLOR_HEADER, font=font_bold)
    cy += header_h + 8
    for line in wrapped.split("\n"):
        draw.text((cx, cy), line, fill=_COLOR_TEXT, font=font_body)
        cy += line_h + 4

    pil = Image.alpha_composite(pil, overlay)

    # Trajectory inset below reasoning bar (left), so it does not cover the text band
    traj_img = Image.fromarray(traj_rgb).convert("RGBA")
    tw, th = traj_img.size
    target_h = max(120, int(h * 0.30))
    scale = target_h / th
    traj_img = traj_img.resize((int(tw * scale), target_h), Image.Resampling.LANCZOS)
    ty = bar_y0 + bar_h + margin
    pil.paste(traj_img, (margin, ty), traj_img)

    return np.asarray(pil.convert("RGB"))


def run_model_rollout(
    model: AlpamayoR1,
    processor,
    data: dict,
    top_p: float = 0.98,
    temperature: float = 0.6,
    num_traj_samples: int = 1,
    max_generation_length: int = 256,
) -> tuple[str, np.ndarray, np.ndarray]:
    """One VLM + trajectory sample at the clip time implied by ``data`` (t0_us, context frames)."""
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=copy.deepcopy(model_inputs),
            top_p=top_p,
            temperature=temperature,
            num_traj_samples=num_traj_samples,
            max_generation_length=max_generation_length,
            return_extra=True,
        )
    reasoning = str(extra["cot"][0, 0, 0]).strip()
    hist_np = data["ego_history_xyz"].cpu().numpy()[0, 0, :, :2].astype(np.float64)
    pred_np = pred_xyz.cpu().numpy()[0, 0, 0, :, :2].astype(np.float64)
    return reasoning, hist_np, pred_np


def _stack_bev_video(frames: list[np.ndarray]) -> np.ndarray:
    """Resize BEV RGB frames to a common (H, W) so mediapy can stack them."""
    if not frames:
        raise ValueError("no BEV frames to stack")
    h = max(f.shape[0] for f in frames)
    w = max(f.shape[1] for f in frames)
    out: list[np.ndarray] = []
    for f in frames:
        if f.shape[0] == h and f.shape[1] == w:
            out.append(f)
        else:
            out.append(
                np.asarray(
                    Image.fromarray(f).resize((w, h), Image.Resampling.LANCZOS),
                    dtype=np.uint8,
                )
            )
    return np.stack(out, axis=0)

def save_standalone_artifacts(
    artifact_dir: Path,
    bev_frames: list[np.ndarray],
    cot_segments: list[tuple[float, int | None, str]],
    *,
    fps: float,
    clip_id: str,
) -> None:
    """Write ``bev.mp4`` (trajectory panel only) and ``cot.txt`` (timed segments)."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if bev_frames:
        mp.write_video(str(artifact_dir / "bev.mp4"), _stack_bev_video(bev_frames), fps=fps)
    lines = [
        f"# Chain-of-thought log\n# clip_id: {clip_id}\n# output_fps: {fps}\n",
        f"# segments: {len(cot_segments)}\n",
    ]
    lines.append("segment,video_time_s,t0_us,cot\n")
    for i, (t_vid, t0_us, text) in enumerate(cot_segments):
        lines.append((f"{i},{t_vid:.6f},{t0_us},{text}\n").strip())
    (artifact_dir / "cot.txt").write_text("".join(lines), encoding="utf-8")
    print(f"Artifacts: {artifact_dir / 'bev.mp4'} , {artifact_dir / 'cot.txt'}")

def load_clip_ids(avdi=None) -> list[str]:
    """Resolve short IDs via PAI_REASON.txt; optionally keep only clips in the dataset ``clip_index``."""
    all_clip_ids_path = "/home/bcostarendon/data/PAI_REASON.txt"
    all_clip_ids = []
    with open(all_clip_ids_path, "r") as f:
        for line in f:
            all_clip_ids.append(line.strip())
    selected_clips_path = "/home/bcostarendon/data/selected_clip_ids.txt"
    selected_clip_ids: list[str] = []
    with open(selected_clips_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for full_clip_id in all_clip_ids:
                if line in full_clip_id:
                    selected_clip_ids.append(full_clip_id)
                    break
    print(f"Found {len(selected_clip_ids)} selected clip ids (from prefix match)")
    if avdi is not None:
        valid = set(avdi.clip_index.index.astype(str))
        ok = [c for c in selected_clip_ids if c in valid]
        for c in selected_clip_ids:
            if c not in valid:
                print(
                    f"Skipping {c!r}: not in Physical AI AV clip_index for this "
                    "revision (PAI_REASON / selection list can include other corpora or dropped clips)."
                )
        print(f"After clip_index filter: {len(ok)} clips")
        return ok
    return selected_clip_ids

def run_alpamayo_inference(
    clip_id: str,
    model: AlpamayoR1,
    processor: object,
    max_frames: int | None = None,
    fps: float = 10.,
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[float, int | None, str]]]:

    hf_token = os.environ.get("HF_TOKEN", None)
    data = load_physical_aiavdataset(
        clip_id,
        token=hf_token,
        return_avdi=True,
    )
    avdi = data["avdi"]
    cam_feature = data["camera_features_ordered"][1]

    clip_feature = avdi.get_clip_feature(
        clip_id, 
        cam_feature, 
        maybe_stream=True
    )
    try:
        timestamps = clip_feature.timestamps
        total_timestamps = len(timestamps)
        if max_frames is not None:
            total_timestamps = min(total_timestamps, max_frames)
    finally:
        clip_feature.close()

    min_t0 = 5_100_000
    curr_timestamp = 0
    while curr_timestamp < total_timestamps and int(timestamps[curr_timestamp]) < min_t0:
        curr_timestamp += 1
    if curr_timestamp >= total_timestamps:
        raise ValueError(
            f"No frame reaches minimum t0_us={min_t0} µs; clip too short for sliding mode."
        )
    if curr_timestamp > 0:
        print(
            f"Skipping first {curr_timestamp} frames (need timestamp >= {min_t0} µs for history before t0)."
        )

    out_frames: list[np.ndarray] = []
    bev_frames: list[np.ndarray] = []
    cot_segments: list[tuple[float, int | None, str]] = []
    reader = None
    traj_panel = None
    reasoning = ""
    for j in range(curr_timestamp, total_timestamps):
        if (j - curr_timestamp) % fps == 0 or traj_panel is None:
            if reader is not None:
                reader.close()
                reader = None

            data_j = load_physical_aiavdataset(
                clip_id,
                t0_us=int(timestamps[j]),
                avdi=avdi,
                token=hf_token,
                return_avdi=False,
                maybe_stream=True,
            )
            reasoning, hist_np, pred_np = run_model_rollout(model, processor, data_j)
            traj_panel = render_trajectory_panel(hist_np, pred_np)
            reader = avdi.get_clip_feature(clip_id, cam_feature, maybe_stream=True)
            r_snip = reasoning if len(reasoning) <= 140 else reasoning[:140] + "…"
            t_vid = (int(timestamps[j]) - int(timestamps[0])) * 1e-6
            cot_segments.append((t_vid, int(timestamps[j]), reasoning))
            print(f"  t={t_vid:.2f}s  CoT: {r_snip}")
            
        assert reader is not None
        img = reader.decode_images_from_frame_indices(np.array([j], dtype=np.int64))[0]
        t_sec = (int(timestamps[j]) - int(timestamps[0])) * 1e-6
        frame_chw = np.transpose(img, (2, 0, 1))
        out_frames.append(composite_overlay(frame_chw, reasoning, t_sec, traj_panel))
        bev_frames.append(np.copy(traj_panel))

    if reader is not None:
        reader.close()

    print(f"Decoded {len(out_frames)} frames (sliding overlay, fps={fps})")
    return out_frames, bev_frames, cot_segments

def main() -> None:
    parser = argparse.ArgumentParser(description="Alpamayo inference + overlay video")
    parser.add_argument(
        "--output",
        type=str,
        default="alpamayo_output",
        help="Output path for artifacts",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output video FPS (default: native camera FPS for full clip, 10 for --model-frames-only)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Short smoke test",
    )
    args = parser.parse_args()

    fps = 30
    max_frames = None
    if args.debug:
        max_frames = 180
        print(
            f"[debug] max_frames={max_frames}"
        )

    # Load clip ids (filter to HF dataset revision so clip_index lookup cannot KeyError)
    avdi_catalog = physical_ai_av_interface(token=os.environ.get("HF_TOKEN"))
    #clip_ids = load_clip_ids(avdi_catalog)

    clip_ids = []
    with open("500_clip_ids.txt", "r") as f:
        for line in f:
            clip_ids.append(line.strip())

    for i, clip_id in enumerate(clip_ids):
        if os.path.exists(os.path.join(args.output, clip_id)):
            continue

        print(f"Processing clip {clip_id}")
        model = AlpamayoR1.from_pretrained(
            "nvidia/Alpamayo-R1-10B", 
            dtype=torch.bfloat16
        ).to("cuda")
        processor = helper.get_processor(model.tokenizer)
        torch.cuda.manual_seed_all(42)

        try:
            (
                out_frames,
                bev_frames,
                cot_segments,
            ) = run_alpamayo_inference(
                clip_id,
                model,
                processor,
                max_frames,
            )
        except Exception as e:
            print(f"Error processing clip {clip_id}: {e}")
            continue

        os.makedirs(os.path.join(args.output, clip_id), exist_ok=True)
        output_video = os.path.join(args.output, clip_id, f"{clip_id}.mp4")
        mp.write_video(output_video, np.stack(out_frames, axis=0), fps=fps)
        print(f"Wrote {output_video} ({len(out_frames)} frames @ {fps} fps)")
        artifacts_dir = os.path.join(args.output, clip_id, "artifacts")

        if bev_frames:
            art_dir = Path(artifacts_dir)
            save_standalone_artifacts(
                art_dir,
                bev_frames,
                cot_segments,
                fps=fps,
                clip_id=clip_id,
            )


if __name__ == "__main__":
    main()
