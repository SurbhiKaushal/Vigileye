#!/usr/bin/env python3
"""VigilEye live runner.

    python run.py                          # webcam, default config
    python run.py --source data/test.mp4   # run against a video file
    python run.py --mode hybrid            # rules + trained LSTM
    python run.py --no-objects             # skip YOLO/hands (low-power mode)
    python run.py --recalibrate            # force a fresh baseline
    python run.py --headless               # no window (embedded deployment)

Keys: q quit | c recalibrate | m mute | h hide HUD | s save frame
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2

from vigileye.alerts import AlertManager
from vigileye.config import load_config
from vigileye.fusion import DriverState
from vigileye.logger import SessionLogger
from vigileye.pipeline import VigilEyePipeline
from vigileye.video import VideoStream
from vigileye.pico_alert import send_to_pico
from vigileye.visualize import (
    draw_detections,
    draw_hud,
    draw_landmarks,
    draw_pose_axis,
    draw_wheel_roi,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VigilEye driver monitoring")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default=None, help="camera index or video path")
    p.add_argument("--mode", choices=["rule", "temporal", "hybrid"], default=None)
    p.add_argument("--driver-id", default="driver_01")
    p.add_argument("--vehicle-id", default="vehicle_01")
    p.add_argument("--no-objects", action="store_true", help="disable YOLO + hands")
    p.add_argument("--no-alerts", action="store_true")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--recalibrate", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--record", default=None, help="write annotated video to this path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.source is not None:
        cfg.camera.source = args.source
    if args.mode is not None:
        cfg.fusion.mode = args.mode
    if args.no_alerts:
        cfg.alerts.enabled = False

    if args.recalibrate:
        Path(cfg.calibration.file).unlink(missing_ok=True)

    stream = VideoStream(
        cfg.camera.source, cfg.camera.width, cfg.camera.height,
        cfg.camera.fps, cfg.camera.flip,
    )
    pipeline = VigilEyePipeline(cfg, use_objects=not args.no_objects)
    alerts = AlertManager(cfg.alerts)
    logger = None if args.no_log else SessionLogger(cfg.logging, args.driver_id, args.vehicle_id)

    writer = None
    fps_hist = deque(maxlen=30)
    show_hud = True
    muted = False
    frames = 0
    t_start = time.time()
    previous_state = None

    if logger:
        print(f"[VigilEye] session {logger.session_id} started")
    print("[VigilEye] running - press 'q' to quit")

    try:
        while True:
            ok, frame, ts = stream.read()
            if not ok:
                break

            t0 = time.perf_counter()
            signals, result = pipeline.process(frame, ts)
            if not pipeline.calibrating:
                current_state = (
                    result.state.value
                    if hasattr(result.state, "value")
                    else str(result.state)
                )

                if current_state != previous_state:
                    send_to_pico(current_state)
                    previous_state = current_state
            fps_hist.append(1.0 / max(time.perf_counter() - t0, 1e-6))
            fps = sum(fps_hist) / len(fps_hist)
            frames += 1

            event = None
            if not muted and not pipeline.calibrating:
                event = alerts.update(ts, result.state, pipeline.time_in_state(ts))

            if logger:
                logger.log_frame(signals, result)
                if event:
                    logger.log_alert(event, result, signals.perclos)
            if event:
                print(f"[ALERT L{event.level}] {event.message}")

            if not args.headless or args.record:
                canvas = frame.copy()
                if show_hud:
                    if pipeline.last_points is not None:
                        draw_landmarks(canvas, pipeline.last_points)
                        if pipeline.last_pose is not None:
                            draw_pose_axis(canvas, pipeline.last_points, pipeline.last_pose)
                    draw_wheel_roi(canvas, pipeline.hands.roi)
                    draw_detections(canvas, pipeline.last_detections)
                    canvas = draw_hud(
                        canvas, signals, result, fps,
                        flashing=alerts.is_flashing(ts),
                        calibrating=pipeline.calibrator.progress(ts) if pipeline.calibrating else -1.0,
                    )

                if args.record:
                    if writer is None:
                        h, w = canvas.shape[:2]
                        writer = cv2.VideoWriter(
                            args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h)
                        )
                    writer.write(canvas)

                if not args.headless:
                    cv2.imshow("VigilEye", canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("c"):
                        pipeline.recalibrate()
                        print("[VigilEye] recalibrating...")
                    if key == ord("m"):
                        muted = not muted
                        print(f"[VigilEye] alerts {'muted' if muted else 'unmuted'}")
                    if key == ord("h"):
                        show_hud = not show_hud
                    if key == ord("s"):
                        name = f"snapshot_{int(time.time())}.png"
                        cv2.imwrite(name, canvas)
                        print(f"[VigilEye] saved {name}")

    except KeyboardInterrupt:
        print("\n[VigilEye] interrupted")
    finally:
        elapsed = max(time.time() - t_start, 1e-6)
        mean_fps = frames / elapsed
        stream.release()
        pipeline.close()
        if writer is not None:
            writer.release()
        if logger:
            logger.close(mean_fps)
        cv2.destroyAllWindows()
        print(f"[VigilEye] {frames} frames in {elapsed:.1f}s ({mean_fps:.1f} FPS)")


if __name__ == "__main__":
    main()
