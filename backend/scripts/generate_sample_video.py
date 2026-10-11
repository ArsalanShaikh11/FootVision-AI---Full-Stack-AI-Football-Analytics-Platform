"""
Generates a realistic synthetic football clip (10 seconds, 25 FPS)
with a green pitch, field markings, and animated player sprites
for testing FootVision AI without needing external downloads.
"""

from pathlib import Path
import cv2
import numpy as np

def create_sample_football_video(output_path: Path, duration_seconds: int = 10, fps: int = 25):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 360
    total_frames = duration_seconds * fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))

    # Define 6 simulated players with trajectory vectors
    players = [
        {"x": 120, "y": 140, "vx": 1.2, "vy": 0.3, "color": (40, 40, 220)},    # Red team
        {"x": 200, "y": 220, "vx": 0.8, "vy": -0.4, "color": (40, 40, 220)},   # Red team
        {"x": 310, "y": 170, "vx": -0.5, "vy": 0.6, "color": (40, 40, 220)},   # Red team
        {"x": 280, "y": 110, "vx": -1.1, "vy": 0.2, "color": (230, 220, 40)},  # Yellow team
        {"x": 390, "y": 200, "vx": -0.7, "vy": -0.3, "color": (230, 220, 40)}, # Yellow team
        {"x": 480, "y": 150, "vx": -1.4, "vy": 0.5, "color": (230, 220, 40)},  # Yellow team
    ]

    ball = {"x": 250.0, "y": 160.0, "vx": 1.5, "vy": 0.4}

    for f in range(total_frames):
        # 1. Pitch Grass Field
        frame = np.full((height, width, 3), (30, 110, 35), dtype=np.uint8)

        # Subtle pitch stripes
        stripe_w = width // 10
        for s in range(10):
            if s % 2 == 0:
                cv2.rectangle(frame, (s * stripe_w, 0), ((s + 1) * stripe_w, height), (35, 125, 40), -1)

        # 2. Football pitch white markings
        # Boundary touchlines
        cv2.rectangle(frame, (20, 20), (width - 20, height - 20), (240, 240, 240), 2)
        # Halfway line
        cv2.line(frame, (width // 2, 20), (width // 2, height - 20), (240, 240, 240), 2)
        # Center circle
        cv2.circle(frame, (width // 2, height // 2), 45, (240, 240, 240), 2)
        cv2.circle(frame, (width // 2, height // 2), 3, (240, 240, 240), -1)
        # Left penalty box
        cv2.rectangle(frame, (20, height // 2 - 60), (120, height // 2 + 60), (240, 240, 240), 2)
        # Right penalty box
        cv2.rectangle(frame, (width - 120, height // 2 - 60), (width - 20, height // 2 + 60), (240, 240, 240), 2)

        # 3. Animate Players
        for p in players:
            p["x"] += p["vx"]
            p["y"] += p["vy"]

            # Boundary bounce
            if p["x"] < 40 or p["x"] > width - 40:
                p["vx"] *= -1
            if p["y"] < 40 or p["y"] > height - 60:
                p["vy"] *= -1

            px, py = int(p["x"]), int(p["y"])

            # Render humanoid figure: Head + Jersey Torso + Shorts + Legs
            # Head (skin tone)
            cv2.circle(frame, (px, py - 20), 6, (170, 195, 235), -1)
            # Torso / Jersey
            cv2.rectangle(frame, (px - 8, py - 14), (px + 8, py + 4), p["color"], -1)
            # Shorts (dark)
            cv2.rectangle(frame, (px - 7, py + 4), (px + 7, py + 12), (30, 30, 30), -1)
            # Legs
            cv2.line(frame, (px - 4, py + 12), (px - 4, py + 22), (170, 195, 235), 2)
            cv2.line(frame, (px + 4, py + 12), (px + 4, py + 22), (170, 195, 235), 2)

        # 4. Animate Football
        ball["x"] += ball["vx"]
        ball["y"] += ball["vy"]
        if ball["x"] < 30 or ball["x"] > width - 30:
            ball["vx"] *= -1
        if ball["y"] < 30 or ball["y"] > height - 30:
            ball["vy"] *= -1

        cv2.circle(frame, (int(ball["x"]), int(ball["y"])), 4, (255, 255, 255), -1)
        cv2.circle(frame, (int(ball["x"]), int(ball["y"])), 4, (0, 0, 0), 1)

        out.write(frame)

    out.release()
    print(f"Sample football match video created at: {output_path} ({total_frames} frames, {width}x{height})")

if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent.parent / "sample_match.mp4"
    create_sample_football_video(target)
