import os
import base64
import openai
import json
import csv
import cv2
from collections import defaultdict
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('basketball_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Initialize OpenAI client ---
logger.debug("Initializing OpenAI client...")
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
if not os.environ.get("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable is not set!")

def extract_frames(video_path, output_dir="frames", step=30):
    """
    Extracts frames from a video at a regular interval and saves them to output_dir.
    """
    logger.debug(f"Starting frame extraction from {video_path}")
    logger.debug(f"Output directory: {output_dir}, Frame step: {step}")
    
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return

    logger.debug("Video opened successfully, starting frame extraction...")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.debug("End of video reached")
            break

        if frame_count % step == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            success = cv2.imwrite(frame_filename, frame)
            if success:
                logger.debug(f"Saved frame {saved_count} to {frame_filename}")
            else:
                logger.error(f"Failed to save frame {saved_count}")
            saved_count += 1

        frame_count += 1

    cap.release()
    logger.info(f"Frame extraction complete. Extracted {saved_count} frames to {output_dir}")
    return output_dir

def encode_image(image_path):
    logger.debug(f"Encoding image: {image_path}")
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            logger.debug("Image encoded successfully")
            return encoded
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {str(e)}")
        raise

def analyze_frame(image_path):
    logger.debug(f"Starting frame analysis for: {image_path}")
    base64_image = encode_image(image_path)

    prompt = """
You are analyzing a youth basketball game frame. Identify:
1. Points scored in this frame, and jersey number of the player (if visible).
2. Passes that are clearly happening (ball in motion between teammates).
3. Rebound attempts or successful rebounds, and jersey number if visible.

Output JSON like:
{
  "points": { "23": 2 },
  "passes": 1,
  "rebounds": { "11": 1 }
}
"""

    try:
        logger.debug("Sending request to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You analyze basketball frames for player stats."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        logger.debug("Received response from OpenAI API")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error during OpenAI API call: {str(e)}")
        raise

def annotate_frame(image_path, annotations, output_path):
    logger.debug(f"Annotating frame: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 255, 0)
    thickness = 2
    y_offset = 30

    for key, val in annotations.items():
        label = f"{key}: {val}"
        cv2.putText(image, label, (10, y_offset), font, font_scale, color, thickness)
        y_offset += 25

    success = cv2.imwrite(output_path, image)
    if success:
        logger.debug(f"Annotated frame saved to: {output_path}")
    else:
        logger.error(f"Failed to save annotated frame to: {output_path}")

def analyze_frames(folder_path, output_dir="output"):
    logger.info(f"Starting frame analysis for folder: {folder_path}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotated_frames"), exist_ok=True)

    points = defaultdict(int)
    passes = 0
    rebounds = defaultdict(int)
    frame_data_list = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            logger.info(f"Analyzing {filename}...")

            try:
                result = analyze_frame(image_path)
                logger.debug(f"Raw result for {filename}: {result}")

                cleaned = result.strip().strip("```").replace("json", "").strip()
                frame_data = json.loads(cleaned)
                frame_data["frame"] = filename
                frame_data_list.append(frame_data)

                for jersey, score in frame_data.get("points", {}).items():
                    points[jersey] += score
                passes += frame_data.get("passes", 0)
                for jersey, count in frame_data.get("rebounds", {}).items():
                    rebounds[jersey] += count

                # Annotate and save
                out_img_path = os.path.join(output_dir, "annotated_frames", f"annotated_{filename}")
                annotate_frame(image_path, {
                    "Points": frame_data.get("points", {}),
                    "Passes": frame_data.get("passes", 0),
                    "Rebounds": frame_data.get("rebounds", {})
                }, out_img_path)

            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)

    # Save .json
    try:
        with open(os.path.join(output_dir, "summary.json"), "w") as jf:
            json.dump(frame_data_list, jf, indent=2)
        logger.info("Summary JSON file saved successfully")
    except Exception as e:
        logger.error(f"Error saving summary.json: {str(e)}")

    # Save .csv
    try:
        with open(os.path.join(output_dir, "summary.csv"), "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=["frame", "points", "passes", "rebounds"])
            writer.writeheader()
            for row in frame_data_list:
                writer.writerow({
                    "frame": row["frame"],
                    "points": json.dumps(row.get("points", {})),
                    "passes": row.get("passes", 0),
                    "rebounds": json.dumps(row.get("rebounds", {})),
                })
        logger.info("Summary CSV file saved successfully")
    except Exception as e:
        logger.error(f"Error saving summary.csv: {str(e)}")

    return points, passes, rebounds

def print_final_stats(csv_path):
    logger.info(f"Reading final stats from: {csv_path}")
    points = defaultdict(int)
    rebounds = defaultdict(int)
    total_passes = 0

    try:
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    # Parse points
                    frame_points = json.loads(row["points"] or "{}")
                    if isinstance(frame_points, dict):
                        for jersey, score in frame_points.items():
                            points[jersey] += score

                    # Parse rebounds
                    frame_rebounds = json.loads(row["rebounds"] or "{}")
                    if isinstance(frame_rebounds, dict):
                        for jersey, count in frame_rebounds.items():
                            rebounds[jersey] += count

                    # Parse passes
                    passes_val = row["passes"].strip()
                    if passes_val.isdigit():
                        total_passes += int(passes_val)

                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error parsing row {row.get('frame', '?')}: {str(e)}")

        # --- Final Output ---
        print("\n🏀 Final Game Stats from Video Analysis")
        
        print("\n1️⃣ Total Points Scored by Jersey:")
        if points:
            for jersey, score in points.items():
                print(f"   - Jersey #{jersey}: {score} point(s)")
        else:
            print("   - No points detected.")

        print("\n2️⃣ Total Passes:")
        print(f"   - {total_passes} pass(es) detected.")

        print("\n3️⃣ Total Rebounds by Jersey:")
        if rebounds:
            for jersey, count in rebounds.items():
                print(f"   - Jersey #{jersey}: {count} rebound(s)")
        else:
            print("   - No rebounds detected.")

    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")

def main():
    logger.info("Starting basketball analysis...")
    
    # Step 1: Extract frames from video
    video_path = "GirlsNav.mp4"
    logger.info(f"Extracting frames from video: {video_path}")
    frames_dir = extract_frames(video_path, output_dir="GirlsNav_frames", step=30)

    # Step 2: Analyze frames
    logger.info("Starting frame analysis...")
    points, total_passes, rebounds = analyze_frames(frames_dir, output_dir="output")

    # Step 3: Print final stats
    logger.info("Generating final statistics...")
    print_final_stats(os.path.join("output", "summary.csv"))
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main() 