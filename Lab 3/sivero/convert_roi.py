import subprocess


files = ["hoy_puls.mp4"]

for filename in files:

    output_name = filename.replace(".mp4", ".txt")

    print(f"\nProcessing {filename}")
    print("Select ROI and press ENTER when done...\n")

    subprocess.run(["python", "read_video_from_roi.py",
                    filename, output_name])

    print(f"Finished {filename}")
