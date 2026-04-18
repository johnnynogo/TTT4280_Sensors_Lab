import subprocess


files = ["Lab\ 3/transmittans/iphone/trans_iphone_movie1.mp4"]

for filename in files:

    output_name = filename.replace(".mp4", ".txt")

    print(f"\nProcessing {filename}")
    print("Select ROI and press ENTER when done...\n")

    subprocess.run(["python", "read_video_from_roi.py",
                    filename, output_name])

    print(f"Finished {filename}")
