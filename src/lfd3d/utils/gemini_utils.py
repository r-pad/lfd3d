import imageio


def upload_video(client, video_file_name):
    video_file = client.files.upload(file=video_file_name)

    while video_file.state == "PROCESSING":
        time.sleep(10)
        video_file = client.files.get(name=video_file.name)

    if video_file.state == "FAILED":
        raise ValueError(video_file.state)
    return video_file


def save_video_for_gemini(frames, output_path):
    """
    Save a sequence of frames as a video file

    Args:
        frames: NumPy array of shape (num_frames, height, width, channels)
        output_path: Path to save the video file
        fps: Frames per second for the video
    """
    # Gemini seems to struggle with longer videos, so adaptively set fps so that its always 20 sec video
    # We'll lose temporal resolution, but atleast our goals should be accurate
    fps = frames.shape[0] // 20
    height, width = frames[0].shape[:2]
    writer = imageio.get_writer(
        output_path,
        format="mp4",
        fps=fps,
        codec="h264",
        quality=7,
        pixelformat="yuv420p",
        macro_block_size=1,
        output_params=["-vf", f"scale={width}:{height}"],
    )

    for frame in frames:
        writer.append_data(frame)
    writer.close()
