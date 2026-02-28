from moviepy import VideoFileClip, AudioFileClip

# Load your video and audio
video = VideoFileClip("output_fast.mp4")
audio = AudioFileClip("Psychoziz & Infinite - More & More.mp3")
if audio.duration < video.duration:
    # In v2.0, looping is often done via the .with_audio_loop() method
    # or by repeating the clip manually
    final_audio = audio.with_effects([lambda clip: clip.loop(duration=video.duration)])
else:
    # Cutting still works the same way
    final_audio = audio.subclipped(0, video.duration) 

# Note: .set_audio() changed to .with_audio() in v2.0
final_video = video.with_audio(final_audio)

# Export
final_video.write_videofile("warped_output.mp4", codec="libx264")