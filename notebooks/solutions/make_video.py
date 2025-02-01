import gymnasium as gym
from gymnasium.utils.save_video import save_video

def make_video(env, policy, max_length, video_folder="videos", name_prefix="rl_policy"):
    s,_ = env.reset()
    for t in range(max_length):
        a = policy(s)
        s2,_,d,_,_ = env.step(a)
        s=s2
        if d:
            break
    frames = env.render()
    nb_frames = len(frames)
    print("Nb frames in video", nb_frames)
    save_video(frames=frames, 
               video_folder=video_folder, 
               fps=env.metadata["render_fps"], 
               name_prefix=name_prefix)
    return frames
