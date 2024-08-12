import os

def latest_ckpt(ckpt_dir):
    ckpts = os.listdir(ckpt_dir)
    ckpts = [(int(f.split('_')[1].split('.')[0]), f) for f in ckpts]
    if not ckpts:
        return None
    _, latest_file = max(ckpts, key=lambda x: x[0])
    return os.path.join(ckpt_dir, latest_file)

