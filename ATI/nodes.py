import json

from .motion import process_tracks
import numpy as np
FIXED_LENGTH = 121
def pad_pts(tr):
    """Convert list of {x,y} to (FIXED_LENGTH,1,3) array, padding/truncating."""
    pts = np.array([[p['x'], p['y'], 1] for p in tr], dtype=np.float32)
    n = pts.shape[0]
    if n < FIXED_LENGTH:
        pad = np.zeros((FIXED_LENGTH - n, 3), dtype=np.float32)
        pts = np.vstack((pts, pad))
    else:
        pts = pts[:FIXED_LENGTH]
    return pts.reshape(FIXED_LENGTH, 1, 3)

class WanVideoATITracks:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("WANVIDEOMODEL", ),
            "tracks": ("STRING",),
            "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 29048, "step": 8, "tooltip": "Height of the image to encode"}),
            "temperature": ("FLOAT", {"default": 220.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
            "topk": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the steps to apply ATI"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the steps to apply ATI"}),
        },
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patchmodel"
    CATEGORY = "WanVideoWrapper"

    def patchmodel(self, model, tracks, width, height, temperature, topk, start_percent, end_percent):
    
        if len(tracks) < 10:
            tracks_data = []
            for coords in tracks:
                coords = json.loads(coords.replace("'", '"'))
                tracks_data.append(coords)
        else:
            coords = json.loads(tracks.replace("'", '"'))

            if tracks_data and isinstance(tracks_data[0], dict) and 'x' in tracks_data[0]:
                # It's a single track, wrap it in a list to make it a list of tracks
                tracks_data = [tracks_data]

        arrs = []
        for track in tracks_data:
            pts = pad_pts(track)
            arrs.append(pts)

        tracks_np = np.stack(arrs, axis=0)

        processed_tracks = process_tracks(tracks_np, (width, height))

        patcher = model.clone()
        patcher.model_options["transformer_options"]["ati_tracks"] = processed_tracks.unsqueeze(0)
        patcher.model_options["transformer_options"]["ati_temperature"] = temperature
        patcher.model_options["transformer_options"]["ati_topk"] = topk
        patcher.model_options["transformer_options"]["ati_start_percent"] = start_percent
        patcher.model_options["transformer_options"]["ati_end_percent"] = end_percent
        
        return (patcher,)
        
NODE_CLASS_MAPPINGS = {
    "WanVideoATITracks": WanVideoATITracks,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoATITracks": "WanVideo ATI Tracks",
    }
