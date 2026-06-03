import pandas as pd
import instaloader
import os
import time
import random


post_url = "https://www.instagram.com/p/DXABUxqDEnT/"

shortcode = post_url.split("/p/")[1].split("/")[0]

L = instaloader.Instaloader(
    download_videos=False,
    download_video_thumbnails=False,
    download_comments=False,
    save_metadata=False
)

post = instaloader.Post.from_shortcode(L.context, shortcode)

# ── METADATA ─────────────────────────────

caption = post.caption
likes = post.likes
date = post.date  # datetime completo
date_only = date.date()
time_only = date.time()

# ── IMAGEN PRINCIPAL ─────────────────────

image_url = post.url

print("CAPTION:\n", caption)
print("\nLIKES:", likes)
print("\nDATE:", date_only)
print("TIME:", time_only)
print("\nIMAGE URL:", image_url)