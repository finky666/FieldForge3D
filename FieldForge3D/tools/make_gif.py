from pathlib import Path
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# ===== NASTAVENIA =====
duration_ms = 1500   # čas jedného obrázka v milisekundách
scale = 1.0         # 1.0 = pôvodná veľkosť, 0.75 = menší GIF

# ===== VÝBER PRIEČINKA =====
root = tk.Tk()
root.withdraw()

folder = filedialog.askdirectory(title="Vyber priečinok s PNG obrázkami")
if not folder:
    raise SystemExit("Nebolo vybraté nič.")

images_dir = Path(folder)
png_files = sorted(images_dir.glob("*.png"))

if not png_files:
    raise SystemExit(f"V priečinku nie sú žiadne PNG: {images_dir}")

output_gif = images_dir.parent / "fieldforge_plugins.gif"

# ===== SPRACOVANIE =====
frames = []
base_size = None

for file in png_files:
    img = Image.open(file).convert("RGBA")

    if scale != 1.0:
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    if base_size is None:
        base_size = img.size
    elif img.size != base_size:
        img = img.resize(base_size, Image.LANCZOS)

    # GIF lepšie znáša P mód až na konci
    frames.append(img)

# konverzia do GIF-compatible palety
gif_frames = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]

gif_frames[0].save(
    output_gif,
    save_all=True,
    append_images=gif_frames[1:],
    duration=duration_ms,
    loop=0,
    optimize=True,
)

print(f"Hotovo: {output_gif}")
print(f"Počet obrázkov: {len(gif_frames)}")