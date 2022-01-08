# Rigid body physics simulator

Requires a Python installation under an OS supporting [Taichi](https://docs.taichi.graphics/).

Running simulator-visualizer
```sh
python3 -m venv ti
source ti/bin/activate
pip3 install -r requirements.txt

python3 sim1-spheres.py     # Spheres
python3 sim2-molecules.py   # Molecules

# In case of failure: try the following environment variable settings
export TI_BACKEND=gpu
export TI_BACKEND=opengl
export TI_BACKEND=cpu       # No GPU acceleration
```

Recording
```sh
export REC=<n_steps>,<file>
export REC=<n_steps>    # <file> defaults to "record.bin"
```

Compiling playback
```sh
gcc playback.c -o playback \
  -O2 -Iraylib/src raylib/build/raylib/libraylib.a \
  -framework OpenGL -framework Cocoa -framework IOKit # macOS
```

Running playback
```sh
./playback [<file>]     # <file> defaults to "record.bin"
```

Recording demo: name the record binary as **sim1/sim1.bin** and
**sim2/sim2.bin** for the two simulations. Also, put the Taichi
screencast images under these two folders respectively.
```sh
./playback sim1/sim1.bin 1
./playback sim2/sim2.bin 2
```

Appendix: converting image sequence to video with FFmpeg
```sh
ffmpeg -r 60 -i out%04d.png -c:v libx264 -vf fps=60 -pix_fmt yuv420p out.mp4
```
