# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kplot.utils import column_width, two_column_width
from kplot.image import show
import matplotlib.pyplot as plt
import numpy as np

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                              Types                              <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                           Definitions                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def ffmpeg(
        file_name: str, 
        source: str = "./frames", destination: str = "./", 
        fps: int = 30
    ) -> None:
    file_path: str = destination + '/' + file_name
    if not file_path.endswith('.mp4'): file_path += ".mp4"
    os.system(f"ffmpeg -loglevel 8 -framerate {fps} -pattern_type glob -i '{source+'/*.png'}' -c:v libx264 -pix_fmt yuv420p -y {file_path}")
def func_video(
        video_name: str, fig: Figure, updater: Callable, N: int, 
        frames: str = "./frames", destination: str = "./", 
        dpi: int = 100, fps: int = 30, verbose=True
    ) -> None:
    if len(glob(f"{frames}/*.png"))>0: os.system(f"rm {frames}/*.png")
    ndigits = len(str(N))
    # if the size isn't divisible by 2 ffmpeg gets mad???
    [wpix, hpix] = (np.array(fig.get_size_inches()) * dpi // 1).astype(int)
    if wpix%2==1: wpix += 1 
    if hpix%2==1: hpix += 1
    fig.set_size_inches(wpix/dpi, hpix/dpi)
    # make the frames and save them to the frames directory
    for i in verbose_bar(range(N), verbose):
        updater(i)
        fig.savefig(f"{frames}/{str(i).zfill(ndigits)}.png", dpi=dpi)
    # make the video
    ffmpeg(video_name, source=frames, destination=destination, fps=fps)
def line_video(
    xs, ys, 
    fname, 
    destination: str = '.',
    ax=None, figsize=(5, 5),
    fps=20, dpi=100,
    **kwargs
):
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    fig = ax.get_figure()
    [line] = ax.plot(xs[0], ys[0], **kwargs)
    def update(f: int):
        line.set_data(xs[f], ys[f])
    func_video(fname, fig, update, len(xs), fps=fps, dpi=dpi, destination=destination)
def lines_video(
        data,
        fname,
        destination: str = '.',
        ax=None, figsize=(5,5),
        fps=20, dpi=100
) -> None:
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    fig = ax.get_figure()
    lines = []
    for (x, y) in data:
        line, = ax.plot(x[0], y[0])
        lines.append(line)
    def update(i: int) -> None:
        for (x, y), line in zip(data, lines):
            line.set_data(x[i], y[i])
    func_video(fname, fig, update, len(data[0][0]), fps=fps, dpi=dpi, destination=destination)
def show_video(
    frames: np.ndarray,
    fname: str,
    ax = None,
    norm = 'linear',
    destination: str = '.',
    fps: int = 30,
    dpi=100,
    **kwargs
):
    # create figure and axis 
    if isinstance(norm, str): norm = auto_norm(norm, frames)
    fig,ax,img = show(frames[0], ax=ax, norm=norm, show=False, **kwargs)
    def update(f: int):
        img.set_array(frames[f])
    func_video(fname, fig, update, len(frames), destination=destination, fps=fps, dpi=dpi)
def particle_video(
    sim,
    particles: list[str] | int,
    fname: str,
    background = None,      
    resume: bool = False,
    res: int = None,
    zfill: int = 10,
    dpi: int = 250,
    # paticle plotting keywords
    color = 'red',
    marker='.',
    ms=10, 
    # assume the rest are keywords for the show function
    **kwds
):
    # check species
    assert sim.input.num_species == 1, "only one species is implemented rn :-("
    # check the time resolution is valid
    if not res: res = sim.input.sp01.track_nstore
    dnp = sim.input.sp01.track_nstore
    dnb = sim.input.ndump
    assert (dnp % res == 0) & (res % dnb == 0), "your resolution must be an integer multiple of track_nstore and ndump must be an integer multiple of res."
    # handle the resuming
    if not resume: 
        os.system(f'rm {frameDir.path}/*')
        i_start = 0
    else:
        last_file = sorted(frameDir.children)[-1]
        last_iter = int(last_file[:-4])
        i_start = last_iter + res
    # find the file
    fn = sim.path+"/Output/Tracks/Sp01/track_Sp01.h5"
    with h5File(fn) as file:
        # extract particle tracks based on the particles argument
        tags = np.array(list(file.keys()))
        match particles:
            case [str(x), ]: selected = particles 
            case int(x): selected = np.random.choice(tags, size=x, replace=False)
        sx, sy = np.array([file[t]['x1'] for t in selected]).T, np.array([file[t]['x2'] for t in selected]).T 
        # setup background
        if not background: background = sim.density
        # initialize plots
        fig,ax,img = show(
            background[0], 
            x=range(0, sim.input.boxsize[0], sim.dx), y=range(0, sim.input.boxsize[1], sim.dy), 
            zorder=1,
            **kwds
        )
        line, = ax.plot(
            sx[0], sy[0], 
            ls='None', color=color, marker=marker, ms=ms,
            zorder = 2
        )
        # execute the loop
        def update(i: int, dnp=dnp, dnb=dnb):
            pind = i * dnp
            line.set_data(sx[pind], sy[pind])
            if pind % dnb == 0:
                bind = pind // dnb 
                img.set_array(background[bind])
        func_video(fname, fig, update, )
    
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Decorators                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def line_video_function(func):
    @wraps(func)
    def line_video_wrapper(*args, ax=None, save="default.gif", **kwargs):
        # Calculate data via func
        xs, ys = func(*args, **kwargs)
        line_video(xs, ys, save, ax=ax)
    return line_video_wrapper
def show_video_function(func):
    @wraps(func)
    def simple_video_wrapper(*args, save="default.gif", norm='linear', **kwargs):
        frames = func(*args, **kwargs)
        show_video(frames, save, norm=norm)
    return simple_video_wrapper

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Classes                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==