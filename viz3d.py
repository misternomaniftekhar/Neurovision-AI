import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

SURFACES = {  # name -> (mask function, colour, opacity)
    "Whole tumor": (lambda m: m > 0, "#FFD400", 0.18),
    "Tumor core": (lambda m: (m == 1) | (m == 3), "#FF3B30", 0.45),
    "Enhancing": (lambda m: m == 3, "#0A84FF", 0.95),
}


def mesh_figure(mask, zooms, which, stride=2):
    sp = np.array(zooms) * stride
    fig = go.Figure()
    for name in which:
        fn, color, opacity = SURFACES[name]
        m = fn(mask[::stride, ::stride, ::stride]).astype(np.float32)
        if m.sum() < 8:
            continue
        m = gaussian_filter(np.pad(m, 1), 1.0)
        if m.max() <= 0.5:
            continue
        verts, faces, _, _ = marching_cubes(m, level=0.5, spacing=tuple(sp))
        verts -= sp  # undo the 1-voxel padding
        fig.add_trace(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                color=color, opacity=opacity, name=name,
                                showlegend=True, flatshading=True))
    fig.update_layout(scene=dict(aspectmode="data", xaxis=dict(visible=False),
                                 yaxis=dict(visible=False), zaxis=dict(visible=False)),
                      margin=dict(l=0, r=0, t=0, b=0), height=560,
                      legend=dict(orientation="h", y=1.02))
    return fig
