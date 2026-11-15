"""HydroTwin — improved interactive preview of the conceptual G1 CAD assembly.

The STEP/GLB model is an illustrative engineering model, not a verified
as-built representation of the Neelum-Jhelum hydropower generating unit.

Requires: numpy, plotly, trimesh
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import trimesh


# Exterior wall elements only, so fading the housing does not also fade
# the shaft, rotor, runner blades, and other internal machinery.
SHELL_PARTS = {
    "Generator__main_housing_cutaway",
    "Generator__stator_backiron_cutaway",
    "Turbine__cutaway_casing",
}


def _hex_color(mesh):
    material = getattr(mesh.visual, "material", None)
    rgba = getattr(material, "baseColorFactor", None)
    if rgba is None:
        return "#8497A8"
    rgb = np.asarray(rgba, dtype=float)[:3]
    if rgb.max() > 1:
        rgb = rgb / 255.0
    return "#" + "".join(f"{int(round(ch * 255)):02X}" for ch in np.clip(rgb, 0, 1))


def cad_preview(glb_path, selected_component="TGB", housing_opacity=0.24):
    """Render one rotatable CAD-style hydropower generating unit.

    Args:
        glb_path: Path to HydroTwin_G1_Interactive.glb.
        selected_component: TGB or UGB; this bearing is highlighted.
        housing_opacity: 0..1 transparency of main housing/casing ONLY.
    """
    selected_component = selected_component.upper()
    if selected_component not in {"TGB", "UGB"}:
        raise ValueError("selected_component must be 'TGB' or 'UGB'")
    if not 0 <= housing_opacity <= 1:
        raise ValueError("housing_opacity must be between 0 and 1")
    glb_path = Path(glb_path)
    if not glb_path.exists():
        raise FileNotFoundError(glb_path)

    scene = trimesh.load(str(glb_path), force="scene")
    grouped = defaultdict(list)
    for part_name, mesh in scene.geometry.items():
        if part_name.startswith("TGB__"):
            role = "TGB"
        elif part_name.startswith("UGB__"):
            role = "UGB"
        elif part_name in SHELL_PARTS:
            role = "HOUSING"
        else:
            role = "MACHINERY"
        grouped[(role, _hex_color(mesh))].append(mesh)

    fig = go.Figure()
    for (role, color), meshes in grouped.items():
        merged = trimesh.util.concatenate(meshes)
        if role == "HOUSING":
            opacity = housing_opacity
            label = "Cutaway Housing"
        elif role in ("TGB", "UGB"):
            opacity = 1.0 if selected_component == role else 0.52
            label = f"{role} Bearing"
        else:
            opacity = 1.0
            label = "Generating Unit Machinery"

        fig.add_trace(go.Mesh3d(
            x=merged.vertices[:, 0],
            y=merged.vertices[:, 1],
            z=merged.vertices[:, 2],
            i=merged.faces[:, 0],
            j=merged.faces[:, 1],
            k=merged.faces[:, 2],
            name=label,
            color=color,
            opacity=opacity,
            flatshading=False,
            lighting=dict(ambient=0.50, diffuse=0.72, specular=0.53,
                          roughness=0.38, fresnel=0.2),
            lightposition=dict(x=10, y=-7, z=17),
            showlegend=False,
            hovertemplate=label + "<extra></extra>",
        ))

    # Position callouts beside the bearing locations; z coordinates are
    # conceptual-model geometry, not real plant dimensions.
    for name, z, color in (
        ("UGB", 9.65, "#FFB454"),
        ("TGB", 3.24, "#58ACFF"),
    ):
        fig.add_trace(go.Scatter3d(
            x=[1.55], y=[-1.40], z=[z],
            mode="markers+text",
            text=[name],
            textposition="middle right",
            textfont=dict(color=color, size=17),
            marker=dict(color=color, size=5),
            showlegend=False,
            hovertemplate=name + "<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="HydroTwin | G1 Vertical Hydropower Assembly (Conceptual CAD)",
            font=dict(size=20),
        ),
        paper_bgcolor="#101826",
        font=dict(color="#E6EDF6"),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#101826",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.55, y=-1.85, z=0.9),
                up=dict(x=0, y=0, z=1),
                projection=dict(type="perspective"),
            ),
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, t=65, b=0),
        height=760,
        showlegend=False,
        annotations=[dict(
            text="Illustrative engineering assembly · not as-built plant geometry",
            x=0.5, y=0.01, xref="paper", yref="paper",
            showarrow=False, font=dict(size=12, color="#90A0B6"),
        )],
    )
    return fig
