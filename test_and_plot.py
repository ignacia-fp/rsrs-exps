from rsrs_config import RSRSBenchmarkConfig
import numpy as np
import subprocess

'''config = RSRSBenchmarkConfig(operator_type=19, 
        dim_arg_type=0,  
        precision = 0, 
        initial_num_samples = 1000, 
        id_tols = [20], 
        geometry=6, 
        min_level=1, 
        h=0.055, 
        depth = 2, 
        kappa = 3.0, 
        factors_cn = False, 
        dense_errors =False, 
        max_tree_depth=16, 
        rrqr=0, 
        f=1.0, 
        n_sources=1, 
        solve = True, 
        symmetric=False, 
        save_samples=False)'''

config = RSRSBenchmarkConfig(operator_type=19,
        dim_arg_type=1,
        precision = 0,
        initial_num_samples = 7000,
        id_tols = [10],
        geometry=6,
        min_level=1,
        h=0.046,
        depth = 2,
        kappa = 10.0,
        factors_cn = False,
        dense_errors =False,
        max_tree_depth=16,
        rrqr=0,
        f=1.0,
        n_sources=5,
        solve = True,
        symmetric=False,
        save_samples=False)

#config = RSRSBenchmarkConfig(operator_type=16, dim_arg_type=1, id_tols = [100, 120], geometry=0, min_level=1, h=0.02, depth = 2, kappa = 0.1, factors_cn = False, dense_errors =False, max_tree_depth=4, rrqr=0, f=1.0, n_sources=1, solve = True, symmetric=False, save_samples=False)
#config.generate_bash_script("run_test_for_plot.sh")
#subprocess.run(["./run_test_for_plot.sh"], check=True)

npoints = 400
'''
pieces, box_limits =  config.save_clipped_mesh_piece_renders(
    plane_z=-10,
    plane_y=0.1,
    elev=25,
    azim=-60,
    transparent=False,
    min_faces=80,   # adjust to drop tiny scraps
    padd = 5.0,
    dpi=600,
    out_dir_name="clipped_mesh_z"
)


slice_paths = config.plot_field_slices_3d(
    tol=20,
    plane="xz", 
    out_name_prefix="field_only_z",
    plane_alpha=0.7,
    plane_points=npoints,
    hide_axes=True,
    box_limits=box_limits,
    dpi=600,
    show_colorbar=True,
    shared_color_scale=False,
    clip_percentiles=(35.0, 99.5)
)'''

slice_paths = config.get_existing_slice_paths("field_only_z", "xz")

bg = "clipped_mesh_z/piece_00.png"

_ = [
    config.composite_images(
        background_relpath=p.name,
        overlay_relpath=bg,
        out_relpath=f"composited/{p.stem}.png"
    )
    for p in slice_paths
]

#-----------------------
'''
pieces, box_limits =  config.save_clipped_mesh_piece_renders(
    plane_y=-10,
    plane_z =-1.2,
    elev=25,
    azim=-60,
    transparent=False,
    min_faces=80,   # adjust to drop tiny scraps
    padd = 5.0,
    out_dir_name="clipped_mesh_y"
)

slice_paths = config.plot_field_slices_3d(
    tol=20,
    plane="xy", 
    out_name_prefix="field_only_y",
    plane_alpha=0.7,
    plane_points=npoints,
    hide_axes=True,
    box_limits=box_limits,
    show_colorbar=False,
    shared_color_scale=False,
    clip_percentiles=(15.0, 99.5)
)'''

slice_paths = config.get_existing_slice_paths("field_only_y", "xy")
bg = "clipped_mesh_y/piece_01.png"

_ = [
    config.composite_images(
        background_relpath=p.name,
        overlay_relpath=bg,
        out_relpath=f"composited/{p.stem}.png"
    )
    for p in slice_paths
]

#config.plot_far_field(tol=20, n_grid_points = 400, plane=0, lims=[-10, 10, -10, 10], c =-0.9888495000000002)

#config.plot_gmres_residuals_first_rhs(save_plot=True)