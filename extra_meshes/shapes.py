
import numpy as _np
from collections import defaultdict as _defaultdict
from pathlib import Path as _Path


def get_gmsh_file():
    """
    Create a new temporary gmsh file.

    Return a 3-tuple (geo_file,geo_name,msh_name), where
    geo_file is a file descriptor to an empty .geo file, geo_name is
    the corresponding filename and msh_name is the name of the
    Gmsh .msh file that will be generated.

    """
    import os
    import tempfile
    import bempp_cl.api

    geo, geo_name = tempfile.mkstemp(suffix=".geo", dir=bempp_cl.api.TMP_PATH, text=True)
    geo_file = os.fdopen(geo, "w")
    msh_name = os.path.splitext(geo_name)[0] + ".msh"
    return (geo_file, geo_name, msh_name)


def __generate_grid_from_gmsh_string(gmsh_string):
    """Return a grid from a string containing a gmsh mesh."""
    import os
    import tempfile
    import bempp_cl.api

    if bempp_cl.api.mpi_rank == 0:
        # First create the grid.
        handle, fname = tempfile.mkstemp(suffix=".msh", dir=bempp_cl.api.TMP_PATH, text=True)
        with os.fdopen(handle, "w") as f:
            f.write(gmsh_string)
    grid = bempp_cl.api.import_grid(fname)
    bempp_cl.api.mpi_comm.Barrier()
    if bempp_cl.api.mpi_rank == 0:
        os.remove(fname)
    return grid


def __generate_grid_from_geo_string(geo_string):
    """Create a grid from a gmsh geo string."""
    import os
    import subprocess
    import bempp_cl.api

    def msh_from_string(geo_string):
        """Create a mesh from a string."""
        gmsh_command = bempp_cl.api.GMSH_PATH
        if gmsh_command is None:
            raise RuntimeError("Gmsh is not found. Cannot generate mesh")
        f, geo_name, msh_name = get_gmsh_file()
        f.write(geo_string)
        f.close()

        fnull = open(os.devnull, "w")
        cmd = gmsh_command + " -2 " + geo_name
        try:
            subprocess.check_call(cmd, shell=True, stdout=fnull, stderr=fnull)
        except:  # noqa: E722
            print("The following command failed: " + cmd)
            fnull.close()
            raise
        os.remove(geo_name)
        fnull.close()
        return msh_name

    msh_name = msh_from_string(geo_string)
    grid = bempp_cl.api.import_grid(msh_name)
    os.remove(msh_name)
    return grid


_F16_BODY_GEO_PATH = _Path(__file__).with_name("f16_body.geo")
_F16_PHYSICAL_SURFACE_TAG = 1102
_F16_ACTIVE_SURFACES = [
    1030, 814, 1074, 696, 979, 942, 938, 826, 816, 824, 818, 822, 934, 820, 650,
    786, 676, 956, 646, 912, 932, 682, 782, 648, 720, 700, 680, 702, 726, 728, 690,
    692, 830, 1023, 1034, 974, 972, 1083, 1047, 1049, 977, 722, 914, 730, 954, 848,
    788, 846, 1073, 678, 926, 804, 736, 746, 732, 740, 778, 664, 652, 662, 658, 748,
    660, 860, 862, 762, 844, 760, 768, 656, 750, 764, 766, 928, 910, 930, 790, 868,
    836, 770, 772, 794, 806, 792, 866, 834, 796, 886, 798, 888, 854, 858, 852, 864,
    850, 880, 878, 876, 870, 884, 874, 882, 856, 800, 802, 774, 776, 840, 838, 842,
    952, 890, 734, 744, 758, 738, 742, 752, 724, 916, 754, 718, 918, 810, 808, 780,
    950, 654, 666, 716, 958, 892, 986, 984, 712, 1043, 1039, 894, 1025, 1041, 1019,
    988, 714, 990, 1079, 1078, 1082, 1084, 1081, 1080, 1077, 1076, 1075, 1055, 1057,
    784, 832, 698, 694, 688, 686, 684, 872, 922, 920, 924, 828, 1021, 936,
]
_F16_REVERSED_SURFACES = [
    984, 977, 979, 972, 974, 646, 1079, 712, 830, 754, 958, 666, 716, 696, 752, 782,
    1075, 1083, 648, 662, 736, 742, 914, 724, 916, 918, 744, 676, 718, 1074, 702, 776,
    686, 688, 954, 738, 678, 890, 740, 838, 1080, 750, 1025, 928, 730, 778, 746, 1077,
    772, 794, 806, 930, 774, 862, 910, 1082, 860, 658, 796, 866, 868, 748, 864, 886,
    804, 894, 888, 858, 892, 922, 1039, 780, 810, 934, 932, 726, 1076, 1084, 870, 872,
    874, 1049, 1047,
]


def __gmsh_integer_list(values):
    """Serialize integer ids as a gmsh list."""
    return ", ".join(str(value) for value in values)


def __generate_f16_geo_string(h):
    """Create the cleaned F16 geo script with explicit active/reversed surfaces."""
    body = _F16_BODY_GEO_PATH.read_text()
    return (
        "cl = "
        + str(h)
        + ";\n"
        + body
        + "\nReverse Surface {"
        + __gmsh_integer_list(_F16_REVERSED_SURFACES)
        + "};\nPhysical Surface("
        + str(_F16_PHYSICAL_SURFACE_TAG)
        + ") = {"
        + __gmsh_integer_list(_F16_ACTIVE_SURFACES)
        + "};\n"
    )


def __validate_closed_oriented_surface_grid(grid, label):
    """Validate that a triangular surface grid is closed, manifold, and consistently oriented."""
    vertices = _np.asarray(grid.vertices, dtype=_np.float64)
    elements = _np.asarray(grid.elements, dtype=_np.uint32)

    edge_to_elements = _defaultdict(list)
    for elem_index, (a, b, c) in enumerate(elements.T):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_to_elements[(min(int(u), int(v)), max(int(u), int(v)))].append(elem_index)

    boundary_edges = [edge for edge, attached in edge_to_elements.items() if len(attached) == 1]
    nonmanifold_edges = [edge for edge, attached in edge_to_elements.items() if len(attached) > 2]
    if boundary_edges or nonmanifold_edges:
        raise RuntimeError(
            f"{label} mesh is not a closed manifold surface "
            f"(boundary_edges={len(boundary_edges)}, nonmanifold_edges={len(nonmanifold_edges)})"
        )

    same_direction_edges = 0
    for attached in edge_to_elements.values():
        if len(attached) != 2:
            continue
        first_index, second_index = attached
        first = elements[:, first_index]
        second = elements[:, second_index]
        shared = [vertex for vertex in first if vertex in set(second)]
        local_first = {int(first[local]): local for local in range(3)}
        local_second = {int(second[local]): local for local in range(3)}
        u, v = int(shared[0]), int(shared[1])
        first_forward = (local_first[v] - local_first[u]) % 3 == 1
        second_forward = (local_second[v] - local_second[u]) % 3 == 1
        if first_forward == second_forward:
            same_direction_edges += 1
    if same_direction_edges:
        raise RuntimeError(
            f"{label} mesh has inconsistent surface orientation "
            f"(same_direction_edges={same_direction_edges})"
        )

    center = vertices.mean(axis=1, keepdims=True)
    v0 = vertices[:, elements[0]] - center
    v1 = vertices[:, elements[1]] - center
    v2 = vertices[:, elements[2]] - center
    signed_volume = _np.einsum("ij,ij->j", v0, _np.cross(v1.T, v2.T).T).sum() / 6.0
    if signed_volume <= 0:
        raise RuntimeError(
            f"{label} mesh is not outward oriented (signed_volume={signed_volume})"
        )

    return grid


def dihedral_with_y(h=0.1):
    stub = """
    Point (1) = {0.009500004128710764, 0.1645000358712893, -5.791675117793049e-09, cl};
    Point (2) = {0.009500003790414172, 0.1644999912095858, 0.2189999870362379, cl};
    Point (3) = {0.009500000600000001, 0.009500000600000001, 0, cl};
    Point (4) = {0.009500015456791903, 0.009499997543208091, 0.2189999861750472, cl};
    Point (5) = {0, 0, 0, cl};
    Point (6) = {1.339085443224919e-09, 0.1550000357609146, -6.576409464307403e-09, cl};
    Point (7) = {7.094896031789588e-09, 0.155000001905104, 0.2189999848714535, cl};
    Point (8) = {-1.10045387594071e-08, 1.100453876005762e-08, 0.1349000488252119, cl};
    Point (9) = {-6.580077595297329e-08, -1.419922405956203e-08, 0.2189999625673587, cl};
    Point (10) = {-1.794777204633391e-08, 0.155000026947772, 0.1349000445450397, cl};
    Point (11) = {-4.919243772744109e-09, 0.03600000391924377, 0.08409999688432511, cl};
    Point (12) = {-2.24710660495786e-09, 0.03600000624710661, 0.1349000205254881, cl};
    Point (13) = {-1.000886436924864e-08, 0.1550000400088644, 0.08410002166924359, cl};
    Point (14) = {-0.01030000147546802, 0.02570000147546801, 0.1349000221490764, cl};
    Point (15) = {-0.01030000095318929, 0.004266398953189287, 0.1349000126130908, cl};
    Point (16) = {-0.03575584648354788, -0.02118944451645212, 0.08409999929311533, cl};
    Point (17) = {-0.03575584672482789, -0.02118944127517211, 0.1349000109895025, cl};
    Point (18) = {-0.01029999686415022, 0.004266395864150223, 0.08410000199909583, cl};
    Point (19) = {0.004266404889736085, -0.01030000788973609, 0.134899993461174, cl};
    Point (21) = {-0.02118944818767472, -0.03575584181232528, 0.08410001301817825, cl};
    Point (22) = {0.02569999747638256, -0.01029999747638257, 0.08410001380338014, cl};
    Point (23) = {0.02570000652619551, -0.01030000652619552, 0.1348999831133942, cl};
    Point (24) = {0.004266400469783174, -0.01030000146978318, 0.0841000154535607, cl};
    Point (25) = {0.03599999647638257, 1.523617435342239e-09, 0.08410001380338016, cl};
    Point (26) = {0.0360000055261955, -7.526195511860928e-09, 0.1348999831133942, cl};
    Point (27) = {0.1645000358712893, 0.009500004128710764, 5.791675118660411e-09, cl};
    Point (28) = {0.1550000357609146, 1.339085442357557e-09, 6.576409465608446e-09, cl};
    Point (29) = {0.1644999699500608, 0.009499991049939141, 0.21899997232515, cl};
    Point (30) = {0.1549999699500609, -8.950060855633657e-09, 0.21899997232515, cl};
    Point (31) = {2.946482551002827e-09, -2.946482550677566e-09, 0.0841000009854376, cl};
    Point (32) = {-0.01030000410954494, 0.02570000010954494, 0.08409999823731536, cl};
    Point (33) = {0.1549999670136561, -9.913656068803733e-09, 0.1348999681583306, cl};
    Point (35) = {0.1550000386973194, 2.302680647721378e-09, 0.08410001074322893, cl};
    Point (39) = {-0.0211894, -0.03575, 0.1349, cl};
    Line (2) = {2, 1};
    Line (3) = {4, 2};
    Line (4) = {4, 29};
    Line (5) = {30, 9};
    Line (6) = {9, 7};
    Line (7) = {2, 7};
    Line (8) = {29, 30};
    Line (9) = {29, 27};
    Line (10) = {30, 33};
    Line (11) = {27, 3};
    Line (12) = {1, 3};
    Line (13) = {3, 4};
    Line (14) = {1, 6};
    Line (15) = {7, 10};
    Line (16) = {10, 13};
    Line (17) = {13, 6};
    Line (18) = {6, 5};
    Line (19) = {33, 35};
    Line (20) = {35, 28};
    Line (21) = {27, 28};
    Line (22) = {28, 5};
    Line (23) = {35, 25};
    Line (24) = {33, 26};
    Line (25) = {25, 26};
    Line (26) = {26, 23};
    Line (27) = {25, 22};
    Line (28) = {22, 23};
    Line (29) = {23, 19};
    Line (30) = {22, 24};
    Line (31) = {12, 14};
    Line (32) = {14, 15};
    Line (33) = {15, 18};
    Line (34) = {18, 32};
    Line (35) = {32, 11};
    Line (36) = {11, 12};
    Line (37) = {14, 32};
    Line (38) = {12, 8};
    Line (39) = {15, 8};
    Line (40) = {18, 31};
    Line (41) = {15, 17};
    Line (45) = {17, 16};
    Line (46) = {19, 8};
    Line (47) = {8, 26};
    Line (48) = {24, 19};
    Line (49) = {16, 21};
    Line (50) = {16, 18};
    Line (51) = {21, 24};
    Line (52) = {8, 9};
    Line (53) = {5, 31};
    Line (54) = {31, 24};
    Line (55) = {25, 31};
    Line (56) = {11, 31};
    Line (57) = {13, 11};
    Line (58) = {10, 12};
    Line (59) = {21, 39};
    Line (60) = {39, 19};
    Line (61) = {17, 39};
    Line (68) = {4, 9};
    Line (69) = {5, 3};
    Line Loop (63) = {3, 2, 12, 13};
    Plane Surface (63) = {63};
    Line Loop (65) = {9, 11, 13, 4};
    Plane Surface (65) = {65};
    Line Loop (67) = {21, -20, -19, -10, -8, 9};
    Plane Surface (67) = {67};
    Line Loop (71) = {22, 69, -11, 21};
    Plane Surface (71) = {71};
    Line Loop (73) = {18, 69, -12, 14};
    Plane Surface (73) = {73};
    Line Loop (75) = {8, 5, -68, 4};
    Plane Surface (75) = {75};
    Line Loop (77) = {68, 6, -7, -3};
    Plane Surface (77) = {77};
    Line Loop (79) = {7, 15, 16, 17, -14, -2};
    Plane Surface (79) = {79};
    Line Loop (81) = {52, -5, 10, 24, -47};
    Plane Surface (81) = {81};
    Line Loop (83) = {25, -24, 19, 23};
    Plane Surface (83) = {83};
    Line Loop (85) = {53, -55, -23, 20, 22};
    Plane Surface (85) = {85};
    Line Loop (87) = {57, 56, -53, -18, -17};
    Plane Surface (87) = {87};
    Line Loop (89) = {57, 36, -58, 16};
    Plane Surface (89) = {89};
    Line Loop (91) = {58, 38, 52, 6, 15};
    Plane Surface (91) = {91};
    Line Loop (93) = {47, 26, 29, 46};
    Plane Surface (93) = {93};
    Line Loop (95) = {26, -28, -27, 25};
    Plane Surface (95) = {95};
    Line Loop (97) = {29, -48, -30, 28};
    Plane Surface (97) = {97};
    Line Loop (99) = {39, -46, -60, -61, -41};
    Plane Surface (99) = {99};
    Line Loop (101) = {45, 49, 59, -61};
    Plane Surface (101) = {101};
    Line Loop (103) = {33, -50, -45, -41};
    Plane Surface (103) = {103};
    Line Loop (105) = {30, -54, -55, 27};
    Plane Surface (105) = {105};
    Line Loop (107) = {37, 35, 36, 31};
    Plane Surface (107) = {107};
    Line Loop (109) = {33, 34, -37, 32};
    Plane Surface (109) = {109};
    Line Loop (111) = {38, -39, -32, -31};
    Plane Surface (111) = {111};
    Line Loop (113) = {48, -60, -59, 51};
    Plane Surface (113) = {113};
    Line Loop (115) = {56, -40, 34, 35};
    Plane Surface (115) = {115};
    Line Loop (117) = {40, 54, -51, -49, 50};
    Plane Surface (117) = {117};
    """

    geometry = "cl = " + str(h) + ";\n" + stub
    return __generate_grid_from_geo_string(geometry)


def device(h=0.1):
    stub = """
    R1 = 0.201;
    r1 = 0.0135;

    // the base cylinder extrusion heights
    h1 = 0.032;
    h2 = h1 + 0.0366/2.0;
    h3 = h1 + 0.0366;
    h4 = 0.102;

    Point(newp) = {0,0,0,cl};
    Point(newp) = {0,-R1,0,cl};
    Point(newp) = {0,-R1+0.058,0,cl};
    Point(newp) = {-0.14,-R1+0.058,0,cl};
    Point(newp) = {0,R1,0,cl};
    Point(newp) = {-R1,0,0,cl};
    Point(newp) = {0.0,-r1,0,cl};
    Point(newp) = {r1,0,0,cl};
    Point(newp) = {0,r1,0,cl};
    Point(newp) = {-r1,0,0,cl};

    // the tip
    Point(newp) = {0.151,0,0,cl};
    TipAngle = 51.5 / 180.0 * Pi;
    LengthSideTip = 0.074;
    Point(newp) = {0.151 - LengthSideTip * Cos(TipAngle/2.0), LengthSideTip * Sin(TipAngle/2.0),0,cl};
    Point(newp) = {0.151 - LengthSideTip * Cos(TipAngle/2.0), -LengthSideTip * Sin(TipAngle/2.0),0,cl};
    Point(newp) = {0.14,-R1+0.058,0,cl};
    Point(newp) = {0.14,R1-0.058,0,cl};

    // same points as above, but extruded following h2
    Point(newp) = {-0.14,-R1+0.058,h2,cl};
    Point(newp) = {-R1,0,h2,cl};
    Point(newp) = {0,R1,h2,cl};
    Point(newp) = {0.14,R1-0.058,h2,cl};

    // same points as above, but extruded following h4
    Point(newp) = {0,-R1,h4,cl};
    Point(newp) = {0,-R1+0.058,h4,cl};
    Point(newp) = {-0.14,-R1+0.058,h4,cl};
    Point(newp) = {0,R1,h4,cl};
    Point(newp) = {-R1,0,h4,cl};

    Point(newp) = {0.151,0,h4,cl};
    Point(newp) = {0.151 - LengthSideTip * Cos(TipAngle/2.0), LengthSideTip * Sin(TipAngle/2.0),h4,cl};
    Point(newp) = {0.151 - LengthSideTip * Cos(TipAngle/2.0), -LengthSideTip * Sin(TipAngle/2.0),h4,cl};
    Point(newp) = {0.14,-R1+0.058,h4,cl};
    Point(newp) = {0.14,R1-0.058,h4,cl};


    // the points for the 2 intakes on the lower cylinder
    AngleBetweenIntakesEdges = 0.174/R1; // [rad]
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), -R1 * Sin(AngleBetweenIntakesEdges/2.0),0.0,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), R1 * Sin(AngleBetweenIntakesEdges/2.0),0.0,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), -R1 * Sin(AngleBetweenIntakesEdges/2.0),h2,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), R1 * Sin(AngleBetweenIntakesEdges/2.0),h2,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), -R1 * Sin(AngleBetweenIntakesEdges/2.0),h4,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), R1 * Sin(AngleBetweenIntakesEdges/2.0),h4,cl};

    AngleBetweenIntakesEdges = (0.174+0.0366)/R1; // [rad]
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), -R1 * Sin(AngleBetweenIntakesEdges/2.0),0.0,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), R1 * Sin(AngleBetweenIntakesEdges/2.0),0.0,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), -R1 * Sin(AngleBetweenIntakesEdges/2.0),h1,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), R1 * Sin(AngleBetweenIntakesEdges/2.0),h1,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), -R1 * Sin(AngleBetweenIntakesEdges/2.0),h2,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), R1 * Sin(AngleBetweenIntakesEdges/2.0),h2,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), -R1 * Sin(AngleBetweenIntakesEdges/2.0),h3,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), R1 * Sin(AngleBetweenIntakesEdges/2.0),h3,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), -R1 * Sin(AngleBetweenIntakesEdges/2.0),h4,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), R1 * Sin(AngleBetweenIntakesEdges/2.0),h4,cl};

    // the intakes
    Point(newp) = {(-R1+0.0834) * Cos(AngleBetweenIntakesEdges/2.0), (R1-0.0834) * Sin(AngleBetweenIntakesEdges/2.0),h1,cl};
    Point(newp) = {(-R1+0.0834) * Cos(AngleBetweenIntakesEdges/2.0), (R1-0.0834) * Sin(AngleBetweenIntakesEdges/2.0),h2,cl};
    Point(newp) = {(-R1+0.0834) * Cos(AngleBetweenIntakesEdges/2.0) - 0.0366/2.0 * Sin(AngleBetweenIntakesEdges/2.0), (R1-0.0834) * Sin(AngleBetweenIntakesEdges/2.0) - 0.0366/2.0 * Cos(AngleBetweenIntakesEdges/2.0),h2,cl};
    Point(newp) = {(-R1+0.0834) * Cos(AngleBetweenIntakesEdges/2.0) + 0.0366/2.0 * Sin(AngleBetweenIntakesEdges/2.0), (R1-0.0834) * Sin(AngleBetweenIntakesEdges/2.0) + 0.0366/2.0 * Cos(AngleBetweenIntakesEdges/2.0),h2,cl};
    Point(newp) = {(-R1+0.0834) * Cos(AngleBetweenIntakesEdges/2.0), (R1-0.0834) * Sin(AngleBetweenIntakesEdges/2.0),h3,cl};


    Point(newp) = {(-R1+0.0834) * Cos(AngleBetweenIntakesEdges/2.0), (-R1+0.0834) * Sin(AngleBetweenIntakesEdges/2.0),h1,cl};
    Point(newp) = {(-R1+0.0834) * Cos(AngleBetweenIntakesEdges/2.0), (-R1+0.0834) * Sin(AngleBetweenIntakesEdges/2.0),h2,cl};
    Point(newp) = {(-R1+0.0834) * Cos(AngleBetweenIntakesEdges/2.0) + 0.0366/2.0 * Sin(AngleBetweenIntakesEdges/2.0), (-R1+0.0834) * Sin(AngleBetweenIntakesEdges/2.0) - 0.0366/2.0 * Cos(AngleBetweenIntakesEdges/2.0),h2,cl};
    Point(newp) = {(-R1+0.0834) * Cos(AngleBetweenIntakesEdges/2.0) - 0.0366/2.0 * Sin(AngleBetweenIntakesEdges/2.0), (-R1+0.0834) * Sin(AngleBetweenIntakesEdges/2.0) + 0.0366/2.0 * Cos(AngleBetweenIntakesEdges/2.0),h2,cl};
    Point(newp) = {(-R1+0.0834) * Cos(AngleBetweenIntakesEdges/2.0), (-R1+0.0834) * Sin(AngleBetweenIntakesEdges/2.0),h3,cl};


    AngleBetweenIntakesEdges = (0.174 + 2.0 * 0.0366)/R1; // [rad]
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), -R1 * Sin(AngleBetweenIntakesEdges/2.0),0.0,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), R1 * Sin(AngleBetweenIntakesEdges/2.0),0.0,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), -R1 * Sin(AngleBetweenIntakesEdges/2.0),h2,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), R1 * Sin(AngleBetweenIntakesEdges/2.0),h2,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), -R1 * Sin(AngleBetweenIntakesEdges/2.0),h4,cl};
    Point(newp) = {-R1 * Cos(AngleBetweenIntakesEdges/2.0), R1 * Sin(AngleBetweenIntakesEdges/2.0),h4,cl};

    // add the centres as we go up...
    Point(newp) = {0.0, 0.0, h2,cl};
    Point(newp) = {0.0, 0.0, h4,cl};

    // the upper cylinder
    R2 = 0.1; // 9.95 cm really :)
    Point(newp) = {0.0506,0,h4,cl};
    Point(newp) = {0.0506-R2,0,h4,cl};
    Point(newp) = {0.0506-R2,-R2,h4,cl};
    Point(newp) = {0.0506-R2,R2,h4,cl};
    Point(newp) = {0.0506-0.1425,0,h4,cl};
    Point(newp) = {0.0506-0.1425,-0.05,h4,cl};
    Point(newp) = {0.0506-0.1425,0.05,h4,cl};

    cos = 0.05/R2;
    sin = Sqrt(1.0 - cos^2);
    Point(newp) = {0.0506 -R2 - sin * R2,0.05,h4,cl};
    Point(newp) = {0.0506 -R2 - sin * R2,-0.05,h4,cl};

    // the curves for the hollow triangle in the upper cylinder
    cos = 0.03/R2;
    sin = Sqrt(1.0 - cos^2);
    Point(newp) = {0.0506 -R2 - cos * R2, sin * R2,h4,cl};
    Point(newp) = {0.0506 -R2 + cos * R2, sin * R2,h4,cl};



    Line(1) = {2,3};
    Line(3) = {14,13};
    Line(4) = {13,11};
    Line(5) = {11,12};
    Line(6) = {12,15};
    Line(7) = {15,19};
    Line(8) = {19,29};
    Line(9) = {29,26};
    Line(10) = {26,25};
    Line(11) = {25,27};
    Line(12) = {27,28};
    Line(14) = {21,20};
    Line(15) = {20,2};
    Line(16) = {3,21};
    Line(17) = {28,14};
    Line(18) = {27,13};
    Line(19) = {25,11};
    Line(20) = {12,26};


    Line(21) = {3,4};
    Line(22) = {22,21};
    Line(23) = {22,16};
    Line(24) = {16,4};
    Line(25) = {60,58};
    Line(26) = {58,56};
    Line(27) = {36,38};
    Line(28) = {42,44};
    Line(29) = {34,32};
    Line(30) = {32,30};
    Line(31) = {24,17};
    Line(32) = {17,6};
    Line(33) = {35,33};
    Line(34) = {33,31};
    Line(35) = {45,43};
    Line(36) = {39,37};
    Line(37) = {61,59};
    Line(38) = {59,57};
    Line(39) = {23,18};
    Line(40) = {18,5};
    Line(41) = {53,58};
    Line(42) = {38,51};
    Line(43) = {54,32};
    Line(44) = {55,42};
    Line(45) = {50,43};
    Line(46) = {48,33};
    Line(47) = {46,39};
    Line(48) = {49,59};
    Line(49) = {8,13};
    Line(50) = {8,12};
    Line(51) = {9,5};
    Line(52) = {7,3};
    Line(53) = {10,6};

    Line(54) = {69,70};
    Line(55) = {69,72};
    Line(56) = {70,71};
    Circle(57) = {7,1,8};
    Circle(58) = {8,1,9};

    Circle(59) = {9,1,10};
    Circle(60) = {10,1,7};
    Circle(61) = {2,1,14};
    Circle(62) = {15,1,5};
    Circle(63) = {5,1,57};
    Circle(64) = {57,1,37};
    Circle(65) = {37,1,31};
    Circle(66) = {31,1,6};
    Circle(67) = {6,1,30};
    Circle(68) = {30,1,36};
    Circle(69) = {36,1,56};
    Circle(70) = {56,1,4};
    Circle(71) = {20,63,28};
    Circle(72) = {29,63,23};
    Circle(73) = {23,63,61};
    Circle(74) = {61,63,45};
    Circle(75) = {45,63,35};
    Circle(76) = {35,63,24};
    Circle(77) = {24,63,34};
    Circle(78) = {34,63,44};
    Circle(79) = {44,63,60};
    Circle(80) = {60,63,22};
    Circle(81) = {42,40,32};
    Circle(82) = {32,40,38};
    Circle(83) = {38,40,58};
    Circle(84) = {58,40,42};
    Circle(85) = {43,41,59};
    Circle(86) = {59,41,39};
    Circle(87) = {39,41,33};
    Circle(88) = {33,41,43};
    Circle(89) = {50,47,49};
    Circle(90) = {49,47,46};
    Circle(91) = {46,47,48};
    Circle(92) = {48,47,50};
    Circle(93) = {55,52,54};
    Circle(94) = {54,52,51};
    Circle(95) = {51,52,53};
    Circle(96) = {53,52,55};

    Circle(97) = {19,62,18};
    Circle(98) = {18,62,59};
    Circle(99) = {33,62,17};
    Circle(100) = {17,62,32};
    Circle(101) = {58,62,16};

    // circles for the upper cylinder


    Circle(102) = {64,65,66};
    Circle(103) = {66,65,72};
    Circle(104) = {71,65,73};
    Circle(105) = {73,65,67};
    Circle(106) = {67,65,74};
    Circle(107) = {74,65,64};
    Line Loop(108) = {22,-16,21,-24,-23};
    Plane Surface(109) = {108};
    Line Loop(110) = {16,14,15,1};
    Plane Surface(111) = {110};
    Line Loop(112) = {12,17,3,-18};
    Plane Surface(113) = {112};
    Line Loop(114) = {11,18,4,-19};
    Plane Surface(115) = {114};
    Line Loop(116) = {10,19,5,20};
    Plane Surface(117) = {116};
    Line Loop(118) = {6,7,8,9,-20};
    Plane Surface(119) = {118};
    Line Loop(120) = {49,4,5,-50};
    Plane Surface(121) = {120};
    Line Loop(122) = {52,-1,61,3,-49,-57};
    Plane Surface(123) = {122};
    Line Loop(124) = {51,-62,-6,-50,58};
    Plane Surface(125) = {124};
    Line Loop(126) = {53,-66,-65,-64,-63,-51,59};
    Plane Surface(127) = {126};
    Line Loop(128) = {53,67,68,69,70,-21,-52,-60};
    Plane Surface(129) = {128};
    Line(130) = {66,21};
    Line(131) = {67,23};

    Line(132) = {72,34};
    Line(133) = {71,35};
    Line(134) = {64,27};
    Line(135) = {64,26};
    Line Loop(136) = {130,14,71,-12,-134,102};
    Plane Surface(137) = {136};
    Line Loop(138) = {131,-72,9,-135,-107,-106};
    Plane Surface(139) = {138};
    Line Loop(140) = {131,73,74,75,-133,104,105};
    Plane Surface(141) = {140};
    Line Loop(142) = {22,-130,103,132,78,79,80};
    Plane Surface(143) = {142};
    Line Loop(144) = {132,-77,-76,-133,-56,-54,55};
    Plane Surface(145) = {144};
    Line Loop(146) = {89,90,91,92};
    Plane Surface(147) = {146};
    Line Loop(148) = {93,94,95,96};
    Plane Surface(149) = {148};
    Line Loop(150) = {72,39,-97,8};
    Ruled Surface(151) = {150};
    Line Loop(152) = {97,40,-62,7};
    Ruled Surface(153) = {152};
    Line Loop(154) = {73,37,-98,-39};
    Ruled Surface(155) = {154};
    Line Loop(156) = {98,38,-63,-40};
    Ruled Surface(157) = {156};
    Line Loop(158) = {76,31,-99,-33};
    Ruled Surface(159) = {158};
    Line Loop(160) = {99,32,-66,-34};
    Ruled Surface(161) = {160};
    Line Loop(162) = {77,29,-100,-31};
    Ruled Surface(163) = {162};
    Line Loop(164) = {100,30,-67,-32};
    Ruled Surface(165) = {164};
    Line Loop(166) = {80,23,-101,-25};
    Ruled Surface(167) = {166};
    Line Loop(168) = {101,24,-70,-26};
    Ruled Surface(169) = {168};
    Line Loop(170) = {78,-28,81,-29};
    Ruled Surface(171) = {170};
    Line Loop(172) = {79,25,84,28};
    Ruled Surface(173) = {172};
    Line Loop(174) = {82,-27,-68,-30};
    Ruled Surface(175) = {174};
    Line Loop(176) = {27,83,26,-69};
    Ruled Surface(177) = {176};
    Line Loop(178) = {65,-34,-87,36};
    Ruled Surface(179) = {178};
    Line Loop(180) = {36,-64,-38,86};
    Ruled Surface(181) = {180};
    Line Loop(182) = {74,35,85,-37};
    Ruled Surface(183) = {182};
    Line Loop(184) = {75,33,88,-35};
    Ruled Surface(185) = {184};
    Line Loop(186) = {45,85,-48,-89};
    Ruled Surface(187) = {186};
    Line Loop(188) = {45,-88,-46,92};
    Ruled Surface(189) = {188};
    Line Loop(190) = {46,-87,-47,91};
    Ruled Surface(191) = {190};
    Line Loop(192) = {48,86,-47,-90};
    Ruled Surface(193) = {192};
    Line Loop(194) = {44,81,-43,-93};
    Ruled Surface(195) = {194};
    Line Loop(196) = {96,44,-84,-41};
    Ruled Surface(197) = {196};
    Line Loop(198) = {43,82,42,-94};
    Ruled Surface(199) = {198};
    Line Loop(200) = {95,41,-83,42};
    Ruled Surface(201) = {200};
    Line Loop(202) = {135,10,11,-134};
    Plane Surface(203) = {202};
    Line Loop(204) = {71,17,-61,-15};
    Ruled Surface(205) = {204};
    """

    geometry = "cl = " + str(h) + ";\n" + stub
    return __generate_grid_from_geo_string(geometry)



def f16(h=0.1):
    geometry = __generate_f16_geo_string(h)
    grid = __generate_grid_from_geo_string(geometry)
    return __validate_closed_oriented_surface_grid(grid, "F16")


def ridged_horn_tem_antenna(h=0.1):
    stub = """
    // some lengths

    a_1 = 0.238; // first aperture
    b_1 = 0.138; // first aperture
    h_1 = 0.0; // depth of 'first aperture'

    a_2 = 0.087; // outer contour of 'second aperture'
    a_2b = 0.085; // inner contour of 'second aperture'
    b_2 = 0.064; // inner contour of 'second aperture'
    h_2 = -0.1555; // depth of 'second aperture'

    a_3 = 0.025; // bottom aperture
    b_3 = 0.016; // bottom aperture
    h_3 = -0.189; // depth of 'bottom aperture'

    w_ridge = 0.0075; // width of ridge
    // aperture

    Point(1) = {-a_1/2.0, -b_1/2.0, h_1, lc_coarse};
    Point(2) = {a_1/2.0, -b_1/2.0, h_1, lc_coarse};
    Point(3) = {a_1/2.0, b_1/2.0, h_1, lc_coarse};
    Point(4) = {-a_1/2.0, b_1/2.0, h_1, lc_coarse};


    // outer contour of 'second aperture'

    Point(5) = {-a_2/2.0, -b_2/2.0, h_2, lc_detail};
    Point(6) = {a_2/2.0, -b_2/2.0, h_2, lc_detail};
    Point(7) = {a_2/2.0, b_2/2.0, h_2, lc_detail};
    Point(8) = {-a_2/2.0, b_2/2.0, h_2, lc_detail};


    // inner contour of 'second aperture'

    Point(9) = {-a_2b/2.0, -b_2/2.0, h_2, lc_detail};
    Point(10) = {a_2b/2.0, -b_2/2.0, h_2, lc_detail};
    Point(11) = {a_2b/2.0, b_2/2.0, h_2, lc_detail};
    Point(12) = {-a_2b/2.0, b_2/2.0, h_2, lc_detail};


    // contour of 'bottom aperture'

    Point(13) = {-a_3/2.0, -b_3/2.0, h_3, lc_detail};
    Point(14) = {a_3/2.0, -b_3/2.0, h_3, lc_detail};
    Point(15) = {a_3/2.0, b_3/2.0, h_3, lc_detail};
    Point(16) = {-a_3/2.0, b_3/2.0, h_3, lc_detail};


    // 2 opposite slopes of 1.6 cm width from inner contour (shortest sides) of 'second aperture' to 'bottom aperture'

    Point(17) = {-a_2b/2.0,-b_3/2.0, h_2,lc_detail};
    Point(18) = {-a_2b/2.0,b_3/2.0, h_2,lc_detail};
    Point(19) = {a_2b/2.0,-b_3/2.0, h_2,lc_detail};
    Point(20) = {a_2b/2.0,b_3/2.0, h_2,lc_detail};
    Point(21) = {-a_3/2.0,-b_3/2.0,-0.183,lc_detail};
    Point(22) = {a_3/2.0,-b_3/2.0,-0.183,lc_detail};
    Point(23) = {a_3/2.0,b_3/2.0,-0.183,lc_detail};
    Point(24) = {-a_3/2.0,b_3/2.0,-0.183,lc_detail};


    // 4 opposite slopes of 8.5 cm width from inner contour (longest sides) of 'second aperture' to 'bottom aperture'

    Point(25) = {-a_2b/2.0, -b_3/2.0, -0.184, lc_detail};
    Point(26) = {-a_2b/2.0, b_3/2.0, -0.184, lc_detail};
    Point(27) = {a_2b/2.0, -b_3/2.0, -0.184, lc_detail};
    Point(28) = {a_2b/2.0, b_3/2.0, -0.184, lc_detail};
    Point(29) = {-a_3/2.0, -b_3/2.0, -0.184, lc_detail};
    Point(30) = {a_3/2.0, -b_3/2.0, -0.184, lc_detail};
    Point(31) = {a_3/2.0, b_3/2.0, -0.184, lc_detail};
    Point(32) = {-a_3/2.0, b_3/2.0, -0.184, lc_detail};

    // Ridge
    // bottom points
    Point(33) = {-w_ridge/2.0, -0.00075, -0.18, lc_detail};
    Point(34) = {w_ridge/2.0, -0.00075, -0.18, lc_detail};
    Point(35) = {w_ridge/2.0, 0.00075, -0.18, lc_detail};
    Point(36) = {-w_ridge/2.0, 0.00075, -0.18, lc_detail};
    Point(37) = {-w_ridge/2.0, -b_3/2.0, -0.18, lc_detail};
    Point(38) = {w_ridge/2.0, -b_3/2.0, -0.18, lc_detail};
    Point(39) = {w_ridge/2.0, b_3/2.0, -0.18, lc_detail};
    Point(40) = {-w_ridge/2.0, b_3/2.0, -0.18, lc_detail};

    // intersection with 'second aperture'
    Point(41) = {-w_ridge/2.0, -b_2/2.0, h_2, lc_detail};
    Point(42) = {w_ridge/2.0, -b_2/2.0, h_2, lc_detail};
    Point(43) = {w_ridge/2.0, b_2/2.0, h_2, lc_detail};
    Point(44) = {-w_ridge/2.0, b_2/2.0, h_2, lc_detail};

    // intersection with 'first aperture'
    Point(45) = {-w_ridge/2.0, -b_1/2.0, h_1, lc_coarse};
    Point(46) = {w_ridge/2.0, -b_1/2.0, h_1, lc_coarse};
    Point(47) = {w_ridge/2.0, b_1/2.0, h_1, lc_coarse};
    Point(48) = {-w_ridge/2.0, b_1/2.0, h_1, lc_coarse};

    // intersection with the 4 opposite slopes
    Point(49) = {-w_ridge/2.0, -b_3/2.0, -0.184, lc_detail};
    Point(50) = {w_ridge/2.0, -b_3/2.0, -0.184, lc_detail};
    Point(51) = {w_ridge/2.0, b_3/2.0, -0.184, lc_detail};
    Point(52) = {-w_ridge/2.0, b_3/2.0, -0.184, lc_detail};

    // intersection with the 'bottom aperture'
    Point(53) = {-w_ridge/2.0, -b_3/2.0, h_3, lc_detail};
    Point(54) = {w_ridge/2.0, -b_3/2.0, h_3, lc_detail};
    Point(55) = {w_ridge/2.0, b_3/2.0, h_3, lc_detail};
    Point(56) = {-w_ridge/2.0, b_3/2.0, h_3, lc_detail};

    // ridge profile (resulting from painful measurements) 
    /*  Columns 1 through 6 (all in cm)
    *
    * -18.00000000000000 -10.10000000000000  -8.60000000000000  -7.40000000000000  -6.65000000000000  -6.10000000000000
    *   0.00075000000000   0.50000000000000   0.75000000000000   1.00000000000000   1.25000000000000   1.45000000000000
    *
    *  Columns 7 through 10
    *
    *  -4.90000000000000  -3.30000000000000  -2.10000000000000                  0
    *   2.00000000000000   3.00000000000000   4.20000000000000   6.90000000000000
    */


    Point(57) = {-w_ridge/2.0, -0.0008, -0.17, lc_detail};
    Point(58) = {w_ridge/2.0, -0.0008, -0.17, lc_detail};
    Point(59) = {w_ridge/2.0, 0.0008, -0.17, lc_detail};
    Point(60) = {-w_ridge/2.0, 0.0008, -0.17, lc_detail};

    Point(61) = {-w_ridge/2.0, -0.005, -0.101, lc_detail};
    Point(62) = {w_ridge/2.0, -0.005, -0.101, lc_detail};
    Point(63) = {w_ridge/2.0, 0.005, -0.101, lc_detail};
    Point(64) = {-w_ridge/2.0, 0.005, -0.101, lc_detail};

    Point(65) = {-w_ridge/2.0, -0.0075, -0.086, lc_detail};
    Point(66) = {w_ridge/2.0, -0.0075, -0.086, lc_detail};
    Point(67) = {w_ridge/2.0, 0.0075, -0.086, lc_detail};
    Point(68) = {-w_ridge/2.0, 0.0075, -0.086, lc_detail};

    Point(69) = {-w_ridge/2.0, -0.01, -0.074, lc_detail};
    Point(70) = {w_ridge/2.0, -0.01, -0.074, lc_detail};
    Point(71) = {w_ridge/2.0, 0.01, -0.074, lc_detail};
    Point(72) = {-w_ridge/2.0, 0.01, -0.074, lc_detail};

    Point(73) = {-w_ridge/2.0, -0.0125, -0.0665, lc_detail};
    Point(74) = {w_ridge/2.0, -0.0125, -0.0665, lc_detail};
    Point(75) = {w_ridge/2.0, 0.0125, -0.0665, lc_detail};
    Point(76) = {-w_ridge/2.0, 0.0125, -0.0665, lc_detail};

    Point(77) = {-w_ridge/2.0, -0.0145, -0.061, lc_detail};
    Point(78) = {w_ridge/2.0, -0.0145, -0.061, lc_detail};
    Point(79) = {w_ridge/2.0, 0.0145, -0.061, lc_detail};
    Point(80) = {-w_ridge/2.0, 0.0145, -0.061, lc_detail};

    Point(81) = {-w_ridge/2.0, -0.02, -0.049, lc_detail};
    Point(82) = {w_ridge/2.0, -0.02, -0.049, lc_detail};
    Point(83) = {w_ridge/2.0, 0.02, -0.049, lc_detail};
    Point(84) = {-w_ridge/2.0, 0.02, -0.049, lc_detail};

    Point(85) = {-w_ridge/2.0, -0.03, -0.033, lc_detail};
    Point(86) = {w_ridge/2.0, -0.03, -0.033, lc_detail};
    Point(87) = {w_ridge/2.0, 0.03, -0.033, lc_detail};
    Point(88) = {-w_ridge/2.0, 0.03, -0.033, lc_detail};

    Point(89) = {-w_ridge/2.0, -0.042, -0.021, lc_detail};
    Point(90) = {w_ridge/2.0, -0.042, -0.021, lc_detail};
    Point(91) = {w_ridge/2.0, 0.042, -0.021, lc_detail};
    Point(92) = {-w_ridge/2.0, 0.042, -0.021, lc_detail};

    // lines

    // not classified



    Line(1) = {1,45};
    Line(2) = {45,46};
    Line(3) = {46,2};
    Line(4) = {2,3};
    Line(5) = {3,47};
    Line(6) = {47,48};
    Line(7) = {48,4};
    Line(8) = {4,1};
    Line(9) = {9,41};
    Line(10) = {41,42};
    Line(11) = {42,10};
    Line(12) = {6,7};
    Line(13) = {11,43};
    Line(14) = {43,44};
    Line(15) = {44,12};
    Line(16) = {8,5};
    Line(17) = {12,18};
    Line(18) = {18,17};
    Line(19) = {17,9};
    Line(20) = {11,20};
    Line(21) = {20,19};
    Line(22) = {19,10};
    Line(23) = {10,6};
    Line(24) = {7,11};
    Line(25) = {12,8};
    Line(26) = {5,9};
    Line(27) = {4,8};
    Line(28) = {3,7};
    Line(29) = {6,2};
    Line(30) = {1,5};
    Line(31) = {45,89};
    Line(32) = {89,85};
    Line(33) = {85,81};
    Line(34) = {81,77};
    Line(35) = {77,73};
    Line(36) = {73,69};
    Line(37) = {69,65};
    Line(38) = {65,61};
    Line(39) = {62,66};
    Line(40) = {66,70};
    Line(41) = {70,74};
    Line(42) = {74,78};
    Line(43) = {78,82};
    Line(44) = {82,86};
    Line(45) = {86,90};
    Line(46) = {90,46};
    Line(47) = {48,92};
    Line(48) = {92,88};
    Line(49) = {88,84};
    Line(50) = {84,80};
    Line(51) = {80,76};
    Line(52) = {76,72};
    Line(53) = {72,68};
    Line(54) = {68,64};
    Line(55) = {47,91};
    Line(56) = {91,87};
    Line(57) = {87,83};
    Line(58) = {83,79};
    Line(59) = {79,75};
    Line(60) = {75,71};
    Line(61) = {71,67};
    Line(62) = {67,63};
    Line(63) = {43,47};
    Line(64) = {48,44};
    Line(65) = {46,42};
    Line(66) = {45,41};
    Line(67) = {18,26};
    Line(68) = {26,12};
    Line(69) = {25,17};
    Line(70) = {25,9};
    Line(71) = {20,28};
    Line(72) = {28,11};
    Line(73) = {19,27};
    Line(74) = {27,10};
    Line(75) = {26,32};
    Line(76) = {32,52};
    Line(77) = {52,51};
    Line(78) = {51,31};
    Line(79) = {31,28};
    Line(80) = {25,29};
    Line(81) = {29,49};
    Line(82) = {49,50};
    Line(83) = {50,30};
    Line(84) = {30,27};
    Line(85) = {13,53};
    Line(86) = {53,54};
    Line(87) = {54,14};
    Line(88) = {14,15};
    Line(89) = {15,55};
    Line(90) = {55,56};
    Line(91) = {56,16};
    Line(92) = {16,13};
    Line(93) = {52,44};
    Line(94) = {43,51};
    Line(95) = {49,41};
    Line(96) = {42,50};
    Line(97) = {32,24};
    Line(98) = {24,18};
    Line(99) = {17,21};
    Line(100) = {21,29};
    Line(101) = {30,22};
    Line(102) = {22,19};
    Line(103) = {31,23};
    Line(104) = {23,20};
    Line(105) = {52,40};
    Line(106) = {40,36};
    Line(107) = {33,37};
    Line(108) = {37,49};
    Line(109) = {50,38};
    Line(110) = {38,34};
    Line(111) = {35,39};
    Line(112) = {39,51};
    Line(113) = {36,60};
    Line(114) = {33,57};
    Line(115) = {35,59};
    Line(116) = {34,58};
    Line(117) = {57,61};
    Line(118) = {58,62};
    Line(119) = {64,60};
    Line(120) = {59,63};
    Line(121) = {16,32};
    Line(122) = {13,29};
    Line(123) = {14,30};
    Line(124) = {15,31};
    Line(137) = {92,91};
    Line(138) = {88,87};
    Line(139) = {84,83};
    Line(140) = {80,79};
    Line(141) = {76,75};
    Line(142) = {72,71};
    Line(143) = {68,67};
    Line(144) = {64,63};
    Line(145) = {60,59};
    Line(146) = {58,57};
    Line(147) = {36,35};
    Line(148) = {34,33};
    Line(149) = {62,61};
    Line(150) = {66,65};
    Line(151) = {70,69};
    Line(152) = {74,73};
    Line(153) = {78,77};
    Line(154) = {82,81};
    Line(155) = {86,85};
    Line(156) = {90,89};
    Line(181) = {39,40};
    Line(182) = {38,37};
    Line(233) = {22,23};
    Line(234) = {21,24};


    Line Loop(125) = {11,23,29,-3,65};
    Plane Surface(126) = {125};
    Line Loop(127) = {10,-65,-2,66};
    //Plane Surface(128) = {127};
    Line Loop(129) = {66,-9,-26,-30,1};
    Plane Surface(130) = {129};
    Line Loop(131) = {28,24,13,63,-5};
    Plane Surface(132) = {131};
    Line Loop(133) = {14,-64,-6,-63};
    //Plane Surface(134) = {133};
    Line Loop(135) = {7,27,-25,-15,-64};
    Plane Surface(136) = {135};
    Line Loop(157) = {64,-93,105,106,113,-119,-54,-53,-52,-51,-50,-49,-48,-47};
    Plane Surface(158) = {157};
    Line Loop(159) = {63,55,56,57,58,59,60,61,62,-120,-115,111,112,-94};
    Plane Surface(160) = {159};
    Line Loop(161) = {55,-137,-47,-6};
    Plane Surface(162) = {161};
    Line Loop(163) = {48,138,-56,-137};
    Plane Surface(164) = {163};
    Line Loop(165) = {57,-139,-49,138};
    Plane Surface(166) = {165};
    Line Loop(167) = {58,-140,-50,139};
    Plane Surface(168) = {167};
    Line Loop(169) = {59,-141,-51,140};
    Plane Surface(170) = {169};
    Line Loop(171) = {142,-60,-141,52};
    Plane Surface(172) = {171};
    Line Loop(173) = {143,-61,-142,53};
    Plane Surface(174) = {173};
    Line Loop(175) = {144,-62,-143,54};
    Plane Surface(176) = {175};
    Line Loop(177) = {145,120,-144,119};
    Plane Surface(178) = {177};
    Line Loop(179) = {115,-145,-113,147};
    Plane Surface(180) = {179};
    Line Loop(183) = {147,111,181,106};
    Plane Surface(184) = {183};
    Line Loop(185) = {112,-77,105,-181};
    Plane Surface(186) = {185};
    Line Loop(187) = {77,-94,14,-93};
    //Plane Surface(188) = {187};
    Line Loop(189) = {2,-46,156,-31};
    Plane Surface(190) = {189};
    Line Loop(191) = {45,156,32,-155};
    Plane Surface(192) = {191};
    Line Loop(193) = {44,155,33,-154};
    Plane Surface(194) = {193};
    Line Loop(195) = {43,154,34,-153};
    Plane Surface(196) = {195};
    Line Loop(197) = {42,153,35,-152};
    Plane Surface(198) = {197};
    Line Loop(199) = {41,152,36,-151};
    Plane Surface(200) = {199};
    Line Loop(201) = {40,151,37,-150};
    Plane Surface(202) = {201};
    Line Loop(203) = {39,150,38,-149};
    Plane Surface(204) = {203};
    Line Loop(205) = {118,149,-117,-146};
    Plane Surface(206) = {205};
    Line Loop(207) = {148,114,-146,-116};
    Plane Surface(208) = {207};
    Line Loop(209) = {110,148,107,-182};
    Plane Surface(210) = {209};
    Line Loop(211) = {82,109,182,108};
    Plane Surface(212) = {211};
    Line Loop(213) = {96,-82,95,10};
    //Plane Surface(214) = {213};
    Line Loop(215) = {65,96,109,110,116,118,39,40,41,42,43,44,45,46};
    Plane Surface(216) = {215};
    Line Loop(217) = {95,-66,31,32,33,34,35,36,37,38,-117,-114,107,108};
    Plane Surface(218) = {217};
    Line Loop(219) = {27,16,-30,-8};
    Plane Surface(220) = {219};
    Line Loop(221) = {12,-28,-4,-29};
    Plane Surface(222) = {221};
    Line Loop(223) = {86,87,88,89,90,91,92,85};
    Plane Surface(224) = {223};
    Line Loop(225) = {72,20,71};
    Plane Surface(226) = {225};
    Line Loop(227) = {73,74,-22};
    Plane Surface(228) = {227};
    Line Loop(229) = {68,17,67};
    Plane Surface(230) = {229};
    Line Loop(231) = {69,19,-70};
    Plane Surface(232) = {231};
    Line Loop(235) = {234,98,18,99};
    Plane Surface(236) = {235};
    Line Loop(237) = {102,-21,-104,-233};
    Plane Surface(238) = {237};
    Line Loop(239) = {233,-103,-124,-88,123,101};
    Plane Surface(240) = {239};
    Line Loop(241) = {121,97,-234,100,-122,-92};
    Plane Surface(242) = {241};
    Line Loop(243) = {123,-83,-82,-81,-122,85,86,87};
    Plane Surface(244) = {243};
    Line Loop(245) = {76,77,78,-124,89,90,91,121};
    Plane Surface(246) = {245};
    Line Loop(247) = {99,100,-80,69};
    Plane Surface(248) = {247};
    Line Loop(249) = {98,67,75,97};
    Plane Surface(250) = {249};
    Line Loop(251) = {102,73,-84,101};
    Plane Surface(252) = {251};
    Line Loop(253) = {19,-26,-16,-25,17,18};
    Plane Surface(254) = {253};
    Line Loop(255) = {12,24,20,21,22,23};
    Plane Surface(256) = {255};
    Line Loop(257) = {80,81,95,-9,-70};
    Plane Surface(258) = {257};
    Line Loop(259) = {11,-74,-84,-83,-96};
    Plane Surface(260) = {259};
    Line Loop(261) = {93,15,-68,75,76};
    Plane Surface(262) = {261};
    Line Loop(263) = {78,79,72,13,94};
    Plane Surface(264) = {263};
    Line Loop(265) = {104,71,-79,103};
    Plane Surface(266) = {265};
    """

    geometry = "lc_coarse = " + str(h) + ";\n" +  "lc_detail = " + str(0.5*h) + ";\n" + stub
    return __generate_grid_from_geo_string(geometry)


def emcc_almond(h=0.1):
    stub = """
    // parameters
    d = 9.936 * 0.0254;

    /************
    * Functions
    ************/
    Function QuarterEllipse1
    x = d*t * t_fact;
    y = 4.83345 * d * ( Sqrt(1 - (t*t_fact/2.08335)^2) - 0.96);
    z = 1.61115 * d * ( Sqrt(1 - (t*t_fact/2.08335)^2) - 0.96);
    Point(point_number) = {x, 0, 0, lc*lc_fact}; // ellipse center
    CenterNumber = point_number;
    Psi_array[] = {0.0, 15.0, 45.0, 90.0};
    For i In {0:3}
        point_number = newp;
        thePointNumber[i] = point_number;
        psi = Psi_array[i]/180.0*Pi ;
        Point(point_number) = {x, y*Cos(psi), z*Sin(psi), lc*lc_fact}; point_number = newp;
    EndFor
    For i In {0:2}
        Ellipse(newreg) = {thePointNumber[i],CenterNumber,thePointNumber[3],thePointNumber[i+1]};
    EndFor
    Return

    Function QuarterEllipse2
    x = d*t * t_fact;
    y = yfact * d * ( Sqrt(1 - (t*t_fact*3.0/1.25)^2) );
    z = zfact * d * ( Sqrt(1 - (t*t_fact*3.0/1.25)^2) );
    Point(point_number) = {x, 0, 0, lc*lc_fact}; // ellipse center
    CenterNumber = point_number;
    Psi_array[] = {0.0, 15.0, 45.0, 90.0};
    For i In {0:3}
        point_number = newp;
        thePointNumber[i] = point_number;
        psi = Psi_array[i]/180.0*Pi ;
        Point(point_number) = {x, y*Cos(psi), z*Sin(psi), lc*lc_fact}; point_number = newp;
    EndFor
    For i In {0:2}
        Ellipse(newreg) = {thePointNumber[i],CenterNumber,thePointNumber[3],thePointNumber[i+1]};
    EndFor
    Return

    /************
    * Geometry
    ************/
    // first part of the almond
    t = 1.75/3.0;

    // tip of the almond
    Point(1) = {d*t, 0.0, 0.0, lc/2.0};
    point_number = 2;

    // 0.975
    lc_fact = 0.5;
    t_fact = 0.975;
    Call QuarterEllipse1 ;

    // 0.95
    lc_fact = 0.75;
    t_fact = 0.95;
    Call QuarterEllipse1 ;

    lc_fact = 1.0;
    t_fact = 0.9;
    For j In {1:10}
    Call QuarterEllipse1 ;
    t_fact -= 0.1;
    EndFor


    // second part of the almond

    t = -1.25/3.0;
    yfact = 0.58/3.0;
    zfact = 0.58/9.0;

    lc_fact = 1.0;
    t_fact = 0.1;
    For j In {1:9}
    Call QuarterEllipse2 ;
    t_fact += 0.1;
    EndFor


    // 0.95
    lc_fact = 1.0;
    t_fact = 0.95;
    Call QuarterEllipse2 ;

    // 0.99
    lc_fact = .75;
    t_fact = 0.99;
    Call QuarterEllipse2 ;

    // tip of the almond
    Point(point_number) = {d*t, 0.0, 0.0, lc/2.0};

    // curves

    CatmullRom(70) = {1,6,11};
    CatmullRom(71) = {1,5,10};
    CatmullRom(72) = {1,3,8};
    CatmullRom(73) = {11,16,21};
    CatmullRom(74) = {10,15,20};
    CatmullRom(75) = {9,14,19};
    CatmullRom(76) = {8,13,18};
    CatmullRom(77) = {21,26,31,36};
    CatmullRom(78) = {20,25,30,35};
    CatmullRom(79) = {19,24,29,34};
    CatmullRom(80) = {18,23,28,33};
    CatmullRom(81) = {36,41,46,51};
    CatmullRom(82) = {35,40,45,50};
    CatmullRom(83) = {34,39,44,49};
    CatmullRom(84) = {33,38,43,48};
    CatmullRom(85) = {51,56,61,66};
    CatmullRom(86) = {50,55,60,65};
    CatmullRom(87) = {49,54,59,64};
    CatmullRom(88) = {48,53,58,63};
    CatmullRom(89) = {66,71,76,81};
    CatmullRom(90) = {65,70,75,80};
    CatmullRom(91) = {64,69,74,79};
    CatmullRom(92) = {63,68,73,78};
    CatmullRom(93) = {81,86,91,96};
    CatmullRom(94) = {80,85,90,95};
    CatmullRom(95) = {79,84,89,94};
    CatmullRom(96) = {78,83,88,93};
    CatmullRom(97) = {93,98,103,108};
    CatmullRom(98) = {94,99,104,109};
    CatmullRom(99) = {95,100,105,110};
    CatmullRom(100) = {96,101,106,111};
    CatmullRom(101) = {108,113,117};
    CatmullRom(102) = {110,115,117};
    CatmullRom(103) = {111,116,117};

    // surfaces
    Line Loop(104) = {70,-6,-71};
    Ruled Surface(105) = {104};
    Line Loop(106) = {71,-5,-4,-72};
    Ruled Surface(107) = {106};
    Line Loop(108) = {6,73,-12,-74};
    Ruled Surface(109) = {108};
    Line Loop(110) = {5,74,-11,-75};
    Ruled Surface(111) = {110};
    Line Loop(112) = {75,-10,-76,4};
    Ruled Surface(113) = {112};
    Line Loop(114) = {12,77,-21,-78};
    Ruled Surface(115) = {114};
    Line Loop(116) = {11,78,-20,-79};
    Ruled Surface(117) = {116};
    Line Loop(118) = {10,79,-19,-80};
    Ruled Surface(119) = {118};
    Line Loop(120) = {21,81,-30,-82};
    Ruled Surface(121) = {120};
    Line Loop(122) = {20,82,-29,-83};
    Ruled Surface(123) = {122};
    Line Loop(124) = {19,83,-28,-84};
    Ruled Surface(125) = {124};
    Line Loop(126) = {30,85,-39,-86};
    Ruled Surface(127) = {126};
    Line Loop(128) = {29,86,-38,-87};
    Ruled Surface(129) = {128};
    Line Loop(130) = {28,87,-37,-88};
    Ruled Surface(131) = {130};
    Line Loop(132) = {39,89,-48,-90};
    Ruled Surface(133) = {132};
    Line Loop(134) = {38,90,-47,-91};
    Ruled Surface(135) = {134};
    Line Loop(136) = {37,91,-46,-92};
    Ruled Surface(137) = {136};
    Line Loop(138) = {48,93,-57,-94};
    Ruled Surface(139) = {138};
    Line Loop(140) = {47,94,-56,-95};
    Ruled Surface(141) = {140};
    Line Loop(142) = {46,95,-55,-96};
    Ruled Surface(143) = {142};
    Line Loop(144) = {57,100,-66,-99};
    Ruled Surface(145) = {144};
    Line Loop(146) = {56,99,-65,-98};
    Ruled Surface(147) = {146};
    Line Loop(148) = {55,98,-64,-97};
    Ruled Surface(149) = {148};
    Line Loop(150) = {101,-102,-65,-64};
    Ruled Surface(151) = {150};
    Line Loop(152) = {103,-102,66};
    Ruled Surface(153) = {152};
    Symmetry {0,1,0,0} {
    Duplicata { Surface{105,107,111,109,113,119,117,115,121,123,125,127,129,131,133,135,137,139,141,143,145,147,149,153,151}; }
    }
    Symmetry {0,0,1,0} {
    Duplicata { Surface{154,158,105,107,111,113,109,168,173,163,119,117,115,188,183,178,203,198,193,121,123,125,213,208,127,129,131,218,233,228,223,133,135,137,143,141,139,238,243,248,263,258,253,145,147,149,272,268,153,151}; }
    }

    """

    geometry = "lc = " + str(h) + ";\n" + stub
    return __generate_grid_from_geo_string(geometry)


def frigate_hull(h=0.1):
    stub = """
    Point (1) = {1.731873000000633/1000.0, 1.13686837721616e-13/1000.0, 3447.047/1000.0, lc};
    Point (2) = {132.5606999999982/1000.0, -5411.502/1000.0, 8089.289999999999/1000.0, lc};
    Point (3) = {100047/1000.0, -5.684341886080801e-14/1000.0, 494.4196000000001/1000.0, lc};
    Point (4) = {103631.3/1000.0, -434.0860999999992/1000.0, 8101.888999999998/1000.0, lc};
    Point (5) = {1.731873000000633/1000.0, -1.13686837721616e-13/1000.0, 3447.047/1000.0, lc};
    Point (6) = {132.5606999999982/1000.0, 5411.502/1000.0, 8089.289999999999/1000.0, lc};
    Point (7) = {100047/1000.0, 5.684341886080801e-14/1000.0, 494.4196000000001/1000.0, lc};
    Point (8) = {103631.3/1000.0, 434.0860999999992/1000.0, 8101.888999999998/1000.0, lc};
    Point (9) = {100047/1000.0, -3.33066907387547e-15/1000.0, 494.4196000000001/1000.0, lc};
    Point (10) = {101764.4/1000.0, -3.33066907387547e-15/1000.0, 1397.42/1000.0, lc};
    Point (11) = {103631.3/1000.0, -434.0861/1000.0, 8101.889/1000.0, lc};
    Point (12) = {104199.8/1000.0, -2.842170943040401e-14/1000.0, 8101.889/1000.0, lc};
    Point (13) = {100047/1000.0, 3.33066907387547e-15/1000.0, 494.4196000000001/1000.0, lc};
    Point (14) = {101764.4/1000.0, 3.33066907387547e-15/1000.0, 1397.42/1000.0, lc};
    Point (15) = {103631.3/1000.0, 434.0861/1000.0, 8101.889/1000.0, lc};
    Point (16) = {104199.8/1000.0, 2.842170943040401e-14/1000.0, 8101.889/1000.0, lc};
    Point (17) = {1.731873000000632/1000.0, 0/1000.0, 3447.047/1000.0, lc};
    Point (18) = {100047/1000.0, -1.4210854715202e-14/1000.0, 494.4196/1000.0, lc};
    Point (19) = {99885.95999999999/1000.0, 0/1000.0, 512.831/1000.0, lc};
    Point (20) = {1.731873000000632/1000.0, 0/1000.0, 3447.047/1000.0, lc};
    Point (21) = {100047/1000.0, 1.4210854715202e-14/1000.0, 494.4196/1000.0, lc};
    Point (22) = {99885.95999999999/1000.0, 0/1000.0, 512.831/1000.0, lc};
    p1 = newp;
    Point (p1 + 1) = {1.63165963139318/1000.0, -283.5765116854973/1000.0, 3458.662366996587/1000.0, lc};
    Point (p1 + 2) = {1.520003921084145/1000.0, -498.8855410120032/1000.0, 3468.523498648208/1000.0, lc};
    Point (p1 + 3) = {1.403968924261698/1000.0, -685.4969471385097/1000.0, 3476.314759706816/1000.0, lc};
    Point (p1 + 4) = {1.287891762934856/1000.0, -859.3644679807019/1000.0, 3483.071511727658/1000.0, lc};
    Point (p1 + 5) = {1.17288998408631/1000.0, -1030.994442268699/1000.0, 3490.007033490641/1000.0, lc};
    Point (p1 + 6) = {1.056340423352394/1000.0, -1212.692660408892/1000.0, 3498.179834881983/1000.0, lc};
    Point (p1 + 7) = {0.9313381893206537/1000.0, -1417.767063315607/1000.0, 3508.779911577/1000.0, lc};
    Point (p1 + 8) = {0.7865311329136876/1000.0, -1658.947234739162/1000.0, 3523.450932483341/1000.0, lc};
    Point (p1 + 9) = {0.6099130946790331/1000.0, -1946.632084315447/1000.0, 3544.370426778933/1000.0, lc};
    Point (p1 + 10) = {0.3979921585482493/1000.0, -2286.905280579152/1000.0, 3574.001317834429/1000.0, lc};
    Point (p1 + 11) = {0.1660035444154647/1000.0, -2679.521020949075/1000.0, 3614.810303217294/1000.0, lc};
    Point (p1 + 12) = {-0.04297697207950973/1000.0, -3116.339660140416/1000.0, 3669.852186837182/1000.0, lc};
    Point (p1 + 13) = {-0.151740724629414/1000.0, -3580.906588158266/1000.0, 3745.554238696463/1000.0, lc};
    Point (p1 + 14) = {-0.04230901767535866/1000.0, -4048.490695776497/1000.0, 3855.384285154983/1000.0, lc};
    Point (p1 + 15) = {0.5841260382148572/1000.0, -4486.984908850851/1000.0, 4025.115149480531/1000.0, lc};
    Point (p1 + 16) = {3.006986207382129/1000.0, -4861.982364388634/1000.0, 4305.815190395672/1000.0, lc};
    Point (p1 + 17) = {10.77168056217393/1000.0, -5144.927919561177/1000.0, 4792.515104938174/1000.0, lc};
    Point (p1 + 18) = {30.70589292923169/1000.0, -5321.465931739946/1000.0, 5629.030264408704/1000.0, lc};
    Point (p1 + 19) = {70.62752560640348/1000.0, -5398.771619974294/1000.0, 6861.87611699337/1000.0, lc};
    CatmullRom (1) = {1, p1 + 1, p1 + 2, p1 + 3, p1 + 4, p1 + 5, p1 + 6, p1 + 7, p1 + 8, p1 + 9, p1 + 10, p1 + 11, p1 + 12, p1 + 13, p1 + 14, p1 + 15, p1 + 16, p1 + 17, p1 + 18, p1 + 19, 2};
    p2 = newp;
    Point (p2 + 1) = {9317.554799738758/1000.0, -139.3694885613147/1000.0, 2942.857971742675/1000.0, lc};
    Point (p2 + 2) = {14724.64862612/1000.0, -210.18325075198/1000.0, 2113.413688138912/1000.0, lc};
    Point (p2 + 3) = {19611.19286465473/1000.0, -246.6193145858631/1000.0, 1125.284989972894/1000.0, lc};
    Point (p2 + 4) = {23249.02097322105/1000.0, -278.5639893578149/1000.0, 339.0656431875292/1000.0, lc};
    Point (p2 + 5) = {25371.65146785838/1000.0, -292.0802043446009/1000.0, -58.01804876601409/1000.0, lc};
    Point (p2 + 6) = {26666.10758849813/1000.0, -290.6949410912934/1000.0, -193.201108439527/1000.0, lc};
    Point (p2 + 7) = {27882.57724661941/1000.0, -281.8773829745387/1000.0, -232.9839480135292/1000.0, lc};
    Point (p2 + 8) = {29606.24744292548/1000.0, -270.129717707284/1000.0, -239.1902570770887/1000.0, lc};
    Point (p2 + 9) = {32696.47337998234/1000.0, -265.6575382761905/1000.0, -217.2925098303248/1000.0, lc};
    Point (p2 + 10) = {37975.08112348829/1000.0, -275.4882892124488/1000.0, -160.2421458246485/1000.0, lc};
    Point (p2 + 11) = {45455.35188076008/1000.0, -290.4158848525699/1000.0, -70.55141422325377/1000.0, lc};
    Point (p2 + 12) = {53898.6780871717/1000.0, -298.3127464947366/1000.0, 26.82317735520494/1000.0, lc};
    Point (p2 + 13) = {61374.29196374536/1000.0, -299.2965718131835/1000.0, 94.45420596456063/1000.0, lc};
    Point (p2 + 14) = {67450.02876813264/1000.0, -298.0797935904579/1000.0, 130.4626228617591/1000.0, lc};
    Point (p2 + 15) = {73712.18212436585/1000.0, -295.6811939986992/1000.0, 178.0523068123641/1000.0, lc};
    Point (p2 + 16) = {81126.28015937647/1000.0, -290.7568407304351/1000.0, 262.6520637541346/1000.0, lc};
    Point (p2 + 17) = {88730.75178590092/1000.0, -273.5900336715672/1000.0, 364.5881605037617/1000.0, lc};
    Point (p2 + 18) = {94710.76120887922/1000.0, -216.3209508941758/1000.0, 448.1572682599385/1000.0, lc};
    Point (p2 + 19) = {98242.99731098629/1000.0, -99.21419481730666/1000.0, 490.0647216050518/1000.0, lc};
    CatmullRom (2) = {1, p2 + 1, p2 + 2, p2 + 3, p2 + 4, p2 + 5, p2 + 6, p2 + 7, p2 + 8, p2 + 9, p2 + 10, p2 + 11, p2 + 12, p2 + 13, p2 + 14, p2 + 15, p2 + 16, p2 + 17, p2 + 18, p2 + 19, 3};
    p3 = newp;
    Point (p3 + 1) = {100399.8267466054/1000.0, -37.64328850050601/1000.0, 983.1517345866399/1000.0, lc};
    Point (p3 + 2) = {100544.0340097644/1000.0, -56.30285882427393/1000.0, 1190.72213194248/1000.0, lc};
    Point (p3 + 3) = {100625.7701109969/1000.0, -68.2070611051324/1000.0, 1329.88586165233/1000.0, lc};
    Point (p3 + 4) = {100693.6069984745/1000.0, -77.83843427273396/1000.0, 1464.231181343476/1000.0, lc};
    Point (p3 + 5) = {100759.0197231369/1000.0, -86.49249278444159/1000.0, 1603.991484609769/1000.0, lc};
    Point (p3 + 6) = {100825.70614491/1000.0, -94.60964911899997/1000.0, 1751.956898573364/1000.0, lc};
    Point (p3 + 7) = {100895.9566714799/1000.0, -102.4669806582255/1000.0, 1912.632956115628/1000.0, lc};
    Point (p3 + 8) = {100974.5414101193/1000.0, -110.6791909967124/1000.0, 2097.381740563592/1000.0, lc};
    Point (p3 + 9) = {101069.5045533751/1000.0, -120.3381459398932/1000.0, 2324.171450127307/1000.0, lc};
    Point (p3 + 10) = {101188.76250063/1000.0, -132.6618952815267/1000.0, 2610.010253999559/1000.0, lc};
    Point (p3 + 11) = {101335.9311808066/1000.0, -148.5497581743961/1000.0, 2962.067516012177/1000.0, lc};
    Point (p3 + 12) = {101508.1752328875/1000.0, -168.2740303823938/1000.0, 3373.13025279561/1000.0, lc};
    Point (p3 + 13) = {101699.2131058543/1000.0, -191.5212492107784/1000.0, 3828.105406941327/1000.0, lc};
    Point (p3 + 14) = {101904.3406224582/1000.0, -217.5712363905989/1000.0, 4314.845869380807/1000.0, lc};
    Point (p3 + 15) = {102124.3726632129/1000.0, -245.4881661908508/1000.0, 4832.611167649558/1000.0, lc};
    Point (p3 + 16) = {102363.8508919292/1000.0, -274.3448949415413/1000.0, 5388.008869205063/1000.0, lc};
    Point (p3 + 17) = {102625.0307187294/1000.0, -303.4721798503643/1000.0, 5981.720214898422/1000.0, lc};
    Point (p3 + 18) = {102903.8999389907/1000.0, -333.1090041066523/1000.0, 6599.32203120271/1000.0, lc};
    Point (p3 + 19) = {103211.730739768/1000.0, -369.2356817058429/1000.0, 7254.295030661341/1000.0, lc};
    CatmullRom (3) = {3, p3 + 1, p3 + 2, p3 + 3, p3 + 4, p3 + 5, p3 + 6, p3 + 7, p3 + 8, p3 + 9, p3 + 10, p3 + 11, p3 + 12, p3 + 13, p3 + 14, p3 + 15, p3 + 16, p3 + 17, p3 + 18, p3 + 19, 4};
    p4 = newp;
    Point (p4 + 1) = {8943.212265505897/1000.0, -5688.370970275137/1000.0, 8105.02317030835/1000.0, lc};
    Point (p4 + 2) = {14563.05619667714/1000.0, -5882.347131614337/1000.0, 8112.514303265203/1000.0, lc};
    Point (p4 + 3) = {19629.41958677865/1000.0, -5989.789597332175/1000.0, 8111.703928320967/1000.0, lc};
    Point (p4 + 4) = {23174.32236964143/1000.0, -6042.643021751901/1000.0, 8105.962909744474/1000.0, lc};
    Point (p4 + 5) = {25169.55268656656/1000.0, -6068.383035845703/1000.0, 8102.637802859644/1000.0, lc};
    Point (p4 + 6) = {26376.63353139325/1000.0, -6086.644889715932/1000.0, 8101.930333060809/1000.0, lc};
    Point (p4 + 7) = {27493.25866503828/1000.0, -6106.966546645771/1000.0, 8101.889018675328/1000.0, lc};
    Point (p4 + 8) = {29062.12248276674/1000.0, -6128.989242662807/1000.0, 8101.889/1000.0, lc};
    Point (p4 + 9) = {31992.07131170032/1000.0, -6145.642305141368/1000.0, 8101.889/1000.0, lc};
    Point (p4 + 10) = {37214.79084566674/1000.0, -6149.706230951064/1000.0, 8101.888880208691/1000.0, lc};
    Point (p4 + 11) = {44816.90092560398/1000.0, -6139.688435629507/1000.0, 8101.884107596338/1000.0, lc};
    Point (p4 + 12) = {53520.63232570445/1000.0, -6116.737385091436/1000.0, 8101.85059305681/1000.0, lc};
    Point (p4 + 13) = {61285.08216637133/1000.0, -6063.203958205218/1000.0, 8101.759152955191/1000.0, lc};
    Point (p4 + 14) = {67730.45268171342/1000.0, -5896.320030635935/1000.0, 8101.65586164279/1000.0, lc};
    Point (p4 + 15) = {74504.92464347548/1000.0, -5485.861324088314/1000.0, 8101.649963258591/1000.0, lc};
    Point (p4 + 16) = {81986.4784891601/1000.0, -4772.275816407996/1000.0, 8101.74865711518/1000.0, lc};
    Point (p4 + 17) = {88682.29454388955/1000.0, -3816.95907481118/1000.0, 8101.844787749058/1000.0, lc};
    Point (p4 + 18) = {94048.0276068235/1000.0, -2738.192504547267/1000.0, 8101.88282661142/1000.0, lc};
    Point (p4 + 19) = {99069.2044944838/1000.0, -1570.252637244806/1000.0, 8101.888807074421/1000.0, lc};
    CatmullRom (4) = {2, p4 + 1, p4 + 2, p4 + 3, p4 + 4, p4 + 5, p4 + 6, p4 + 7, p4 + 8, p4 + 9, p4 + 10, p4 + 11, p4 + 12, p4 + 13, p4 + 14, p4 + 15, p4 + 16, p4 + 17, p4 + 18, p4 + 19, 4};
    p5 = newp;
    Point (p5 + 1) = {1.63165963139318/1000.0, 283.5765116854973/1000.0, 3458.662366996587/1000.0, lc};
    Point (p5 + 2) = {1.520003921084145/1000.0, 498.8855410120032/1000.0, 3468.523498648208/1000.0, lc};
    Point (p5 + 3) = {1.403968924261698/1000.0, 685.4969471385097/1000.0, 3476.314759706816/1000.0, lc};
    Point (p5 + 4) = {1.287891762934856/1000.0, 859.3644679807019/1000.0, 3483.071511727658/1000.0, lc};
    Point (p5 + 5) = {1.17288998408631/1000.0, 1030.994442268699/1000.0, 3490.007033490641/1000.0, lc};
    Point (p5 + 6) = {1.056340423352394/1000.0, 1212.692660408892/1000.0, 3498.179834881983/1000.0, lc};
    Point (p5 + 7) = {0.9313381893206537/1000.0, 1417.767063315607/1000.0, 3508.779911577/1000.0, lc};
    Point (p5 + 8) = {0.7865311329136876/1000.0, 1658.947234739162/1000.0, 3523.450932483341/1000.0, lc};
    Point (p5 + 9) = {0.6099130946790331/1000.0, 1946.632084315447/1000.0, 3544.370426778933/1000.0, lc};
    Point (p5 + 10) = {0.3979921585482493/1000.0, 2286.905280579152/1000.0, 3574.001317834429/1000.0, lc};
    Point (p5 + 11) = {0.1660035444154647/1000.0, 2679.521020949075/1000.0, 3614.810303217294/1000.0, lc};
    Point (p5 + 12) = {-0.04297697207950973/1000.0, 3116.339660140416/1000.0, 3669.852186837182/1000.0, lc};
    Point (p5 + 13) = {-0.151740724629414/1000.0, 3580.906588158266/1000.0, 3745.554238696463/1000.0, lc};
    Point (p5 + 14) = {-0.04230901767535866/1000.0, 4048.490695776497/1000.0, 3855.384285154983/1000.0, lc};
    Point (p5 + 15) = {0.5841260382148572/1000.0, 4486.984908850851/1000.0, 4025.115149480531/1000.0, lc};
    Point (p5 + 16) = {3.006986207382129/1000.0, 4861.982364388634/1000.0, 4305.815190395672/1000.0, lc};
    Point (p5 + 17) = {10.77168056217393/1000.0, 5144.927919561177/1000.0, 4792.515104938174/1000.0, lc};
    Point (p5 + 18) = {30.70589292923169/1000.0, 5321.465931739946/1000.0, 5629.030264408704/1000.0, lc};
    Point (p5 + 19) = {70.62752560640348/1000.0, 5398.771619974294/1000.0, 6861.87611699337/1000.0, lc};
    CatmullRom (5) = {5, p5 + 1, p5 + 2, p5 + 3, p5 + 4, p5 + 5, p5 + 6, p5 + 7, p5 + 8, p5 + 9, p5 + 10, p5 + 11, p5 + 12, p5 + 13, p5 + 14, p5 + 15, p5 + 16, p5 + 17, p5 + 18, p5 + 19, 6};
    p6 = newp;
    Point (p6 + 1) = {9317.554799738758/1000.0, 139.3694885613147/1000.0, 2942.857971742675/1000.0, lc};
    Point (p6 + 2) = {14724.64862612/1000.0, 210.18325075198/1000.0, 2113.413688138912/1000.0, lc};
    Point (p6 + 3) = {19611.19286465473/1000.0, 246.6193145858631/1000.0, 1125.284989972894/1000.0, lc};
    Point (p6 + 4) = {23249.02097322105/1000.0, 278.5639893578149/1000.0, 339.0656431875292/1000.0, lc};
    Point (p6 + 5) = {25371.65146785838/1000.0, 292.0802043446009/1000.0, -58.01804876601409/1000.0, lc};
    Point (p6 + 6) = {26666.10758849813/1000.0, 290.6949410912934/1000.0, -193.201108439527/1000.0, lc};
    Point (p6 + 7) = {27882.57724661941/1000.0, 281.8773829745387/1000.0, -232.9839480135292/1000.0, lc};
    Point (p6 + 8) = {29606.24744292548/1000.0, 270.129717707284/1000.0, -239.1902570770887/1000.0, lc};
    Point (p6 + 9) = {32696.47337998234/1000.0, 265.6575382761905/1000.0, -217.2925098303248/1000.0, lc};
    Point (p6 + 10) = {37975.08112348829/1000.0, 275.4882892124488/1000.0, -160.2421458246485/1000.0, lc};
    Point (p6 + 11) = {45455.35188076008/1000.0, 290.4158848525699/1000.0, -70.55141422325377/1000.0, lc};
    Point (p6 + 12) = {53898.6780871717/1000.0, 298.3127464947366/1000.0, 26.82317735520494/1000.0, lc};
    Point (p6 + 13) = {61374.29196374536/1000.0, 299.2965718131835/1000.0, 94.45420596456063/1000.0, lc};
    Point (p6 + 14) = {67450.02876813264/1000.0, 298.0797935904579/1000.0, 130.4626228617591/1000.0, lc};
    Point (p6 + 15) = {73712.18212436585/1000.0, 295.6811939986992/1000.0, 178.0523068123641/1000.0, lc};
    Point (p6 + 16) = {81126.28015937647/1000.0, 290.7568407304351/1000.0, 262.6520637541346/1000.0, lc};
    Point (p6 + 17) = {88730.75178590092/1000.0, 273.5900336715672/1000.0, 364.5881605037617/1000.0, lc};
    Point (p6 + 18) = {94710.76120887922/1000.0, 216.3209508941758/1000.0, 448.1572682599385/1000.0, lc};
    Point (p6 + 19) = {98242.99731098629/1000.0, 99.21419481730666/1000.0, 490.0647216050518/1000.0, lc};
    CatmullRom (6) = {5, p6 + 1, p6 + 2, p6 + 3, p6 + 4, p6 + 5, p6 + 6, p6 + 7, p6 + 8, p6 + 9, p6 + 10, p6 + 11, p6 + 12, p6 + 13, p6 + 14, p6 + 15, p6 + 16, p6 + 17, p6 + 18, p6 + 19, 7};
    p7 = newp;
    Point (p7 + 1) = {100399.8267466054/1000.0, 37.64328850050601/1000.0, 983.1517345866399/1000.0, lc};
    Point (p7 + 2) = {100544.0340097644/1000.0, 56.30285882427393/1000.0, 1190.72213194248/1000.0, lc};
    Point (p7 + 3) = {100625.7701109969/1000.0, 68.2070611051324/1000.0, 1329.88586165233/1000.0, lc};
    Point (p7 + 4) = {100693.6069984745/1000.0, 77.83843427273396/1000.0, 1464.231181343476/1000.0, lc};
    Point (p7 + 5) = {100759.0197231369/1000.0, 86.49249278444159/1000.0, 1603.991484609769/1000.0, lc};
    Point (p7 + 6) = {100825.70614491/1000.0, 94.60964911899997/1000.0, 1751.956898573364/1000.0, lc};
    Point (p7 + 7) = {100895.9566714799/1000.0, 102.4669806582255/1000.0, 1912.632956115628/1000.0, lc};
    Point (p7 + 8) = {100974.5414101193/1000.0, 110.6791909967124/1000.0, 2097.381740563592/1000.0, lc};
    Point (p7 + 9) = {101069.5045533751/1000.0, 120.3381459398932/1000.0, 2324.171450127307/1000.0, lc};
    Point (p7 + 10) = {101188.76250063/1000.0, 132.6618952815267/1000.0, 2610.010253999559/1000.0, lc};
    Point (p7 + 11) = {101335.9311808066/1000.0, 148.5497581743961/1000.0, 2962.067516012177/1000.0, lc};
    Point (p7 + 12) = {101508.1752328875/1000.0, 168.2740303823938/1000.0, 3373.13025279561/1000.0, lc};
    Point (p7 + 13) = {101699.2131058543/1000.0, 191.5212492107784/1000.0, 3828.105406941327/1000.0, lc};
    Point (p7 + 14) = {101904.3406224582/1000.0, 217.5712363905989/1000.0, 4314.845869380807/1000.0, lc};
    Point (p7 + 15) = {102124.3726632129/1000.0, 245.4881661908508/1000.0, 4832.611167649558/1000.0, lc};
    Point (p7 + 16) = {102363.8508919292/1000.0, 274.3448949415413/1000.0, 5388.008869205063/1000.0, lc};
    Point (p7 + 17) = {102625.0307187294/1000.0, 303.4721798503643/1000.0, 5981.720214898422/1000.0, lc};
    Point (p7 + 18) = {102903.8999389907/1000.0, 333.1090041066523/1000.0, 6599.32203120271/1000.0, lc};
    Point (p7 + 19) = {103211.730739768/1000.0, 369.2356817058429/1000.0, 7254.295030661341/1000.0, lc};
    CatmullRom (7) = {7, p7 + 1, p7 + 2, p7 + 3, p7 + 4, p7 + 5, p7 + 6, p7 + 7, p7 + 8, p7 + 9, p7 + 10, p7 + 11, p7 + 12, p7 + 13, p7 + 14, p7 + 15, p7 + 16, p7 + 17, p7 + 18, p7 + 19, 8};
    p8 = newp;
    Point (p8 + 1) = {8943.212265505897/1000.0, 5688.370970275137/1000.0, 8105.02317030835/1000.0, lc};
    Point (p8 + 2) = {14563.05619667714/1000.0, 5882.347131614337/1000.0, 8112.514303265203/1000.0, lc};
    Point (p8 + 3) = {19629.41958677865/1000.0, 5989.789597332175/1000.0, 8111.703928320967/1000.0, lc};
    Point (p8 + 4) = {23174.32236964143/1000.0, 6042.643021751901/1000.0, 8105.962909744474/1000.0, lc};
    Point (p8 + 5) = {25169.55268656656/1000.0, 6068.383035845703/1000.0, 8102.637802859644/1000.0, lc};
    Point (p8 + 6) = {26376.63353139325/1000.0, 6086.644889715932/1000.0, 8101.930333060809/1000.0, lc};
    Point (p8 + 7) = {27493.25866503828/1000.0, 6106.966546645771/1000.0, 8101.889018675328/1000.0, lc};
    Point (p8 + 8) = {29062.12248276674/1000.0, 6128.989242662807/1000.0, 8101.889/1000.0, lc};
    Point (p8 + 9) = {31992.07131170032/1000.0, 6145.642305141368/1000.0, 8101.889/1000.0, lc};
    Point (p8 + 10) = {37214.79084566674/1000.0, 6149.706230951064/1000.0, 8101.888880208691/1000.0, lc};
    Point (p8 + 11) = {44816.90092560398/1000.0, 6139.688435629507/1000.0, 8101.884107596338/1000.0, lc};
    Point (p8 + 12) = {53520.63232570445/1000.0, 6116.737385091436/1000.0, 8101.85059305681/1000.0, lc};
    Point (p8 + 13) = {61285.08216637133/1000.0, 6063.203958205218/1000.0, 8101.759152955191/1000.0, lc};
    Point (p8 + 14) = {67730.45268171342/1000.0, 5896.320030635935/1000.0, 8101.65586164279/1000.0, lc};
    Point (p8 + 15) = {74504.92464347548/1000.0, 5485.861324088314/1000.0, 8101.649963258591/1000.0, lc};
    Point (p8 + 16) = {81986.4784891601/1000.0, 4772.275816407996/1000.0, 8101.74865711518/1000.0, lc};
    Point (p8 + 17) = {88682.29454388955/1000.0, 3816.95907481118/1000.0, 8101.844787749058/1000.0, lc};
    Point (p8 + 18) = {94048.0276068235/1000.0, 2738.192504547267/1000.0, 8101.88282661142/1000.0, lc};
    Point (p8 + 19) = {99069.2044944838/1000.0, 1570.252637244806/1000.0, 8101.888807074421/1000.0, lc};
    CatmullRom (8) = {6, p8 + 1, p8 + 2, p8 + 3, p8 + 4, p8 + 5, p8 + 6, p8 + 7, p8 + 8, p8 + 9, p8 + 10, p8 + 11, p8 + 12, p8 + 13, p8 + 14, p8 + 15, p8 + 16, p8 + 17, p8 + 18, p8 + 19, 8};
    p9 = newp;
    Point (p9 + 1) = {100167.6471505963/1000.0, 0/1000.0, 507.6358599504135/1000.0, lc};
    Point (p9 + 2) = {100285.0763422741/1000.0, 0/1000.0, 524.2763149901309/1000.0, lc};
    Point (p9 + 3) = {100399.1960974999/1000.0, 0/1000.0, 544.3488410104093/1000.0, lc};
    Point (p9 + 4) = {100509.9185091162/1000.0, 0/1000.0, 567.8570241378407/1000.0, lc};
    Point (p9 + 5) = {100617.1594542225/1000.0, 0/1000.0, 594.8001191965965/1000.0, lc};
    Point (p9 + 6) = {100720.8387979568/1000.0, 0/1000.0, 625.1730215329168/1000.0, lc};
    Point (p9 + 7) = {100820.8805861771/1000.0, 0/1000.0, 658.9662524555755/1000.0, lc};
    Point (p9 + 8) = {100917.2132260961/1000.0, 0/1000.0, 696.1659584679518/1000.0, lc};
    Point (p9 + 9) = {101009.7696539848/1000.0, 0/1000.0, 736.7539243870834/1000.0, lc};
    Point (p9 + 10) = {101098.487489133/1000.0, 0/1000.0, 780.7076003634065/1000.0, lc};
    Point (p9 + 11) = {101183.3091733322/1000.0, 0/1000.0, 828.0001427325635/1000.0, lc};
    Point (p9 + 12) = {101264.1820952265/1000.0, 0/1000.0, 878.6004685484742/1000.0, lc};
    Point (p9 + 13) = {101341.0586989712/1000.0, 0/1000.0, 932.473323565588/1000.0, lc};
    Point (p9 + 14) = {101413.8965767258/1000.0, 0/1000.0, 989.579363358646/1000.0, lc};
    Point (p9 + 15) = {101482.6585446081/1000.0, 0/1000.0, 1049.875247191144/1000.0, lc};
    Point (p9 + 16) = {101547.312701835/1000.0, 0/1000.0, 1113.313744169709/1000.0, lc};
    Point (p9 + 17) = {101607.8324728728/1000.0, 0/1000.0, 1179.843851151494/1000.0, lc};
    Point (p9 + 18) = {101664.1966325262/1000.0, 0/1000.0, 1249.41092180609/1000.0, lc};
    Point (p9 + 19) = {101716.3893139912/1000.0, 0/1000.0, 1321.956806172898/1000.0, lc};
    CatmullRom (9) = {9, p9 + 1, p9 + 2, p9 + 3, p9 + 4, p9 + 5, p9 + 6, p9 + 7, p9 + 8, p9 + 9, p9 + 10, p9 + 11, p9 + 12, p9 + 13, p9 + 14, p9 + 15, p9 + 16, p9 + 17, p9 + 18, p9 + 19, 10};
    p10 = newp;
    Point (p10 + 1) = {100399.8267466054/1000.0, -37.64328850050601/1000.0, 983.1517345866399/1000.0, lc};
    Point (p10 + 2) = {100544.0340097644/1000.0, -56.30285882427393/1000.0, 1190.72213194248/1000.0, lc};
    Point (p10 + 3) = {100625.7701109969/1000.0, -68.2070611051324/1000.0, 1329.88586165233/1000.0, lc};
    Point (p10 + 4) = {100693.6069984745/1000.0, -77.83843427273396/1000.0, 1464.231181343476/1000.0, lc};
    Point (p10 + 5) = {100759.0197231369/1000.0, -86.49249278444159/1000.0, 1603.991484609769/1000.0, lc};
    Point (p10 + 6) = {100825.70614491/1000.0, -94.60964911899997/1000.0, 1751.956898573364/1000.0, lc};
    Point (p10 + 7) = {100895.9566714799/1000.0, -102.4669806582255/1000.0, 1912.632956115628/1000.0, lc};
    Point (p10 + 8) = {100974.5414101193/1000.0, -110.6791909967124/1000.0, 2097.381740563592/1000.0, lc};
    Point (p10 + 9) = {101069.5045533751/1000.0, -120.3381459398932/1000.0, 2324.171450127307/1000.0, lc};
    Point (p10 + 10) = {101188.76250063/1000.0, -132.6618952815267/1000.0, 2610.010253999559/1000.0, lc};
    Point (p10 + 11) = {101335.9311808066/1000.0, -148.5497581743961/1000.0, 2962.067516012177/1000.0, lc};
    Point (p10 + 12) = {101508.1752328875/1000.0, -168.2740303823938/1000.0, 3373.13025279561/1000.0, lc};
    Point (p10 + 13) = {101699.2131058543/1000.0, -191.5212492107784/1000.0, 3828.105406941327/1000.0, lc};
    Point (p10 + 14) = {101904.3406224582/1000.0, -217.5712363905989/1000.0, 4314.845869380807/1000.0, lc};
    Point (p10 + 15) = {102124.3726632129/1000.0, -245.4881661908508/1000.0, 4832.611167649558/1000.0, lc};
    Point (p10 + 16) = {102363.8508919292/1000.0, -274.3448949415413/1000.0, 5388.008869205063/1000.0, lc};
    Point (p10 + 17) = {102625.0307187294/1000.0, -303.4721798503643/1000.0, 5981.720214898422/1000.0, lc};
    Point (p10 + 18) = {102903.8999389907/1000.0, -333.1090041066523/1000.0, 6599.32203120271/1000.0, lc};
    Point (p10 + 19) = {103211.730739768/1000.0, -369.2356817058429/1000.0, 7254.295030661341/1000.0, lc};
    CatmullRom (10) = {9, p10 + 1, p10 + 2, p10 + 3, p10 + 4, p10 + 5, p10 + 6, p10 + 7, p10 + 8, p10 + 9, p10 + 10, p10 + 11, p10 + 12, p10 + 13, p10 + 14, p10 + 15, p10 + 16, p10 + 17, p10 + 18, p10 + 19, 11};
    p11 = newp;
    Point (p11 + 1) = {103675.9209166124/1000.0, -420.6142732306811/1000.0, 8101.889/1000.0, lc};
    Point (p11 + 2) = {103720.0564503957/1000.0, -405.868447415886/1000.0, 8101.889/1000.0, lc};
    Point (p11 + 3) = {103763.4753074087/1000.0, -389.8673941217564/1000.0, 8101.889/1000.0, lc};
    Point (p11 + 4) = {103805.9411872879/1000.0, -372.6430122811637/1000.0, 8101.889/1000.0, lc};
    Point (p11 + 5) = {103847.2163261744/1000.0, -354.2406104321777/1000.0, 8101.889/1000.0, lc};
    Point (p11 + 6) = {103887.0653198091/1000.0, -334.7188783768622/1000.0, 8101.889/1000.0, lc};
    Point (p11 + 7) = {103925.2591153438/1000.0, -314.1495259224355/1000.0, 8101.889/1000.0, lc};
    Point (p11 + 8) = {103961.5790437228/1000.0, -292.6165800368197/1000.0, 8101.889/1000.0, lc};
    Point (p11 + 9) = {103995.8207540876/1000.0, -270.2153470630386/1000.0, 8101.889000000001/1000.0, lc};
    Point (p11 + 10) = {104027.7979088458/1000.0, -247.0510625827172/1000.0, 8101.889/1000.0, lc};
    Point (p11 + 11) = {104057.3455035446/1000.0, -223.2372669481808/1000.0, 8101.889000000001/1000.0, lc};
    Point (p11 + 12) = {104084.3226894926/1000.0, -198.8939582391097/1000.0, 8101.889/1000.0, lc};
    Point (p11 + 13) = {104108.6149984405/1000.0, -174.1455853674513/1000.0, 8101.889/1000.0, lc};
    Point (p11 + 14) = {104130.1358961013/1000.0, -149.1189513980853/1000.0, 8101.889/1000.0, lc};
    Point (p11 + 15) = {104148.8276228517/1000.0, -123.9411003343402/1000.0, 8101.889/1000.0, lc};
    Point (p11 + 16) = {104164.6613132524/1000.0, -98.7372594729372/1000.0, 8101.889/1000.0, lc};
    Point (p11 + 17) = {104177.6364186037/1000.0, -73.62890418278722/1000.0, 8101.889/1000.0, lc};
    Point (p11 + 18) = {104187.7794863365/1000.0, -48.73200317368178/1000.0, 8101.889/1000.0, lc};
    Point (p11 + 19) = {104195.1423747387/1000.0, -24.15549083078653/1000.0, 8101.889/1000.0, lc};
    CatmullRom (11) = {11, p11 + 1, p11 + 2, p11 + 3, p11 + 4, p11 + 5, p11 + 6, p11 + 7, p11 + 8, p11 + 9, p11 + 10, p11 + 11, p11 + 12, p11 + 13, p11 + 14, p11 + 15, p11 + 16, p11 + 17, p11 + 18, p11 + 19, 12};
    p12 = newp;
    Point (p12 + 1) = {101894.5253650667/1000.0, 0/1000.0, 1718.093468752226/1000.0, lc};
    Point (p12 + 2) = {101983.9099956683/1000.0, 0/1000.0, 1944.089196601366/1000.0, lc};
    Point (p12 + 3) = {102047.7391203998/1000.0, 0/1000.0, 2121.423957979851/1000.0, lc};
    Point (p12 + 4) = {102097.762074111/1000.0, 0/1000.0, 2271.745937714456/1000.0, lc};
    Point (p12 + 5) = {102141.9012679034/1000.0, 0/1000.0, 2407.187791802282/1000.0, lc};
    Point (p12 + 6) = {102184.809697928/1000.0, 0/1000.0, 2537.39082976903/1000.0, lc};
    Point (p12 + 7) = {102229.12703304/1000.0, 0/1000.0, 2670.763095115024/1000.0, lc};
    Point (p12 + 8) = {102276.7662514937/1000.0, 0/1000.0, 2815.182173316108/1000.0, lc};
    Point (p12 + 9) = {102329.822163086/1000.0, 0/1000.0, 2978.63920510494/1000.0, lc};
    Point (p12 + 10) = {102390.9664599981/1000.0, 0/1000.0, 3169.80380480494/1000.0, lc};
    Point (p12 + 11) = {102463.7312968615/1000.0, 0/1000.0, 3398.553831849483/1000.0, lc};
    Point (p12 + 12) = {102552.5943002166/1000.0, 0/1000.0, 3676.007201306099/1000.0, lc};
    Point (p12 + 13) = {102662.5623910316/1000.0, 0/1000.0, 4013.290082725893/1000.0, lc};
    Point (p12 + 14) = {102798.5576449433/1000.0, 0/1000.0, 4419.80639623195/1000.0, lc};
    Point (p12 + 15) = {102964.7474391686/1000.0, 0/1000.0, 4901.404174084206/1000.0, lc};
    Point (p12 + 16) = {103163.6367409908/1000.0, 0/1000.0, 5458.12675938652/1000.0, lc};
    Point (p12 + 17) = {103394.9850365313/1000.0, 0/1000.0, 6081.658346485272/1000.0, lc};
    Point (p12 + 18) = {103654.5303327061/1000.0, 0/1000.0, 6752.85835290347/1000.0, lc};
    Point (p12 + 19) = {103930.8646857862/1000.0, 0/1000.0, 7440.533306577965/1000.0, lc};
    CatmullRom (12) = {10, p12 + 1, p12 + 2, p12 + 3, p12 + 4, p12 + 5, p12 + 6, p12 + 7, p12 + 8, p12 + 9, p12 + 10, p12 + 11, p12 + 12, p12 + 13, p12 + 14, p12 + 15, p12 + 16, p12 + 17, p12 + 18, p12 + 19, 12};
    p13 = newp;
    Point (p13 + 1) = {100167.6471505963/1000.0, 0/1000.0, 507.6358599504135/1000.0, lc};
    Point (p13 + 2) = {100285.0763422741/1000.0, 0/1000.0, 524.2763149901309/1000.0, lc};
    Point (p13 + 3) = {100399.1960974999/1000.0, 0/1000.0, 544.3488410104093/1000.0, lc};
    Point (p13 + 4) = {100509.9185091162/1000.0, 0/1000.0, 567.8570241378407/1000.0, lc};
    Point (p13 + 5) = {100617.1594542225/1000.0, 0/1000.0, 594.8001191965965/1000.0, lc};
    Point (p13 + 6) = {100720.8387979568/1000.0, 0/1000.0, 625.1730215329168/1000.0, lc};
    Point (p13 + 7) = {100820.8805861771/1000.0, 0/1000.0, 658.9662524555755/1000.0, lc};
    Point (p13 + 8) = {100917.2132260961/1000.0, 0/1000.0, 696.1659584679518/1000.0, lc};
    Point (p13 + 9) = {101009.7696539848/1000.0, 0/1000.0, 736.7539243870834/1000.0, lc};
    Point (p13 + 10) = {101098.487489133/1000.0, 0/1000.0, 780.7076003634065/1000.0, lc};
    Point (p13 + 11) = {101183.3091733322/1000.0, 0/1000.0, 828.0001427325635/1000.0, lc};
    Point (p13 + 12) = {101264.1820952265/1000.0, 0/1000.0, 878.6004685484742/1000.0, lc};
    Point (p13 + 13) = {101341.0586989712/1000.0, 0/1000.0, 932.473323565588/1000.0, lc};
    Point (p13 + 14) = {101413.8965767258/1000.0, 0/1000.0, 989.579363358646/1000.0, lc};
    Point (p13 + 15) = {101482.6585446081/1000.0, 0/1000.0, 1049.875247191144/1000.0, lc};
    Point (p13 + 16) = {101547.312701835/1000.0, 0/1000.0, 1113.313744169709/1000.0, lc};
    Point (p13 + 17) = {101607.8324728728/1000.0, 0/1000.0, 1179.843851151494/1000.0, lc};
    Point (p13 + 18) = {101664.1966325262/1000.0, 0/1000.0, 1249.41092180609/1000.0, lc};
    Point (p13 + 19) = {101716.3893139912/1000.0, 0/1000.0, 1321.956806172898/1000.0, lc};
    CatmullRom (13) = {13, p13 + 1, p13 + 2, p13 + 3, p13 + 4, p13 + 5, p13 + 6, p13 + 7, p13 + 8, p13 + 9, p13 + 10, p13 + 11, p13 + 12, p13 + 13, p13 + 14, p13 + 15, p13 + 16, p13 + 17, p13 + 18, p13 + 19, 14};
    p14 = newp;
    Point (p14 + 1) = {100399.8267466054/1000.0, 37.64328850050601/1000.0, 983.1517345866399/1000.0, lc};
    Point (p14 + 2) = {100544.0340097644/1000.0, 56.30285882427393/1000.0, 1190.72213194248/1000.0, lc};
    Point (p14 + 3) = {100625.7701109969/1000.0, 68.2070611051324/1000.0, 1329.88586165233/1000.0, lc};
    Point (p14 + 4) = {100693.6069984745/1000.0, 77.83843427273396/1000.0, 1464.231181343476/1000.0, lc};
    Point (p14 + 5) = {100759.0197231369/1000.0, 86.49249278444159/1000.0, 1603.991484609769/1000.0, lc};
    Point (p14 + 6) = {100825.70614491/1000.0, 94.60964911899997/1000.0, 1751.956898573364/1000.0, lc};
    Point (p14 + 7) = {100895.9566714799/1000.0, 102.4669806582255/1000.0, 1912.632956115628/1000.0, lc};
    Point (p14 + 8) = {100974.5414101193/1000.0, 110.6791909967124/1000.0, 2097.381740563592/1000.0, lc};
    Point (p14 + 9) = {101069.5045533751/1000.0, 120.3381459398932/1000.0, 2324.171450127307/1000.0, lc};
    Point (p14 + 10) = {101188.76250063/1000.0, 132.6618952815267/1000.0, 2610.010253999559/1000.0, lc};
    Point (p14 + 11) = {101335.9311808066/1000.0, 148.5497581743961/1000.0, 2962.067516012177/1000.0, lc};
    Point (p14 + 12) = {101508.1752328875/1000.0, 168.2740303823938/1000.0, 3373.13025279561/1000.0, lc};
    Point (p14 + 13) = {101699.2131058543/1000.0, 191.5212492107784/1000.0, 3828.105406941327/1000.0, lc};
    Point (p14 + 14) = {101904.3406224582/1000.0, 217.5712363905989/1000.0, 4314.845869380807/1000.0, lc};
    Point (p14 + 15) = {102124.3726632129/1000.0, 245.4881661908508/1000.0, 4832.611167649558/1000.0, lc};
    Point (p14 + 16) = {102363.8508919292/1000.0, 274.3448949415413/1000.0, 5388.008869205063/1000.0, lc};
    Point (p14 + 17) = {102625.0307187294/1000.0, 303.4721798503643/1000.0, 5981.720214898422/1000.0, lc};
    Point (p14 + 18) = {102903.8999389907/1000.0, 333.1090041066523/1000.0, 6599.32203120271/1000.0, lc};
    Point (p14 + 19) = {103211.730739768/1000.0, 369.2356817058429/1000.0, 7254.295030661341/1000.0, lc};
    CatmullRom (14) = {13, p14 + 1, p14 + 2, p14 + 3, p14 + 4, p14 + 5, p14 + 6, p14 + 7, p14 + 8, p14 + 9, p14 + 10, p14 + 11, p14 + 12, p14 + 13, p14 + 14, p14 + 15, p14 + 16, p14 + 17, p14 + 18, p14 + 19, 15};
    p15 = newp;
    Point (p15 + 1) = {103675.9209166124/1000.0, 420.6142732306811/1000.0, 8101.889/1000.0, lc};
    Point (p15 + 2) = {103720.0564503957/1000.0, 405.868447415886/1000.0, 8101.889/1000.0, lc};
    Point (p15 + 3) = {103763.4753074087/1000.0, 389.8673941217564/1000.0, 8101.889/1000.0, lc};
    Point (p15 + 4) = {103805.9411872879/1000.0, 372.6430122811637/1000.0, 8101.889/1000.0, lc};
    Point (p15 + 5) = {103847.2163261744/1000.0, 354.2406104321777/1000.0, 8101.889/1000.0, lc};
    Point (p15 + 6) = {103887.0653198091/1000.0, 334.7188783768622/1000.0, 8101.889/1000.0, lc};
    Point (p15 + 7) = {103925.2591153438/1000.0, 314.1495259224355/1000.0, 8101.889/1000.0, lc};
    Point (p15 + 8) = {103961.5790437228/1000.0, 292.6165800368197/1000.0, 8101.889/1000.0, lc};
    Point (p15 + 9) = {103995.8207540876/1000.0, 270.2153470630386/1000.0, 8101.889000000001/1000.0, lc};
    Point (p15 + 10) = {104027.7979088458/1000.0, 247.0510625827172/1000.0, 8101.889/1000.0, lc};
    Point (p15 + 11) = {104057.3455035446/1000.0, 223.2372669481808/1000.0, 8101.889000000001/1000.0, lc};
    Point (p15 + 12) = {104084.3226894926/1000.0, 198.8939582391097/1000.0, 8101.889/1000.0, lc};
    Point (p15 + 13) = {104108.6149984405/1000.0, 174.1455853674513/1000.0, 8101.889/1000.0, lc};
    Point (p15 + 14) = {104130.1358961013/1000.0, 149.1189513980853/1000.0, 8101.889/1000.0, lc};
    Point (p15 + 15) = {104148.8276228517/1000.0, 123.9411003343402/1000.0, 8101.889/1000.0, lc};
    Point (p15 + 16) = {104164.6613132524/1000.0, 98.7372594729372/1000.0, 8101.889/1000.0, lc};
    Point (p15 + 17) = {104177.6364186037/1000.0, 73.62890418278722/1000.0, 8101.889/1000.0, lc};
    Point (p15 + 18) = {104187.7794863365/1000.0, 48.73200317368178/1000.0, 8101.889/1000.0, lc};
    Point (p15 + 19) = {104195.1423747387/1000.0, 24.15549083078653/1000.0, 8101.889/1000.0, lc};
    CatmullRom (15) = {15, p15 + 1, p15 + 2, p15 + 3, p15 + 4, p15 + 5, p15 + 6, p15 + 7, p15 + 8, p15 + 9, p15 + 10, p15 + 11, p15 + 12, p15 + 13, p15 + 14, p15 + 15, p15 + 16, p15 + 17, p15 + 18, p15 + 19, 16};
    p16 = newp;
    Point (p16 + 1) = {101894.5253650667/1000.0, 0/1000.0, 1718.093468752226/1000.0, lc};
    Point (p16 + 2) = {101983.9099956683/1000.0, 0/1000.0, 1944.089196601366/1000.0, lc};
    Point (p16 + 3) = {102047.7391203998/1000.0, 0/1000.0, 2121.423957979851/1000.0, lc};
    Point (p16 + 4) = {102097.762074111/1000.0, 0/1000.0, 2271.745937714456/1000.0, lc};
    Point (p16 + 5) = {102141.9012679034/1000.0, 0/1000.0, 2407.187791802282/1000.0, lc};
    Point (p16 + 6) = {102184.809697928/1000.0, 0/1000.0, 2537.39082976903/1000.0, lc};
    Point (p16 + 7) = {102229.12703304/1000.0, 0/1000.0, 2670.763095115024/1000.0, lc};
    Point (p16 + 8) = {102276.7662514937/1000.0, 0/1000.0, 2815.182173316108/1000.0, lc};
    Point (p16 + 9) = {102329.822163086/1000.0, 0/1000.0, 2978.63920510494/1000.0, lc};
    Point (p16 + 10) = {102390.9664599981/1000.0, 0/1000.0, 3169.80380480494/1000.0, lc};
    Point (p16 + 11) = {102463.7312968615/1000.0, 0/1000.0, 3398.553831849483/1000.0, lc};
    Point (p16 + 12) = {102552.5943002166/1000.0, 0/1000.0, 3676.007201306099/1000.0, lc};
    Point (p16 + 13) = {102662.5623910316/1000.0, 0/1000.0, 4013.290082725893/1000.0, lc};
    Point (p16 + 14) = {102798.5576449433/1000.0, 0/1000.0, 4419.80639623195/1000.0, lc};
    Point (p16 + 15) = {102964.7474391686/1000.0, 0/1000.0, 4901.404174084206/1000.0, lc};
    Point (p16 + 16) = {103163.6367409908/1000.0, 0/1000.0, 5458.12675938652/1000.0, lc};
    Point (p16 + 17) = {103394.9850365313/1000.0, 0/1000.0, 6081.658346485272/1000.0, lc};
    Point (p16 + 18) = {103654.5303327061/1000.0, 0/1000.0, 6752.85835290347/1000.0, lc};
    Point (p16 + 19) = {103930.8646857862/1000.0, 0/1000.0, 7440.533306577965/1000.0, lc};
    CatmullRom (16) = {14, p16 + 1, p16 + 2, p16 + 3, p16 + 4, p16 + 5, p16 + 6, p16 + 7, p16 + 8, p16 + 9, p16 + 10, p16 + 11, p16 + 12, p16 + 13, p16 + 14, p16 + 15, p16 + 16, p16 + 17, p16 + 18, p16 + 19, 16};
    p17 = newp;
    Point (p17 + 1) = {9317.554799738758/1000.0, -139.3694885613147/1000.0, 2942.857971742675/1000.0, lc};
    Point (p17 + 2) = {14724.64862612/1000.0, -210.18325075198/1000.0, 2113.413688138912/1000.0, lc};
    Point (p17 + 3) = {19611.19286465473/1000.0, -246.6193145858631/1000.0, 1125.284989972894/1000.0, lc};
    Point (p17 + 4) = {23249.02097322105/1000.0, -278.5639893578149/1000.0, 339.0656431875292/1000.0, lc};
    Point (p17 + 5) = {25371.65146785838/1000.0, -292.0802043446009/1000.0, -58.01804876601409/1000.0, lc};
    Point (p17 + 6) = {26666.10758849813/1000.0, -290.6949410912934/1000.0, -193.201108439527/1000.0, lc};
    Point (p17 + 7) = {27882.57724661941/1000.0, -281.8773829745387/1000.0, -232.9839480135292/1000.0, lc};
    Point (p17 + 8) = {29606.24744292548/1000.0, -270.129717707284/1000.0, -239.1902570770887/1000.0, lc};
    Point (p17 + 9) = {32696.47337998234/1000.0, -265.6575382761905/1000.0, -217.2925098303248/1000.0, lc};
    Point (p17 + 10) = {37975.08112348829/1000.0, -275.4882892124488/1000.0, -160.2421458246485/1000.0, lc};
    Point (p17 + 11) = {45455.35188076008/1000.0, -290.4158848525699/1000.0, -70.55141422325377/1000.0, lc};
    Point (p17 + 12) = {53898.6780871717/1000.0, -298.3127464947366/1000.0, 26.82317735520494/1000.0, lc};
    Point (p17 + 13) = {61374.29196374536/1000.0, -299.2965718131835/1000.0, 94.45420596456063/1000.0, lc};
    Point (p17 + 14) = {67450.02876813264/1000.0, -298.0797935904579/1000.0, 130.4626228617591/1000.0, lc};
    Point (p17 + 15) = {73712.18212436585/1000.0, -295.6811939986992/1000.0, 178.0523068123641/1000.0, lc};
    Point (p17 + 16) = {81126.28015937647/1000.0, -290.7568407304351/1000.0, 262.6520637541346/1000.0, lc};
    Point (p17 + 17) = {88730.75178590092/1000.0, -273.5900336715672/1000.0, 364.5881605037617/1000.0, lc};
    Point (p17 + 18) = {94710.76120887922/1000.0, -216.3209508941758/1000.0, 448.1572682599385/1000.0, lc};
    Point (p17 + 19) = {98242.99731098629/1000.0, -99.21419481730666/1000.0, 490.0647216050518/1000.0, lc};
    CatmullRom (17) = {17, p17 + 1, p17 + 2, p17 + 3, p17 + 4, p17 + 5, p17 + 6, p17 + 7, p17 + 8, p17 + 9, p17 + 10, p17 + 11, p17 + 12, p17 + 13, p17 + 14, p17 + 15, p17 + 16, p17 + 17, p17 + 18, p17 + 19, 18};
    p18 = newp;
    Point (p18 + 1) = {100038.948/1000.0, 0/1000.0, 495.34017/1000.0, lc};
    Point (p18 + 2) = {100030.896/1000.0, 0/1000.0, 496.26074/1000.0, lc};
    Point (p18 + 3) = {100022.844/1000.0, 0/1000.0, 497.18131/1000.0, lc};
    Point (p18 + 4) = {100014.792/1000.0, 0/1000.0, 498.10188/1000.0, lc};
    Point (p18 + 5) = {100006.74/1000.0, 0/1000.0, 499.02245/1000.0, lc};
    Point (p18 + 6) = {99998.68799999999/1000.0, 0/1000.0, 499.94302/1000.0, lc};
    Point (p18 + 7) = {99990.636/1000.0, 0/1000.0, 500.86359/1000.0, lc};
    Point (p18 + 8) = {99982.584/1000.0, 0/1000.0, 501.78416/1000.0, lc};
    Point (p18 + 9) = {99974.53200000001/1000.0, 0/1000.0, 502.70473/1000.0, lc};
    Point (p18 + 10) = {99966.48000000001/1000.0, 0/1000.0, 503.6253/1000.0, lc};
    Point (p18 + 11) = {99958.428/1000.0, 0/1000.0, 504.54587/1000.0, lc};
    Point (p18 + 12) = {99950.376/1000.0, 0/1000.0, 505.46644/1000.0, lc};
    Point (p18 + 13) = {99942.32400000001/1000.0, 0/1000.0, 506.38701/1000.0, lc};
    Point (p18 + 14) = {99934.27200000001/1000.0, 0/1000.0, 507.30758/1000.0, lc};
    Point (p18 + 15) = {99926.22/1000.0, 0/1000.0, 508.22815/1000.0, lc};
    Point (p18 + 16) = {99918.16800000001/1000.0, 0/1000.0, 509.14872/1000.0, lc};
    Point (p18 + 17) = {99910.11600000001/1000.0, 0/1000.0, 510.06929/1000.0, lc};
    Point (p18 + 18) = {99902.064/1000.0, 0/1000.0, 510.98986/1000.0, lc};
    Point (p18 + 19) = {99894.012/1000.0, 0/1000.0, 511.91043/1000.0, lc};
    CatmullRom (18) = {18, p18 + 1, p18 + 2, p18 + 3, p18 + 4, p18 + 5, p18 + 6, p18 + 7, p18 + 8, p18 + 9, p18 + 10, p18 + 11, p18 + 12, p18 + 13, p18 + 14, p18 + 15, p18 + 16, p18 + 17, p18 + 18, p18 + 19, 19};
    p19 = newp;
    Point (p19 + 1) = {9318.805498911808/1000.0, 0/1000.0, 2936.253456825882/1000.0, lc};
    Point (p19 + 2) = {14738.50810010411/1000.0, 0/1000.0, 2111.74837431424/1000.0, lc};
    Point (p19 + 3) = {19623.46326334996/1000.0, 0/1000.0, 1124.871964999602/1000.0, lc};
    Point (p19 + 4) = {23265.17461182307/1000.0, 0/1000.0, 339.0304501252059/1000.0, lc};
    Point (p19 + 5) = {25411.33693511236/1000.0, 0/1000.0, -57.90299974473172/1000.0, lc};
    Point (p19 + 6) = {26721.2498839577/1000.0, 0/1000.0, -192.9737408251548/1000.0, lc};
    Point (p19 + 7) = {27929.86353924959/1000.0, 0/1000.0, -232.6497529716127/1000.0, lc};
    Point (p19 + 8) = {29636.07631906199/1000.0, 0/1000.0, -238.8281846293573/1000.0, lc};
    Point (p19 + 9) = {32712.25091011242/1000.0, 0/1000.0, -217.0278961163639/1000.0, lc};
    Point (p19 + 10) = {37983.65663790161/1000.0, 0/1000.0, -160.1623840070317/1000.0, lc};
    Point (p19 + 11) = {45463.98657871379/1000.0, 0/1000.0, -70.64118864945422/1000.0, lc};
    Point (p19 + 12) = {53911.95089646555/1000.0, 0/1000.0, 26.66805113801968/1000.0, lc};
    Point (p19 + 13) = {61390.93681082637/1000.0, 0/1000.0, 94.34534722928311/1000.0, lc};
    Point (p19 + 14) = {67467.02972142941/1000.0, 0/1000.0, 130.4489038904908/1000.0, lc};
    Point (p19 + 15) = {73729.79407107766/1000.0, 0/1000.0, 178.093648191419/1000.0, lc};
    Point (p19 + 16) = {81142.24379960362/1000.0, 0/1000.0, 262.6517539250688/1000.0, lc};
    Point (p19 + 17) = {88734.19252399549/1000.0, 0/1000.0, 364.5081196046716/1000.0, lc};
    Point (p19 + 18) = {94503.10149100069/1000.0, 0/1000.0, 447.2131248251829/1000.0, lc};
    Point (p19 + 19) = {96284.19956287603/1000.0, 0/1000.0, 480.8841776329334/1000.0, lc};
    CatmullRom (19) = {17, p19 + 1, p19 + 2, p19 + 3, p19 + 4, p19 + 5, p19 + 6, p19 + 7, p19 + 8, p19 + 9, p19 + 10, p19 + 11, p19 + 12, p19 + 13, p19 + 14, p19 + 15, p19 + 16, p19 + 17, p19 + 18, p19 + 19, 19};
    p20 = newp;
    Point (p20 + 1) = {9317.554799738758/1000.0, 139.3694885613147/1000.0, 2942.857971742675/1000.0, lc};
    Point (p20 + 2) = {14724.64862612/1000.0, 210.18325075198/1000.0, 2113.413688138912/1000.0, lc};
    Point (p20 + 3) = {19611.19286465473/1000.0, 246.6193145858631/1000.0, 1125.284989972894/1000.0, lc};
    Point (p20 + 4) = {23249.02097322105/1000.0, 278.5639893578149/1000.0, 339.0656431875292/1000.0, lc};
    Point (p20 + 5) = {25371.65146785838/1000.0, 292.0802043446009/1000.0, -58.01804876601409/1000.0, lc};
    Point (p20 + 6) = {26666.10758849813/1000.0, 290.6949410912934/1000.0, -193.201108439527/1000.0, lc};
    Point (p20 + 7) = {27882.57724661941/1000.0, 281.8773829745387/1000.0, -232.9839480135292/1000.0, lc};
    Point (p20 + 8) = {29606.24744292548/1000.0, 270.129717707284/1000.0, -239.1902570770887/1000.0, lc};
    Point (p20 + 9) = {32696.47337998234/1000.0, 265.6575382761905/1000.0, -217.2925098303248/1000.0, lc};
    Point (p20 + 10) = {37975.08112348829/1000.0, 275.4882892124488/1000.0, -160.2421458246485/1000.0, lc};
    Point (p20 + 11) = {45455.35188076008/1000.0, 290.4158848525699/1000.0, -70.55141422325377/1000.0, lc};
    Point (p20 + 12) = {53898.6780871717/1000.0, 298.3127464947366/1000.0, 26.82317735520494/1000.0, lc};
    Point (p20 + 13) = {61374.29196374536/1000.0, 299.2965718131835/1000.0, 94.45420596456063/1000.0, lc};
    Point (p20 + 14) = {67450.02876813264/1000.0, 298.0797935904579/1000.0, 130.4626228617591/1000.0, lc};
    Point (p20 + 15) = {73712.18212436585/1000.0, 295.6811939986992/1000.0, 178.0523068123641/1000.0, lc};
    Point (p20 + 16) = {81126.28015937647/1000.0, 290.7568407304351/1000.0, 262.6520637541346/1000.0, lc};
    Point (p20 + 17) = {88730.75178590092/1000.0, 273.5900336715672/1000.0, 364.5881605037617/1000.0, lc};
    Point (p20 + 18) = {94710.76120887922/1000.0, 216.3209508941758/1000.0, 448.1572682599385/1000.0, lc};
    Point (p20 + 19) = {98242.99731098629/1000.0, 99.21419481730666/1000.0, 490.0647216050518/1000.0, lc};
    CatmullRom (20) = {20, p20 + 1, p20 + 2, p20 + 3, p20 + 4, p20 + 5, p20 + 6, p20 + 7, p20 + 8, p20 + 9, p20 + 10, p20 + 11, p20 + 12, p20 + 13, p20 + 14, p20 + 15, p20 + 16, p20 + 17, p20 + 18, p20 + 19, 21};
    p21 = newp;
    Point (p21 + 1) = {100038.948/1000.0, 0/1000.0, 495.34017/1000.0, lc};
    Point (p21 + 2) = {100030.896/1000.0, 0/1000.0, 496.26074/1000.0, lc};
    Point (p21 + 3) = {100022.844/1000.0, 0/1000.0, 497.18131/1000.0, lc};
    Point (p21 + 4) = {100014.792/1000.0, 0/1000.0, 498.10188/1000.0, lc};
    Point (p21 + 5) = {100006.74/1000.0, 0/1000.0, 499.02245/1000.0, lc};
    Point (p21 + 6) = {99998.68799999999/1000.0, 0/1000.0, 499.94302/1000.0, lc};
    Point (p21 + 7) = {99990.636/1000.0, 0/1000.0, 500.86359/1000.0, lc};
    Point (p21 + 8) = {99982.584/1000.0, 0/1000.0, 501.78416/1000.0, lc};
    Point (p21 + 9) = {99974.53200000001/1000.0, 0/1000.0, 502.70473/1000.0, lc};
    Point (p21 + 10) = {99966.48000000001/1000.0, 0/1000.0, 503.6253/1000.0, lc};
    Point (p21 + 11) = {99958.428/1000.0, 0/1000.0, 504.54587/1000.0, lc};
    Point (p21 + 12) = {99950.376/1000.0, 0/1000.0, 505.46644/1000.0, lc};
    Point (p21 + 13) = {99942.32400000001/1000.0, 0/1000.0, 506.38701/1000.0, lc};
    Point (p21 + 14) = {99934.27200000001/1000.0, 0/1000.0, 507.30758/1000.0, lc};
    Point (p21 + 15) = {99926.22/1000.0, 0/1000.0, 508.22815/1000.0, lc};
    Point (p21 + 16) = {99918.16800000001/1000.0, 0/1000.0, 509.14872/1000.0, lc};
    Point (p21 + 17) = {99910.11600000001/1000.0, 0/1000.0, 510.06929/1000.0, lc};
    Point (p21 + 18) = {99902.064/1000.0, 0/1000.0, 510.98986/1000.0, lc};
    Point (p21 + 19) = {99894.012/1000.0, 0/1000.0, 511.91043/1000.0, lc};
    CatmullRom (21) = {21, p21 + 1, p21 + 2, p21 + 3, p21 + 4, p21 + 5, p21 + 6, p21 + 7, p21 + 8, p21 + 9, p21 + 10, p21 + 11, p21 + 12, p21 + 13, p21 + 14, p21 + 15, p21 + 16, p21 + 17, p21 + 18, p21 + 19, 22};
    p22 = newp;
    Point (p22 + 1) = {9318.805498911808/1000.0, 0/1000.0, 2936.253456825882/1000.0, lc};
    Point (p22 + 2) = {14738.50810010411/1000.0, 0/1000.0, 2111.74837431424/1000.0, lc};
    Point (p22 + 3) = {19623.46326334996/1000.0, 0/1000.0, 1124.871964999602/1000.0, lc};
    Point (p22 + 4) = {23265.17461182307/1000.0, 0/1000.0, 339.0304501252059/1000.0, lc};
    Point (p22 + 5) = {25411.33693511236/1000.0, 0/1000.0, -57.90299974473172/1000.0, lc};
    Point (p22 + 6) = {26721.2498839577/1000.0, 0/1000.0, -192.9737408251548/1000.0, lc};
    Point (p22 + 7) = {27929.86353924959/1000.0, 0/1000.0, -232.6497529716127/1000.0, lc};
    Point (p22 + 8) = {29636.07631906199/1000.0, 0/1000.0, -238.8281846293573/1000.0, lc};
    Point (p22 + 9) = {32712.25091011242/1000.0, 0/1000.0, -217.0278961163639/1000.0, lc};
    Point (p22 + 10) = {37983.65663790161/1000.0, 0/1000.0, -160.1623840070317/1000.0, lc};
    Point (p22 + 11) = {45463.98657871379/1000.0, 0/1000.0, -70.64118864945422/1000.0, lc};
    Point (p22 + 12) = {53911.95089646555/1000.0, 0/1000.0, 26.66805113801968/1000.0, lc};
    Point (p22 + 13) = {61390.93681082637/1000.0, 0/1000.0, 94.34534722928311/1000.0, lc};
    Point (p22 + 14) = {67467.02972142941/1000.0, 0/1000.0, 130.4489038904908/1000.0, lc};
    Point (p22 + 15) = {73729.79407107766/1000.0, 0/1000.0, 178.093648191419/1000.0, lc};
    Point (p22 + 16) = {81142.24379960362/1000.0, 0/1000.0, 262.6517539250688/1000.0, lc};
    Point (p22 + 17) = {88734.19252399549/1000.0, 0/1000.0, 364.5081196046716/1000.0, lc};
    Point (p22 + 18) = {94503.10149100069/1000.0, 0/1000.0, 447.2131248251829/1000.0, lc};
    Point (p22 + 19) = {96284.19956287603/1000.0, 0/1000.0, 480.8841776329334/1000.0, lc};
    CatmullRom (22) = {20, p22 + 1, p22 + 2, p22 + 3, p22 + 4, p22 + 5, p22 + 6, p22 + 7, p22 + 8, p22 + 9, p22 + 10, p22 + 11, p22 + 12, p22 + 13, p22 + 14, p22 + 15, p22 + 16, p22 + 17, p22 + 18, p22 + 19, 22};
    Line Loop (1) = {1, 4, -3, -2};
    Ruled Surface (1) = {1};
    Line Loop (2) = {5, 8, -7, -6};
    Ruled Surface (2) = {2};
    Line Loop (3) = {9, 12, -11, -10};
    Ruled Surface (3) = {3};
    Line Loop (4) = {13, 16, -15, -14};
    Ruled Surface (4) = {4};
    Line Loop (5) = {17, 18, -19};
    Ruled Surface (5) = {5};
    Line Loop (6) = {20, 21, -22};
    Ruled Surface (6) = {6};
    """

    geometry = "lc = " + str(h) + ";\n" + stub
    return __generate_grid_from_geo_string(geometry)


def plane(h=0.1):

    stub ="""
    Point(1) = {0, 0, 0, lc1};

    Point(2) = {0, 8.15, 0.275, lc1};

    Point(3) = {0.4, 3.5, 0, lc2};

    Point(4) = {-0.4, 3.5, 0, lc1};

    Point(5) = {-0.4, 4.75, 0, lc1};

    Point(6) = {0.4, 4.75, 0, lc2};

    Point(7) = {7.4, 4.45, 0, lc2};

    Point(8) = {-7.4, 4.45, 0, lc1};

    Point(9) = {-7.4, 3.9, 0, lc1};

    Point(10) = {7.4, 3.9, 0, lc2};

    Point(15) = {0.5, 0.35, 0, lc1};

    Point(16) = {-0.5, 0.35, 0, lc1};

    Point(17) = {-0.6, 0.95, 0, lc1};

    Point(18) = {0.6, 0.95, 0, lc1};

    Point(19) = {0.5, 2, 0, lc1};

    Point(20) = {-0.5, 2, 0, lc1};

    Point(21) = {-0.3, 6.9, 0, lc1};

    Point(22) = {0.3, 6.9, 0, lc1};

    Point(23) = {0.15, 7.6, 0, lc1};

    Point(24) = {-0.15, 7.6, 0, lc1};

    Point(25) = {0, 7.9, 0.08, lc1};

    Point(26) = {0, 7.9, 0.47, lc1};

    Point(27) = {0.1, 8.1, 0.275, lc1};

    Point(28) = {-0.1, 8.1, 0.275, lc1};

    Point(29) = {2, 6.9, -1.2, lc1};

    Point(30) = {-2, 6.9, -1.2, lc1};

    Point(31) = {-2, 7.6, -1.2, lc1};

    Point(32) = {2, 7.6, -1.2, lc1};

    Point(33) = {0, 0.8, 0.68, lc1};

    Point(66) = {0, 1.1, 0.75, lc1};

    Point(68) = {0, 3.25, 0.55, lc1};

    Point(70) = {0, 4.4, 0.55, lc1};

    Point(72) = {0.37, 3.5, 0.1, lc2};

    Point(73) = {-0.37, 3.5, 0.1, lc1};

    Point(74) = {-0.37, 4.75, 0.1, lc1};

    Point(75) = {0.37, 4.75, 0.1, lc2};

    Point(78) = {7.4, 4.45, 0.05, lc2};

    Point(79) = {-7.4, 4.45, 0.05, lc1};

    Point(115) = {-7.4, 3.9, 0.05, lc1};

    Point(116) = {7.4, 3.9, 0.05, lc2};

    Point(117) = {0, 3.65, 0.68, lc1};

    Point(178) = {2, 6.9, -1.15, lc1};

    Point(179) = {-2, 6.9, -1.15, lc1};

    Point(180) = {-2, 7.6, -1.15, lc1};

    Point(181) = {2, 7.6, -1.15, lc1};

    Point(182) = {0.37, 6.9, 0.1, lc1};

    Point(183) = {-0.37, 6.9, 0.1, lc1};

    Point(184) = {-0.27, 7.6, 0.1, lc1};

    Point(185) = {0.27, 7.6, 0.1, lc1};

    Point(186) = {0, 1.1, 0.75, lc1};

    Point(187) = {0.45, 1.1, 0.4, lc1};

    Point(188) = {-0.45, 1.1, 0.4, lc1};

    Point(189) = {-0.55, 1.1, 0.1, lc1};

    Point(190) = {0.55, 1.1, 0.1, lc1};

    Point(252) = {0, 0.35, -0.35, lc1};

    Point(259) = {0.3, 0.35, -0.22, lc1};

    Point(260) = {-0.3, 0.35, -0.22, lc1};

    Point(261) = {0.4, 0.35, -0.07000000000000001, lc1};

    Point(262) = {-0.4, 0.35, -0.07000000000000001, lc1};

    Point(263) = {0.45, 0.35, 0.1, lc1};

    Point(264) = {-0.45, 0.35, 0.1, lc1};

    Point(265) = {0, 3.7, -0.06, lc1};

    Point(266) = {0, 2, -0.22, lc1};

    Point(267) = {0, 6.9, -0.05, lc1};

    Point(269) = {0, 8.1, 0.375, lc1};

    Point(270) = {0, 8.1, 0.275, lc1};

    Point(271) = {0, 8.1, 0.175, lc1};

    Point(272) = {0, 7.6, 0, lc1};

    Point(273) = {0, 7.8, 0.5, lc1};

    Point(274) = {0, 7.8, 0.05, lc1};

    Point(275) = {0.225, 7.8, 0.275, lc1};

    Point(276) = {-0.225, 7.8, 0.275, lc1};

    Point(277) = {0, 7.8, 0.275, lc1};

    Point(278) = {0.6, 1.1, 0, lc1};

    Point(279) = {-0.6, 1.1, 0, lc1};

    Point(280) = {0, 1.1, -0.3, lc1};

    Point(281) = {0.3, 1.1, -0.17, lc1};

    Point(282) = {-0.3, 1.1, -0.17, lc1};

    Point(283) = {0.45, 1.1, -0.05, lc1};

    Point(284) = {-0.45, 1.1, -0.05, lc1};

    Point(285) = {0, 3.25, -0.1, lc1};

    Point(286) = {0.3, 3.25, 0.35, lc1};

    Point(287) = {-0.3, 3.25, 0.35, lc1};

    Point(288) = {-0.37, 3.25, 0.1, lc1};

    Point(289) = {0.37, 3.25, 0.1, lc1};

    Point(290) = {-0.3, 3.25, -0.025, lc1};

    Point(291) = {0.3, 3.25, -0.025, lc1};

    Point(292) = {0.2, 3.25, -0.07000000000000001, lc1};

    Point(293) = {-0.2, 3.25, -0.07000000000000001, lc1};

    Point(294) = {0.4, 3.25, 0, lc1};

    Point(295) = {-0.4, 3.25, 0, lc1};

    Point(296) = {0, 4.75, 0.55, lc1};

    Point(297) = {-0.3, 4.75, 0.35, lc1};

    Point(298) = {0.3, 4.75, 0.35, lc1};

    Point(299) = {0, 4.75, -0.05, lc1};

    Point(300) = {0.2, 3.65, 0.5, lc1};

    Point(301) = {-0.2, 3.65, 0.5, lc1};

    Point(302) = {0.3, 3.65, 0.35, lc1};

    Point(303) = {-0.3, 3.65, 0.35, lc1};

    Point(304) = {0.37, 3.65, 0.127, lc1};

    Point(305) = {-0.37, 3.65, 0.127, lc1};

    Point(306) = {0, 3.65, 0.55, lc1};

    Point(307) = {0, 6.9, 0.55, lc1};

    Point(311) = {0.27, 6.9, 0.35, lc1};

    Point(312) = {-0.27, 6.9, 0.35, lc1};

    Point(313) = {0.195, 7.9, 0.275, lc1};

    Point(314) = {-0.195, 7.9, 0.275, lc1};

    Point(315) = {0, 7.9, 0.275, lc1};

    Point(316) = {0, 7.6, 0.55, lc1};

    Point(317) = {-0.2, 7.6, 0.35, lc1};

    Point(318) = {0.2, 7.6, 0.35, lc1};

    Point(319) = {0.3, 0.35, 0.3, lc1};

    Point(320) = {-0.3, 0.35, 0.3, lc1};

    Point(321) = {0, 0.35, 0.45, lc1};

    Point(322) = {0, 0.1, 0.1, lc1};

    Point(323) = {0.3, 1.1, 0.63, lc1};

    Point(324) = {-0.3, 1.1, 0.63, lc1};

    Point(325) = {0.15, 1.1, 0.72, lc1};

    Point(326) = {-0.15, 1.1, 0.72, lc1};

    Point(327) = {0.15, 0.35, 0.4, lc1};

    Point(328) = {-0.15, 0.35, 0.4, lc1};

    Point(329) = {-0.15, 0.35, -0.32, lc1};

    Point(330) = {0.15, 0.35, -0.32, lc1};

    Point(331) = {0.2, 1.1, -0.25, lc1};

    Point(332) = {-0.2, 1.1, -0.25, lc1};

    Point(333) = {0.15, 3.25, 0.5, lc1};

    Point(334) = {-0.15, 3.25, 0.5, lc1};

    Point(335) = {-0.15, 4.75, 0.5, lc1};

    Point(336) = {0.15, 4.75, 0.5, lc1};

    Point(337) = {0.1, 3.65, 0.63, lc1};

    Point(338) = {-0.1, 3.65, 0.63, lc1};

    Point(339) = {0.15, 6.9, 0.5, lc1};

    Point(340) = {-0.15, 6.9, 0.5, lc1};

    Point(341) = {-0.12, 7.6, 0.5, lc1};

    Point(342) = {0.12, 7.6, 0.5, lc1};

    Point(343) = {0.05, 6.9, -0.05, lc1};

    Point(344) = {-0.05, 6.9, -0.05, lc1};

    Point(345) = {-0.05, 7.6, 0, lc1};

    Point(346) = {0.05, 7.6, 0, lc1};

    Point(347) = {0.025, 7.6, -0.8, lc1};

    Point(348) = {-0.025, 7.6, -0.8, lc1};

    Point(349) = {-0.025, 6.9, -0.8, lc1};

    Point(350) = {0.025, 6.9, -0.8, lc1};

    Point(351) = {-7.4, 4.1, -0.02, lc1};

    Point(352) = {-7.4, 4.1, 0.08, lc1};

    Point(353) = {-0.4, 3.8, -0.02, lc1};

    Point(354) = {-0.37, 3.9, 0.15, lc1};

    Point(355) = {7.4, 4.1, 0.08, lc2};

    Point(356) = {7.4, 4.1, -0.02, lc2};

    Point(357) = {0.37, 3.9, 0.15, lc2};

    Point(358) = {0.4, 3.8, -0.02, lc2};

    Point(359) = {0.2, 0.06, 0, lc1};

    Point(360) = {-0.2, 0.06, 0, lc1};

    Point(362) = {0.25, 0.16, 0.1, lc1};

    Point(363) = {-0.25, 0.16, 0.1, lc1};

    Point(445) = {0.0495, 7.9, 0.4636, lc1};

    Point(446) = {-0.0495, 7.9, 0.4636, lc1};

    Point(447) = {-0.0495, 7.9, 0.0864, lc1};

    Point(448) = {0.0495, 7.9, 0.0864, lc1};

    Point(449) = {0.0496, 7.9, 0.8636, lc1};

    Point(450) = {-0.0496, 7.9, 0.8636, lc1};

    Point(451) = {-0.03, 7.88, 1.0636, lc1};

    Point(452) = {0.03, 7.88, 1.0636, lc1};

    Point(453) = {0.03, 7.88, -0.5135999999999999, lc1};

    Point(454) = {-0.03, 7.88, -0.5135999999999999, lc1};

    Point(456) = {-0.0495, 7.9, -0.3136, lc1};

    Point(457) = {0.0495, 7.9, -0.3136, lc1};

    Point(458) = {0.0496, 7.8, 0.0555, lc1};

    Point(459) = {-0.0496, 7.8, 0.0555, lc1};

    Point(460) = {-0.0496, 7.8, 0.4945, lc1};

    Point(461) = {0.0496, 7.8, 0.4945, lc1};

    Point(462) = {0.03, 7.82, -0.5135999999999999, lc1};

    Point(463) = {-0.03, 7.82, -0.5135999999999999, lc1};

    Point(464) = {-0.03, 7.82, 1.0636, lc1};

    Point(465) = {0.03, 7.82, 1.0636, lc1};

    Point(466) = {0.0496, 7.8, -0.3136, lc1};

    Point(467) = {-0.0496, 7.8, -0.3136, lc1};

    Point(468) = {-0.0496, 7.8, 0.8636, lc1};

    Point(469) = {0.0496, 7.8, 0.8636, lc1};

    Point(470) = {0, 0.08, -0.14, lc1};

    Point(471) = {0, 0.6, -0.4, lc1};

    Point(472) = {0.2, 4.75, -0.04, lc1};

    Point(473) = {-0.2, 4.75, -0.04, lc1};

    Point(474) = {0.2, 6.9, -0.04, lc1};

    Point(475) = {-0.2, 6.9, -0.04, lc1};

    Point(476) = {0.3, 3.5, -0.025, lc1};

    Point(477) = {-0.3, 3.5, -0.025, lc1};

    Point(478) = {-0.2, 3.5, -0.07000000000000001, lc1};

    Point(479) = {0.2, 3.5, -0.07000000000000001, lc1};

    Point(480) = {0, 3.5, -0.07000000000000001, lc1};

    Point(481) = {-0.05, 5.4, -0.05, lc1};

    Point(482) = {0.05, 5.4, -0.05, lc1};

    Point(483) = {0.13, 4, 0.52, lc1};

    Point(484) = {-0.13, 4, 0.52, lc1};

    Point(485) = {-0.1, 3.4, 0.52, lc1};

    Point(486) = {0.1, 3.4, 0.52, lc1};

    Point(487) = {0, 4, 0.63, lc1};

    Point(488) = {0, 3.4, 0.63, lc1};

    Point(489) = {0, 0.2, 0.3, lc1};

    Point(23502) = {-0.3, 4, 0.35, lc1};

    Point(23503) = {0.3, 4, 0.35, lc1};

    Point(23504) = {0, 7.88, -0.5135999999999999, lc1};

    Point(23505) = {0, 7.82, -0.5135999999999999, lc1};

    Point(23506) = {0, 7.88, 1.0636, lc1};

    Point(23507) = {0, 7.82, 1.0636, lc1};

    Point(23508) = {0, 7.8, 0.8636, lc1};

    Point(23509) = {0, 7.9, 0.8636, lc1};

    Point(23510) = {0, 7.8, -0.3136, lc1};

    Point(23511) = {0, 7.9, -0.3136, lc1};

    Line (1) = {4, 9};

    Line (4) = {3, 10};

    Line (5) = {6, 7};

    Line (7) = {6, 22};

    Line (15) = {73, 4};

    Line (16) = {74, 79};

    Line (18) = {79, 8};

    Line (19) = {115, 9};

    Line (20) = {115, 73};

    Line (22) = {21, 30};

    Line (23) = {24, 31};

    Line (24) = {31, 30};

    Line (26) = {29, 32};

    Line (27) = {23, 32};

    Line (29) = {72, 116};

    Line (30) = {75, 78};

    Line (32) = {78, 7};

    Line (33) = {116, 10};

    Line (39) = {183, 179};

    Line (40) = {179, 180};

    Line (41) = {180, 31};

    Line (42) = {179, 30};

    Line (43) = {183, 184};

    Line (44) = {178, 181};

    Line (45) = {181, 185};

    Line (46) = {178, 182};

    Line (47) = {182, 185};

    Line (48) = {23, 22};

    Line (49) = {24, 21};

    Line (50) = {178, 29};

    Line (51) = {181, 32};

    Circle (62) = {269, 270, 28} Plane{0, 0, 1};

    Circle (63) = {28, 270, 271} Plane{0, 0, 1};

    Circle (64) = {271, 270, 27} Plane{0, 0, 1};

    Circle (65) = {27, 270, 269} Plane{0, 0, 1};

    Line (78) = {184, 180};

    Line (85) = {72, 3};

    Line (88) = {8, 5};

    CatmullRom (101) = {285, 293, 290, 295};

    CatmullRom (102) = {285, 292, 291, 294};

    Line (118) = {70, 296};

    Line (126) = {22, 29};

    CatmullRom (138) = {2, 27, 313};

    CatmullRom (139) = {2, 271, 25};

    CatmullRom (140) = {2, 269, 26};

    CatmullRom (141) = {2, 28, 314};

    Line (146) = {296, 296};

    Line (147) = {296, 307};

    Line (149) = {74, 183};

    Line (150) = {75, 182};

    Line (151) = {5, 21};

    CatmullRom (159) = {68, 186};

    CatmullRom (172) = {16, 262, 260, 329, 252};

    CatmullRom (174) = {278, 283, 281, 331, 280};

    CatmullRom (175) = {279, 284, 282, 332, 280};

    CatmullRom (180) = {300, 337, 117};

    CatmullRom (181) = {301, 338, 117};

    Line (188) = {346, 343};

    Line (189) = {345, 344};

    Line (190) = {344, 349};

    Line (191) = {343, 350};

    Line (192) = {345, 348};

    Line (193) = {346, 347};

    Line (194) = {348, 349};

    Line (195) = {350, 347};

    Line (196) = {350, 349};

    Line (197) = {347, 348};

    Line (198) = {307, 316};

    Line (200) = {273, 26};

    Line (201) = {25, 274};

    Line (202) = {313, 275};

    Line (203) = {314, 276};

    CatmullRom (204) = {79, 352, 115};

    CatmullRom (205) = {8, 351, 9};

    CatmullRom (207) = {74, 354, 305, 73};

    CatmullRom (208) = {116, 355, 78};

    CatmullRom (209) = {10, 356, 7};

    CatmullRom (211) = {75, 357, 304, 72};

    Circle (216) = {26, 315, 445} Plane{0, 0, 1};

    Circle (217) = {445, 315, 313} Plane{0, 0, 1};

    Circle (218) = {313, 315, 448} Plane{0, 0, 1};

    Circle (219) = {448, 315, 25} Plane{0, 0, 1};

    Circle (220) = {25, 315, 447} Plane{0, 0, 1};

    Circle (221) = {447, 315, 314} Plane{0, 0, 1};

    Circle (222) = {314, 315, 446} Plane{0, 0, 1};

    Circle (223) = {446, 315, 26} Plane{0, 0, 1};

    CatmullRom (226) = {446, 450, 451};

    CatmullRom (227) = {445, 449, 452};

    CatmullRom (229) = {448, 457, 453};

    CatmullRom (230) = {447, 456, 454};

    Circle (232) = {273, 277, 461} Plane{0, 0, 1};

    Circle (233) = {461, 277, 275} Plane{0, 0, 1};

    Circle (234) = {275, 277, 458} Plane{0, 0, 1};

    Circle (235) = {458, 277, 274} Plane{0, 0, 1};

    Circle (236) = {274, 277, 459} Plane{0, 0, 1};

    Circle (237) = {459, 277, 276} Plane{0, 0, 1};

    Circle (238) = {276, 277, 460} Plane{0, 0, 1};

    Circle (239) = {460, 277, 273} Plane{0, 0, 1};

    CatmullRom (240) = {461, 469, 465};

    CatmullRom (241) = {460, 468, 464};

    Line (243) = {452, 465};

    Line (244) = {451, 464};

    Line (245) = {445, 461};

    Line (246) = {446, 460};

    CatmullRom (248) = {458, 466, 462};

    CatmullRom (249) = {459, 467, 463};

    Line (250) = {448, 458};

    Line (251) = {447, 459};

    Line (252) = {453, 462};

    Line (253) = {454, 463};

    CatmullRom (254) = {252, 330, 259, 261, 15};

    CatmullRom (255) = {1, 359, 15};

    CatmullRom (256) = {15, 18, 278};

    CatmullRom (258) = {321, 33, 186};

    Line (260) = {75, 6};

    CatmullRom (261) = {296, 336, 298, 75};

    Line (262) = {5, 74};

    CatmullRom (263) = {74, 297, 335, 296};

    CatmullRom (268) = {279, 17, 16};

    CatmullRom (269) = {16, 360, 1};

    CatmullRom (276) = {1, 470, 252};

    CatmullRom (277) = {252, 471, 280};

    CatmullRom (278) = {278, 19, 294};

    CatmullRom (280) = {279, 20, 295};

    CatmullRom (286) = {280, 266, 285};

    CatmullRom (298) = {6, 472, 299};

    CatmullRom (299) = {299, 473, 5};

    Line (308) = {343, 344};

    CatmullRom (310) = {22, 474, 343};

    CatmullRom (311) = {344, 475, 21};

    CatmullRom (354) = {480, 265, 299};

    CatmullRom (355) = {4, 477, 478, 480};

    CatmullRom (356) = {480, 479, 476, 3};

    CatmullRom (357) = {5, 353, 4};

    Line (358) = {295, 4};

    CatmullRom (359) = {6, 358, 3};

    Line (360) = {294, 3};

    Line (362) = {480, 285};

    CatmullRom (1000371) = {343, 482, 299};

    CatmullRom (1000372) = {299, 481, 344};

    CatmullRom (2000397) = {70, 487, 117};

    CatmullRom (2000398) = {68, 488, 117};

    CatmullRom (2000399) = {70, 483, 300};

    CatmullRom (2000400) = {300, 486, 68};

    CatmullRom (2000401) = {70, 484, 301};

    CatmullRom (2000402) = {301, 485, 68};

    Line (2000411) = {289, 190};

    Line (2000412) = {289, 72};

    Line (2000413) = {189, 288};

    Line (2000414) = {288, 73};

    CatmullRom (2000415) = {68, 333, 286, 289};

    Line (2000416) = {289, 294};

    CatmullRom (2000417) = {68, 334, 287, 288};

    Line (2000418) = {288, 295};

    CatmullRom (2000419) = {186, 325, 323, 187, 190};

    Line (2000420) = {190, 278};

    CatmullRom (2000421) = {321, 327, 319, 263};

    CatmullRom (2000422) = {322, 362, 263};

    Line (2000423) = {263, 15};

    Line (2000424) = {263, 190};

    CatmullRom (2000435) = {186, 326, 324, 188, 189};

    Line (2000436) = {189, 279};

    CatmullRom (2000443) = {322, 363, 264};

    CatmullRom (2000444) = {264, 320, 328, 321};

    Line (2000445) = {264, 189};

    Line (2000446) = {16, 264};

    Line (2000452) = {322, 1};

    CatmullRom (2000453) = {321, 489, 322};

    CatmullRom (3000466) = {300, 302, 72};

    CatmullRom (3000469) = {301, 303, 73};

    CatmullRom (3000473) = {301, 23502, 74};

    CatmullRom (3000474) = {300, 23503, 75};

    CatmullRom (5000480) = {307, 340, 312, 183};

    Line (5000481) = {183, 21};

    CatmullRom (5000482) = {307, 339, 311, 182};

    Line (5000483) = {182, 22};

    CatmullRom (5000484) = {316, 342, 318, 185};

    Line (5000485) = {185, 23};

    CatmullRom (5000486) = {316, 341, 317, 184};

    Line (5000487) = {184, 24};

    Line (6000508) = {453, 23504};

    Line (6000509) = {23504, 23505};

    Line (6000510) = {23505, 462};

    Line (6000511) = {23505, 463};

    Line (6000512) = {454, 23504};

    Line (6000527) = {23506, 452};

    Line (6000528) = {465, 23507};

    Line (6000529) = {23507, 23506};

    Line (6000530) = {23506, 451};

    Line (6000531) = {23507, 464};

    Line (7000546) = {23, 346};

    Line (7000548) = {346, 458};

    CatmullRom (7000549) = {346, 272, 345};

    Line (7000550) = {345, 458};

    Line (7000551) = {345, 459};

    Line (7000560) = {24, 345};

    Line (7000563) = {275, 185};

    Line (7000566) = {316, 461};

    Line (7000567) = {316, 460};

    Line (7000572) = {184, 276};

    Line (8000575) = {459, 24};

    Line (8000580) = {458, 23};

    CatmullRom (9000586) = {274, 23510, 23505};

    CatmullRom (9000604) = {273, 23508, 23507};

    CatmullRom (9000605) = {26, 23509, 23506};

    CatmullRom (10000622) = {23504, 23511, 25};

    Line Loop (1000289) = {278, -102, -286, -174};

    Surface (289) = {1000289};

    Line Loop (1000291) = {256, 174, -277, 254};

    Surface (291) = {1000291};

    Line Loop (1000293) = {277, -175, 268, 172};

    Surface (293) = {1000293};

    Line Loop (1000295) = {175, 286, 101, -280};

    Surface (295) = {1000295};

    Line Loop (1000321) = {47, -45, -44, 46};

    Surface (321) = {1000321};

    Line Loop (1000323) = {50, 26, -51, -44};

    Plane Surface (323) = {1000323};

    Line Loop (1000325) = {40, 41, 24, -42};

    Plane Surface (325) = {1000325};

    Line Loop (1000327) = {22, -24, -23, 49};

    Surface (327) = {1000327};

    Line Loop (1000329) = {43, 78, -40, -39};

    Surface (329) = {1000329};

    Line Loop (1000344) = {191, 195, -193, 188};

    Plane Surface (344) = {1000344};

    Line Loop (1000346) = {189, 190, -194, -192};

    Plane Surface (346) = {1000346};

    Line Loop (1000348) = {195, 197, 194, -196};

    Plane Surface (348) = {1000348};

    Line Loop (1000350) = {308, 190, -196, -191};

    Surface (350) = {1000350};

    Line Loop (1000352) = {207, -20, -204, -16};

    Surface (352) = {1000352};

    /* Line Loop (1000353) = {207, -20, -204, -16};

    Surface (353) = {1000353};   is obviously duplicated */

    Line Loop (1000364) = {358, 355, 362, 101};

    Surface (364) = {1000364};

    Line Loop (1000366) = {362, 102, 360, -356};

    Surface (366) = {1000366};

    Line Loop (1000368) = {357, 355, 354, 299};

    Surface (368) = {1000368};

    Line Loop (1000370) = {298, -354, 356, -359};

    Surface (370) = {1000370};

    Line Loop (2000374) = {1000372, -308, 1000371}; // CHANGED ORDER

    Surface (1000374) = {2000374};

    Line Loop (2000376) = {1000371, -298, 7, 310};

    Surface (1000376) = {2000376};

    Line Loop (2000378) = {1000372, 311, -151, -299};

    Surface (1000378) = {2000378};

    Line Loop (2000380) = {357, 1, -205, 88};

    Surface (1000380) = {2000380};

    Line Loop (2000382) = {15, 1, -19, 20};

    Plane Surface (1000382) = {2000382};

    Line Loop (2000384) = {262, 16, 18, 88};

    Plane Surface (1000384) = {2000384};

    Line Loop (2000386) = {205, -19, -204, 18};

    Surface (1000386) = {2000386};

    Line Loop (2000388) = {209, -32, -208, 33};

    Surface (1000388) = {2000388};

    Line Loop (2000390) = {359, 4, 209, -5};

    Surface (1000390) = {2000390};

    Line Loop (2000392) = {30, -208, -29, -211};

    Surface (1000392) = {2000392};

    Line Loop (2000394) = {5, -32, -30, 260};

    Plane Surface (1000394) = {2000394};

    Line Loop (2000396) = {33, -4, -85, 29};

    Plane Surface (1000396) = {2000396};

    Line Loop (3000404) = {2000399, 180, -2000397};

    Surface (2000404) = {3000404};

    Line Loop (3000406) = {2000401, 181, -2000397};

    Surface (2000406) = {3000406};

    Line Loop (3000408) = {2000402, 2000398, -181};

    Surface (2000408) = {3000408};

    Line Loop (3000410) = {2000400, 2000398, -180};

    Surface (2000410) = {3000410};

    Line Loop (3000426) = {2000419, -2000424, -2000421, 258};

    Surface (2000426) = {3000426};

    Line Loop (3000428) = {2000424, 2000420, -256, -2000423};

    Surface (2000428) = {3000428};

    Line Loop (3000430) = {2000419, -2000411, -2000415, 159};

    Surface (2000430) = {3000430};

    Line Loop (3000432) = {2000420, 278, -2000416, 2000411};

    Surface (2000432) = {3000432};

    Line Loop (3000434) = {360, -85, -2000412, 2000416};

    Surface (2000434) = {3000434};

    Line Loop (3000438) = {2000417, -2000413, -2000435, -159};

    Surface (2000438) = {3000438};

    Line Loop (3000440) = {2000418, -280, -2000436, 2000413};

    Surface (2000440) = {3000440};

    Line Loop (3000442) = {2000418, 358, -15, -2000414};

    Surface (2000442) = {3000442};

    Line Loop (3000448) = {258, 2000435, -2000445, 2000444};

    Surface (2000448) = {3000448};

    Line Loop (3000450) = {2000445, 2000436, 268, 2000446};

    Surface (2000450) = {3000450};

    Line Loop (3000455) = {2000453, 2000443, 2000444};

    Surface (2000455) = {3000455};

    Line Loop (3000457) = {2000422, -2000421, 2000453};

    Surface (2000457) = {3000457};

    Line Loop (3000459) = {269, -2000452, 2000443, -2000446};

    Surface (2000459) = {3000459};

    Line Loop (3000461) = {255, -2000423, -2000422, 2000452};

    Surface (2000461) = {3000461};

    Line Loop (3000463) = {172, -276, -269};

    Surface (2000463) = {3000463};

    Line Loop (3000465) = {276, 254, -255};

    Surface (2000465) = {3000465};

    Line Loop (4000468) = {3000466, -2000412, -2000415, -2000400};

    Surface (3000468) = {4000468};

    Line Loop (4000471) = {2000417, 2000414, -3000469, 2000402};

    Surface (3000471) = {4000471};

    Line Loop (5000473) = {261, -3000474, -2000399, 118};

    Surface (4000473) = {5000473};

    Line Loop (5000475) = {211, -3000466, 3000474};

    Surface (4000475) = {5000475};

    Line Loop (5000477) = {3000473, 263, -118, 2000401};

    Surface (4000477) = {5000477};

    Line Loop (5000479) = {3000473, 207, -3000469};

    Surface (4000479) = {5000479};

    Line Loop (6000489) = {149, -5000480, -147, -263};

    Surface (5000489) = {6000489};

    Line Loop (6000491) = {262, 149, 5000481, -151};

    Surface (5000491) = {6000491};

    Line Loop (6000493) = {5000483, -7, -260, 150};

    Surface (5000493) = {6000493};

    Line Loop (6000495) = {5000482, -150, -261, 147};

    Surface (5000495) = {6000495};

    Line Loop (6000497) = {47, -5000484, -198, 5000482};

    Surface (5000497) = {6000497};

    Line Loop (6000499) = {198, 5000486, -43, -5000480};

    Surface (5000499) = {6000499};

    Line Loop (6000501) = {23, -41, -78, 5000487};

    Surface (5000501) = {6000501};

    Line Loop (6000503) = {5000481, 22, -42, -39};

    Surface (5000503) = {6000503};

    Line Loop (6000505) = {50, -126, -5000483, -46};

    Surface (5000505) = {6000505};

    Line Loop (6000507) = {5000485, 27, -51, 45};

    Surface (5000507) = {6000507};

    Line Loop (6000509) = {140, 216, 217, -138};

    Surface (5000509) = {6000509};

    Line Loop (6000511) = {141, 222, 223, -140};

    Surface (5000511) = {6000511};

    Line Loop (6000513) = {139, -219, -218, -138};

    Surface (5000513) = {6000513};

    Line Loop (6000515) = {220, 221, -141, 139};

    Surface (5000515) = {6000515};

    Line Loop (6000517) = {217, 202, -233, -245};

    Surface (5000517) = {6000517};

    Line Loop (6000519) = {202, 234, -250, -218};

    Surface (5000519) = {6000519};

    Line Loop (6000521) = {222, 246, -238, -203};

    Surface (5000521) = {6000521};

    Line Loop (6000523) = {237, -203, -221, 251};

    Surface (5000523) = {6000523};

    Line Loop (7000516) = {6000512, 6000509, 6000511, -253};

    Plane Surface (6000516) = {7000516};

    Line Loop (7000518) = {6000508, 6000509, 6000510, -252};

    Plane Surface (6000518) = {7000518};

    Line Loop (7000533) = {6000529, 6000527, 243, 6000528};

    Plane Surface (6000533) = {7000533};

    Line Loop (7000535) = {6000530, 244, -6000531, 6000529};

    Plane Surface (6000535) = {7000535};

    Line Loop (8000553) = {7000549, 7000550, -7000548};

    Surface (7000553) = {8000553};

    Line Loop (8000555) = {7000551, -236, -235, -7000550};

    Surface (7000555) = {8000555};

    Line Loop (8000557) = {7000549, 192, -197, -193};

    Surface (7000557) = {8000557};

    Line Loop (8000559) = {188, -310, -48, 7000546};

    Surface (7000559) = {8000559};

    Line Loop (8000562) = {311, -49, 7000560, 189};

    Surface (7000562) = {8000562};

    Line Loop (8000569) = {7000566, 233, 7000563, -5000484};

    Surface (7000569) = {8000569};

    Line Loop (8000571) = {7000566, -232, -239, -7000567};

    Surface (7000571) = {8000571};

    Line Loop (8000574) = {238, -7000567, 5000486, 7000572};

    Surface (7000574) = {8000574};

    Line Loop (9000577) = {7000560, 7000551, 8000575};

    Surface (8000577) = {9000577};

    Line Loop (9000579) = {5000487, -8000575, 237, -7000572};

    Surface (8000579) = {9000579};

    Line Loop (9000582) = {7000548, 8000580, 7000546};

    Surface (8000582) = {9000582};

    Line Loop (9000584) = {8000580, -5000485, -7000563, 234};

    Surface (8000584) = {9000584};

    Line Loop (10000588) = {230, 6000512};

    //Plane Surface (9000588) = {10000588}; // BAD!!!


    Line Loop (10000607) = {6000528, -9000604, 232, 240};

    Surface (9000607) = {10000607};

    Line Loop (10000608) = {245, 240, -243, -227};

    Surface (9000608) = {10000608};

    Line Loop (10000610) = {6000531, -241, 239, 9000604};

    Surface (9000610) = {10000610};

    Line Loop (10000612) = {6000527, -227, -216, 9000605};

    Surface (9000612) = {10000612};

    Line Loop (10000614) = {6000530, -226, 223, 9000605};

    Surface (9000614) = {10000614};

    Line Loop (10000615) = {244, -241, -246, 226};

    Surface (9000615) = {10000615};

    Line Loop (11000623) = {6000511, -249, -236, 9000586};

    Surface (10000623) = {11000623};

    Line Loop (11000624) = {248, -6000510, -9000586, -235};

    Surface (10000624) = {11000624};

    Line Loop (11000625) = {230, 253, -249, -251};

    Surface (10000625) = {11000625};

    Line Loop (11000626) = {250, 248, -252, -229};

    Surface (10000626) = {11000626};

    Line Loop (11000628) = {6000512, 10000622, 220, 230};

    Surface (10000628) = {11000628};

    Line Loop (11000630) = {6000508, 10000622, -219, 229};

    Surface (10000630) = {11000630};

    Line Loop (12000632) = {26, -27, 48, 126};

    Surface (11000632) = {12000632};
    """

    geometry = "lc1 = " + str(h) + ";\n" + "lc2 = " + str(h) + ";\n" + stub
    return __generate_grid_from_geo_string(geometry)



def cylinder(r1=1.0, r2=1.0, lc=1.0, h= 5.0):
    stub ='''
    a1 = r1 / Sqrt(2);
    a2 = r2 / Sqrt(2);

    // Bottom
    Point(1) = {0, 0, 0, lc};
    Point(3) = { a1,  a1, 0, lc};
    Point(5) = {-a1,  a1, 0, lc};
    Point(7) = {-a1, -a1, 0, lc};
    Point(9) = { a1, -a1, 0, lc};

    // Top
    Point(10) = {0, 0, H, lc};
    Point(12) = { a2,  a2, H, lc};
    Point(14) = {-a2,  a2, H, lc};
    Point(16) = {-a2, -a2, H, lc};
    Point(18) = { a2, -a2, H, lc};

    // Top arcs
    Circle(17) = {12, 10, 14};
    Circle(18) = {14, 10, 16};
    Circle(19) = {16, 10, 18};
    Circle(20) = {18, 10, 12};

    // Bottom arcs
    Circle(21) = {3, 1, 5};
    Circle(22) = {5, 1, 7};
    Circle(23) = {7, 1, 9};
    Circle(24) = {9, 1, 3};

    // Side edges
    Line(25) = {7, 16};
    Line(27) = {9, 18};
    Line(29) = {3, 12};
    Line(31) = {5, 14};

    // Side surfaces
    Line Loop(59) = {27, 20, -29, -24};
    Ruled Surface(60) = {59};

    Line Loop(61) = {29, 17, -31, -21};
    Ruled Surface(62) = {61};

    Line Loop(63) = {31, 18, -25, -22};
    Ruled Surface(64) = {63};

    Line Loop(65) = {25, 19, -27, -23};
    Ruled Surface(66) = {65};

    Physical Surface("sides") = {60, 62, 64, 66};
    '''

    geometry = "lc = " + str(lc) + ";\n"  + "r1 = " + str(r1) + ";\n" + "r2 = " + str(r2) + ";\n" + "H = " + str(h) + ";\n" + stub
    return __generate_grid_from_geo_string(geometry)
