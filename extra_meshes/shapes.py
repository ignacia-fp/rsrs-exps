
import numpy as _np


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
    stub = """
    // the nose
    Point(1) = {8.05141, 0.00278256, -1.36246, cl};
    Point(2) = {7.6903, 4.85811e-06, -1.38746, cl};
    Point(3) = {7.6903, 0.0333377, -1.3819, cl};
    Point(4) = {7.6903, 0.0583376, -1.36246, cl};
    Point(5) = {7.6903, -0.0583282, -1.36246, cl};
    Point(6) = {7.6903, -0.0333283, -1.3819, cl};
    Point(7) = {7.6903, 0.0750047, -1.33746, cl};
    Point(8) = {7.6903, 0.0777827, -1.30135, cl};
    Point(9) = {7.6903, 0.0694486, -1.27913, cl};
    Point(10) = {7.6903, 0.0472267, -1.25968, cl};
    Point(11) = {7.6903, 0.0277829, -1.25135, cl};
    Point(12) = {7.6903, 4.87073e-06, -1.24302, cl};
    Point(13) = {7.6903, -0.0277731, -1.25135, cl};
    Point(14) = {7.6903, -0.0472173, -1.25968, cl};
    Point(15) = {7.6903, -0.0694392, -1.27913, cl};
    Point(16) = {7.6903, -0.0777733, -1.30135, cl};
    Point(17) = {7.6903, -0.0749953, -1.33746, cl};
    Point(18) = {7.34308, -0.127773, -1.41246, cl};
    Point(19) = {7.34308, -0.0722172, -1.43746, cl};
    Point(20) = {7.34308, 0.191672, -1.34302, cl};
    Point(21) = {7.34308, 0.205561, -1.24024, cl};
    Point(22) = {7.34308, 0.127783, -1.41246, cl};
    Point(23) = {7.34308, 0.0722266, -1.43746, cl};
    Point(24) = {7.34308, -0.116662, -1.12913, cl};
    Point(25) = {7.34308, -0.174995, -1.17635, cl};
    Point(26) = {7.34308, -0.0555514, -1.10968, cl};
    Point(27) = {7.34308, 0.00278265, -1.10135, cl};
    Point(28) = {7.34308, 0.0555606, -1.10968, cl};
    Point(29) = {7.34308, 0.116672, -1.12913, cl};
    Point(30) = {7.34308, 0.175005, -1.17635, cl};
    Point(31) = {7.34308, -0.191662, -1.34302, cl};
    Point(32) = {7.34308, -0.205551, -1.24024, cl};
    Point(33) = {7.34308, 0.00278262, -1.44857, cl};
    CatmullRom(1) = {1,12,27};
    CatmullRom(2) = {1,10,29};
    CatmullRom(3) = {1,8,21};
    CatmullRom(4) = {1,4,22};
    CatmullRom(5) = {1,2,33};
    CatmullRom(6) = {1,5,18};
    CatmullRom(7) = {1,16,32};
    CatmullRom(8) = {1,14,24};
    CatmullRom(9) = {27,26,24};
    CatmullRom(10) = {24,25,32};
    CatmullRom(11) = {32,31,18};
    CatmullRom(12) = {18,19,33};
    CatmullRom(13) = {33,23,22};
    CatmullRom(14) = {22,20,21};
    CatmullRom(15) = {21,30,29};
    CatmullRom(16) = {29,28,27};


    // still continuing the nose...
    Point(34) = {6.99586, 0.105561, -1.00135, cl};
    Point(35) = {6.99586, -0.105551, -1.00135, cl};
    Point(36) = {6.99586, 0.00278269, -0.990239, cl};
    Point(37) = {6.99586, 0.319449, -1.19024, cl};
    Point(38) = {6.99586, 0.302783, -1.31246, cl};
    Point(39) = {6.99586, 0.216672, -1.41802, cl};
    Point(40) = {6.99586, 0.261116, -1.07913, cl};
    Point(41) = {6.99586, 0.194449, -1.0319, cl};
    Point(42) = {6.99586, -0.216662, -1.41802, cl};
    Point(43) = {6.99586, -0.302772, -1.31246, cl};
    Point(44) = {6.99586, -0.319439, -1.19024, cl};
    Point(45) = {6.99586, -0.261104, -1.07913, cl};
    Point(46) = {6.99586, -0.194439, -1.0319, cl};
    Point(47) = {6.99586, 0.00278265, -1.48468, cl};
    Point(48) = {6.99586, -0.138884, -1.45968, cl};
    Point(49) = {6.99586, 0.138894, -1.45968, cl};
    Point(50) = {6.64862, 4.67277e-06, -0.88746, cl};
    Point(51) = {6.64862, -0.147217, -0.906904, cl};
    Point(52) = {6.64863, 0.147227, -0.906904, cl};
    Point(53) = {6.64863, 0.247227, -0.940238, cl};
    Point(54) = {6.64863, 0.355561, -1.02635, cl};
    Point(55) = {6.64863, 0.408338, -1.15135, cl};
    Point(56) = {6.64862, 0.388894, -1.28468, cl};
    Point(57) = {6.64862, 0.275005, -1.41802, cl};
    Point(58) = {6.64862, -0.355548, -1.02635, cl};
    Point(59) = {6.64862, -0.247217, -0.940238, cl};
    Point(60) = {6.64862, -0.388885, -1.28468, cl};
    Point(61) = {6.64862, -0.274995, -1.41802, cl};
    Point(62) = {6.64862, -0.408327, -1.15135, cl};
    Point(63) = {6.64862, -0.161106, -1.47357, cl};
    Point(64) = {6.64862, 4.61959e-06, -1.49579, cl};
    Point(65) = {6.64862, 0.161116, -1.47357, cl};

    CatmullRom(17) = {27,36,50};
    CatmullRom(18) = {29,41,53};
    CatmullRom(19) = {21,37,55};
    CatmullRom(20) = {22,39,57};
    CatmullRom(21) = {33,47,64};
    CatmullRom(22) = {18,42,61};
    CatmullRom(23) = {32,44,62};
    CatmullRom(24) = {24,46,59};
    CatmullRom(25) = {50,51,59};
    CatmullRom(26) = {59,58,62};
    CatmullRom(27) = {62,60,61};
    CatmullRom(28) = {61,63,64};
    CatmullRom(29) = {64,65,57};
    CatmullRom(30) = {57,56,55};
    CatmullRom(31) = {55,54,53};
    CatmullRom(32) = {53,52,50};


    // still continuing the nose...
    Point(66) = {6.30141, 4.67277e-06, -0.795793, cl};
    Point(67) = {6.30141, -0.163884, -0.815238, cl};
    Point(68) = {6.30141, 0.163894, -0.815238, cl};
    Point(69) = {6.30141, 0.311116, -0.870793, cl};
    Point(70) = {6.30141, 0.444449, -1.00413, cl};
    Point(71) = {6.30141, 0.455561, -1.27079, cl};
    Point(72) = {6.30141, 0.480561, -1.13746, cl};
    Point(73) = {6.30141, 0.325005, -1.41524, cl};
    Point(74) = {6.30141, -0.311106, -0.870793, cl};
    Point(75) = {6.30141, -0.44444, -1.00413, cl};
    Point(76) = {6.30141, -0.455552, -1.27079, cl};
    Point(77) = {6.30141, -0.324992, -1.41524, cl};
    Point(78) = {6.30141, -0.480548, -1.13746, cl};
    Point(79) = {6.30141, -0.166662, -1.47913, cl};
    Point(80) = {6.30141, 4.67277e-06, -1.49302, cl};
    Point(81) = {6.30141, 0.166672, -1.47913, cl};
    Point(82) = {5.95419, 0.394449, -0.845793, cl};
    Point(83) = {5.95419, 0.500005, -0.995793, cl};
    Point(84) = {5.95419, 0.219449, -0.740238, cl};
    Point(85) = {5.95419, 0.522227, -1.12635, cl};
    Point(86) = {5.95419, 0.497227, -1.25968, cl};
    Point(87) = {5.95419, 0.352783, -1.40413, cl};
    Point(88) = {5.95419, -0.00277331, -0.704127, cl};
    Point(89) = {5.95419, -0.499996, -0.995793, cl};
    Point(90) = {5.95419, -0.394439, -0.845793, cl};
    Point(91) = {5.95419, -0.219439, -0.740238, cl};
    Point(92) = {5.95419, -0.352773, -1.40413, cl};
    Point(93) = {5.95419, -0.497214, -1.25968, cl};
    Point(94) = {5.95419, -0.52222, -1.12635, cl};
    Point(95) = {5.95419, 0.194449, -1.45968, cl};
    Point(96) = {5.95419, 4.68621e-06, -1.47913, cl};
    Point(97) = {5.95419, -0.194439, -1.45968, cl};


    CatmullRom(33) = {50,66,88};
    CatmullRom(34) = {59,74,90};
    CatmullRom(35) = {62,78,94};
    CatmullRom(36) = {53,69,82};
    CatmullRom(37) = {55,72,85};
    CatmullRom(38) = {57,73,87};
    CatmullRom(39) = {64,80,96};
    CatmullRom(40) = {61,77,92};
    CatmullRom(41) = {88,84,82};
    CatmullRom(42) = {82,83,85};
    CatmullRom(43) = {85,86,87};
    CatmullRom(44) = {87,95,96};
    CatmullRom(45) = {96,97,92};
    CatmullRom(46) = {92,93,94};
    CatmullRom(47) = {94,89,90};
    CatmullRom(48) = {90,91,88};

    // still continuing the nose...
    Point(98) = {5.60697, 0.530561, -0.970793, cl};
    Point(99) = {5.60697, 0.455561, -0.820793, cl};
    Point(100) = {5.60697, 0.258338, -0.676349, cl};
    Point(101) = {5.60697, 0.522227, -1.27079, cl};
    Point(102) = {5.60697, 0.388894, -1.38746, cl};
    Point(103) = {5.60697, 0.561116, -1.12357, cl};
    Point(104) = {5.60697, -0.00277327, -0.623571, cl};
    Point(105) = {5.60697, -0.455548, -0.820793, cl};
    Point(106) = {5.60697, -0.258327, -0.676349, cl};
    Point(107) = {5.60697, -0.530556, -0.970793, cl};
    Point(108) = {5.60697, -0.522222, -1.27079, cl};
    Point(109) = {5.60697, -0.388885, -1.38746, cl};
    Point(110) = {5.60697, -0.561111, -1.12357, cl};
    Point(111) = {5.60697, 0.00278278, -1.46246, cl};
    Point(112) = {5.60697, 0.216672, -1.44579, cl};
    Point(113) = {5.60697, -0.216662, -1.44579, cl};
    Point(114) = {5.25974, 0.00278257, -0.545793, cl};
    Point(115) = {5.25974, 0.575005, -0.973572, cl};
    Point(116) = {5.25974, 0.516672, -0.826349, cl};
    Point(117) = {5.25974, 0.330561, -0.634682, cl};
    Point(118) = {5.25974, 0.591672, -1.12357, cl};
    Point(119) = {5.25974, 0.547227, -1.28468, cl};
    Point(120) = {5.25974, 0.427783, -1.37913, cl};
    Point(121) = {5.25974, 0.188894, -1.44024, cl};
    Point(122) = {5.25974, -0.516667, -0.826349, cl};
    Point(123) = {5.25974, -0.330551, -0.634682, cl};
    Point(124) = {5.25974, -0.575, -0.973571, cl};
    Point(125) = {5.25974, -0.591667, -1.12357, cl};
    Point(126) = {5.25974, -0.427774, -1.37913, cl};
    Point(127) = {5.25974, -0.547222, -1.28468, cl};
    Point(128) = {5.25974, -0.188884, -1.44024, cl};
    Point(129) = {5.25974, 0.00278282, -1.45135, cl};

    CatmullRom(49) = {88,104,114};
    CatmullRom(50) = {82,99,116};
    CatmullRom(51) = {85,103,118};
    CatmullRom(52) = {87,102,120};
    CatmullRom(53) = {96,111,129};
    CatmullRom(54) = {92,109,126};
    CatmullRom(55) = {94,110,125};
    CatmullRom(56) = {90,105,122};
    CatmullRom(57) = {114,123,122};
    CatmullRom(58) = {122,124,125};
    CatmullRom(59) = {125,127,126};
    CatmullRom(60) = {126,128,129};
    CatmullRom(61) = {129,121,120};
    CatmullRom(62) = {120,119,118};
    CatmullRom(63) = {118,115,116};
    CatmullRom(64) = {114,117,116};


    Point(130) = {4.91252, 0.233338, -0.470793, cl};
    Point(131) = {4.91252, 0.188894, -0.423571, cl};
    Point(132) = {4.91252, 0.269449, -0.529127, cl};
    Point(133) = {4.91252, -0.00277318, -0.348571, cl};
    Point(134) = {4.91252, -0.0972171, -0.365238, cl};
    Point(135) = {4.91252, 0.0972266, -0.365238, cl};
    Point(136) = {4.91252, 0.605561, -0.943016, cl};
    Point(137) = {4.91252, 0.550005, -0.804127, cl};
    Point(138) = {4.91252, 0.447227, -0.654127, cl};
    Point(139) = {4.91252, 0.663894, -1.11802, cl};
    Point(140) = {4.91252, 0.625005, -1.09579, cl};
    Point(141) = {4.91252, 0.627783, -1.12913, cl};
    Point(142) = {4.91252, 0.447227, -1.37079, cl};
    Point(143) = {4.91252, 0.575005, -1.2819, cl};
    Point(144) = {4.91252, 0.169449, -1.4319, cl};
    Point(145) = {4.91252, 0.00278285, -1.43746, cl};
    Point(146) = {4.91252, -0.26944, -0.529127, cl};
    Point(147) = {4.91252, -0.233328, -0.470793, cl};
    Point(148) = {4.91252, -0.188884, -0.423571, cl};
    Point(149) = {4.91252, -0.447218, -0.654127, cl};
    Point(150) = {4.91252, -0.549997, -0.804127, cl};
    Point(151) = {4.91252, -0.605556, -0.943016, cl};
    Point(152) = {4.91252, -0.663889, -1.11802, cl};
    Point(153) = {4.91252, -0.627778, -1.12913, cl};
    Point(154) = {4.91252, -0.625, -1.09579, cl};
    Point(155) = {4.91252, -0.574999, -1.2819, cl};
    Point(156) = {4.91252, -0.447218, -1.37079, cl};
    Point(157) = {4.91252, -0.169439, -1.4319, cl};
    Point(158) = {4.5653, 0.00278267, -0.143016, cl};
    Point(159) = {4.5653, 0.141672, -0.173571, cl};
    Point(160) = {4.5653, 0.333338, -0.404127, cl};
    Point(161) = {4.5653, 0.352783, -0.518016, cl};
    Point(162) = {4.5653, 0.461116, -0.609682, cl};
    Point(163) = {4.5653, 0.263894, -0.273571, cl};
    Point(164) = {4.5653, -0.141662, -0.173571, cl};
    Point(165) = {4.5653, 0.611116, -0.870793, cl};
    Point(166) = {4.5653, 0.647227, -1.03468, cl};
    Point(167) = {4.5653, 0.552783, -0.73746, cl};
    Point(168) = {4.5653, 0.725005, -1.11246, cl};
    Point(169) = {4.5653, 0.641672, -1.16802, cl};
    Point(170) = {4.5653, 0.591672, -1.27635, cl};
    Point(171) = {4.5653, 0.447227, -1.36802, cl};
    Point(172) = {4.5653, 0.205561, -1.41524, cl};
    Point(173) = {4.5653, -0.00277324, -1.42635, cl};
    Point(174) = {4.5653, -0.461103, -0.609682, cl};
    Point(175) = {4.5653, -0.352774, -0.518016, cl};
    Point(176) = {4.5653, -0.333328, -0.404127, cl};
    Point(177) = {4.5653, -0.263884, -0.273571, cl};
    Point(178) = {4.5653, -0.647222, -1.03468, cl};
    Point(179) = {4.5653, -0.611108, -0.870793, cl};
    Point(180) = {4.5653, -0.552775, -0.73746, cl};
    Point(181) = {4.5653, -0.205551, -1.41524, cl};
    Point(182) = {4.5653, -0.447218, -1.36802, cl};
    Point(183) = {4.5653, -0.591664, -1.27635, cl};
    Point(184) = {4.5653, -0.641667, -1.16802, cl};
    Point(185) = {4.5653, -0.725, -1.11246, cl};

    CatmullRom(65) = {114,133,158};
    CatmullRom(66) = {118,140,166};
    CatmullRom(67) = {118,141,169};
    CatmullRom(68) = {118,139,168};
    CatmullRom(69) = {122,150,180};
    CatmullRom(70) = {116,137,167};
    CatmullRom(71) = {114,132,161};
    CatmullRom(72) = {114,146,175};
    CatmullRom(73) = {125,154,178};
    CatmullRom(74) = {125,152,185};
    CatmullRom(75) = {125,153,184};
    CatmullRom(76) = {126,156,182};
    CatmullRom(77) = {129,145,173};
    CatmullRom(78) = {120,142,171};
    CatmullRom(79) = {158,159,163,160,161};
    CatmullRom(80) = {158,164,177,176,175};
    CatmullRom(81) = {161,162,167};
    CatmullRom(82) = {167,165,166};
    CatmullRom(83) = {175,174,180};
    CatmullRom(84) = {180,179,178};
    CatmullRom(85) = {184,183,182};
    CatmullRom(86) = {182,181,173};
    CatmullRom(87) = {173,172,171};
    CatmullRom(88) = {171,170,169};


    // the hood: contd.....
    Point(186) = {4.21807, 0.197227, -0.031904, cl};
    Point(187) = {4.21807, 4.6537e-06, 0.023651, cl};
    Point(188) = {4.21807, 0.330561, -0.156904, cl};
    Point(189) = {4.21807, 0.391672, -0.501349, cl};
    Point(190) = {4.21807, 0.500005, -0.615238, cl};
    Point(191) = {4.21807, 0.397227, -0.326349, cl};
    Point(192) = {4.21807, 0.630561, -0.854127, cl};
    Point(193) = {4.21807, 0.658338, -0.993017, cl};
    Point(194) = {4.21807, 0.580561, -0.740238, cl};
    Point(195) = {4.21807, 0.752783, -1.10968, cl};
    Point(196) = {4.21807, 0.666672, -1.15968, cl};
    Point(197) = {4.21807, 0.597227, -1.27079, cl};
    Point(198) = {4.21807, 0.472227, -1.34857, cl};
    Point(199) = {4.21807, 0.230561, -1.39857, cl};
    Point(200) = {4.21807, 4.85494e-06, -1.41246, cl};
    Point(201) = {4.21807, -0.330549, -0.156904, cl};
    Point(202) = {4.21807, -0.197217, -0.031904, cl};
    Point(203) = {4.21807, -0.499992, -0.615238, cl};
    Point(204) = {4.21807, -0.391659, -0.501349, cl};
    Point(205) = {4.21807, -0.397214, -0.326349, cl};
    Point(206) = {4.21807, -0.658333, -0.993016, cl};
    Point(207) = {4.21807, -0.630554, -0.854127, cl};
    Point(208) = {4.21807, -0.580553, -0.740238, cl};
    Point(209) = {4.21807, -0.472214, -1.34857, cl};
    Point(210) = {4.21807, -0.597222, -1.27079, cl};
    Point(211) = {4.21807, -0.230551, -1.39857, cl};
    Point(212) = {4.21807, -0.666665, -1.15968, cl};
    Point(213) = {4.21807, -0.752778, -1.10968, cl};
    Point(214) = {3.87086, -0.802778, -1.10413, cl};
    Point(215) = {3.87086, -0.680556, -1.16802, cl};
    Point(216) = {3.87086, -0.274995, -1.37913, cl};
    Point(217) = {3.87086, -0.480548, -1.32635, cl};
    Point(218) = {3.87086, -0.608333, -1.25135, cl};
    Point(219) = {3.87086, -0.588889, -0.709682, cl};
    Point(220) = {3.87086, -0.697222, -0.998572, cl};
    Point(221) = {3.87086, -0.641666, -0.843016, cl};
    Point(222) = {3.87086, -0.42777, -0.273571, cl};
    Point(223) = {3.87086, -0.499995, -0.568016, cl};
    Point(224) = {3.87086, -0.405549, -0.473571, cl};
    Point(225) = {3.87086, -0.358325, -0.079127, cl};
    Point(226) = {3.87086, -0.199995, 0.06254, cl};
    Point(227) = {3.87086, 0.275005, -1.37913, cl};
    Point(228) = {3.87086, 0.00278262, -1.39302, cl};
    Point(229) = {3.87086, 0.608338, -1.25135, cl};
    Point(230) = {3.87086, 0.480561, -1.32635, cl};
    Point(231) = {3.87086, 0.802783, -1.10413, cl};
    Point(232) = {3.87086, 0.680561, -1.16802, cl};
    Point(233) = {3.87086, 0.588894, -0.709682, cl};
    Point(234) = {3.87086, 0.641672, -0.843016, cl};
    Point(235) = {3.87086, 0.697227, -0.998572, cl};
    Point(236) = {3.87086, 0.427783, -0.273571, cl};
    Point(237) = {3.87086, 0.405561, -0.473571, cl};
    Point(238) = {3.87086, 0.500005, -0.568016, cl};
    Point(239) = {3.87086, 0.358338, -0.0791269, cl};
    Point(240) = {3.87086, 4.69405e-06, 0.11254, cl};
    Point(241) = {3.87086, 0.200005, 0.0625401, cl};
    CatmullRom(89) = {158,187,240};
    CatmullRom(90) = {161,189,237};
    CatmullRom(91) = {167,194,233};
    CatmullRom(92) = {166,193,235};
    CatmullRom(93) = {168,195,231};
    CatmullRom(94) = {169,196,232};
    CatmullRom(95) = {171,198,230};
    CatmullRom(96) = {173,200,228};
    CatmullRom(97) = {182,209,217};
    CatmullRom(98) = {185,213,214};
    CatmullRom(99) = {184,212,215};
    CatmullRom(100) = {178,206,220};
    CatmullRom(101) = {180,208,219};
    CatmullRom(102) = {175,204,224};
    CatmullRom(103) = {240,226,225,222,224};
    CatmullRom(104) = {240,241,239,236,237};
    CatmullRom(105) = {237,238,233};
    CatmullRom(106) = {233,234,235};
    CatmullRom(107) = {232,229,230};
    CatmullRom(108) = {230,227,228};
    CatmullRom(109) = {228,216,217};
    CatmullRom(110) = {217,218,215};
    CatmullRom(111) = {220,221,219};
    CatmullRom(112) = {219,223,224};


    // the hood reloaded
    Point(242) = {3.52363, -0.855556, -1.10135, cl};
    Point(243) = {3.52363, -0.722222, -1.14579, cl};
    Point(244) = {3.52363, -0.619444, -1.22635, cl};
    Point(245) = {3.52363, -0.449995, -1.31524, cl};
    Point(246) = {3.52363, -0.233328, -1.36802, cl};
    Point(247) = {3.52363, -0.727778, -0.984683, cl};
    Point(248) = {3.52363, -0.602777, -0.690238, cl};
    Point(249) = {3.52363, -0.655556, -0.831904, cl};
    Point(250) = {3.52363, -0.419436, -0.245793, cl};
    Point(251) = {3.52363, -0.402774, -0.431904, cl};
    Point(252) = {3.52363, -0.511112, -0.540238, cl};
    Point(253) = {3.52363, -0.219439, 0.084762, cl};
    Point(254) = {3.52363, -0.355551, -0.043016, cl};
    Point(255) = {3.52363, 4.59598e-06, -1.3819, cl};
    Point(256) = {3.52363, 0.233338, -1.36802, cl};
    Point(257) = {3.52363, 0.450005, -1.31524, cl};
    Point(258) = {3.52363, 0.619449, -1.22635, cl};
    Point(259) = {3.52363, 0.722227, -1.14579, cl};
    Point(260) = {3.52363, 0.855561, -1.10135, cl};
    Point(261) = {3.52363, 0.727783, -0.984682, cl};
    Point(262) = {3.52363, 0.655561, -0.831904, cl};
    Point(263) = {3.52363, 0.602783, -0.690238, cl};
    Point(264) = {3.52363, 0.511116, -0.540238, cl};
    Point(265) = {3.52363, 0.402783, -0.431904, cl};
    Point(266) = {3.52363, 0.419449, -0.245793, cl};
    Point(267) = {3.52363, 0.355561, -0.0430159, cl};
    Point(268) = {3.52363, 0.219449, 0.084762, cl};
    Point(269) = {3.52363, 4.7293e-06, 0.143096, cl};
    Point(270) = {3.32919, -0.419438, -1.92357, cl};
    Point(271) = {3.32919, -0.199995, -1.99579, cl};
    Point(272) = {3.32919, -0.625, -1.79857, cl};
    Point(273) = {3.32919, -0.255548, -1.47079, cl};
    Point(274) = {3.32919, -0.449993, -1.44857, cl};
    Point(275) = {3.32919, -0.6, -1.47635, cl};
    Point(276) = {3.32919, -0.694444, -1.62635, cl};
    Point(277) = {3.32919, 0.419449, -1.92357, cl};
    Point(278) = {3.32919, 4.55886e-06, -2.01524, cl};
    Point(279) = {3.32919, 0.200005, -1.99579, cl};
    Point(280) = {3.32919, 0.255561, -1.47079, cl};
    Point(281) = {3.32919, 4.60621e-06, -1.47357, cl};
    Point(282) = {3.32919, 0.625005, -1.79857, cl};
    Point(283) = {3.32919, 0.694449, -1.62635, cl};
    Point(284) = {3.32919, 0.600005, -1.47635, cl};
    Point(285) = {3.32919, 0.450005, -1.44857, cl};
    Point(286) = {3.32919, -0.152773, -1.36802, cl};
    Point(287) = {3.32919, -0.730556, -1.14857, cl};
    Point(288) = {3.32919, -0.880556, -1.09024, cl};
    Point(289) = {3.32919, -0.761111, -1.12913, cl};
    Point(290) = {3.32919, -0.597222, -1.2319, cl};
    Point(291) = {3.32919, -0.644444, -1.19857, cl};
    Point(292) = {3.32919, -0.461103, -1.29857, cl};
    Point(293) = {3.32919, -0.311107, -1.34579, cl};
    Point(294) = {3.32919, -0.244439, -1.3569, cl};
    Point(295) = {3.32919, -0.449996, -1.30413, cl};
    Point(296) = {3.32919, -0.747222, -0.968016, cl};
    Point(297) = {3.32919, -0.777778, -1.01524, cl};
    Point(298) = {3.32919, -0.708333, -0.909682, cl};
    Point(299) = {3.32919, -0.666667, -0.823571, cl};
    Point(300) = {3.32919, -0.591666, -0.659682, cl};
    Point(301) = {3.32919, -0.611111, -0.693016, cl};
    Point(302) = {3.32919, -0.655556, -0.801349, cl};
    Point(304) = {3.32919, -0.505553, -0.523571, cl};
    Point(305) = {3.32919, -0.477772, -0.48746, cl};
    Point(306) = {3.32919, -0.550001, -0.584682, cl};
    Point(307) = {3.32919, -0.411106, -0.245794, cl};
    Point(308) = {3.32919, -0.391662, -0.404127, cl};
    Point(311) = {3.32919, -0.333328, -0.0291272, cl};
    Point(312) = {3.32919, -0.2166615, 0.08198455, cl};
    Point(313) = {3.32919, 0.152783, -1.36802, cl};
    Point(314) = {3.32919, 0.311116, -1.34579, cl};
    Point(315) = {3.32919, 0.244449, -1.3569, cl};
    Point(316) = {3.32919, 4.61422e-06, -1.3819, cl};
    Point(317) = {3.32919, 0.461116, -1.29857, cl};
    Point(318) = {3.32919, 0.450005, -1.30413, cl};
    Point(319) = {3.32919, 0.597227, -1.2319, cl};
    Point(320) = {3.32919, 0.730561, -1.14857, cl};
    Point(321) = {3.32919, 0.644449, -1.19857, cl};
    Point(322) = {3.32919, 0.880561, -1.09024, cl};
    Point(323) = {3.32919, 0.761116, -1.12913, cl};
    Point(324) = {3.32919, 0.747227, -0.968016, cl};
    Point(325) = {3.32919, 0.591672, -0.659682, cl};
    Point(326) = {3.32919, 0.666672, -0.823571, cl};
    Point(327) = {3.32919, 0.777783, -1.01524, cl};
    Point(328) = {3.32919, 0.708338, -0.909682, cl};
    Point(329) = {3.32919, 0.655561, -0.801349, cl};
    Point(330) = {3.32919, 0.611116, -0.693016, cl};
    Point(331) = {3.32919, 0.505561, -0.523571, cl};
    Point(333) = {3.32919, 0.550005, -0.584682, cl};
    Point(334) = {3.32919, 0.477783, -0.48746, cl};
    Point(335) = {3.32919, 0.391672, -0.404127, cl};
    Point(336) = {3.32919, 0.411116, -0.245793, cl};
    Point(338) = {3.32919, 0.333338, -0.0291271, cl};
    Point(340) = {3.32919, -0.00277331, 0.148651, cl};
    Point(341) = {3.32919, 0.2166715, 0.08198455, cl};
    CatmullRom(113) = {228,255,316};
    CatmullRom(114) = {231,260,322};
    CatmullRom(115) = {214,242,288};
    CatmullRom(116) = {240,269,340};
    CatmullRom(117) = {237,265,335};
    CatmullRom(118) = {224,251,308};
    CatmullRom(119) = {340,312,311,307,308};
    CatmullRom(120) = {340,341,338,336,335};


    // the hood reloaded
    Point(342) = {3.17641, -0.0249953, -1.47357, cl};
    Point(343) = {3.17641, -0.463885, -1.92357, cl};
    Point(344) = {3.17641, -0.266662, -2.0069, cl};
    Point(345) = {3.17641, -0.644443, -1.79302, cl};
    Point(346) = {3.17641, -0.216662, -1.46524, cl};
    Point(347) = {3.17641, -0.438885, -1.44579, cl};
    Point(348) = {3.17641, -0.624997, -1.47913, cl};
    Point(349) = {3.17641, -0.705556, -1.61802, cl};
    Point(350) = {3.17641, 0.463894, -1.92357, cl};
    Point(351) = {3.17641, 0.266672, -2.0069, cl};
    Point(352) = {3.17641, 0.00278263, -2.04302, cl};
    Point(353) = {3.17641, 0.216672, -1.46524, cl};
    Point(354) = {3.17641, 0.0250046, -1.47357, cl};
    Point(355) = {3.17641, 0.644449, -1.79302, cl};
    Point(356) = {3.17641, 0.705561, -1.61802, cl};
    Point(357) = {3.17641, 0.625005, -1.47913, cl};
    Point(358) = {3.17641, 0.438894, -1.44579, cl};
    Point(359) = {3.17641, -0.0277734, -1.37357, cl};
    Point(360) = {3.17641, -0.902778, -1.09302, cl};
    Point(361) = {3.17641, -0.752778, -1.13468, cl};
    Point(362) = {3.17641, -0.622223, -1.20968, cl};
    Point(363) = {3.17641, -0.480552, -1.28468, cl};
    Point(364) = {3.17641, -0.338883, -1.33468, cl};
    Point(365) = {3.17641, -0.188884, -1.36246, cl};
    Point(366) = {3.17641, -0.8, -1.01802, cl};
    Point(367) = {3.17641, -0.725, -0.915238, cl};
    Point(368) = {3.17641, -0.616667, -0.681904, cl};
    Point(369) = {3.17641, -0.663889, -0.801349, cl};
    Point(370) = {3.17641, -0.55, -0.565238, cl};
    Point(371) = {3.17641, -0.474995, -0.470793, cl};
    Point(372) = {3.17641, -0.380548, -0.379127, cl};
    Point(373) = {3.17641, -0.394436, -0.226349, cl};
    Point(374) = {3.17641, -0.233328, 0.06254, cl};
    Point(375) = {3.17641, -0.358329, -0.084682, cl};
    Point(376) = {3.17641, 0.338894, -1.33468, cl};
    Point(377) = {3.17641, 0.0277826, -1.37357, cl};
    Point(378) = {3.17641, 0.188894, -1.36246, cl};
    Point(379) = {3.17641, 0.480561, -1.28468, cl};
    Point(380) = {3.17641, 0.902783, -1.09302, cl};
    Point(381) = {3.17641, 0.622227, -1.20968, cl};
    Point(382) = {3.17641, 0.752782, -1.13468, cl};
    Point(383) = {3.17641, 0.725004, -0.915238, cl};
    Point(384) = {3.17641, 0.800005, -1.01802, cl};
    Point(385) = {3.17641, 0.663894, -0.801349, cl};
    Point(386) = {3.17641, 0.616672, -0.681904, cl};
    Point(387) = {3.17641, 0.475005, -0.470793, cl};
    Point(388) = {3.17641, 0.550005, -0.565238, cl};
    Point(389) = {3.17641, 0.394449, -0.226349, cl};
    Point(390) = {3.17641, 0.380561, -0.379127, cl};
    Point(391) = {3.17641, 0.358338, -0.084682, cl};
    Point(392) = {3.17641, 0.233338, 0.0625401, cl};
    Point(393) = {3.17641, 4.76188e-06, 0.143096, cl};
    Point(394) = {2.82919, -0.116662, -1.4569, cl};
    Point(395) = {2.82919, -0.283327, -2.04024, cl};
    Point(396) = {2.82919, -0.511111, -1.93468, cl};
    Point(397) = {2.82919, -0.680554, -1.79579, cl};
    Point(398) = {2.82919, -0.299994, -1.45413, cl};
    Point(399) = {2.82919, -0.480548, -1.44857, cl};
    Point(400) = {2.82919, -0.727778, -1.61246, cl};
    Point(401) = {2.82919, -0.655556, -1.49579, cl};
    Point(402) = {2.82919, 0.511116, -1.93468, cl};
    Point(403) = {2.82919, 0.283338, -2.04024, cl};
    Point(404) = {2.82919, 4.59995e-06, -2.0819, cl};
    Point(405) = {2.82919, 0.116672, -1.4569, cl};
    Point(406) = {2.82919, 0.300005, -1.45413, cl};
    Point(407) = {2.82919, 0.680561, -1.79579, cl};
    Point(408) = {2.82919, 0.727783, -1.61246, cl};
    Point(409) = {2.82919, 0.480561, -1.44857, cl};
    Point(410) = {2.82919, 0.655561, -1.49579, cl};
    Point(411) = {2.82919, -0.119439, -1.34857, cl};
    Point(412) = {2.82919, -0.811111, -1.12079, cl};
    Point(413) = {2.82919, -0.958333, -1.09024, cl};
    Point(414) = {2.82919, -0.669442, -1.17079, cl};
    Point(415) = {2.82919, -0.538889, -1.24024, cl};
    Point(416) = {2.82919, -0.263881, -1.32635, cl};
    Point(417) = {2.82919, -0.40555, -1.29302, cl};
    Point(418) = {2.82919, -0.738888, -0.884682, cl};
    Point(419) = {2.82919, -0.833333, -0.995793, cl};
    Point(420) = {2.82919, -0.669444, -0.765238, cl};
    Point(421) = {2.82919, -0.608333, -0.63746, cl};
    Point(422) = {2.82919, -0.347214, -0.315238, cl};
    Point(423) = {2.82919, -0.541667, -0.515238, cl};
    Point(424) = {2.82919, -0.449995, -0.406904, cl};
    Point(425) = {2.82919, -0.347218, -0.176349, cl};
    Point(426) = {2.82919, -0.299993, -0.054127, cl};
    Point(427) = {2.82919, -0.211106, 0.045873, cl};
    Point(428) = {2.82919, 0.958337, -1.09024, cl};
    Point(429) = {2.82919, 0.405561, -1.29302, cl};
    Point(430) = {2.82919, 0.263894, -1.32635, cl};
    Point(431) = {2.82919, 0.119449, -1.34857, cl};
    Point(432) = {2.82919, 0.538894, -1.24024, cl};
    Point(433) = {2.82919, 0.811116, -1.12079, cl};
    Point(434) = {2.82919, 0.669448, -1.17079, cl};
    Point(435) = {2.82919, 0.738894, -0.884682, cl};
    Point(436) = {2.82919, 0.833337, -0.995793, cl};
    Point(437) = {2.82919, 0.608338, -0.63746, cl};
    Point(438) = {2.82919, 0.669449, -0.765238, cl};
    Point(439) = {2.82919, 0.347227, -0.315238, cl};
    Point(440) = {2.82919, 0.450005, -0.406904, cl};
    Point(441) = {2.82919, 0.541672, -0.515238, cl};
    Point(442) = {2.82919, 0.347227, -0.176349, cl};
    Point(443) = {2.82919, 0.300005, -0.054127, cl};
    Point(444) = {2.82919, 4.79228e-06, 0.118096, cl};
    Point(445) = {2.82919, 0.211116, 0.0458731, cl};

    // the hood reloaded
    Point(446) = {2.48197, -0.558333, -1.92913, cl};
    Point(447) = {2.48197, -0.255549, -2.07357, cl};
    Point(448) = {2.48197, -0.577778, -1.45135, cl};
    Point(449) = {2.48197, -0.708333, -1.5319, cl};
    Point(450) = {2.48197, -0.75, -1.65413, cl};
    Point(451) = {2.48197, -0.708333, -1.78468, cl};
    Point(452) = {2.48197, 0.558338, -1.92913, cl};
    Point(453) = {2.48197, 4.6301e-06, -2.10968, cl};
    Point(454) = {2.48197, 0.255561, -2.07357, cl};
    Point(455) = {2.48197, 0.708338, -1.78468, cl};
    Point(456) = {2.48197, 0.750005, -1.65413, cl};
    Point(457) = {2.48197, 0.708338, -1.5319, cl};
    Point(458) = {2.48197, 0.577783, -1.45135, cl};
    Point(459) = {2.48197, -0.627777, -1.18468, cl};
    Point(460) = {2.48197, -1.02222, -1.09024, cl};
    Point(461) = {2.48197, -0.886112, -1.10968, cl};
    Point(462) = {2.48197, -0.755556, -1.14024, cl};
    Point(463) = {2.48197, -0.497214, -1.23468, cl};
    Point(464) = {2.48197, -0.366659, -1.27913, cl};
    Point(465) = {2.48197, -0.238884, -1.3069, cl};
    Point(466) = {2.48197, -0.238884, -1.41802, cl};
    Point(467) = {2.48197, -0.424996, -1.4319, cl};
    Point(468) = {2.48197, -0.683333, -0.751349, cl};
    Point(469) = {2.48197, -0.777778, -0.881904, cl};
    Point(470) = {2.48197, -0.880556, -0.998571, cl};
    Point(471) = {2.48197, -0.294437, -0.26246, cl};
    Point(472) = {2.48197, -0.424995, -0.356904, cl};
    Point(473) = {2.48197, -0.530554, -0.473571, cl};
    Point(474) = {2.48197, -0.611111, -0.609682, cl};
    Point(475) = {2.48197, -0.25277, -0.0680161, cl};
    Point(476) = {2.48197, -0.288882, -0.148571, cl};
    Point(477) = {2.48197, 1.02223, -1.09024, cl};
    Point(478) = {2.48197, 0.886116, -1.10968, cl};
    Point(479) = {2.48197, 0.425005, -1.4319, cl};
    Point(480) = {2.48197, 0.238894, -1.41802, cl};
    Point(481) = {2.48197, 0.238894, -1.3069, cl};
    Point(482) = {2.48197, 0.366672, -1.27913, cl};
    Point(483) = {2.48197, 0.497227, -1.23468, cl};
    Point(484) = {2.48197, 0.627783, -1.18468, cl};
    Point(485) = {2.48197, 0.755561, -1.14024, cl};
    Point(486) = {2.48197, 0.880561, -0.998572, cl};
    Point(487) = {2.48197, 0.777782, -0.881904, cl};
    Point(488) = {2.48197, 0.683338, -0.751349, cl};
    Point(489) = {2.48197, 0.294449, -0.26246, cl};
    Point(490) = {2.48197, 0.425005, -0.356904, cl};
    Point(491) = {2.48197, 0.611116, -0.609682, cl};
    Point(492) = {2.48197, 0.530561, -0.473571, cl};
    Point(493) = {2.48197, 0.163894, 0.018096, cl};
    Point(494) = {2.48197, 0.00278288, 0.070873, cl};
    Point(495) = {2.48197, -0.163884, 0.018096, cl};
    Point(496) = {2.48197, 0.288894, -0.148571, cl};
    Point(497) = {2.48197, 0.252783, -0.068016, cl};
    Point(498) = {2.13473, -0.711111, -1.8069, cl};
    Point(499) = {2.13473, -0.508333, -1.22079, cl};
    Point(500) = {2.13473, -0.394437, -1.39024, cl};
    Point(501) = {2.13473, -0.8, -0.870793, cl};
    Point(502) = {2.13473, 1.10556, -1.09024, cl};
    Point(503) = {2.13473, 0.388894, -0.306904, cl};
    Point(504) = {2.13473, 0.511116, -0.43746, cl};
    Point(505) = {2.13473, 0.605561, -0.584682, cl};
    Point(506) = {2.13473, 0.127783, -0.0291271, cl};
    Point(507) = {2.13473, 0.194449, -0.093016, cl};
    Point(508) = {2.13473, 0.230561, -0.154127, cl};
    Point(509) = {2.13473, -0.524995, -1.97079, cl};
    Point(510) = {2.13473, -0.252773, -2.09579, cl};
    Point(511) = {2.13473, -0.675, -1.48746, cl};
    Point(512) = {2.13473, -0.744444, -1.55968, cl};
    Point(513) = {2.13473, -0.763889, -1.6819, cl};
    Point(514) = {2.13473, 0.525005, -1.97079, cl};
    Point(515) = {2.13473, 4.66147e-06, -2.12357, cl};
    Point(516) = {2.13473, 0.252783, -2.09579, cl};
    Point(517) = {2.13473, 0.763894, -1.6819, cl};
    Point(518) = {2.13473, 0.711116, -1.80691, cl};
    Point(519) = {2.13473, 0.744449, -1.55968, cl};
    Point(520) = {2.13473, 0.675005, -1.48746, cl};
    Point(521) = {2.13473, -1.10556, -1.09024, cl};
    Point(522) = {2.13473, -0.986111, -1.10413, cl};
    Point(523) = {2.13473, -0.869444, -1.12079, cl};
    Point(524) = {2.13473, -0.75, -1.14579, cl};
    Point(525) = {2.13473, -0.627776, -1.17635, cl};
    Point(526) = {2.13473, -0.388882, -1.25135, cl};
    Point(527) = {2.13473, -0.549998, -1.42357, cl};
    Point(528) = {2.13473, -0.697222, -0.734682, cl};
    Point(529) = {2.13473, -0.922222, -0.990239, cl};
    Point(530) = {2.13473, -0.241662, -0.215238, cl};
    Point(531) = {2.13473, -0.388884, -0.306904, cl};
    Point(532) = {2.13473, -0.605556, -0.584682, cl};
    Point(533) = {2.13473, -0.511111, -0.43746, cl};
    Point(534) = {2.13473, -0.230551, -0.154127, cl};
    Point(535) = {2.13473, -0.194439, -0.093016, cl};
    Point(536) = {2.13473, 0.388894, -1.25135, cl};
    Point(537) = {2.13473, 0.550005, -1.42357, cl};
    Point(538) = {2.13473, 0.394449, -1.39024, cl};
    Point(539) = {2.13473, 0.869449, -1.12079, cl};
    Point(540) = {2.13473, 0.508338, -1.22079, cl};
    Point(541) = {2.13473, 0.627783, -1.17635, cl};
    Point(542) = {2.13473, 0.750005, -1.14579, cl};
    Point(543) = {2.13473, 0.697227, -0.734682, cl};
    Point(544) = {2.13473, 0.800005, -0.870793, cl};
    Point(545) = {2.13473, 0.983338, -1.10135, cl};
    Point(546) = {2.13473, 0.922227, -0.990238, cl};
    Point(547) = {2.13473, 0.00278258, 0.00420702, cl};
    Point(548) = {2.13473, -0.127773, -0.0291271, cl};
    Point(549) = {2.13473, 0.241672, -0.215238, cl};


    // the hood reloaded
    Point(550) = {1.78752, -0.644442, -1.89024, cl};
    Point(551) = {1.78752, -0.249995, -2.10135, cl};
    Point(552) = {1.78752, -0.508329, -1.99302, cl};
    Point(553) = {1.78752, -0.733333, -1.50135, cl};
    Point(554) = {1.78752, -0.758333, -1.75413, cl};
    Point(555) = {1.78752, -0.788889, -1.61246, cl};
    Point(556) = {1.78752, 0.644449, -1.89024, cl};
    Point(557) = {1.78752, 0.508338, -1.99302, cl};
    Point(558) = {1.78752, 0.250005, -2.10135, cl};
    Point(559) = {1.78752, 4.6938e-06, -2.12635, cl};
    Point(560) = {1.78752, 0.758338, -1.75413, cl};
    Point(561) = {1.78752, 0.733338, -1.50135, cl};
    Point(562) = {1.78752, 0.788894, -1.61246, cl};
    Point(563) = {1.78752, -0.113884, -0.106904, cl};
    Point(564) = {1.78752, -0.147217, -0.143016, cl};
    Point(565) = {1.78752, -0.172217, -0.179127, cl};
    Point(566) = {1.78752, -0.772222, -1.14857, cl};
    Point(567) = {1.78752, -1.21944, -1.09302, cl};
    Point(568) = {1.78752, -1.10556, -1.10413, cl};
    Point(569) = {1.78752, -0.883333, -1.12913, cl};
    Point(570) = {1.78752, -0.997222, -1.11524, cl};
    Point(571) = {1.78752, -0.661111, -1.17357, cl};
    Point(572) = {1.78752, -0.55, -1.20413, cl};
    Point(573) = {1.78752, -0.652778, -1.4319, cl};
    Point(574) = {1.78752, -0.55, -1.37357, cl};
    Point(575) = {1.78752, -0.716667, -0.740238, cl};
    Point(576) = {1.78752, -1.00556, -1.00968, cl};
    Point(577) = {1.78752, -0.85, -0.890238, cl};
    Point(578) = {1.78752, -0.347218, -0.270793, cl};
    Point(579) = {1.78752, -0.611111, -0.573571, cl};
    Point(580) = {1.78752, -0.491663, -0.409682, cl};
    Point(581) = {1.78752, 0.997227, -1.11524, cl};
    Point(582) = {1.78752, 0.550005, -1.20413, cl};
    Point(583) = {1.78752, 0.550005, -1.37357, cl};
    Point(584) = {1.78752, 0.652783, -1.4319, cl};
    Point(585) = {1.78752, 0.661116, -1.17357, cl};
    Point(586) = {1.78752, 0.883338, -1.12913, cl};
    Point(587) = {1.78752, 0.772227, -1.14857, cl};
    Point(588) = {1.78752, 0.716672, -0.740238, cl};
    Point(589) = {1.78752, 0.850005, -0.890238, cl};
    Point(590) = {1.78752, 1.21944, -1.09302, cl};
    Point(591) = {1.78752, 1.10556, -1.10413, cl};
    Point(592) = {1.78752, 1.00834, -1.00968, cl};
    Point(593) = {1.78752, 0.347227, -0.270793, cl};
    Point(594) = {1.78752, 0.491672, -0.409682, cl};
    Point(595) = {1.78752, 0.611116, -0.573571, cl};
    Point(597) = {1.78752, -0.0638843, -0.081904, cl};
    Point(598) = {1.78752, 4.6938e-06, -0.0680161, cl};
    Point(599) = {1.78752, 0.0638938, -0.0819039, cl};
    Point(600) = {1.78752, 0.113894, -0.106904, cl};
    Point(601) = {1.78752, 0.147227, -0.143016, cl};
    Point(602) = {1.78752, 0.172213, -0.179127, cl};
    Point(603) = {1.4403, -0.163884, -2.10135, cl};
    Point(604) = {1.4403, -0.533333, -1.97357, cl};
    Point(605) = {1.4403, -0.358325, -2.06246, cl};
    Point(606) = {1.4403, -0.691666, -1.83468, cl};
    Point(607) = {1.4403, -0.791667, -1.5569, cl};
    Point(608) = {1.4403, -0.791667, -1.67635, cl};
    Point(609) = {1.4403, 0.358338, -2.06246, cl};
    Point(610) = {1.4403, 0.533338, -1.97357, cl};
    Point(611) = {1.4403, -0.00277333, -2.11246, cl};
    Point(612) = {1.4403, 0.163894, -2.10135, cl};
    Point(613) = {1.4403, 0.791672, -1.5569, cl};
    Point(614) = {1.4403, 0.691672, -1.83468, cl};
    Point(615) = {1.4403, 0.791672, -1.67635, cl};
    Point(616) = {1.4403, 0.744449, -1.44302, cl};
    Point(617) = {1.4403, -0.744444, -1.44302, cl};
    Point(618) = {1.4403, -1.31944, -1.09024, cl};
    Point(619) = {1.4403, -0.666667, -1.17913, cl};
    Point(620) = {1.4403, -0.661111, -1.36246, cl};
    Point(621) = {1.4403, -1.14722, -1.10968, cl};
    Point(622) = {1.4403, -1.05278, -1.12079, cl};
    Point(623) = {1.4403, -0.961111, -1.1319, cl};
    Point(624) = {1.4403, -0.858333, -1.14302, cl};
    Point(625) = {1.4403, -0.758333, -1.15968, cl};
    Point(626) = {1.4403, -0.713889, -0.731904, cl};
    Point(627) = {1.4403, -0.869444, -0.88746, cl};
    Point(628) = {1.4403, -1.05833, -1.00135, cl};
    Point(629) = {1.4403, -0.586111, -0.551349, cl};
    Point(630) = {1.4403, -0.274995, -0.245793, cl};
    Point(631) = {1.4403, -0.452772, -0.379127, cl};
    Point(632) = {1.4403, 0.858338, -1.14302, cl};
    Point(633) = {1.4403, 0.661116, -1.36246, cl};
    Point(634) = {1.4403, 0.666672, -1.17913, cl};
    Point(635) = {1.4403, 0.758338, -1.15968, cl};
    Point(636) = {1.4403, 0.869448, -0.88746, cl};
    Point(637) = {1.4403, 0.713894, -0.731904, cl};
    Point(638) = {1.4403, 1.31944, -1.09024, cl};
    Point(639) = {1.4403, 0.961116, -1.1319, cl};
    Point(640) = {1.4403, 1.05279, -1.12079, cl};
    Point(641) = {1.4403, 1.14445, -1.10968, cl};
    Point(642) = {1.4403, 1.06112, -1.00413, cl};
    Point(643) = {1.4403, 0.275005, -0.245793, cl};
    Point(644) = {1.4403, 0.586116, -0.551349, cl};
    Point(645) = {1.4403, 0.452783, -0.379127, cl};
    Point(646) = {1.4403, 0.0888937, -0.179127, cl};
    Point(647) = {1.4403, 0.0611157, -0.168016, cl};
    Point(648) = {1.4403, 0.0388938, -0.16246, cl};
    Point(649) = {1.4403, 0.0194487, -0.156904, cl};
    Point(650) = {1.4403, -0.00277316, -0.151349, cl};
    Point(651) = {1.4403, -0.0194392, -0.156904, cl};
    Point(652) = {1.4403, -0.0388844, -0.16246, cl};
    Point(653) = {1.4403, -0.0611062, -0.168016, cl};
    Point(654) = {1.4403, -0.0888842, -0.179127, cl};


    // the hood reloaded
    Point(655) = {1.09308, -0.205551, -2.09024, cl};
    Point(656) = {1.09308, -0.541668, -1.96802, cl};
    Point(657) = {1.09308, -0.405548, -2.04024, cl};
    Point(658) = {1.09308, -0.683332, -1.84024, cl};
    Point(659) = {1.09308, -0.777778, -1.71524, cl};
    Point(660) = {1.09308, -0.788889, -1.47357, cl};
    Point(661) = {1.09308, -0.802778, -1.58746, cl};
    Point(662) = {1.09308, 0.205561, -2.09024, cl};
    Point(663) = {1.09308, 4.76091e-06, -2.10413, cl};
    Point(664) = {1.09308, 0.405561, -2.04024, cl};
    Point(665) = {1.09308, 0.541672, -1.96802, cl};
    Point(666) = {1.09308, 0.777783, -1.71524, cl};
    Point(667) = {1.09308, 0.683338, -1.84024, cl};
    Point(668) = {1.09308, 0.802783, -1.58746, cl};
    Point(669) = {1.09308, 0.788894, -1.47357, cl};
    Point(670) = {1.09308, -1.41389, -1.09024, cl};
    Point(671) = {1.09308, -0.738889, -1.36802, cl};
    Point(672) = {1.09308, -0.741667, -1.17913, cl};
    Point(673) = {1.09308, -1.075, -1.1319, cl};
    Point(674) = {1.09308, -1.15, -1.12357, cl};
    Point(675) = {1.09308, -1.0, -1.14024, cl};
    Point(676) = {1.09308, -0.827778, -1.16246, cl};
    Point(677) = {1.09308, -0.913889, -1.14857, cl};
    Point(678) = {1.09308, -0.716667, -0.726349, cl};
    Point(679) = {1.09308, -0.886111, -0.879127, cl};
    Point(680) = {1.09308, -1.09444, -0.984682, cl};
    Point(681) = {1.09308, -0.572222, -0.548571, cl};
    Point(682) = {1.09308, -0.427773, -0.370793, cl};
    Point(683) = {1.09308, -0.233328, -0.245793, cl};
    Point(684) = {1.09308, 0.738894, -1.36802, cl};
    Point(685) = {1.09308, 0.827783, -1.16246, cl};
    Point(686) = {1.09308, 0.741672, -1.17913, cl};
    Point(687) = {1.09308, 0.716672, -0.726349, cl};
    Point(688) = {1.09308, 0.886116, -0.879128, cl};
    Point(689) = {1.09308, 0.913894, -1.14857, cl};
    Point(690) = {1.09308, 1.41389, -1.09024, cl};
    Point(691) = {1.09308, 1.00001, -1.14024, cl};
    Point(692) = {1.09308, 1.07501, -1.1319, cl};
    Point(693) = {1.09308, 1.15001, -1.12357, cl};
    Point(694) = {1.09308, 1.09445, -0.987461, cl};
    Point(695) = {1.09308, 0.233338, -0.245793, cl};
    Point(696) = {1.09308, 0.427783, -0.370793, cl};
    Point(697) = {1.09308, 0.572227, -0.548571, cl};
    Point(698) = {1.09308, -0.00277313, -0.198571, cl};
    Point(699) = {0.745857, -0.183328, -2.0819, cl};
    Point(700) = {0.745857, -0.374996, -2.04302, cl};
    Point(701) = {0.745857, -0.533333, -1.95968, cl};
    Point(702) = {0.745857, -0.677778, -1.82913, cl};
    Point(703) = {0.745857, -0.761111, -1.71524, cl};
    Point(704) = {0.745857, -0.802778, -1.59579, cl};
    Point(705) = {0.745857, -0.8, -1.47357, cl};
    Point(706) = {0.745857, 0.183338, -2.0819, cl};
    Point(707) = {0.745857, 4.79446e-06, -2.09302, cl};
    Point(708) = {0.745857, 0.533338, -1.95968, cl};
    Point(709) = {0.745857, 0.375005, -2.04302, cl};
    Point(710) = {0.745857, 0.761116, -1.71524, cl};
    Point(711) = {0.745857, 0.677783, -1.82913, cl};
    Point(712) = {0.745857, 0.800004, -1.47357, cl};
    Point(713) = {0.745857, 0.802783, -1.59579, cl};
    Point(714) = {0.745857, -1.47222, -1.08746, cl};
    Point(715) = {0.745857, -0.780556, -1.38468, cl};
    Point(716) = {0.745857, -1.15, -1.13468, cl};
    Point(717) = {0.745857, -1.075, -1.14857, cl};
    Point(718) = {0.745857, -1.00278, -1.15413, cl};
    Point(719) = {0.745857, -0.930556, -1.16524, cl};
    Point(720) = {0.745857, -0.783333, -1.18468, cl};
    Point(721) = {0.745857, -0.855556, -1.17357, cl};
    Point(722) = {0.745857, -0.725, -0.718016, cl};
    Point(723) = {0.745857, -0.905556, -0.868016, cl};
    Point(724) = {0.745857, -1.18889, -0.990238, cl};
    Point(725) = {0.745857, -0.55833, -0.545793, cl};
    Point(726) = {0.745857, -0.413881, -0.393016, cl};
    Point(727) = {0.745857, -0.233328, -0.276349, cl};
    Point(728) = {0.745857, 0.780561, -1.38468, cl};
    Point(729) = {0.745857, 0.855561, -1.17357, cl};
    Point(730) = {0.745857, 0.783338, -1.18468, cl};
    Point(731) = {0.745857, 0.725005, -0.718016, cl};
    Point(732) = {0.745857, 0.905561, -0.868016, cl};
    Point(733) = {0.745857, 1.47222, -1.08746, cl};
    Point(734) = {0.745857, 0.930561, -1.16524, cl};
    Point(735) = {0.745857, 1.00279, -1.15413, cl};
    Point(736) = {0.745857, 1.15001, -1.13468, cl};
    Point(737) = {0.745857, 1.07501, -1.14857, cl};
    Point(738) = {0.745857, 1.18889, -0.990238, cl};
    Point(739) = {0.745857, 4.63061e-06, -0.229127, cl};
    Point(740) = {0.745857, 0.233338, -0.276349, cl};
    Point(741) = {0.745857, 0.413894, -0.393016, cl};
    Point(742) = {0.745857, 0.558338, -0.545793, cl};


    //
    Point(743) = {0.526409, -1.47222, -1.08746, cl};
    Point(744) = {0.526409, 1.47222, -1.08746, cl};
    Point(745) = {0.421689, -1.47222, -1.05008, cl};
    Point(746) = {0.421689, -1.47222, -1.10679, cl};
    Point(747) = {0.421689, 1.47222, -1.05008, cl};
    Point(748) = {0.421689, 1.47222, -1.10679, cl};
    Point(749) = {0.398626, -0.219439, -0.301349, cl};
    Point(750) = {0.398626, 0.744449, -0.731904, cl};
    Point(751) = {0.398626, 0.802783, -1.37357, cl};
    Point(752) = {0.398626, 0.805561, -1.20413, cl};
    Point(753) = {0.398626, 0.872227, -1.19302, cl};
    Point(754) = {0.398626, 1.50278, -1.04857, cl};
    Point(755) = {0.398626, 1.50278, -1.10135, cl};
    Point(756) = {0.398626, 0.941672, -1.1819, cl};
    Point(757) = {0.398626, 1.01112, -1.17357, cl};
    Point(758) = {0.398626, 1.08334, -1.16246, cl};
    Point(759) = {0.398626, 1.15556, -1.15413, cl};
    Point(760) = {0.398626, 0.952783, -0.873571, cl};
    Point(761) = {0.398626, 1.19723, -0.968016, cl};
    Point(762) = {0.398626, 0.219449, -0.301349, cl};
    Point(763) = {0.398626, 4.66028e-06, -0.26246, cl};
    Point(764) = {0.398626, 0.580561, -0.584682, cl};
    Point(765) = {0.398626, 0.397227, -0.404127, cl};
    Point(766) = {0.398625, -0.194439, -2.07635, cl};
    Point(767) = {0.398625, -0.516667, -1.96524, cl};
    Point(768) = {0.398625, -0.380548, -2.0319, cl};
    Point(769) = {0.398625, -0.805556, -1.46524, cl};
    Point(770) = {0.398625, -0.8, -1.5819, cl};
    Point(771) = {0.398625, -0.738889, -1.72079, cl};
    Point(772) = {0.398625, -0.661111, -1.83468, cl};
    Point(773) = {0.398625, 4.82753e-06, -2.08746, cl};
    Point(774) = {0.398625, 0.194449, -2.07635, cl};
    Point(775) = {0.398625, 0.380561, -2.0319, cl};
    Point(776) = {0.398625, 0.516672, -1.96524, cl};
    Point(777) = {0.398625, 0.805561, -1.46524, cl};
    Point(778) = {0.398625, 0.800005, -1.5819, cl};
    Point(779) = {0.398625, 0.661115, -1.83468, cl};
    Point(780) = {0.398625, 0.738894, -1.72079, cl};
    Point(781) = {0.398625, -1.50278, -1.10135, cl};
    Point(782) = {0.398625, -1.50278, -1.04857, cl};
    Point(783) = {0.398625, -0.802778, -1.37357, cl};
    Point(784) = {0.398625, -1.15556, -1.15135, cl};
    Point(785) = {0.398625, -1.08333, -1.16246, cl};
    Point(786) = {0.398625, -1.01111, -1.17357, cl};
    Point(787) = {0.398625, -0.941667, -1.1819, cl};
    Point(788) = {0.398625, -0.872222, -1.19302, cl};
    Point(789) = {0.398625, -0.805556, -1.20413, cl};
    Point(790) = {0.398625, -0.744444, -0.731904, cl};
    Point(791) = {0.398625, -0.952778, -0.873571, cl};
    Point(792) = {0.398625, -1.19722, -0.965238, cl};
    Point(793) = {0.398625, -0.397214, -0.404127, cl};
    Point(794) = {0.398625, -0.580557, -0.584682, cl};
    Point(795) = {0.316687, -1.47222, -1.03319, cl};
    Point(796) = {0.316687, -1.47222, -1.11129, cl};
    Point(797) = {0.316687, 1.47222, -1.03319, cl};
    Point(798) = {0.316687, 1.47222, -1.11129, cl};
    Point(799) = {0.211967, -1.47222, -1.11577, cl};
    Point(800) = {0.211967, -1.47222, -1.01828, cl};
    Point(801) = {0.211967, 1.47222, -1.11577, cl};
    Point(802) = {0.211967, 1.47222, -1.01828, cl};
    Point(803) = {0.106965, -1.47222, -1.00877, cl};
    Point(804) = {0.106965, 1.97222, -1.08746, cl};
    Point(805) = {0.106965, 1.47222, -1.11615, cl};
    Point(806) = {0.106965, 1.47222, -1.00877, cl};
    Point(807) = {0.106964, -1.97222, -1.08746, cl};
    Point(808) = {0.106964, -1.47222, -1.11615, cl};
    Point(809) = {0.0514048, 1.49722, -1.10968, cl};
    Point(810) = {0.0514048, 1.49722, -1.00413, cl};
    Point(811) = {0.0514048, 0.783337, -1.57357, cl};
    Point(812) = {0.0514048, -0.244439, -0.343016, cl};
    Point(813) = {0.0514048, 0.730561, -0.720793, cl};
    Point(814) = {0.0514048, 0.802783, -1.44024, cl};
    Point(815) = {0.0514048, 0.805561, -1.32913, cl};
    Point(816) = {0.0514048, 0.875005, -1.21246, cl};
    Point(817) = {0.0514048, 0.813894, -1.22357, cl};
    Point(818) = {0.0514048, 0.944449, -1.19857, cl};
    Point(819) = {0.0514048, 1.01668, -1.19024, cl};
    Point(820) = {0.0514048, 1.08611, -1.17357, cl};
    Point(821) = {0.0514048, 1.15, -1.16524, cl};
    Point(822) = {0.0514048, 0.988894, -0.873571, cl};
    Point(823) = {0.0514048, 1.26667, -0.959682, cl};
    Point(824) = {0.0514048, -0.00277337, -0.293016, cl};
    Point(825) = {0.0514048, 0.244449, -0.343016, cl};
    Point(826) = {0.0514048, 0.569449, -0.595793, cl};
    Point(827) = {0.0514048, 0.408338, -0.443016, cl};
    Point(828) = {0.0514048, 0.197227, -2.06524, cl};
    Point(829) = {0.0514048, 0.511116, -1.95413, cl};
    Point(830) = {0.0514048, 0.380561, -2.02079, cl};
    Point(831) = {0.0514048, 0.713894, -1.72635, cl};
    Point(832) = {0.0514048, 0.638894, -1.84024, cl};
    Point(833) = {0.0514048, -0.730556, -0.720793, cl};
    Point(834) = {0.0514048, -0.408328, -0.443016, cl};
    Point(835) = {0.0514048, -0.569444, -0.595793, cl};
    Point(836) = {0.0514048, -0.197217, -2.06524, cl};
    Point(837) = {0.0514048, -0.380548, -2.02079, cl};
    Point(838) = {0.0514048, -0.511111, -1.95413, cl};
    Point(839) = {0.0514048, -0.783333, -1.57357, cl};
    Point(840) = {0.0514048, -0.713889, -1.72635, cl};
    Point(841) = {0.0514048, -0.638889, -1.84024, cl};
    Point(842) = {0.0514048, 4.86108e-06, -2.07635, cl};
    Point(843) = {0.0514048, -1.49722, -1.00413, cl};
    Point(844) = {0.0514048, -1.49722, -1.10968, cl};
    Point(845) = {0.0514048, -0.805556, -1.32913, cl};
    Point(846) = {0.0514048, -0.802778, -1.44024, cl};
    Point(847) = {0.0514048, -1.15, -1.16524, cl};
    Point(848) = {0.0514048, -1.01667, -1.19024, cl};
    Point(849) = {0.0514048, -1.08611, -1.17357, cl};
    Point(850) = {0.0514048, -0.944444, -1.19857, cl};
    Point(851) = {0.0514048, -0.813889, -1.22357, cl};
    Point(852) = {0.0514048, -0.875, -1.21246, cl};
    Point(853) = {0.0514048, -0.988889, -0.873571, cl};
    Point(854) = {0.0514048, -1.26667, -0.959682, cl};
    Point(855) = {0.0125192, 1.97222, -1.05352, cl};
    Point(856) = {0.0125192, 1.97222, -1.10456, cl};
    Point(857) = {0.0125192, -1.97222, -1.05352, cl};
    Point(858) = {0.0125192, -1.97222, -1.10456, cl};
    Point(859) = {-0.081644, 1.97222, -1.03809, cl};
    Point(860) = {-0.081644, 1.97222, -1.10838, cl};
    Point(861) = {-0.081644, -1.97222, -1.03809, cl};
    Point(862) = {-0.081644, -1.97222, -1.10838, cl};
    Point(863) = {-0.102763, -1.47222, -1.11879, cl};
    Point(864) = {-0.102763, -1.47222, -0.990857, cl};
    Point(865) = {-0.102763, 1.47222, -1.11879, cl};
    Point(866) = {-0.102763, 1.47222, -0.990858, cl};
    Point(867) = {-0.176095, -1.97222, -1.11216, cl};
    Point(868) = {-0.176095, -1.97222, -1.02443, cl};
    Point(869) = {-0.176095, 1.97222, -1.11216, cl};
    Point(870) = {-0.176095, 1.97222, -1.02443, cl};
    Point(871) = {-0.270536, 1.97222, -1.11228, cl};
    Point(872) = {-0.270536, 1.97222, -1.01564, cl};
    Point(873) = {-0.270536, -1.97222, -1.11228, cl};
    Point(874) = {-0.270536, -1.97222, -1.01564, cl};
    Point(875) = {-0.295816, -0.330551, -2.02357, cl};
    Point(876) = {-0.295816, -0.488884, -1.95135, cl};
    Point(877) = {-0.295816, -0.769444, -1.57357, cl};
    Point(878) = {-0.295816, -0.627778, -1.8319, cl};
    Point(879) = {-0.295816, -0.694444, -1.72635, cl};
    Point(880) = {-0.295816, -0.163884, -2.05968, cl};
    Point(881) = {-0.295816, 0.488894, -1.95135, cl};
    Point(882) = {-0.295816, 0.330561, -2.02357, cl};
    Point(883) = {-0.295816, 0.163894, -2.05968, cl};
    Point(884) = {-0.295816, 0.00278263, -2.06802, cl};
    Point(885) = {-0.295816, 0.769449, -1.57357, cl};
    Point(886) = {-0.295816, 0.694449, -1.72635, cl};
    Point(887) = {-0.295816, 0.627783, -1.8319, cl};
    Point(888) = {-0.295816, -1.33889, -0.95135, cl};
    Point(889) = {-0.295816, -1.49722, -1.11524, cl};
    Point(890) = {-0.295816, -1.5, -0.981905, cl};
    Point(891) = {-0.295816, -0.813889, -1.24024, cl};
    Point(892) = {-0.295816, -0.791667, -1.44302, cl};
    Point(893) = {-0.295816, -0.802778, -1.33468, cl};
    Point(894) = {-0.295816, -1.15, -1.17635, cl};
    Point(895) = {-0.295816, -1.08611, -1.18746, cl};
    Point(896) = {-0.295816, -1.02222, -1.20413, cl};
    Point(897) = {-0.295816, -0.883334, -1.22635, cl};
    Point(898) = {-0.295816, -0.952778, -1.21246, cl};
    Point(899) = {-0.295816, -1.06667, -0.884682, cl};
    Point(900) = {-0.295816, -0.772222, -0.745793, cl};
    Point(901) = {-0.295816, -0.25277, -0.368016, cl};
    Point(902) = {-0.295816, -0.577776, -0.609682, cl};
    Point(903) = {-0.295816, -0.419439, -0.470793, cl};
    Point(904) = {-0.295816, 0.772227, -0.745793, cl};
    Point(905) = {-0.295816, 0.883338, -1.22635, cl};
    Point(906) = {-0.295816, 0.813894, -1.24024, cl};
    Point(907) = {-0.295816, 0.802783, -1.33468, cl};
    Point(908) = {-0.295816, 0.791672, -1.44302, cl};
    Point(909) = {-0.295816, 1.49722, -1.11524, cl};
    Point(910) = {-0.295816, 0.952783, -1.21246, cl};
    Point(911) = {-0.295816, 1.02222, -1.20413, cl};
    Point(912) = {-0.295816, 1.15, -1.17635, cl};
    Point(913) = {-0.295816, 1.08611, -1.18746, cl};
    Point(914) = {-0.295816, 1.5, -0.981905, cl};
    Point(915) = {-0.295816, 1.06668, -0.884682, cl};
    Point(916) = {-0.295816, 1.33889, -0.951349, cl};
    Point(917) = {-0.295816, 4.72083e-06, -0.315238, cl};
    Point(918) = {-0.295816, 0.252783, -0.368016, cl};
    Point(919) = {-0.295816, 0.419449, -0.470793, cl};
    Point(920) = {-0.295816, 0.577783, -0.609682, cl};
    Point(921) = {-0.312485, -1.47222, -0.976917, cl};
    Point(922) = {-0.312485, -1.47222, -1.12029, cl};
    Point(923) = {-0.312485, 1.47222, -0.976916, cl};
    Point(924) = {-0.312485, 1.47222, -1.12029, cl};
    Point(925) = {-0.459144, -1.97222, -1.1142, cl};
    Point(926) = {-0.459144, -1.97222, -0.999057, cl};
    Point(927) = {-0.459144, 1.97222, -1.1142, cl};
    Point(928) = {-0.459144, 1.97222, -0.999058, cl};
    Point(929) = {-0.643032, -0.236106, -0.376349, cl};
    Point(930) = {-0.643032, 0.780561, -1.43746, cl};
    Point(931) = {-0.643032, 0.816672, -0.768016, cl};
    Point(932) = {-0.643032, 0.800005, -1.33468, cl};
    Point(933) = {-0.643032, 0.813894, -1.25413, cl};
    Point(934) = {-0.643032, 0.877783, -1.24024, cl};
    Point(935) = {-0.643032, 1.5, -1.11524, cl};
    Point(936) = {-0.643032, 0.944448, -1.22635, cl};
    Point(937) = {-0.643032, 1.01111, -1.21524, cl};
    Point(938) = {-0.643032, 1.08611, -1.19857, cl};
    Point(939) = {-0.643032, 1.15278, -1.18468, cl};
    Point(940) = {-0.643032, 1.49444, -0.96246, cl};
    Point(941) = {-0.643032, 1.29722, -0.926349, cl};
    Point(942) = {-0.643032, 1.10001, -0.881904, cl};
    Point(943) = {-0.643032, 0.236116, -0.376349, cl};
    Point(944) = {-0.643032, 4.75244e-06, -0.326349, cl};
    Point(945) = {-0.643032, 0.566672, -0.606904, cl};
    Point(946) = {-0.643032, 0.391672, -0.46246, cl};
    Point(947) = {-0.643032, -0.486103, -1.94302, cl};
    Point(948) = {-0.643032, -0.299992, -2.02357, cl};
    Point(949) = {-0.643032, -0.75, -1.56802, cl};
    Point(950) = {-0.643032, -0.683333, -1.70968, cl};
    Point(951) = {-0.643032, -0.611111, -1.82635, cl};
    Point(952) = {-0.643032, 0.486116, -1.94302, cl};
    Point(953) = {-0.643032, -0.155551, -2.05135, cl};
    Point(954) = {-0.643032, 4.60115e-06, -2.0569, cl};
    Point(955) = {-0.643032, 0.155561, -2.05135, cl};
    Point(956) = {-0.643032, 0.300005, -2.02357, cl};
    Point(957) = {-0.643032, 0.750005, -1.56802, cl};
    Point(958) = {-0.643032, 0.611116, -1.82635, cl};
    Point(959) = {-0.643032, 0.683337, -1.70968, cl};
    Point(960) = {-0.643032, -0.780556, -1.43746, cl};
    Point(961) = {-0.643032, -1.29722, -0.926349, cl};
    Point(962) = {-0.643032, -1.49444, -0.96246, cl};
    Point(963) = {-0.643032, -1.5, -1.11524, cl};
    Point(964) = {-0.643032, -0.877778, -1.24024, cl};
    Point(965) = {-0.643032, -0.813889, -1.25413, cl};
    Point(966) = {-0.643032, -0.8, -1.33468, cl};
    Point(967) = {-0.643032, -0.944444, -1.22635, cl};
    Point(968) = {-0.643032, -1.15278, -1.18468, cl};
    Point(969) = {-0.643032, -1.08611, -1.19857, cl};
    Point(970) = {-0.643032, -1.01111, -1.21524, cl};
    Point(971) = {-0.643032, -0.816667, -0.768016, cl};
    Point(972) = {-0.643032, -1.1, -0.881904, cl};
    Point(973) = {-0.643032, -0.391661, -0.46246, cl};
    Point(974) = {-0.643032, -0.566667, -0.606904, cl};
    Point(975) = {-0.648032, 1.97222, -0.986029, cl};
    Point(976) = {-0.648032, 1.97222, -1.11506, cl};
    Point(977) = {-0.648032, -1.97222, -0.986029, cl};
    Point(978) = {-0.648032, -1.97222, -1.11506, cl};
    Point(979) = {-0.731924, -1.47222, -1.12068, cl};
    Point(980) = {-0.731924, -1.47222, -0.958354, cl};
    Point(981) = {-0.731924, 1.47222, -1.12068, cl};
    Point(982) = {-0.731924, 1.47222, -0.958354, cl};
    Point(983) = {-0.990263, -0.311104, -2.01524, cl};
    Point(984) = {-0.990263, -0.480549, -1.93468, cl};
    Point(985) = {-0.990263, -0.763889, -1.45968, cl};
    Point(986) = {-0.990263, -0.733333, -1.57357, cl};
    Point(987) = {-0.990263, -0.594444, -1.8319, cl};
    Point(988) = {-0.990263, -0.672222, -1.70413, cl};
    Point(989) = {-0.990263, 0.480561, -1.93468, cl};
    Point(990) = {-0.990263, 0.0027827, -2.04857, cl};
    Point(991) = {-0.990263, -0.172217, -2.04024, cl};
    Point(992) = {-0.990263, 0.311116, -2.01524, cl};
    Point(993) = {-0.990263, 0.172227, -2.04024, cl};
    Point(994) = {-0.990263, 0.672227, -1.70413, cl};
    Point(995) = {-0.990263, 0.594449, -1.8319, cl};
    Point(996) = {-0.990263, 0.733338, -1.57357, cl};
    Point(997) = {-0.990263, 0.763894, -1.45968, cl};
    Point(998) = {-0.990263, -1.5, -0.951349, cl};
    Point(999) = {-0.990263, -1.50278, -1.11524, cl};
    Point(1000) = {-0.990263, -0.875, -1.25135, cl};
    Point(1001) = {-0.990263, -0.791667, -1.34024, cl};
    Point(1002) = {-0.990263, -0.808333, -1.27079, cl};
    Point(1003) = {-0.990263, -0.944444, -1.23746, cl};
    Point(1004) = {-0.990263, -1.15833, -1.19024, cl};
    Point(1005) = {-0.990263, -1.01944, -1.22357, cl};
    Point(1006) = {-0.990263, -1.09167, -1.20135, cl};
    Point(1007) = {-0.990263, -0.736111, -0.720793, cl};
    Point(1008) = {-0.990263, -0.966667, -0.829127, cl};
    Point(1009) = {-0.990263, -1.21944, -0.898571, cl};
    Point(1010) = {-0.990263, -0.219439, -0.376349, cl};
    Point(1011) = {-0.990263, -0.547222, -0.593016, cl};
    Point(1012) = {-0.990263, -0.369436, -0.451349, cl};
    Point(1013) = {-0.990263, 0.736116, -0.720793, cl};
    Point(1014) = {-0.990263, 0.791671, -1.34024, cl};
    Point(1015) = {-0.990263, 0.875004, -1.25135, cl};
    Point(1016) = {-0.990263, 0.808338, -1.27079, cl};
    Point(1017) = {-0.990263, 1.50278, -1.11524, cl};
    Point(1018) = {-0.990263, 0.944449, -1.23746, cl};
    Point(1019) = {-0.990263, 1.01944, -1.22357, cl};
    Point(1020) = {-0.990263, 1.15834, -1.19024, cl};
    Point(1021) = {-0.990263, 1.09168, -1.20135, cl};
    Point(1022) = {-0.990263, 1.5, -0.951349, cl};
    Point(1023) = {-0.990263, 0.966672, -0.829127, cl};
    Point(1024) = {-0.990263, 1.21944, -0.898571, cl};
    Point(1025) = {-0.990263, 4.78405e-06, -0.33746, cl};
    Point(1026) = {-0.990263, 0.219449, -0.376349, cl};
    Point(1027) = {-0.990263, 0.369449, -0.451349, cl};
    Point(1028) = {-0.990263, 0.547227, -0.593016, cl};
    Point(1029) = {-1.02554, -1.97222, -1.11451, cl};
    Point(1030) = {-1.02554, -1.97222, -0.968413, cl};
    Point(1031) = {-1.02554, 1.97222, -1.11451, cl};
    Point(1032) = {-1.02554, 1.97222, -0.968413, cl};
    Point(1033) = {-1.0347, -3.33333, -1.08746, cl};
    Point(1034) = {-1.0347, 3.33333, -1.08746, cl};
    Point(1035) = {-1.10054, -3.33333, -1.06308, cl};
    Point(1036) = {-1.10054, -3.33333, -1.09868, cl};
    Point(1037) = {-1.10054, 3.33333, -1.06308, cl};
    Point(1038) = {-1.10054, 3.33333, -1.09868, cl};
    Point(1039) = {-1.15137, -1.47222, -1.11799, cl};
    Point(1040) = {-1.15137, -1.47222, -0.95021, cl};
    Point(1041) = {-1.15137, 1.47222, -1.11799, cl};
    Point(1042) = {-1.15137, 1.47222, -0.95021, cl};
    Point(1043) = {-1.16638, 3.33333, -1.05161, cl};
    Point(1044) = {-1.16638, 3.33333, -1.10064, cl};
    Point(1045) = {-1.16638, -3.33333, -1.05161, cl};
    Point(1046) = {-1.16638, -3.33333, -1.10064, cl};
    Point(1047) = {-1.2322, -3.33333, -1.04139, cl};
    Point(1048) = {-1.2322, -3.33333, -1.10259, cl};
    Point(1049) = {-1.2322, 3.33334, -1.04139, cl};
    Point(1050) = {-1.2322, 3.33334, -1.10259, cl};
    Point(1051) = {-1.29803, -3.33333, -1.03453, cl};
    Point(1052) = {-1.29803, -3.33333, -1.10194, cl};
    Point(1053) = {-1.29803, 3.33334, -1.03453, cl};
    Point(1054) = {-1.29803, 3.33334, -1.10194, cl};
    Point(1055) = {-1.33748, -0.486103, -1.92357, cl};
    Point(1056) = {-1.33748, -0.347214, -1.99302, cl};
    Point(1057) = {-1.33748, -0.741667, -1.47635, cl};
    Point(1058) = {-1.33748, -0.7, -1.59857, cl};
    Point(1059) = {-1.33748, -0.644444, -1.72357, cl};
    Point(1060) = {-1.33748, -0.574998, -1.83468, cl};
    Point(1061) = {-1.33748, 0.486116, -1.92357, cl};
    Point(1062) = {-1.33748, -0.183328, -2.02913, cl};
    Point(1063) = {-1.33748, 4.66777e-06, -2.04024, cl};
    Point(1064) = {-1.33748, 0.183338, -2.02913, cl};
    Point(1065) = {-1.33748, 0.347227, -1.99302, cl};
    Point(1066) = {-1.33748, 0.575005, -1.83468, cl};
    Point(1067) = {-1.33748, 0.644449, -1.72357, cl};
    Point(1068) = {-1.33748, 0.700005, -1.59857, cl};
    Point(1069) = {-1.33748, 0.741672, -1.47635, cl};
    Point(1070) = {-1.33748, -1.49722, -0.954127, cl};
    Point(1071) = {-1.33748, -1.5, -1.10968, cl};
    Point(1072) = {-1.33748, -0.775, -1.37079, cl};
    Point(1073) = {-1.33748, -0.941667, -1.24302, cl};
    Point(1074) = {-1.33748, -0.866667, -1.26246, cl};
    Point(1075) = {-1.33748, -0.794444, -1.27913, cl};
    Point(1076) = {-1.33748, -1.01389, -1.22913, cl};
    Point(1077) = {-1.33748, -1.15556, -1.19024, cl};
    Point(1078) = {-1.33748, -1.08889, -1.2069, cl};
    Point(1079) = {-1.33748, -0.713889, -0.704127, cl};
    Point(1080) = {-1.33748, -0.941667, -0.815238, cl};
    Point(1081) = {-1.33748, -1.18611, -0.881904, cl};
    Point(1082) = {-1.33748, -0.222217, -0.379127, cl};
    Point(1083) = {-1.33748, -0.388881, -0.46246, cl};
    Point(1084) = {-1.33748, -0.527775, -0.576349, cl};
    Point(1085) = {-1.33748, 0.713894, -0.704127, cl};
    Point(1086) = {-1.33748, 0.775005, -1.37079, cl};
    Point(1087) = {-1.33748, 0.794449, -1.27913, cl};
    Point(1088) = {-1.33748, 0.866672, -1.26246, cl};
    Point(1089) = {-1.33748, 1.5, -1.10968, cl};
    Point(1090) = {-1.33748, 0.941672, -1.24302, cl};
    Point(1091) = {-1.33748, 1.01389, -1.22913, cl};
    Point(1092) = {-1.33748, 1.08889, -1.2069, cl};
    Point(1093) = {-1.33748, 1.15557, -1.19024, cl};
    Point(1094) = {-1.33748, 1.49722, -0.954127, cl};
    Point(1095) = {-1.33748, 0.941671, -0.815238, cl};
    Point(1096) = {-1.33748, 1.18611, -0.881904, cl};
    Point(1097) = {-1.33748, 0.222227, -0.379127, cl};
    Point(1098) = {-1.33748, 4.81639e-06, -0.340238, cl};
    Point(1099) = {-1.33748, 0.527783, -0.576349, cl};
    Point(1100) = {-1.33748, 0.388894, -0.46246, cl};
    Point(1101) = {-1.40304, -1.97222, -1.11118, cl};
    Point(1102) = {-1.40304, -1.97222, -0.960182, cl};
    Point(1103) = {-1.40304, 1.97222, -1.11118, cl};
    Point(1104) = {-1.40304, 1.97222, -0.960182, cl};
    Point(1105) = {-1.4297, -3.33333, -1.02147, cl};
    Point(1106) = {-1.4297, -3.33333, -1.10178, cl};
    Point(1107) = {-1.4297, 3.33334, -1.02147, cl};
    Point(1108) = {-1.4297, 3.33334, -1.10178, cl};
    Point(1109) = {-1.56138, -3.33333, -1.01112, cl};
    Point(1110) = {-1.56138, -3.33333, -1.10113, cl};
    Point(1111) = {-1.56138, 3.33333, -1.01112, cl};
    Point(1112) = {-1.56138, 3.33333, -1.10113, cl};
    Point(1113) = {-1.57081, -1.47222, -1.11056, cl};
    Point(1114) = {-1.57081, -1.47222, -0.953358, cl};
    Point(1115) = {-1.57081, 1.47222, -1.11056, cl};
    Point(1116) = {-1.57081, 1.47222, -0.953357, cl};
    Point(1117) = {-1.6847, -0.188884, -2.02357, cl};
    Point(1118) = {-1.6847, -0.336107, -1.99024, cl};
    Point(1119) = {-1.6847, -0.477773, -1.91802, cl};
    Point(1120) = {-1.6847, -0.727778, -1.4819, cl};
    Point(1121) = {-1.6847, -0.683333, -1.6069, cl};
    Point(1122) = {-1.6847, -0.572222, -1.82913, cl};
    Point(1123) = {-1.6847, -0.633333, -1.72913, cl};
    Point(1124) = {-1.6847, 0.477783, -1.91802, cl};
    Point(1125) = {-1.6847, 4.70108e-06, -2.0319, cl};
    Point(1126) = {-1.6847, 0.336116, -1.99024, cl};
    Point(1127) = {-1.6847, 0.188894, -2.02357, cl};
    Point(1128) = {-1.6847, 0.633338, -1.72913, cl};
    Point(1129) = {-1.6847, 0.572227, -1.82913, cl};
    Point(1130) = {-1.6847, 0.727783, -1.4819, cl};
    Point(1131) = {-1.6847, 0.683338, -1.6069, cl};
    Point(1132) = {-1.6847, -1.49722, -0.959682, cl};
    Point(1133) = {-1.6847, -1.49722, -1.10413, cl};
    Point(1134) = {-1.6847, -0.758333, -1.3819, cl};
    Point(1135) = {-1.6847, -0.930556, -1.24857, cl};
    Point(1136) = {-1.6847, -0.777778, -1.28468, cl};
    Point(1137) = {-1.6847, -0.85, -1.27079, cl};
    Point(1138) = {-1.6847, -1.00278, -1.2319, cl};
    Point(1139) = {-1.6847, -1.08056, -1.20968, cl};
    Point(1140) = {-1.6847, -1.15556, -1.19024, cl};
    Point(1141) = {-1.6847, -0.716667, -0.701349, cl};
    Point(1142) = {-1.6847, -1.13611, -0.868016, cl};
    Point(1143) = {-1.6847, -0.933333, -0.806904, cl};
    Point(1144) = {-1.6847, -0.238884, -0.381904, cl};
    Point(1145) = {-1.6847, -0.533334, -0.576349, cl};
    Point(1146) = {-1.6847, -0.388885, -0.459682, cl};
    Point(1147) = {-1.6847, 0.716672, -0.701349, cl};
    Point(1148) = {-1.6847, 0.758338, -1.3819, cl};
    Point(1149) = {-1.6847, 0.850005, -1.27079, cl};
    Point(1150) = {-1.6847, 0.777783, -1.28468, cl};
    Point(1151) = {-1.6847, 1.49722, -1.10413, cl};
    Point(1152) = {-1.6847, 0.930561, -1.24857, cl};
    Point(1153) = {-1.6847, 1.00279, -1.2319, cl};
    Point(1154) = {-1.6847, 1.15557, -1.19024, cl};
    Point(1155) = {-1.6847, 1.08056, -1.20968, cl};
    Point(1156) = {-1.6847, 1.49722, -0.959682, cl};
    Point(1157) = {-1.6847, 0.933338, -0.806904, cl};
    Point(1158) = {-1.6847, 1.13612, -0.868016, cl};
    Point(1159) = {-1.6847, 4.84872e-06, -0.343016, cl};
    Point(1160) = {-1.6847, 0.238894, -0.381904, cl};
    Point(1161) = {-1.6847, 0.388894, -0.459682, cl};
    Point(1162) = {-1.6847, 0.533338, -0.576349, cl};
    Point(1163) = {-1.78054, -1.97222, -1.10362, cl};
    Point(1164) = {-1.78054, -1.97222, -0.962132, cl};
    Point(1165) = {-1.78054, 1.97222, -1.10362, cl};
    Point(1166) = {-1.78054, 1.97222, -0.962132, cl};
    Point(1167) = {-1.8247, -3.33333, -0.99606, cl};
    Point(1168) = {-1.8247, -3.33333, -1.09797, cl};
    Point(1169) = {-1.8247, 3.33334, -0.99606, cl};
    Point(1170) = {-1.8247, 3.33333, -1.09797, cl};
    Point(1171) = {-1.99026, -1.47222, -1.1006, cl};
    Point(1172) = {-1.99026, -1.47222, -0.965707, cl};
    Point(1173) = {-1.99026, 1.47222, -1.1006, cl};
    Point(1174) = {-1.99026, 1.47222, -0.965707, cl};
    Point(1175) = {-2.03192, -0.333326, -1.9819, cl};
    Point(1176) = {-2.03192, -0.711111, -1.49857, cl};
    Point(1177) = {-2.03192, -0.663889, -1.6319, cl};
    Point(1178) = {-2.03192, 0.461116, -1.92079, cl};
    Point(1179) = {-2.03192, 0.166672, -2.01524, cl};
    Point(1180) = {-2.03192, 0.333338, -1.9819, cl};
    Point(1181) = {-2.03192, -0.166662, -2.01524, cl};
    Point(1182) = {-2.03192, -0.00277333, -2.02079, cl};
    Point(1183) = {-2.03192, 0.550005, -1.84024, cl};
    Point(1184) = {-2.03192, 0.611116, -1.74857, cl};
    Point(1185) = {-2.03192, 0.663894, -1.6319, cl};
    Point(1186) = {-2.03192, 0.711116, -1.49857, cl};
    Point(1187) = {-2.03192, -0.744444, -1.37913, cl};
    Point(1188) = {-2.03192, -0.925, -1.24857, cl};
    Point(1189) = {-2.03192, -0.844444, -1.27079, cl};
    Point(1190) = {-2.03192, -0.769444, -1.29024, cl};
    Point(1191) = {-2.03192, -1.0, -1.2319, cl};
    Point(1192) = {-2.03192, -0.672222, -0.670793, cl};
    Point(1193) = {-2.03192, -0.894444, -0.784682, cl};
    Point(1194) = {-2.03192, -1.15278, -0.879127, cl};
    Point(1195) = {-2.03192, -0.219439, -0.373571, cl};
    Point(1196) = {-2.03192, -0.391659, -0.454127, cl};
    Point(1197) = {-2.03192, -0.527773, -0.565238, cl};
    Point(1198) = {-2.03192, 0.672227, -0.670793, cl};
    Point(1199) = {-2.03192, 0.894448, -0.784682, cl};
    Point(1200) = {-2.03192, 0.744449, -1.37913, cl};
    Point(1201) = {-2.03192, 0.769449, -1.29024, cl};
    Point(1202) = {-2.03192, 0.844449, -1.27079, cl};
    Point(1203) = {-2.03192, 1.5, -1.09302, cl};
    Point(1204) = {-2.03192, 0.925004, -1.24857, cl};
    Point(1205) = {-2.03192, 1.0, -1.2319, cl};
    Point(1206) = {-2.03192, 1.08056, -1.2069, cl};
    Point(1207) = {-2.03192, 1.15279, -1.18746, cl};
    Point(1208) = {-2.03192, 1.5, -0.970793, cl};
    Point(1209) = {-2.03192, 1.15279, -0.870793, cl};
    Point(1210) = {-2.03192, 0.219449, -0.373571, cl};
    Point(1211) = {-2.03192, 4.88155e-06, -0.340238, cl};
    Point(1212) = {-2.03192, 0.527783, -0.565238, cl};
    Point(1213) = {-2.03192, 0.391672, -0.454127, cl};
    Point(1214) = {-2.03192, -0.461103, -1.92079, cl};
    Point(1215) = {-2.03192, -0.611111, -1.74857, cl};
    Point(1216) = {-2.03192, -0.549998, -1.84024, cl};
    Point(1217) = {-2.03192, -1.5, -0.970793, cl};
    Point(1218) = {-2.03192, -1.5, -1.09302, cl};
    Point(1219) = {-2.03192, -1.15278, -1.18468, cl};
    Point(1220) = {-2.03192, -1.08056, -1.2069, cl};
    Point(1221) = {-2.08803, -3.33333, -0.987544, cl};
    Point(1222) = {-2.08803, -3.33333, -1.09288, cl};
    Point(1223) = {-2.08803, 3.33334, -0.987543, cl};
    Point(1224) = {-2.08803, 3.33334, -1.09288, cl};
    Point(1225) = {-2.15804, -1.97222, -1.09377, cl};
    Point(1226) = {-2.15804, -1.97222, -0.972369, cl};
    Point(1227) = {-2.15804, 1.97222, -1.09377, cl};
    Point(1228) = {-2.15804, 1.97222, -0.972368, cl};
    Point(1229) = {-2.19999, -1.47222, -1.09546, cl};
    Point(1230) = {-2.19999, -1.47222, -0.974741, cl};
    Point(1231) = {-2.19999, 1.47222, -1.09546, cl};
    Point(1232) = {-2.19999, 1.47222, -0.974742, cl};
    Point(1233) = {-2.3466, -1.97222, -1.08873, cl};
    Point(1234) = {-2.3466, -1.97222, -0.980082, cl};
    Point(1235) = {-2.3466, 1.97222, -1.08873, cl};
    Point(1236) = {-2.3466, 1.97222, -0.980082, cl};
    Point(1237) = {-2.3514, -3.33333, -0.986142, cl};
    Point(1238) = {-2.3514, -3.33333, -1.08484, cl};
    Point(1239) = {-2.3514, 3.33333, -0.986141, cl};
    Point(1240) = {-2.3514, 3.33333, -1.08484, cl};
    Point(1241) = {-2.3791, 4.58393e-06, -0.379127, cl};
    Point(1242) = {-2.3791, -0.477773, -1.89857, cl};
    Point(1243) = {-2.3791, -0.330552, -1.97357, cl};
    Point(1244) = {-2.3791, -0.608333, -1.72635, cl};
    Point(1245) = {-2.3791, -0.56389, -1.80135, cl};
    Point(1246) = {-2.3791, -0.7, -1.4819, cl};
    Point(1247) = {-2.3791, -0.652775, -1.61802, cl};
    Point(1248) = {-2.3791, 0.477783, -1.89857, cl};
    Point(1249) = {-2.3791, -0.183328, -2.0069, cl};
    Point(1250) = {-2.3791, -0.00277329, -2.01524, cl};
    Point(1251) = {-2.3791, 0.183338, -2.0069, cl};
    Point(1252) = {-2.3791, 0.330561, -1.97357, cl};
    Point(1253) = {-2.3791, 0.563894, -1.80135, cl};
    Point(1254) = {-2.3791, 0.608337, -1.72635, cl};
    Point(1255) = {-2.3791, 0.652783, -1.61802, cl};
    Point(1256) = {-2.3791, 0.700005, -1.4819, cl};
    Point(1257) = {-2.3791, -1.49722, -0.98746, cl};
    Point(1258) = {-2.3791, -1.5, -1.08468, cl};
    Point(1259) = {-2.3791, -0.730555, -1.37913, cl};
    Point(1260) = {-2.3791, -0.908333, -1.25968, cl};
    Point(1261) = {-2.3791, -0.830556, -1.27913, cl};
    Point(1262) = {-2.3791, -0.75, -1.30135, cl};
    Point(1263) = {-2.3791, -0.988889, -1.23746, cl};
    Point(1264) = {-2.3791, -1.15278, -1.17913, cl};
    Point(1265) = {-2.3791, -1.15, -1.19302, cl};
    Point(1266) = {-2.3791, -1.07778, -1.20968, cl};
    Point(1267) = {-2.3791, -0.672222, -0.668016, cl};
    Point(1268) = {-2.3791, -0.927778, -0.795793, cl};
    Point(1269) = {-2.3791, -1.15278, -0.865238, cl};
    Point(1270) = {-2.3791, -1.15556, -0.901349, cl};
    Point(1271) = {-2.3791, -0.219439, -0.368016, cl};
    Point(1272) = {-2.3791, -0.391659, -0.451349, cl};
    Point(1273) = {-2.3791, -0.536112, -0.568016, cl};
    Point(1274) = {-2.3791, 0.672227, -0.668016, cl};
    Point(1275) = {-2.3791, 0.730561, -1.37913, cl};
    Point(1276) = {-2.3791, 0.750005, -1.30135, cl};
    Point(1277) = {-2.3791, 0.830561, -1.27913, cl};
    Point(1278) = {-2.3791, 1.5, -1.08468, cl};
    Point(1279) = {-2.3791, 0.908338, -1.25968, cl};
    Point(1280) = {-2.3791, 0.988894, -1.23746, cl};
    Point(1281) = {-2.3791, 1.07778, -1.20968, cl};
    Point(1282) = {-2.3791, 1.15001, -1.19302, cl};
    Point(1283) = {-2.3791, 1.15279, -1.17913, cl};
    Point(1284) = {-2.3791, 1.49722, -0.98746, cl};
    Point(1285) = {-2.3791, 0.927783, -0.795793, cl};
    Point(1286) = {-2.3791, 1.15557, -0.901349, cl};
    Point(1287) = {-2.3791, 1.15279, -0.865238, cl};
    Point(1288) = {-2.3791, 0.219449, -0.368016, cl};
    Point(1289) = {-2.3791, 4.58782e-06, -0.334682, cl};
    Point(1290) = {-2.3791, 0.536116, -0.568016, cl};
    Point(1291) = {-2.3791, 0.391672, -0.451349, cl};
    Point(1292) = {-2.4097, -1.47222, -1.09057, cl};
    Point(1293) = {-2.4097, -1.47222, -0.985628, cl};
    Point(1294) = {-2.4097, 1.47222, -1.09057, cl};
    Point(1295) = {-2.4097, 1.47222, -0.985627, cl};

    Point(1296) = {-2.43471, -5.0, -1.08746, cl};
    Point(1297) = {-2.43471, 5.0, -1.08746, cl};
    Point(1298) = {-2.4655, -5.0, -1.07476, cl};
    Point(1299) = {-2.4655, -5.0, -1.09143, cl};
    Point(1300) = {-2.4655, 5.00001, -1.07476, cl};
    Point(1301) = {-2.4655, 5.00001, -1.09143, cl};
    Point(1302) = {-2.4964, -5.0, -1.06817, cl};
    Point(1303) = {-2.4964, -5.0, -1.09114, cl};
    Point(1304) = {-2.4964, 5.00001, -1.06817, cl};
    Point(1305) = {-2.4964, 5.00001, -1.09114, cl};
    Point(1306) = {-2.5272, -5.0, -1.06216, cl};
    Point(1307) = {-2.5272, -5.0, -1.09082, cl};
    Point(1308) = {-2.5272, 5.00001, -1.06216, cl};
    Point(1309) = {-2.5272, 5.00001, -1.09082, cl};
    Point(1310) = {-2.5355, -1.97222, -1.08399, cl};
    Point(1311) = {-2.5355, -1.97222, -0.989542, cl};
    Point(1312) = {-2.5355, 1.97222, -1.08399, cl};
    Point(1313) = {-2.5355, 1.97222, -0.989541, cl};
    Point(1314) = {-2.558, -5.0, -1.05773, cl};
    Point(1315) = {-2.558, -5.0, -1.0893, cl};
    Point(1316) = {-2.558, 5.00001, -1.05773, cl};
    Point(1317) = {-2.558, 5.00001, -1.0893, cl};
    Point(1318) = {-2.6147, -3.33333, -0.990505, cl};
    Point(1319) = {-2.6147, -3.33333, -1.07519, cl};
    Point(1320) = {-2.6147, 3.33334, -0.990504, cl};
    Point(1321) = {-2.6147, 3.33334, -1.07519, cl};
    Point(1322) = {-2.6197, -5.0, -1.04921, cl};
    Point(1323) = {-2.6197, -5.0, -1.08682, cl};
    Point(1324) = {-2.6197, 5.0, -1.04921, cl};
    Point(1325) = {-2.6197, 5.00001, -1.08682, cl};
    Point(1326) = {-2.6814, 5.0, -1.04188, cl};
    Point(1327) = {-2.6814, 5.0, -1.08404, cl};
    Point(1328) = {-2.6814, -5.0, -1.04188, cl};
    Point(1329) = {-2.6814, -5.0, -1.08404, cl};
    Point(1330) = {-2.7264, -0.333325, -1.96246, cl};
    Point(1331) = {-2.7264, -0.486104, -1.87913, cl};
    Point(1332) = {-2.7264, -0.552776, -1.80135, cl};
    Point(1333) = {-2.7264, -0.608332, -1.71524, cl};
    Point(1334) = {-2.7264, -0.644442, -1.61246, cl};
    Point(1335) = {-2.7264, -0.686111, -1.49579, cl};
    Point(1336) = {-2.7264, 0.486116, -1.87913, cl};
    Point(1337) = {-2.7264, 4.80125e-06, -2.00413, cl};
    Point(1338) = {-2.7264, -0.186106, -1.99579, cl};
    Point(1339) = {-2.7264, 0.333338, -1.96246, cl};
    Point(1340) = {-2.7264, 0.186116, -1.99579, cl};
    Point(1341) = {-2.7264, 0.608338, -1.71524, cl};
    Point(1342) = {-2.7264, 0.552783, -1.80135, cl};
    Point(1343) = {-2.7264, 0.686116, -1.49579, cl};
    Point(1344) = {-2.7264, 0.644449, -1.61246, cl};
    Point(1345) = {-2.7264, -1.50556, -1.07913, cl};
    Point(1346) = {-2.7264, -1.50278, -1.0069, cl};
    Point(1347) = {-2.7264, -0.719444, -1.36524, cl};
    Point(1348) = {-2.7264, -0.9, -1.2569, cl};
    Point(1349) = {-2.7264, -0.738889, -1.30135, cl};
    Point(1350) = {-2.7264, -0.819444, -1.2819, cl};
    Point(1351) = {-2.7264, -0.980556, -1.23746, cl};
    Point(1352) = {-2.7264, -1.15833, -1.15968, cl};
    Point(1353) = {-2.7264, -1.06389, -1.21524, cl};
    Point(1354) = {-2.7264, -1.15556, -1.19302, cl};
    Point(1355) = {-2.7264, -0.738889, -0.698571, cl};
    Point(1356) = {-2.7264, -0.958333, -0.795793, cl};
    Point(1357) = {-2.7264, -1.15278, -0.93746, cl};
    Point(1358) = {-2.7264, -1.15278, -0.856904, cl};
    Point(1359) = {-2.7264, -0.238884, -0.368016, cl};
    Point(1360) = {-2.7264, -0.538884, -0.559682, cl};
    Point(1361) = {-2.7264, -0.408329, -0.451349, cl};
    Point(1362) = {-2.7264, 0.900005, -1.2569, cl};
    Point(1363) = {-2.7264, 0.738894, -0.698571, cl};
    Point(1364) = {-2.7264, 0.719449, -1.36524, cl};
    Point(1365) = {-2.7264, 0.819449, -1.2819, cl};
    Point(1366) = {-2.7264, 0.738894, -1.30135, cl};
    Point(1367) = {-2.7264, 1.50556, -1.07913, cl};
    Point(1368) = {-2.7264, 0.980561, -1.23746, cl};
    Point(1369) = {-2.7264, 1.06389, -1.21524, cl};
    Point(1370) = {-2.7264, 1.15833, -1.15968, cl};
    Point(1371) = {-2.7264, 1.15557, -1.19302, cl};
    Point(1372) = {-2.7264, 1.50278, -1.0069, cl};
    Point(1373) = {-2.7264, 0.958338, -0.795793, cl};
    Point(1374) = {-2.7264, 1.15279, -0.856904, cl};
    Point(1375) = {-2.7264, 1.15279, -0.93746, cl};
    Point(1376) = {-2.7264, -0.00277344, -0.329127, cl};
    Point(1377) = {-2.7264, 0.238894, -0.368016, cl};
    Point(1378) = {-2.7264, 0.408338, -0.451349, cl};
    Point(1379) = {-2.7264, 0.538894, -0.559682, cl};
    Point(1380) = {-2.7464, -3.33333, -0.994483, cl};
    Point(1381) = {-2.7464, -3.33333, -1.07027, cl};
    Point(1382) = {-2.7464, 3.33333, -0.994482, cl};
    Point(1383) = {-2.7464, 3.33333, -1.07027, cl};
    Point(1384) = {-2.8047, -5.0, -1.03037, cl};
    Point(1385) = {-2.8047, -5.0, -1.0781, cl};
    Point(1386) = {-2.8047, 5.0, -1.03037, cl};
    Point(1387) = {-2.8047, 5.0, -1.0781, cl};
    Point(1388) = {-2.8291, -1.47222, -1.08372, cl};
    Point(1389) = {-2.8291, -1.47222, -1.01309, cl};
    Point(1390) = {-2.8291, 1.47222, -1.08372, cl};
    Point(1391) = {-2.8291, 1.47222, -1.01309, cl};
    Point(1392) = {-2.878, -3.33333, -0.999657, cl};
    Point(1393) = {-2.878, -3.33333, -1.06554, cl};
    Point(1394) = {-2.878, 3.33333, -0.999658, cl};
    Point(1395) = {-2.878, 3.33333, -1.06554, cl};
    Point(1396) = {-2.913, -1.97222, -1.07694, cl};
    Point(1397) = {-2.913, -1.97222, -1.01337, cl};
    Point(1398) = {-2.913, 1.97222, -1.07694, cl};
    Point(1399) = {-2.913, 1.97222, -1.01337, cl};
    Point(1400) = {-2.928, 5.0, -1.02113, cl};
    Point(1401) = {-2.928, 5.0, -1.07046, cl};
    Point(1402) = {-2.928, -5.0, -1.02113, cl};
    Point(1403) = {-2.928, -5.0, -1.07046, cl};

    Point(1406) = {-3.03891, -1.47222, -1.08387, cl};
    Point(1407) = {-3.03891, -1.47222, -1.03077, cl};
    Point(1408) = {-3.03891, 1.47222, -1.08387, cl};
    Point(1409) = {-3.03891, 1.47222, -1.03077, cl};
    Point(1410) = {-3.0514, -5.0, -1.01557, cl};
    Point(1411) = {-3.0514, -5.0, -1.0618, cl};
    Point(1412) = {-3.0514, 5.00001, -1.01557, cl};
    Point(1413) = {-3.0514, 5.00001, -1.0618, cl};
    Point(1414) = {-3.0736, -0.449993, -1.89024, cl};
    Point(1415) = {-3.0736, -0.316659, -1.95968, cl};
    Point(1416) = {-3.0736, -0.597222, -1.70413, cl};
    Point(1417) = {-3.0736, -0.536111, -1.80968, cl};
    Point(1418) = {-3.0736, -0.675, -1.49024, cl};
    Point(1419) = {-3.0736, -0.641667, -1.59857, cl};
    Point(1420) = {-3.0736, 0.450005, -1.89024, cl};
    Point(1421) = {-3.0736, 0.152783, -1.99024, cl};
    Point(1422) = {-3.0736, 0.316672, -1.95968, cl};
    Point(1423) = {-3.0736, -0.152773, -1.99024, cl};
    Point(1424) = {-3.0736, 4.83456e-06, -1.99579, cl};
    Point(1425) = {-3.0736, 0.536116, -1.80968, cl};
    Point(1426) = {-3.0736, 0.597227, -1.70413, cl};
    Point(1427) = {-3.0736, 0.641672, -1.59857, cl};
    Point(1428) = {-3.0736, 0.675005, -1.49024, cl};
    Point(1429) = {-3.0736, -1.50278, -1.03468, cl};
    Point(1430) = {-3.0736, -1.5, -1.07913, cl};
    Point(1431) = {-3.0736, -1.15556, -0.990238, cl};
    Point(1432) = {-3.0736, -0.977778, -1.24024, cl};
    Point(1433) = {-3.0736, -0.894444, -1.2569, cl};
    Point(1434) = {-3.0736, -0.808333, -1.28468, cl};
    Point(1435) = {-3.0736, -1.15278, -1.13746, cl};
    Point(1436) = {-3.0736, -1.15278, -1.19579, cl};
    Point(1437) = {-3.0736, -1.07222, -1.21802, cl};
    Point(1438) = {-3.0736, -0.724999, -1.30413, cl};
    Point(1439) = {-3.0736, -0.705555, -1.37913, cl};
    Point(1440) = {-3.0736, -1.15, -0.854127, cl};
    Point(1441) = {-3.0736, -0.736111, -0.701349, cl};
    Point(1442) = {-3.0736, -0.938889, -0.784682, cl};
    Point(1443) = {-3.0736, -0.230551, -0.365238, cl};
    Point(1444) = {-3.0736, -0.399996, -0.445793, cl};
    Point(1445) = {-3.0736, -0.541666, -0.565238, cl};
    Point(1446) = {-3.0736, 0.736116, -0.701349, cl};
    Point(1447) = {-3.0736, 0.705561, -1.37913, cl};
    Point(1448) = {-3.0736, 0.725005, -1.30413, cl};
    Point(1449) = {-3.0736, 0.808338, -1.28468, cl};
    Point(1450) = {-3.0736, 0.894449, -1.2569, cl};
    Point(1451) = {-3.0736, 1.50278, -1.03468, cl};
    Point(1452) = {-3.0736, 1.15556, -0.990239, cl};
    Point(1453) = {-3.0736, 1.5, -1.07913, cl};
    Point(1454) = {-3.0736, 0.977782, -1.24024, cl};
    Point(1455) = {-3.0736, 1.07223, -1.21802, cl};
    Point(1456) = {-3.0736, 1.15279, -1.19579, cl};
    Point(1457) = {-3.0736, 1.15278, -1.13746, cl};
    Point(1458) = {-3.0736, 1.15001, -0.854127, cl};
    Point(1459) = {-3.0736, 0.938894, -0.784682, cl};
    Point(1460) = {-3.0736, 0.230561, -0.365238, cl};
    Point(1461) = {-3.0736, -0.00277341, -0.329127, cl};
    Point(1462) = {-3.0736, 0.541672, -0.565238, cl};
    Point(1463) = {-3.0736, 0.400005, -0.445793, cl};
    Point(1464) = {-3.1016, -0.472216, -1.79857, cl};
    Point(1465) = {-3.1016, 0.472227, -1.79857, cl};
    Point(1466) = {-3.1016, -1.97222, -1.07666, cl};
    Point(1467) = {-3.1016, -1.97222, -1.02887, cl};
    Point(1468) = {-3.1016, 1.97222, -1.07666, cl};
    Point(1469) = {-3.1016, 1.97222, -1.02887, cl};
    Point(1470) = {-3.1414, -3.33333, -1.01357, cl};
    Point(1471) = {-3.1414, -3.33333, -1.05791, cl};
    Point(1472) = {-3.1414, 3.33334, -1.01357, cl};
    Point(1473) = {-3.1414, 3.33334, -1.05791, cl};
    Point(1474) = {-3.1747, 5.0, -1.01277, cl};
    Point(1475) = {-3.1747, 5.0, -1.05243, cl};
    Point(1476) = {-3.1747, -5.0, -1.01277, cl};
    Point(1477) = {-3.1747, -5.0, -1.05243, cl};

    Point(1478) = {-3.2364, -5.0, -1.01219, cl};
    Point(1479) = {-3.2364, -5.0, -1.04768, cl};
    Point(1480) = {-3.2364, 5.0, -1.01218, cl};
    Point(1481) = {-3.2364, 5.0, -1.04768, cl};
    Point(1482) = {-3.2486, -1.47222, -1.04914, cl};
    Point(1483) = {-3.2486, -1.47222, -1.08479, cl};
    Point(1484) = {-3.2486, 1.47222, -1.04914, cl};
    Point(1485) = {-3.2486, 1.47222, -1.08479, cl};
    Point(1486) = {-3.273, -3.33333, -1.02299, cl};
    Point(1487) = {-3.273, -3.33333, -1.05632, cl};
    Point(1488) = {-3.273, 3.33334, -1.02298, cl};
    Point(1489) = {-3.273, 3.33334, -1.05632, cl};
    Point(1490) = {-3.2905, -1.97222, -1.04503, cl};
    Point(1491) = {-3.2905, -1.97222, -1.07712, cl};
    Point(1492) = {-3.2905, 1.97222, -1.04503, cl};
    Point(1493) = {-3.2905, 1.97222, -1.07712, cl};
    Point(1494) = {-3.29751, -0.463828, -1.80082, cl};
    Point(1495) = {-3.29751, -0.480607, -1.79632, cl};
    Point(1496) = {-3.29751, 0.480616, -1.79632, cl};
    Point(1497) = {-3.29751, 0.463838, -1.80082, cl};
    Point(1498) = {-3.298, -5.0, -1.01217, cl};
    Point(1499) = {-3.298, -5.0, -1.04303, cl};
    Point(1500) = {-3.298, 5.0, -1.01217, cl};
    Point(1501) = {-3.298, 5.0, -1.04303, cl};
    Point(1502) = {-3.4047, -3.33333, -1.03285, cl};
    Point(1503) = {-3.4047, -3.33333, -1.05524, cl};
    Point(1504) = {-3.4047, 3.33333, -1.03285, cl};
    Point(1505) = {-3.4047, 3.33333, -1.05524, cl};
    Point(1506) = {-3.4208, -0.308325, -1.94857, cl};
    Point(1507) = {-3.4208, -0.430548, -1.89302, cl};
    Point(1508) = {-3.4208, -0.586111, -1.71524, cl};
    Point(1509) = {-3.4208, -0.522222, -1.81246, cl};
    Point(1510) = {-3.4208, -0.672222, -1.47913, cl};
    Point(1511) = {-3.4208, -0.636111, -1.60135, cl};
    Point(1512) = {-3.4208, 0.430561, -1.89302, cl};
    Point(1513) = {-3.4208, -0.158328, -1.97913, cl};
    Point(1514) = {-3.4208, 0.0027826, -1.98746, cl};
    Point(1515) = {-3.4208, 0.158338, -1.97913, cl};
    Point(1516) = {-3.4208, 0.308338, -1.94857, cl};
    Point(1517) = {-3.4208, 0.522227, -1.81246, cl};
    Point(1518) = {-3.4208, 0.586116, -1.71524, cl};
    Point(1519) = {-3.4208, 0.636116, -1.60135, cl};
    Point(1520) = {-3.4208, 0.672227, -1.47913, cl};
    Point(1521) = {-3.4208, -1.50278, -1.06524, cl};
    Point(1522) = {-3.4208, -1.5, -1.08468, cl};
    Point(1523) = {-3.4208, -1.15556, -1.04579, cl};
    Point(1524) = {-3.4208, -0.891667, -1.2569, cl};
    Point(1525) = {-3.4208, -0.805556, -1.27635, cl};
    Point(1526) = {-3.4208, -0.980556, -1.23746, cl};
    Point(1527) = {-3.4208, -1.15556, -1.11246, cl};
    Point(1528) = {-3.4208, -1.15278, -1.19857, cl};
    Point(1529) = {-3.4208, -1.06111, -1.21802, cl};
    Point(1530) = {-3.4208, -0.719444, -1.29579, cl};
    Point(1531) = {-3.4208, -0.697221, -1.37913, cl};
    Point(1532) = {-3.4208, -1.15, -0.856904, cl};
    Point(1533) = {-3.4208, -0.769444, -0.720793, cl};
    Point(1534) = {-3.4208, -0.952778, -0.795793, cl};
    Point(1535) = {-3.4208, -0.216662, -0.359682, cl};
    Point(1536) = {-3.4208, -0.427774, -0.46246, cl};
    Point(1537) = {-3.4208, -0.555556, -0.581904, cl};
    Point(1538) = {-3.4208, 0.769449, -0.720793, cl};
    Point(1539) = {-3.4208, 0.697227, -1.37913, cl};
    Point(1540) = {-3.4208, 0.719449, -1.29579, cl};
    Point(1541) = {-3.4208, 0.805561, -1.27635, cl};
    Point(1542) = {-3.4208, 0.891672, -1.2569, cl};
    Point(1543) = {-3.4208, 1.15557, -1.04579, cl};
    Point(1544) = {-3.4208, 1.15556, -1.11246, cl};
    Point(1545) = {-3.4208, 0.980561, -1.23746, cl};
    Point(1546) = {-3.4208, 1.06111, -1.21802, cl};
    Point(1547) = {-3.4208, 1.15279, -1.19857, cl};
    Point(1548) = {-3.4208, 1.5, -1.08468, cl};
    Point(1549) = {-3.4208, 1.50278, -1.06524, cl};
    Point(1550) = {-3.4208, 0.952783, -0.795793, cl};
    Point(1551) = {-3.4208, 1.15001, -0.856904, cl};
    Point(1552) = {-3.4208, 0.216672, -0.359682, cl};
    Point(1553) = {-3.4208, 4.68629e-06, -0.326349, cl};
    Point(1554) = {-3.4208, 0.555561, -0.581904, cl};
    Point(1555) = {-3.4208, 0.427783, -0.46246, cl};
    Point(1556) = {-3.4214, -5.0, -1.0138, cl};
    Point(1557) = {-3.4214, -5.0, -1.03457, cl};
    Point(1558) = {-3.4214, 5.0, -1.0138, cl};
    Point(1559) = {-3.4214, 5.0, -1.03457, cl};
    Point(1560) = {-3.4583, -1.47222, -1.06811, cl};
    Point(1561) = {-3.4583, -1.47222, -1.08631, cl};
    Point(1562) = {-3.4583, 1.47222, -1.06811, cl};
    Point(1563) = {-3.4583, 1.47222, -1.08631, cl};
    Point(1564) = {-3.4791, -1.97222, -1.06171, cl};
    Point(1565) = {-3.4791, -1.97222, -1.0781, cl};
    Point(1566) = {-3.4791, 1.97222, -1.06171, cl};
    Point(1567) = {-3.4791, 1.97222, -1.0781, cl};
    Point(1568) = {-3.483, -5.0, -1.01579, cl};
    Point(1569) = {-3.483, -5.0, -1.03141, cl};
    Point(1570) = {-3.483, 5.0, -1.01579, cl};
    Point(1571) = {-3.483, 5.0, -1.03141, cl};
    Point(1572) = {-3.4939, 0.486616, -1.79471, cl};
    Point(1573) = {-3.4939, 0.457838, -1.80243, cl};
    Point(1574) = {-3.4939, -0.457829, -1.80243, cl};
    Point(1575) = {-3.4939, -0.486607, -1.79471, cl};
    Point(1578) = {-3.52971, -0.583111, -2.21246, cl};
    Point(1579) = {-3.52971, 0.583116, -2.21246, cl};
    Point(1580) = {-3.5364, -3.33333, -1.04311, cl};
    Point(1581) = {-3.5364, -3.33333, -1.05454, cl};
    Point(1582) = {-3.5364, 3.33333, -1.04311, cl};
    Point(1583) = {-3.5364, 3.33333, -1.05454, cl};
    Point(1584) = {-3.5447, -5.0, -1.01783, cl};
    Point(1585) = {-3.5447, -5.0, -1.02831, cl};
    Point(1586) = {-3.5447, 5.00001, -1.01783, cl};
    Point(1587) = {-3.5447, 5.00001, -1.02831, cl};
    Point(1588) = {-3.6064, 5.00001, -1.02034, cl};
    Point(1589) = {-3.6064, -5.0, -1.02034, cl};

    Point(1590) = {-3.668, -3.33333, -1.0536, cl};
    Point(1591) = {-3.668, -5.0, -1.02279, cl};
    Point(1592) = {-3.668, -1.47222, -1.08746, cl};
    Point(1593) = {-3.668, -1.97222, -1.07874, cl};
    Point(1594) = {-3.668, 1.47222, -1.08746, cl};
    Point(1595) = {-3.668, 1.97222, -1.07874, cl};
    Point(1596) = {-3.668, 3.33333, -1.0536, cl};
    Point(1597) = {-3.668, 5.00001, -1.02279, cl};
    Point(1598) = {-3.66941, -0.458326, -1.85968, cl};
    Point(1599) = {-3.66941, -0.322218, -1.9319, cl};
    Point(1600) = {-3.66941, -0.541667, -1.77357, cl};
    Point(1601) = {-3.66941, -0.591666, -1.69024, cl};
    Point(1602) = {-3.66941, -0.636108, -1.59302, cl};
    Point(1603) = {-3.66941, -0.669444, -1.47635, cl};
    Point(1604) = {-3.66941, 0.458338, -1.85968, cl};
    Point(1605) = {-3.66941, -0.186106, -1.96524, cl};
    Point(1606) = {-3.66941, 4.56513e-06, -1.97913, cl};
    Point(1607) = {-3.66941, 0.322227, -1.9319, cl};
    Point(1608) = {-3.66941, 0.186116, -1.96524, cl};
    Point(1609) = {-3.66941, 0.591672, -1.69024, cl};
    Point(1610) = {-3.66941, 0.541672, -1.77357, cl};
    Point(1611) = {-3.66941, 0.669449, -1.47635, cl};
    Point(1612) = {-3.66941, 0.636116, -1.59302, cl};
    Point(1613) = {-3.66941, -1.5, -1.08468, cl};
    Point(1614) = {-3.66941, -0.888889, -1.25135, cl};
    Point(1615) = {-3.66941, -0.802778, -1.27079, cl};
    Point(1616) = {-3.66941, -0.977778, -1.23468, cl};
    Point(1617) = {-3.66941, -1.15556, -1.08746, cl};
    Point(1618) = {-3.66941, -1.15556, -1.19579, cl};
    Point(1619) = {-3.66941, -1.06944, -1.21246, cl};
    Point(1620) = {-3.66941, -0.694442, -1.38746, cl};
    Point(1621) = {-3.66941, -0.719444, -1.28746, cl};
    Point(1622) = {-3.66941, -1.15278, -0.859682, cl};
    Point(1623) = {-3.66941, -0.936111, -0.790238, cl};
    Point(1624) = {-3.66941, -0.747222, -0.715238, cl};
    Point(1625) = {-3.66941, -0.263884, -0.370793, cl};
    Point(1626) = {-3.66941, -0.572222, -0.598571, cl};
    Point(1627) = {-3.66941, -0.430548, -0.459682, cl};
    Point(1628) = {-3.66941, 0.747227, -0.715238, cl};
    Point(1629) = {-3.66941, 0.694449, -1.38746, cl};
    Point(1630) = {-3.66941, 0.719449, -1.28746, cl};
    Point(1631) = {-3.66941, 0.805561, -1.27079, cl};
    Point(1632) = {-3.66941, 0.888894, -1.25135, cl};
    Point(1633) = {-3.66941, 1.15556, -1.0819, cl};
    Point(1634) = {-3.66941, 0.977783, -1.23468, cl};
    Point(1635) = {-3.66941, 1.06944, -1.21246, cl};
    Point(1636) = {-3.66941, 1.15557, -1.19579, cl};
    Point(1637) = {-3.66941, 1.5, -1.08468, cl};
    Point(1638) = {-3.66941, 0.936116, -0.790238, cl};
    Point(1639) = {-3.66941, 1.15279, -0.859682, cl};
    Point(1640) = {-3.66941, 4.70962e-06, -0.326349, cl};
    Point(1641) = {-3.66941, 0.263894, -0.370793, cl};
    Point(1642) = {-3.66941, 0.430561, -0.459682, cl};
    Point(1643) = {-3.66941, 0.572227, -0.598571, cl};
    Point(1644) = {-3.67251, -0.577028, -2.2141, cl};
    Point(1645) = {-3.67251, -0.579583, -2.2134, cl};
    Point(1646) = {-3.67251, 0.589199, -2.21082, cl};
    Point(1647) = {-3.67251, 0.586644, -2.21152, cl};
    Point(1648) = {-3.69001, -0.454243, -1.80338, cl};
    Point(1649) = {-3.69001, -0.490186, -1.79377, cl};
    Point(1650) = {-3.69001, 0.490199, -1.79377, cl};
    Point(1651) = {-3.69001, 0.454255, -1.80338, cl};

    Point(1652) = {-3.7514, -0.640444, -2.42643, cl};
    Point(1653) = {-3.7514, 0.640449, -2.42643, cl};
    Point(1654) = {-3.768, -0.430548, -1.87635, cl};
    Point(1655) = {-3.768, -0.519441, -1.79579, cl};
    Point(1656) = {-3.768, -0.580556, -1.7069, cl};
    Point(1657) = {-3.768, 0.430561, -1.87635, cl};
    Point(1658) = {-3.768, 0.00278264, -1.97913, cl};
    Point(1659) = {-3.768, 0.291672, -1.94024, cl};
    Point(1660) = {-3.768, 0.150005, -1.97079, cl};
    Point(1661) = {-3.768, 0.580561, -1.7069, cl};
    Point(1662) = {-3.768, 0.666672, -1.48468, cl};
    Point(1663) = {-3.768, 0.625005, -1.6069, cl};
    Point(1664) = {-3.768, -0.786111, -1.27635, cl};
    Point(1665) = {-3.768, -1.04444, -1.22357, cl};
    Point(1666) = {-3.768, -1.15556, -1.20413, cl};
    Point(1667) = {-3.768, -0.691667, -1.37635, cl};
    Point(1668) = {-3.768, -0.716667, -1.29024, cl};
    Point(1669) = {-3.768, -0.733332, -0.71246, cl};
    Point(1670) = {-3.768, -1.15278, -0.865238, cl};
    Point(1671) = {-3.768, -0.941667, -0.798571, cl};
    Point(1672) = {-3.768, -0.244439, -0.370793, cl};
    Point(1673) = {-3.768, -0.422214, -0.46246, cl};
    Point(1674) = {-3.768, 0.733338, -0.71246, cl};
    Point(1675) = {-3.768, 0.691672, -1.37635, cl};
    Point(1676) = {-3.768, 0.866672, -1.25968, cl};
    Point(1677) = {-3.768, 0.958338, -1.24024, cl};
    Point(1678) = {-3.768, 1.15556, -1.0819, cl};
    Point(1679) = {-3.768, 1.15556, -1.20413, cl};
    Point(1680) = {-3.768, 1.04445, -1.22357, cl};
    Point(1681) = {-3.768, 0.941672, -0.798571, cl};
    Point(1682) = {-3.768, 1.15278, -0.865238, cl};
    Point(1683) = {-3.768, -0.00277334, -0.323571, cl};
    Point(1684) = {-3.768, 0.244449, -0.370793, cl};
    Point(1685) = {-3.768, 0.422227, -0.46246, cl};
    Point(1686) = {-3.768, 0.550005, -0.581904, cl};
    Point(1687) = {-3.76801, -0.291659, -1.94024, cl};
    Point(1688) = {-3.76801, -0.625, -1.6069, cl};
    Point(1689) = {-3.76801, -0.666667, -1.48468, cl};
    Point(1690) = {-3.76801, -0.149995, -1.97079, cl};
    Point(1691) = {-3.76801, 0.519449, -1.79579, cl};
    Point(1692) = {-3.76801, -0.958333, -1.24024, cl};
    Point(1693) = {-3.76801, -0.866667, -1.25968, cl};
    Point(1694) = {-3.76801, -1.15556, -1.0819, cl};
    Point(1695) = {-3.76801, -0.549997, -0.581904, cl};
    Point(1696) = {-3.76801, 0.716672, -1.29024, cl};
    Point(1697) = {-3.76801, 0.786116, -1.27635, cl};
    Point(1698) = {-3.815, -0.577083, -2.21407, cl};
    Point(1699) = {-3.815, -0.589139, -2.21085, cl};
    Point(1700) = {-3.815, 0.589144, -2.21085, cl};
    Point(1701) = {-3.815, 0.577088, -2.21407, cl};

    Delete {
    Point{755,754,809,909,810,935,1017,914,940,1151,1203,1278,1156,1367,1208,1284};
    }
    Delete {
    Point{1453,1548,1372};
    }
    Delete {
    Point{1451};
    }
    Delete {
    Point{1549};
    }
    Delete {
    Point{1637};
    }
    Delete {
    Point{781,844,889,782,963,843,1133,1218,890,1258,1345,1613,962,1430};
    }
    Delete {
    Point{1522,1429,1521,1132,1346};
    }
    Delete {
    Point{1217};
    }
    Delete {
    Point{1257};
    }
    Line(256) = {1034,1297};
    Line(257) = {1594,1595};
    Line(258) = {1595,1596};
    Line(259) = {1596,1597};
    Line(260) = {1033,1296};
    Line(261) = {1591,1590};
    Line(262) = {1590,1593};
    Line(263) = {1593,1592};
    CatmullRom(264) = {340,393,444,494};
    CatmullRom(265) = {335,390,439,489};
    CatmullRom(266) = {308,372,422,471};
    CatmullRom(267) = {494,493,497,496,489};
    CatmullRom(268) = {494,495,475,476,471};
    CatmullRom(269) = {494,547,598};
    CatmullRom(270) = {489,549,602};
    CatmullRom(271) = {471,530,565};
    CatmullRom(272) = {598,599,600,601,602};
    CatmullRom(273) = {598,597,563,564,565};
    CatmullRom(274) = {602,646,698};
    CatmullRom(275) = {565,654,698};
    CatmullRom(276) = {598,650,698};
    CatmullRom(277) = {322,380,428,477};
    CatmullRom(278) = {288,360,413,460};
    CatmullRom(279) = {460,521,567};
    CatmullRom(280) = {477,502,590};
    CatmullRom(281) = {567,618,670};
    CatmullRom(282) = {590,638,690};
    CatmullRom(283) = {670,714,743};
    CatmullRom(284) = {690,733,744};
    CatmullRom(285) = {743,807,1033};
    CatmullRom(286) = {744,804,1034};
    CatmullRom(287) = {698,739,763};
    CatmullRom(291) = {316,281};
    CatmullRom(292) = {278,352,404,453};
    CatmullRom(293) = {281,342,394,466};
    CatmullRom(294) = {281,354,405,480};
    CatmullRom(295) = {453,515,559};
    CatmullRom(296) = {559,611,663};
    CatmullRom(297) = {663,707,773};
    CatmullRom(301) = {480,538,583};
    CatmullRom(302) = {583,633,684};
    CatmullRom(303) = {583,582};
    CatmullRom(304) = {684,686};
    CatmullRom(305) = {751,752};
    CatmullRom(306) = {684,728,751};
    CatmullRom(307) = {686,730,752};
    CatmullRom(308) = {582,634,686};
    CatmullRom(321) = {466,500,574};
    CatmullRom(322) = {574,620,671};
    CatmullRom(323) = {574,572};
    CatmullRom(324) = {671,672};
    CatmullRom(325) = {671,715,783};
    CatmullRom(326) = {783,789};
    CatmullRom(327) = {672,720,789};
    CatmullRom(340) = {672,619,572};
    CatmullRom(341) = {572,526,465};
    CatmullRom(342) = {465,466};
    CatmullRom(343) = {316,359,411,465};
    CatmullRom(344) = {481,480};
    CatmullRom(345) = {481,536,582};
    CatmullRom(346) = {316,377,431,481};
    Line(347) = {1169,1386};
    Line(348) = {1169,1032};
    Line(349) = {1031,1170};
    Line(350) = {1387,1170};
    CatmullRom(351) = {804,855,859,870,872,928,975,1032};
    CatmullRom(352) = {804,856,860,869,871,927,976,1031};
    CatmullRom(353) = {1032,1104,1166,1228,1236,1313,1399,1469,1492,1566,1595};
    CatmullRom(354) = {1031,1103,1165,1227,1235,1312,1398,1468,1493,1567,1595};
    CatmullRom(355) = {1034,1037,1043,1049,1053,1107,1111,1169};
    CatmullRom(356) = {1034,1038,1044,1050,1054,1108,1112,1170};
    CatmullRom(357) = {1169,1223,1239,1320,1382,1394,1472,1488,1504,1582,1596};
    CatmullRom(358) = {1170,1224,1240,1321,1383,1395,1473,1489,1505,1583,1596};
    CatmullRom(359) = {1297,1300,1304,1308,1316,1324,1326,1386};
    CatmullRom(360) = {1386,1400,1412,1474,1480,1500,1558,1570,1586,1588,1597};
    CatmullRom(361) = {1297,1301,1305,1309,1317,1325,1327,1327,1387};
    CatmullRom(362) = {1387,1401,1413,1475,1481,1501,1559,1571,1587,1597};
    Line(363) = {1030,1167};
    Line(364) = {1384,1167};
    Line(365) = {1029,1168};
    Line(366) = {1168,1385};
    CatmullRom(367) = {807,857,861,868,874,926,977,1030};
    CatmullRom(368) = {807,858,862,867,873,925,978,1029};
    CatmullRom(369) = {1030,1102,1164,1226,1234,1311,1397,1467,1490,1564,1593};
    CatmullRom(370) = {1593,1565,1491,1466,1396,1310,1233,1225,1163,1101,1029};
    CatmullRom(371) = {1033,1035,1045,1047,1051,1105,1109,1167};
    CatmullRom(372) = {1033,1036,1046,1048,1052,1106,1110,1168};
    CatmullRom(373) = {1296,1298,1302,1306,1314,1322,1328,1384};
    CatmullRom(374) = {1296,1299,1303,1307,1315,1323,1329,1385};
    CatmullRom(375) = {1167,1221,1237,1318,1380,1392,1470,1486,1502,1580,1590};
    CatmullRom(376) = {1590,1581,1503,1487,1471,1393,1381,1319,1238,1222,1168};
    CatmullRom(377) = {1384,1402,1410,1476,1478,1498,1556,1568,1584,1589,1591};
    CatmullRom(378) = {1591,1585,1569,1557,1499,1479,1477,1411,1403,1385};
    Line(388) = {1032,982};
    Line(389) = {1030,980};
    Line(390) = {1029,979};
    Line(391) = {1031,981};
    CatmullRom(392) = {744,747,797,802,806,866,923,982};
    CatmullRom(393) = {744,748,798,801,805,865,924,981};
    CatmullRom(394) = {743,745,795,800,803,864,921,980};
    CatmullRom(395) = {743,746,796,799,808,863,922,979};
    CatmullRom(396) = {982,1042,1116,1174,1232,1295,1391,1409,1484,1562,1594};
    CatmullRom(397) = {981,1041,1115,1173,1231,1294,1390,1408,1485,1563,1594};
    CatmullRom(398) = {980,1040,1114,1172,1230,1293,1389,1407,1482,1560,1592};
    CatmullRom(399) = {1592,1561,1483,1406,1388,1292,1229,1171,1113,1039,979};
    CatmullRom(400) = {763,749,793,794};
    CatmullRom(401) = {763,762,765,764};
    CatmullRom(402) = {794,790,791,792,743};
    CatmullRom(403) = {764,750,760,761,744};
    CatmullRom(404) = {698,683,682,681};
    CatmullRom(405) = {681,678,679,680,670};
    CatmullRom(406) = {565,578,580,579};
    CatmullRom(407) = {579,575,577,576,567};
    CatmullRom(408) = {471,472,473,474};
    CatmullRom(409) = {474,468,469,470,460};
    CatmullRom(410) = {698,695,696,697};
    CatmullRom(411) = {697,687,688,694,690};
    CatmullRom(412) = {602,593,594,595};
    CatmullRom(413) = {595,588,589,592,590};
    CatmullRom(414) = {489,490,492,491};
    CatmullRom(415) = {491,488,487,486,477};
    CatmullRom(416) = {763,824,917,944};
    CatmullRom(417) = {944,929,973,974};
    CatmullRom(418) = {944,943,946,945};
    CatmullRom(419) = {974,971,972,961,980};
    CatmullRom(420) = {945,931,942,941,982};
    CatmullRom(421) = {974,902,835,794};
    CatmullRom(422) = {945,920,826,764};
    CatmullRom(423) = {794,725,681};
    CatmullRom(424) = {681,629,579};
    CatmullRom(425) = {579,532,474};
    CatmullRom(426) = {764,742,697};
    CatmullRom(427) = {697,644,595};
    CatmullRom(428) = {595,505,491};
    Delete {
    Line{111};
    }
    Delete {
    Line{100};
    }
    Delete {
    Line{84};
    }
    Delete {
    Line{73};
    }
    Delete {
    Line{106};
    }
    Delete {
    Line{92};
    }
    Delete {
    Line{82};
    }
    Delete {
    Line{66};
    }
    CatmullRom(429) = {167,165,166,168};
    CatmullRom(430) = {233,234,235,231};
    CatmullRom(431) = {219,221,220,214};
    CatmullRom(432) = {180,179,178,185};
    CatmullRom(433) = {752,817,906,933};
    CatmullRom(434) = {933,932};
    CatmullRom(435) = {932,907,815,751};
    CatmullRom(436) = {789,851,891,965};
    CatmullRom(437) = {965,966};
    CatmullRom(438) = {966,893,845,783};
    CatmullRom(439) = {773,842,884,954};
    CatmullRom(440) = {465,464,463,459,462,461,460};
    CatmullRom(441) = {572,571,566,569,570,568,567};
    CatmullRom(442) = {672,676,677,675,673,674,670};
    CatmullRom(443) = {481,482,483,484,485,478,477};
    CatmullRom(444) = {582,585,587,586,581,591,590};
    CatmullRom(445) = {686,685,689,691,692,693,690};
    CatmullRom(446) = {944,1025,1098,1159,1211};
    CatmullRom(447) = {1211,1210,1213,1212};
    CatmullRom(448) = {1211,1195,1196,1197};
    CatmullRom(449) = {1197,1192,1193,1194,1172};
    CatmullRom(450) = {1212,1198,1199,1209,1174};
    CatmullRom(451) = {974,1011,1084,1145,1197};
    CatmullRom(452) = {945,1028,1099,1162,1212};
    Delete {
    Line{396};
    }
    Delete {
    Line{397};
    }
    Delete {
    Line{353};
    }
    Delete {
    Line{354};
    }
    Delete {
    Line{357};
    }
    Delete {
    Line{358};
    }
    Delete {
    Line{398,399,370,369,376,375,377,378};
    }
    Delete {
    Line{360};
    }
    Delete {
    Line{362};
    }
    CatmullRom(453) = {1174,1228,1320,1474};
    CatmullRom(454) = {1173,1227,1321,1475};
    CatmullRom(455) = {982,1042,1116,1174};
    CatmullRom(456) = {981,1041,1115,1173};
    CatmullRom(457) = {1032,1104,1166,1228};
    CatmullRom(458) = {1031,1103,1165,1227};
    CatmullRom(459) = {1169,1223,1239,1320};
    CatmullRom(460) = {1170,1224,1240,1321};
    CatmullRom(461) = {1386,1400,1412,1474};
    CatmullRom(462) = {1387,1401,1413,1475};
    CatmullRom(463) = {1228,1236,1313,1399,1469,1566,1595};
    CatmullRom(464) = {1227,1235,1312,1398,1468,1493,1567,1595};
    CatmullRom(465) = {1320,1382,1394,1472,1488,1504,1582,1596};
    CatmullRom(466) = {1321,1383,1395,1473,1489,1505,1583,1596};
    CatmullRom(467) = {1474,1480,1500,1558,1570,1586,1588,1597};
    CatmullRom(468) = {1475,1481,1501,1559,1571,1587,1597};
    CatmullRom(469) = {1174,1232,1295,1391,1409,1484,1562,1594};
    CatmullRom(470) = {1173,1231,1294,1390,1408,1485,1563,1594};
    CatmullRom(471) = {1172,1226,1318,1476};
    CatmullRom(472) = {1171,1225,1319,1477};
    CatmullRom(473) = {1384,1402,1410,1476};
    CatmullRom(474) = {1385,1403,1411,1477};
    CatmullRom(475) = {1476,1478,1498,1556,1568,1584,1589,1591};
    CatmullRom(476) = {1477,1479,1499,1557,1569,1585,1591};
    CatmullRom(477) = {1167,1221,1237,1318};
    CatmullRom(478) = {1168,1222,1238,1319};
    CatmullRom(479) = {1318,1380,1392,1470,1486,1502,1580,1590};
    CatmullRom(480) = {1319,1381,1393,1471,1487,1503,1581,1590};
    CatmullRom(481) = {1030,1102,1164,1226};
    CatmullRom(482) = {1029,1101,1163,1225};
    CatmullRom(483) = {1226,1234,1311,1397,1467,1490,1564,1593};
    CatmullRom(484) = {1225,1233,1310,1396,1466,1491,1565,1593};
    CatmullRom(485) = {980,1040,1114,1172};
    CatmullRom(486) = {979,1039,1113,1171};
    CatmullRom(487) = {1172,1230,1293,1389,1407,1482,1560,1592};
    CatmullRom(488) = {1171,1229,1292,1388,1406,1483,1561,1592};
    CatmullRom(489) = {965,1002,1075,1136,1190};
    CatmullRom(490) = {966,1001,1072,1134,1187};
    CatmullRom(491) = {1190,1187};
    CatmullRom(492) = {933,1016,1087,1150,1201};
    CatmullRom(493) = {1201,1200};
    CatmullRom(494) = {1200,1148,1086,1014,932};
    CatmullRom(495) = {1211,1289,1376,1461,1553,1640,1683};
    CatmullRom(496) = {1209,1286,1375,1452,1543,1633};
    CatmullRom(497) = {1194,1270,1357,1431,1523,1617};
    Delete {
    Line{495};
    }
    CatmullRom(498) = {1622,1532,1440,1358,1269,1194};
    CatmullRom(499) = {1622,1617};
    CatmullRom(500) = {1617,1592};
    CatmullRom(501) = {1639,1551,1458,1374,1287,1209};
    CatmullRom(502) = {1639,1633};
    CatmullRom(503) = {1594,1633};

    CatmullRom(506) = {308,305,306,301};
    CatmullRom(507) = {335,334,333,330};
    CatmullRom(508) = {219,248,301};
    CatmullRom(509) = {301,368,421,474};
    CatmullRom(510) = {233,263,330};
    CatmullRom(511) = {330,386,437,491};
    CatmullRom(512) = {1212,1290,1379,1462,1554,1643};
    CatmullRom(513) = {1643,1628,1638,1639};
    CatmullRom(514) = {1197,1273,1360,1445,1537,1626};
    CatmullRom(515) = {1626,1624,1623,1622};
    CatmullRom(516) = {1211,1289,1376,1461,1553,1640};
    CatmullRom(517) = {1640,1625,1627,1626};
    CatmullRom(518) = {1640,1641,1642,1643};
    CatmullRom(519) = {933,934,936,937,939,981};
    CatmullRom(520) = {752,753,756,757,759,744};
    CatmullRom(521) = {277,350,402,452};
    CatmullRom(522) = {452,514,557};
    CatmullRom(523) = {557,610,665};
    CatmullRom(524) = {665,708,776};
    CatmullRom(525) = {776,829,881,952};
    CatmullRom(526) = {952,989,1061,1124,1178};
    CatmullRom(527) = {932,930,957,959,958,952};
    CatmullRom(528) = {952,956,955,954};
    CatmullRom(529) = {751,777,778,780,779,776};
    CatmullRom(530) = {776,775,774,773};
    CatmullRom(531) = {684,669,668,666,667,665};
    CatmullRom(532) = {665,664,662,663};
    CatmullRom(533) = {583,584,561,562,560,556,557};
    CatmullRom(534) = {557,558,559};
    CatmullRom(535) = {480,479,458,457,456,455,452};
    CatmullRom(536) = {452,454,453};
    CatmullRom(537) = {270,343,396,446};
    CatmullRom(538) = {446,509,552};
    CatmullRom(539) = {552,604,656};
    CatmullRom(540) = {656,701,767};
    CatmullRom(541) = {767,838,876,947};
    CatmullRom(542) = {947,984,1055,1119,1214};
    CatmullRom(543) = {947,948,953,954};
    CatmullRom(544) = {773,766,768,767};
    CatmullRom(545) = {656,657,655,663};
    CatmullRom(546) = {559,551,552};
    CatmullRom(547) = {453,447,446};
    CatmullRom(548) = {270,271,278};
    CatmullRom(549) = {278,279,277};
    CatmullRom(550) = {947,951,950,949,960,966};
    CatmullRom(551) = {767,772,771,770,769,783};
    CatmullRom(552) = {656,658,659,661,660,671};
    CatmullRom(553) = {552,550,554,555,553,573,574};
    CatmullRom(554) = {446,451,450,449,448,467,466};
    CatmullRom(555) = {270,272,276,275,274,273,281};
    CatmullRom(556) = {277,282,283,284,285,280,281};
    CatmullRom(557) = {1187,1176,1177,1215,1216,1214};
    CatmullRom(558) = {1200,1186,1185,1184,1183,1178};
    CatmullRom(559) = {954,990,1063,1125,1182};
    CatmullRom(560) = {1214,1175,1181,1182};
    CatmullRom(561) = {1182,1179,1180,1178};
    CatmullRom(562) = {1190,1262,1349,1438,1530,1621};
    CatmullRom(563) = {1621,1620};
    CatmullRom(564) = {1620,1531,1439,1347,1259,1187};
    CatmullRom(565) = {1201,1276,1366,1448,1540,1630};
    CatmullRom(566) = {1630,1629};
    CatmullRom(567) = {1629,1539,1447,1364,1275,1200};
    CatmullRom(568) = {1182,1250,1337,1424,1514,1606};
    CatmullRom(569) = {1201,1202,1204,1205,1206,1207,1173};
    CatmullRom(570) = {1190,1189,1188,1191,1220,1219,1171};
    Delete {
    Point{1141,1143,1142,1140,1139,1138,1135,1137,1060,1059,1122,1123,1121,1120,1117,1118,1126,1097,1127,1160,1161,1100,1147,1157,1154,1158,1155,1153,1152,1149,1131,1130,1128,1129,1144,1146};
    }
    Delete {
    Point{833,853,854,847,849,848,852,850,897,898,896,895,894,888,899,900,903,901,812,834,841,840,839,846,892,877,879,878,918,919,880,875,825,827,836,837,904,813,830,828,883,882,814,811,831,832,887,886,885,908,916,823,821,822,915,912,913,911,910,905,816,818,819,820};
    }
    Delete {
    Point{722,723,724,716,717,718,719,721,705,704,703,702,700,699,706,709,726,727,740,741,731,732,738,736,737,735,734,729,712,713,710,711};
    }
    Delete {
    Point{637,635,616,615,613,632,636,639,642,640,641,645,643,609,612,603,605,630,631,614,626,627,628,621,622,623,624,625,617,607,608,606,647,648,649,651,652,653};
    }
    Delete {
    Point{548,535,534,531,533,528,501,529,522,523,524,525,499,527,511,512,513,498,510,516,506,507,508,504,543,540,537,541,520,544,542,519,518,517,539,546,545,503};
    }
    Delete {
    Point{427,426,424,423,395,398,416,417,399,415,400,397,414,420,418,412,419,401,403,407,408,410,433,436,434,409,432,435,438,429,406,430,441,440,442,443,445,425};
    }
    Delete {
    Point{267,246,268,245,244,264,256,266,257,262,258,261,259,253,250,252,254,247,249,243};
    }
    Delete {
    Point{355,356,351,357,358,384,382,381,379,344,353,376,383,378,346,365,345,388,347,364,349,348,363,387,362,389,391,361,369,373,375,374,371,370,392,367,366,385};
    }
    Delete {
    Point{202,201,186,205,188,203,191,207,206,190,210,211,192,193,199,197};
    }
    Delete {
    Point{136,138,130,131,135,134,148,147,149,151,155,143,144,154,157};
    }
    Delete {
    Point{106,100,98,101,107,108,112,113};
    }
    Delete {
    Point{70,76,71,79,75,68,67,81};
    }
    CatmullRom(571) = {789,788,787,786,784,743};
    CatmullRom(572) = {965,964,967,970,968,979};






    // we begin the tail
    cl = 0.0600666666666;
    Point(1702) = {-3.8861, -0.453049, -1.80371, cl};
    Point(1703) = {-3.8861, -0.491384, -1.79343, cl};
    Point(1704) = {-3.8861, 0.491394, -1.79343, cl};
    Point(1705) = {-3.8861, 0.453061, -1.80371, cl};
    Point(1706) = {-3.9578, -0.575581, -2.21449, cl};
    Point(1707) = {-3.9578, -0.590639, -2.21043, cl};
    Point(1708) = {-3.9578, 0.590644, -2.21043, cl};
    Point(1709) = {-3.9578, 0.575588, -2.21449, cl};
    Point(1710) = {-4.0822, -0.454242, -1.80338, cl};
    Point(1711) = {-4.0822, -0.490189, -1.79377, cl};
    Point(1712) = {-4.0822, 0.490199, -1.79377, cl};
    Point(1713) = {-4.0822, 0.454255, -1.80338, cl};
    Point(1714) = {-4.0928, -0.100981, -0.379127, cl};
    Point(1715) = {-4.0928, 0.100988, -0.379127, cl};
    Point(1716) = {-4.10031, -0.575056, -2.21463, cl};
    Point(1717) = {-4.10031, -0.591167, -2.21029, cl};
    Point(1718) = {-4.10031, 0.591172, -2.21029, cl};
    Point(1719) = {-4.10031, 0.575061, -2.21463, cl};
    Point(1720) = {-4.1153, -0.297217, -1.92357, cl};
    Point(1721) = {-4.1153, -0.399996, -1.87913, cl};
    Point(1722) = {-4.1153, -0.572222, -1.70413, cl};
    Point(1723) = {-4.1153, -0.49444, -1.80135, cl};
    Point(1724) = {-4.1153, -0.655556, -1.49302, cl};
    Point(1725) = {-4.1153, -0.622219, -1.59579, cl};
    Point(1726) = {-4.1153, 0.400005, -1.87913, cl};
    Point(1727) = {-4.1153, 0.158338, -1.9569, cl};
    Point(1728) = {-4.1153, 0.297227, -1.92357, cl};
    Point(1729) = {-4.1153, -0.158328, -1.9569, cl};
    Point(1730) = {-4.1153, 0.494449, -1.80135, cl};
    Point(1731) = {-4.1153, 0.572227, -1.70413, cl};
    Point(1732) = {-4.1153, 0.622227, -1.59579, cl};
    Point(1733) = {-4.1153, 0.655561, -1.49302, cl};
    Point(1734) = {-4.1153, -0.894444, -1.25413, cl};
    Point(1735) = {-4.1153, -0.802778, -1.26802, cl};
    Point(1736) = {-4.1153, -1.15, -1.09024, cl};
    Point(1737) = {-4.1153, -1.14722, -1.20968, cl};
    Point(1738) = {-4.1153, -1.05278, -1.22357, cl};
    Point(1739) = {-4.1153, -0.988889, -1.23746, cl};
    Point(1740) = {-4.1153, -0.708333, -1.28468, cl};
    Point(1741) = {-4.1153, -0.686111, -1.37635, cl};
    Point(1742) = {-4.1153, -0.697222, -0.71246, cl};
    Point(1743) = {-4.1153, -0.919444, -0.809682, cl};
    Point(1744) = {-4.1153, -1.15833, -0.879127, cl};
    Point(1745) = {-4.1153, -0.205551, -0.36246, cl};
    Point(1746) = {-4.1153, -0.422217, -0.465238, cl};
    Point(1747) = {-4.1153, -0.566667, -0.606904, cl};
    Point(1748) = {-4.1153, 0.686116, -1.37635, cl};
    Point(1749) = {-4.1153, 0.708338, -1.28468, cl};
    Point(1750) = {-4.1153, 0.802783, -1.26802, cl};
    Point(1751) = {-4.1153, 0.894449, -1.25413, cl};
    Point(1752) = {-4.1153, 0.988894, -1.23746, cl};
    Point(1753) = {-4.1153, 1.05278, -1.22357, cl};
    Point(1754) = {-4.1153, 1.15, -1.09024, cl};
    Point(1755) = {-4.1153, 1.15833, -0.879127, cl};
    Point(1756) = {-4.1153, 0.919449, -0.809682, cl};
    Point(1757) = {-4.1153, 0.205561, -0.36246, cl};
    Point(1758) = {-4.1153, -0.00277331, -0.329127, cl};
    Point(1759) = {-4.1153, 4.60818e-06, -1.96524, cl};
    Point(1760) = {-4.1153, 0.697227, -0.71246, cl};
    Point(1761) = {-4.1153, 1.14722, -1.20968, cl};
    Point(1762) = {-4.1153, 0.566672, -0.606904, cl};
    Point(1763) = {-4.1153, 0.422227, -0.465238, cl};
    Point(1764) = {-4.243, -0.575582, -2.21449, cl};
    Point(1765) = {-4.243, -0.590636, -2.21043, cl};
    Point(1766) = {-4.243, 0.590644, -2.21043, cl};
    Point(1767) = {-4.243, 0.575587, -2.21449, cl};
    Point(1768) = {-4.2566, 4.80709e-06, 0.158373, cl};
    Point(1769) = {-4.2786, -0.457826, -1.80243, cl};
    Point(1770) = {-4.2786, -0.486603, -1.79471, cl};
    Point(1771) = {-4.2786, 0.486616, -1.79471, cl};
    Point(1772) = {-4.2786, 0.457838, -1.80243, cl};
    Point(1773) = {-4.3855, -0.577083, -2.21407, cl};
    Point(1774) = {-4.3855, -0.589139, -2.21085, cl};
    Point(1775) = {-4.3855, 0.589144, -2.21085, cl};
    Point(1776) = {-4.3855, 0.577088, -2.21407, cl};
    Point(1777) = {-4.4625, 4.64246e-06, -1.94579, cl};
    Point(1778) = {-4.4625, 0.152783, -1.93746, cl};
    Point(1779) = {-4.4625, 0.313894, -1.89857, cl};
    Point(1780) = {-4.4625, 0.558338, -1.70135, cl};
    Point(1781) = {-4.4625, 0.422227, -1.84302, cl};
    Point(1782) = {-4.4625, 0.488894, -1.78468, cl};
    Point(1783) = {-4.4625, 0.616672, -1.58468, cl};
    Point(1784) = {-4.4625, 0.652783, -1.49302, cl};
    Point(1785) = {-4.4625, -0.663887, -0.709682, cl};
    Point(1786) = {-4.4625, -0.91389, -0.820793, cl};
    Point(1787) = {-4.4625, -0.249995, -0.379127, cl};
    Point(1788) = {-4.4625, -0.427772, -0.473571, cl};
    Point(1789) = {-4.4625, -0.55, -0.598571, cl};
    Point(1790) = {-4.4625, 0.663894, -0.709682, cl};
    Point(1791) = {-4.4625, 0.686116, -1.37079, cl};
    Point(1792) = {-4.4625, 1.15557, -1.09024, cl};
    Point(1793) = {-4.4625, 0.977783, -1.23468, cl};
    Point(1794) = {-4.4625, 1.05833, -1.22357, cl};
    Point(1795) = {-4.4625, 1.14168, -1.21802, cl};
    Point(1796) = {-4.4625, 0.913893, -0.820793, cl};
    Point(1797) = {-4.4625, 1.17501, -0.895793, cl};
    Point(1798) = {-4.4625, 0.250005, -0.379127, cl};
    Point(1799) = {-4.4625, 0.550005, -0.598571, cl};
    Point(1800) = {-4.4625, 0.427783, -0.473571, cl};
    Point(1801) = {-4.4625, -0.313881, -1.89857, cl};
    Point(1802) = {-4.4625, -0.422218, -1.84302, cl};
    Point(1803) = {-4.4625, -0.55833, -1.70135, cl};
    Point(1804) = {-4.4625, -0.488881, -1.78468, cl};
    Point(1805) = {-4.4625, -0.652776, -1.49302, cl};
    Point(1806) = {-4.4625, -0.616666, -1.58468, cl};
    Point(1807) = {-4.4625, -0.152773, -1.93746, cl};
    Point(1808) = {-4.4625, -0.00555134, -0.331904, cl};
    Point(1809) = {-4.4625, -0.886111, -1.24857, cl};
    Point(1810) = {-4.4625, -0.802779, -1.2569, cl};
    Point(1811) = {-4.4625, -0.977778, -1.23468, cl};
    Point(1812) = {-4.4625, -1.15556, -1.09024, cl};
    Point(1813) = {-4.4625, -1.14167, -1.21802, cl};
    Point(1814) = {-4.4625, -1.05833, -1.22357, cl};
    Point(1815) = {-4.4625, -0.708333, -1.27079, cl};
    Point(1816) = {-4.4625, -0.686111, -1.37079, cl};
    Point(1817) = {-4.4625, -1.175, -0.895793, cl};
    Point(1818) = {-4.4625, 0.708338, -1.27079, cl};
    Point(1819) = {-4.4625, 0.802783, -1.2569, cl};
    Point(1820) = {-4.4625, 0.886116, -1.24857, cl};
    Point(1821) = {-4.4625, 0.00556058, -0.331904, cl};
    Point(1822) = {-4.47471, -0.463825, -1.80082, cl};
    Point(1823) = {-4.47471, -0.480603, -1.79632, cl};
    Point(1824) = {-4.47471, 0.480616, -1.79632, cl};
    Point(1825) = {-4.47471, 0.463838, -1.80082, cl};
    Point(1826) = {-4.5283, -0.577025, -2.2141, cl};
    Point(1827) = {-4.5283, -0.579579, -2.2134, cl};
    Point(1828) = {-4.5283, 0.589199, -2.21082, cl};
    Point(1829) = {-4.5283, 0.586644, -2.21152, cl};
    Point(1830) = {-4.593, -0.0255421, 0.158373, cl};
    Point(1831) = {-4.593, 0.0255518, 0.158373, cl};
    Point(1832) = {-4.6639, -0.127026, -0.379127, cl};
    Point(1833) = {-4.6639, 0.127036, -0.379127, cl};
    Point(1834) = {-4.6708, -0.472217, -1.79857, cl};
    Point(1835) = {-4.6708, 0.472227, -1.79857, cl};
    Point(1836) = {-4.6708, 0.583116, -2.21246, cl};
    Point(1837) = {-4.6708, -0.58311, -2.21246, cl};
    Point(1838) = {-4.8097, -0.27777, -1.88468, cl};
    Point(1839) = {-4.8097, -0.377773, -1.84024, cl};
    Point(1840) = {-4.8097, -0.486105, -1.75968, cl};
    Point(1841) = {-4.8097, -0.547222, -1.68468, cl};
    Point(1842) = {-4.8097, -0.608332, -1.57913, cl};
    Point(1843) = {-4.8097, -0.65, -1.47913, cl};
    Point(1844) = {-4.8097, 0.00278274, -1.92357, cl};
    Point(1845) = {-4.8097, -0.136106, -1.91524, cl};
    Point(1846) = {-4.8097, 0.277783, -1.88468, cl};
    Point(1847) = {-4.8097, 0.136116, -1.91524, cl};
    Point(1848) = {-4.8097, 0.486116, -1.75968, cl};
    Point(1849) = {-4.8097, 0.377783, -1.84024, cl};
    Point(1850) = {-4.8097, -0.00555131, -0.334682, cl};
    Point(1851) = {-4.8097, -0.794444, -1.24857, cl};
    Point(1852) = {-4.8097, -0.880556, -1.24302, cl};
    Point(1853) = {-4.8097, -0.958333, -1.2319, cl};
    Point(1854) = {-4.8097, -1.15278, -1.09024, cl};
    Point(1855) = {-4.8097, -1.02778, -1.22913, cl};
    Point(1856) = {-4.8097, -1.13333, -1.22079, cl};
    Point(1857) = {-4.8097, -0.680556, -1.36246, cl};
    Point(1858) = {-4.8097, -0.702778, -1.25968, cl};
    Point(1859) = {-4.8097, -0.669444, -0.734682, cl};
    Point(1860) = {-4.8097, -1.18056, -0.909682, cl};
    Point(1861) = {-4.8097, -0.908333, -0.83746, cl};
    Point(1862) = {-4.8097, -0.541667, -0.595793, cl};
    Point(1863) = {-4.8097, 0.669449, -0.734682, cl};
    Point(1864) = {-4.8097, 0.702783, -1.25968, cl};
    Point(1865) = {-4.8097, 0.880561, -1.24302, cl};
    Point(1866) = {-4.8097, 0.794449, -1.24857, cl};
    Point(1867) = {-4.8097, 1.15279, -1.09024, cl};
    Point(1868) = {-4.8097, 0.958338, -1.2319, cl};
    Point(1869) = {-4.8097, 1.13334, -1.22079, cl};
    Point(1870) = {-4.8097, 1.02779, -1.22913, cl};
    Point(1871) = {-4.8097, 0.908338, -0.83746, cl};
    Point(1872) = {-4.8097, 1.18056, -0.909682, cl};
    Point(1873) = {-4.8097, 0.00556061, -0.334682, cl};
    Point(1874) = {-4.8097, 0.247227, -0.381904, cl};
    Point(1875) = {-4.8097, 0.411116, -0.470793, cl};
    Point(1876) = {-4.8097, 0.541672, -0.595793, cl};
    Point(1877) = {-4.8097, 0.547227, -1.68468, cl};
    Point(1878) = {-4.8097, 0.650005, -1.47913, cl};
    Point(1879) = {-4.8097, 0.608338, -1.57913, cl};
    Point(1880) = {-4.8097, -0.247217, -0.381904, cl};
    Point(1881) = {-4.8097, -0.411106, -0.470793, cl};
    Point(1882) = {-4.8097, 0.68056, -1.36246, cl};
    Point(1883) = {-4.9297, -0.0437673, 0.158373, cl};
    Point(1884) = {-4.9297, 0.0437767, 0.158373, cl};
    Point(1885) = {-5.085, 1.15278, -1.10252, cl};
    Point(1886) = {-5.085, -1.15278, -1.10252, cl};
    Point(1887) = {-5.1569, -0.502778, -1.71246, cl};
    Point(1888) = {-5.1569, -0.388885, -1.80691, cl};
    Point(1889) = {-5.1569, -0.291661, -1.84857, cl};
    Point(1890) = {-5.1569, -0.644444, -1.46524, cl};
    Point(1891) = {-5.1569, -0.605556, -1.56246, cl};
    Point(1892) = {-5.1569, -0.563888, -1.63746, cl};
    Point(1893) = {-5.1569, 0.161116, -1.88468, cl};
    Point(1894) = {-5.1569, -0.161106, -1.88468, cl};
    Point(1895) = {-5.1569, 0.00278277, -1.89579, cl};
    Point(1896) = {-5.1569, 0.291672, -1.84857, cl};
    Point(1897) = {-5.1569, 0.388894, -1.80691, cl};
    Point(1898) = {-5.1569, 0.502783, -1.71246, cl};
    Point(1899) = {-5.1569, 0.563894, -1.63746, cl};
    Point(1900) = {-5.1569, -1.02778, -1.22913, cl};
    Point(1901) = {-5.1569, -1.15556, -1.08746, cl};
    Point(1902) = {-5.1569, -1.13056, -1.22357, cl};
    Point(1903) = {-5.1569, -0.941667, -1.2319, cl};
    Point(1904) = {-5.1569, -0.866667, -1.23468, cl};
    Point(1905) = {-5.1569, -0.783333, -1.23746, cl};
    Point(1906) = {-5.1569, -0.697222, -1.24579, cl};
    Point(1907) = {-5.1569, -0.677778, -1.36524, cl};
    Point(1908) = {-5.1569, -0.95, -0.870793, cl};
    Point(1909) = {-5.1569, -1.18056, -0.923571, cl};
    Point(1910) = {-5.1569, -0.588888, -0.665238, cl};
    Point(1911) = {-5.1569, -0.691667, -0.784682, cl};
    Point(1912) = {-5.1569, -0.474995, -0.531904, cl};
    Point(1913) = {-5.1569, 0.644449, -1.46524, cl};
    Point(1914) = {-5.1569, 0.605561, -1.56246, cl};
    Point(1915) = {-5.1569, -0.00555128, -0.345793, cl};
    Point(1916) = {-5.1569, -0.272214, -0.398571, cl};
    Point(1917) = {-5.1569, 0.677783, -1.36524, cl};
    Point(1918) = {-5.1569, 0.697227, -1.24579, cl};
    Point(1919) = {-5.1569, 0.783338, -1.23746, cl};
    Point(1920) = {-5.1569, 0.866672, -1.23468, cl};
    Point(1921) = {-5.1569, 0.691671, -0.784682, cl};
    Point(1922) = {-5.1569, 0.588894, -0.665238, cl};
    Point(1923) = {-5.1569, 1.15557, -1.08746, cl};
    Point(1924) = {-5.1569, 0.941671, -1.2319, cl};
    Point(1925) = {-5.1569, 1.02779, -1.22913, cl};
    Point(1926) = {-5.1569, 1.13057, -1.22357, cl};
    Point(1927) = {-5.1569, 1.18056, -0.923571, cl};
    Point(1928) = {-5.1569, 0.950005, -0.870793, cl};
    Point(1929) = {-5.1569, 0.475005, -0.531904, cl};
    Point(1930) = {-5.1569, 0.272227, -0.398571, cl};
    Point(1931) = {-5.1569, 0.00556064, -0.345793, cl};
    Point(1932) = {-5.235, -0.137078, -0.379127, cl};
    Point(1933) = {-5.235, 0.137088, -0.379127, cl};
    Point(1934) = {-5.2661, -0.0547233, 0.158373, cl};
    Point(1935) = {-5.2661, 0.0547328, 0.158373, cl};
    Point(1936) = {-5.3633, 1.15278, -1.07675, cl};
    Point(1937) = {-5.3633, 1.15278, -1.12829, cl};
    Point(1938) = {-5.3633, -1.15278, -1.12829, cl};
    Point(1939) = {-5.3633, -1.15278, -1.07675, cl};
    Point(1940) = {-5.5041, -0.488882, -1.69579, cl};
    Point(1941) = {-5.5041, -0.305552, -1.81246, cl};
    Point(1942) = {-5.5041, -0.40555, -1.76246, cl};
    Point(1943) = {-5.5041, -0.638887, -1.4569, cl};
    Point(1944) = {-5.5041, -0.55, -1.62635, cl};
    Point(1945) = {-5.5041, -0.597222, -1.55135, cl};
    Point(1946) = {-5.5041, -0.169439, -1.85135, cl};
    Point(1947) = {-5.5041, 0.00278281, -1.86802, cl};
    Point(1948) = {-5.5041, -1.00278, -1.2319, cl};
    Point(1949) = {-5.5041, -1.13056, -1.2319, cl};
    Point(1950) = {-5.5041, -1.15556, -1.09024, cl};
    Point(1951) = {-5.5041, -0.916667, -1.22913, cl};
    Point(1952) = {-5.5041, -0.775, -1.22913, cl};
    Point(1953) = {-5.5041, -0.844444, -1.22913, cl};
    Point(1954) = {-5.5041, -0.697221, -1.2319, cl};
    Point(1955) = {-5.5041, -0.669444, -1.37079, cl};
    Point(1956) = {-5.5041, -1.175, -0.943016, cl};
    Point(1957) = {-5.5041, -0.925, -0.890238, cl};
    Point(1958) = {-5.5041, 0.169449, -1.85135, cl};
    Point(1959) = {-5.5041, 0.305561, -1.81246, cl};
    Point(1960) = {-5.5041, 0.488894, -1.69579, cl};
    Point(1961) = {-5.5041, 0.405561, -1.76246, cl};
    Point(1962) = {-5.5041, 0.638894, -1.4569, cl};
    Point(1963) = {-5.5041, 0.597227, -1.55135, cl};
    Point(1964) = {-5.5041, 0.550005, -1.62635, cl};
    Point(1965) = {-5.5041, -0.00555124, -0.354127, cl};
    Point(1966) = {-5.5041, -0.677778, -0.818016, cl};
    Point(1967) = {-5.5041, -0.577778, -0.670793, cl};
    Point(1968) = {-5.5041, -0.238884, -0.398571, cl};
    Point(1969) = {-5.5041, -0.466662, -0.540238, cl};
    Point(1970) = {-5.5041, 0.669449, -1.37079, cl};
    Point(1971) = {-5.5041, 0.697227, -1.2319, cl};
    Point(1972) = {-5.5041, 0.844449, -1.22913, cl};
    Point(1973) = {-5.5041, 0.775005, -1.22913, cl};
    Point(1974) = {-5.5041, 0.577783, -0.670793, cl};
    Point(1975) = {-5.5041, 0.677783, -0.818016, cl};
    Point(1976) = {-5.5041, 1.15557, -1.09024, cl};
    Point(1977) = {-5.5041, 0.916672, -1.22913, cl};
    Point(1978) = {-5.5041, 1.13057, -1.2319, cl};
    Point(1979) = {-5.5041, 1.00279, -1.2319, cl};
    Point(1980) = {-5.5041, 0.925004, -0.890238, cl};
    Point(1981) = {-5.5041, 1.175, -0.943017, cl};
    Point(1982) = {-5.5041, 0.466672, -0.540238, cl};
    Point(1983) = {-5.5041, 0.00556068, -0.354127, cl};
    Point(1984) = {-5.5041, 0.238894, -0.398571, cl};
    Point(1985) = {-5.6025, -0.0583563, 0.158373, cl};
    Point(1986) = {-5.6025, 0.0583658, 0.158373, cl};
    Point(1987) = {-5.6025, -0.0748312, 0.158373, cl};
    Point(1988) = {-5.6025, 0.0748407, 0.158373, cl};
    Point(1989) = {-5.6416, 1.15278, -1.05833, cl};
    Point(1990) = {-5.6416, 1.15278, -1.14671, cl};
    Point(1991) = {-5.6416, -1.15278, -1.14671, cl};
    Point(1992) = {-5.6416, -1.15278, -1.05833, cl};
    Point(1993) = {-5.8064, -0.137078, -0.379127, cl};
    Point(1994) = {-5.8064, 0.137088, -0.379127, cl};
    Point(1995) = {-5.85141, 0.633338, -1.43746, cl};
    Point(1996) = {-5.85141, -0.633333, -1.43746, cl};
    Point(1997) = {-5.85141, -0.619444, -1.47079, cl};
    Point(1998) = {-5.85141, -0.525, -1.61802, cl};
    Point(1999) = {-5.85141, -0.369438, -1.74579, cl};
    Point(2000) = {-5.85141, -0.219439, -1.8069, cl};
    Point(2001) = {-5.85141, -0.486107, -1.65968, cl};
    Point(2002) = {-5.85141, -0.283325, -1.78468, cl};
    Point(2003) = {-5.85141, -0.394436, -1.7319, cl};
    Point(2004) = {-5.85141, -0.558333, -1.5819, cl};
    Point(2005) = {-5.85141, -0.602775, -1.50968, cl};
    Point(2006) = {-5.85141, -0.155551, -1.81802, cl};
    Point(2007) = {-5.85141, -0.00277328, -1.83468, cl};
    Point(2008) = {-5.85141, 0.219449, -1.80691, cl};
    Point(2009) = {-5.85141, 0.369449, -1.74579, cl};
    Point(2010) = {-5.85141, 0.525005, -1.61802, cl};
    Point(2011) = {-5.85141, 0.619449, -1.47079, cl};
    Point(2012) = {-5.85141, 0.283338, -1.78468, cl};
    Point(2013) = {-5.85141, 0.155561, -1.81802, cl};
    Point(2014) = {-5.85141, 0.486116, -1.65968, cl};
    Point(2015) = {-5.85141, 0.394449, -1.7319, cl};
    Point(2016) = {-5.85141, 0.602783, -1.50968, cl};
    Point(2017) = {-5.85141, 0.558338, -1.5819, cl};
    Point(2018) = {-5.85141, -1.05556, -1.2319, cl};
    Point(2019) = {-5.85141, -0.983333, -1.26524, cl};
    Point(2020) = {-5.85141, -0.841667, -1.20968, cl};
    Point(2021) = {-5.85141, -0.691667, -1.04857, cl};
    Point(2022) = {-5.85141, -0.672222, -1.31246, cl};
    Point(2023) = {-5.85141, -0.966667, -1.26524, cl};
    Point(2024) = {-5.85141, -1.15278, -1.09302, cl};
    Point(2025) = {-5.85141, -1.03611, -1.22913, cl};
    Point(2026) = {-5.85141, -1.12778, -1.23746, cl};
    Point(2027) = {-5.85141, -0.791667, -1.19857, cl};
    Point(2028) = {-5.85141, -0.880556, -1.21802, cl};
    Point(2029) = {-5.85141, -0.669443, -1.3319, cl};
    Point(2030) = {-5.85141, -0.694444, -1.19579, cl};
    Point(2031) = {-5.85141, -0.569444, -0.676349, cl};
    Point(2032) = {-5.85141, -0.694444, -0.879127, cl};
    Point(2033) = {-5.85141, -0.936111, -0.918016, cl};
    Point(2034) = {-5.85141, -1.17222, -0.970793, cl};
    Point(2035) = {-5.85141, -0.925, -0.915238, cl};
    Point(2036) = {-5.85141, -0.675, -0.868016, cl};
    Point(2037) = {-5.85141, -0.563887, -0.66246, cl};
    Point(2038) = {-5.85141, -0.416662, -0.509682, cl};
    Point(2039) = {-5.85141, -0.258328, -0.420793, cl};
    Point(2040) = {-5.85141, 0.672227, -1.31246, cl};
    Point(2041) = {-5.85141, 0.841671, -1.20968, cl};
    Point(2042) = {-5.85141, 0.691672, -1.04857, cl};
    Point(2043) = {-5.85141, 0.669449, -1.3319, cl};
    Point(2044) = {-5.85141, 0.694449, -1.20135, cl};
    Point(2045) = {-5.85141, 0.880561, -1.21802, cl};
    Point(2046) = {-5.85141, 0.791672, -1.19857, cl};
    Point(2047) = {-5.85141, 0.697227, -0.876349, cl};
    Point(2048) = {-5.85141, 0.569449, -0.676349, cl};
    Point(2049) = {-5.85141, 0.563894, -0.66246, cl};
    Point(2050) = {-5.85141, 0.675004, -0.868016, cl};
    Point(2051) = {-5.85141, 0.983338, -1.26524, cl};
    Point(2052) = {-5.85141, 1.05557, -1.2319, cl};
    Point(2053) = {-5.85141, 0.966672, -1.26524, cl};
    Point(2054) = {-5.85141, 1.03611, -1.22913, cl};
    Point(2055) = {-5.85141, 1.15278, -1.09302, cl};
    Point(2056) = {-5.85141, 1.12779, -1.23746, cl};
    Point(2057) = {-5.85141, 0.936115, -0.918016, cl};
    Point(2058) = {-5.85141, 0.925005, -0.915238, cl};
    Point(2059) = {-5.85141, 1.17223, -0.970793, cl};
    Point(2060) = {-5.85141, 0.416672, -0.509682, cl};
    Point(2061) = {-5.85141, -0.00277315, -0.365238, cl};
    Point(2062) = {-5.85141, 0.258338, -0.420793, cl};
    Point(2063) = {-5.9203, 1.15278, -1.04728, cl};
    Point(2064) = {-5.9203, 1.15278, -1.15774, cl};
    Point(2065) = {-5.9203, -1.15278, -1.15774, cl};
    Point(2066) = {-5.9203, -1.15278, -1.04728, cl};
    Point(2067) = {-5.93911, -0.0547233, 0.158373, cl};
    Point(2068) = {-5.93911, 0.0547329, 0.158373, cl};
    Point(2069) = {-5.93911, -0.0807533, 0.158373, cl};
    Point(2070) = {-5.93911, 0.0807629, 0.158373, cl};
    Point(2071) = {-6.19861, 1.15278, -1.04361, cl};
    Point(2072) = {-6.19861, 1.15279, -1.16143, cl};
    Point(2073) = {-6.19861, -1.15278, -1.16143, cl};
    Point(2074) = {-6.19861, -1.15278, -1.04361, cl};
    Point(2075) = {-6.19861, -0.199995, -1.76246, cl};
    Point(2076) = {-6.19861, -0.397214, -1.67635, cl};
    Point(2077) = {-6.19861, -0.597221, -1.4569, cl};
    Point(2078) = {-6.19861, -0.513887, -1.57635, cl};
    Point(2079) = {-6.19861, 0.397227, -1.67635, cl};
    Point(2080) = {-6.19861, 4.81921e-06, -1.78746, cl};
    Point(2081) = {-6.19861, 0.200005, -1.76246, cl};
    Point(2082) = {-6.19861, 0.513894, -1.57635, cl};
    Point(2083) = {-6.19861, 0.597227, -1.4569, cl};
    Point(2084) = {-6.19861, -0.752778, -1.03746, cl};
    Point(2085) = {-6.19861, -0.680556, -1.04302, cl};
    Point(2086) = {-6.19861, -1.05, -1.24579, cl};
    Point(2087) = {-6.19861, -0.969445, -1.2819, cl};
    Point(2088) = {-6.19861, -1.15278, -1.08746, cl};
    Point(2089) = {-6.19861, -1.125, -1.23468, cl};
    Point(2090) = {-6.19861, -0.888889, -1.20413, cl};
    Point(2091) = {-6.19861, -0.825, -1.14857, cl};
    Point(2092) = {-6.19861, -0.675, -1.19579, cl};
    Point(2093) = {-6.19861, -0.655556, -1.31802, cl};
    Point(2094) = {-6.19861, -0.836111, -0.951349, cl};
    Point(2095) = {-6.19861, -1.00833, -0.959683, cl};
    Point(2096) = {-6.19861, -1.16944, -0.993017, cl};
    Point(2097) = {-6.19861, -0.563887, -0.695793, cl};
    Point(2098) = {-6.19861, -0.658333, -0.918016, cl};
    Point(2099) = {-6.19861, -0.252773, -0.443016, cl};
    Point(2100) = {-6.19861, -0.424996, -0.543016, cl};
    Point(2101) = {-6.19861, 0.836116, -0.95135, cl};
    Point(2102) = {-6.19861, 0.680561, -1.04302, cl};
    Point(2103) = {-6.19861, 0.752783, -1.03746, cl};
    Point(2104) = {-6.19861, 0.655561, -1.31802, cl};
    Point(2105) = {-6.19861, 0.675004, -1.19579, cl};
    Point(2106) = {-6.19861, 0.825004, -1.14857, cl};
    Point(2107) = {-6.19861, 0.888894, -1.20413, cl};
    Point(2108) = {-6.19861, 0.658338, -0.918016, cl};
    Point(2109) = {-6.19861, 0.563894, -0.695793, cl};
    Point(2110) = {-6.19861, 0.96945, -1.2819, cl};
    Point(2111) = {-6.19861, 1.05001, -1.24579, cl};
    Point(2112) = {-6.19861, 1.12501, -1.23468, cl};
    Point(2113) = {-6.19861, 1.15278, -1.08746, cl};
    Point(2114) = {-6.19861, 1.16945, -0.993016, cl};
    Point(2115) = {-6.19861, 1.00833, -0.959682, cl};
    Point(2116) = {-6.19861, 0.425005, -0.543016, cl};
    Point(2117) = {-6.19861, 0.252783, -0.443016, cl};
    Point(2118) = {-6.19861, -0.00277312, -0.390238, cl};
    Point(2119) = {-6.2755, -0.0437672, 0.158373, cl};
    Point(2120) = {-6.2755, 0.0437769, 0.158373, cl};
    Point(2121) = {-6.2755, -0.0807532, 0.158373, cl};
    Point(2122) = {-6.2755, 0.0807626, 0.158373, cl};
    Point(2123) = {-6.3775, -0.114231, -0.379127, cl};
    Point(2124) = {-6.3775, 0.114241, -0.379127, cl};
    Point(2125) = {-6.4769, 1.15278, -1.04728, cl};
    Point(2126) = {-6.4769, 1.15279, -1.15774, cl};
    Point(2127) = {-6.4769, -1.15278, -1.15774, cl};
    Point(2128) = {-6.4769, -1.15278, -1.04728, cl};
    Point(2129) = {-6.5319, -0.186106, -1.71524, cl};
    Point(2130) = {-6.5319, -0.327772, -1.65413, cl};
    Point(2131) = {-6.5319, -0.449995, -1.56246, cl};
    Point(2132) = {-6.5319, -0.536111, -1.46246, cl};
    Point(2133) = {-6.5319, 0.186116, -1.71524, cl};
    Point(2134) = {-6.5319, 4.85461e-06, -1.74024, cl};
    Point(2135) = {-6.5319, -1.04444, -1.19024, cl};
    Point(2136) = {-6.5319, -0.975, -1.19857, cl};
    Point(2137) = {-6.5319, -1.12778, -1.2069, cl};
    Point(2138) = {-6.5319, -1.14722, -1.10968, cl};
    Point(2139) = {-6.5319, -0.802778, -1.10413, cl};
    Point(2140) = {-6.5319, -0.9, -1.1569, cl};
    Point(2141) = {-6.5319, -0.641667, -1.19302, cl};
    Point(2142) = {-6.5319, -0.602778, -1.33468, cl};
    Point(2143) = {-6.5319, -0.75, -1.01802, cl};
    Point(2144) = {-6.5319, -0.833333, -0.979127, cl};
    Point(2145) = {-6.5319, -1.16389, -1.02357, cl};
    Point(2146) = {-6.5319, -1.02778, -0.993016, cl};
    Point(2147) = {-6.5319, -0.625, -0.91246, cl};
    Point(2148) = {-6.5319, -0.641667, -1.01802, cl};
    Point(2149) = {-6.5319, 0.327783, -1.65413, cl};
    Point(2150) = {-6.5319, 0.536115, -1.46246, cl};
    Point(2151) = {-6.5319, 0.450005, -1.56246, cl};
    Point(2152) = {-6.5319, -0.552775, -0.743016, cl};
    Point(2153) = {-6.5319, -0.408325, -0.581904, cl};
    Point(2154) = {-6.5319, -0.241662, -0.479127, cl};
    Point(2155) = {-6.5319, 0.641672, -1.19302, cl};
    Point(2156) = {-6.5319, 0.602783, -1.33468, cl};
    Point(2157) = {-6.5319, 0.900005, -1.1569, cl};
    Point(2158) = {-6.5319, 0.802783, -1.10413, cl};
    Point(2159) = {-6.5319, 0.552783, -0.743016, cl};
    Point(2160) = {-6.5319, 0.833338, -0.979127, cl};
    Point(2161) = {-6.5319, 0.750005, -1.01524, cl};
    Point(2162) = {-6.5319, 0.641672, -1.01802, cl};
    Point(2163) = {-6.5319, 0.625005, -0.91246, cl};
    Point(2164) = {-6.5319, 0.975005, -1.19857, cl};
    Point(2165) = {-6.5319, 1.04445, -1.19024, cl};
    Point(2166) = {-6.5319, 1.14723, -1.10968, cl};
    Point(2167) = {-6.5319, 1.12778, -1.2069, cl};
    Point(2168) = {-6.5319, 1.02779, -0.993017, cl};
    Point(2169) = {-6.5319, 1.1639, -1.02357, cl};
    Point(2170) = {-6.5319, 0.408338, -0.581904, cl};
    Point(2171) = {-6.5319, 0.241672, -0.479127, cl};
    Point(2172) = {-6.5319, 4.6422e-06, -0.431904, cl};
    Point(2173) = {-6.6122, -0.0255423, 0.158373, cl};
    Point(2174) = {-6.6122, 0.0255517, 0.158373, cl};
    Point(2175) = {-6.6122, -0.0672952, 0.158373, cl};
    Point(2176) = {-6.6122, 0.0673017, 0.158373, cl};
    Point(2177) = {-6.6628, -3.00472, -1.42907, cl};
    Point(2178) = {-6.6628, 3.00472, -1.42907, cl};
    Point(2179) = {-6.7319, 4.58405e-06, 2.42643, cl};
    Point(2180) = {-6.7464, 3.00472, -1.42135, cl};
    Point(2181) = {-6.7464, 3.00472, -1.43679, cl};
    Point(2182) = {-6.7464, -3.00472, -1.43679, cl};
    Point(2183) = {-6.7464, -3.00472, -1.42135, cl};
    Point(2184) = {-6.7553, 1.15279, -1.05833, cl};
    Point(2185) = {-6.7553, 1.15279, -1.14671, cl};
    Point(2186) = {-6.7553, -1.15278, -1.14671, cl};
    Point(2187) = {-6.7553, -1.15278, -1.05833, cl};
    Point(2188) = {-6.83, 3.00472, -1.41582, cl};
    Point(2189) = {-6.83, 3.00472, -1.44232, cl};
    Point(2190) = {-6.83, -3.00472, -1.44232, cl};
    Point(2191) = {-6.83, -3.00472, -1.41582, cl};
    Point(2192) = {-6.8708, -0.441659, -1.4819, cl};
    Point(2193) = {-6.8708, -0.324996, -1.58468, cl};
    Point(2194) = {-6.8708, -0.213884, -1.64579, cl};
    Point(2195) = {-6.8708, 0.441672, -1.4819, cl};
    Point(2196) = {-6.8708, 4.56472e-06, -1.6819, cl};
    Point(2197) = {-6.8708, 0.213894, -1.64579, cl};
    Point(2198) = {-6.8708, 0.325005, -1.58468, cl};
    Point(2199) = {-6.8708, -0.783333, -1.0569, cl};
    Point(2200) = {-6.8708, -1.14167, -1.16802, cl};
    Point(2201) = {-6.8708, -1.06111, -1.15135, cl};
    Point(2202) = {-6.8708, -1.15833, -1.05413, cl};
    Point(2203) = {-6.8708, -1.14722, -1.11246, cl};
    Point(2204) = {-6.8708, -0.952778, -1.12635, cl};
    Point(2205) = {-6.8708, -0.861111, -1.09857, cl};
    Point(2206) = {-6.8708, -0.588888, -1.15413, cl};
    Point(2207) = {-6.8708, -0.561112, -1.27635, cl};
    Point(2208) = {-6.8708, -0.508334, -1.39302, cl};
    Point(2209) = {-6.8708, -1.01667, -1.02357, cl};
    Point(2210) = {-6.8708, -0.75, -1.02079, cl};
    Point(2211) = {-6.8708, -0.866668, -1.00413, cl};
    Point(2212) = {-6.8708, -0.491662, -0.76246, cl};
    Point(2213) = {-6.8708, -0.563889, -0.904127, cl};
    Point(2214) = {-6.8708, -0.586112, -1.01802, cl};
    Point(2215) = {-6.8708, -0.202773, -0.529127, cl};
    Point(2216) = {-6.8708, -0.35277, -0.609682, cl};
    Point(2217) = {-6.8708, 0.588894, -1.15413, cl};
    Point(2218) = {-6.8708, 0.508338, -1.39302, cl};
    Point(2219) = {-6.8708, 0.561116, -1.27635, cl};
    Point(2220) = {-6.8708, 0.861116, -1.09857, cl};
    Point(2221) = {-6.8708, 0.783338, -1.0569, cl};
    Point(2222) = {-6.8708, 0.491672, -0.76246, cl};
    Point(2223) = {-6.8708, 0.586116, -1.01802, cl};
    Point(2224) = {-6.8708, 0.563894, -0.904127, cl};
    Point(2225) = {-6.8708, 0.866672, -1.00413, cl};
    Point(2226) = {-6.8708, 0.750005, -1.02079, cl};
    Point(2227) = {-6.8708, 1.01667, -1.02357, cl};
    Point(2228) = {-6.8708, 0.952783, -1.12635, cl};
    Point(2229) = {-6.8708, 1.06111, -1.15135, cl};
    Point(2230) = {-6.8708, 1.14168, -1.16802, cl};
    Point(2231) = {-6.8708, 1.14723, -1.11246, cl};
    Point(2232) = {-6.8708, 1.15833, -1.05413, cl};
    Point(2233) = {-6.8708, 0.352783, -0.609682, cl};
    Point(2234) = {-6.8708, 0.202783, -0.529127, cl};
    Point(2235) = {-6.8708, 4.6689e-06, -0.490238, cl};
    Point(2236) = {-6.89691, -0.0125174, 2.42643, cl};
    Point(2237) = {-6.89691, 0.0125266, 2.42643, cl};
    Point(2238) = {-6.9133, 3.00472, -1.41249, cl};
    Point(2239) = {-6.9133, 3.00472, -1.44565, cl};
    Point(2240) = {-6.91331, -3.00472, -1.44565, cl};
    Point(2241) = {-6.91331, -3.00472, -1.41249, cl};
    Point(2242) = {-6.94861, 4.68591e-06, -0.379127, cl};
    Point(2243) = {-6.94861, 4.7329e-06, 0.158373, cl};
    Point(2244) = {-6.9969, 3.00473, -1.4114, cl};
    Point(2245) = {-6.9969, 3.00472, -1.44674, cl};
    Point(2246) = {-6.9969, -3.00472, -1.44674, cl};
    Point(2247) = {-6.9969, -3.00472, -1.4114, cl};
    Point(2248) = {-7.0336, 1.15279, -1.07675, cl};
    Point(2249) = {-7.0336, 1.15279, -1.12829, cl};
    Point(2250) = {-7.0336, -1.15278, -1.12829, cl};
    Point(2251) = {-7.0336, -1.15278, -1.07675, cl};
    Point(2252) = {-7.0616, -0.0214493, 2.42643, cl};
    Point(2253) = {-7.0616, 0.0214589, 2.42643, cl};
    Point(2254) = {-7.0805, 3.00473, -1.41249, cl};
    Point(2255) = {-7.0805, 3.00473, -1.44565, cl};
    Point(2256) = {-7.0805, -3.00472, -1.44565, cl};
    Point(2257) = {-7.0805, -3.00472, -1.41249, cl};
    Point(2258) = {-7.1639, 3.00472, -1.41582, cl};
    Point(2259) = {-7.1639, 3.00472, -1.44232, cl};
    Point(2260) = {-7.1639, -3.00472, -1.44232, cl};
    Point(2261) = {-7.1639, -3.00472, -1.41582, cl};
    Point(2262) = {-7.1708, -0.447214, -1.47635, cl};
    Point(2263) = {-7.1708, -0.197217, -1.64857, cl};
    Point(2264) = {-7.1708, -0.319438, -1.58468, cl};
    Point(2265) = {-7.1708, 0.447227, -1.47635, cl};
    Point(2266) = {-7.1708, 4.59287e-06, -1.6819, cl};
    Point(2267) = {-7.1708, 0.197227, -1.64857, cl};
    Point(2268) = {-7.1708, -1.04167, -1.05968, cl};
    Point(2269) = {-7.1708, -0.794444, -1.03468, cl};
    Point(2270) = {-7.1708, -1.03611, -1.08468, cl};
    Point(2271) = {-7.1708, -1.14722, -1.1069, cl};
    Point(2272) = {-7.1708, -1.14722, -1.09302, cl};
    Point(2273) = {-7.1708, -1.14722, -1.07913, cl};
    Point(2274) = {-7.1708, -0.869444, -1.05135, cl};
    Point(2275) = {-7.1708, -0.947222, -1.06802, cl};
    Point(2276) = {-7.1708, -0.586108, -1.16524, cl};
    Point(2277) = {-7.1708, -0.519442, -1.36802, cl};
    Point(2278) = {-7.1708, -0.561111, -1.27357, cl};
    Point(2279) = {-7.1708, -0.872222, -1.0319, cl};
    Point(2280) = {-7.1708, -0.752778, -1.01802, cl};
    Point(2281) = {-7.1708, -0.499995, -0.770793, cl};
    Point(2282) = {-7.1708, -0.586111, -1.02635, cl};
    Point(2283) = {-7.1708, -0.566664, -0.920794, cl};
    Point(2284) = {-7.1708, -0.363882, -0.618016, cl};
    Point(2285) = {-7.1708, -0.211106, -0.531904, cl};
    Point(2286) = {-7.1708, 0.869449, -1.05135, cl};
    Point(2287) = {-7.1708, 0.794448, -1.03468, cl};
    Point(2288) = {-7.1708, 0.586116, -1.16524, cl};
    Point(2289) = {-7.1708, 0.561116, -1.27357, cl};
    Point(2290) = {-7.1708, 0.519449, -1.36802, cl};
    Point(2291) = {-7.1708, 0.500005, -0.770793, cl};
    Point(2292) = {-7.1708, 0.566672, -0.920793, cl};
    Point(2293) = {-7.1708, 0.586116, -1.02635, cl};
    Point(2294) = {-7.1708, 0.750005, -1.02079, cl};
    Point(2295) = {-7.1708, 0.872227, -1.0319, cl};
    Point(2296) = {-7.1708, 1.04168, -1.05968, cl};
    Point(2297) = {-7.1708, 0.947227, -1.06802, cl};
    Point(2298) = {-7.1708, 1.03612, -1.08468, cl};
    Point(2299) = {-7.1708, 1.14722, -1.1069, cl};
    Point(2300) = {-7.1708, 1.14722, -1.07913, cl};
    Point(2301) = {-7.1708, 1.14722, -1.09302, cl};
    Point(2302) = {-7.1708, 0.363894, -0.618016, cl};
    Point(2303) = {-7.1708, -0.00277336, -0.490238, cl};
    Point(2304) = {-7.1708, 0.211116, -0.531904, cl};
    Point(2305) = {-7.1708, 0.319449, -1.58468, cl};
    Point(2306) = {-7.2266, -0.0268191, 2.42643, cl};
    Point(2307) = {-7.2266, 0.0268287, 2.42643, cl};
    Point(2308) = {-7.2475, 3.00472, -1.42135, cl};
    Point(2309) = {-7.2475, 3.00472, -1.43679, cl};
    Point(2310) = {-7.2475, -3.00472, -1.43679, cl};
    Point(2311) = {-7.2475, -3.00472, -1.42135, cl};
    Point(2312) = {-7.31191, 0.181394, -1.65024, cl};
    Point(2313) = {-7.31191, 0.432505, -1.49052, cl};
    Point(2314) = {-7.31191, 0.319449, -1.58274, cl};
    Point(2315) = {-7.31191, -0.432494, -1.49052, cl};
    Point(2316) = {-7.31191, -0.319439, -1.58274, cl};
    Point(2317) = {-7.31191, 0.0011157, -1.67663, cl};
    Point(2318) = {-7.31191, -0.181384, -1.65024, cl};
    Point(2319) = {-7.31191, 0.501116, -1.40135, cl};
    Point(2320) = {-7.31191, 0.555838, -1.29274, cl};
    Point(2321) = {-7.31191, 0.585283, -1.17079, cl};
    Point(2322) = {-7.31191, 0.906394, -1.05746, cl};
    Point(2323) = {-7.31191, 1.07306, -1.08357, cl};
    Point(2324) = {-7.31191, 0.986949, -1.07329, cl};
    Point(2325) = {-7.31191, 0.836949, -1.04968, cl};
    Point(2326) = {-7.31191, 0.753616, -1.03357, cl};
    Point(2327) = {-7.31191, 0.589727, -1.03218, cl};
    Point(2328) = {-7.31191, 0.558338, -0.893849, cl};
    Point(2329) = {-7.31191, 0.476394, -0.733849, cl};
    Point(2330) = {-7.31191, 0.353338, -0.60746, cl};
    Point(2331) = {-7.31191, 0.202783, -0.525793, cl};
    Point(2332) = {-7.31191, 4.70995e-06, -0.494128, cl};
    Point(2333) = {-7.31191, -0.202773, -0.525793, cl};
    Point(2334) = {-7.31191, -0.353329, -0.60746, cl};
    Point(2335) = {-7.31191, -0.476381, -0.733849, cl};
    Point(2336) = {-7.31191, -0.558333, -0.893849, cl};
    Point(2337) = {-7.31191, -1.07306, -1.08357, cl};
    Point(2338) = {-7.31191, -0.989722, -1.07329, cl};
    Point(2339) = {-7.31191, -0.906389, -1.05746, cl};
    Point(2340) = {-7.31191, -0.834167, -1.04968, cl};
    Point(2341) = {-7.31191, -0.753611, -1.03635, cl};
    Point(2342) = {-7.31191, -0.592774, -1.03496, cl};
    Point(2343) = {-7.31191, -0.585274, -1.17079, cl};
    Point(2344) = {-7.31191, -0.501111, -1.40135, cl};
    Point(2345) = {-7.31191, -0.55583, -1.29274, cl};
    Point(2346) = {-7.3122, 1.15278, -1.10252, cl};
    Point(2347) = {-7.3122, -1.15278, -1.10252, cl};
    Point(2348) = {-7.33111, 3.00472, -1.42907, cl};
    Point(2349) = {-7.33111, -3.00472, -1.42907, cl};
    Point(2350) = {-7.39161, -0.0286011, 2.42643, cl};
    Point(2351) = {-7.39161, 0.0286107, 2.42643, cl};
    Point(2352) = {-7.5566, -0.0268194, 2.42643, cl};
    Point(2353) = {-7.5566, 0.0268288, 2.42643, cl};
    Point(2354) = {-7.72141, -0.0214492, 2.42643, cl};
    Point(2355) = {-7.72141, 0.0214586, 2.42643, cl};
    Point(2356) = {-7.88641, -0.0125173, 2.42643, cl};
    Point(2357) = {-7.88641, 0.0125267, 2.42643, cl};
    Point(2358) = {-8.05141, 4.70787e-06, 2.42643, cl};
    Line(573) = {1653,1836};
    Line(574) = {1652,1837};
    Delete {
    Point{1683,1685,1672,1681,1695,1671,1697,1670,1676,1680,1677,1679,1669,1678,1682,1673,1686,1684,1674,1696,1675,1662,1663,1661,1691,1657,1665,1666,1656,1655,1694,1693,1664,1668,1660,1654,1690,1658,1659,1688,1692,1689,1667,1687};
    }
    CatmullRom(575) = {2179,2236,2252,2306,2350};
    CatmullRom(576) = {2350,2352,2354,2356,2358};
    CatmullRom(577) = {2358,2357,2355,2353,2351};
    CatmullRom(578) = {2351,2307,2253,2237,2179};
    Line(579) = {2358,2243};
    Line(580) = {2179,1768};
    Line(581) = {2351,1986};
    Line(582) = {1985,2350};
    CatmullRom(583) = {1768,1831,1884,1935,1986};
    CatmullRom(584) = {1768,1830,1883,1934,1985};
    CatmullRom(585) = {1986,2068,2120,2174,2243};
    CatmullRom(586) = {1985,2067,2119,2173,2243};
    Delete {
    Point{1645,1644};
    }
    Delete {
    Point{1826,1827};
    }
    Delete {
    Point{1829,1828,1646,1647};
    }
    CatmullRom(587) = {1579,1700,1708,1718,1766,1775,1836};
    CatmullRom(588) = {1836,1776,1767,1719,1709,1701,1579};
    CatmullRom(589) = {1653,1579};
    CatmullRom(590) = {1578,1698,1706,1716,1764,1773,1837};
    CatmullRom(591) = {1837,1774,1765,1717,1707,1699,1578};
    CatmullRom(592) = {1652,1578};
    Delete {
    Point{295};
    }
    Delete {
    Point{317};
    }
    Delete {
    Line{94,107,67,88};
    }
    Delete {
    Line{75,99,85,110};
    }
    CatmullRom(593) = {171,170,169,168};
    CatmullRom(594) = {182,183,184,185};
    CatmullRom(595) = {230,229,232,231};
    CatmullRom(596) = {217,218,215,214};
    CatmullRom(597) = {230,316};
    CatmullRom(598) = {217,316};
    CatmullRom(599) = {301,299,298,296,297,288};
    CatmullRom(600) = {330,326,328,324,327,322};
    CatmullRom(601) = {288,289,287,291,290,292,293,294,286,316};
    CatmullRom(602) = {316,313,315,314,318,319,321,320,323,322};
    CatmullRom(603) = {1886,1939,1992,2066,2074,2128,2187,2251,2347};
    CatmullRom(604) = {2347,2250,2186,2127,2073,2065,1991,1938,1886};
    CatmullRom(605) = {2349,2311,2261,2257,2247,2241,2191,2183,2177};
    CatmullRom(606) = {2177,2182,2190,2240,2246,2256,2260,2310,2349};
    CatmullRom(607) = {2349,2347};
    CatmullRom(608) = {2177,1886};
    CatmullRom(609) = {1885,1936,1989,2063,2071,2125,2184,2248,2346};
    CatmullRom(610) = {2346,2249,2185,2126,2072,2064,1990,1937,1885};
    CatmullRom(611) = {2178,2180,2188,2238,2244,2254,2258,2308,2348};
    CatmullRom(612) = {2178,2181,2189,2239,2245,2255,2259,2309,2348};
    CatmullRom(613) = {1885,2178};
    CatmullRom(614) = {2346,2348};
    CatmullRom(615) = {1207,1283,1370,1457,1544,1633};
    CatmullRom(616) = {1207,1282,1371,1456,1547,1636};
    CatmullRom(617) = {1633,1636};
    CatmullRom(618) = {1219,1264,1352,1435,1527,1617};
    CatmullRom(619) = {1219,1265,1354,1436,1528,1618};
    CatmullRom(620) = {1618,1617};
    CatmullRom(621) = {1621,1615,1614,1616,1619,1618};
    CatmullRom(622) = {1630,1631,1632,1634,1635,1636};
    Delete {
    Point{1754,1867};
    }
    Delete {
    Point{1736,1854};
    }
    Delete {
    Point{1923,1976,2055,2113,2166,2231};
    }
    Delete {
    Point{1901,1950,2024,2088,2138,2203};
    }
    CatmullRom(623) = {1639,1755,1797,1872,1927};
    CatmullRom(624) = {1927,1885};
    CatmullRom(625) = {1636,1761,1795,1869,1926};
    CatmullRom(626) = {1926,1885};
    CatmullRom(627) = {1630,1749,1818,1864,1918};
    CatmullRom(628) = {1918,1917};
    CatmullRom(629) = {1917,1882,1791,1748,1629};
    CatmullRom(630) = {1918,1920,1926};
    CatmullRom(631) = {1909,1860,1817,1744,1622};
    CatmullRom(632) = {1909,1886};
    CatmullRom(633) = {1886,1902};
    CatmullRom(634) = {1902,1856,1813,1737,1618};
    CatmullRom(635) = {1906,1904,1902};
    CatmullRom(636) = {1906,1858,1815,1740,1621};
    CatmullRom(637) = {1620,1741,1816,1857,1907};
    CatmullRom(638) = {1907,1906};

    Point(1404) = {-2.95031, -0.0333533, -0.38, cl};
    Point(1405) = {-2.95031, 0.0333607, -0.38, cl};
    Point(1576) = {-3.52161, -0.0680812, -0.379127, cl};
    Point(1577) = {-3.52161, 0.0680876, -0.379127, cl};
    CatmullRom(639) = {1289,1404,1576,1625};
    CatmullRom(640) = {1289,1405,1577,1641};
    Line Loop(641) = {447,512,-518,-516};
    Ruled Surface(642) = {641};
    Line Loop(643) = {516,517,-514,-448};
    Ruled Surface(644) = {643};
    Delete {
    Point{1241};
    }
    Delete {
    Point{1291,1288,1271,1272,1361,1359,1377,1378};
    }
    Line Loop(645) = {556,294,535,-521};
    Ruled Surface(646) = {645};
    Line Loop(647) = {302,531,-523,-533};
    Ruled Surface(648) = {647};
    Line Loop(649) = {293,-554,-537,555};
    Ruled Surface(650) = {649};
    Line Loop(651) = {104,117,-120,-116};
    Ruled Surface(652) = {651};
    Line Loop(653) = {403,392,-420,422};
    Ruled Surface(654) = {653};
    Line Loop(655) = {80,-72,65};
    Ruled Surface(656) = {655};
    Line Loop(657) = {65,79,-71};
    Ruled Surface(658) = {657};
    Line Loop(659) = {89,103,-102,-80};
    Ruled Surface(660) = {659};
    Line Loop(661) = {89,104,-90,-79};
    Ruled Surface(662) = {661};
    Line Loop(663) = {119,-118,-103,116};
    Ruled Surface(664) = {663};
    Line Loop(665) = {400,-421,-417,-416};
    Ruled Surface(666) = {665};
    Line Loop(667) = {608,603,-607,605};
    Ruled Surface(668) = {667};
    Line Loop(669) = {611,-614,-609,613};
    Ruled Surface(670) = {669};
    Line Loop(671) = {610,613,612,-614};
    Ruled Surface(672) = {671};
    Line Loop(673) = {604,-608,606,607};
    Ruled Surface(674) = {673};
    Line Loop(675) = {548,292,547,-537};
    Ruled Surface(676) = {675};
    Line Loop(677) = {295,546,-538,-547};
    Ruled Surface(678) = {677};
    Line Loop(679) = {292,-536,-521,-549};
    Ruled Surface(680) = {679};
    Line Loop(681) = {295,-534,-522,536};
    Ruled Surface(682) = {681};
    Line Loop(683) = {534,296,-532,-523};
    Ruled Surface(684) = {683};
    Line Loop(685) = {296,-545,-539,-546};
    Ruled Surface(686) = {685};
    Line Loop(687) = {545,297,544,-540};
    Ruled Surface(688) = {687};
    Line Loop(689) = {297,-530,-524,532};
    Ruled Surface(690) = {689};
    Line Loop(691) = {530,439,-528,-525};
    Ruled Surface(692) = {691};
    Line Loop(693) = {544,541,543,-439};
    Ruled Surface(694) = {693};
    Line Loop(695) = {543,559,-560,-542};
    Ruled Surface(696) = {695};
    Line Loop(697) = {528,559,561,-526};
    Ruled Surface(698) = {697};
    Line Loop(699) = {529,525,-527,435};
    Ruled Surface(700) = {699};
    Line Loop(701) = {306,529,-524,-531};
    Ruled Surface(702) = {701};
    Line Loop(703) = {580,583,-581,578};
    Ruled Surface(704) = {703};
    Line Loop(705) = {582,-575,580,584};
    Ruled Surface(706) = {705};
    Line Loop(707) = {581,585,-579,577};
    Ruled Surface(708) = {707};
    Line Loop(709) = {582,576,579,-586};
    Ruled Surface(710) = {709};
    Line Loop(711) = {446,447,-452,-418};
    Ruled Surface(712) = {711};
    Line Loop(713) = {448,-451,-417,446};
    Ruled Surface(714) = {713};
    Line Loop(715) = {418,422,-401,416};
    Ruled Surface(716) = {715};
    Line Loop(717) = {401,426,-410,287};
    Ruled Surface(718) = {717};
    Line Loop(719) = {400,423,-404,287};
    Ruled Surface(720) = {719};
    Line Loop(721) = {404,424,-406,275};
    Ruled Surface(722) = {721};
    Line Loop(723) = {410,427,-412,274};
    Ruled Surface(724) = {723};
    Line Loop(725) = {275,-276,273};
    Ruled Surface(726) = {725};
    Line Loop(727) = {274,-276,272};
    Ruled Surface(728) = {727};
    Line Loop(729) = {272,-270,-267,269};
    Ruled Surface(730) = {729};
    Line Loop(731) = {273,-271,-268,269};
    Ruled Surface(732) = {731};
    Line Loop(733) = {268,-266,-119,264};
    Ruled Surface(734) = {733};
    Line Loop(735) = {267,-265,-120,264};
    Ruled Surface(736) = {735};
    Line Loop(737) = {412,428,-414,270};
    Ruled Surface(738) = {737};
    Line Loop(739) = {414,-511,-507,265};
    Ruled Surface(740) = {739};
    Line Loop(741) = {413,-280,-415,-428};
    Ruled Surface(742) = {741};
    Line Loop(743) = {415,-277,-600,511};
    Ruled Surface(744) = {743};
    Line Loop(745) = {600,-114,-430,510};
    Ruled Surface(746) = {745};
    Line Loop(747) = {105,-91,-81,90};
    Ruled Surface(748) = {747};
    Line Loop(749) = {81,-70,-64,71};
    Ruled Surface(750) = {749};
    Line Loop(751) = {411,-282,-413,-427};
    Ruled Surface(752) = {751};
    Line Loop(753) = {403,-284,-411,-426};
    Ruled Surface(754) = {753};
    Line Loop(755) = {450,-455,-420,452};
    Ruled Surface(756) = {755};
    Line Loop(757) = {271,406,425,-408};
    Ruled Surface(758) = {757};
    Line Loop(759) = {266,408,-509,-506};
    Ruled Surface(760) = {759};
    Line Loop(761) = {506,-508,112,118};
    Ruled Surface(762) = {761};
    Line Loop(763) = {102,-112,-101,-83};
    Ruled Surface(764) = {763};
    Line Loop(765) = {431,-98,-432,101};
    Ruled Surface(766) = {765};
    Line Loop(767) = {72,83,-69,-57};
    Ruled Surface(768) = {767};
    Line Loop(769) = {49,57,-56,48};
    Ruled Surface(770) = {769};
    Line Loop(771) = {48,-33,25,34};
    Ruled Surface(772) = {771};
    Line Loop(773) = {41,-36,32,33};
    Ruled Surface(774) = {773};
    Line Loop(775) = {64,-50,-41,49};
    Ruled Surface(776) = {775};
    Line Loop(777) = {430,-93,-429,91};
    Ruled Surface(778) = {777};
    Line Loop(779) = {345,-303,-301,-344};
    Ruled Surface(780) = {779};
    Line Loop(781) = {301,533,-522,-535};
    Ruled Surface(782) = {781};
    Line Loop(783) = {494,527,526,-558};
    Ruled Surface(784) = {783};
    Line Loop(785) = {509,409,-278,-599};
    Ruled Surface(786) = {785};
    Line Loop(787) = {599,-115,-431,508};
    Ruled Surface(788) = {787};
    Line Loop(789) = {69,432,-74,-58};
    Ruled Surface(790) = {789};
    Line Loop(791) = {58,-55,47,56};
    Ruled Surface(792) = {791};
    Line Loop(793) = {47,-34,26,35};
    Ruled Surface(794) = {793};
    Line Loop(795) = {46,-35,27,40};
    Ruled Surface(796) = {795};
    Line Loop(797) = {26,-23,-10,24};
    Ruled Surface(798) = {797};
    Line Loop(799) = {25,-24,-9,17};
    Ruled Surface(800) = {799};
    Line Loop(801) = {32,-17,-16,18};
    Ruled Surface(802) = {801};
    Line Loop(803) = {507,-510,-105,117};
    Ruled Surface(804) = {803};
    Line Loop(805) = {42,-37,31,36};
    Ruled Surface(806) = {805};
    Line Loop(807) = {304,-308,-303,302};
    Ruled Surface(808) = {807};
    Line Loop(809) = {307,-305,-306,304};
    Ruled Surface(810) = {809};
    Line Loop(811) = {449,-485,-419,451};
    Ruled Surface(812) = {811};
    Line Loop(813) = {419,-394,-402,-421};
    Ruled Surface(814) = {813};
    Line Loop(815) = {402,-283,-405,-423};
    Ruled Surface(816) = {815};
    Line Loop(817) = {405,-281,-407,-424};
    Ruled Surface(818) = {817};
    Line Loop(819) = {407,-279,-409,-425};
    Ruled Surface(820) = {819};
    Line Loop(821) = {321,-553,-538,554};
    Ruled Surface(822) = {821};
    Line Loop(823) = {553,322,-552,-539};
    Ruled Surface(824) = {823};
    Line Loop(825) = {325,-551,-540,552};
    Ruled Surface(826) = {825};
    Line Loop(827) = {326,436,437,438};
    Ruled Surface(828) = {827};
    Line Loop(829) = {438,-551,541,550};
    Ruled Surface(830) = {829};
    Line Loop(831) = {490,557,-542,550};
    Ruled Surface(832) = {831};
    Line Loop(833) = {55,59,-54,46};
    Ruled Surface(834) = {833};
    Line Loop(835) = {74,-594,-76,-59};
    Ruled Surface(836) = {835};
    Line Loop(837) = {60,77,-86,-76};
    Ruled Surface(838) = {837};
    Line Loop(839) = {44,53,61,-52};
    Ruled Surface(840) = {839};
    Line Loop(841) = {77,87,-78,-61};
    Ruled Surface(842) = {841};
    Line Loop(843) = {96,-108,-95,-87};
    Ruled Surface(844) = {843};
    Line Loop(845) = {109,598,-113};
    Ruled Surface(846) = {845};
    Line Loop(847) = {108,113,-597};
    Ruled Surface(848) = {847};
    Line Loop(849) = {54,60,-53,45};
    Ruled Surface(850) = {849};
    Line Loop(851) = {40,-45,-39,-28};
    Ruled Surface(852) = {851};
    Line Loop(853) = {22,28,-21,-12};
    Ruled Surface(854) = {853};
    Line Loop(855) = {27,-22,-11,23};
    Ruled Surface(856) = {855};
    Line Loop(857) = {13,20,-29,-21};
    Ruled Surface(858) = {857};
    Line Loop(859) = {62,68,-593,-78};
    Ruled Surface(860) = {859};
    Line Loop(861) = {593,93,-595,-95};
    Ruled Surface(862) = {861};
    Line Loop(863) = {29,38,44,-39};
    Ruled Surface(864) = {863};
    Line Loop(865) = {37,43,-38,30};
    Ruled Surface(866) = {865};
    Line Loop(867) = {51,-62,-52,-43};
    Ruled Surface(868) = {867};
    Line Loop(869) = {2,-15,-3};
    Ruled Surface(870) = {869};
    Line Loop(871) = {1,-16,-2};
    Ruled Surface(872) = {871};
    Line Loop(873) = {8,-9,-1};
    Ruled Surface(874) = {873};
    Line Loop(875) = {8,10,-7};
    Ruled Surface(876) = {875};
    Line Loop(877) = {7,11,-6};
    Ruled Surface(878) = {877};
    Line Loop(879) = {6,12,-5};
    Ruled Surface(880) = {879};
    Line Loop(881) = {5,13,-4};
    Ruled Surface(882) = {881};
    Line Loop(883) = {4,14,-3};
    Ruled Surface(884) = {883};
    Line Loop(885) = {19,-30,-20,14};
    Ruled Surface(886) = {885};
    Line Loop(887) = {18,-31,-19,15};
    Ruled Surface(888) = {887};
    Line Loop(889) = {86,96,109,-97};
    Ruled Surface(890) = {889};
    Line Loop(891) = {433,434,435,305};
    Ruled Surface(892) = {891};
    Line Loop(893) = {492,493,494,-434};
    Ruled Surface(894) = {893};
    Line Loop(895) = {565,566,567,-493};
    Ruled Surface(896) = {895};
    Line Loop(897) = {627,628,629,-566};
    Ruled Surface(898) = {897};
    Delete {
    Line{259};
    }
    Delete {
    Line{258};
    }
    Delete {
    Line{257};
    }
    CatmullRom(899) = {1594,1595,1596,1597};
    Delete {
    Line{262};
    }
    Delete {
    Line{263};
    }
    Delete {
    Line{261};
    }
    CatmullRom(900) = {1592,1593,1590,1591};
    Line Loop(901) = {471,475,-900,-487};
    Ruled Surface(902) = {901};
    Line Loop(903) = {453,467,-899,-469};
    Ruled Surface(904) = {903};
    Line Loop(905) = {454,468,-899,-470};
    Ruled Surface(906) = {905};
    Line Loop(907) = {488,900,-476,-472};
    Ruled Surface(908) = {907};
    Line Loop(909) = {596,-98,-594,97};
    Ruled Surface(910) = {909};
    Line Loop(911) = {443,-277,-602,346};
    Ruled Surface(912) = {911};
    Line Loop(913) = {280,-444,-345,443};
    Ruled Surface(914) = {913};
    Line Loop(915) = {282,-445,-308,444};
    Ruled Surface(916) = {915};
    Line Loop(917) = {284,-520,-307,445};
    Ruled Surface(918) = {917};
    Line Loop(919) = {281,-442,340,441};
    Ruled Surface(920) = {919};
    Line Loop(921) = {340,-323,322,324};
    Ruled Surface(922) = {921};
    Line Loop(923) = {327,-326,-325,324};
    Ruled Surface(924) = {923};
    Line Loop(925) = {283,-571,-327,442};
    Ruled Surface(926) = {925};
    Line Loop(927) = {70,429,-68,63};
    Ruled Surface(928) = {927};
    Line Loop(929) = {50,-63,-51,-42};
    Ruled Surface(930) = {929};
    Line Loop(931) = {346,344,-294,-291};
    Ruled Surface(932) = {931};
    Line Loop(933) = {341,342,321,323};
    Ruled Surface(934) = {933};
    Line Loop(935) = {343,342,-293,-291};
    Ruled Surface(936) = {935};
    Line Loop(937) = {395,-572,-436,571};
    Ruled Surface(938) = {937};
    Line Loop(939) = {572,486,-570,-489};
    Ruled Surface(940) = {939};
    Line Loop(941) = {491,-490,-437,489};
    Ruled Surface(942) = {941};
    Line Loop(943) = {563,564,-491,562};
    Ruled Surface(944) = {943};
    Line Loop(945) = {636,563,637,638};
    Ruled Surface(946) = {945};
    Line Loop(947) = {634,-621,-636,635};
    Ruled Surface(948) = {947};
    Line Loop(949) = {440,279,-441,341};
    Ruled Surface(950) = {949};
    Line Loop(951) = {596,115,601,-598};
    Ruled Surface(952) = {951};
    Line Loop(953) = {595,114,-602,-597};
    Ruled Surface(954) = {953};
    Line Loop(955) = {278,-440,-343,-601};
    Ruled Surface(956) = {955};
    Line Loop(957) = {520,393,-519,-433};
    Ruled Surface(958) = {957};
    Line Loop(959) = {519,456,-569,-492};
    Ruled Surface(960) = {959};
    Line Loop(961) = {615,617,-616};
    Ruled Surface(962) = {961};
    Line Loop(963) = {496,-502,501};
    Ruled Surface(964) = {963};
    Line Loop(965) = {498,497,-499};
    Ruled Surface(966) = {965};
    Line Loop(967) = {618,-620,-619};
    Ruled Surface(968) = {967};
    Delete {
    Line{389};
    }
    Delete {
    Line{363};
    }
    Delete {
    Line{364};
    }
    CatmullRom(969) = {980,1030,1167,1384};
    Delete {
    Line{390};
    }
    Delete {
    Line{365};
    }
    Delete {
    Line{366};
    }
    CatmullRom(970) = {1385,1168,1029,979};
    Line Loop(971) = {969,473,-471,-485};
    Ruled Surface(972) = {971};
    Line Loop(973) = {472,-474,970,486};
    Ruled Surface(974) = {973};
    Delete {
    Line{285};
    }
    Delete {
    Line{260};
    }
    CatmullRom(975) = {807,1033,1296};
    Delete {
    Line{975};
    }
    CatmullRom(975) = {743,807,1033,1296};
    Line Loop(976) = {395,-970,-374,-975};
    Ruled Surface(977) = {976};
    Line Loop(978) = {373,-969,-394,975};
    Ruled Surface(979) = {978};
    Delete {
    Surface{960};
    }
    Delete {
    Line{569};
    }
    Delete {
    Surface{948};
    }
    Delete {
    Surface{908,940};
    }
    Delete {
    Line{570};
    }
    Delete {
    Surface{812};
    }
    Delete {
    Line{449};
    }
    Delete {
    Surface{756};
    }
    Delete {
    Line{450};
    }
    Delete {
    Line{286};
    }
    Delete {
    Line{256};
    }
    Delete {
    Line{391};
    }
    Delete {
    Line{349};
    }
    Delete {
    Line{350};
    }
    CatmullRom(980) = {981,1031,1170,1387};
    CatmullRom(981) = {1297,1034,804,744};
    Delete {
    Line{388};
    }
    Delete {
    Line{348};
    }
    Delete {
    Line{347};
    }
    CatmullRom(982) = {982,1032,1169,1386};
    Line Loop(983) = {982,-359,981,392};
    Ruled Surface(984) = {983};
    Line Loop(985) = {980,-361,981,393};
    Ruled Surface(986) = {985};
    Line Loop(987) = {454,-462,-980,456};
    Ruled Surface(988) = {987};
    Line Loop(989) = {461,-453,-455,982};
    Ruled Surface(990) = {989};
    CatmullRom(991) = {1212,1198,1199,1209};
    Line Loop(992) = {513,501,-991,512};
    Ruled Surface(993) = {992};
    CatmullRom(994) = {1201,1204,1206,1207};
    Line Loop(995) = {622,-616,-994,565};
    Ruled Surface(996) = {995};
    CatmullRom(997) = {1197,1192,1193,1194};
    CatmullRom(998) = {1190,1188,1220,1219};
    Line Loop(999) = {997,-498,-515,-514};
    Ruled Surface(1000) = {999};
    Line Loop(1001) = {998,619,-621,-562};
    Ruled Surface(1002) = {1001};
    Ruled Surface(1003) = {907};
    Line(1004) = {1207,1173};
    Line Loop(1005) = {1004,470,503,-615};
    Ruled Surface(1006) = {1005};
    Line(1007) = {1209,1174};
    Line Loop(1008) = {1007,469,503,-496};
    Ruled Surface(1009) = {1008};
    Line(1010) = {1194,1172};
    Line Loop(1011) = {1010,487,-500,-497};
    Ruled Surface(1012) = {1011};
    Line(1013) = {1171,1219};
    Line Loop(1014) = {488,-500,-618,-1013};
    Ruled Surface(1015) = {1014};
    CatmullRom(1016) = {1194,1081,980};
    CatmullRom(1017) = {1209,1096,982};
    Line Loop(1018) = {420,-1017,-991,-452};
    Ruled Surface(1019) = {1018};
    Line Loop(1020) = {1017,455,-1007};
    Ruled Surface(1021) = {1020};
    Line Loop(1022) = {1016,-419,451,997};
    Ruled Surface(1023) = {1022};
    Line Loop(1024) = {485,-1010,1016};
    Ruled Surface(1025) = {1024};
    CatmullRom(1026) = {1219,1077,1004};
    Line(1027) = {1004,965};
    Line(1028) = {1004,979};
    Line Loop(1029) = {1013,1026,1028,486};
    Ruled Surface(1030) = {1029};
    Line Loop(1031) = {998,1026,1027,489};
    Ruled Surface(1032) = {1031};
    Line Loop(1033) = {1027,572,-1028};
    Ruled Surface(1034) = {1033};
    Delete {
    Surface{1034};
    }
    Ruled Surface(1034) = {1033};
    CatmullRom(1035) = {1207,1093,1020};
    CatmullRom(1036) = {1020,981};
    CatmullRom(1037) = {1020,933};
    Line Loop(1038) = {1037,519,-1036};
    Ruled Surface(1039) = {1038};
    Line Loop(1040) = {994,1035,1037,492};
    Ruled Surface(1041) = {1040};
    Line Loop(1042) = {1004,-456,-1036,-1035};
    Ruled Surface(1043) = {1042};
    Line(1044) = {1476,1477};
    Line(1045) = {1384,1385};
    Line Loop(1046) = {374,-1045,-373};
    Ruled Surface(1047) = {1046};
    Line Loop(1048) = {1045,474,-1044,-473};
    Ruled Surface(1049) = {1048};
    Line Loop(1050) = {476,-475,1044};
    Ruled Surface(1051) = {1050};
    Line(1052) = {1386,1387};
    Line(1053) = {1474,1475};
    Line Loop(1054) = {361,-1052,-359};
    Ruled Surface(1055) = {1054};
    Line Loop(1056) = {1052,462,-1053,-461};
    Ruled Surface(1057) = {1056};
    Line Loop(1058) = {1053,468,-467};
    Ruled Surface(1059) = {1058};
    CatmullRom(1060) = {1885,1792,1633};
    Line Loop(1061) = {1060,617,625,626};
    Ruled Surface(1062) = {1061};
    Line Loop(1063) = {623,624,1060,-502};
    Ruled Surface(1064) = {1063};
    CatmullRom(1065) = {1886,1812,1617};
    Line Loop(1066) = {631,499,-1065,-632};
    Ruled Surface(1067) = {1066};
    Line Loop(1068) = {1065,-620,-634,-633};
    Ruled Surface(1069) = {1068};
    Ruled Surface(1070) = {947};
    Line Loop(1071) = {630,-625,-622,627};
    Ruled Surface(1072) = {1071};
    CatmullRom(1073) = {1835,1825,1772,1713,1705,1651,1573,1497,1465};
    CatmullRom(1074) = {1835,1824,1771,1712,1704,1650,1572,1496,1465};
    CatmullRom(1075) = {1579,1465};
    CatmullRom(1076) = {1835,1836};
    CatmullRom(1077) = {1834,1822,1769,1710,1702,1648,1574,1494,1464};
    CatmullRom(1078) = {1464,1495,1575,1649,1703,1711,1770,1823,1834};
    CatmullRom(1079) = {1834,1837};
    CatmullRom(1080) = {1578,1464};
    Line Loop(1081) = {1076,588,1075,-1073};
    Ruled Surface(1082) = {1081};
    Line Loop(1083) = {1074,-1075,587,-1076};
    Ruled Surface(1084) = {1083};
    Line Loop(1085) = {1078,1079,591,1080};
    Ruled Surface(1086) = {1085};
    Line Loop(1087) = {1077,-1080,590,-1079};
    Ruled Surface(1088) = {1087};
    //+
    Physical Point(1) -= {2285, 2334};
    //+
    Physical Point(1) -= {2242};
    //+
    Physical Point(1) -= {2242};
    //+
    Physical Point(1) -= {2242};
    //+
    Recursive Delete {
    Point{2333}; 
    }
    //+
    Recursive Delete {
    Point{2242}; 
    }
    //+
    Recursive Delete {
    Point{2285}; 
    }
    //+
    Recursive Delete {
    Point{2235}; 
    }
    //+
    Recursive Delete {
    Point{2334}; 
    }
    //+
    Recursive Delete {
    Point{2171}; 
    }
    //+
    Recursive Delete {
    Point{2284}; 
    }
    //+
    Recursive Delete {
    Point{2215}; 
    }
    //+
    Recursive Delete {
    Point{2172}; Point{2124}; Point{2117}; Point{2216}; Point{2154}; Point{2118}; Point{2123}; Point{2153}; Point{1994}; Point{2099}; Point{2061}; Point{1993}; Point{2039}; Point{1983}; Point{1965}; 
    }
    //+
    Recursive Delete {
    Point{2100}; Point{1984}; Point{2152}; Point{1929}; Point{2097}; Point{1875}; Point{1930}; Point{1933}; Point{2038}; Point{1915}; Point{1931}; Point{1968}; 
    }
    //+
    Recursive Delete {
    Point{1932}; Point{1874}; Point{2037}; Point{2031}; Point{1800}; Point{1969}; Point{1833}; Point{1798}; Point{1967}; Point{1763}; Point{1966}; Point{2035}; Point{2033}; 
    }
    //+
    Recursive Delete {
    Point{2096}; 
    }
    //+
    Recursive Delete {
    Point{1912}; Point{1880}; Point{1808}; Point{1832}; 
    }
    //+
    Recursive Delete {
    Point{1916}; Point{1850}; Point{1873}; 
    }
    //+
    Recursive Delete {
    Point{1821}; Point{1757}; Point{1910}; Point{1715}; Point{1881}; Point{1862}; Point{1758}; Point{1787}; Point{1714}; Point{1788}; Point{1745}; 
    }
    //+
    Recursive Delete {
    Point{2095}; Point{2032}; Point{2036}; Point{2145}; Point{2094}; Point{2146}; Point{2202}; Point{2098}; 
    }
    //+
    Recursive Delete {
    Point{2144}; Point{2209}; Point{1876}; 
    }
    //+
    Recursive Delete {
    Point{2147}; Point{2211}; 
    }
    //+
    Recursive Delete {
    Point{2273}; 
    }
    //+
    Recursive Delete {
    Point{2271}; Point{2272}; 
    }
    //+
    Recursive Delete {
    Point{2268}; Point{2270}; Point{2199}; 
    }
    //+
    Recursive Delete {
    Point{2148}; Point{2204}; Point{1863}; Point{2205}; Point{2210}; Point{2337}; Point{2275}; Point{1922}; Point{2338}; Point{2274}; Point{2279}; Point{2339}; Point{2269}; Point{2213}; Point{1982}; Point{2212}; Point{2062}; Point{2214}; Point{2201}; Point{2139}; Point{2200}; Point{2085}; Point{1790}; Point{2140}; 
    }
    //+
    Recursive Delete {
    Point{2280}; Point{1974}; Point{2283}; Point{2282}; Point{2341}; Point{2340}; Point{2342}; Point{2048}; Point{2049}; Point{2336}; Point{2281}; Point{2060}; Point{2335}; Point{2116}; Point{2170}; Point{2234}; Point{2303}; Point{2332}; Point{2304}; Point{2233}; Point{2331}; Point{2302}; Point{2330}; Point{2222}; Point{2159}; Point{2109}; 
    }
    //+
    Recursive Delete {
    Point{2329}; Point{2291}; Point{2328}; Point{2292}; Point{2224}; Point{2163}; Point{2162}; Point{2101}; Point{2108}; Point{2058}; Point{1981}; Point{2103}; Point{2102}; Point{2161}; Point{2223}; Point{2160}; 
    }
    //+
    Recursive Delete {
    Point{2225}; Point{2287}; Point{2294}; Point{2326}; Point{2286}; Point{2295}; 
    }
    //+
    Recursive Delete {
    Point{2325}; Point{2227}; Point{2297}; Point{2322}; Point{2296}; Point{2324}; Point{2298}; 
    }
    //+
    Recursive Delete {
    Point{2323}; Point{2300}; Point{2299}; 
    }
    //+
    Recursive Delete {
    Point{2232}; Point{2230}; Point{2229}; Point{2169}; Point{2228}; Point{2168}; Point{2220}; Point{2221}; Point{2288}; Point{2226}; Point{2321}; Point{2293}; Point{2327}; 
    }
    //+
    Recursive Delete {
    Point{1449}; Point{1450}; Point{1454}; Point{1455}; Point{1546}; Point{1545}; Point{1542}; Point{1541}; Point{1365}; Point{1362}; Point{1368}; Point{1369}; Point{1281}; Point{1280}; Point{1279}; Point{1277}; 
    }
    //+
    Recursive Delete {
    Point{2044}; Point{1971}; Point{1973}; Point{1972}; Point{1977}; Point{1979}; Point{1978}; Point{2056}; Point{2052}; Point{2054}; Point{2114}; Point{2059}; Point{2106}; Point{2045}; Point{2041}; Point{2046}; Point{2053}; Point{2051}; Point{2107}; Point{2158}; Point{2110}; Point{2157}; Point{2111}; Point{2164}; Point{2112}; Point{2165}; Point{2167}; 
    }
    //+
    Recursive Delete {
    Point{2115}; Point{2057}; 
    }
    //+
    Recursive Delete {
    Point{2043}; Point{1995}; Point{2016}; Point{2017}; Point{2010}; Point{2014}; Point{2015}; Point{2009}; Point{2012}; Point{2008}; Point{1959}; Point{1961}; Point{2013}; Point{1897}; Point{1898}; Point{1960}; Point{1964}; Point{1963}; Point{2011}; Point{2040}; Point{1970}; Point{1962}; Point{1913}; Point{1914}; Point{1899}; Point{1878}; Point{1879}; Point{1877}; Point{1848}; Point{1784}; Point{1783}; Point{1780}; Point{1782}; Point{1731}; Point{1732}; Point{1733}; Point{1611}; Point{1612}; Point{1609}; Point{1520}; Point{1519}; Point{1518}; Point{1426}; Point{1427}; Point{1428}; Point{1343}; Point{1344}; Point{1256}; 
    }
    //+
    Recursive Delete {
    Point{1610}; Point{1341}; Point{1254}; Point{1255}; Point{1253}; Point{1248}; Point{2021}; Point{1336}; Point{1342}; Point{1425}; Point{2084}; Point{1420}; Point{1512}; Point{2092}; Point{1516}; Point{2029}; Point{2022}; Point{2090}; Point{1422}; Point{2091}; Point{2030}; Point{2027}; Point{2020}; Point{2028}; Point{1952}; Point{1954}; Point{1339}; 
    }
    //+
    Recursive Delete {
    Point{1517}; 
    }
    //+
    Recursive Delete {
    Point{2105}; Point{2104}; Point{2083}; Point{2082}; Point{2079}; Point{2081}; Point{1607}; Point{1608}; Point{1515}; Point{1421}; Point{1340}; Point{1251}; Point{1911}; Point{1252}; Point{1728}; Point{1727}; Point{1759}; Point{1729}; Point{1720}; Point{1955}; Point{1801}; Point{1807}; Point{2143}; Point{1838}; Point{1845}; Point{1777}; Point{1844}; Point{1778}; Point{1779}; Point{1846}; Point{1847}; 
    }
    //+
    Recursive Delete {
    Point{1997}; Point{2093}; Point{1889}; Point{2141}; Point{1894}; 
    }
    //+
    Recursive Delete {
    Point{1944}; Point{1996}; Point{2004}; Point{1940}; Point{2005}; Point{1888}; Point{2001}; Point{1998}; Point{1942}; Point{2077}; 
    }
    //+
    Recursive Delete {
    Point{2142}; Point{2206}; Point{1895}; Point{1893}; Point{1958}; 
    }
    //+
    Recursive Delete {
    Surface{670}; 
    }
    //+
    Recursive Delete {
    Surface{672}; 
    }
    //+
    Recursive Delete {
    Point{2301}; 
    }
    //+
    Recursive Delete {
    Point{2155}; 
    }
    //+
    Recursive Delete {
    Surface{1084}; 
    }
    //+
    Recursive Delete {
    Curve{588}; 
    }
    //+
    Recursive Delete {
    Curve{1075}; 
    }
    //+
    Recursive Delete {
    Surface{1082}; 
    }
    //+
    Recursive Delete {
    Curve{589}; 
    }
    //+
    Recursive Delete {
    Curve{573}; 
    }
    //+
    Recursive Delete {
    Point{1849}; 
    }
    //+
    Recursive Delete {
    Point{1781}; 
    }
    //+
    Recursive Delete {
    Point{1726}; 
    }
    //+
    Recursive Delete {
    Point{1604}; 
    }
    //+
    Recursive Delete {
    Point{1730}; 
    }
    //+
    Recursive Delete {
    Point{2042}; 
    }
    //+
    Recursive Delete {
    Point{1896}; 
    }
    //+
    Recursive Delete {
    Surface{1086}; 
    }
    //+
    Recursive Delete {
    Surface{1088}; 
    }
    //+
    Recursive Delete {
    Curve{592}; 
    }
    //+
    Recursive Delete {
    Curve{574}; 
    }
    //+
    Recursive Delete {
    Curve{568}; 
    }
    //+
    Recursive Delete {
    Surface{706}; 
    }
    //+
    Recursive Delete {
    Surface{704}; 
    }
    //+
    Recursive Delete {
    Surface{708}; 
    }
    //+
    Recursive Delete {
    Surface{710}; 
    }
    //+
    Recursive Delete {
    Surface{668}; 
    }
    //+
    Recursive Delete {
    Surface{674}; 
    }
    //+
    Recursive Delete {
    Point{2217}; Point{2156}; Point{2150}; Point{2151}; Point{2149}; Point{2133}; Point{2343}; Point{2278}; Point{2276}; Point{2080}; Point{2207}; Point{2075}; Point{2131}; Point{2132}; Point{2007}; Point{2006}; Point{2000}; Point{2078}; Point{2002}; Point{2003}; Point{1999}; Point{2136}; Point{1946}; Point{1947}; Point{1941}; Point{2087}; Point{2086}; Point{2089}; Point{1943}; Point{1945}; Point{2023}; Point{2025}; Point{2019}; 
    }
    //+
    Recursive Delete {
    Point{2026}; Point{2018}; Point{1887}; 
    }
    //+
    Recursive Delete {
    Point{2034}; Point{1951}; 
    }
    //+
    Recursive Delete {
    Point{1953}; Point{1890}; 
    }
    //+
    Recursive Delete {
    Point{1948}; Point{1840}; 
    }
    //+
    Recursive Delete {
    Point{1839}; Point{1891}; Point{1892}; 
    }
    //+
    Recursive Delete {
    Point{1949}; Point{1841}; Point{1842}; Point{1802}; 
    }
    //+
    Recursive Delete {
    Point{1908}; 
    }
    //+
    Recursive Delete {
    Point{1957}; Point{1861}; Point{1859}; 
    }
    //+
    Recursive Delete {
    Point{1804}; 
    }
    //+
    Recursive Delete {
    Point{2137}; Point{2135}; 
    }
    //+
    Recursive Delete {
    Point{2076}; 
    }
    //+
    Recursive Delete {
    Point{1786}; Point{1785}; Point{1789}; Point{1747}; Point{1742}; Point{1743}; Point{1746}; 
    }
    //+
    Recursive Delete {
    Surface{1032}; Point{1605}; Point{1599}; Point{1600}; Point{1598}; Point{1601}; Point{1602}; Point{1603}; Point{1724}; Point{1725}; Point{1722}; Point{1723}; Point{1721}; Point{1513}; Point{1506}; Point{1507}; 
    }
    //+
    Recursive Delete {
    Point{1509}; Point{1508}; Point{1511}; Point{1510}; Point{1418}; Point{1335}; 
    }
    //+
    Recursive Delete {
    Point{1417}; Point{1414}; Point{1415}; Point{1423}; Point{1330}; Point{1331}; Point{1332}; Point{1245}; Point{1242}; Point{1243}; Point{1249}; Point{1338}; 
    }
    //+
    Curve Loop(1088) = {549, 556, -555, 548};
    //+
    Plane Surface(1073) = {1088};
    //+
    Physical Curve(1089) = {555, 548, 549, 556};
    //+
    Physical Curve(1090) = {517, 515, 631, 632, 633, 635, 638, 637, 564, 557, 560, 561, 558, 567, 629, 628, 630, 626, 624, 623, 513, 518};
    """

    geometry = "cl = " + str(h) + ";\n" + stub
    return __generate_grid_from_geo_string(geometry)

    
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