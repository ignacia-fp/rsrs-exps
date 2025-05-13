use std::process::Command;
use std::env;
use cc;

fn main() {
    // Ensure build.rs reruns if relevant files or env variables change
    println!("cargo:rerun-if-changed=kernel_interface.c");
    println!("cargo:rerun-if-env-changed=PYTHON_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=PYTHON_LIB_DIR");

    // Get the Python include path
    let python_include = Command::new("python")
        .arg("-c")
        .arg("import sysconfig; print(sysconfig.get_path('include'))")
        .output()
        .expect("Failed to get Python include path")
        .stdout;
    let python_include = String::from_utf8_lossy(&python_include).trim().to_string();

    // Get the Python library directory (used for linking)
    let python_libdir = Command::new("python")
        .arg("-c")
        .arg("import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
        .output()
        .expect("Failed to get Python library path")
        .stdout;
    let python_libdir = String::from_utf8_lossy(&python_libdir).trim().to_string();

    // Get the Python library name (e.g. python3.11)
    let python_libname = Command::new("python")
        .arg("-c")
        .arg("import sysconfig; print('python' + sysconfig.get_config_var('LDVERSION'))")
        .output()
        .expect("Failed to get Python lib name")
        .stdout;
    let python_libname = String::from_utf8_lossy(&python_libname).trim().to_string();

    // Get the NumPy include path
    let numpy_include = Command::new("python")
        .arg("-c")
        .arg("import numpy; print(numpy.get_include())")
        .output()
        .expect("Failed to get NumPy include path")
        .stdout;
    let numpy_include = String::from_utf8_lossy(&numpy_include).trim().to_string();

    // Pass to the linker
    println!("cargo:rustc-link-search=native={}", python_libdir);
    println!("cargo:rustc-link-lib=dylib={}", python_libname);

    // Compile C code with the correct includes
    cc::Build::new()
        .file("kernel_interface.c")
        .include(&python_include)
        .include(&numpy_include)
        .compile("libkernel_interface.a");
}
