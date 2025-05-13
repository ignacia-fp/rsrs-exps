use cc;

fn main() {
    // Paths for Python and Numpy
    let python_include = "/opt/homebrew/Cellar/python@3.13/3.13.3/Frameworks/Python.framework/Versions/3.13/include/python3.13";
    let python_lib =
        "/opt/homebrew/Cellar/python@3.13/3.13.3/Frameworks/Python.framework/Versions/3.13/lib";
    let numpy_include = "/opt/homebrew/lib/python3.13/site-packages/numpy/_core/include";

    // Print necessary configuration for `cargo` to detect
    println!("cargo:rerun-if-changed=kernel_interface.c");
    println!("cargo:rerun-if-env-changed=PYTHON_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=PYTHON_LIB_DIR");

    // Add include paths for both Python and Numpy
    println!("cargo:include={}", python_include);
    println!("cargo:include={}", numpy_include);
    println!("cargo:lib={}", python_lib);

    // Link to Python
    println!("cargo:rustc-link-search=native=/opt/homebrew/Cellar/python@3.13/3.13.3/Frameworks/Python.framework/Versions/3.13/lib");
    println!("cargo:rustc-link-lib=dylib=python3.13");

    // Compile the C code
    cc::Build::new()
        .file("kernel_interface.c")
        .include(python_include)
        .include(numpy_include) // Include Numpy headers
        .compile("libkernel_interface.a");
}
