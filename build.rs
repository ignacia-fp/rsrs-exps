use cc;
use std::process::Command;
use std::fs::File;
use std::io::Write;


/// Generates a Rust enum from Python class names in a given file.
fn generate_operator_enum(python_file: &str, output_file: &str) {
    let output = Command::new("python3")
        .arg("python/class_names.py")
        .arg(python_file)
        .output()
        .expect("Failed to run Python class name extractor");

    if !output.status.success() {
        panic!("Python error:\n{}", String::from_utf8_lossy(&output.stderr));
    }

    let class_names = String::from_utf8_lossy(&output.stdout);
    let mut file = File::create(output_file)
        .unwrap_or_else(|_| panic!("Failed to create {}", output_file));

    writeln!(file, "#[derive(Clone, strum_macros::AsRefStr, strum_macros::EnumString, Debug)]
    pub enum StructuredOperatorType {{").unwrap();
    for name in class_names.lines() {
        writeln!(file, "    {},", name).unwrap();
    }
    writeln!(file, "}}").unwrap();

    println!("cargo:rerun-if-changed={}", python_file);
    println!("cargo:rerun-if-changed=scripts/class_names.py");
}


fn main() {
    // Ensure build.rs reruns if relevant files or env variables change
    println!("cargo:rerun-if-changed=c_src/structured_operator_interface.c");
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

    generate_operator_enum("python/structured_operators.py", "src/io/structured_operators_types.rs");

    // Compile C code with the correct includes
    cc::Build::new()
        .file("c_src/structured_operator_interface.c")
        .include(&python_include)
        .include(&numpy_include)
        .compile("libstructured_operator_interface.a");
}
