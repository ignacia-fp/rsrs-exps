use bempp_rsrs::rsrs::rsrs_cycle::{RsrsArgs, RsrsOptions};
use regex::Regex;
use rlst::prelude::*;
use rsrs_exps::io::structured_operators_types::StructuredOperatorType;
use rsrs_exps::test_prep::Precision;
use rsrs_exps::test_prep::Results;
use rsrs_exps::test_prep::TestFramework;
use rsrs_exps::test_prep::TestFrameworkImpl;
use rsrs_exps::test_prep::{DataType, OutputOptions, ScenarioArgs, ScenarioOptions, Solve};
use std::collections::HashMap;
use std::env;
use std::fs;


fn extract_operator_types(python_code: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();

    // Regex to match class declarations inheriting from BaseStructuredOperator
    let class_regex = Regex::new(r"class (\w+)\(BaseStructuredOperator\):").unwrap();

    // Regex to match assignment of self.operator_type
    let operator_type_regex = Regex::new(r#"self\.operator_type\s*=\s*['"](\w+)['"]"#).unwrap();

    let mut current_class = None;

    for line in python_code.lines() {
        if let Some(cap) = class_regex.captures(line) {
            current_class = Some(cap[1].to_string());
        } else if let Some(cap) = operator_type_regex.captures(line) {
            if let Some(class_name) = &current_class {
                map.insert(class_name.clone(), cap[1].to_string());
                current_class = None; // Reset for next class
            }
        }
    }

    map
}

#[derive(Debug, Clone, Copy)]
enum ScalarType {
    F32,
    F64,
    C32,
    C64,
}

fn determine_type_behavior(
    operator_type: &StructuredOperatorType,
    precision: &Precision,
    type_map: &HashMap<String, String>,
) -> Result<ScalarType, String> {
    let key = operator_type.as_ref().to_string();

    match type_map.get(&key).map(|s| s.as_str()) {
        Some("real") => match precision {
            Precision::Single => Ok(ScalarType::F32),
            Precision::Double => Ok(ScalarType::F64),
        },
        Some("complex") => match precision {
            Precision::Single => Ok(ScalarType::C32),
            Precision::Double => Ok(ScalarType::C64),
        },
        Some(other) => Err(format!("Unknown numeric nature: {other}")),
        None => Err(format!("Unknown operator type: {key}")),
    }
}

/*#[derive(Debug)]
enum TypedScenarioArgs {
    Real(ScenarioArgs<f64>),
    Complex(ScenarioArgs<c64>),
}

#[derive(Debug)]
enum TypedRsrsArgs {
    Real(RsrsArgs<f64>),
    Complex(RsrsArgs<c64>),
}*/


fn build_and_run_test() {
    let args: Vec<String> = env::args().collect();

    let python_source = fs::read_to_string("python/structured_operators.py").unwrap();
    let type_map = extract_operator_types(&python_source);

    if args.len() > 1 {
        let data_type: DataType =
            serde_json::from_str(&args[1]).expect("Failed to deserialize data type args");

        let dtype = determine_type_behavior(
            &data_type.structured_operator_type,
            &data_type.precision,
            &type_map,
        )
        .unwrap();

        match dtype {
            ScalarType::F32 | ScalarType::C32 => panic!("Single precision not implemented"),

            ScalarType::F64 => {
                let scenario_args = serde_json::from_str::<ScenarioArgs<f64>>(&args[2])
                    .expect("Failed to deserialize scenario args");
                let rsrs_args = serde_json::from_str::<RsrsArgs<f64>>(&args[3])
                    .expect("Failed to deserialize rsrs args");
                let output_options =
                    serde_json::from_str(&args[4]).expect("Failed to deserialize output args");
                let scenario_options = ScenarioOptions::new(Some(scenario_args), data_type);
                let rsrs_options = RsrsOptions::<f64>::new(Some(rsrs_args));
                let mut test_framework = TestFramework::new(scenario_options, rsrs_options, output_options);
                test_framework.run_tests();
            }

            ScalarType::C64 => {
                let scenario_args = serde_json::from_str::<ScenarioArgs<c64>>(&args[2])
                    .expect("Failed to deserialize scenario args");
                let rsrs_args = serde_json::from_str::<RsrsArgs<c64>>(&args[3])
                    .expect("Failed to deserialize rsrs args");
                let output_options =
                    serde_json::from_str(&args[4]).expect("Failed to deserialize output args");

                let scenario_options = ScenarioOptions::new(Some(scenario_args), data_type);
                let rsrs_options = RsrsOptions::<c64>::new(Some(rsrs_args));
                let mut test_framework = TestFramework::new(scenario_options, rsrs_options, output_options);
                test_framework.run_tests();
            }
        }
    } else {
        let structured_operator_type = StructuredOperatorType::KiFMMLaplaceOperator;
        let precision = Precision::Double;

        let data_type = DataType {
            structured_operator_type: structured_operator_type.clone(),
            precision: precision.clone(),
        };

        let dtype = determine_type_behavior(&structured_operator_type, &precision, &type_map)
            .expect("Could not determine ScalarType");

        match dtype {
            ScalarType::F32 | ScalarType::C32 => panic!("Single precision not implemented"),

            ScalarType::F64 => {
                let scenario_options =
                    ScenarioOptions::new(None::<ScenarioArgs<f64>>, data_type.clone());
                let rsrs_options = RsrsOptions::<f64>::new(None::<RsrsArgs<f64>>);
                let output_options = OutputOptions::new(Solve::False, false, false, Results::All);
                let mut test_framework = TestFramework::new(scenario_options, rsrs_options, output_options);
                test_framework.run_tests();
            }

            ScalarType::C64 => {
                let scenario_options =
                    ScenarioOptions::new(None::<ScenarioArgs<c64>>, data_type.clone());
                let rsrs_options = RsrsOptions::<c64>::new(None::<RsrsArgs<c64>>);
                let output_options = OutputOptions::new(Solve::False, false, false, Results::All);
                let mut test_framework = TestFramework::new(scenario_options, rsrs_options, output_options);
                test_framework.run_tests();
            }
        }
    };

    
}

fn main() {
    build_and_run_test();
}
