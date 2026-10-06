use bzip2::read::BzDecoder;
use std::{error::Error, path::Path, time::Duration};
use tar::Archive;

const MODEL_WEIGHTS_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2";
const MODELS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models");
const MODEL_WEIGHTS_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
);

fn main() -> Result<(), Box<dyn Error>> {
    if !Path::new(MODEL_WEIGHTS_PATH).exists() {
        println!("Downloading weights...");
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(30 * 60))
            .build()?;
        let response = client.get(MODEL_WEIGHTS_URL).send()?.error_for_status()?;
        let bz_decoder = BzDecoder::new(response);
        let mut tar = Archive::new(bz_decoder);

        tar.unpack(MODELS_DIR)?;
    }
    Ok(())
}
