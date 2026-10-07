use bzip2::read::BzDecoder;
use sha2::{Digest, Sha256};
use std::{error::Error, path::Path, time::Duration};
use tar::Archive;

const MODEL_WEIGHTS_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2";
const MODELS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models");
const MODEL_WEIGHTS_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
);

const VAD_URL: &str = "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx";
const VAD_NAME: &str = "silero-vad-v6.2.3.onnx";
const VAD_SHA256: &str = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3";

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
    ensure_vad()?;
    Ok(())
}

/// Pinned, checksum-verified weights; never publish a partial download as the model.
fn ensure_vad() -> Result<(), Box<dyn Error>> {
    let path = Path::new(MODELS_DIR).join(VAD_NAME);
    let bytes = if path.exists() {
        std::fs::read(&path)?
    } else {
        println!("cargo:warning=Downloading Silero VAD v6.2.3 (2.3 MB)");
        reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(120))
            .build()?
            .get(VAD_URL)
            .send()?
            .error_for_status()?
            .bytes()?
            .to_vec()
    };
    let hash = format!("{:x}", Sha256::digest(&bytes));
    if hash != VAD_SHA256 {
        return Err(format!(
            "Silero VAD checksum mismatch at {}; remove the file and rebuild",
            path.display()
        )
        .into());
    }
    if !path.exists() {
        std::fs::create_dir_all(MODELS_DIR)?;
        let temporary = path.with_extension("onnx.part");
        std::fs::write(&temporary, &bytes)?;
        std::fs::rename(temporary, path)?;
    }
    Ok(())
}
