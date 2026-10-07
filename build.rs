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
    ensure_cues()?;
    Ok(())
}

fn ensure_cues() -> Result<(), Box<dyn Error>> {
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").ok_or("missing OUT_DIR")?);
    for (source, target, expected) in [
        (
            "click4",
            "cue-start.wav",
            "0d80e2c82426316b140b0686e10f83924ef794e9a9dfe13aaaa794b18200b048",
        ),
        (
            "click3",
            "cue-done.wav",
            "8d0676a5bcbfedad3e65b7b73e93a044216d7f192c99a3a05caf21e2aa4a8dda",
        ),
    ] {
        let path = out.join(target);
        let bytes = if path.exists() {
            std::fs::read(&path)?
        } else {
            let url = format!(
                "https://raw.githubusercontent.com/Calinou/kenney-ui-audio/8c3d81b9159d058c444f89d12d518276b0b09345/addons/kenney_ui_audio/{source}.wav"
            );
            reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()?
                .get(url)
                .send()?
                .error_for_status()?
                .bytes()?
                .to_vec()
        };
        if format!("{:x}", Sha256::digest(&bytes)) != expected {
            return Err("cue audio checksum mismatch".into());
        }
        if !path.exists() {
            std::fs::write(path, bytes)?;
        }
    }
    Ok(())
}
