use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use fastembed::{EmbeddingModel, ModelInfo, TextEmbedding};
use figment::{
    Figment,
    providers::{Env, Format as _, Serialized, Toml},
};
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;

/// Crawls a source repository, generating summaries to insert into a semantic search database.
#[skip_serializing_none] // This is the solution!
#[derive(Clone, Parser, Debug, Serialize, Deserialize)]
// #[command(version, about, long_about = None)]
pub struct Config {
    /// Print the effective configuration and exit
    #[arg(long, action=clap::ArgAction::SetTrue)]
    pub dump_config: Option<bool>,

    /// Path to local cache for storing embedding models
    #[arg(long)]
    pub fastembed_cache: Option<PathBuf>,

    /// The embedding model identifier (e.g. Xenova/all-MiniLM-L6-v2) or enum code (e.g. AllMiniLML6V2Q)
    #[arg(long)]
    pub embed_model: Option<String>,

    /// URL to the qdrant server instance
    #[arg(long)]
    pub qdrant_url: Option<String>,

    /// Name of collection in qdrant
    #[arg(long)]
    pub collection: Option<String>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            dump_config: Default::default(),
            collection: Some("myproject".into()),
            qdrant_url: Some("http://localhost:6334".into()),
            embed_model: Default::default(),
            fastembed_cache: dirs::cache_dir().map(|d| d.join("fastembed")),
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        Ok(Figment::new()
            .merge(Serialized::defaults(Config::default()))
            .merge(Toml::file(
                dirs::config_dir()
                    .map(|p| p.join("emberlain"))
                    .unwrap_or_default()
                    .join("config.toml"),
            ))
            .merge(Env::prefixed("EMB_"))
            .merge(Serialized::defaults(Config::parse()))
            .select(std::env::var("EMB_PROFILE").unwrap_or_default())
            .extract()?)
    }
}

// TODO: common lib
pub fn get_embed_info(config: &Config) -> Option<ModelInfo<EmbeddingModel>> {
    let model_name = config.embed_model.as_ref()?.to_lowercase();
    let all_embeddings = TextEmbedding::list_supported_models();
    all_embeddings
        .iter()
        .find(|model| {
            model.model_code.to_lowercase().ends_with(&model_name)
                || format!("{:?}", model.model)
                    .to_lowercase()
                    .ends_with(&model_name)
        })
        .cloned()
}
