pub mod parse;
pub mod traverse;
pub mod workers;

use std::sync::{Arc, LazyLock};

use clap::Parser;
use fastembed::{EmbeddingModel, ModelInfo, TextEmbedding};
use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
use indicatif::{MultiProgress, ProgressBar};
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;
pub use traverse::*;

use rig::Embed;

#[skip_serializing_none]
#[derive(Embed, serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct CodeSnippet {
    pub path: String,
    pub class: Option<String>,
    pub name: String,
    pub body: String,

    #[embed]
    pub summary: String,

    #[serde(skip_serializing)]
    pub hash: Vec<u8>,
}

pub struct Progressor {
    pub multi: MultiProgress,
    pub file_progress: ProgressBar,
}

impl Default for Progressor {
    fn default() -> Self {
        let multi = MultiProgress::new();
        let file_progress = multi.add(ProgressBar::no_length());

        Self {
            multi,
            file_progress,
        }
    }
}

pub enum SnippetProgress {
    Snippet {
        progress: Option<ProgressBar>,
        snippet: CodeSnippet,
    },
    EndOfFile {
        progressor: Arc<Option<Progressor>>,
        progress: Option<ProgressBar>,
    },
}

// TODO: Split into worker specific configs
// TODO: project description
#[skip_serializing_none] // This is the solution!
#[derive(Clone, Parser, Debug, Serialize, Deserialize)]
// #[command(version, about, long_about = None)]
pub struct Config {
    #[arg(long, action=clap::ArgAction::SetTrue)]
    pub dry_run: Option<bool>,

    #[arg(long, action=clap::ArgAction::SetTrue)]
    pub progress: Option<bool>,

    #[arg(long, action=clap::ArgAction::SetTrue)]
    pub dump_config: Option<bool>,

    // #[arg(long, default_value = "http://10.10.10.100:11434")] // No, masks config file
    /// Base URL of the language model server
    #[arg(long)]
    pub llm_base_url: Option<String>,

    /// Name of the language model for summarizing code snippets
    #[arg(long)]
    pub llm_model: Option<String>,

    #[arg(long)]
    pub fastembed_cache: Option<String>,

    #[arg(long)]
    pub collection: Option<String>,

    #[arg(long)]
    pub embed_model: Option<String>,

    pub target_path: Option<String>,
}

#[skip_serializing_none] // This is the solution!
#[derive(Clone, Parser, Debug, Serialize, Deserialize)]
pub struct CommonConfig {}

#[skip_serializing_none] // This is the solution!
#[derive(Clone, Parser, Debug, Serialize, Deserialize)]
pub struct SummarizerConfig {
    // #[arg(long, default_value = "http://10.10.10.100:11434")] // No, masks config file
    /// Base URL of the language model server
    #[arg(long)]
    pub llm_base_url: Option<String>,

    /// Name of the language model for summarizing code snippets
    #[arg(long)]
    pub llm_model: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dry_run: Default::default(),
            progress: Default::default(),
            dump_config: Default::default(),
            llm_model: Some("devstral:latest".into()),
            llm_base_url: Some("http://localhost:11434".into()),
            collection: Some("myproject".into()),
            embed_model: Default::default(),
            fastembed_cache: dirs::cache_dir().and_then(|mut d| {
                d.push("fastembed");
                d.into_os_string().into_string().ok()
            }),
            target_path: Some("./".into()),
        }
    }
}

// TODO: move these statics back to the bin and pass down to the workers by argument
pub static CONFIG: LazyLock<Config> = LazyLock::new(|| {
    Figment::new()
        .merge(Serialized::defaults(Config::default()))
        .merge(Toml::file(
            dirs::config_dir()
                .map(|p| p.join("emberlain"))
                .unwrap_or_default()
                .join("config.toml"),
        ))
        .merge(Env::prefixed("EMB_"))
        .merge(Serialized::defaults(Config::parse()))
        // TODO: select profile from env
        .extract()
        .unwrap()
});

pub static LLM_MODEL: LazyLock<String> = LazyLock::new(|| CONFIG.llm_model.clone().unwrap());
pub static LLM_BASE_URL: LazyLock<String> = LazyLock::new(|| CONFIG.llm_base_url.clone().unwrap());
pub static EMBED_INFO: LazyLock<ModelInfo<EmbeddingModel>> = LazyLock::new(|| {
    let model_name = CONFIG.embed_model.as_ref().unwrap().to_lowercase();
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
        .unwrap_or_else(|| {
            panic!(
                "The embedding model '{}' is not valid",
                CONFIG.embed_model.as_ref().unwrap()
            )
        })
});

pub static EMBED_MODEL: LazyLock<EmbeddingModel> = LazyLock::new(|| EMBED_INFO.model.clone());
pub static EMBED_DIMS: LazyLock<usize> = LazyLock::new(|| EMBED_INFO.dim);
pub static COLLECTION_NAME: LazyLock<String> = LazyLock::new(|| CONFIG.collection.clone().unwrap());

#[cfg(test)]
#[path = "../tests/utils/mod.rs"]
mod test_utils;
