use std::path::PathBuf;

use anyhow::{Context as _, Result};
use chrono::{DateTime, Utc};
use clap::Parser;
use figment::{
    Figment,
    providers::{Env, Format as _, Serialized, Toml},
};
use humantime::parse_duration;
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;

/// Crawls a source repository, generating summaries to insert into a semantic search database.
#[skip_serializing_none] // This is the solution!
#[derive(Clone, Parser, Debug, Serialize, Deserialize)]
// #[command(version, about, long_about = None)]
pub struct Config {
    /// Just crawl the repository but don't summarize or insert vectors
    #[arg(long, action=clap::ArgAction::SetTrue)]
    pub dry_run: Option<bool>,

    /// Display progress bars
    #[arg(long, action=clap::ArgAction::SetTrue)]
    pub progress: Option<bool>,

    /// Print the effective configuration and exit
    #[arg(long, action=clap::ArgAction::SetTrue)]
    pub dump_config: Option<bool>,

    /// Re-summarize and embed previously processed snippets
    #[arg(long, action=clap::ArgAction::SetTrue)]
    pub reprocess: Option<bool>,

    /// Remove stale entries after walking the repo
    ///
    /// Can be "all" or a duration like "30days", "1month", etc.
    /// using https://docs.rs/humantime/latest/humantime/fn.parse_duration.html.
    ///
    /// The cutoff is based on when an entry was first absent during a crawl,
    /// rather than when it was added or actually removed from the repositoy.
    #[arg(long)]
    pub prune: Option<String>,

    /// Number of concurrent summarization tasks. Set to the number of LLM instances available.
    #[arg(long)]
    pub summary_workers: Option<u32>,

    /// Instructions given to the summary agent describing its persona.
    ///
    /// Using a heredoc or multiline config string is recommended over a program argument.
    #[arg(long)]
    pub persona: Option<String>,

    /// Base URL of the language model server
    #[arg(long)]
    pub llm_base_url: Option<String>,

    /// Name of the language model for summarizing code snippets
    #[arg(long)]
    pub llm_model: Option<String>,

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

    /// Path to the language specification YAML file
    #[arg(long)]
    pub lang_spec: Option<PathBuf>,

    /// Override the base directory used to calculate relative paths within the target path.
    ///
    /// Otherwise, this is discovered automatically by traversing the file system.
    #[arg(long)]
    pub repo_root: Option<PathBuf>,

    /// Path of the repository to index
    pub target_path: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dry_run: Default::default(),
            progress: Default::default(),
            dump_config: Default::default(),
            reprocess: Default::default(),
            prune: Default::default(),
            summary_workers: Some(1),
            persona: None,
            llm_model: Some("devstral:latest".into()),
            llm_base_url: Some("http://localhost:11434".into()),
            collection: Some("myproject".into()),
            qdrant_url: Some("http://localhost:6334".into()),
            embed_model: Default::default(),
            fastembed_cache: dirs::cache_dir().map(|d| d.join("fastembed")),
            lang_spec: Default::default(),
            repo_root: None,
            target_path: Some("./".into()),
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

    pub fn pruning_cutoff(&self) -> Result<Option<DateTime<Utc>>> {
        if let Some(dur) = self.prune.as_ref() {
            if dur == "all" || dur == "now" {
                Ok(Some(DateTime::<Utc>::MAX_UTC))
            } else {
                let delta = parse_duration(dur).context("Invalid duration. See https://docs.rs/humantime/latest/humantime/fn.parse_duration.html")?;
                let delta = chrono::Duration::from_std(delta)?;
                Ok(Utc::now().checked_sub_signed(delta))
            }
        } else {
            Ok(None)
        }
    }
}
