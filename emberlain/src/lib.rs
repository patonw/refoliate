use anyhow::{Context, Result};
use async_trait::async_trait;
use cached::proc_macro::cached;
use chrono::{DateTime, Utc};
use clap::Parser;
use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
use humantime::parse_duration;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use indoc::indoc;
use qdrant_client::{
    Qdrant,
    qdrant::{
        CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Distance, FieldType,
        VectorParamsBuilder,
    },
};
use rig::{Embed, agent::Agent, completion::Prompt, providers::ollama};
use rig::{client::CompletionClient, providers::ollama::CompletionModel};
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, skip_serializing_none};
use std::{borrow::Cow, path::PathBuf, sync::Arc, time::Duration};
use uuid::Uuid;

pub mod parse;
pub mod traverse;
pub mod workers;

pub use traverse::*;

#[cached]
fn make_id_hash(
    path: String,
    interface: Option<String>,
    class: Option<String>,
    attributes: Vec<String>,
    name: String,
) -> Vec<u8> {
    let data = format!(
        "{path}\n{}\n{}\n{attributes:?}\n{name}",
        interface.as_deref().unwrap_or_default(),
        class.as_deref().unwrap_or_default()
    );

    blake3::hash(data.as_bytes()).as_bytes().to_vec()
}

#[serde_as]
#[skip_serializing_none]
#[derive(Embed, serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct CodeSnippet {
    /// If the file is in a repository, this is relative to the repo root.
    /// Otherwise, relative to the target path argument.
    pub path: String,

    /// Name of the interface/trait/etc if this is a member function
    pub interface: Option<String>,

    /// Name of the class/struct/etc if this is a member function
    pub class: Option<String>,

    pub attributes: Vec<String>,

    /// The name of this function/method/type/etc
    pub name: String,

    /// The contents of the snippet
    pub body: String,

    /// An LLM generated summary
    #[embed]
    pub summary: String,

    // #[serde(skip_serializing)]
    #[serde_as(as = "serde_with::hex::Hex")]
    pub hash: Vec<u8>,
}

impl CodeSnippet {
    pub fn uuid(&self) -> Result<Uuid> {
        let CodeSnippet {
            path,
            interface,
            class,
            attributes,
            name,
            ..
        } = self;

        let hash = make_id_hash(
            path.clone(),
            interface.clone(),
            class.clone(),
            attributes.clone(),
            name.clone(),
        );

        Ok(Uuid::new_v8(hash[..16].try_into()?))
    }

    pub fn body(&self) -> Cow<String> {
        // TODO: universal commenting or wrap in markdown
        match (&self.interface, &self.class) {
            (Some(trait_type), Some(class_type)) => Cow::Owned(format!(
                "/// self: {trait_type} for {class_type}\n{}",
                &self.body
            )),
            (None, Some(class_type)) => {
                Cow::Owned(format!("/// self: {class_type}\n{}", &self.body))
            }
            _ => Cow::Borrowed(&self.body),
        }
    }
}

pub struct Progressor {
    pub multi: MultiProgress,
    pub file_progress: ProgressBar,
}

impl Default for Progressor {
    fn default() -> Self {
        let multi = MultiProgress::new();
        let file_progress = multi.add(ProgressBar::no_length());

        // Something funny going on with the duration calculation:
        // Fluctuates between seconds and days, even at half way point.
        // It's based off of steps per second instead of elapsed time and percent complete
        file_progress.set_style(
            ProgressStyle::with_template("[{elapsed_precise}] {wide_bar} {pos}/{len} ({percent}%)")
                .unwrap(),
        );
        file_progress.enable_steady_tick(Duration::from_secs(5));

        Self {
            multi,
            file_progress,
        }
    }
}

pub enum SnippetProgress {
    StartOfFile {
        file_path: PathBuf,
        progressor: Arc<Option<Progressor>>,
        progress: Option<ProgressBar>,
    },
    Snippet {
        progress: Option<ProgressBar>,
        snippet: CodeSnippet,
    },
    EndOfFile {
        progressor: Arc<Option<Progressor>>,
        progress: Option<ProgressBar>,
    },
}

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

pub fn get_ollama_agent(config: &Config) -> Agent<CompletionModel> {
    let preamble = config.persona.as_deref().unwrap_or(indoc! {r##"
            You are a helpful software engineer mentoring a new team-mate.
            Without preamble or introduction, summarize provided code snippets, in a few sentences.
            Explain what it does and how it works in general terms without referring to specific values.
            If it uses higher level concepts and design patterns like observers, pipelines, etc. make note of that.
            Be sure to mention key types and functions used, if applicable.
            But do not dwell on basic concepts or things that are obvious like what it means for something to be public.
            Keep your explanation in paragraph format, using complete sentences.
        ""##});
    ollama::Client::from_url(config.llm_base_url.as_ref().unwrap())
        .agent(config.llm_model.as_ref().unwrap())
        .max_tokens(1024)
        // TODO: Really need to work on taming the LLM's verbiage
        .preamble(preamble)
        .build()
}

// Allows both static dispatch via generics or dynamic via boxing
#[async_trait]
pub trait DynAgent: Send + Sync {
    async fn prompt(&self, body: &str) -> Result<String>;
}

#[async_trait(?Send)]
pub trait DynChecker: Send + Sync {
    async fn call(&self) -> Result<()>;
}

#[async_trait(?Send)]
impl<T: AsyncFn() -> Result<()> + Send + Sync> DynChecker for T {
    async fn call(&self) -> Result<()> {
        (self)().await
    }
}

#[async_trait]
impl<M: rig::completion::CompletionModel> DynAgent for Agent<M> {
    async fn prompt(&self, body: &str) -> Result<String> {
        Ok(Prompt::prompt(self, body).await?)
    }
}

// The part that supports dynamic dispatch via unsized trait objects.
#[async_trait]
impl<T: DynAgent + ?Sized> DynAgent for Box<T> {
    async fn prompt(&self, body: &str) -> Result<String> {
        (**self).prompt(body).await
    }
}

// // Probably unnecessary given the flexibility of boxed closure style
//
// #[async_trait(?Send)]
// pub trait AgentFactory {
//     fn build(&self) -> Box<dyn DynAgent>;
//
//     async fn check(&self) -> Result<()>;
// }
//
// #[async_trait(?Send)]
// impl AgentFactory for Box<dyn AgentFactory> {
//     fn build(&self) -> Box<dyn DynAgent> {
//         (**self).build()
//     }
//
//     async fn check(&self) -> Result<()> {
//         (**self).check().await
//     }
// }
//
// #[async_trait(?Send)]
// impl AgentFactory for BoxAgentFactory {
//     fn build(&self) -> Box<dyn DynAgent> {
//         (self.builder)()
//     }
//
//     async fn check(&self) -> Result<()> {
//         self.checker.call().await
//     }
// }

pub struct AgentFactory {
    /// Creates an agent instance based on config info supplied during factory initialization.
    builder: Box<dyn Fn() -> Box<dyn DynAgent>>,

    /// Checks that the provider information supplied when creating the factory is valid.
    /// i.e. may check API urls, tokens, model names, etc using remote calls.
    checker: Box<dyn DynChecker>,
}

impl AgentFactory {
    pub fn new(builder: Box<dyn Fn() -> Box<dyn DynAgent>>, checker: Box<dyn DynChecker>) -> Self {
        Self { builder, checker }
    }

    pub fn build(&self) -> Box<dyn DynAgent> {
        (self.builder)()
    }

    pub async fn check(&self) -> Result<()> {
        self.checker.call().await
    }
}

pub fn get_agent_factory(config: &Config) -> Result<AgentFactory> {
    let checker = {
        let base_url = config.llm_base_url.clone().unwrap();
        async move || {
            reqwest::get(&base_url)
                .await
                .context("Unable to connect to LLM provider")?
                .text()
                .await?;
            Ok(())
        }
    };

    let config = config.clone();
    let builder = move || Box::new(get_ollama_agent(&config)) as Box<dyn DynAgent>;

    Ok(AgentFactory::new(Box::new(builder), Box::new(checker)))
}

pub async fn init_collection(client: &Qdrant, collection: &str, dims: u64) -> Result<()> {
    if !client.collection_exists(collection).await? {
        client
            .create_collection(
                CreateCollectionBuilder::new(collection)
                    .vectors_config(VectorParamsBuilder::new(dims, Distance::Cosine)),
            )
            .await?;
    }

    for field in ["path", "name", "hash", "attributes"] {
        // Hoping this works on an empty collection and doesn't blow up if an index already exists
        client
            .create_field_index(CreateFieldIndexCollectionBuilder::new(
                collection,
                field,
                FieldType::Keyword,
            ))
            .await?;
    }

    client
        .create_field_index(CreateFieldIndexCollectionBuilder::new(
            collection,
            "__removed",
            FieldType::Datetime,
        ))
        .await?;

    Ok(())
}

#[cfg(test)]
#[path = "../tests/utils/mod.rs"]
mod test_utils;
