use anyhow::{Context, Result};
use async_trait::async_trait;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use indoc::indoc;
use qdrant_client::{
    Qdrant,
    qdrant::{
        CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Distance, FieldType,
        VectorParamsBuilder,
    },
};
use rig::{agent::Agent, completion::Prompt, providers::ollama};
use rig::{client::CompletionClient, providers::ollama::CompletionModel};
use std::{path::PathBuf, sync::Arc, time::Duration};

pub mod config;
pub mod parse;
pub mod snippet;
pub mod template;
pub mod traverse;
pub mod workers;

pub use config::*;
pub use snippet::*;
pub use traverse::*;

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
