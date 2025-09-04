use anyhow::Result;
use async_trait::async_trait;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use indoc::indoc;
use qdrant_client::{
    Qdrant,
    qdrant::{
        CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Distance, FieldType,
        MultiVectorComparator, MultiVectorConfigBuilder, VectorParamsBuilder, VectorsConfigBuilder,
    },
};
use rig::{agent::Agent, completion::Prompt, extractor::Extractor};
use rig::{
    client::{
        builder::{BoxAgentBuilder, DynClientBuilder},
        completion::CompletionModelHandle,
    },
    extractor::ExtractorBuilder,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
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

const EXTRACTOR_PREAMBLE: &str = indoc! {"
    The provided text is a summary of a code snippet.
    Your task is to generate 3 synthetic queries that a developer
    might use to search for this specific entry within a larger codebase.
    Avoid relying on keyword search terms and favor concepts and meaning.
    Focus on the specific purpose of the snippet.
    Avoid general queries about design patterns, language idioms, error handling, etc.
    Each query should be unique, and together,
    they should span the semantic breadth of the summary.
    Ensure that each query has enough semantic context to distinguish it from
    general questions that can be answered without this specific snippet.
"};

const SUMMARY_PREAMBLE: &str = indoc! {r##"
    You are a helpful software engineer mentoring a new team-mate.
    Without preamble or introduction, summarize provided code snippets, in a few sentences.
    Focus on its purpose, but also explain what it does and how it works in general terms without
    referring to specific literal values.

    Be sure to mention key types and functions used, if applicable.
    But do not dwell on basic concepts or things that are obvious like what it means for something to be public.
    Keep your explanation in paragraph format, using complete sentences.
"##};

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

// Allows both static dispatch via generics or dynamic via boxing
#[async_trait]
pub trait DynAgent: Send + Sync {
    async fn prompt(&self, body: &str) -> Result<String>;
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

#[async_trait]
pub trait DynExtractor<T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync>: Send + Sync {
    async fn extract(&self, body: &str) -> Result<T>;
}

#[async_trait]
impl<M: rig::completion::CompletionModel, T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync>
    DynExtractor<T> for Extractor<M, T>
{
    async fn extract(&self, body: &str) -> Result<T> {
        Ok(Extractor::extract(self, body).await?)
    }
}

// The part that supports dynamic dispatch via unsized trait objects.
#[async_trait]
impl<T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync, A: DynExtractor<T> + ?Sized>
    DynExtractor<T> for Box<A>
{
    async fn extract(&self, body: &str) -> Result<T> {
        (**self).extract(body).await
    }
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

#[derive(Clone, Debug, Default)]
pub struct AgentFactory {
    provider: String,

    model: String,

    summary_preamble: Option<String>,

    synth_preamble: Option<String>,
}

// TODO: upgrade to rig >= 0.19 and impl VerifyClient
impl AgentFactory {
    pub fn new(config: &Config) -> Self {
        Self {
            provider: config.llm_provider.clone().unwrap(),
            model: config.llm_model.clone().unwrap(),
            summary_preamble: config.persona.clone(),
            // TODO: config extractor PREAMBLE
            ..Default::default()
        }
    }

    pub fn model(&self) -> anyhow::Result<CompletionModelHandle<'static>> {
        let client = DynClientBuilder::new();
        let model = client.completion(&self.provider, &self.model)?;

        Ok(CompletionModelHandle {
            inner: Arc::from(model),
        })
    }

    pub fn agent(&self) -> anyhow::Result<BoxAgentBuilder<'static>> {
        Ok(DynClientBuilder::new().agent(&self.provider, &self.model)?)
    }

    pub fn summarizer(&self) -> anyhow::Result<BoxAgentBuilder<'static>> {
        Ok(self
            .agent()?
            .preamble(self.summary_preamble.as_deref().unwrap_or(SUMMARY_PREAMBLE)))
    }

    pub fn extractor<T>(&self) -> Result<ExtractorBuilder<T, CompletionModelHandle<'static>>>
    where
        T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync + 'static,
    {
        let builder = ExtractorBuilder::new(self.model()?)
            .preamble(self.synth_preamble.as_deref().unwrap_or(EXTRACTOR_PREAMBLE));

        Ok(builder)
    }
}

pub async fn init_collection(client: &Qdrant, collection: &str, dims: u64) -> Result<()> {
    if !client.collection_exists(collection).await? {
        // let vectors_config = VectorParamsBuilder::new(dims, Distance::Cosine);
        let mut vectors_config = VectorsConfigBuilder::default();
        vectors_config.add_named_vector_params(
            "default",
            VectorParamsBuilder::new(dims, Distance::Cosine).build(),
        );
        vectors_config.add_named_vector_params(
            "aliases",
            VectorParamsBuilder::new(dims, Distance::Cosine)
                .multivector_config(MultiVectorConfigBuilder::new(MultiVectorComparator::MaxSim))
                .build(),
        );

        client
            .create_collection(
                CreateCollectionBuilder::new(collection).vectors_config(vectors_config),
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
