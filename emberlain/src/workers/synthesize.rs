use flume::{Receiver, Sender};
use log::info;
use rig::{client::CompletionClient as _, providers};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use crate::CodeSnippet;
use crate::DynAgent;
use crate::SnippetProgress;

const LLM_MODEL: &str = "my-qwen3-coder:30b";
const LLM_BASE_URL: &str = "http://10.10.10.100:11434";

const PREAMBLE: &str = "\
        The provided text is a summary of a code snippet.\n\
        Your task is to generate 3 synthetic queries that a developer \
        might use to search for this specific entry within a larger codebase.\n\
        Avoid relying on keyword search terms and favor concepts and meaning. \n\
        Focus on the specific purpose of the snippet. \n\
        Avoid general queries about design patterns, language idioms, error handling, etc. \n\
        Each query should be unique, and together, \
        they should span the semantic breadth of the summary. \n\
        Ensure that each query has enough semantic context to distinguish it from \
        general questions that can be answered without this specific snippet.\
    ";

#[derive(Deserialize, Serialize, JsonSchema, PartialEq, Debug, Clone)]
struct Synthetics {
    queries: Vec<String>,
}

#[derive(TypedBuilder)]
pub struct SynthWorker<A: DynAgent> {
    #[allow(dead_code)]
    agent: A,

    #[builder(default)]
    enabled: bool,
}

impl<A: DynAgent> SynthWorker<A> {
    pub async fn run(
        &self,
        receiver: Receiver<SnippetProgress>,
        sender: Sender<SnippetProgress>,
    ) -> anyhow::Result<()> {
        let llm = providers::ollama::Client::from_url(LLM_BASE_URL);
        let extractor = llm
            .extractor::<Synthetics>(LLM_MODEL)
            .preamble(PREAMBLE)
            .build();

        while let Ok(msg) = receiver.recv_async().await {
            match msg {
                SnippetProgress::Snippet { progress, snippet }
                    if self.enabled && !snippet.summary.is_empty() =>
                {
                    let snippet = match extractor.extract(&snippet.summary).await {
                        Ok(synthetics) => {
                            let queries = synthetics.queries;
                            log::info!("Synthetics: {queries:?}");

                            CodeSnippet { queries, ..snippet }
                        }
                        Err(err) => {
                            log::warn!("Could not synthesize queries: {err:?}");
                            snippet
                        }
                    };

                    sender
                        .send_async(SnippetProgress::Snippet { progress, snippet })
                        .await?;
                }
                _ => {
                    sender.send_async(msg).await?;
                }
            }
        }

        info!("No more snippets to summarize");
        Ok(())
    }
}
