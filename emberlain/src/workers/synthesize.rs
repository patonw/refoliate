use flume::{Receiver, Sender};
use log::info;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use crate::CodeSnippet;
use crate::DynExtractor;
use crate::SnippetProgress;

#[derive(Deserialize, Serialize, JsonSchema, PartialEq, Debug, Clone)]
pub struct Synthetics {
    queries: Vec<String>,
}

#[derive(TypedBuilder)]
pub struct SynthWorker<T: DynExtractor<Synthetics>> {
    extractor: T,

    #[builder(default)]
    enabled: bool,

    #[builder(default)]
    reprocess: bool,
}

impl<T: DynExtractor<Synthetics>> SynthWorker<T> {
    pub async fn run(
        &self,
        receiver: Receiver<SnippetProgress>,
        sender: Sender<SnippetProgress>,
    ) -> anyhow::Result<()> {
        let extractor = &self.extractor;

        while let Ok(msg) = receiver.recv_async().await {
            match msg {
                SnippetProgress::Snippet {
                    progress, snippet, ..
                } if (self.enabled && !snippet.summary.is_empty())
                    && (self.reprocess || snippet.queries.is_empty()) =>
                {
                    log::info!("Synthesizing queries for {snippet:?}");
                    let snippet = match extractor.extract(&snippet.summary).await {
                        Ok(synthetics) => {
                            let queries = synthetics.queries;
                            log::info!("Synthetics: {queries:?}");

                            Box::new(CodeSnippet {
                                queries,
                                ..*snippet
                            })
                        }
                        Err(err) => {
                            log::warn!("Could not synthesize queries: {err:?}");
                            snippet
                        }
                    };

                    sender
                        .send_async(SnippetProgress::Snippet {
                            progress,
                            snippet,
                            clean: false,
                        })
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
