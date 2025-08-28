use flume::{Receiver, Sender};
use log::{info, warn};
use typed_builder::TypedBuilder;

use crate::DynAgent;
use crate::{CodeSnippet, SnippetProgress};

#[derive(TypedBuilder)]
pub struct SummaryWorker<A: DynAgent> {
    agent: A,

    #[builder(default)]
    dry_run: bool,
}

impl<A: DynAgent> SummaryWorker<A> {
    pub async fn run(
        &self,
        receiver: Receiver<SnippetProgress>,
        sender: Sender<SnippetProgress>,
    ) -> anyhow::Result<()> {
        while let Ok(msg) = receiver.recv_async().await {
            match msg {
                SnippetProgress::Snippet {
                    progress, snippet, ..
                } => {
                    // Skip trivial declarations: one-liners, aliases, forward declarations, etc.
                    if snippet.body.lines().count() <= 4 {
                        // TODO: principled cutoff logic. Ideally exclude code with a single
                        // statement, not counting signature, braces, comments, etc
                        continue;
                    }

                    let body = snippet.body();

                    let options = textwrap::Options::new(100)
                        .initial_indent(">>> ")
                        .subsequent_indent("... ");
                    info!("{}", textwrap::fill(&body, &options));

                    if !self.dry_run {
                        match self.agent.prompt(&body).await {
                            Ok(resp) => {
                                let snippet = CodeSnippet {
                                    summary: resp,
                                    ..snippet
                                };
                                sender
                                    .send_async(SnippetProgress::Snippet { snippet, progress })
                                    .await
                                    .unwrap();
                            }
                            Err(err) => warn!("Could not summarize snippet: {err:?}"),
                        }
                    }
                }
                _ => {
                    sender.send_async(msg).await.unwrap();
                }
            }
        }

        info!("No more snippets to summarize");
        Ok(())
    }
}
