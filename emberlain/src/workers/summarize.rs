use std::time::Duration;

use flume::{Receiver, Sender};
use indicatif::ProgressBar;
use log::{info, warn};
use typed_builder::TypedBuilder;

use crate::DynAgent;
use crate::{CodeSnippet, Progressor, SnippetProgress};

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
        sender: Sender<CodeSnippet>,
    ) -> anyhow::Result<()> {
        // TODO: check and skip existing snippets
        // Not worth batching when using ollama with such weak hardware, but later...
        while let Ok(msg) = receiver.recv_async().await {
            match msg {
                SnippetProgress::StartOfFile {
                    file_path: _,
                    progressor: _,
                    progress,
                } => {
                    // Otherwise we include time spent queued, waiting for other files to finish
                    progress.inspect(|p| {
                        p.reset();
                        p.enable_steady_tick(Duration::from_secs(1));
                    });
                }
                SnippetProgress::Snippet {
                    progress, snippet, ..
                } => {
                    // Skip one-liners, aliases, forward declarations, etc.
                    if !snippet.body.contains("\n") {
                        continue;
                    }

                    let count = snippet.body.len();

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
                                sender.send_async(snippet).await.unwrap();
                            }
                            Err(err) => warn!("Could not summarize snippet: {err:?}"),
                        }
                    }

                    // Would prefer using file position since there could be large comments at the
                    // top level, but parser can jump around quite a bit depending on the traversal.
                    progress.as_ref().inspect(|p| p.inc(count as u64));
                }
                // Of course, having multiple workers means we can get to the end of the file while
                // there are still pending tasks.
                SnippetProgress::EndOfFile {
                    progressor,
                    progress,
                } => {
                    if let Some(bar) = progress.as_ref()
                        && let Some(Progressor {
                            multi,
                            file_progress,
                        }) = progressor.as_ref()
                    {
                        // Emulate detaching a finished bar from the multi by creating a dummy
                        multi.remove(bar);
                        file_progress.inc(1);

                        multi.suspend(|| {
                            let tombstone = bar
                                .length()
                                .map(ProgressBar::new)
                                .unwrap_or_else(ProgressBar::no_length)
                                .with_prefix(bar.prefix())
                                .with_style(bar.style())
                                .with_elapsed(bar.elapsed());

                            tombstone.finish_with_message(bar.message());
                        });
                    }
                }
            }
        }

        info!("No more snippets to summarize");
        Ok(())
    }
}
