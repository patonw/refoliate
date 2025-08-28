use std::time::Duration;

use flume::Receiver;
use indicatif::ProgressBar;
use typed_builder::TypedBuilder;

use crate::Progressor;
use crate::SnippetProgress;

#[derive(TypedBuilder)]
pub struct ProgressWorker {}

impl ProgressWorker {
    pub async fn run(self, receiver: Receiver<SnippetProgress>) -> anyhow::Result<Self> {
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
                SnippetProgress::Snippet { progress, snippet } => {
                    let count = snippet.body.len();
                    progress.as_ref().inspect(|p| p.inc(count as u64));
                }
            }
        }
        Ok(self)
    }
}
