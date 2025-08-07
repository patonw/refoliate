use indicatif::ProgressBar;
use indoc::indoc;
use log::{info, warn};
use rig::{completion::Prompt, prelude::*, providers::ollama};
use tokio::sync::mpsc::{Receiver, Sender};

use crate::{CONFIG, CodeSnippet, LLM_BASE_URL, LLM_MODEL, Progressor, SnippetProgress};

pub async fn summary_worker(
    mut snippet_rx: Receiver<SnippetProgress>,
    summary_tx: Sender<CodeSnippet>,
) -> anyhow::Result<()> {
    let agent = ollama::Client::from_url(&LLM_BASE_URL)
            .agent(&LLM_MODEL)
            .max_tokens(1024)
            // TODO: Really need to work on taming the LLM's verbiage
            .preamble(indoc! {r##"
                You are a helpful software engineer mentoring a new team-mate.
                Without preamble or introduction, summarize provided code snippets, in a few sentences.
                Explain what it does and how it works in general terms without referring to specific values.
                If it uses higher level concepts and design patterns like observers, pipelines, etc. make note of that.
                Be sure to mention key types and functions used, if applicable.
                But do not dwell on basic concepts or things that are obvious like what it means for something to be public.
                Keep your explanation in paragraph format, using complete sentences.
            ""##})
            .build();

    // TODO: check and skip existing snippets
    // Not worth batching when using ollama with such weak hardware, but later...
    while let Some(msg) = snippet_rx.recv().await {
        match msg {
            SnippetProgress::Snippet {
                progress, snippet, ..
            } => {
                // Skip one-liners, aliases, forward declarations, etc.
                if !snippet.body.contains("\n") {
                    continue;
                }

                let count = snippet.body.len();

                // TODO: strip out comments since we want to rely on LLM to interpret code rather than
                // regurgitating out-of-date or deceptive descriptions
                let body = if let Some(self_type) = &snippet.class {
                    // TODO: universal commenting or wrap in markdown
                    format!("/// self: {self_type}\n{}", &snippet.body)
                } else {
                    snippet.body.clone()
                };

                let hash = blake3::hash(body.as_bytes()).as_bytes().to_vec();

                let options = textwrap::Options::new(100)
                    .initial_indent(">>> ")
                    .subsequent_indent("... ");
                info!("{}", textwrap::fill(&body, &options));

                if !CONFIG.dry_run.unwrap_or(false) {
                    match agent.prompt(body).await {
                        Ok(resp) => {
                            let snippet = CodeSnippet {
                                summary: resp,
                                hash,
                                ..snippet
                            };
                            summary_tx.send(snippet).await.unwrap();
                        }
                        Err(err) => warn!("Could not summarize snippet: {err:?}"),
                    }
                }

                // Would prefer using file position since there could be large comments at the
                // top level, but parser can jump around quite a bit depending on the traversal.
                progress.as_ref().inspect(|p| p.inc(count as u64));
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
        }
    }

    info!("No more snippets to summarize");

    Ok(())
}
