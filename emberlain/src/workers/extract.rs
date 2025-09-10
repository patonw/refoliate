use anyhow::{Context, Result};
use flume::Receiver;
use flume::Sender;
use indicatif::ProgressBar;
use itertools::Itertools;
use itertools::MinMaxResult;
use log::warn;
use std::path::Path;
use typed_builder::TypedBuilder;

use crate::{
    CodeSnippet, SnippetProgress, SourceWalker,
    parse::{cb::FileMatchArgs, process_node},
};

#[derive(TypedBuilder)]
pub struct ExtractingWorker {
    walker: SourceWalker,
}

impl ExtractingWorker {
    pub async fn run(
        &mut self,
        receiver: Receiver<SnippetProgress>,
        sender: Sender<SnippetProgress>,
        repo_root: impl AsRef<Path>,
    ) -> Result<()> {
        while let Ok(msg) = receiver.recv_async().await {
            match msg {
                SnippetProgress::StartOfFile {
                    file_path,
                    progressor,
                    progress,
                } => {
                    sender
                        .send_async(SnippetProgress::StartOfFile {
                            file_path: file_path.clone(),
                            progressor: progressor.clone(),
                            progress: progress.clone(),
                        })
                        .await?;

                    if let Err(err) = extract_file(
                        &sender,
                        &mut self.walker,
                        repo_root.as_ref(),
                        file_path,
                        progress.clone(),
                    )
                    .await
                    {
                        warn!("{err:?}");
                    }
                    sender
                        .send_async(SnippetProgress::EndOfFile {
                            progressor: progressor.clone(),
                            progress: progress.clone(),
                        })
                        .await?;
                }
                msg @ SnippetProgress::MissingFile { .. } => {
                    sender.send_async(msg).await?;
                }
                _ => {
                    log::warn!("Unexpected message received by ExtractingWorker");
                    sender.send_async(msg).await?;
                }
            }
        }

        Ok::<_, anyhow::Error>(())
    }
}

async fn extract_file(
    snippet_tx: &Sender<SnippetProgress>,
    src_walk: &mut SourceWalker,
    root_path: impl AsRef<Path>,
    file_path: impl AsRef<Path>,
    progress: Option<ProgressBar>,
) -> Result<()> {
    let abs_path = root_path.as_ref().join(file_path.as_ref());

    // TODO: handle missing files

    let (source_code, tree, query) = src_walk
        .parse_file(abs_path)
        .await
        .context("Failed to parse file")?;

    // dbg!(tree.root_node().to_sexp());

    let entry = FileMatchArgs {
        file_path: file_path.as_ref(),
        source: source_code.as_slice(),
        tree: &tree,
        query: query.as_ref(),
    };

    process_node(
        entry.tree.root_node(),
        entry.source,
        entry.query,
        vec![],
        &async move |node_match| {
            let n = node_match.query_match;
            let p = entry.file_path;
            let q = entry.query;
            let src = entry.source;

            // log::debug!("^_- Match {n:?} at {p:?}");
            let mut attrs: Vec<String> = Vec::new();
            let mut interface: Option<String> = None;
            let mut class: Option<String> = None;
            let mut ident: Option<String> = None;
            // let mut kind: Option<String> = None;
            let mut body: Option<String> = None;
            let mut bounds = Vec::new();

            // Maybe match destructuring should be part of SourceWalker
            for cap in &n.captures {
                let index = cap.index as usize;
                if q.capture_names().len() <= index {
                    continue;
                }

                // dbg!(cap.node.to_sexp());
                let cap_name = q.capture_names()[index];
                let parts: Vec<&str> = cap_name.split(".").collect();
                match parts.as_slice() {
                    // What's more common "attribute", "annotation", "decorator"?
                    ["attribute"] | ["annotation"] => {
                        if let Ok(n) = cap.node.utf8_text(src) {
                            attrs.push(n.to_string());
                        }
                    }
                    ["definition", _] => {
                        // kind = Some(k.to_string());
                        bounds.push(cap.node.start_byte());
                        bounds.push(cap.node.end_byte());
                    }
                    ["name", "definition", _] => {
                        if let Ok(n) = cap.node.utf8_text(src) {
                            ident = Some(n.to_string());
                        }
                    }
                    ["name", "reference", "interface"] => {
                        if let Ok(n) = cap.node.utf8_text(src) {
                            interface = Some(n.to_string());
                        }
                    }
                    ["name", "reference", "class"] => {
                        if let Ok(n) = cap.node.utf8_text(src) {
                            class = Some(n.to_string());
                        }
                    }
                    _ => {
                        warn!("Don't know what to do with this capture: {cap_name}")
                    }
                }
            }

            if let MinMaxResult::MinMax(a, b) = bounds.into_iter().minmax()
                && let Ok(txt) = str::from_utf8(&src[a..b])
            {
                body = Some(txt.to_string());
            }

            // log::debug!("o.O Match results kind: {kind:?} identier: {ident:?} attrs: {attrs:?}");
            if let Some(body) = &body {
                let snippet = CodeSnippet {
                    path: p.display().to_string(),
                    interface,
                    class,
                    attributes: attrs,
                    name: ident.clone().unwrap_or("???".to_string()),
                    body: body.clone(),
                    ..Default::default()
                };

                let msg = SnippetProgress::Snippet {
                    progress: progress.clone(),
                    snippet: Box::new(snippet),
                    clean: true,
                };

                snippet_tx.send_async(msg).await.unwrap();
            }
        },
    )
    .await;

    Ok(())
}
