use anyhow::{Context, Result};
use flume::Sender;
use ignore::{DirEntry, Walk};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use itertools::MinMaxResult;
use log::info;
use log::warn;
use std::{path::Path, sync::Arc};
use typed_builder::TypedBuilder;

use crate::{
    CodeSnippet, Progressor, SnippetProgress, SourceWalker,
    parse::{cb::FileMatchArgs, process_node},
};

#[derive(TypedBuilder)]
pub struct ExtractingWorker {
    walker: SourceWalker,
}

// Can't inline with iter_repo due to borrowing restrictions
fn filter_repo(walk: Walk) -> impl Iterator<Item = DirEntry> {
    walk.filter_map(|entry| entry.ok())
        .filter(|entry| !entry.path().is_dir())
}

impl ExtractingWorker {
    pub async fn count_files(&mut self, target_path: impl AsRef<Path>) -> Result<usize> {
        let walk = filter_repo(self.walker.iter_repo(target_path.as_ref())?);
        Ok(walk.count())
    }

    pub async fn run(
        &mut self,
        progressor: Arc<Option<Progressor>>,
        sender: Sender<SnippetProgress>,
        repo_root: impl AsRef<Path>,
        target_path: impl AsRef<Path>,
    ) -> Result<()> {
        let walk = filter_repo(self.walker.iter_repo(target_path.as_ref())?);
        for file_path in walk {
            if let Err(err) = extract_file(
                &progressor,
                &sender,
                &mut self.walker,
                repo_root.as_ref(),
                file_path,
            )
            .await
            {
                warn!("{err:?}");
            }
        }

        info!("Done walking {:?}", target_path.as_ref());

        Ok::<_, anyhow::Error>(())
    }
}

async fn extract_file(
    progressor: &Arc<Option<Progressor>>,
    snippet_tx: &Sender<SnippetProgress>,
    src_walk: &mut SourceWalker,
    root_path: impl AsRef<Path>,
    file_path: ignore::DirEntry,
) -> Result<()> {
    let (source_code, tree, query) = src_walk
        .parse_file(file_path.path())
        .await
        .context("Failed to parse file")?;

    // dbg!(tree.root_node().to_sexp());

    let file_path = file_path.path();
    let file_path = file_path.strip_prefix(root_path).unwrap_or(file_path);

    let entry = FileMatchArgs {
        file_path,
        source: source_code.as_slice(),
        tree: &tree,
        query: query.as_ref(),
    };

    let snip_tx = snippet_tx.clone();
    let progress = make_file_progress(progressor, &entry);
    let prog = progress.clone();

    snippet_tx
        .send_async(SnippetProgress::StartOfFile {
            file_path: file_path.to_path_buf(),
            progressor: progressor.clone(),
            progress: progress.clone(),
        })
        .await?;

    process_node(
        entry.tree.root_node(),
        entry.source,
        entry.query,
        vec![],
        &async move |node_match| {
            let progress = prog.clone();
            let n = node_match.query_match;
            let p = entry.file_path;
            let q = entry.query;
            let src = entry.source;

            log::debug!("^_- Match {n:?} at {p:?}");
            let mut attrs: Vec<String> = Vec::new();
            let mut interface: Option<String> = None;
            let mut class: Option<String> = None;
            let mut ident: Option<String> = None;
            let mut kind: Option<String> = None;
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
                    ["definition", k] => {
                        kind = Some(k.to_string());
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

            log::debug!("o.O Match results kind: {kind:?} identier: {ident:?} attrs: {attrs:?}");
            if let Some(body) = &body {
                let snippet = CodeSnippet {
                    path: p.display().to_string(),
                    interface,
                    class,
                    attributes: attrs,
                    name: ident.clone().unwrap_or("???".to_string()),
                    body: body.clone(),
                    summary: "".to_string(),
                    hash: Default::default(),
                };

                let msg = SnippetProgress::Snippet {
                    progress: progress.clone(),
                    snippet,
                };

                snip_tx.send_async(msg).await.unwrap();
            }
        },
    )
    .await;

    snippet_tx
        .send_async(SnippetProgress::EndOfFile {
            progressor: progressor.clone(),
            progress: progress.clone(),
        })
        .await?;

    // Why is this different from the tests?

    Ok(())
}

fn make_file_progress(
    progressor: &Arc<Option<Progressor>>,
    entry: &FileMatchArgs<'_>,
) -> Option<ProgressBar> {
    if let Some(bar) = progressor.as_ref() {
        let byte_progress = bar
            .multi
            .insert_from_back(1, ProgressBar::new(entry.source.len() as u64));

        byte_progress.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:30.cyan/blue} {decimal_bytes:>10}/{decimal_total_bytes:10} {wide_msg}",
            )
            .unwrap(),
        );
        byte_progress.set_message(format!("{:?}", entry.file_path));
        Some(byte_progress)
    } else {
        None
    }
}
