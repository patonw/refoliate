use anyhow::Context;
use indicatif::{ProgressBar, ProgressStyle};
use log::warn;
use log::{debug, info};
use std::{path::Path, sync::Arc};
use tokio::sync::mpsc::Sender;

use crate::{
    CodeSnippet, Progressor, SnippetProgress, SourceWalker,
    parse::{cb::FileMatchArgs, process_node},
};

// TODO: Extract to lib to reuse in distributed mode
pub async fn counting_worker(target_path: impl AsRef<Path>) -> anyhow::Result<usize> {
    let langspec = include_str!("../../etc/languages.yml");
    let mut src_walk = SourceWalker::default();
    src_walk.load_languages(langspec)?;

    Ok(src_walk
        .iter_repo(target_path)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| !entry.path().is_dir())
        .count())
}

pub async fn extracting_worker(
    progressor: Arc<Option<Progressor>>,
    snippet_tx: Sender<SnippetProgress>,
    target_path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let langspec = include_str!("../../etc/languages.yml");
    let mut src_walk = SourceWalker::default();
    src_walk.load_languages(langspec)?;

    for file_path in src_walk.iter_repo(target_path.as_ref())? {
        if let Err(err) = extract_file(&progressor, &snippet_tx, &mut src_walk, file_path).await {
            warn!("{err:?}");
        }
    }

    info!("Done walking {:?}", target_path.as_ref());

    Ok::<_, anyhow::Error>(())
}

async fn extract_file(
    progressor: &Arc<Option<Progressor>>,
    snippet_tx: &Sender<SnippetProgress>,
    src_walk: &mut SourceWalker,
    file_path: std::result::Result<ignore::DirEntry, ignore::Error>,
) -> anyhow::Result<()> {
    let file_path = file_path.context("Error while walking repo")?;
    if file_path.path().is_dir() {
        debug!("Directory: {file_path:?}");
        return Ok(());
    }

    let (source_code, tree, query) = src_walk
        .parse_file(file_path.path())
        .await
        .context("Failed to parse file")?;

    let entry = FileMatchArgs {
        file_path: file_path.path(),
        source: source_code.as_slice(),
        tree: &tree,
        query: query.as_ref(),
    };

    let snip_tx = snippet_tx.clone();
    let progress = make_file_progress(progressor, &entry);
    let prog = progress.clone();

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
            info!("^_- Match {n:?} at {p:?}");
            let mut class: Option<String> = None;
            let mut ident: Option<String> = None;
            let mut kind: Option<String> = None;
            let mut body: Option<String> = None;

            // Maybe match destructuring should be part of SourceWalker
            for cap in &n.captures {
                let index = cap.index as usize;
                if q.capture_names().len() <= index {
                    continue;
                }

                let cap_name = q.capture_names()[index];
                let parts: Vec<&str> = cap_name.split(".").collect();
                match parts.as_slice() {
                    ["definition", k] => {
                        kind = Some(k.to_string());
                        if let Ok(n) = cap.node.utf8_text(src) {
                            body = Some(n.to_string());
                        }
                    }
                    ["name", "definition", _] => {
                        if let Ok(n) = cap.node.utf8_text(src) {
                            ident = Some(n.to_string());
                        }
                    }
                    ["name", "reference", _] => {
                        if let Ok(n) = cap.node.utf8_text(src) {
                            class = Some(n.to_string());
                        }
                    }
                    _ => {
                        warn!("Don't know what to do with this capture: {cap_name}")
                    }
                }
            }

            info!("o.O Match results kind: {kind:?} identier: {ident:?}");
            if let Some(body) = &body {
                let snippet = CodeSnippet {
                    path: p.display().to_string(),
                    class,
                    name: ident.clone().unwrap_or("???".to_string()),
                    body: body.clone(),
                    summary: "".to_string(),
                    hash: Default::default(),
                };

                let msg = SnippetProgress::Snippet {
                    progress: progress.clone(),
                    snippet,
                };

                snip_tx.send(msg).await.unwrap();
            }
        },
    )
    .await;
    snippet_tx
        .send(SnippetProgress::EndOfFile {
            progressor: progressor.clone(),
            progress: progress.clone(),
        })
        .await
        .unwrap();

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
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .unwrap(),
        );
        byte_progress.set_message(format!("{:?}", entry.file_path));
        Some(byte_progress)
    } else {
        None
    }
}
