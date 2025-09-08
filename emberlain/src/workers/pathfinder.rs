use std::{path::Path, sync::Arc};

use anyhow::Result;
use flume::Sender;
use ignore::{DirEntry, Walk, WalkBuilder, types::Types};
use indicatif::{ProgressBar, ProgressStyle};
use typed_builder::TypedBuilder;

use crate::{Progressor, SnippetProgress};

#[derive(TypedBuilder)]
pub struct Pathfinder {
    types: Types,
}

impl Pathfinder {
    pub async fn count_files(&self, target_path: impl AsRef<Path>) -> Result<usize> {
        let walk = WalkBuilder::new(target_path.as_ref())
            .types(self.types.clone())
            .build();
        let walk = filter_repo(walk);
        Ok(walk.count())
    }

    pub async fn run(
        &self,
        progressor: Arc<Option<Progressor>>,
        sender: Sender<SnippetProgress>,
        repo_root: impl AsRef<Path>,
        target_path: impl AsRef<Path>,
    ) -> Result<()> {
        // TODO: fetch previous paths from DB
        let walk = WalkBuilder::new(target_path.as_ref())
            .types(self.types.clone())
            .build();
        let walk = filter_repo(walk);
        for file_path in walk {
            if file_path.path().is_dir() {
                continue;
            }

            let file_path = file_path.path();
            let file_size = file_path.metadata()?.len();
            let file_path = file_path
                .strip_prefix(repo_root.as_ref())
                .unwrap_or(file_path);

            let progress = make_file_progress(progressor.clone(), file_path, file_size);

            sender
                .send_async(SnippetProgress::StartOfFile {
                    file_path: file_path.to_path_buf(),
                    progressor: progressor.clone(),
                    progress,
                })
                .await?;
            log::info!("Sent {file_path:?}");
        }

        log::info!("Done walking {:?}", target_path.as_ref());

        Ok::<_, anyhow::Error>(())
    }
}

// Can't inline with iter_repo due to borrowing restrictions
fn filter_repo(walk: Walk) -> impl Iterator<Item = DirEntry> {
    walk.filter_map(|entry| entry.ok())
        .filter(|entry| !entry.path().is_dir())
}

fn make_file_progress(
    progressor: Arc<Option<Progressor>>,
    file_path: impl AsRef<Path>,
    file_size: u64,
) -> Option<ProgressBar> {
    if let Some(bar) = progressor.as_ref() {
        let byte_progress = bar.multi.insert_from_back(1, ProgressBar::new(file_size));

        byte_progress.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:30.cyan/blue} {decimal_bytes:>10}/{decimal_total_bytes:10} {wide_msg}",
            )
            .unwrap(),
        );
        byte_progress.set_message(format!("{:?}", file_path.as_ref()));
        Some(byte_progress)
    } else {
        None
    }
}
