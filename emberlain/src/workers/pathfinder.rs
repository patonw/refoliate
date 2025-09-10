use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Result;
use flume::Sender;
use ignore::{DirEntry, Walk, WalkBuilder, types::Types};
use indicatif::{ProgressBar, ProgressStyle};
use qdrant_client::{
    Qdrant,
    qdrant::{FacetCountsBuilder, FacetHit},
};
use typed_builder::TypedBuilder;

use crate::{Progressor, SnippetProgress};

#[derive(TypedBuilder)]
pub struct Pathfinder {
    types: Types,
    qdrant: Qdrant,
    collection: String,
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
        let rel_target = target_path
            .as_ref()
            .strip_prefix(repo_root.as_ref())
            .unwrap_or(target_path.as_ref())
            .to_owned();

        // fetch previous paths from DB
        let resp = self
            .qdrant
            .facet(FacetCountsBuilder::new(&self.collection, "path").limit(1_000_000))
            .await?;

        let db_paths: BTreeSet<PathBuf> = resp
            .hits
            .into_iter()
            .filter_map(facet_hit_path)
            // Only consider files in the target subtree as missing
            .filter(|p| p.strip_prefix(&rel_target).is_ok())
            .collect();

        log::info!("Existing paths: {db_paths:?}");

        let walk = WalkBuilder::new(target_path.as_ref())
            .types(self.types.clone())
            .build();
        let walk = filter_repo(walk);
        let file_sizes: BTreeMap<_, _> = walk
            .filter_map(|p| {
                let file_path = p.path();

                let file_size = file_path.metadata().ok()?.len();
                let file_path = file_path
                    .strip_prefix(repo_root.as_ref())
                    .map(|p| p.to_path_buf())
                    .unwrap_or(file_path.to_owned());

                Some((file_path, file_size))
            })
            .collect();

        let fs_keys: BTreeSet<PathBuf> = file_sizes.keys().map(|p| p.to_owned()).collect();

        for file_path in db_paths.union(&fs_keys) {
            if file_path.is_dir() {
                continue;
            }

            if !file_sizes.contains_key(file_path) {
                sender
                    .send_async(SnippetProgress::MissingFile {
                        file_path: file_path.to_path_buf(),
                    })
                    .await?;
            } else {
                let file_size = file_sizes.get(file_path).unwrap();

                let progress =
                    make_file_progress(progressor.clone(), file_path.as_path(), *file_size);

                sender
                    .send_async(SnippetProgress::StartOfFile {
                        file_path: file_path.into(),
                        progressor: progressor.clone(),
                        progress,
                    })
                    .await?;
            }
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

fn facet_hit_path(hit: FacetHit) -> Option<PathBuf> {
    use qdrant_client::qdrant::facet_value::Variant;
    if let Variant::StringValue(p) = hit.value?.variant? {
        Some(PathBuf::from(p))
    } else {
        None
    }
}
