use chrono::Utc;
use flume::Receiver;
use flume::Sender;
use itertools::Itertools;
use log::debug;
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::Condition;
use qdrant_client::qdrant::DeletePayloadPointsBuilder;
use qdrant_client::qdrant::Filter;
use qdrant_client::qdrant::PointsIdsList;
use qdrant_client::qdrant::ScrollPointsBuilder;
use qdrant_client::qdrant::SetPayloadPointsBuilder;
use serde_json::json;
use typed_builder::TypedBuilder;

use crate::CodeSnippet;
use crate::SnippetProgress;
use crate::template::Templater;

#[derive(TypedBuilder)]
pub struct DedupWorker<'a> {
    reprocess: bool,
    qdrant: Qdrant,
    collection: String,
    templater: Templater<'a>,
}

impl<'a> DedupWorker<'a> {
    pub async fn run(
        self,
        receiver: Receiver<SnippetProgress>,
        sender: Sender<SnippetProgress>,
    ) -> anyhow::Result<Self> {
        while let Ok(msg) = receiver.recv_async().await {
            let msg = match msg {
                SnippetProgress::StartOfFile {
                    file_path,
                    progressor,
                    progress,
                } => {
                    // mark old snippets as out-of-date
                    let points = self
                        .qdrant
                        .scroll(
                            ScrollPointsBuilder::new(&self.collection).filter(Filter::must([
                                Condition::is_empty("__removed"),
                                Condition::matches("path", file_path.display().to_string()),
                            ])),
                        )
                        .await?;

                    let point_ids = points.result.into_iter().filter_map(|p| p.id).collect_vec();
                    debug!("marking points for {file_path:?}: {point_ids:?}");

                    if !point_ids.is_empty() {
                        self.qdrant
                            .set_payload(
                                // If point is reprocessed, this flag disappears, allowing us to
                                // distinguish between live and stale code
                                SetPayloadPointsBuilder::new(
                                    &self.collection,
                                    Payload::try_from(json!({
                                        "__removed": Utc::now().to_rfc3339(),
                                    }))
                                    .unwrap(),
                                )
                                .points_selector(PointsIdsList { ids: point_ids })
                                .wait(true), // Necessary?
                            )
                            .await?;
                    }

                    SnippetProgress::StartOfFile {
                        file_path,
                        progressor,
                        progress,
                    }
                }
                SnippetProgress::Snippet { progress, snippet } if !self.reprocess => {
                    let snippet = self.templater.render(snippet)?;
                    let body = snippet.body();
                    let hash = blake3::hash(body.as_bytes()).as_bytes().to_vec();
                    let hash_hex = hex::encode(&hash);
                    let points = self
                        .qdrant
                        .scroll(
                            ScrollPointsBuilder::new(&self.collection).filter(Filter::must([
                                // Maybe too strict?
                                Condition::matches("path", snippet.path.clone()),
                                // Something wrong with path field, not 100% match...
                                // ah, multiple instances of the contents exist, causing collisions
                                Condition::matches("name", snippet.name.clone()),
                                Condition::matches("hash", hash_hex.clone()),
                            ])),
                        )
                        .await?;

                    let point_ids = points.result.into_iter().filter_map(|p| p.id).collect_vec();

                    if !point_ids.is_empty() {
                        log::debug!("Skip existing point with hash of {hash_hex}");
                        self.qdrant
                            .delete_payload(
                                DeletePayloadPointsBuilder::new(
                                    &self.collection,
                                    vec!["__removed".to_string()],
                                )
                                .points_selector(PointsIdsList { ids: point_ids })
                                .wait(true),
                            )
                            .await?;
                        progress
                            .as_ref()
                            .inspect(|p| p.inc(snippet.body.len() as u64));

                        // don't send dups unless reprocessing
                        continue;
                    }

                    log::info!(
                        "New snippet. path: {} name: {} hash: {}",
                        &snippet.path,
                        &snippet.name,
                        &hash_hex
                    );
                    // Otherwise, this is new, so process normally
                    let snippet = CodeSnippet { hash, ..snippet };
                    SnippetProgress::Snippet { progress, snippet }
                }
                _ => msg,
            };

            sender.send_async(msg).await?;
        }
        Ok(self)
    }
}
