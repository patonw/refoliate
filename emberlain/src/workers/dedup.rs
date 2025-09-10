use chrono::Utc;
use flume::Receiver;
use flume::Sender;
use itertools::Itertools;
use log::debug;
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::Condition;
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
                SnippetProgress::MissingFile { file_path } => {
                    log::debug!("File {file_path:?} is missing. Marking for removal.");
                    self.qdrant
                        .set_payload(
                            SetPayloadPointsBuilder::new(
                                &self.collection,
                                Payload::try_from(json!({
                                    "__removed": Utc::now().to_rfc3339(),
                                }))
                                .unwrap(),
                            )
                            .points_selector(Filter::must([
                                Condition::is_empty("__removed"),
                                Condition::matches("path", file_path.display().to_string()),
                            ])),
                        )
                        .await?;

                    continue;
                }
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
                SnippetProgress::Snippet {
                    progress, snippet, ..
                } if !self.reprocess => {
                    let snippet = self.templater.render(*snippet)?;
                    let body = snippet.body();
                    let hash = blake3::hash(body.as_bytes()).as_bytes().to_vec();
                    let hash_hex = hex::encode(&hash);
                    let points = self
                        .qdrant
                        .scroll(
                            ScrollPointsBuilder::new(&self.collection)
                                .filter(Filter::must([
                                    // Maybe too strict?
                                    Condition::matches("path", snippet.path.clone()),
                                    // Something wrong with path field, not 100% match...
                                    // ah, multiple instances of the contents exist, causing collisions
                                    Condition::matches("name", snippet.name.clone()),
                                    Condition::matches("hash", hash_hex.clone()),
                                ]))
                                .with_payload(true),
                        )
                        .await?;

                    let snippet = if !points.result.is_empty() {
                        log::debug!(
                            "Existing point with hash of {hash_hex}: {:?}",
                            points.result
                        );

                        if points.result.len() > 1 {
                            let point_ids = points
                                .result
                                .iter()
                                .filter_map(|p| p.id.clone())
                                .collect_vec();
                            log::warn!(
                                "Too many matching points for snippet {snippet:?}: {point_ids:?}",
                            )
                        }

                        let point = points.result.first().unwrap();

                        let summary = point
                            .payload
                            .get("summary")
                            .and_then(|v| v.as_str())
                            .cloned()
                            .unwrap_or_default();

                        let queries: Vec<String> = point
                            .payload
                            .get("queries")
                            .and_then(|v| {
                                let value = v.clone().into_json();
                                let queries = serde_json::from_value::<Vec<String>>(value);
                                queries.ok()
                            })
                            .unwrap_or_default();

                        // log::debug!("Retrieved summary: {summary}\nqueries: {queries:?}");

                        CodeSnippet {
                            hash,
                            summary,
                            queries,
                            ..snippet
                        }
                    } else {
                        log::info!(
                            "New snippet. path: {} name: {} hash: {}",
                            &snippet.path,
                            &snippet.name,
                            &hash_hex
                        );
                        CodeSnippet { hash, ..snippet }
                    };

                    log::debug!(
                        "Merged snippet: {}",
                        serde_json::to_string_pretty(&snippet).unwrap_or("???".to_string())
                    );

                    SnippetProgress::Snippet {
                        progress,
                        snippet: Box::new(snippet),
                        clean: true,
                    }
                }
                _ => msg,
            };

            sender.send_async(msg).await?;
        }
        Ok(self)
    }
}
