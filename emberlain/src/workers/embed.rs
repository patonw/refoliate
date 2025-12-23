use fastembed::TextEmbedding;
use flume::{Receiver, Sender};
use log::{info, warn};
use qdrant_client::{
    Payload, Qdrant,
    qdrant::{PointStruct, UpsertPointsBuilder, Vector},
};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use typed_builder::TypedBuilder;

use crate::SnippetProgress;

#[derive(TypedBuilder)]
pub struct EmbeddingWorker {
    embedding: Arc<Mutex<TextEmbedding>>,
    qdrant: Qdrant,
    collection: String,
}

impl EmbeddingWorker {
    pub async fn run(
        &self,
        receiver: Receiver<SnippetProgress>,
        sender: Sender<SnippetProgress>,
    ) -> anyhow::Result<()> {
        while let Ok(msg) = receiver.recv_async().await {
            if let SnippetProgress::Snippet { snippet, .. } = &msg {
                let result = async {
                    let options = textwrap::Options::new(100)
                        .initial_indent(">.< ")
                        .subsequent_indent("-.- ");

                    info!("X.X ID = {:?}", snippet.uuid());
                    info!("{}", textwrap::fill(&snippet.summary, &options));

                    // this could be cleaner
                    let mut texts = vec![snippet.summary.as_str()];
                    texts.extend(snippet.queries.iter().map(|s| s.as_str()));

                    let embeddings = {
                        let mut embedder = self.embedding.lock().unwrap();
                        embedder.embed(texts, None)?
                    };

                    let embedding = embeddings[0].clone();

                    let id = snippet.uuid()?.to_string();
                    let value = serde_json::to_value(snippet)?;
                    let payload = Payload::try_from(value)?;

                    let vectors = HashMap::from([
                        ("default".to_string(), Vector::new_dense(embedding)),
                        ("aliases".to_string(), Vector::new_multi(embeddings)),
                    ]);
                    let point = PointStruct::new(id, vectors, payload);

                    let request =
                        UpsertPointsBuilder::new(self.collection.as_str(), vec![point]).build();
                    self.qdrant.upsert_points(request).await?;

                    Ok::<_, anyhow::Error>(())
                }
                .await;

                if let Err(e) = result {
                    warn!("Unable to handle snippet: {e:?}");
                };
            }

            sender.send_async(msg).await.unwrap();
        }

        Ok(())
    }
}
