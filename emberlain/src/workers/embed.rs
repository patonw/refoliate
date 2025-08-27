use fastembed::TextEmbedding;
use flume::Receiver;
use log::{info, warn};
use qdrant_client::{
    Payload, Qdrant,
    qdrant::{PointStruct, UpsertPointsBuilder, Vector},
};
use std::{collections::HashMap, sync::Arc};
use typed_builder::TypedBuilder;

use crate::CodeSnippet;

#[derive(TypedBuilder)]
pub struct EmbeddingWorker {
    embedding: Arc<TextEmbedding>,
    qdrant: Qdrant,
    collection: String,
}

impl EmbeddingWorker {
    pub async fn run(&self, receiver: Receiver<CodeSnippet>) -> anyhow::Result<()> {
        while let Ok(msg) = receiver.recv_async().await {
            let result = async {
                let options = textwrap::Options::new(100)
                    .initial_indent(">.< ")
                    .subsequent_indent("-.- ");

                info!("X.X ID = {:?}", msg.uuid());
                info!("{}", textwrap::fill(&msg.summary, &options));

                // this could be cleaner
                let embedding = self
                    .embedding
                    .embed(vec![&msg.summary], None)?
                    .pop()
                    .unwrap();

                let id = msg.uuid()?.to_string();
                let value = serde_json::to_value(msg)?;
                let payload = Payload::try_from(value)?;

                let vectors = HashMap::from([
                    ("default".to_string(), Vector::new_dense(embedding.clone())),
                    (
                        "aliases".to_string(),
                        Vector::new_multi(vec![embedding.clone()]),
                    ),
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

        Ok(())
    }
}
