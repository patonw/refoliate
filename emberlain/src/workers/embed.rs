use itertools::Itertools;
use log::{info, warn};
use qdrant_client::{
    Payload, Qdrant,
    qdrant::{
        CreateCollectionBuilder, Distance, PointStruct, UpsertPointsBuilder, VectorParamsBuilder,
    },
};
use rig::embeddings::EmbeddingsBuilder;
use rig_fastembed::Client;
use tokio::sync::mpsc::Receiver;
use uuid::Uuid;

use crate::{COLLECTION_NAME, CodeSnippet, EMBED_DIMS, EMBED_MODEL};

pub async fn embedding_worker(mut summary_rx: Receiver<CodeSnippet>) -> anyhow::Result<()> {
    // Initialize the Fastembed client
    let fastembed_client = Client::new();

    let embedding_model = fastembed_client.embedding_model(&EMBED_MODEL);

    let client = Qdrant::from_url("http://localhost:6334").build()?;
    if !client.collection_exists(COLLECTION_NAME.as_str()).await? {
        client
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION_NAME.as_str()).vectors_config(
                    VectorParamsBuilder::new(*EMBED_DIMS as u64, Distance::Cosine),
                ),
            )
            .await?;
    }

    // let query_params = QueryPointsBuilder::new(COLLECTION_NAME.as_str()).with_payload(true);
    // let vector_store =
    //     QdrantVectorStore::new(client, embedding_model.clone(), query_params.build());

    while let Some(msg) = summary_rx.recv().await {
        let result = async {
            // println!("Summarized: {msg:?}");
            let id = u64::from_be_bytes(msg.hash[..8].try_into()?);

            let options = textwrap::Options::new(100)
                .initial_indent(">.< ")
                .subsequent_indent("-.- ");

            info!(
                "X.X ID = {id} HASH = {:?}",
                blake3::Hash::from_slice(&msg.hash)
            );
            info!("{}", textwrap::fill(&msg.summary, &options));

            let documents = EmbeddingsBuilder::new(embedding_model.clone())
                .document(msg)?
                .build()
                .await?;

            if let Some((doc, embeds)) = documents.first() {
                // let id = u64::from_be_bytes(doc.hash[..8].try_into()?);
                let id = Uuid::new_v8(doc.hash[..16].try_into()?).to_string();
                let value = serde_json::to_value(doc)?;
                let payload = Payload::try_from(value)?;
                let embedding = embeds.first().vec.iter().map(|x| *x as f32).collect_vec();

                let point = PointStruct::new(id, embedding, payload);
                let request =
                    UpsertPointsBuilder::new(COLLECTION_NAME.as_str(), vec![point]).build();
                client.upsert_points(request).await?;
            }

            // vector_store
            //     .insert_documents(documents)
            //     .await
            //     .map_err(|err| anyhow!("Couldn't insert documents: {err}"))?;
            Ok::<_, anyhow::Error>(())
        }
        .await;

        if let Err(e) = result {
            warn!("Unable to handle snippet: {e:?}");
        };
    }

    Ok(())
}
