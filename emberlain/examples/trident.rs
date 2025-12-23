use anyhow::anyhow;
use qdrant_client::{
    Qdrant,
    qdrant::{CreateCollectionBuilder, Distance, QueryPointsBuilder, VectorParamsBuilder},
};
use rig::{
    Embed,
    vector_store::{InsertDocuments, VectorStoreIndex},
};
use rig_qdrant::QdrantVectorStore;

const EMBEDDING_DIMS: u64 = 384;
const COLLECTION_NAME: &str = "my_collection";

#[derive(Embed, serde::Deserialize, serde::Serialize, Debug)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> //Result<(), QdrantError>
{
    use rig::embeddings::EmbeddingsBuilder;
    use rig_fastembed::{Client, FastembedModel};

    // Initialize the Fastembed client
    let fastembed_client = Client::new();

    let embedding_model = fastembed_client.embedding_model(&FastembedModel::AllMiniLML6V2Q);

    // Example of top level client
    // You may also use tonic-generated client from `src/qdrant.rs`
    let client = Qdrant::from_url("http://localhost:6334").build()?;
    if !client.collection_exists(COLLECTION_NAME).await? {
        client
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION_NAME)
                    .vectors_config(VectorParamsBuilder::new(EMBEDDING_DIMS, Distance::Cosine)),
            )
            .await?;
    }

    let documents = EmbeddingsBuilder::new(embedding_model.clone())
        .document(Word {
            id: "0981d983-a5f8-49eb-89ea-f7d3b2196d2e".to_string(),
            definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        })?
        .document(Word {
            id: "62a36d43-80b6-4fd6-990c-f75bb02287d1".to_string(),
            definition: "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        })?
        .document(Word {
            id: "f9e17d59-32e5-440c-be02-b2759a654824".to_string(),
            definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        })?
        .build()
        .await?;
    let query_params = QueryPointsBuilder::new(COLLECTION_NAME).with_payload(true);
    let vector_store = QdrantVectorStore::new(client, embedding_model, query_params.build());

    vector_store
        .insert_documents(documents)
        .await
        .map_err(|err| anyhow!("Couldn't insert documents: {err}"))?;

    let results = vector_store
        .top_n::<Word>("What is a linglingdong?", 1)
        .await?;

    println!("Results: {results:?}");

    Ok(())
}
