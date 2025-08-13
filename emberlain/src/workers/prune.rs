use chrono::{DateTime, Utc};
use qdrant_client::{
    Qdrant,
    qdrant::{self, Condition, DatetimeRange, DeletePointsBuilder, Filter},
};
use typed_builder::TypedBuilder;

#[derive(TypedBuilder)]
pub struct PruningWorker {
    cutoff: DateTime<Utc>,
    qdrant: Qdrant,
    collection: String,
}

impl PruningWorker {
    pub async fn run(&self) -> anyhow::Result<()> {
        log::info!("pruning points older than {}", &self.cutoff);
        let prune = qdrant::Timestamp {
            seconds: self.cutoff.timestamp(),
            ..Default::default()
        };

        self.qdrant
            .delete_points(
                DeletePointsBuilder::new(&self.collection)
                    .points(Filter::must([Condition::datetime_range(
                        "__removed",
                        DatetimeRange {
                            lte: Some(prune),
                            ..Default::default()
                        },
                    )]))
                    .wait(true),
            )
            .await?;

        Ok(())
    }
}
