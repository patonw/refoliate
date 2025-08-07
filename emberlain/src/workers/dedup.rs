use flume::Receiver;
use flume::Sender;
use typed_builder::TypedBuilder;

use crate::SnippetProgress;

#[derive(TypedBuilder)]
pub struct DedupWorker {
    #[allow(dead_code)]
    reprocess: bool,
}

impl DedupWorker {
    pub async fn run(
        self,
        receiver: Receiver<SnippetProgress>,
        sender: Sender<SnippetProgress>,
    ) -> anyhow::Result<Self> {
        // TODO:
        // - hash content
        // - Mark stale records when starting file
        // - Lookup and duplicates
        // - Update progress bars if needed.
        while let Ok(msg) = receiver.recv_async().await {
            sender.send_async(msg).await?;
        }
        Ok(self)
    }
}
